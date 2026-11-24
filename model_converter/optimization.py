"""Bounded scale search and data-free, SVD-guided learned rounding.

Original implementation using Kitchen's layouts and codecs, not CTQ runtime code.
"""

from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import time

from .formats import (FORMATS, MICROSCALED_FORMATS, ROTATED_FORMATS, ADJUSTABLE_ROTATED_FORMATS,
                      quantization_config, storage_contract, resolve_convrot_group)


MODES = ("standard", "balanced", "thorough")
MODE_CHOICES = [(name.title(), name) for name in MODES]
OPTIMIZER_CHOICES = [("Prodigy + Schedule-Free", "prodigy"), ("AdamW", "adamw")]
THOROUGH_FORMATS = FORMATS
GROUPED_INT8_FORMATS = ("asym_w4a8_int8", "w6a8_int8")
TILE_ELEMENTS = 1024 * 1024
SVD_RANK = 16
SCALE_RATIOS = (1., .975, .95, .9, .85, .8, .75, .7)
PRODIGY_VERSION = "2.0.1"
DEFAULT_STEPS = {"prodigy": 2000, "adamw": 2000}
REGULARIZER_RAMP_STEPS = 400


def validate_settings(mode, optimizer, use_speed, steps, seed, early_stop=True, patience=100, min_delta=.001):
    if mode not in MODES:
        raise ValueError(f"Optimization mode must be one of {MODES}")
    if optimizer not in ("prodigy", "adamw"):
        raise ValueError("Optimization optimizer must be prodigy or adamw")
    if type(use_speed) is not bool:
        raise ValueError("optimization_use_speed must be true or false")
    if type(steps) is not int or not 1 <= steps <= 4000:
        raise ValueError("optimization_steps must be an integer from 1 to 4000")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("optimization_seed must be an integer from 0 to 4294967295")
    if type(early_stop) is not bool:
        raise ValueError("optimization_early_stop must be true or false")
    if type(patience) is not int or not 25 <= patience <= 4000:
        raise ValueError("optimization_patience must be an integer from 25 to 4000")
    if type(min_delta) not in (int, float) or not math.isfinite(min_delta) or not 0 <= min_delta < 1:
        raise ValueError("optimization_min_delta must be a finite relative improvement from 0 (inclusive) to 1 (exclusive)")


def settings(request, formats, probe=False):
    result = {"mode": request.optimization_mode, "optimizer": request.optimization_optimizer,
              "use_speed": request.optimization_use_speed, "steps": request.optimization_steps,
              "seed": request.optimization_seed, "tile_target_elements": TILE_ELEMENTS,
              "early_stop": request.optimization_early_stop, "patience": request.optimization_patience,
              "min_delta": request.optimization_min_delta, "minimum_steps": 100,
              "svd_rank_limit": SVD_RANK, "scale_ratios": list(SCALE_RATIOS),
              "calibration": "none (data-free)", "applied": request.format != "original"}
    mode = result["mode"]
    if request.format == "original":
        if mode != "standard":
            raise ValueError("Original precision export requires Standard; there are no quantization parameters to optimize")
    if mode == "thorough":
        unsupported = set(formats) - set(THOROUGH_FORMATS)
        if unsupported:
            raise ValueError("Thorough is unavailable for " + ", ".join(sorted(unsupported))
                             + ": no verified learned-rounding adapter.")
        if probe:
            optimizer_class(result)
    result["objective"] = "tile_svd_weighted_mse" if mode == "thorough" else "weight_mse"
    result["optimizer_active"] = mode == "thorough"
    result["optimizer_name"] = dict((value, label) for label, value in OPTIMIZER_CHOICES)[result["optimizer"]]
    result["prodigy_version"] = PRODIGY_VERSION if mode == "thorough" and result["optimizer"] == "prodigy" else None
    result["optimizer_parameters"] = optimizer_parameters(result) if mode == "thorough" else None
    result["objective_parameters"] = ({"projection_weight": 1., "weight_mse_growth_limit": 1.05,
        "binary_regularizer_weight": .01, "binary_regularizer_ramp_steps": min(result["steps"], REGULARIZER_RAMP_STEPS),
        "evaluate_every_steps": 25, "svd_iterations": 2} if mode == "thorough" else {})
    return result


def optimizer_class(options):
    if options["optimizer"] == "adamw":
        from torch.optim import AdamW
        return AdamW
    try:
        version = importlib.metadata.version("prodigy-plus-schedule-free")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != PRODIGY_VERSION:
        raise ValueError(f"Thorough with Prodigy + Schedule-Free requires optional prodigy-plus-schedule-free=={PRODIGY_VERSION}; found {version or 'not installed'}. "
                         "Install that version separately without upgrading host dependencies, or select AdamW, Balanced or Standard.")
    from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
    return ProdigyPlusScheduleFree


def optimizer_parameters(options):
    common = dict(lr=.05, betas=(.9, .99), eps=1e-8, weight_decay=0.)
    if options["optimizer"] == "prodigy":
        common.update(lr=1., d0=1e-4, use_schedulefree=False, use_speed=options["use_speed"],
                      factored=True, factored_fp32=True, stochastic_rounding=False, split_groups=False)
    return common


def _rotate(weight, format, group):
    if format not in ROTATED_FORMATS:
        return weight
    # Audited Kitchen helper; capability probes exercise this API and its layout.
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight
    return _rotate_weight(weight, _build_hadamard(group, weight.device, weight.dtype), group)


def _units(params, format):
    """Return exactly representable stored scales, before broadcasting to weights."""
    import comfy_kitchen as ck
    import torch
    if format == "nvfp4":
        rows, cols = params.orig_shape
        return ck.from_blocked(params.block_scale, (rows + 15) // 16 * 16, (cols + 15) // 16).float()
    if format == "mxfp8":
        rows, cols = params.orig_shape
        raw = ck.from_blocked(params.scale, (rows + 31) // 32 * 32, (cols + 31) // 32).view(torch.uint8)
        return torch.pow(2., raw.float() - 127.)
    return params.scale.float()


def _expanded(units, shape, format, params):
    if format == "nvfp4":
        return (units * params.scale).repeat_interleave(16, dim=1)
    if format == "mxfp8":
        return units.repeat_interleave(32, dim=1)
    if format in ("convrot_w4a4", "int8_convrot"):
        return units.reshape(-1, 1)
    return units


def _with_units(params, units, format):
    import comfy_kitchen as ck
    import torch
    if format == "nvfp4":
        return replace(params, block_scale=ck.to_blocked(units.to(torch.float8_e4m3fn), flatten=False))
    if format == "mxfp8":
        raw = (units.log2().round() + 127).clamp(0, 254).to(torch.uint8)
        return replace(params, scale=ck.to_blocked(raw, flatten=False).view(params.scale.dtype))
    return replace(params, scale=units.reshape(params.scale.shape).float())


def _levels(format, device):
    import torch
    if format in ("int8_tensorwise", "int8_convrot", "convrot_w4a4"):
        return torch.arange(-7 if format == "convrot_w4a4" else -128,
                            8 if format == "convrot_w4a4" else 128, device=device, dtype=torch.float32)
    if format == "nvfp4":
        return torch.tensor([-6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.], device=device)
    dtype = torch.float8_e4m3fn if format == "mxfp8" else getattr(torch, format)
    values = torch.arange(256, device=device).to(torch.uint8).view(dtype).float()
    return values[torch.isfinite(values)].unique(sorted=True)


def _neighbors(values, levels):
    import torch
    index = torch.searchsorted(levels, values.contiguous()).clamp(0, len(levels) - 1)
    return levels[(index - 1).clamp_min(0)], levels[index]


def _nearest(values, levels, format):
    import torch
    if format in ("int8_tensorwise", "int8_convrot", "convrot_w4a4"):
        return values.round().clamp(float(levels[0]), float(levels[-1]))
    if format.startswith("float8_") or format == "mxfp8":
        dtype = torch.float8_e4m3fn if format == "mxfp8" else getattr(torch, format)
        return values.clamp(float(levels[0]), float(levels[-1])).to(dtype).float()
    lower, upper = _neighbors(values, levels)
    return torch.where(values - lower <= upper - values, lower, upper)


def _encode(values, format):
    import torch
    from comfy_kitchen.float_utils import pack_uint4
    if format == "convrot_w4a4":
        return pack_uint4((values.to(torch.int8) & 15).to(torch.uint8), hi_first=False).view(torch.int8)
    if format == "nvfp4":
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=values.device)
        codes = torch.searchsorted(lut, values.abs().contiguous()).to(torch.uint8)
        return pack_uint4(codes | ((values < 0).to(torch.uint8) << 3), hi_first=True)
    if format in ("int8_tensorwise", "int8_convrot"):
        return values.to(torch.int8)
    return values.to(torch.float8_e4m3fn if format == "mxfp8" else getattr(torch, format))


def _unit_error(decoded, target, format):
    difference = (decoded - target).square()
    if format in ("nvfp4", "mxfp8"):
        return difference.reshape(target.shape[0], -1, 16 if format == "nvfp4" else 32).sum(-1)
    if format in ("convrot_w4a4", "int8_convrot"):
        return difference.sum(-1).reshape(-1, 1)
    return difference.sum()


def _balanced(target, layout, qdata, params, format, checkpoint, ratios=SCALE_RATIOS):
    import torch
    units = _units(params, format)
    levels = _levels(format, target.device)
    # Baseline codes remain available even at exact rounding ties.
    baseline = _rotate(layout.dequantize(qdata, params).float(), format,
                       getattr(params, "convrot_groupsize", 256))
    if baseline.shape != target.shape:
        raise ValueError("Kitchen reconstruction disagrees with padded optimization shape")
    best_error = _unit_error(baseline, target, format)
    best_units = units.clone()
    best_values = baseline / _expanded(units, target.shape, format, params).clamp_min(torch.finfo(torch.float32).tiny)
    best_values = _nearest(best_values, levels, format)
    for ratio in ratios[1:]:
        checkpoint(f"Balanced scale search ({ratio:g})")
        candidate = units * ratio
        if format == "nvfp4":
            candidate = candidate.clamp(max=448).to(torch.float8_e4m3fn).float()
        elif format == "mxfp8":
            # E8M0 supports powers of two only; include the adjacent lower exponent.
            candidate = torch.pow(2., (units.log2() - (1 if ratio <= .8 else 0)).clamp(-127, 127))
        candidate_params = _with_units(params, candidate, format)
        scale = _expanded(candidate, target.shape, format, candidate_params)
        values = _nearest(target / scale.clamp_min(torch.finfo(torch.float32).tiny), levels, format)
        error = _unit_error(values * scale, target, format)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_units = torch.where(better.reshape(units.shape), candidate, best_units)
        mask = better.repeat_interleave(16 if format == "nvfp4" else 32, dim=1) if format in MICROSCALED_FORMATS else better
        best_values = torch.where(mask, values, best_values)
    return _encode(best_values, format), _with_units(params, best_units, format)


def _balanced_grouped(weight, layout, qdata, params, format, checkpoint):
    """Search through Kitchen's complete grouped INT8 decode, including both scales."""
    import torch
    from .backend import quantize_layout
    error = (layout.dequantize(qdata, params).float() - weight).square().sum(-1)
    for ratio in SCALE_RATIOS[1:]:
        checkpoint(f"Balanced grouped-scale search ({ratio:g})")
        limit = weight.abs().amax(-1, keepdim=True) * ratio
        _, codes, candidate = quantize_layout(weight.clamp(-limit, limit), format)
        score = (layout.dequantize(codes, candidate).float() - weight).square().sum(-1)
        better = score < error
        qdata = torch.where(better[:, None], codes, qdata)
        params = replace(params, scale=torch.where(better[:, None], candidate.scale, params.scale),
                         s_channel=torch.where(better, candidate.s_channel, params.s_channel))
        error = torch.where(better, score, error)
    return qdata, params


def _objective(decoded, target, vectors):
    difference = decoded - target
    mse = difference.square().mean()
    return mse + (difference @ vectors).square().mean(), mse


def _grouped_neighbors(target, params, bits):
    """Bracket each weight on its group's actual INT8-decoded grid, retaining code IDs."""
    import torch
    if (params.codebook is not None or params.correction is not None or params.mma_packed
            or params.group_size != 16 or params.convrot_groupsize != 256):
        raise ValueError("Grouped learned rounding requires canonical symmetric uniform storage, group 16 and rotation 256")
    # Kitchen's symmetric 6-bit quantizer uses codes 1..63 (levels -31..31).
    first = 1 if bits == 6 else 0
    codes = torch.arange(first, 1 << bits, device=target.device, dtype=torch.int32)
    levels = ((codes.float() - (1 << (bits - 1))) * params.scale.float().reshape(-1, 1))
    levels = levels.round().clamp(-127, 127)
    levels *= params.s_channel.float()[:, None].expand_as(params.scale).reshape(-1, 1)
    values = target.reshape(-1, params.group_size).contiguous()
    upper_index = torch.searchsorted(levels.contiguous(), values).clamp(max=len(codes) - 1)
    lower_index = (upper_index - 1).clamp_min(0)
    return (levels.gather(1, lower_index).reshape_as(target),
            levels.gather(1, upper_index).reshape_as(target),
            codes[lower_index].reshape_as(target), codes[upper_index].reshape_as(target))


def _learn(target, original, layout, qdata, params, format, group, options, tile_seed, checkpoint, progress, observe):
    import torch
    grouped = format in GROUPED_INT8_FORMATS
    if grouped:
        bits = 4 if format == "asym_w4a8_int8" else 6
        lower, upper, lower_codes, upper_codes = _grouped_neighbors(target, params, bits)
        scale = torch.ones((), device=target.device)
    else:
        levels = _levels(format, target.device)
        scale = _expanded(_units(params, format), target.shape, format, params)
        lower, upper = _neighbors(target / scale.clamp_min(torch.finfo(torch.float32).tiny), levels)
    fraction = ((target / scale.clamp_min(torch.finfo(torch.float32).tiny) - lower)
                / (upper - lower).clamp_min(1e-12)).clamp(1e-4, 1 - 1e-4)
    logits = torch.logit(fraction).detach().requires_grad_(True)
    checkpoint("Thorough: bounded data-free SVD")
    rank = min(SVD_RANK, *target.shape)
    _, _, vectors = torch.svd_lowrank(target, q=rank, niter=2)
    vectors = vectors.detach()
    best_codes = qdata
    reconstruction = layout.dequantize(qdata, params).float()
    decoded = _rotate(reconstruction, format, group)
    best_score, _ = _objective(decoded, target, vectors)
    best_score = best_score.item()
    base_mse = (reconstruction[:len(original), :original.shape[1]] - original).square().mean().item()
    baseline_score = best_score
    significant_score, last_improvement = best_score, 0
    early_stop = options.get("early_stop", True)
    patience, min_delta = options.get("patience", 100), options.get("min_delta", .001)
    steps, accepted, reason = 0, 0, None
    stop_reason = "max_steps"
    started = time.monotonic()

    def event(status):
        observe({"step": steps, "max_steps": options["steps"], "status": status,
                 "stop_reason": stop_reason if status == "completed" else None,
                 "best_objective": best_score, "balanced_objective": baseline_score,
                 "last_improvement_step": last_improvement, "patience": patience,
                 "min_delta": min_delta, "early_stop": early_stop,
                 "elapsed_seconds": time.monotonic() - started})

    event("running")
    energy = target.square().mean().clamp_min(1e-20)
    cls = optimizer_class(options)
    exact = early_stop and baseline_score == 0
    optimizer = None if exact else cls([logits], **optimizer_parameters(options))
    if exact:
        stop_reason = "exact_reconstruction"
    for step in range(0 if exact else options["steps"]):
        checkpoint(f"Thorough learned rounding {step + 1}/{options['steps']}")
        optimizer.zero_grad(set_to_none=True)
        probability = logits.sigmoid()
        soft = (lower + probability * (upper - lower)) * scale
        score, _ = _objective(soft, target, vectors)
        regularizer = (probability * (1 - probability)).mean()
        # A higher cap extends the same trajectory instead of delaying rounding regularization.
        ramp = min(1., (step + 1) / min(options["steps"], REGULARIZER_RAMP_STEPS))
        loss = score / energy + .01 * ramp * regularizer
        if not torch.isfinite(loss):
            reason = "non-finite optimization loss; retained best accepted discrete candidate"
            stop_reason = "non_finite_loss"
            break
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            reason = "non-finite optimization gradient; retained best accepted discrete candidate"
            stop_reason = "non_finite_gradient"
            break
        optimizer.step()
        steps += 1
        if not torch.isfinite(logits).all():
            reason = "non-finite optimization state; retained best accepted discrete candidate"
            stop_reason = "non_finite_state"
            break
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == options["steps"]:
            progress((step + 1) / options["steps"], "Learning quantization rounding")
            with torch.no_grad():
                if grouped:
                    from comfy_kitchen.tensor.w4a8_packed import _pack_codes
                    codes = _pack_codes(torch.where(logits >= 0, upper_codes, lower_codes), bits)
                else:
                    codes = _encode(torch.where(logits >= 0, upper, lower), format)
                reconstruction = layout.dequantize(codes, params).float()
                actual = _rotate(reconstruction, format, group)
                candidate_score, _ = _objective(actual, target, vectors)
                candidate_mse = (reconstruction[:len(original), :original.shape[1]] - original).square().mean().item()
                # Limit weight-error growth while improving the declared SVD objective.
                if (torch.isfinite(candidate_score) and candidate_score.item() < best_score
                        and candidate_mse <= base_mse * 1.05):
                    best_codes, best_score = codes, candidate_score.item()
                    accepted += 1
                # Small accepted improvements still survive; only material cumulative gain resets patience.
                if best_score < significant_score * (1 - min_delta):
                    significant_score, last_improvement = best_score, steps
                if early_stop and steps < options["steps"] and (best_score == 0 or (steps >= 100 and steps - last_improvement >= patience)):
                    stop_reason = "exact_reconstruction" if best_score == 0 else "plateau"
                event("running")
                if stop_reason != "max_steps":
                    break
    event("completed")
    if not accepted and reason is None:
        reason = "learned rounding did not improve the accepted Balanced candidate"
    return best_codes, {"steps": steps, "planned_steps": options["steps"], "stop_reason": stop_reason,
                        "early_stopped": stop_reason in ("plateau", "exact_reconstruction"),
                        "last_improvement_step": last_improvement, "accepted_checkpoints": accepted,
                        "balanced_objective": baseline_score, "final_objective": best_score,
                        "svd_rank": rank, "fallback": reason, "tile_seed": tile_seed}


def quantize_optimized(read_rows, shape, name, format, device, options, checkpoint=lambda stage: None,
                       group=256, tile_elements=TILE_ELEMENTS, progress=lambda fraction, phase: None,
                       observe=lambda event: None):
    """Read bounded rows; serialize one layer's packed CPU output, never its FP32 matrix."""
    import comfy_kitchen as ck
    import torch
    from comfy_kitchen.tensor import QuantizedTensor
    from .backend import quantize_layout

    options = {"early_stop": True, "patience": 100, "min_delta": .001, **options, "minimum_steps": 100}
    validate_settings(options["mode"], options["optimizer"], options["use_speed"], options["steps"], options["seed"],
                      options.get("early_stop", True), options.get("patience", 100), options.get("min_delta", .001))
    options["optimizer_name"] = dict((value, label) for label, value in OPTIMIZER_CHOICES)[options["optimizer"]]
    resolve_convrot_group(1024, group)
    if format not in ADJUSTABLE_ROTATED_FORMATS and group != 256:
        raise ValueError(f"{format} requires fixed ConvRot group 256 in this exporter")
    if options["mode"] not in ("balanced", "thorough"):
        raise ValueError("Optimized exporter requires Balanced or Thorough")
    if options["mode"] == "thorough" and format not in THOROUGH_FORMATS:
        raise ValueError(f"Thorough has no verified learned-rounding adapter for {format}")
    if options["mode"] == "thorough":
        optimizer_class(options)
    started = time.monotonic()
    n, k = shape
    rows = max(128, tile_elements // k // 128 * 128)
    scalar = format.startswith("float8_") or format == "int8_tensorwise"
    global_scale = None
    totals = Counter()
    fallbacks = Counter()
    tile_reports = []
    output = {}

    def read(start, stop):
        checkpoint(f"Read optimization rows {start}:{stop}")
        value = read_rows(start, stop).to(device=device, dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite source weight: {name}")
        return value

    with ck.use_backend("eager") if format in MICROSCALED_FORMATS else nullcontext():
        if scalar or format == "nvfp4":
            maximum = 0.
            for start in range(0, n, rows):
                tile = read(start, min(start + rows, n))
                maximum = max(maximum, tile.abs().amax().item())
                del tile
            progress(.05, "Checking quantization scales")
            sample = torch.tensor([[-maximum, maximum]], device=device)
            _, _, parameters = quantize_layout(sample, format, group)
            global_scale = parameters.scale
            del sample, parameters
        chosen_scale = global_scale
        if scalar:
            errors = [0.] * len(SCALE_RATIOS)
            for start in range(0, n, rows):
                tile = read(start, min(start + rows, n))
                for index, ratio in enumerate(SCALE_RATIOS):
                    checkpoint(f"Balanced global-scale search ({ratio:g}), rows {start}:{start + len(tile)}")
                    layout, codes, params = quantize_layout(tile, format, group, global_scale * ratio)
                    errors[index] += (layout.dequantize(codes, params).float() - tile).square().sum(dtype=torch.float64).item()
                del tile, codes, params
                progress(.05 + .15 * min(start + rows, n) / n, "Searching quantization scales")
            chosen_scale = global_scale * SCALE_RATIOS[min(range(len(errors)), key=errors.__getitem__)]
        specs = storage_contract(format, shape)
        dtypes = {"I8": torch.int8, "U8": torch.uint8, "F32": torch.float32,
                  "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
        output = {name + suffix: torch.empty(spec["shape"], dtype=dtypes[spec["dtype"]]) for suffix, spec in specs.items()}
        for start in range(0, n, rows):
            stop = min(start + rows, n)
            progress(.2 + .8 * start / n, "Optimizing quantization scales")
            weight = read(start, stop)
            layout, base_codes, base_params = quantize_layout(weight, format, group, global_scale)
            baseline_decoded = layout.dequantize(base_codes, base_params).float()[:stop-start, :k]
            totals["baseline_sse"] += (baseline_decoded - weight).square().sum(dtype=torch.float64).item()
            del baseline_decoded
            if scalar:
                _, codes, params = quantize_layout(weight, format, group, chosen_scale)
            elif format in GROUPED_INT8_FORMATS:
                codes, params = _balanced_grouped(weight, layout, base_codes, base_params, format, checkpoint)
            else:
                target = _rotate(weight, format, group)
                padded = layout.dequantize(base_codes, base_params).shape
                target = torch.nn.functional.pad(target, (0, padded[1]-k, 0, padded[0]-len(weight)))
                codes, params = _balanced(target, layout, base_codes, base_params, format, checkpoint)
                # Check actual Kitchen reconstruction; a search approximation cannot beat its audit.
                actual = layout.dequantize(codes, params).float()[:len(weight), :k]
                base = layout.dequantize(base_codes, base_params).float()[:len(weight), :k]
                if (actual - weight).square().sum() > (base - weight).square().sum():
                    codes, params = base_codes, base_params
                    fallbacks["scale search retained Standard after final-grid audit"] += 1
                del actual, base, target
            balanced_decoded = layout.dequantize(codes, params).float()[:len(weight), :k]
            totals["balanced_sse"] += (balanced_decoded - weight).square().sum(dtype=torch.float64).item()
            del balanced_decoded, base_codes, base_params
            info = {"row_start": start, "row_stop": stop, "steps": 0, "planned_steps": 0,
                    "stop_reason": "scale_search_complete" if options["mode"] == "balanced" else "zero_source"}
            if options["mode"] == "thorough" and weight.abs().amax() > 0:
                target = _rotate(weight, format, group)
                padded = layout.dequantize(codes, params).shape
                target = torch.nn.functional.pad(target, (0, padded[1]-k, 0, padded[0]-len(weight)))
                tile_seed = (options["seed"] + int.from_bytes(hashlib.sha256(f"{name}:{start}".encode()).digest()[:4], "little")) % 2**32
                devices = [weight.device.index] if weight.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(tile_seed)
                    codes, learned = _learn(target, weight, layout, codes, params, format, group,
                                            options, tile_seed, checkpoint,
                                            lambda fraction, phase: progress(.2 + .8 * (start + (stop-start) * (.1 + .9*fraction)) / n, phase),
                                            lambda event: observe({**event, "name": name, "tile_index": start // rows + 1,
                                                "tile_count": (n + rows - 1) // rows, "row_start": start, "row_stop": stop}))
                info.update(learned)
                totals["balanced_objective_sum"] += learned["balanced_objective"] * target.numel()
                totals["final_objective_sum"] += learned["final_objective"] * target.numel()
                totals["objective_elements"] += target.numel()
                if learned["fallback"]:
                    fallbacks[learned["fallback"]] += 1
                del target
            totals["steps"] += info["steps"]
            decoded = layout.dequantize(codes, params).float()[:len(weight), :k]
            if not torch.isfinite(decoded).all():
                raise ValueError(f"Non-finite optimized reconstruction: {name}")
            totals["final_sse"] += (decoded - weight).square().sum(dtype=torch.float64).item()
            quant = QuantizedTensor(codes, layout.__name__, params)
            state = quant.state_dict(name)
            if set(state) != set(output):
                raise ValueError("Optimized Kitchen serialization has missing or unexpected fields")
            for key, value in state.items():
                destination = output[key] if value.ndim == 0 else output[key][start:start + value.shape[0]]
                if destination.shape != value.shape:
                    raise ValueError(f"Optimized storage alignment mismatch: {key}")
                if value.element_size() == 1:
                    destination.view(torch.uint8).copy_(value.view(torch.uint8).cpu())
                else:
                    destination.copy_(value.cpu())
            tile_reports.append(info)
            del state, quant, codes, params, decoded, weight
            progress(.2 + .8 * stop / n, "Saving optimized weights")
    output[name.removesuffix("weight") + "comfy_quant"] = torch.tensor(
        list(json.dumps(quantization_config(format, group), sort_keys=True).encode()), dtype=torch.uint8)
    for key, value in output.items():
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"Non-finite optimized storage: {key}")
    count = n * k
    report = {**options, "elapsed_seconds": time.monotonic() - started, "rows_per_tile": rows,
              "binary_regularizer_ramp_steps": min(options["steps"], REGULARIZER_RAMP_STEPS) if options["mode"] == "thorough" else None,
              "optimizer_parameters": optimizer_parameters(options) if options["mode"] == "thorough" else None,
              "baseline_weight_mse": totals["baseline_sse"] / count,
              "balanced_weight_mse": totals["balanced_sse"] / count,
              "final_weight_mse": totals["final_sse"] / count, "actual_steps": totals["steps"],
              "planned_steps": sum(tile["planned_steps"] for tile in tile_reports),
              "early_stopped_tiles": sum(tile.get("early_stopped", False) for tile in tile_reports),
              "stop_reasons": dict(Counter(tile["stop_reason"] for tile in tile_reports)) if options["mode"] == "thorough" else {},
              "balanced_objective": totals["balanced_objective_sum"] / max(1, totals["objective_elements"]),
              "final_objective": totals["final_objective_sum"] / max(1, totals["objective_elements"]),
              "fallbacks": dict(fallbacks), "tiles": tile_reports,
              "weight_mse_growth_limit": 1.05 if options["mode"] == "thorough" else 1.}
    if options["mode"] == "balanced":
        report["balanced_objective"] = report["final_objective"] = report["final_weight_mse"]
    if not all(math.isfinite(report[key]) for key in ("baseline_weight_mse", "balanced_weight_mse", "final_weight_mse",
                                                      "balanced_objective", "final_objective")):
        raise FloatingPointError(f"Non-finite final optimization metrics: {name}")
    if report["balanced_weight_mse"] >= report["baseline_weight_mse"]:
        report["fallbacks"]["scale search found no improvement over Standard"] = 1
    return output, report["final_weight_mse"], report

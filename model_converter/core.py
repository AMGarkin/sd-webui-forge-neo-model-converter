"""Deterministic plans and strict conversion without host imports."""

from dataclasses import asdict, dataclass
from collections import Counter
import json
from pathlib import Path
import time
import os
import tempfile
import shutil
import errno

from .backend import FORMATS, capabilities, quantize, quantize_nvfp4_tiled, NVFP4_TILE_ELEMENTS, versions
from .io import inspect, read_header, StreamWriter, TensorReader
from .layouts import inspect_canonical, CanonicalReader, input_files
from .profiles import select_profile, detect_profile, universal_selection, dimension_anchor, qwen3vl_identity, PROFILES
from .policies import PrecisionPolicy
from .references import ReferenceProtection
from .profile_registry import profile_label
from .paths import clean_path, absolute_paths
from .devices import normalize_device, resolve_device
from .optimization import DEFAULT_STEPS, validate_settings, settings as optimization_settings, quantize_optimized
from .formats import (ROTATED_FORMATS, ADJUSTABLE_ROTATED_FORMATS, CONVROT_GROUP_SIZES,
                      resolve_convrot_group, export_engine, payload_bytes, quantization_config, storage_contract)


@dataclass
class Request:
    source: str
    output: str
    format: str = "float8_e4m3fn"
    device: str = "auto"
    profile: str = "auto"
    allow_experimental: bool = False
    save_report: bool = False
    precision_policy: str = "uniform"
    keep_first_blocks: int = 0
    keep_last_blocks: int = 0
    preserve_patterns: str = ""
    layer_rules: str = ""
    pruned_protection: bool = False
    protect_general: bool = True
    protect_model: bool = True
    use_profile_formats: bool = False
    source_kind: str = "local"
    hf_repo: str = ""
    hf_revision: str = ""
    hf_component: str = ""
    hf_set: str = ""
    hf_cache_dir: str = ""
    hf_keep: bool = True
    source_provenance: dict = None
    protect_layers: bool = True
    convrot_group_size: int = 256
    dynamic_convrot: bool = False
    optimization_mode: str = "standard"
    optimization_optimizer: str = "prodigy"
    optimization_use_speed: bool = False
    optimization_steps: int | None = None
    optimization_seed: int = 0
    optimization_early_stop: bool = True
    optimization_patience: int = 100
    optimization_min_delta: float = .001

    def __post_init__(self):
        self.source = clean_path(self.source)
        self.output = clean_path(self.output)
        self.device = normalize_device(self.device)
        resolve_convrot_group(1024, self.convrot_group_size, self.dynamic_convrot)
        if self.optimization_steps is None and self.optimization_optimizer in ("prodigy", "adamw"):
            self.optimization_steps = DEFAULT_STEPS[self.optimization_optimizer]
        validate_settings(self.optimization_mode, self.optimization_optimizer, self.optimization_use_speed,
                          self.optimization_steps, self.optimization_seed, self.optimization_early_stop,
                          self.optimization_patience, self.optimization_min_delta)
        if self.source_kind not in ("local", "huggingface"):
            raise ValueError("Unknown source kind")
        if type(self.hf_keep) is not bool:
            raise ValueError("hf_keep must be true or false")
        if self.source_kind == "huggingface" and not self.hf_cache_dir:
            raise ValueError("HF requires an application-owned cache directory")


def plan(request, probe=True, emit=lambda e: None, cancelled=lambda: None):
    cancelled()
    if request.source_kind == "huggingface":
        optimization_settings(request, (), probe)
        from .huggingface import Acquisition
        with Acquisition(request, emit, cancelled) as acquisition:
            prepared = acquisition.prepare(download=False)
            return prepared if isinstance(prepared, dict) else _plan_local(prepared, probe=probe)
    from .huggingface import local_lease
    with local_lease(request.source):
        return _plan_local(request, probe=probe)


def _plan_local(request, probe=True):
    request = Request(**asdict(request))
    request.source, request.output = absolute_paths(request.source, request.output)
    requested_device = request.device
    requested_profile = request.profile
    request.device = "cpu" if request.format == "original" else resolve_device(request.device)
    if request.format not in (*FORMATS, "original"):
        raise ValueError("Unsupported format; no silent fallback is permitted")
    tensors, source_metadata, source_layout = inspect_canonical(request.source)
    output = Path(request.output).resolve()
    if output.suffix != ".safetensors" or not output.parent.is_dir():
        raise ValueError("Output requires an existing directory and .safetensors suffix")
    if output.exists() or str(output) in {spec["file"] for spec in tensors.values()}:
        raise FileExistsError("Output already exists or aliases a source")
    if request.save_report and Path(str(output) + ".report.json").exists():
        raise FileExistsError("Output report already exists")
    if request.profile == "auto":
        request.profile = detect_profile(tensors, request.format)
    for field in ("pruned_protection", "protect_general", "protect_model", "use_profile_formats", "protect_layers"):
        if type(getattr(request, field)) is not bool:
            raise ValueError(f"{field} must be true or false")
    if not request.protect_layers or request.format == "original":
        request.protect_general = request.protect_model = request.pruned_protection = request.use_profile_formats = False
        request.keep_first_blocks = request.keep_last_blocks = 0
        request.preserve_patterns = ""
        if request.precision_policy == "preserve_attention" or request.format == "original":
            request.precision_policy = "uniform"
        if request.format == "original":
            request.layer_rules = ""
    native = request.profile in PROFILES
    if native:
        select_profile(tensors, request.profile)  # Signature validation, not a projection allowlist.
    references = ReferenceProtection(request.profile, tensors, request.pruned_protection,
                                     request.protect_model, request.use_profile_formats)
    catalog = references.star
    entries, estimate, precision = _plan_layers(request, tensors, references, native)
    extra = {"requested": request.pruned_protection, "available": references.extra_available,
             "enabled": references.extra_enabled, "additional_preserved_weights": 0}
    if references.extra_enabled:
        baseline = ReferenceProtection(request.profile, tensors, False, request.protect_model, request.use_profile_formats)
        try:
            baseline_entries, _, _ = _plan_layers(request, tensors, baseline, native)
            extra["additional_preserved_weights"] = sum(current["action"] == "preserve" and previous["action"] == "quantize"
                for current, previous in zip(entries, baseline_entries))
        except ValueError as error:
            extra["additional_preserved_weights"] = None
            extra["comparison_note"] = str(error)
    if request.format != "original" and not any(e["action"] == "quantize" for e in entries):
        raise ValueError("No eligible linear weights for this policy and format. The profile or advanced rules may preserve all candidates. Convolutional weights (common in VAE/vae_approx), lookup tables, small matrices and incompatible rotation widths are preserved. Try the optional preview or another format; this converter does not quantize convolutions.")
    format_counts = {format: sum(e["action"] == "quantize" and e["format"] == format for e in entries)
                     for format in FORMATS if any(e["format"] == format for e in entries)}
    convrot_groups = {fmt: sorted({e["convrot_group_size"] for e in entries
                                  if e["action"] == "quantize" and e["format"] == fmt})
                      for fmt in format_counts if fmt in ROTATED_FORMATS}
    optimization = optimization_settings(request, format_counts, probe)
    availability = capabilities(request.device, formats=tuple(format_counts), convrot_groups=convrot_groups) if probe and format_counts else {}
    for format, capability in availability.items():
        if not capability["available"]:
            raise ValueError(f"{format}: {capability['reason']}")
    if probe and request.optimization_mode != "standard":
        import torch
        for fmt in format_counts:
            for group in convrot_groups.get(fmt, (256,)):
                width = max(256, group)
                sample = torch.linspace(-1, 1, 8 * width).reshape(8, width)
                try:
                    quantize_optimized(lambda start, stop: sample[start:stop], sample.shape, "probe.weight", fmt,
                                       request.device, {**optimization, "steps": 1, "early_stop": False}, group=group)
                except Exception as error:
                    raise ValueError(f"{request.optimization_mode.title()} {fmt} API probe failed: {error}") from error
            availability[fmt]["optimization_probe"] = {"mode": request.optimization_mode, "available": True,
                "steps": 1 if request.optimization_mode == "thorough" else 0}
    engines = {format: export_engine(format) for format in format_counts}
    policy_notes = precision.notes()
    if requested_profile == "auto" and request.profile == "universal":
        policy_notes.insert(0, "No known standalone model profile matched; using Universal selection.")
    policy_notes.extend(references.notes(tensors))
    policy_warnings = precision.notes() + references.warnings(tensors)
    if not request.protect_layers and request.format != "original":
        policy_warnings.append("All layer protections are disabled, including embeddings and packed dimension probes. Quality or target loading can fail. Unsupported tensor types and format dimensions still cannot be quantized.")
    incompatible_widths = Counter(e["shape"][1] for e in entries if e["action"] == "preserve"
        and e["reason"].startswith("Preserve matrix whose input width is not divisible by rotation group "))
    if incompatible_widths:
        widths = ", ".join(f"{width} ({count})" for width, count in sorted(incompatible_widths.items()))
        policy_warnings.append(f"{sum(incompatible_widths.values())} matrices kept in source precision: input widths are incompatible with the selected ConvRot groups. "
                               f"Widths (matrix counts): {widths}. This is a format requirement, not layer protection. "
                               "Choose FP8 or INT8 without ConvRot, or set compatible per-layer formats. No format was substituted.")
    packing_limits = sum(e["reason"].startswith("Preserve matrix whose input width is not divisible by packing group") for e in entries)
    if packing_limits:
        policy_warnings.append(f"{packing_limits} matrices kept in source precision: independent packing dimensions still apply (INT4: 64, W4A8: 16, W6A8: 32).")
    fixed_groups = [fmt for fmt in format_counts if fmt in ROTATED_FORMATS and fmt not in ADJUSTABLE_ROTATED_FORMATS]
    if fixed_groups and (request.dynamic_convrot or request.convrot_group_size != 256):
        policy_warnings.append("W4A8/W6A8 retain ConvRot group 256: the audited target CUDA loaders reject other groups. Adjustable groups apply only to INT8 ConvRot and INT4 ConvRot.")
    if request.pruned_protection and not references.extra_enabled:
        policy_warnings.append("Extra protection is inactive: " + ("model-specific protection is disabled." if not request.protect_model
                               else "this profile defines no extra protection rules."))
    if request.pruned_protection and catalog is None:
        policy_notes.append("Extra pruned-model protection requires a matching Star recipe; the selected policy has none.")
    loader_notes = []
    loader_notes.extend(source_layout.get("warnings", []))
    if source_layout["adapter"] in ("universal-identity", "transformers-identity"):
        loader_notes.append(source_layout["status"])
    if "encoder.feature_extractor.mel_mean" in tensors and "decoder.layernorm_embedding.weight" in tensors:
        loader_notes.append("SheetSage2: the inspected ComfyUI audio-encoder loader uses ordinary manual_cast operations and does not consume comfy_quant. This export needs a quantization-aware loader; audio inference is not validated.")
    identity = qwen3vl_identity(tensors)
    label = profile_label(request.profile)
    if identity:
        label = identity["label"] + " · " + label
    return {"request": asdict(request), "layers": entries, "capabilities": availability,
            "source_metadata": source_metadata, "source_layout": source_layout,
            "device_selection": {"requested": requested_device, "resolved": request.device},
            "policy_selection": {"requested": requested_profile, "resolved": request.profile, "label": label},
            "model_identity": identity,
            "optimization_selection": optimization,
            "convrot_selection": {"requested_group_size": request.convrot_group_size, "dynamic": request.dynamic_convrot,
                                   "maximum_group_size": max(CONVROT_GROUP_SIZES), "groups_by_format": convrot_groups,
                                   "counts": dict(sorted(Counter(str(e["convrot_group_size"]) for e in entries
                                       if e["action"] == "quantize" and e["format"] in ROTATED_FORMATS).items()))},
            "precision_selection": precision.details(), "format_counts": format_counts,
            "profile_reference": references.star_details(tensors),
            "protection_selection": references.details(tensors),
            "policy_notes": policy_notes, "policy_warnings": policy_warnings,
            "protection_settings": {"applied": request.format != "original", "all": request.protect_layers, "general": request.protect_general, "model": request.protect_model,
                                    "profile_formats": request.use_profile_formats, "extra": extra},
            "loader_notes": loader_notes,
            "export_engine": "safetensors streaming (no quantization)" if not engines else next(iter(engines.values())) if len(set(engines.values())) == 1 else "mixed Kitchen engines",
            "export_engines": engines,
            "source_reader": "owned CPU buffers from positional reads",
            "nvfp4_processing": {"passes": 2, "global_scale": True, "tile_target_elements": NVFP4_TILE_ELEMENTS,
                                  "row_alignment": 128} if "nvfp4" in format_counts else None,
            "estimated_payload_bytes": estimate, "largest_input_tensor_bytes": max(
                e["data_offsets"][1] - e["data_offsets"][0] for e in entries),
            "compatibility": "experimental; this run performs structural validation only; see docs/USER_GUIDE.md#compatibility for separate loader and user-reported generation evidence",
            "warnings": ["Payload estimate excludes header and is not a VRAM or speed estimate",
                         "Two output-sized disk buffers are required during finalization",
                         *([] if request.format == "original" else ["full_precision_matrix_mult=true: quantized storage with compute-dtype matmul"])] + (
                         ["Universal selection infers linear weights from names and shapes. The host must support comfy_quant for these modules; key remapping and full-model compatibility are not implied."]
                         if request.profile == "universal" and request.format != "original" else []) + loader_notes + policy_notes,
            "versions": versions()}


def _plan_layers(request, tensors, references, native):
    precision = PrecisionPolicy(request, tensors)
    entries = []
    estimate = 0
    for name, spec in sorted(tensors.items()):
        shape = spec["shape"]
        if request.format == "original":
            estimate += spec["data_offsets"][1] - spec["data_offsets"][0]
            entries.append({"name": name, **spec, "action": "preserve", "format": None,
                            "reason": "Original precision: canonical export without quantization", "protections": []})
            continue
        layer_format = request.format
        group_size = None
        selected, reason = universal_selection(name, spec, "float8_e4m3fn",
                                                minimum_dimension=(8 if native else 64) if request.protect_layers else 1,
                                                protect_general=request.protect_general, protect_lookup=request.protect_layers)
        protections = references.protection(name)
        if protections:
            selected, reason = False, protections[0]
        if not selected:
            precision.record_protected(name)
        if selected:
            layer_format, catalog_reason = references.format_override(name, layer_format)
            reason = catalog_reason or reason
            layer_format, precision_reason = precision.select(name, layer_format)
            selected = layer_format is not None
            reason = precision_reason or reason
        if selected:
            if request.protect_layers:
                references.check_format(layer_format)
            protections = references.protection(name, layer_format)
            if protections:
                selected = False
                precision.block_assignment()
                reason = protections[0]
        if selected and layer_format in ROTATED_FORMATS:
            group_size = resolve_convrot_group(shape[1], request.convrot_group_size, request.dynamic_convrot) if layer_format in ADJUSTABLE_ROTATED_FORMATS else resolve_convrot_group(shape[1])
            group_label = (f"dynamic minimum {request.convrot_group_size} (maximum 1024)" if request.dynamic_convrot
                           else str(request.convrot_group_size)) if layer_format in ADJUSTABLE_ROTATED_FORMATS else "256"
        if selected and request.protect_layers and dimension_anchor(name):
            if storage_contract(layer_format, shape)[""]["shape"] != shape:
                selected = False
                precision.block_assignment()
                reason = "Preserve model-detection dimension anchor: packed/padded storage would change the shape read by the host"
            elif layer_format in ROTATED_FORMATS and group_size is None:
                selected = False
                precision.block_assignment()
                reason = f"Preserve model-detection dimension anchor: input width is incompatible with rotation group {group_label}"
        if selected and layer_format in ROTATED_FORMATS and group_size is None:
            if (native and request.protect_layers) or reason.startswith("Layer rule"):
                raise ValueError(f"Rotation group {group_label} must divide input width for {layer_format}: {name}")
            selected = False
            precision.block_assignment()
            reason = f"Preserve matrix whose input width is not divisible by rotation group {group_label} for {layer_format}"
        packing = {"convrot_w4a4": 64, "asym_w4a8_int8": 16, "w6a8_int8": 32}.get(layer_format, 1)
        if selected and shape[1] % packing:
            selected = False
            precision.block_assignment()
            reason = f"Preserve matrix whose input width is not divisible by packing group {packing} for {layer_format}"
        size = spec["data_offsets"][1] - spec["data_offsets"][0]
        if selected:
            size = payload_bytes(layer_format, shape)
            size += len(json.dumps(quantization_config(layer_format, group_size or 256), sort_keys=True).encode("utf-8"))
        estimate += size
        if not selected and not protections:
            protections = [reason]
        entries.append({"name": name, **spec, "action": "quantize" if selected else "preserve", "reason": reason,
                        "format": layer_format if selected else None, "protections": protections,
                        "convrot_group_size": group_size if selected else None})
    return entries, estimate, precision


def validate_output(path, conversion_plan):
    import torch
    from safetensors import safe_open
    header, metadata = read_header(path)
    if metadata != conversion_plan["source_metadata"]:
        raise ValueError("Source metadata changed during export")
    expected = set()
    with safe_open(path, framework="pt", device="cpu") as standard_reader:
        if set(standard_reader.keys()) != set(header):
            raise ValueError("Safetensors parser disagrees with output keys")
    with TensorReader(path) as reader:
        for entry in conversion_plan["layers"]:
            name, shape = entry["name"], entry["shape"]
            expected.add(name)
            if entry["action"] == "preserve":
                if header[name]["shape"] != shape or header[name]["dtype"] != entry["dtype"]:
                    raise ValueError(f"Preserved tensor changed structure: {name}")
                continue
            module = name.removesuffix("weight")
            config_key = module + "comfy_quant"
            expected.add(config_key)
            config_tensor = reader.get_tensor(config_key)
            if config_tensor.dtype != torch.uint8 or config_tensor.ndim != 1:
                raise ValueError("Configuration must be a uint8 vector")
            config = json.loads(bytes(config_tensor.tolist()).decode("utf-8"))
            format = entry.get("format", conversion_plan["request"]["format"])
            if config != quantization_config(format, entry.get("convrot_group_size") or 256):
                raise ValueError(f"Configuration mismatch: {config_key}")
            for suffix, spec in storage_contract(format, shape).items():
                key = name + suffix
                expected.add(key)
                if header.get(key, {}).get("shape") != spec["shape"] or header.get(key, {}).get("dtype") != spec["dtype"]:
                    raise ValueError(f"Invalid storage shape or dtype: {key}")
                value = reader.get_tensor(key)
                if not torch.isfinite(value.float()).all():
                    raise ValueError(f"Non-finite output tensor: {key}")
                if suffix:
                    if format == "mxfp8" and (value == 255).any():
                        raise ValueError(f"Invalid E8M0 scale encoding: {key}")
                    if (value.float() < 0).any() or (suffix == "_scale_2" and not (value > 0).all()):
                        raise ValueError(f"Invalid scale: {key}")
    if set(header) != expected:
        raise ValueError("Unexpected or missing output tensors")


def convert(request, emit=lambda event: None, cancelled=lambda: None, work_dir=None):
    from .huggingface import Acquisition, local_lease
    if not request.allow_experimental:
        raise ValueError("Explicit experimental opt-in required until whole-model validation is recorded")
    if request.source_kind == "huggingface":
        optimization_settings(request, (), probe=True)
        output = Path(request.output).resolve()
        if output.suffix != ".safetensors" or not output.parent.is_dir():
            raise ValueError("Output requires an existing directory and .safetensors suffix")
        if output.exists() or (request.save_report and Path(str(output) + ".report.json").exists()):
            raise FileExistsError("Output or report already exists; no source downloaded")
        with Acquisition(request, emit, cancelled) as acquisition:
            prepared = acquisition.prepare()
            # Download occupies the first 20 percent; conversion stays monotonic.
            def progress(event):
                if event.get("event") == "progress":
                    event = {**event, "fraction": .2 + .8 * event.get("fraction", 0)}
                emit(event)
            report = _convert_local(prepared, progress, cancelled, work_dir, defer_report=True)
            try:
                report["source_cleanup"] = acquisition.cleanup_success()
            except Exception as error:
                report["status"] = "failed"
                report["source_cleanup"] = str(error)
                failure = RuntimeError(f"Validated checkpoint saved at {prepared.output}, but source cleanup failed: {error}. Inspect the retained source before retrying cleanup.")
                failure.conversion_report = report
                raise failure from None
            if prepared.save_report:
                try:
                    _publish_report(report, work_dir)
                except Exception as error:
                    report.update(status="failed", failure_stage="report_publication")
                    failure = RuntimeError(f"Validated checkpoint saved at {prepared.output}, but report publication failed: {error}")
                    failure.conversion_report = report
                    raise failure from None
            return report
    with local_lease(request.source):
        return _convert_local(request, emit, cancelled, work_dir)


def _convert_local(request, emit=lambda event: None, cancelled=lambda: None, work_dir=None, defer_report=False):
    import torch
    if not request.allow_experimental:
        raise ValueError("Explicit experimental opt-in required until whole-model validation is recorded")
    started = time.monotonic()
    request = Request(**asdict(request))
    emit({"event": "progress", "phase": "Checking model and available formats", "fraction": 0})
    conversion_plan = _plan_local(request)
    request = Request(**conversion_plan["request"])
    emit({"event": "progress", "phase": f"Using {request.device.upper()} · {conversion_plan['policy_selection']['label']}", "fraction": 0})
    emit({"event": "prepared", "device": request.device, "profile": request.profile,
          "profile_label": conversion_plan["policy_selection"]["label"],
          "format": request.format, "export_engine": conversion_plan["export_engine"],
          "precision_policy": request.precision_policy, "format_counts": conversion_plan["format_counts"],
          "policy_warnings": conversion_plan["policy_warnings"],
          "protection_settings": conversion_plan["protection_settings"],
          "convrot_selection": conversion_plan["convrot_selection"],
          "optimization_selection": conversion_plan["optimization_selection"],
          "quantize": sum(e["action"] == "quantize" for e in conversion_plan["layers"]),
          "preserve": sum(e["action"] == "preserve" for e in conversion_plan["layers"])})
    required_disk = conversion_plan["estimated_payload_bytes"] * 2 + 16 * 1024 * 1024
    free_disk = shutil.disk_usage(Path(request.output).resolve().parent).free
    if free_disk < required_disk:
        raise OSError(errno.ENOSPC, f"Not enough free disk space. Approximately {required_disk / 1024**3:.2f} GiB required; {free_disk / 1024**3:.2f} GiB available.")
    cancelled()
    report = {"status": "running", "plan": conversion_plan, "layers": [], "fallbacks": [],
              "validation": {"structural": False, "model_loader": False, "generation": False}}
    source_stats = {f: (Path(f).stat().st_size, Path(f).stat().st_mtime_ns)
                    for e in conversion_plan["layers"] for f in input_files(e)}
    writer = StreamWriter(request.output, work_dir=work_dir, metadata=conversion_plan["source_metadata"])
    total_bytes = sum(e["data_offsets"][1] - e["data_offsets"][0] for e in conversion_plan["layers"])
    completed_bytes = 0
    stage = "read"
    try:
        for index, entry in enumerate(conversion_plan["layers"]):
            cancelled()
            stage = "read"
            last_step_event = 0.
            def checkpoint(operation):
                nonlocal last_step_event
                cancelled()
                if operation.startswith("Thorough learned rounding"):
                    now = time.monotonic()
                    if now - last_step_event < .25:
                        return
                    last_step_event = now
                emit({"event": "tensor", "name": entry["name"], "shape": entry["shape"], "operation": operation})
            def optimization_progress(fraction, phase):
                size = entry["data_offsets"][1] - entry["data_offsets"][0]
                emit({"event": "progress", "phase": phase,
                      "fraction": .85 * (completed_bytes + fraction * size) / max(total_bytes, 1),
                      "completed": index, "total": len(conversion_plan["layers"]), "name": entry["name"]})
            checkpoint("Read source tensor")
            stat = Path(entry["file"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != source_stats[entry["file"]]:
                raise ValueError("Source changed during conversion")
            with CanonicalReader(entry) as reader:
                mse = None
                tensor = None
                layer_format = entry["format"]
                optimization_report = None
                if entry["action"] == "quantize" and request.optimization_mode != "standard":
                    stage = "quantize"
                    result, mse, optimization_report = quantize_optimized(
                        lambda start, stop: reader.get_tensor(entry["name"], start, stop),
                        entry["shape"], entry["name"], layer_format, request.device,
                        conversion_plan["optimization_selection"], checkpoint,
                        group=entry.get("convrot_group_size") or 256, progress=optimization_progress,
                        observe=lambda event: emit({**event, "event": "optimization"}))
                    for reason, count in optimization_report["fallbacks"].items():
                        report["fallbacks"].append({"name": entry["name"], "reason": reason, "occurrences": count})
                elif entry["action"] == "quantize" and layer_format == "nvfp4":
                    stage = "quantize"
                    result, mse = quantize_nvfp4_tiled(
                        lambda start, stop: reader.get_tensor(entry["name"], start, stop),
                        entry["shape"], entry["name"], request.device, checkpoint)
                else:
                    tensor = reader.get_tensor(entry["name"])
                    checkpoint("Check source values")
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"Non-finite source tensor: {entry['name']}")
                    if entry["action"] == "quantize":
                        stage = "quantize"
                        result, mse = quantize(tensor, entry["name"], layer_format, request.device, checkpoint,
                                               convrot_group_size=entry.get("convrot_group_size") or 256)
                if entry["action"] == "quantize":
                    stage = "write"
                    for name, value in result.items():
                        checkpoint("Write quantized tensors")
                        writer.add(name, value)
                    del result, value
                else:
                    stage = "write"
                    checkpoint("Write preserved tensor")
                    writer.add(entry["name"], tensor)
                del tensor
            report["layers"].append({"name": entry["name"], "action": entry["action"], "format": entry["format"],
                                     "convrot_group_size": entry.get("convrot_group_size"),
                                     "reason": entry["reason"], "final_weight_mse": mse,
                                     "optimization": optimization_report})
            completed_bytes += entry["data_offsets"][1] - entry["data_offsets"][0]
            emit({"event": "progress", "phase": "Saving original-precision tensors" if request.format == "original" else "Converting weights", "fraction": 0.85 * completed_bytes / max(total_bytes, 1),
                  "completed": index + 1, "total": len(conversion_plan["layers"]), "name": entry["name"]})
        for source, expected in source_stats.items():
            stat = Path(source).stat()
            if (stat.st_size, stat.st_mtime_ns) != expected:
                raise ValueError("Source changed during conversion")
        stage = "validate_and_publish"
        writer.finish(lambda path: validate_output(path, conversion_plan), cancelled, emit)
    except Exception as error:
        report["status"] = "failed_or_cancelled"
        report["counts"] = {action: sum(e["action"] == action for e in report["layers"])
                            for action in ("quantize", "preserve")}
        report["counts"]["failed"] = int(stage in ("read", "quantize", "write") and type(error).__name__ != "Cancelled")
        report["counts"]["unprocessed"] = len(conversion_plan["layers"]) - len(report["layers"]) - report["counts"]["failed"]
        report["failure_stage"] = stage
        report["last_attempted_tensor"] = entry["name"] if "entry" in locals() else None
        report["error"] = f"{type(error).__name__}: {error}"
        error.conversion_report = report
        raise
    finally:
        writer.close()
    report.update(status="completed", elapsed_seconds=time.monotonic() - started,
                  output_bytes=Path(request.output).stat().st_size)
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(request.device) if request.device.startswith("cuda") else None
    report["validation"]["structural"] = True
    report["counts"] = {action: sum(e["action"] == action for e in report["layers"])
                        for action in ("quantize", "preserve")}
    report["counts"]["failed"] = 0
    optimized = [e["optimization"] for e in report["layers"] if e["optimization"]]
    report["optimization"] = {"mode": request.optimization_mode, "optimized_layers": len(optimized),
        "actual_steps": sum(e["actual_steps"] for e in optimized),
        "planned_steps": sum(e["planned_steps"] for e in optimized),
        "early_stopped_tiles": sum(e["early_stopped_tiles"] for e in optimized),
        "elapsed_seconds": sum(e["elapsed_seconds"] for e in optimized),
        "fallbacks": len(report["fallbacks"]),
        "improved_weight_mse_layers": sum(e["final_weight_mse"] < e["baseline_weight_mse"] for e in optimized),
        "improved_objective_layers": sum(e["final_objective"] < e["balanced_objective"] for e in optimized)}
    if not request.save_report or defer_report:
        return report
    return _publish_report(report, work_dir)


def _publish_report(report, work_dir=None):
    report_path = Path(report["plan"]["request"]["output"] + ".report.json")
    # Report publication is a separate transaction. Any failure remains an error,
    # even though the already validated checkpoint may have been published.
    with tempfile.TemporaryDirectory(prefix=".converter-report-", dir=work_dir or report_path.parent) as folder:
        pending = Path(folder) / "report.json"
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, report_path)
    return report

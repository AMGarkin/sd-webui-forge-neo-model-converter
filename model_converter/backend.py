"""Explicit Kitchen APIs and canonical ComfyUI serialization."""

import importlib.metadata
import json
import platform
from contextlib import nullcontext
from .devices import resolve_device
from .formats import (FORMATS, MICROSCALED_FORMATS, ADJUSTABLE_ROTATED_FORMATS,
                      export_engine, quantization_config, resolve_convrot_group)


NVFP4_TILE_ELEMENTS = 1024 * 1024


def quantize_nvfp4_tiled(read_rows, shape, name, device, checkpoint=lambda stage: None,
                         tile_elements=NVFP4_TILE_ELEMENTS):
    """Two-pass global-scale export with bounded FP32 working buffers.

    A tile starts at a multiple of 128 rows, so its canonical blocked scales
    occupy consecutive complete row blocks in the final scale tensor. Keep
    Kitchen's global scale shared across all tiles, never a per-tile scale.
    """
    import comfy_kitchen as ck
    import torch
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
    from comfy_kitchen.tensor import QuantizedTensor
    from .formats import storage_contract

    n, k = shape
    rows_per_tile = max(128, tile_elements // k // 128 * 128)
    maximum = 0.
    for start in range(0, n, rows_per_tile):
        stop = min(start + rows_per_tile, n)
        checkpoint(f"Scan NVFP4 global scale, rows {start}:{stop}")
        tile = read_rows(start, stop).to(device=device, dtype=torch.float32)
        if not torch.isfinite(tile).all():
            raise ValueError(f"Non-finite source weight: {name}")
        maximum = max(maximum, tile.abs().amax().item())
        del tile
    maximum_tensor = torch.tensor(maximum, dtype=torch.float32, device=device)
    scale = (maximum_tensor / (F4_E2M1_MAX * F8_E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.where(maximum_tensor == 0, torch.ones_like(scale), scale)
    specs = storage_contract("nvfp4", shape)
    packed = torch.empty(specs[""]["shape"], dtype=torch.uint8)
    scale_bytes = torch.zeros(specs["_scale"]["shape"], dtype=torch.uint8)
    squared_error = 0.
    with ck.use_backend("eager"):
        for start in range(0, n, rows_per_tile):
            stop = min(start + rows_per_tile, n)
            checkpoint(f"Kitchen NVFP4 quantization, rows {start}:{stop}")
            tile = read_rows(start, stop).to(device=device, dtype=torch.float32)
            if not torch.isfinite(tile).all():
                raise ValueError(f"Non-finite source weight: {name}")
            quant = QuantizedTensor.from_float(tile, "TensorCoreNVFP4Layout", scale=scale)
            state = quant.state_dict(name)
            if not torch.isfinite(state[name + "_scale"].float()).all():
                raise ValueError(f"Non-finite NVFP4 scales: {name}")
            checkpoint(f"Reconstruct NVFP4 and measure error, rows {start}:{stop}")
            decoded = quant.dequantize().float()
            if not torch.isfinite(decoded).all():
                raise ValueError(f"Non-finite reconstruction: {name}")
            squared_error += (decoded - tile).square().sum(dtype=torch.float64).item()
            qdata, block_scales = state[name], state[name + "_scale"]
            packed[start:start + qdata.shape[0]].copy_(qdata.cpu())
            scale_bytes[start:start + block_scales.shape[0]].copy_(block_scales.view(torch.uint8).cpu())
            del state, quant, decoded, tile, qdata, block_scales
    config = torch.tensor(list(json.dumps(quantization_config("nvfp4"), sort_keys=True).encode()), dtype=torch.uint8)
    return {name: packed, name + "_scale": scale_bytes.view(torch.float8_e4m3fn),
            name + "_scale_2": scale.cpu(), name.removesuffix("weight") + "comfy_quant": config}, squared_error / (n * k)


def quantize(weight, name, format, device, checkpoint=lambda stage: None, convrot_group_size=256, group_scale_dtype="F32"):
    import comfy_kitchen as ck
    resolve_convrot_group(1024, convrot_group_size)
    if format not in ADJUSTABLE_ROTATED_FORMATS and convrot_group_size != 256:
        raise ValueError(f"{format} does not support adjustable ConvRot groups in the target CUDA loader")
    with ck.use_backend("eager") if format in MICROSCALED_FORMATS else nullcontext():
        return _quantize(weight, name, format, device, checkpoint, convrot_group_size, group_scale_dtype)


def _quantize(weight, name, format, device, checkpoint, convrot_group_size, group_scale_dtype):
    import torch
    from comfy_kitchen.tensor import QuantizedTensor

    checkpoint("Convert weight to FP32")
    weight = weight.to(device=device, dtype=torch.float32)
    if not torch.isfinite(weight).all():
        raise ValueError(f"Non-finite source weight: {name}")
    config = quantization_config(format, convrot_group_size)
    checkpoint("Kitchen quantization")
    layout, qdata, params = quantize_layout(weight, format, convrot_group_size, group_scale_dtype=group_scale_dtype)
    quant = QuantizedTensor(qdata, layout.__name__, params)
    checkpoint("Serialize quantized tensors")
    result = quant.state_dict(name)
    if format == "mxfp8":
        # Raw E8M0 bytes are part of the loader contract.
        result[name + "_scale"] = result[name + "_scale"].view(torch.uint8)
    if any(key.endswith("_correction") for key in result):
        raise ValueError("Loader does not support weight_correction")
    for key, tensor in result.items():
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"Non-finite quantized tensor: {key}")
    checkpoint("Dequantize for error measurement")
    reconstruction = quant.dequantize().float()
    if not torch.isfinite(reconstruction).all():
        raise ValueError("Non-finite reconstruction")
    checkpoint("Measure final weight error")
    error = (reconstruction - weight).square().mean().item()
    result[name.removesuffix("weight") + "comfy_quant"] = torch.tensor(
        list(json.dumps(config, sort_keys=True).encode("utf-8")), dtype=torch.uint8)
    return result, error


def quantize_layout(weight, format, convrot_group_size=256, scale=None, group_scale_dtype="F32"):
    """Share exact base settings with optimized candidates through Kitchen layouts."""
    import torch
    from comfy_kitchen.tensor.base import get_layout_class

    kwargs = {}
    if format.startswith("float8_"):
        dtype = getattr(torch, format)
        if scale is None:
            scale = (weight.abs().amax() / torch.finfo(dtype).max).clamp_min(torch.finfo(torch.float32).tiny)
        layout_name, kwargs = "TensorCoreFP8Layout", dict(dtype=dtype, scale=scale)
    elif format in ("int8_tensorwise", "int8_convrot"):
        rotated = format == "int8_convrot"
        layout_name = "TensorWiseINT8Layout"
        kwargs = dict(per_channel=rotated, convrot=rotated, convrot_groupsize=convrot_group_size, scale=scale)
    elif format in ("asym_w4a8_int8", "w6a8_int8"):
        if group_scale_dtype not in ("F32", "F8_E4M3"):
            raise ValueError("Grouped scale dtype must be F32 or F8_E4M3")
        layout_name = "AsymW4A8Int8Layout"
        kwargs = dict(group_size=16, convrot_groupsize=256, symmetric=True,
                      bits=4 if format == "asym_w4a8_int8" else 6, codebook=False,
                      scale_dtype=torch.float8_e4m3fn if group_scale_dtype == "F8_E4M3" else torch.float32,
                      stochastic_rounding=0, scale_search=False)
    elif format == "convrot_w4a4":
        layout_name = "TensorCoreConvRotW4A4Layout"
        kwargs = dict(convrot_groupsize=convrot_group_size, quant_group_size=64, stochastic_rounding=0, linear_dtype="int4")
    elif format == "nvfp4":
        from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
        if scale is None:
            maximum = weight.abs().amax()
            scale = (maximum / (F4_E2M1_MAX * F8_E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny)
            scale = torch.where(maximum == 0, torch.ones_like(scale), scale)
        layout_name, kwargs = "TensorCoreNVFP4Layout", dict(scale=scale)
    elif format == "mxfp8":
        layout_name = "TensorCoreMXFP8Layout"
    else:
        raise ValueError(f"Format has no verified exporter: {format}")
    layout = get_layout_class(layout_name)
    qdata, params = layout.quantize(weight, **kwargs)
    return layout, qdata, params


def capabilities(device="cpu", formats=None, convrot_groups=None, group_scale_dtypes=None):
    device = resolve_device(device)
    result = {}
    for format in (FORMATS if formats is None else formats):
        try:
            import torch
            groups = (convrot_groups or {}).get(format, (256,))
            for group in groups:
                width = max(256, group)
                sample = torch.linspace(-1, 1, 16 * width).reshape(16, width)
                quantize(sample, "probe.weight", format, device, convrot_group_size=group,
                         group_scale_dtype=(group_scale_dtypes or {}).get(format, "F32"))
            result[format] = {"available": True, "reason": "Kitchen quantize/serialize/dequantize probe passed",
                              "export_engine": export_engine(format), "convrot_groups": list(groups),
                              "group_scale_dtype": (group_scale_dtypes or {}).get(format, "F32")}
        except Exception as error:
            result[format] = {"available": False, "reason": f"{type(error).__name__}: {error}"}
    return result


def versions():
    result = {"python": platform.python_version(), "converter": "0.1.0.dev0",
              "comfy_contract_reference": "e9afbccc2669677076feb08e52658f129cd8c12b",
              "layer_loader_test_reference": "860a7fdc7772bcb53f74a675d0592d89f94fa265",
              "forge_loader_test_reference": "a057d9ede903eb51c38ceb20e186b6841665d250"}
    for package in ("torch", "comfy-kitchen", "safetensors", "prodigy-plus-schedule-free"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result

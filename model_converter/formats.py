"""Canonical on-disk tensor contracts, independent of host/runtime imports."""

import math


FORMATS = ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise", "int8_convrot",
           "asym_w4a8_int8", "w6a8_int8", "convrot_w4a4", "nvfp4", "mxfp8")
ROTATED_FORMATS = ("int8_convrot", "asym_w4a8_int8", "w6a8_int8", "convrot_w4a4")
ADJUSTABLE_ROTATED_FORMATS = ("int8_convrot", "convrot_w4a4")
CONVROT_GROUP_SIZES = (4, 16, 64, 256, 1024)
MICROSCALED_FORMATS = ("nvfp4", "mxfp8")

FORMAT_SUFFIXES = {
    "float8_e4m3fn": "fp8_e4m3", "float8_e5m2": "fp8_e5m2",
    "int8_tensorwise": "int8", "int8_convrot": "int8_convrot",
    "asym_w4a8_int8": "w4a8_convrot", "w6a8_int8": "w6a8_convrot",
    "convrot_w4a4": "w4a4_convrot", "nvfp4": "nvfp4", "mxfp8": "mxfp8",
}


def roundup(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def resolve_convrot_group(width, group_size=256, dynamic=False):
    if type(group_size) is not int or group_size not in CONVROT_GROUP_SIZES:
        raise ValueError(f"ConvRot group size must be one of {CONVROT_GROUP_SIZES}")
    if type(dynamic) is not bool:
        raise ValueError("dynamic_convrot must be true or false")
    # Bound the cached dense Hadamard matrix to 1024 squared elements.
    choices = reversed(CONVROT_GROUP_SIZES) if dynamic else (group_size,)
    return next((size for size in choices if size >= group_size and width % size == 0), None)


def quantization_config(format, convrot_group_size=256):
    if format not in FORMATS:
        raise ValueError(f"Unsupported quantization format: {format}")
    config = {"format": "int8_tensorwise" if format == "int8_convrot" else format,
              "full_precision_matrix_mult": True}
    if format == "int8_tensorwise":
        config["convrot"] = False
    if format in ROTATED_FORMATS:
        config["convrot_groupsize"] = convrot_group_size
    if format == "int8_convrot":
        config["convrot"] = True
    if format in ("asym_w4a8_int8", "w6a8_int8"):
        config["group_size"] = 16
    if format == "convrot_w4a4":
        config["linear_dtype"] = "int4"
    return config


def export_engine(format):
    # Explicit software export avoids Blackwell-only cast kernels selected by
    # some Kitchen installations on older CUDA devices. No retry/fallback.
    return "kitchen-eager" if format in MICROSCALED_FORMATS else "kitchen-dispatch"


def storage_contract(format, shape):
    """Map weight suffixes to canonical safetensors dtype and storage shape."""
    n, k = shape
    if format in MICROSCALED_FORMATS:
        block = 16 if format == "nvfp4" else 32
        rows, cols = roundup(n, block), roundup(k, block)
        result = {
            "": {"dtype": "U8" if format == "nvfp4" else "F8_E4M3",
                 "shape": [rows, cols // 2 if format == "nvfp4" else cols]},
            "_scale": {"dtype": "F8_E4M3" if format == "nvfp4" else "U8",
                       "shape": [roundup(rows, 128), roundup(cols // block, 4)]},
        }
        if format == "nvfp4":
            result["_scale_2"] = {"dtype": "F32", "shape": []}
        return result
    if format in ("asym_w4a8_int8", "w6a8_int8"):
        return {"": {"dtype": "I8", "shape": [n, k // 2 if format == "asym_w4a8_int8" else 3 * k // 4]},
                "_s_rel": {"dtype": "F32", "shape": [n, k // 16]},
                "_s_channel": {"dtype": "F32", "shape": [n]}}
    if format == "convrot_w4a4":
        return {"": {"dtype": "I8", "shape": [n, k // 2]},
                "_scale": {"dtype": "F32", "shape": [n]}}
    if format not in FORMATS:
        raise ValueError(f"Unsupported quantization format: {format}")
    return {"": {"dtype": {"float8_e4m3fn": "F8_E4M3", "float8_e5m2": "F8_E5M2"}.get(format, "I8"),
                 "shape": [n, k]},
            "_scale": {"dtype": "F32", "shape": [n, 1] if format == "int8_convrot" else []}}


def payload_bytes(format, shape):
    return sum(math.prod(spec["shape"]) * (4 if spec["dtype"] == "F32" else 1)
               for spec in storage_contract(format, shape).values())

"""Manual MiniMax H3 precision map; existing Kitchen exporters supply the weights."""


PROFILE = "minimax-h3-compact-w4w6int8-v1"
SOURCE_COMMIT = "3415250db9e2cb89b886f5375c1306fc933e215f"
SOURCE_URL = "https://huggingface.co/LokkenJP/TenEros_Max_optimized_learned/blob/" + SOURCE_COMMIT + "/10Eros_Max_8stepTurbo_v1.0_hybrid_activation_learned_compact_w4w6int8_14.50gb.safetensors.quantization.json"
DESCRIPTION = ("LokkenJP ErosMax v1 Compact precision map for 50 main blocks and two token-refiner blocks. "
               "Enable Use profile format overrides: 64 W4A8, 122 W6A8 and 22 INT8 ConvRot matrices. "
               "Other tensors retain source precision. Requires fixed ConvRot 256. Uses this converter's "
               "symmetric grids with FP8 E4M3 group scales and FP32 channel scales, without the author's learned W4 codebook, activation captures "
               "or discrete polishing. Standard/Balanced/Thorough retain their existing meaning; this is not "
               "a reproduction of the learned checkpoint's values or quality.")


def format_map():
    w4_qkv = {3, 4, 5, 6, 7, 8, 10, 11, 12, 16, 17, 18, 20, 21, 22, 23, 25, 31, 33, 34}
    int8_out = {0, 1, 2, 33, 40, 47, 48, 49}
    result = {}
    for block in range(50):
        for projection in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            fmt = "w6a8_int8"
            if block in (0, 49) or (projection == "attn.out_proj" and block in int8_out):
                fmt = "int8_convrot"
            elif (projection == "attn.qkv_proj" and block in w4_qkv) or (projection == "mlp.fc1" and 2 <= block <= 45):
                fmt = "asym_w4a8_int8"
            result[f"blocks.{block}.{projection}.weight"] = fmt
    for block in range(2):
        for projection in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            result[f"token_refiner.blocks.{block}.{projection}.weight"] = "int8_convrot"
    return result


FORMATS = format_map()


def validate_layout(tensors):
    """A source-specific map must not silently truncate or extrapolate block indices."""
    if not FORMATS.keys() <= tensors.keys() or any(
        key not in tensors for key in ("video_patch_proj.weight", "condition_proj.weight", "rope.inv_freq")
    ):
        raise ValueError("MiniMax Compact requires a canonical 50-block H3 transformer and two complete token-refiner blocks")
    stacks = {key.split(".")[1] for key in tensors if key.startswith("blocks.")}
    refiners = {key.split(".")[2] for key in tensors if key.startswith("token_refiner.blocks.")}
    if stacks != {str(i) for i in range(50)} or refiners != {"0", "1"}:
        raise ValueError("MiniMax Compact precision map is specific to 50 main blocks and two token-refiner blocks")
    anchor = tensors["video_patch_proj.weight"]["shape"]
    if len(anchor) != 2:
        raise ValueError("Invalid MiniMax Compact hidden-width anchor")
    hidden = anchor[0]
    for stack, count in (("blocks", 50), ("token_refiner.blocks", 2)):
        for block in range(count):
            prefix = f"{stack}.{block}."
            shapes = [tensors[prefix + name + ".weight"]["shape"] for name in
                      ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")]
            if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
                raise ValueError(f"Invalid MiniMax Compact projection shapes: {prefix}")
            qkv, out, fc1, fc2 = shapes
            if qkv != [3 * out[1], hidden] or out[0] != hidden or fc1 != [2 * fc2[1], hidden] or fc2[0] != hidden:
                raise ValueError(f"Inconsistent MiniMax Compact attention/MLP dimensions: {prefix}")


def details():
    return {"profile": PROFILE, "source": SOURCE_URL, "commit": SOURCE_COMMIT,
            "scope": "exact 208-matrix precision assignment; independently implemented from published facts",
            "expected_counts": {"asym_w4a8_int8": 64, "w6a8_int8": 122, "int8_convrot": 22},
            "adaptations": "symmetric Kitchen grids, FP8 group scales, FP32 channel scales, no learned codebook or activation calibration; quality is not reproduced"}

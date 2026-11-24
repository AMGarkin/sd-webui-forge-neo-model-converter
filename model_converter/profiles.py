"""Universal linear-weight selection plus optional architecture policies."""

import re
import json
from pathlib import Path
from .formats import ROTATED_FORMATS
from .reference_data import general_protection, ctq_filter


PROFILE_INFO = {
    "moss-image-comfy-v1": ("MOSS-Image SFT (Comfy layout)",
                            "Convert block projections; preserve conditioning, modulation, norms and output."),
    "flux1-bfl-conservative-v1": ("FLUX.1 (BFL layout)",
                                   "Convert double/single block projections; preserve input, conditioning, modulation and output."),
    "flux2-bfl-conservative-v1": ("FLUX.2 / Klein (BFL layout)",
                                   "Convert double/single block projections; preserve global modulation and model boundaries."),
    "z-image-conservative-v1": ("Z-Image / Turbo (Comfy layout)",
                                "Convert main block attention and MLPs; preserve noise/context refiners, modulation and boundaries."),
    "anima-conservative-v1": ("Anima (Comfy / Forge layout)",
                              "Convert main block attention and MLPs; preserve the LLM adapter, modulation and boundaries."),
    "qwen3-text-conservative-v1": ("Qwen3 text encoder",
                                   "Convert attention and MLP projections; preserve embeddings, norms and output heads."),
}
PROFILES = tuple(PROFILE_INFO)


def profile_description(profile):
    if profile.startswith("ctq:"):
        name = profile.removeprefix("ctq:")
        return (f"CTQ {name}: {ctq_filter(name)['help']}. Combined with matching Star family protections "
                "and general safety rules. Protected tensors retain source precision; no keys are removed or remapped.")
    if profile.startswith("star:"):
        from .catalog import catalog_description
        return catalog_description(profile)
    if profile == "auto":
        return "Detect the model profile; otherwise use Universal. Choose general protection, model-specific protection and profile format rules below."
    if profile == "universal":
        return "Select eligible 2-D linear weights without model-specific rules. General sensitive-layer protection is optional; lookup and storage safety always apply."
    return PROFILE_INFO[profile][1]


def detect_profile(tensors, format=None):
    if qwen3vl_identity(tensors):
        return "ctq:qwen_vlm"
    matches = []
    for profile in PROFILES:
        try:
            select_profile(tensors, profile)
            matches.append(profile)
        except ValueError:
            pass
    if len(matches) > 1:
        raise ValueError("Multiple model profiles match. Select a profile explicitly.")
    profile = matches[0] if matches else "universal"
    if profile == "flux2-bfl-conservative-v1" and format in ("asym_w4a8_int8", "w6a8_int8"):
        return "star:Flux 2 Tight"
    return profile


def qwen3vl_identity(tensors):
    for language, visual in (("model.language_model.", "model.visual."), ("model.", "visual.")):
        anchors = (language + "layers.0.self_attn.q_norm.weight", language + "layers.0.self_attn.k_norm.weight",
                   visual + "deepstack_merger_list.0.norm.weight", visual + "merger.linear_fc2.weight")
        if all(k in tensors for k in anchors):
            shape = tensors.get(language + "embed_tokens.weight", {}).get("shape", [])
            if len(shape) != 2:
                continue
            width = shape[1]
            variant = {2048: "2B", 2560: "4B", 4096: "8B", 5120: "32B"}.get(width, "")
            return {"family": "qwen3_vl", "variant": variant,
                    "label": "Qwen3-VL" + (" " + variant if variant else "")}
    return None


def recognized_layout(tensors, label, required, roots, pattern, prefixes=("", "model.diffusion_model.", "diffusion_model.")):
    """Check standalone layout signatures without loading weight values."""
    matches = [prefix for prefix in prefixes if all(prefix + key in tensors for key in required)]
    if len(matches) != 1:
        raise ValueError(f"Input is not a recognized standalone {label} layout; Diffusers/AIO need adapters")
    prefix = matches[0]
    if any(not name.startswith(tuple(prefix + root for root in roots)) for name in tensors):
        raise ValueError(f"Unexpected components in {label}; AIO and mixed state dicts are unsupported")
    for name in required:
        shape = tensors[prefix + name]["shape"]
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"Invalid {label} matrix: {prefix + name}")
    return prefix, re.compile(pattern)


def dimension_anchor(name):
    """Weights whose raw shape is inspected before quantized modules exist.

    Audited host detectors infer latent/context dimensions from these weights.
    Retain these probes when a packed or padded format changes their shape;
    comfy_quant cannot repair a module that was constructed with the wrong K.
    """
    return any(name == key or name.endswith("." + key) for key in (
        "img_in.weight", "txt_in.weight", "vector_in.in_layer.weight", "input_proj.weight",
        "x_embedder.weight", "context_embedder.weight",
        "txt_in.input_embedder.weight", "cap_embedder.1.weight",
        "transformer_blocks.0.attn2.to_k.weight", "x_embedder.proj.1.weight", "first.weight",
        "condition_proj.weight", "time_embedder.proj_in.weight",
    ))


def universal_selection(name, spec, format, minimum_dimension=64, protect_general=True, protect_lookup=True):
    """Select likely linear weights without requiring a known architecture.

    Names and shapes cannot prove a module type. Preserve lookup tables and
    common sensitive parameters; the target host still needs quantized Linear
    operations for selected weights. This policy does not remap keys.
    """
    shape = spec["shape"]
    if spec["dtype"] not in ("F32", "F16", "BF16"):
        return False, "Preserve non-quantized buffer dtype"
    if not (name == "weight" or name.endswith(".weight")):
        return False, "Preserve bias or buffer"
    if len(shape) != 2:
        return False, "Preserve convolution, normalization, or other non-matrix weight"
    protection = general_protection(name, optional=protect_general, protect_lookup=protect_lookup)
    if protection:
        return False, "Preserve " + protection
    if min(shape) < minimum_dimension:
        return False, f"Preserve small matrix (minimum dimension is {minimum_dimension})"
    if format in ROTATED_FORMATS and shape[1] % 256:
        return False, "Preserve matrix whose input width is not divisible by rotation group 256"
    return True, "General protection policy selects an eligible linear weight"


def select_profile(tensors, profile):
    if profile == "moss-image-comfy-v1":
        required = json.loads(Path(__file__).with_name("moss_image_shapes.json").read_text(encoding="utf-8"))
        for index in range(42):
            for suffix, shape in {"proj.qkv": [9216, 3072], "proj.o": [3072, 3072],
                                  "proj.ffn_up_gate": [16384, 3072], "proj.ffn_down": [3072, 8192],
                                  "norm1": [3072], "norm2": [3072], "q_norm": [128], "k_norm": [128]}.items():
                required[f"layers.{index}.{suffix}.weight"] = shape
        for key, shape in required.items():
            if key not in tensors or tensors[key]["shape"] != shape:
                raise ValueError(f"MOSS-Image SFT canonical layout mismatch: {key}")
        if set(tensors) != set(required):
            raise ValueError("Expected standalone 379-tensor MOSS-Image SFT checkpoint")
        return "", re.compile(r"layers\.\d+\.proj\.(?:qkv|o|ffn_up_gate|ffn_down)\.weight$")
    flux_pattern = r"(?:double_blocks\.\d+\.(?:img|txt)_(?:attn\.(?:qkv|proj)|mlp\.(?:0|2))|single_blocks\.\d+\.linear[12])\.weight$"
    if profile == "flux2-bfl-conservative-v1":
        return recognized_layout(tensors, "FLUX.2 / Klein BFL", (
            "img_in.weight", "txt_in.weight", "double_blocks.0.img_attn.qkv.weight",
            "single_blocks.0.linear1.weight", "final_layer.linear.weight",
            "double_stream_modulation_img.lin.weight", "double_stream_modulation_txt.lin.weight",
            "single_stream_modulation.lin.weight"), (
                "img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.",
                "double_blocks.", "single_blocks.", "final_layer.", "pe_embedder.",
                "double_stream_modulation_img.", "double_stream_modulation_txt.", "single_stream_modulation."), flux_pattern)
    if profile == "z-image-conservative-v1":
        return recognized_layout(tensors, "Z-Image", (
            "x_embedder.weight", "cap_embedder.1.weight", "t_embedder.mlp.0.weight",
            "layers.0.attention.qkv.weight", "layers.0.feed_forward.w1.weight",
            "noise_refiner.0.attention.qkv.weight", "context_refiner.0.attention.qkv.weight",
            "final_layer.linear.weight"), (
                "layers.", "noise_refiner.", "context_refiner.", "x_embedder.", "cap_embedder.",
                "t_embedder.", "final_layer.", "x_pad_token", "cap_pad_token"),
            r"layers\.\d+\.(?:attention\.(?:qkv|out)|feed_forward\.w[123])\.weight$")
    if profile == "anima-conservative-v1":
        return recognized_layout(tensors, "Anima", (
            "blocks.0.self_attn.q_proj.weight", "blocks.0.cross_attn.k_proj.weight",
            "blocks.0.mlp.layer1.weight", "llm_adapter.embed.weight", "final_layer.linear.weight"), (
                "blocks.", "llm_adapter.", "x_embedder.", "t_embedder.", "t_embedding_norm.",
                "final_layer.", "pos_embedder."),
            r"blocks\.\d+\.(?:(?:self_attn|cross_attn)\.(?:q_proj|k_proj|v_proj|output_proj)|mlp\.layer[12])\.weight$")
    if profile == "qwen3-text-conservative-v1":
        if not all(key in tensors for key in ("model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight")):
            raise ValueError("Input lacks the Qwen3 Q/K norm layout signature")
        return recognized_layout(tensors, "Qwen3 text encoder", (
            "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight", "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight"), (
                "model.layers.", "model.embed_tokens.", "model.norm.", "model.rotary_emb.", "lm_head."),
            r"model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$",
            prefixes=("",))
    if profile != "flux1-bfl-conservative-v1":
        raise ValueError("Unsupported model profile")
    prefixes = [p for p in ("", "model.diffusion_model.", "diffusion_model.")
                if all(p + key in tensors for key in (
                    "img_in.weight", "txt_in.weight", "double_blocks.0.img_attn.qkv.weight",
                    "single_blocks.0.linear1.weight", "final_layer.linear.weight"))]
    if len(prefixes) != 1:
        raise ValueError("Input is not a recognized standalone FLUX.1 BFL state dict; Diffusers/AIO need adapters")
    prefix = prefixes[0]
    roots = ("img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.",
             "double_blocks.", "single_blocks.", "final_layer.", "pe_embedder.")
    if any(not key.startswith(tuple(prefix + root for root in roots)) for key in tensors):
        raise ValueError("Unexpected components or keys; AIO and mixed state dicts are unsupported")
    return prefix, re.compile(flux_pattern)

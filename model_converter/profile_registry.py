"""One UI entry per model recipe; legacy source IDs remain supported by the API."""

from functools import lru_cache

from .catalog import catalog_data
from .profiles import PROFILE_INFO
from .reference_data import ctq_data
from .references import CTQ_STARS


ALIASES = {
    "star:MOSS-Image": "moss-image-comfy-v1",
    "star:Flux1 / Flux2": "flux2-bfl-conservative-v1",
    "star:Anima": "anima-conservative-v1",
    "star:Z-Image": "z-image-conservative-v1",
    "star:P-H3-Image": "star:minimax_h3_image",
    "ctq:qwen35": "ctq:qwen_vlm",
    "ctq:zimage_refiner": "z-image-conservative-v1",
}
# Filters with additional component-specific protection remain distinct.
for filter_name, star_name in CTQ_STARS.items():
    if filter_name not in ("gemma4", "qwen_vlm", "qwen35", "t5xxl", "mistral"):
        ALIASES["ctq:" + filter_name] = "star:" + star_name

CTQ_LABELS = {
    "gemma4": "Gemma4 text / multimodal encoder", "qwen_vlm": "Qwen VLM text encoder",
    "t5xxl": "T5-XXL text encoder", "mistral": "Mistral text encoder",
    "visual": "Visual encoder (preserve MLP)", "hunyuan": "Hunyuan Video 1.5",
    "radiance": "Radiance", "nerf_large": "NeRF (large)", "nerf_small": "NeRF (small)",
    "distillation_small": "Chroma (guidance-only protection)",
}
STAR_LABELS = {
    "Flux 2 Tight": "FLUX.2 / Klein (W4A8/W6A8: include MLP)",
    "Text-Encoder": "Text encoder", "Text-Encoder HQ": "Text encoder HQ",
    "Text-Encoder MaxQ": "Text encoder MaxQ", "SDXL (Not NVFP4)": "SDXL",
    "minimax_h3": "MiniMax H3", "minimax_h3_image": "MiniMax H3 Image",
    "minimax_h3_te": "MiniMax H3 text encoder", "minimax_h3_vae": "MiniMax H3 VAE",
    "minimax_h3_ref_nvfp4_fp8": "MiniMax H3 (FP8 attention recipe)",
    "minimax_h3_ref_nvfp4_int8convrot": "MiniMax H3 (INT8 attention recipe)",
    "minimax_h3_int4_tensorwise_experimental": "MiniMax H3 (experimental protection recipe)",
    "minimax_h3_int8convrot_int4fc2": "MiniMax H3 (INT4 fc2 recipe)",
    "minimax_h3_int8convrot_int4mlp": "MiniMax H3 (INT4 MLP recipe)",
}


def canonical_ui_profile(profile):
    while profile in ALIASES:
        profile = ALIASES[profile]
    return profile


@lru_cache(maxsize=1)
def model_profile_choices():
    choices = {key: info[0] for key, info in PROFILE_INFO.items()}
    for name in catalog_data()[0]["models"]:
        key = "star:" + name
        if canonical_ui_profile(key) == key:
            choices[key] = STAR_LABELS.get(name, name)
    for name in ctq_data()[0]["filters"]:
        key = "ctq:" + name
        if canonical_ui_profile(key) == key:
            choices[key] = CTQ_LABELS.get(name, name.replace("_", " "))
    return [("Detect model profile (otherwise universal)", "auto"), ("Universal", "universal"),
            *sorted(((label, key) for key, label in choices.items()), key=lambda item: item[0].casefold())]


def profile_label(profile):
    return dict((key, label) for label, key in model_profile_choices()).get(canonical_ui_profile(profile), profile)

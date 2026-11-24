"""Compose family-specific reference protections with explicit provenance."""

import re

from .catalog import CatalogProfile
from .formats import FORMATS
from .reference_data import ctq_data, ctq_filter, TOOLS_SOURCE, TOOLS_COMMIT
from . import compact


# Map architecture families, never similarly named but unrelated components.
NATIVE_REFERENCES = {
    "moss-image-comfy-v1": ("MOSS-Image", ()),
    "flux1-bfl-conservative-v1": ("Flux1 / Flux2", ()),
    "flux2-bfl-conservative-v1": ("Flux1 / Flux2", ("flux2",)),
    "z-image-conservative-v1": ("Z-Image", ("zimage", "zimage_refiner")),
    "anima-conservative-v1": ("Anima", ("anima",)),
    "qwen3-text-conservative-v1": ("Text-Encoder", ("generic_text",)),
}
STAR_FILTERS = {
    "Anima": ("anima",), "Boogu-Image": ("boogu",), "Chroma": ("distillation_large",),
    "Flux1 / Flux2": ("flux2",), "Flux 2 Tight": ("flux2",),
    "Flux 3 Klein": ("flux2",), "Flux 3 Klein MaxQ": ("flux2",),
    "Ideogram-4": ("ideogram4",), "Krea-2": ("krea2",), "Lens": ("lens",),
    "LTX-Video (All Versions)": ("ltxv2",), "LTX-2.5": ("ltxv2",),
    "ltx25_gemma4_te": ("gemma4",),
    "Qwen-Image": ("qwen",), "Qwen-Image W4A8": ("qwen",),
    "Qwen-Image-2.1": ("qwen",), "Qwen-Image-2.1 MaxQ": ("qwen",),
    "Text-Encoder": ("generic_text",), "Text-Encoder HQ": ("generic_text",),
    "Text-Encoder MaxQ": ("generic_text",), "Wan (All Versions)": ("wan",),
    "Z-Image": ("zimage", "zimage_refiner"), "Z-Image MaxQ": ("zimage", "zimage_refiner"),
    "minimax_h3": ("minimaxh3",), "minimax_h3_ref_nvfp4_fp8": ("minimaxh3",),
    "minimax_h3_ref_nvfp4_int8convrot": ("minimaxh3",),
    "minimax_h3_int4_tensorwise_experimental": ("minimaxh3",),
    "minimax_h3_int8convrot_int4fc2": ("minimaxh3",),
    "minimax_h3_int8convrot_int4mlp": ("minimaxh3",),
    "minimax_h3_te": ("generic_text",),
}
CTQ_STARS = {
    "anima": "Anima", "boogu": "Boogu-Image", "distillation_large": "Chroma",
    "flux2": "Flux1 / Flux2", "ideogram4": "Ideogram-4", "krea2": "Krea-2", "lens": "Lens",
    "ltxv2": "LTX-Video (All Versions)", "minimaxh3": "minimax_h3", "qwen": "Qwen-Image",
    "wan": "Wan (All Versions)", "zimage": "Z-Image", "zimage_refiner": "Z-Image",
    "gemma4": "Text-Encoder", "qwen_vlm": "Text-Encoder", "qwen35": "Text-Encoder",
    "t5xxl": "Text-Encoder", "mistral": "Text-Encoder", "generic_text": "Text-Encoder",
}
LANGUAGE_STACK = re.compile(r"(?:^|\.)(?P<stack>(?:model\.)?language_model\.layers|model\.layers)\.(?P<index>\d+)\.")


class ReferenceProtection:
    def __init__(self, profile, tensors, pruned=False, protect_model=True, use_profile_formats=True):
        self.profile = profile
        self.protect_model = protect_model
        self.use_profile_formats = use_profile_formats
        self.native = profile in NATIVE_REFERENCES
        self.compact = profile == compact.PROFILE
        star_name, filters = None, ()
        if self.native:
            star_name, filters = NATIVE_REFERENCES[profile]
        elif profile.startswith("star:"):
            star_name = profile.removeprefix("star:")
            filters = STAR_FILTERS.get(star_name, ())
        elif profile.startswith("ctq:"):
            filter_name = profile.removeprefix("ctq:")
            ctq_filter(filter_name)  # Reject an unknown choice, never fall back.
            filters = (filter_name,)
            star_name = CTQ_STARS.get(filter_name)
        elif profile != "universal" and not self.compact:
            raise ValueError(f"Unsupported model profile: {profile}")
        self.star = CatalogProfile("star:" + star_name, pruned) if star_name else None
        self.filters = filters
        self.rules = []
        self.dynamic_blocks = {}
        for filter_name in filters:
            config = ctq_filter(filter_name)
            for field in ("exclude", "highprec", "remove"):
                for token in config.get(field, []):
                    self.rules.append((filter_name, field, token))
            if config.get("exclude_final_language_layer"):
                for name in tensors:
                    match = LANGUAGE_STACK.search(name)
                    if match:
                        stack = name[:match.start("index") - 1]
                        self.dynamic_blocks[stack] = max(int(match["index"]),
                            self.dynamic_blocks.get(stack, -1))
        self.pattern_matches = {f"{f}:{field}:{token}": sum(self._matches(name, f, token) for name in tensors)
                                for f, field, token in self.rules}

    @staticmethod
    def _matches(name, filter_name, token):
        if token in name:
            return True
        # Both audited hosts use blocks.* for Anima; CTQ's recipe uses
        # net.blocks.*. Add this explicit layout alias without rewriting keys.
        if filter_name == "anima" and token.startswith("net.blocks."):
            canonical = name.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.").removeprefix("net.")
            return canonical.startswith(token.removeprefix("net."))
        return False

    def protection(self, name, format=None):
        if not self.protect_model:
            return []
        if self.compact:
            return [] if name in compact.FORMATS else ["MiniMax Compact preserves tensors outside its 208-matrix precision map"]
        reasons = []
        if self.star:
            reason = self.star.protection(name, format)
            if reason:
                reasons.append(reason)
        for filter_name, field, token in self.rules:
            if self._matches(name, filter_name, token):
                action = "retain rather than delete" if field == "remove" else "preserve original precision"
                reasons.append(f"CTQ {filter_name}: {action}; {field} matches {token}")
        match = LANGUAGE_STACK.search(name)
        if match:
            stack = name[:match.start("index") - 1]
            if int(match["index"]) == self.dynamic_blocks.get(stack):
                reasons.append(f"CTQ Qwen VLM: preserve final language block {stack}.{match['index']}")
        if self.native and self.profile.startswith(("flux1-", "flux2-")):
            if ".img_mod." in name or ".txt_mod." in name:
                reasons.append("Native conservative FLUX profile preserves block modulation")
        if self.native and self.profile == "anima-conservative-v1":
            if any(token in name for token in ("adaln_mod", ".modulation.")):
                reasons.append("Native conservative Anima profile preserves adaptive modulation")
        return reasons

    def format_override(self, name, format):
        if self.compact and self.use_profile_formats and name in compact.FORMATS:
            target = compact.FORMATS[name]
            return target, f"MiniMax Compact precision map: {name} -> {target}"
        return self.star.format_override(name, format) if self.star and self.use_profile_formats else (format, None)

    @property
    def format_overrides_available(self):
        return self.compact or bool(self.star and any(self.star.recipe.get(field) for field, _ in self.star.overrides))

    @property
    def extra_available(self):
        return bool(self.star and self.star.recipe.get("pruned_extra_blacklist"))

    @property
    def extra_enabled(self):
        return bool(self.protect_model and self.extra_available and self.star.pruned)

    def check_format(self, format):
        if self.star:
            self.star.check_format(format)

    def details(self, tensors):
        data, digest = ctq_data()
        return {"profile": self.profile, "model_protection_enabled": self.protect_model,
                "compact": compact.details() if self.compact else None,
                "profile_format_rules_enabled": self.use_profile_formats,
                "star": self.star_details(tensors),
                "ctq": {"source": data["source"], "commit": data["commit"], "data_sha256": digest,
                        "license": data["license"], "filters": list(self.filters),
                        "pattern_matches": self.pattern_matches, "final_language_blocks": self.dynamic_blocks},
                "general_criteria": {"source": TOOLS_SOURCE, "commit": TOOLS_COMMIT,
                    "implementation": "independent name-criterion implementation; no upstream Python copied",
                    "adaptations": "all eligible linear weights, not indexed blocks only; original lookup tables retained; t_emb uses an underscore boundary to avoid text_embeddings_connector"},
                "covered_formats": list(FORMATS),
                "precedence": "storage/loader and family protection > general eligibility > user precision rules > reference format recipe > selected format",
                "dtype_policy": "all protected tensors retain source dtype and values"}

    def star_details(self, tensors):
        if not self.star:
            return None
        return {**self.star.details(tensors), "protection_enabled": self.protect_model,
                "format_rules_enabled": self.use_profile_formats,
                "pruned_extra_protection": self.extra_enabled}

    def notes(self, tensors):
        if self.compact:
            return [compact.DESCRIPTION]
        source_notes = self.star.notes(tensors) if self.star else []
        notes = source_notes if self.protect_model else [note for note in source_notes
            if self.use_profile_formats and ("AWQ" in note or "tensorwise INT4" in note)]
        for filter_name in self.filters if self.protect_model else ():
            matches = [count for key, count in self.pattern_matches.items() if key.startswith(filter_name + ":")]
            if matches and not any(matches) and not self.dynamic_blocks:
                notes.append(f"CTQ {filter_name}: no model-specific protection patterns matched; check the selected family and tensor names.")
        if self.filters and self.protect_model:
            notes.append("Combined reference protections: CTQ " + ", ".join(self.filters) +
                         (f" + Star {self.star.name}." if self.star else ".") +
                         " Family protections override general eligibility; protected tensors retain source precision.")
        if self.protect_model and any(field == "remove" for _, field, _ in self.rules):
            notes.append("CTQ removal rules are preservation rules here: no decoder, head or other source tensor is deleted.")
        if self.protect_model and any(ctq_filter(name).get("preserve_dtype") for name in self.filters):
            notes.append("CTQ protected-dtype hints retain original source precision; no implicit precision casting is applied.")
        if not self.native and (self.filters or self.star):
            notes.append("Reference profiles are manual family selections; matching tensor names do not prove full-model compatibility.")
        if self.native and self.protect_model:
            notes.append("Native signatures validate the layout; linear selection uses protection rules, without a projection-name allowlist.")
        return notes

    def warnings(self, tensors):
        # Attribution and precision descriptions belong in diagnostics. Only
        # mismatched recipes and unimplemented requested features are warnings.
        return [note for note in self.notes(tensors)
                if ((self.protect_model and "no protection patterns matched" in note)
                    or (self.protect_model and "no model-specific protection patterns matched" in note)
                    or (self.use_profile_formats and ("AWQ is not" in note or "tensorwise INT4" in note)))]

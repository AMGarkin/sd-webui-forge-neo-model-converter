"""Adapt licensed StarNodes profile data without importing its converter code."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .profiles import universal_selection


SOURCE_URL = "https://github.com/Starnodes2024/comfyui-starnodes-modelconverter"
SOURCE_COMMIT = "b3852dae4383520c8c9aa2003795edf525d097fe"


@lru_cache(maxsize=1)
def catalog_data():
    raw = Path(__file__).with_name("starnodes_models.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def catalog_choices():
    data, _ = catalog_data()
    return [(f"Star: {name}", "star:" + name) for name in sorted(data["models"], key=str.casefold)]


def catalog_description(profile):
    recipe = CatalogProfile(profile)
    scope = (" This variant changes MLP protection for W4A8/W6A8; other formats can match the base profile."
             if recipe.name == "Flux 2 Tight" else "")
    return (f"{recipe.name}: reference protection rules for this model family. "
            "Matching CTQ/Star protections and profile format recipes can be enabled below. Protected layers retain their original precision; "
            "custom rules can refine eligible layers. No architecture/key remapping is performed." + scope)


class CatalogProfile:
    def __init__(self, profile, pruned=False):
        data, self.source_hash = catalog_data()
        self.name = profile.removeprefix("star:")
        if self.name not in data["models"]:
            raise ValueError(f"Unknown Star model profile: {self.name}")
        self.recipe = {**data["default"], **data["models"][self.name]}
        self.pruned = pruned
        self.overrides = (
            ("force_int8_layers", "int8_tensorwise"),
            ("int4_convrot_layers", "convrot_w4a4"),
            ("int8_convrot_layers", "int8_convrot"),
            ("fp8_layers", "float8_e4m3fn"),
        )

    def protection(self, name, format=None):
        fields = ["blacklist", "keep_fp32", "keep_fp16"]
        if self.pruned:
            fields.append("pruned_extra_blacklist")
        if format in ("asym_w4a8_int8", "w6a8_int8"):
            fields.extend(("w4a8_blacklist", "w4a8_keep_fp32", "w4a8_keep_fp16"))
        for field in fields:
            for token in self.recipe.get(field, []):
                # Upstream uses case-sensitive literal substring matching, not
                # globs. Keep that contract for imported protection patterns.
                if token in name:
                    return f"Star {self.name} preserves original precision: {field} matches {token}"
        return None

    def selection(self, name, spec):
        reason = self.protection(name)
        if reason:
            return False, reason
        return universal_selection(name, spec, "float8_e4m3fn")

    def format_override(self, name, format):
        for field, target in self.overrides:
            for token in self.recipe.get(field, []):
                if token in name:
                    return target, f"Star {self.name} format rule: {field} matches {token} -> {target}"
        return format, None

    def check_format(self, format):
        if format == "nvfp4":
            if self.name == "SDXL (Not NVFP4)":
                raise ValueError("The Star SDXL profile excludes NVFP4; select another supported format")
            if self.recipe.get("nvfp4_group_size") not in (None, 16):
                raise ValueError("This Star recipe requires an unsupported NVFP4 group size")

    def details(self, tensors):
        protection_fields = ("blacklist", "keep_fp32", "keep_fp16", "pruned_extra_blacklist",
                             "w4a8_blacklist", "w4a8_keep_fp32", "w4a8_keep_fp16")
        fields = (*protection_fields, *(field for field, _ in self.overrides))
        matches = {field: {token: sum(token in name for name in tensors)
                          for token in self.recipe.get(field, [])} for field in fields}
        return {"name": self.name, "source": SOURCE_URL, "commit": SOURCE_COMMIT,
                "source_sha256": self.source_hash, "license": "MIT; see docs/STARNODES_LICENSE.md",
                "pruned_extra_protection": self.pruned, "pattern_matches": matches,
                "matching": "case-sensitive literal substring (upstream semantics)",
                "dtype_hints": "keep_fp16/keep_fp32 preserve source precision; no implicit casting",
                "eligibility": "additional Universal exclusions and dimension-anchor safety remain enforced",
                "verification": "profile rule engine tested; full-model validation is separate"}

    def notes(self, tensors):
        notes = []
        if not any(self.protection(name) for name in tensors):
            notes.append(f"Star {self.name}: no protection patterns matched; check that the selected family matches the checkpoint tensor names.")
        if any(self.recipe.get(field) for field in ("keep_fp16", "keep_fp32", "w4a8_keep_fp16", "w4a8_keep_fp32")):
            notes.append("Star profile precision hints retain source dtypes; protected tensors are not implicitly cast to FP16/FP32.")
        if self.recipe.get("awq_prescale_layers") or self.recipe.get("awq_w4a16_layers"):
            notes.append("This Star profile also describes AWQ controls; AWQ is not an implemented target and its prescaling is not applied.")
        if "int4_tensorwise" in self.name:
            notes.append("The reference profile name mentions experimental tensorwise INT4; only its protection rules are used. The output uses the selected supported format, not a tensorwise INT4 exporter.")
        return notes

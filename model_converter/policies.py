"""Explicit, host-independent precision recipes applied to eligible layers only."""

from fnmatch import fnmatchcase
import json
import re

from .formats import FORMATS, FORMAT_SUFFIXES
from .io import unique_object


PRECISION_CHOICES = [
    ("Selected format for all eligible layers", "uniform"),
    ("Keep attention in original precision", "preserve_attention"),
    ("Mixed: attention in FP8 e4m3, other layers in selected format", "attention_fp8"),
    ("Mixed: attention in INT8 ConvRot, other layers in selected format", "attention_int8"),
]
BLOCK = re.compile(r"^(.*?(?:^|\.)(?:layers|blocks|block|double_blocks|single_blocks))\.(\d+)\.")
ATTENTION = re.compile(r"(?:^|\.)(?:attention|attn|self_attn|cross_attn|img_attn|txt_attn)(?:\.|$)")


def is_attention(name):
    # Flux single-stream projections fuse attention and MLP, so protect/mix the
    # whole tensor. Slicing it would violate the canonical loader contract.
    return bool(ATTENTION.search(name) or re.search(r"(?:^|\.)proj\.(?:qkv|o)\.weight$", name)
                or re.search(r"(?:^|\.)single_blocks\.\d+\.linear[12]\.weight$", name))


class PrecisionPolicy:
    def __init__(self, request, tensors):
        self.protect_layers = request.protect_layers
        if request.precision_policy not in dict((value, label) for label, value in PRECISION_CHOICES):
            raise ValueError("Unsupported precision policy")
        self.preset = request.precision_policy
        self.block_counts = (request.keep_first_blocks, request.keep_last_blocks)
        for count in self.block_counts:
            if isinstance(count, bool) or not isinstance(count, (int, float)) or not 0 <= count <= 10000 or int(count) != count:
                raise ValueError("Protected block counts must be nonnegative whole numbers (at most 10000)")
        if not isinstance(request.preserve_patterns, str) or not isinstance(request.layer_rules, str):
            raise ValueError("Layer patterns and JSON rules must be text")
        self.patterns = [line.strip() for line in request.preserve_patterns.splitlines() if line.strip()]
        self.rules = []
        try:
            raw = json.loads(request.layer_rules, object_pairs_hook=unique_object) if request.layer_rules.strip() else []
        except ValueError as error:
            raise ValueError(f"Invalid layer rules JSON: {error}") from error
        if not isinstance(raw, list) or len(raw) > 256 or len(self.patterns) > 256:
            raise ValueError("Layer rules must be a JSON list with at most 256 rules/preserve patterns")
        aliases = {suffix: format for format, suffix in FORMAT_SUFFIXES.items()}
        for rule in raw:
            if not isinstance(rule, dict) or set(rule) != {"pattern", "format"}:
                raise ValueError('Each layer rule requires exactly "pattern" and "format"')
            target = rule["format"]
            if not isinstance(target, str):
                raise ValueError("Layer rule format must be a supported format or preserve")
            target = aliases.get(target, target)
            if target not in (*FORMATS, "preserve"):
                raise ValueError(f"Unsupported layer rule format: {target}")
            self.rules.append({"pattern": rule["pattern"], "format": target, "matched": 0, "applied": 0, "blocked": 0})
        for pattern in [*self.patterns, *(rule["pattern"] for rule in self.rules)]:
            if not isinstance(pattern, str) or not pattern.strip() or len(pattern) > 512:
                raise ValueError("Each layer pattern must be nonempty text of at most 512 characters")
        self.preserve_matches = {pattern: sum(fnmatchcase(name, pattern) for name in tensors) for pattern in self.patterns}
        for rule in self.rules:
            rule["matched"] = sum(fnmatchcase(name, rule["pattern"]) for name in tensors)
        stacks = {}
        for name in tensors:
            match = BLOCK.match(name)
            if match:
                stacks.setdefault(match[1], set()).add(int(match[2]))
        if any(self.block_counts) and not stacks:
            raise ValueError("No recognized transformer block stack for first/last-block protection")
        self.protected_blocks = {}
        for stack, indexes in sorted(stacks.items()):
            ordered = sorted(indexes)
            first, last = map(int, self.block_counts)
            protected = set(ordered[:first]) | (set(ordered[-last:]) if last else set())
            if protected:
                self.protected_blocks[stack] = sorted(protected)

    def select(self, name, format):
        self.current_rule = None
        for pattern in self.patterns:
            if self.protect_layers and fnmatchcase(name, pattern):
                return None, f"Keep original precision: preserve pattern {pattern}"
        match = BLOCK.match(name)
        if self.protect_layers and match and int(match[2]) in self.protected_blocks.get(match[1], ()):
            return None, f"Keep original precision: protected boundary block {match[1]}.{match[2]}"
        for rule in self.rules:
            if not self.protect_layers and rule["format"] == "preserve":
                continue
            if fnmatchcase(name, rule["pattern"]):
                rule["applied"] += 1
                self.current_rule = rule
                return (None if rule["format"] == "preserve" else rule["format"]), f"Layer rule {rule['pattern']}: {rule['format']}"
        if is_attention(name) and self.preset != "uniform":
            if self.preset == "preserve_attention":
                return None, "Keep original precision: attention policy (includes fused attention/MLP projections)"
            target = "float8_e4m3fn" if self.preset == "attention_fp8" else "int8_convrot"
            return target, f"Mixed precision policy: attention in {target} (includes fused attention/MLP projections)"
        return format, None

    def record_protected(self, name):
        """Count a format override blocked before precision resolution."""
        for rule in self.rules:
            if fnmatchcase(name, rule["pattern"]):
                if rule["format"] != "preserve":
                    rule["blocked"] += 1
                break

    def block_assignment(self):
        if self.current_rule is not None:
            self.current_rule["applied"] -= 1
            self.current_rule["blocked"] += 1

    def details(self):
        return {"preset": self.preset, "protected_blocks": self.protected_blocks,
                "preserve_patterns": self.preserve_matches, "layer_rules": self.rules,
                "precedence": "storage/loader and CTQ/Star/native protection > general eligibility > preserve patterns > boundary blocks > first matching rule > attention preset > reference format rule > selected format"}

    def notes(self):
        notes = [f"Preserve pattern matched no tensors: {pattern}" for pattern, count in self.preserve_matches.items() if not count]
        for rule in self.rules:
            if not rule["matched"]:
                notes.append(f"Layer rule matched no tensors: {rule['pattern']}")
            elif not rule["applied"]:
                notes.append(f"Layer rule affected no eligible layers (protected or shadowed): {rule['pattern']}")
            if rule["blocked"]:
                notes.append(f"Layer rule blocked by profile or storage-shape safety for {rule['blocked']} tensors: {rule['pattern']}")
        return notes

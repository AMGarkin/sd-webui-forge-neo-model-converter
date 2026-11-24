"""Reference rule data and independently implemented general selection criteria."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path


TOOLS_COMMIT = "d6797787e6bdb1a1fb0094d588a26f8e71a1c757"
TOOLS_SOURCE = "https://github.com/Comfy-Org/comfy-model-tools/blob/" + TOOLS_COMMIT + "/quant_int8_convrot.py"


@lru_cache(maxsize=1)
def ctq_data():
    raw = Path(__file__).with_name("ctq_filters.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def ctq_filter(name):
    filters = ctq_data()[0]["filters"]
    if name not in filters:
        raise ValueError(f"Unknown CTQ model filter: {name}")
    result = filters[name]
    return filters[result["alias_for"]] if "alias_for" in result else result


def ctq_choices():
    return [("CTQ: " + name.replace("_", " "), "ctq:" + name)
            for name in sorted(ctq_data()[0]["filters"])]


def tools_protection(name):
    """Implement the owner's general selection criteria without copying GPL code.

    Check name segments, rather than an attention/MLP projection allowlist.
    Modulation, SwiGLU gate projections, main-stream refiners and embedding
    connectors remain candidates unless a model-specific rule protects them.
    """
    fragments = ("scale_shift", "rope", "rotary", "rel_pos", "pos_embed", "posembed",
                 "embedder", "gate_logits", "router", "routing", "logit", "temperature",
                 "temb", "guidance", "register", "refiner_blocks", "adapter")
    for segment in name.lower().split(".")[:-1]:
        for token in fragments:
            if token in segment:
                return f"comfy-model-tools general criterion: {token}"
        # A bare substring t_emb would also match tex[t_emb]eddings_connector.
        # Match the conditioning prefix at an underscore boundary instead.
        if segment.startswith("t_emb") or "_t_emb" in segment:
            return "comfy-model-tools general criterion: timestep embedding"
        if any(part.startswith("time") for part in segment.split("_")):
            return "comfy-model-tools general criterion: timestep/conditioning MLP"
        for token in ("final", "head", "proj_out", "out_layer"):
            if (segment == token or segment.startswith(token + "_") or segment.endswith("_" + token)
                    or "_" + token + "_" in segment):
                return f"comfy-model-tools general criterion: output boundary {token}"
    return None


def general_protection(name, optional=True, protect_lookup=True):
    # Keep actual lookup tables distinct from *_embeddings_connector and
    # embedding_projection, which may be ordinary linear projections.
    tables = {"embed", "embedding", "embeddings", "embed_tokens", "embed_tokens_per_layer",
              "embed_positions", "word_embeddings", "token_embedding", "tok_embeddings",
              "position_embedding", "position_embeddings", "wte", "wpe", "shared"}
    if protect_lookup and any(segment in tables for segment in name.lower().split(".")[:-1]):
        return "Lookup-table safety: preserve embedding or lookup table"
    if not optional:
        return None
    tools = tools_protection(name)
    if tools:
        return tools
    for token in ctq_data()[0]["base_exclusions"]:
        if token in name:
            return f"CTQ common protection: {token}"
    return None

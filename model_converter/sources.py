"""Storage grouping, independent of architecture and transport."""

import json
from pathlib import Path, PurePosixPath
import re

SHARD = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.safetensors$")


def safe_relative(value):
    """Reject Windows and POSIX traversal before touching a repository path."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Invalid repository/index path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("Repository/index path traversal is forbidden")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(p.endswith((".", " ")) or p.split(".")[0].upper() in reserved or any(
            ord(c) < 32 or c in '<>"|?*' for c in p) for p in path.parts):
        raise ValueError("Repository/index path is unsafe on Windows")
    return path


def group_files(names):
    """Return deterministic logical sets without opening weights or merging variants.

    Invalid/incomplete groups remain visible; selecting them gives a concrete error.
    An index is authoritative for nonstandard shard filenames.
    """
    names = sorted(set(names))
    groups = {}
    for name in names:
        safe_relative(name)
        path = PurePosixPath(name)
        if name.endswith(".safetensors.index.json"):
            stem = path.name.removesuffix(".safetensors.index.json")
            groups.setdefault((str(path.parent), stem), {"files": [], "index": None})["index"] = name
        elif name.endswith(".safetensors"):
            match = SHARD.fullmatch(path.name)
            stem = match[1] if match else path.stem
            groups.setdefault((str(path.parent), stem), {"files": [], "index": None})["files"].append(name)
    result = []
    for (folder, stem), group in sorted(groups.items()):
        files = group["files"]
        shards = [SHARD.fullmatch(PurePosixPath(f).name) for f in files]
        error = None
        if any(shards):
            totals = {int(m[3]) for m in shards if m}
            if not all(shards) or len(totals) != 1:
                error = "Mixed single/sharded files or conflicting shard counts"
            elif not next(iter(totals)) or {int(m[2]) for m in shards} != set(range(1, next(iter(totals)) + 1)):
                error = "Incomplete shard set"
        result.append({"folder": "" if folder == "." else folder, "stem": stem,
                       **group, "error": error})
    return result


def local_sets(directory):
    directory = Path(directory).resolve(strict=True)
    names = [p.name for p in directory.iterdir() if p.is_file() and p.name.endswith(
        (".safetensors", ".safetensors.index.json"))]
    groups = group_files(names)
    # Explicit indexes may reference nonstandard shard names. Hide those individual
    # files in discovery, but never trust their mapping without inspection.
    referenced = set()
    for group in groups:
        if group["index"]:
            try:
                data = json.loads((directory / group["index"]).read_text(encoding="utf-8"))
                referenced.update(data["weight_map"].values())
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return [g for g in groups if g["index"] or not set(g["files"]) <= referenced]


def group_source(directory, group):
    if group["index"]:
        return Path(directory) / group["index"]
    if group["files"]:
        # The first shard is a stable handle, never a request to quantize one shard.
        return Path(directory) / group["files"][0]
    raise ValueError("Empty source set")


def resolve_local(source):
    source = Path(source).resolve(strict=True)
    if source.is_dir():
        groups = local_sets(source)
        if len(groups) != 1:
            raise ValueError("Component directory has multiple or no weight sets; select a logical set/index explicitly")
        source = group_source(source, groups[0])
    match = SHARD.fullmatch(source.name)
    if match:
        groups = [g for g in local_sets(source.parent) if g["stem"] == match[1]]
        if len(groups) != 1 or groups[0]["error"]:
            raise ValueError(groups[0]["error"] if groups else "Shard set not found")
        group = groups[0]
        return group_source(source.parent, group), [source.parent / n for n in group["files"]]
    return source, None


def component_config(source):
    """Read data only; never import repository Python or trust_remote_code."""
    source, _ = resolve_local(source)
    config_path = source.parent / "config.json"
    if config_path.is_file() and config_path.stat().st_size > 16 * 1024**2:
        raise ValueError("Component config is too large")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if not isinstance(config, dict):
        raise ValueError("Component config must be a JSON object")
    # model_index is evidence of the declared component class, not a key map.
    for root in (source.parent, source.parent.parent):
        index = root / "model_index.json"
        if index.is_file():
            data = json.loads(index.read_text(encoding="utf-8"))
            entry = data.get(source.parent.name)
            if not config.get("_class_name") and isinstance(entry, list) and len(entry) == 2:
                config["_class_name"] = entry[1]
            break
    return config


def ltx_connector_source(source):
    """A declared LTX2 dependency, never a component-wide weight-file glob."""
    source, _ = resolve_local(source)
    folder = source.parent.parent / "connectors"
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("LTX-2 transformer requires the matching sibling connectors component. Retain/download transformer and connectors from the same revision; missing connector weights cannot be invented.")
    groups = local_sets(folder)
    if len(groups) != 1 or groups[0]["error"]:
        raise ValueError("LTX-2 connectors dependency needs exactly one complete weight set; resolve its variant explicitly")
    handle = group_source(folder, groups[0])
    if component_config(handle).get("_class_name") != "LTX2TextConnectors":
        raise ValueError("LTX-2 sibling connectors config must declare LTX2TextConnectors")
    return handle

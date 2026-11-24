"""Resolve paths before crossing the host/worker working-directory boundary."""

import os
from pathlib import Path


def clean_path(value):
    value = str(value or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    return value


def resolve_source(value, base_dir=None, choices=()):
    value = clean_path(value)
    if not value:
        raise ValueError("Select a model or paste its file path first.")
    # An editable Gradio dropdown can submit its display label on a later run.
    # Resolve that label through the current inventory, never through worker cwd.
    key = Path(value).as_posix()
    key = key.casefold() if os.name == "nt" else key
    matches = set()
    for label, path in choices:
        label_key = Path(label).as_posix()
        label_key = label_key.casefold() if os.name == "nt" else label_key
        if label_key == key:
            matches.add(str(Path(path).resolve()))
    if len(matches) > 1:
        raise ValueError("Ambiguous model label. Paste the model's absolute path.")
    if matches:
        return matches.pop()
    source = Path(value)
    if not source.is_absolute():
        source = Path(base_dir or Path.cwd()) / source
    return str(source.resolve())


def resolve_output(value, source, base_dir=None):
    value = clean_path(value)
    if not value:
        raise ValueError("An output filename is required")
    output = Path(value)
    if not output.is_absolute():
        # A bare filename belongs beside the source. A relative directory is
        # rooted at the caller's base (Forge root in the UI, invocation cwd in CLI).
        root = (Path(source) if Path(source).is_dir() else Path(source).parent) if output.parent == Path(".") else Path(base_dir or Path.cwd())
        output = root / output
    return str(output.resolve())


def absolute_paths(source, output, base_dir=None):
    source = resolve_source(source, base_dir)
    return source, resolve_output(output, source, base_dir)

"""Header inspection and bounded, no-clobber safetensors output."""

import json
import math
import os
from pathlib import Path
import struct
import tempfile


SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
         "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        size, = struct.unpack("<Q", raw)
        if not 2 <= size <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError("Invalid header size")
        header = json.loads(stream.read(size), object_pairs_hook=unique_object)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("Invalid metadata")
    spans = []
    for name, spec in header.items():
        shape, dtype, offsets = spec["shape"], spec["dtype"], spec["data_offsets"]
        if dtype not in SIZES or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError(f"Invalid dtype or shape: {name}")
        if len(offsets) != 2 or any(type(x) is not int or x < 0 for x in offsets):
            raise ValueError(f"Invalid offsets: {name}")
        start, end = offsets
        if end - start != math.prod(shape) * SIZES[dtype]:
            raise ValueError(f"Invalid tensor length: {name}")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("Overlapping or non-contiguous tensor data")
        cursor = end
    if cursor + size + 8 != path.stat().st_size:
        raise ValueError("Truncated or trailing tensor data")
    return header, metadata


def inspect(source, with_metadata=False):
    from .sources import resolve_local, safe_relative, SHARD
    source, shard_files = resolve_local(source)
    if source.name.endswith(".safetensors.index.json"):
        index = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        mapping = index["weight_map"]
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(k, str) or not k or not isinstance(v, str) for k, v in mapping.items()):
            raise ValueError("Empty or invalid shard map")
        files = {}
        for filename in set(mapping.values()):
            safe_relative(filename)
            path = (source.parent / filename).resolve(strict=True)
            if path.parent != source.parent or path.suffix != ".safetensors":
                raise ValueError("Shard must be a safetensors file in the index directory")
            files[path] = read_header(path)
        # Validate standard shard numbering even for an explicit index.
        standard = [SHARD.fullmatch(p.name) for p in files]
        if any(standard):
            if not all(standard) or len({(m[1], m[3]) for m in standard}) != 1:
                raise ValueError("Conflicting indexed shard sets")
            if {int(m[2]) for m in standard} != set(range(1, int(standard[0][3]) + 1)):
                raise ValueError("Incomplete indexed shard set")
    elif source.suffix == ".safetensors":
        files = {p: read_header(p) for p in (shard_files or [source])}
        mapping = None
    else:
        raise ValueError("Only safetensors checkpoints and shard indexes are accepted")
    tensors = {}
    combined_metadata = {}
    for path, (header, metadata) in files.items():
        if any("quant" in key.lower() for key in metadata):
            raise ValueError("Already quantized metadata is not accepted")
        for key, value in metadata.items():
            if key in combined_metadata and combined_metadata[key] != value:
                raise ValueError(f"Conflicting metadata across shards: {key}")
            combined_metadata[key] = value
        for name, spec in header.items():
            if name in tensors:
                raise ValueError(f"Duplicate tensor across shards: {name}")
            if mapping is not None and mapping.get(name) != path.name:
                raise ValueError(f"Shard index mismatch: {name}")
            if spec["dtype"].startswith("F8_") or (
                (name == "weight" or name.endswith(".weight")) and spec["dtype"] not in {"F32", "F16", "BF16"}
            ) or any(
                marker in name for marker in ("comfy_quant", "weight_scale", "weight_s_rel", "scaled_fp8")
            ):
                raise ValueError(f"Quantized or unsupported tensor: {name}")
            tensors[name] = {**spec, "file": str(path)}
    if mapping is not None and set(mapping) != set(tensors):
        raise ValueError("Missing tensors referenced by shard index")
    if mapping is not None and "total_size" in index.get("metadata", {}):
        total = index["metadata"]["total_size"]
        if type(total) is not int or total != sum(s["data_offsets"][1] - s["data_offsets"][0] for s in tensors.values()):
            raise ValueError("Shard index total_size disagrees with tensor payload")
    if not tensors:
        raise ValueError("Empty checkpoint")
    return (tensors, combined_metadata) if with_metadata else tensors


class TensorReader:
    """Read owned CPU tensor buffers without Torch's mapped-storage slicing.

    The standard safetensors parser remains part of output verification; input
    headers/offsets are validated before these ordinary positional reads.
    """

    def __init__(self, path, specs=None):
        self.path = Path(path)
        self.specs = read_header(path)[0] if specs is None else specs
        self.stream = self.path.open("rb")
        raw = self.stream.read(8)
        if len(raw) != 8:
            self.stream.close()
            raise ValueError("Truncated tensor header")
        self.data_start = 8 + struct.unpack("<Q", raw)[0]

    def get_tensor(self, name, start=None, stop=None):
        import torch
        spec = self.specs[name]
        names = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
                 "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
                 "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}
        shape = list(spec["shape"])
        offset = spec["data_offsets"][0]
        if start is not None or stop is not None:
            if not shape or start is None or stop is None or not 0 <= start <= stop <= shape[0]:
                raise ValueError("Invalid tensor row slice")
            offset += start * math.prod(shape[1:]) * SIZES[spec["dtype"]]
            shape[0] = stop - start
        value = torch.empty(shape, dtype=getattr(torch, names[spec["dtype"]]), device="cpu")
        buffer = memoryview(value.reshape(-1).view(torch.uint8).numpy())
        self.stream.seek(self.data_start + offset)
        cursor = 0
        while cursor < len(buffer):
            count = self.stream.readinto(buffer[cursor:cursor + 8 * 1024 * 1024])
            if not count:
                raise ValueError(f"Truncated tensor data: {name}")
            cursor += count
        return value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


class StreamWriter:
    """Spool tensors to disk; construct one standard file without a model-sized dict."""

    def __init__(self, output, work_dir=None, metadata=None):
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError(self.output)
        self.temp = None if work_dir else tempfile.TemporaryDirectory(prefix=".converter-", dir=self.output.parent)
        root = Path(work_dir or self.temp.name)
        if root.is_symlink() or root.resolve().stat().st_dev != self.output.parent.stat().st_dev:
            raise ValueError("Working directory must be on the output filesystem and cannot be a symlink")
        root = root.resolve()
        self.spool = root / "payload"
        self.pending = root / "pending.safetensors"
        self.stream = self.spool.open("xb")
        self.header = {}
        if metadata:
            self.header["__metadata__"] = dict(metadata)

    def add(self, name, tensor):
        import torch
        if name in self.header:
            raise ValueError(f"Output tensor collision: {name}")
        dtypes = {torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
                  torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2",
                  torch.int64: "I64", torch.int32: "I32", torch.int16: "I16",
                  torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}
        value = tensor.detach().cpu().contiguous()
        start = self.stream.tell()
        raw = value.reshape(-1).view(torch.uint8).numpy()
        self.stream.write(memoryview(raw))
        self.header[name] = {"dtype": dtypes[value.dtype], "shape": list(value.shape),
                             "data_offsets": [start, self.stream.tell()]}

    def finish(self, validate, cancelled, emit=lambda event: None):
        self.stream.close()
        header = json.dumps(self.header, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        total = self.spool.stat().st_size
        copied = 0
        emit({"event": "progress", "phase": "Writing checkpoint", "fraction": 0.85})
        with self.pending.open("xb") as target, self.spool.open("rb") as source:
            target.write(struct.pack("<Q", len(header)))
            target.write(header)
            while chunk := source.read(8 * 1024 * 1024):
                cancelled()
                target.write(chunk)
                copied += len(chunk)
                emit({"event": "progress", "phase": "Writing checkpoint", "fraction": 0.85 + 0.12 * copied / max(total, 1)})
            target.flush()
            os.fsync(target.fileno())
        emit({"event": "progress", "phase": "Validating checkpoint", "fraction": 0.97})
        validate(self.pending)
        cancelled()
        # A hard link publishes atomically and refuses to overwrite a racing writer.
        os.link(self.pending, self.output)

    def close(self):
        self.stream.close()
        if self.temp is not None:
            self.temp.cleanup()
        # For UI jobs the parent owns this directory and cleans it after process exit,
        # including native crashes that cannot execute Python finally blocks.

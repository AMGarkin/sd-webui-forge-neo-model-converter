"""Pinned, application-owned HF component acquisition using Hub 0.36 APIs.

The shared global HF cache is read-only to this application. Deletion is limited
to inventoried files acquired by the successful, exclusively leased job itself.
"""

from dataclasses import asdict
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid
import time
from urllib.parse import urlparse

from .sources import safe_relative, group_files, group_source, indexed_weight_files
from .io import inspect, unique_object

SCHEMA = 1
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _json(path, data):
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _read(path):
    if path.stat().st_size > 100 * 1024**2:
        raise ValueError("Source metadata exceeds the 100 MiB limit")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


class SourceLease:
    """OS-held exclusive lock: termination releases it without stale-lock guessing.

    The stable lock inode is never deleted, avoiding unlock/unlink races on POSIX.
    The JSON marker is diagnostic only. Legacy opaque leases fail closed.
    """

    def __init__(self, entry, owner):
        self.marker, self.owner = entry / ".lease", owner
        lock = entry / ".source-lock"
        if lock.is_symlink() or self.marker.is_symlink():
            raise ValueError("Unsafe retained source lease path")
        self.stream = lock.open("a+b")
        try:
            self.stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise ValueError("Retained source is in use by another job; retry after it finishes") from None
            if self.marker.exists():
                try:
                    prior = _read(self.marker)
                except ValueError:
                    prior = {}
                if not isinstance(prior, dict) or prior.get("protocol") != "os-lock-v1":
                    raise ValueError("Legacy interrupted source lease requires inspection before removal")
            _json(self.marker, {"protocol": "os-lock-v1", "owner": owner, "pid": os.getpid()})
        except BaseException:
            self.stream.close()
            raise

    def owns(self):
        return self.marker.exists() and _read(self.marker).get("owner") == self.owner

    def close(self):
        try:
            if self.owns():
                self.marker.unlink()
        finally:
            self.stream.close()


def _validate_repo(repo):
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(repo)
    if repo.count("/") != 1:
        raise ValueError("Enter a Hugging Face owner/repository ID")


def download_file(repo, name, commit, target, expected, emit_bytes, cancelled):
    import requests
    try:
        return _download_file(repo, name, commit, target, expected, emit_bytes, cancelled)
    except requests.RequestException as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        detail = f"HTTP {status}" if status else "connection/timeout error"
        # A CDN exception URL can contain signed credentials. Keep it out of both
        # the Forge console and JSON protocol; authentication remains host-owned.
        raise ValueError(f"HF download failed for {name}: {detail}. Check connectivity and existing HF access; source data was preserved for retry.") from None


def _download_file(repo, name, commit, target, expected, emit_bytes, cancelled):
    """Cancellable/resumable HTTP through public Hub 0.36 metadata/auth APIs.

    hf_hub_download in 0.36.2 has no per-file tqdm_class argument. In particular,
    snapshot_download's tqdm_class only reports file counts, not transfer bytes.
    Own the transfer instead of monkeypatching Hub globals or Xet callbacks.
    """
    from huggingface_hub import hf_hub_url, get_hf_file_metadata
    from huggingface_hub.utils import get_session, build_hf_headers
    url = hf_hub_url(repo, name, revision=commit)
    cancelled()
    metadata = get_hf_file_metadata(url, timeout=10)
    if metadata.commit_hash != commit or metadata.size != expected["size"]:
        raise ValueError(f"HF file metadata revision/size mismatch: {name}")
    # Preserve interrupted bytes only in an application-owned per-entry area.
    partial_root = target.parents[len(safe_relative(name).parts) - 1] / ".downloads"
    partial_root.mkdir(exist_ok=True)
    partial = partial_root / (_hash(name) + ".incomplete")
    record = partial.with_suffix(".json")
    identity = {"repo": repo, "commit": commit, "name": name, "size": expected["size"]}
    if partial.is_symlink() or partial_root.is_symlink():
        raise ValueError("Unsafe partial download path")
    if partial.exists() and (not record.exists() or _read(record) != identity):
        raise ValueError("Partial download ownership/revision mismatch")
    _json(record, identity)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected["size"]:
        raise ValueError("Partial download is larger than its immutable source")
    headers = build_hf_headers()
    # Signed CDN locations do not need the user's HF bearer token. requests also
    # strips authorization on cross-host redirects from the authenticated Hub.
    location = metadata.location
    if urlparse(location).netloc != urlparse(url).netloc:
        headers.pop("authorization", None)
    if offset:
        headers["Range"] = f"bytes={offset}-"
    if offset < expected["size"]:
        with get_session().get(location, headers=headers, stream=True, timeout=(10, 10)) as response:
            response.raise_for_status()
            if response.status_code == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or int(match[1]) != offset or int(match[3]) != expected["size"]:
                    raise ValueError("HF server returned an inconsistent partial response")
            elif response.status_code == 200:
                offset = 0  # Server declined resume; replace only our partial file.
            else:
                raise ValueError(f"Unexpected HF download status {response.status_code}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("ab" if offset else "wb") as stream:
                received = offset
                for chunk in response.iter_content(chunk_size=256 * 1024):
                    cancelled()
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > expected["size"]:
                        raise ValueError("HF response exceeds its declared size")
                    stream.write(chunk)
                    emit_bytes(received)
                stream.flush()
                os.fsync(stream.fileno())
    cancelled()
    if not partial.exists() or partial.stat().st_size != expected["size"]:
        raise ValueError(f"Incomplete HF download: {name}")
    digest = expected.get("sha256")
    if digest:
        hasher = hashlib.sha256()
        with partial.open("rb") as stream:
            while chunk := stream.read(8 * 1024**2):
                cancelled()
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise ValueError(f"HF SHA256 mismatch; partial source retained for inspection: {name}")
    # Publishing a source is also no-clobber. A pre-existing file is never adopted.
    os.link(partial, target)
    partial.unlink()
    record.unlink()
    return target


def inventory(repo, revision, cache_root, api=None):
    """Resolve one immutable commit and list isolated folder/set choices.

    An offline fallback is explicit and reported. Cached catalog data never
    substitutes a different requested revision or repository.
    """
    from huggingface_hub import HfApi
    _validate_repo(repo)
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = root / ("catalog-" + _hash(repo + "@" + (revision or "main")) + ".json")
    try:
        info = (api or HfApi()).model_info(repo, revision=revision or "main", files_metadata=True)
        if not COMMIT.fullmatch(info.sha or ""):
            raise ValueError("HF did not return an immutable commit")
        if COMMIT.fullmatch(revision or "") and info.sha != revision:
            raise ValueError("HF metadata commit differs from the requested immutable revision")
        files = {s.rfilename: {"size": s.size, "blob": getattr(s, "blob_id", None),
                              "sha256": getattr(getattr(s, "lfs", None), "sha256", None)}
                 for s in info.siblings}
        for name in files:
            safe_relative(name)
        result = {"schema": SCHEMA, "repo": repo, "requested_revision": revision or "main",
                  "commit": info.sha, "files": files, "offline": False}
        _json(catalog, result)
        _json(root / ("catalog-" + _hash(repo + "@" + info.sha) + ".json"),
              {**result, "requested_revision": info.sha})
    except Exception as error:
        # Only connection/offline failures may use metadata cache. Authentication,
        # nonexistent repositories/revisions and malformed metadata stay failures.
        from huggingface_hub.errors import OfflineModeIsEnabled, LocalEntryNotFoundError
        import requests
        if not isinstance(error, (requests.ConnectionError, requests.Timeout, OfflineModeIsEnabled, LocalEntryNotFoundError)):
            raise
        if not catalog.is_file():
            raise ValueError("HF is offline and no matching repository catalog is retained") from error
        result = _read(catalog)
        if result.get("repo") != repo or result.get("requested_revision") != (revision or "main"):
            raise ValueError("Cached HF catalog identity mismatch")
        result["offline"] = True
    result["sets"] = group_files(result["files"])
    return result


def select_set(catalog, component, stem):
    component = component or ""
    if component:
        safe_relative(component)
    candidates = [g for g in catalog["sets"] if g["folder"] == component]
    if stem:
        candidates = [g for g in candidates if g["stem"] == stem]
    if len(candidates) != 1:
        raise ValueError("Choose exactly one component weight set/variant; unrelated checkpoints are never combined")
    group = candidates[0]
    if group["error"]:
        raise ValueError(group["error"])
    if re.search(r"(?:^|[_.-])(?:adapter|lora|lycoris)(?:[_.-]|$)", group["stem"], re.IGNORECASE):
        raise ValueError("Adapter/LoRA weight sets are not standalone model components")
    return group


def _ancestor(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def disk_preflight(cache_dir, output, missing_bytes, source_bytes, reserve_output=True):
    """Reserve on actual filesystems, counting Windows copies and partials safely.

    Existing incomplete bytes already reduce disk free space. Reserve a full
    replacement for every incomplete file rather than assuming resumability or
    symlink privileges. The canonical writer's exact preflight still runs later.
    Before headers are local, estimate each of the two output buffers from the
    selected source size. Preserved tensors keep their dtype; they are not cast
    to float32. The exact format-aware check still runs before conversion.
    """
    requirements = {}
    for path, needed, purpose in ((cache_dir, missing_bytes, "missing source files"),
                                  (Path(output).parent, source_bytes * 2 + 64 * 1024**2 if reserve_output else 0, "estimated output and staging")):
        existing = _ancestor(path)
        device = existing.stat().st_dev
        entry = requirements.setdefault(device, {"path": str(existing), "bytes": 0, "purposes": [], "breakdown": {}})
        entry["bytes"] += needed
        entry["purposes"].append(purpose)
        entry["breakdown"][purpose] = needed
    for entry in requirements.values():
        entry["free"] = shutil.disk_usage(entry["path"]).free
        if entry["free"] < entry["bytes"]:
            breakdown = ", ".join(f"{purpose}: {size / 1024**3:.2f} GiB" for purpose, size in entry["breakdown"].items())
            raise OSError(errno.ENOSPC, f"HF preflight: {entry['bytes'] / 1024**3:.2f} GiB required on {entry['path']}; {entry['free'] / 1024**3:.2f} GiB available ({breakdown}; selected component/variant and required dependencies only)")
    return list(requirements.values())


def provenance(source):
    """Find a complete retained source record for disk-mode naming and routing."""
    path = Path(source).resolve()
    for parent in (path if path.is_dir() else path.parent, *path.parents):
        manifest = parent / "source-manifest.json"
        if manifest.is_file():
            data = _read(manifest)
            relative = path.relative_to(parent).as_posix()
            if data.get("schema") == SCHEMA and data.get("complete") and (
                    relative in data.get("files", {}) or path.is_dir() and relative == (data.get("component") or ".")):
                return {k: data[k] for k in ("repo", "commit", "component", "stem")}
    return None


@contextmanager
def local_lease(source):
    """Disk reuse participates in the same exclusive ownership lease as HF jobs."""
    path = Path(source).resolve()
    lock = None
    owner = uuid.uuid4().hex
    for parent in (path if path.is_dir() else path.parent, *path.parents):
        if (parent / "ownership.json").is_file():
            lock = SourceLease(parent, owner)
            break
    try:
        yield
    finally:
        if lock:
            lock.close()


class Acquisition:
    def __init__(self, request, emit=lambda e: None, cancelled=lambda: None):
        self.request, self.emit, self.cancelled = request, emit, cancelled
        self.root = Path(request.hf_cache_dir).resolve()
        self.entry = None
        self.lock = None
        self.created = False
        self.owner = uuid.uuid4().hex
        self.manifest = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.lock:
            self.lock.close()

    def _event(self, message):
        self.emit({"event": "source", "message": message})

    def prepare(self, download=True):
        from huggingface_hub import try_to_load_from_cache
        self.cancelled()
        self._event("Resolving Hugging Face component metadata")
        catalog = inventory(self.request.hf_repo, self.request.hf_revision, self.root)
        if catalog["offline"]:
            self._event("HF offline: using retained metadata for the same immutable source revision")
        group = select_set(catalog, self.request.hf_component, self.request.hf_set)
        commit = catalog["commit"]
        identity = {"repo": catalog["repo"], "commit": commit, "component": group["folder"], "stem": group["stem"]}
        self.entry = self.root / ("source-" + _hash(json.dumps(identity, sort_keys=True)))
        if self.entry.is_symlink() or (self.entry.exists() and not self.entry.is_dir()):
            raise ValueError("Invalid application cache entry")
        self.entry.mkdir(exist_ok=True)
        self.lock = SourceLease(self.entry, self.owner)
        marker = self.entry / "ownership.json"
        if marker.exists():
            owned = _read(marker)
            if owned.get("identity") != identity:
                raise ValueError("Cache ownership identity mismatch")
        else:
            # Never adopt pre-existing unowned files as converter-owned inputs.
            if any(p.name not in (".lease", ".source-lock") for p in self.entry.iterdir()):
                raise ValueError("Refusing to adopt a nonempty unowned cache directory")
            _json(marker, {"schema": SCHEMA, "owner": self.owner, "identity": identity})
            self.created = True
        old = _read(self.entry / "source-manifest.json") if (self.entry / "source-manifest.json").exists() else {}
        records = dict(old.get("files", {}))
        metadata = []
        folder = group["folder"]
        for n in (f"{folder}/config.json" if folder else "config.json", "model_index.json"):
            if n in catalog["files"] and n not in metadata:
                metadata.append(n)
        names = sorted(set(([group["index"]] if group["index"] else group["files"]) + metadata))
        # Nonstandard indexed sets must not cause a blind component-wide download.
        # Fetch the small index only after its own preflight, then validate each name.
        def local(n):
            safe_relative(n)
            target = self.entry.joinpath(*n.split("/"))
            if not target.resolve().is_relative_to(self.entry) or any(p.is_symlink() for p in (target, *target.parents) if p != self.entry.parent):
                raise ValueError("Cache path or symlink escapes ownership boundary")
            return target
        def valid(n):
            path = local(n)
            size = catalog["files"][n]["size"]
            record = records.get(n, {})
            return path.is_file() and path.stat().st_size == size and record.get("size") == size and record.get("mtime_ns") == path.stat().st_mtime_ns
        def sizes(selected):
            for n in selected:
                if n not in catalog["files"] or type(catalog["files"][n].get("size")) is not int or catalog["files"][n]["size"] < 0:
                    raise ValueError(f"HF preflight cannot succeed: missing file size for {n}")
                limit = 100 * 1024**2 if n.endswith(".index.json") else 16 * 1024**2
                if n.endswith(".json") and catalog["files"][n]["size"] > limit:
                    raise ValueError(f"HF metadata file exceeds its bounded size limit: {n}")
            return sum(catalog["files"][n]["size"] for n in selected), sum(catalog["files"][n]["size"] for n in selected if not valid(n))
        def check_space():
            total, missing = sizes(names)
            # Complete weights can use the exact canonical output plan, not a source-sized guess.
            reserve = any(n.endswith(".safetensors") and not valid(n) for n in names)
            return disk_preflight(self.entry, self.request.output, missing, total, reserve_output=reserve)
        total, missing = sizes(names)
        preflight = check_space()
        completed = sum(catalog["files"][n]["size"] for n in names if valid(n))
        source_progress = 0
        outer = self
        class Progress:
            def __init__(self, total):
                self.n = 0
                self.last_emit = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n):
                nonlocal source_progress
                outer.cancelled()
                self.n += n
                if time.monotonic() - self.last_emit < .15 and completed + self.n < total:
                    return
                self.last_emit = time.monotonic()
                source_progress = max(source_progress, min(.19, .19 * (completed + self.n) / max(total, 1)))
                outer.emit({"event": "progress", "phase": f"Downloading HF source · {(completed + self.n) / 1024**3:.2f} / {total / 1024**3:.2f} GiB",
                            "fraction": source_progress})
        def fetch(n):
            nonlocal completed
            self.cancelled()
            if valid(n):
                return
            target = local(n)
            if target.exists():
                raise ValueError(f"Retained cache file changed or has no ownership record; refusing to overwrite: {n}")
            self._event(f"Downloading {n} at {commit[:12]}")
            # Cache hits are copied on Windows; account for their destination bytes
            # and expose progress/cancellation rather than relying on symlinks.
            cached = try_to_load_from_cache(catalog["repo"], n, revision=commit)
            if isinstance(cached, str) and Path(cached).stat().st_size == catalog["files"][n]["size"] and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_name(".converter-copy-" + self.owner + ".part")
                try:
                    digest = hashlib.sha256()
                    with Path(cached).open("rb") as src, pending.open("xb") as dst, Progress(total=catalog["files"][n]["size"]) as progress:
                        while chunk := src.read(8 * 1024**2):
                            self.cancelled()
                            dst.write(chunk)
                            digest.update(chunk)
                            progress.update(len(chunk))
                        dst.flush()
                        os.fsync(dst.fileno())
                    if catalog["files"][n].get("sha256") and digest.hexdigest() != catalog["files"][n]["sha256"]:
                        raise ValueError(f"Shared HF cache checksum mismatch: {n}; global cache was not modified")
                    os.link(pending, target)
                finally:
                    pending.unlink(missing_ok=True)
                path = target
            else:
                if catalog["offline"]:
                    raise ValueError(f"HF offline source is incomplete: {n}")
                progress = Progress(total=catalog["files"][n]["size"])
                path = Path(download_file(catalog["repo"], n, commit, target, catalog["files"][n],
                    lambda received: progress.update(received - progress.n), self.cancelled))
            if path.resolve() != target.resolve() or target.stat().st_size != catalog["files"][n]["size"]:
                raise ValueError(f"HF download size/path mismatch: {n}")
            records[n] = {"size": target.stat().st_size, "mtime_ns": target.stat().st_mtime_ns, "owner": self.owner}
            completed += records[n]["size"]
            self.manifest = {"schema": SCHEMA, **identity, "files": records, "complete": False}
            _json(self.entry / "source-manifest.json", self.manifest)
            self.cancelled()
        if group["index"]:
            # Inspect the index even during metadata-only preview. It is bounded
            # metadata, never weights, and its bytes were included in preflight.
            fetch(group["index"])
            mapping = _read(local(group["index"])).get("weight_map")
            indexed = indexed_weight_files(group, mapping)
            names = sorted(set(indexed + metadata + [group["index"]]))
            total, missing = sizes(names)
            preflight = check_space()
        for n in metadata:
            fetch(n)
        config_name = f"{folder}/config.json" if folder else "config.json"
        if config_name in metadata:
            from .layouts import validate_config
            config = _read(local(config_name))
            validate_config(config)
            if config.get("_class_name") == "MiniMaxH3DiTModel" and config.get("audio_enabled") is False:
                if "pdd_config.json" not in catalog["files"]:
                    raise ValueError("MiniMax H3 Image requires repository pdd_config.json before weights can be downloaded")
                names=sorted(set(names+["pdd_config.json"]))
                total,missing=sizes(names)
                preflight=check_space()
                fetch("pdd_config.json")
            if config.get("_class_name") == "LTX2VideoTransformer3DModel":
                dependency_folder = str(Path(folder).parent / "connectors").replace("\\", "/").removeprefix("./")
                groups = [g for g in catalog["sets"] if g["folder"] == dependency_folder]
                if len(groups) != 1 or groups[0]["error"]:
                    raise ValueError("LTX-2 requires exactly one complete sibling connectors set at the same revision; resolve missing/ambiguous connector variants before downloading weights")
                dep = groups[0]
                dep_config = dependency_folder + "/config.json"
                if dep_config not in catalog["files"]:
                    raise ValueError("LTX-2 connectors config is missing")
                additions = [dep_config] + ([dep["index"]] if dep["index"] else dep["files"])
                names = sorted(set(names + additions))
                total, missing = sizes(names)
                preflight = check_space()
                fetch(dep_config)
                if _read(local(dep_config)).get("_class_name") != "LTX2TextConnectors":
                    raise ValueError("LTX-2 connectors class mismatch")
                if dep["index"]:
                    fetch(dep["index"])
                    index = _read(local(dep["index"])).get("weight_map")
                    indexed = indexed_weight_files(dep, index)
                    names = sorted(set(names + indexed))
                    total, missing = sizes(names)
                    preflight = check_space()
                self._event("LTX-2 dependency: including matching connectors and text projection from the same commit")
        complete = all(valid(n) for n in names)
        total, missing = sizes(names)
        if not download and not complete:
            return {"source_preflight": {**identity, "files": names, "missing_bytes": missing,
                                        "disk": preflight, "offline": catalog["offline"]},
                    "layers": [], "request": asdict(self.request),
                    "loader_notes": ["Metadata preview only. Weights are downloaded when conversion starts; no architecture/loader claim is made."]}
        if download:
            for n in names:
                fetch(n)
        handle = local(group["index"] or group["files"][0])
        inspect(handle)  # Validate all keys, shards and index before marking reusable.
        if config_name in metadata and _read(local(config_name)).get("_class_name") == "LTX2VideoTransformer3DModel":
            from .sources import ltx_connector_source
            inspect(ltx_connector_source(handle))
        self.manifest = {"schema": SCHEMA, **identity, "files": {n: records[n] for n in names}, "complete": True}
        _json(self.entry / "source-manifest.json", self.manifest)
        self._event("HF source complete; reusing retained weights" if complete else "HF source downloaded and structurally validated")
        data = asdict(self.request)
        data.update(source=str(handle), source_kind="local", source_provenance={**identity, "offline_catalog": catalog["offline"]})
        return type(self.request)(**data)

    def cleanup_success(self):
        if self.request.hf_keep:
            return "retained"
        if not self.entry or not self.manifest or not self.manifest.get("complete"):
            raise ValueError("Refusing cleanup of an incomplete source")
        marker = _read(self.entry / "ownership.json")
        identity = {k: self.manifest[k] for k in ("repo", "commit", "component", "stem")}
        if marker.get("identity") != identity or not self.lock.owns() or self.entry.is_symlink():
            raise ValueError("HF cleanup ownership changed; source retained")
        owned = {n: spec for n, spec in self.manifest["files"].items() if spec.get("owner") == self.owner}
        if not owned:
            self._event("Keep source is off, but pre-existing retained inputs were preserved for safety")
            return "retained pre-existing source for safety"
        # All tensor readers have closed and the exclusive lease remains held.
        # Remove only individually inventoried downloads. Hub's entry-local
        # metadata and incomplete retry data remain. No recursive deletion occurs.
        for n, spec in owned.items():
            target = self.entry.joinpath(*safe_relative(n).parts)
            if (target.is_symlink() or not target.resolve().is_relative_to(self.entry)
                    or target.stat().st_size != spec["size"] or target.stat().st_mtime_ns != spec["mtime_ns"]):
                raise ValueError("Unsafe HF cleanup path; source retained")
        self.manifest["complete"] = False
        _json(self.entry / "source-manifest.json", self.manifest)
        for n in owned:
            (self.entry / n).unlink()
            del self.manifest["files"][n]
        self.manifest["cleaned_after_success"] = True
        _json(self.entry / "source-manifest.json", self.manifest)
        self._event("Removed this job's application-owned source files after validated output publication")
        return "removed downloaded inputs"

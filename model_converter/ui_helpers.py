"""Host-independent presentation helpers for the thin Forge integration."""

import html
import json
import os
from pathlib import Path
import re
import time

from .paths import clean_path, resolve_source, resolve_output
from .formats import FORMAT_SUFFIXES, PRECISION_DTYPES
from .sources import local_sets, group_source, SHARD
from .console import convrot_summary, optimization_summary


FORMAT_CHOICES = [("FP8 e4m3fn", "float8_e4m3fn"), ("FP8 e5m2", "float8_e5m2"),
                  ("INT8", "int8_tensorwise"), ("INT8 ConvRot", "int8_convrot"),
                  ("W4A8 ConvRot", "asym_w4a8_int8"), ("W6A8 ConvRot", "w6a8_int8"),
                  ("INT4 ConvRot (W4A4)", "convrot_w4a4"), ("MXFP8", "mxfp8"), ("NVFP4", "nvfp4"),
                  ("BF16 (dtype conversion)", "bf16"), ("FP16 (dtype conversion)", "fp16"),
                  ("Original precision (no quantization)", "original")]


def choice_value(choices, value):
    """Accept display labels from Forge defaults as well as protocol values."""
    return next((key for label, key in choices if value == label), value)


def discover_models(roots):
    choices = {}
    excluded = {"lora", "loras", "lycoris", "vae", "vae_approx"}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir() or root.name.casefold() in excluded:
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d.casefold() not in excluded)
            for group in local_sets(directory):
                path = group_source(directory, group)
                label = group["stem"] if group["index"] or any(SHARD.fullmatch(Path(n).name) for n in group["files"]) else path.name
                from .huggingface import provenance
                origin = provenance(path)
                prefix = (f"HF {origin['repo']}@{origin['commit'][:12]}/{origin['component'] or 'root'}"
                          if origin else f"{root.name}/{Path(directory).relative_to(root).as_posix()}")
                choices[str(path.resolve())] = f"{prefix}/{label}"
    return sorted(((label, path) for path, label in choices.items()), key=lambda item: (item[0].lower(), item[1]))


def model_folders(models):
    folders = {}
    for label, path in models:
        parent = str(Path(path).resolve().parent)
        folders.setdefault(parent, Path(label).parent.as_posix())
    # Duplicate root labels can occur with custom model roots. Include enough
    # path information to keep the folder choice unambiguous.
    counts = {}
    for label in folders.values():
        counts[label] = counts.get(label, 0) + 1
    return sorted(((label if counts[label] == 1 else f"{label} ({path})", path)
                   for path, label in folders.items()), key=lambda item: item[0].casefold())


def files_in_folder(models, folder):
    if not folder:
        return []
    folder = Path(resolve_source(folder, choices=model_folders(models)))
    return sorted({(Path(label).name, str(Path(path).resolve())) for label, path in models
                   if Path(path).resolve().parent == folder}, key=lambda item: item[0].casefold())


def selected_model_source(value, folder, models, host_root):
    value = clean_path(value)
    folder = clean_path(folder)
    if folder:
        folder = resolve_source(folder, base_dir=host_root, choices=model_folders(models))
    if folder and Path(clean_path(folder)).is_dir():
        models = [*models, *discover_models([clean_path(folder)])]
    base = folder if folder and Path(value).parent == Path(".") else host_root
    return resolve_source(value, base_dir=base, choices=[*models, *files_in_folder(models, folder)])


def output_path(source, format, override="", save_report=False, base_dir=None):
    source = resolve_source(source, base_dir)
    if Path(source).is_dir():
        from .sources import resolve_local
        source = str(resolve_local(source)[0])
    return _output_path(Path(source), format, override, save_report, base_dir)


def _output_path(source, format, override, save_report, base_dir, replace_precision=True):
    value = clean_path(override)
    if value:
        directory = Path(value)
        if not directory.is_absolute():
            directory = Path(base_dir or Path.cwd()) / directory
        if directory.is_dir() or value.endswith(("/", "\\")) or not directory.suffix:
            directory.mkdir(parents=True, exist_ok=True)
            source = directory / source.name
        else:
            return resolve_output(value, source, base_dir)
    stem = source.name.removesuffix(".safetensors.index.json").removesuffix(".safetensors")
    if SHARD.fullmatch(source.name):
        stem = SHARD.fullmatch(source.name)[1]
    # Replace only a trailing precision token, keeping model names and directories intact.
    if replace_precision:
        stem = re.sub(r"(?:^|[_.\s-]+)(?:bf16|fp16|fp32)$", "", stem, flags=re.IGNORECASE)
    suffix = FORMAT_SUFFIXES.get(format, format)
    output_stem = f"{stem}_{suffix}" if stem else suffix
    candidate = source.with_name(f"{output_stem}.safetensors")
    index = 2
    while candidate.exists() or (save_report and Path(str(candidate) + ".report.json").exists()):
        candidate = source.with_name(f"{output_stem}_{index}.safetensors")
        index += 1
    return str(candidate.resolve())


def hf_output_path(origin, format, models_root, override="", save_report=False, base_dir=None, roots=None):
    """Repository-derived names and host routes, with explicit overrides preserved."""
    component = origin.get("component", "")
    leaf = component.rsplit("/", 1)[-1]
    category = "text_encoder" if re.fullmatch(r"text_encoder(?:_\d+)?", leaf) else "Stable-diffusion"
    directory = Path((roots or {}).get(category) or Path(models_root) / category).resolve()
    sanitize = lambda value: re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "component"
    stem = sanitize(origin["repo"].rsplit("/", 1)[-1]) + ("_" + sanitize(component) if component else "")
    source = directory / f"{stem}.safetensors"
    return _output_path(source, format, override, save_report, base_dir, replace_precision=False)


def progress_html(snapshot=None, show_status=True):
    snapshot = snapshot or {}
    fraction = min(1, max(0, snapshot.get("progress", 0)))
    percent = int(fraction * 100)
    status = snapshot.get("status", "idle")
    label = snapshot.get("phase", "Ready to convert")
    if snapshot.get("cancel_requested") and not snapshot.get("done"):
        label = "Cancelling after the current operation"
    elif snapshot.get("done"):
        label = snapshot.get("message", status)
    eta = ""
    if status == "running" and not snapshot.get("done"):
        elapsed = time.monotonic() - snapshot.get("started", time.monotonic())
        if 0.02 < fraction < 0.95 and elapsed > 3:
            remaining = int(elapsed * (1 - fraction) / fraction)
            eta = f"Approx. {remaining // 60}m {remaining % 60}s remaining (estimate)"
        elif fraction >= 0.95:
            eta = "Finishing validation and saving"
        else:
            eta = "Estimating time remaining…"
    color = "#b91c1c" if status == "failed" else "var(--button-primary-background-fill, var(--primary-500, #f97316))"
    status_line = f"<div>{html.escape(label)}</div>" if show_status else ""
    return (f'<div aria-live="polite">{status_line}'
            f'<div role="progressbar" aria-label="Conversion progress" aria-valuenow="{percent}" '
            f'aria-valuemin="0" aria-valuemax="100" style="width:100%;height:18px;overflow:hidden;'
            f'border-radius:var(--button-large-radius, 8px);background:var(--background-fill-secondary, #e5e7eb)">'
            f'<div style="height:100%;width:{percent}%;background:{color};transition:width 0.25s"></div></div>'
            f'<div>{percent}% {html.escape(eta)}</div></div>') + optimization_progress_html(snapshot)


def optimization_progress_html(snapshot):
    current = snapshot.get("optimization_progress")
    if not current:
        return ""
    step, maximum = current.get("step", 0), max(1, current.get("max_steps", 1))
    percent = min(100, max(0, int(100 * step / maximum)))
    reason = current.get("stop_reason")
    labels = {"plateau": "Early stop: improvement stalled", "exact_reconstruction": "Stopped: exact reconstruction",
            "max_steps": "Maximum steps reached", "non_finite_loss": "Stopped: non-finite loss",
            "non_finite_gradient": "Stopped: non-finite gradient", "non_finite_state": "Stopped: non-finite state"}
    text = labels.get(reason, "Optimizing")
    if snapshot.get("done") and snapshot.get("status") in ("cancelled", "failed"):
        text = "Cancelled" if snapshot["status"] == "cancelled" else "Conversion failed"
    name = html.escape(str(current.get("name", "")))
    tile = f"block {current.get('tile_index', 1)}/{current.get('tile_count', 1)}"
    base, best = current.get("balanced_objective", 0), current.get("best_objective", 0)
    gain = f"; objective improved {100 * (1-best/base):.2f}%" if base > 0 and 0 <= best <= base else ""
    if current.get("status") == "running" and current.get("early_stop"):
        gain += f"; no significant gain {step-current.get('last_improvement_step', 0)}/{current.get('patience', 100)} steps"
    previous = snapshot.get("last_optimization")
    previous_line = ""
    if previous and current.get("status") == "running":
        message = (f"Previous block: {previous.get('name', '')} · {labels.get(previous.get('stop_reason'), 'Stopped')} "
                   f"at {previous.get('step', 0)}/{previous.get('max_steps', 0)} steps")
        previous_line = f'<div style="overflow-wrap:anywhere">{html.escape(message)}</div>'
    return (f'<div aria-live="polite" style="margin-top:12px"><div style="overflow-wrap:anywhere">Optimization: {name} · {tile}</div>'
            f'<div role="progressbar" aria-label="Current optimization" aria-valuenow="{step}" aria-valuemin="0" '
            f'aria-valuemax="{maximum}" style="width:100%;height:12px;overflow:hidden;border-radius:8px;'
            f'background:var(--background-fill-secondary, #e5e7eb)"><div style="height:100%;width:{percent}%;'
            f'background:var(--button-primary-background-fill, var(--primary-500, #f97316));transition:width 0.25s"></div></div>'
            f'<div>{html.escape(text)} · {step}/{maximum} steps{html.escape(gain)}</div>{previous_line}</div>')


def status_text(snapshot=None):
    if not snapshot:
        return "Ready"
    if snapshot.get("done"):
        return {"completed": snapshot.get("message", "Completed"), "failed": "Conversion failed",
                "cancelled": "Cancelled"}.get(snapshot.get("status"), "Finished")
    if snapshot.get("cancel_requested"):
        return "Cancelling after the current operation"
    return snapshot.get("phase", "Starting conversion")


def summary(snapshot):
    if not snapshot.get("done"):
        return ""
    report = snapshot.get("report") or {}
    message = snapshot.get("message", "")
    if snapshot.get("status") == "completed":
        if report.get("source_preflight"):
            source = report["source_preflight"]
            message = (f"HF metadata preview: {source['repo']} at {source['commit'][:12]}, {source['component'] or 'root'}. "
                       f"Missing source: {source['missing_bytes'] / 1024**3:.2f} GiB. No weights downloaded; architecture checks run after acquisition.")
        elif report.get("status") == "completed":
            counts = report["counts"]
            activity = (f"{counts['preserve']} tensors saved in original precision; no quantization. "
                        if report['plan']['request']['format'] == 'original' else
                        f"{counts['quantize']} weights converted; {counts['preserve']} tensors preserved. ")
            if report['plan']['request']['format'] in PRECISION_DTYPES:
                activity = (f"{counts['cast']} tensors cast to {report['plan']['request']['format'].upper()}; "
                            f"{counts['preserve']} tensors already in target dtype or non-floating retained; no quantization. ")
            message = (f"Saved: {report['plan']['request']['output']}\n"
                       f"{activity}"
                       f"Size: {report['output_bytes'] / 1024**3:.2f} GiB. "
                       f"Device: {report['plan']['request']['device'].upper()}.")
        else:
            selected = sum(e["action"] == "quantize" for e in report.get("layers", []))
            message = (f"Preview ready: {selected} weights to convert; {len(report.get('layers', [])) - selected} tensors preserved. "
                       f"Device: {report.get('request', {}).get('device', 'unknown').upper()}. Nothing has been written.")
            if report.get('request', {}).get('format') == 'original':
                message = f"Preview ready: {len(report.get('layers', []))} tensors to save in original precision. No quantization. Nothing has been written."
            elif report.get('request', {}).get('format') in PRECISION_DTYPES:
                cast = sum(e['action'] == 'cast' for e in report.get('layers', []))
                message = (f"Preview ready: {cast} tensors to cast to {report['request']['format'].upper()}; "
                           f"{len(report.get('layers', [])) - cast} tensors retained. No quantization. Nothing has been written.")
    conversion_plan = report.get("plan", report)
    optimization = optimization_summary(conversion_plan.get("optimization_selection", {}))
    if optimization:
        message += "\n" + optimization
    optimized = report.get("optimization", {})
    if optimized.get("optimized_layers"):
        message += (f"\nWeight MSE improved vs Standard in {optimized['improved_weight_mse_layers']}/{optimized['optimized_layers']} layers. "
                    + (f"SVD objective improved vs Balanced in {optimized['improved_objective_layers']}/{optimized['optimized_layers']} layers. " if optimized["mode"] == "thorough" else "")
                    + "These metrics do not establish image quality.")
        if optimized["mode"] == "thorough":
            message += (f"\nLearned steps: {optimized['actual_steps']}/{optimized.get('planned_steps', 0)}. "
                        f"Blocks stopped early: {optimized.get('early_stopped_tiles', 0)}.")
    if report.get("fallbacks"):
        from collections import Counter
        reasons = Counter()
        for fallback in report["fallbacks"]:
            reasons[fallback["reason"]] += fallback.get("occurrences", 1)
        message += "\nCandidate retentions/fallbacks: " + "; ".join(f"{reason} ({count})" for reason, count in reasons.items())
    formats = conversion_plan.get("format_counts", {})
    if len(formats) > 1:
        message += "\nMixed formats: " + ", ".join(f"{FORMAT_SUFFIXES.get(format, format)}: {count}" for format, count in formats.items())
    rotation = convrot_summary(conversion_plan.get("convrot_selection", {}))
    if rotation:
        message += "\n" + rotation
    notes = [*conversion_plan.get("loader_notes", []), *conversion_plan.get("policy_warnings", [])]
    extra = conversion_plan.get("protection_settings", {}).get("extra", {})
    if extra.get("enabled"):
        count = extra.get("additional_preserved_weights")
        message += (f"\nExtra protection: {count} additional weights preserved." if count is not None
                    else "\nExtra protection: count unavailable; see diagnostic details.")
    return message + "".join("\n" + warning for warning in [*notes, *snapshot.get("warnings", [])])

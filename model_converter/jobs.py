"""Supervise workers and clean only each job's own temporary directory."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .core import Request
from .diagnostics import exception_message, exit_message
from .console import log_console
from .paths import absolute_paths
from .formats import FORMAT_SUFFIXES
from .console import protection_summary, convrot_summary, optimization_summary
from .job_storage import JobLock, JOB_MARKER


def cleanup_owned(folder):
    """Retry transient Windows file locks after the worker has fully exited."""
    for attempt in range(5):
        try:
            folder.cleanup()
            return None
        except OSError as error:
            if attempt == 4:
                return f"Could not remove temporary files in {folder.name}: {error}"
            time.sleep(0.2 * (attempt + 1))


class Jobs:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, request, command, temp_dir=None):
        if command not in ("plan", "convert"):
            raise ValueError("Invalid worker command")
        request = Request(**asdict(request))
        if request.source_kind == "local":
            request.source, request.output = absolute_paths(request.source, request.output)
        else:
            request.output = str(Path(request.output).resolve())
            request.hf_cache_dir = str(Path(request.hf_cache_dir).resolve())
        if temp_dir:
            temp_dir = Path(temp_dir).resolve()
            temp_dir.mkdir(parents=True, exist_ok=True)
        folder = tempfile.TemporaryDirectory(prefix="model-converter-job-", dir=temp_dir)
        root = Path(folder.name)
        scratch = None
        stderr = None
        owner_lock = None
        storage_warnings = []
        try:
            owner_lock = JobLock(root / "parent.lock")
            (root / "worker.lock").touch()
            (root / "owner.json").write_text(JOB_MARKER, encoding="utf-8")
            if command == "convert":
                output_dir = Path(request.output).parent
                scratch_root = output_dir
                if temp_dir:
                    if root.stat().st_dev == output_dir.stat().st_dev:
                        scratch_root = root
                    else:
                        storage_warnings.append(f"Temporary directory {temp_dir} is on a different filesystem; output staging remains beside {request.output} for atomic publication. After a host crash, abandoned staging there requires manual cleanup.")
                scratch = tempfile.TemporaryDirectory(prefix=".converter-", dir=scratch_root)
            (root / "request.json").write_text(json.dumps(asdict(request)), encoding="utf-8")
            stderr = (root / "stderr.log").open("w", encoding="utf-8")
            args = [sys.executable, "-m", "model_converter.worker", command,
                    "--request", str(root / "request.json"), "--cancel-file", str(root / "cancel"),
                    "--job-dir", str(root)]
            if scratch:
                args.extend(["--work-dir", scratch.name])
            process = subprocess.Popen(args, cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE,
                                       stderr=stderr, text=True, encoding="utf-8", errors="replace",
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except Exception as error:
            if stderr:
                stderr.close()
            if scratch:
                cleanup_owned(scratch)
            if owner_lock:
                owner_lock.close()
            cleanup_owned(folder)
            log_console("Error", exception_message(error))
            raise
        for warning in storage_warnings:
            log_console("Warning", warning)
        log_console("Started", f"{'Conversion' if command == 'convert' else 'Preview'} process {process.pid}: {request.source} -> {request.format}; device={request.device}")
        job_id = uuid.uuid4().hex
        job = {"root": root, "events": [], "done": False, "report": None,
               "cancel_requested": False, "started": time.monotonic(), "progress": 0.0,
               "phase": "Checking model", "status": "running", "message": "", "diagnostics": "",
               "process": process, "scratch": scratch.name if scratch else None, "last_operation": None}
        job["optimization_progress"] = None
        job["last_optimization"] = None
        with self.lock:
            self.jobs[job_id] = job

        def collect():
            terminal = None
            try:
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError("Not an event object")
                    except ValueError:
                        event = {"event": "log", "message": line.rstrip()}
                    with self.lock:
                        kind = event.get("event")
                        if kind == "tensor":
                            job["last_operation"] = event
                        if kind == "optimization":
                            job["optimization_progress"] = event
                            if event.get("status") == "completed":
                                job["last_optimization"] = event
                        if kind == "source":
                            log_console("Source", event.get("message", ""))
                            job["phase"] = event.get("message", "Preparing source")
                        if kind == "progress":
                            job["progress"] = min(0.99, max(job["progress"], event.get("fraction", 0)))
                            job["phase"] = event.get("phase", "Converting weights")
                        if kind == "prepared":
                            log_console("Ready", f"{event.get('device', '?').upper()}, {event.get('export_engine', '?')}, profile={event.get('profile_label', event.get('profile', '?'))}, precision={event.get('precision_policy', 'uniform')}; {event.get('quantize', 0)} weights to convert, {event.get('preserve', 0)} tensors preserved")
                            if event.get("protection_settings"):
                                log_console("Settings", protection_summary(event["protection_settings"]))
                            if convrot_summary(event.get("convrot_selection", {})):
                                log_console("Settings", convrot_summary(event["convrot_selection"]))
                            if optimization_summary(event.get("optimization_selection", {})):
                                log_console("Settings", optimization_summary(event["optimization_selection"]))
                            if len(event.get("format_counts", {})) > 1:
                                log_console("Formats", ", ".join(f"{FORMAT_SUFFIXES.get(format, format)}={count}" for format, count in event["format_counts"].items()))
                            for note in event.get("policy_warnings", []):
                                log_console("Warning", note)
                        if kind in ("result", "error", "cancelled"):
                            terminal = kind
                            job["message"] = event.get("message", "")
                            if kind == "result":
                                job["report"] = event["result"]
                            elif event.get("report"):
                                job["report"] = event["report"]
                        job["events"].append(event)
                        job["events"] = job["events"][-100:]
                code = process.wait()
                with self.lock:
                    job["exit_code"] = code
                    if terminal == "cancelled" and code == 2:
                        job["status"] = "cancelled"
                    elif code == 0 and terminal == "result":
                        job.update(status="completed", progress=1.0,
                                   message="Conversion complete" if command == "convert" else "Preview ready")
                    else:
                        job["status"] = "failed"
                        if terminal != "error" or code not in (0, 1):
                            job["message"] = exit_message(code, terminal == "result")
                            if job["last_operation"]:
                                operation = job["last_operation"]
                                job["message"] += f" Last operation: {operation['operation']}; tensor {operation['name']} {operation['shape']}."
                        job["events"].append({"event": "worker_exit", "exit_code": code, "message": job["message"]})
            except Exception as error:
                # Never remove a live worker's files even if protocol collection fails.
                process.wait()
                with self.lock:
                    job.update(status="failed", message=f"Could not monitor the conversion process: {error}")
            finally:
                process.stdout.close()
                stderr.close()
                try:
                    with (root / "stderr.log").open("rb") as log:
                        log.seek(max(0, log.seek(0, 2) - 16384))
                        diagnostics = log.read().decode("utf-8", errors="replace")
                except OSError as error:
                    diagnostics = str(error)
                warnings = list(storage_warnings)
                owner_lock.close()
                for owned in (scratch, folder):
                    if owned:
                        warning = cleanup_owned(owned)
                        if warning:
                            warnings.append(warning)
                with self.lock:
                    job["diagnostics"] = diagnostics
                    job["warnings"] = warnings
                    job["elapsed"] = time.monotonic() - job["started"]
                    if job["status"] == "completed":
                        report = job["report"] or {}
                        if command == "convert":
                            counts = report.get("counts", {})
                            activity = (f"{counts.get('preserve', 0)} tensors saved in original precision" if request.format == "original" else
                                        f"{counts.get('quantize', 0)} weights converted, {counts.get('preserve', 0)} tensors preserved")
                            log_console("Completed", f"{activity}; {report.get('output_bytes', 0) / 1024**3:.2f} GiB in {job['elapsed']:.1f}s; saved {request.output}")
                            optimization = report.get("optimization", {})
                            if optimization.get("optimized_layers"):
                                learned = (f"{optimization['actual_steps']}/{optimization.get('planned_steps', 0)} learned steps; "
                                           f"{optimization.get('early_stopped_tiles', 0)} tiles stopped early; "
                                           if optimization["mode"] == "thorough" else "0 learned steps; ")
                                objective = (f"SVD objective improved vs Balanced in {optimization['improved_objective_layers']}/{optimization['optimized_layers']} layers; "
                                             if optimization["mode"] == "thorough" else "")
                                log_console("Optimization", f"{optimization['mode'].title()}; {objective}weight MSE improved vs Standard in {optimization['improved_weight_mse_layers']}/{optimization['optimized_layers']} layers; "
                                            f"{learned}{optimization['fallbacks']} reported candidate retentions/fallbacks")
                        else:
                            log_console("Preview", f"{sum(e['action'] == 'quantize' for e in report.get('layers', []))} weights selected; no checkpoint written")
                            if report.get("protection_settings"):
                                log_console("Settings", protection_summary(report["protection_settings"]))
                            if convrot_summary(report.get("convrot_selection", {})):
                                log_console("Settings", convrot_summary(report["convrot_selection"]))
                            if optimization_summary(report.get("optimization_selection", {})):
                                log_console("Settings", optimization_summary(report["optimization_selection"]))
                            for note in report.get("policy_warnings", []):
                                log_console("Warning", note)
                        for note in report.get("plan", report).get("loader_notes", []):
                            log_console("Warning", note)
                    elif job["status"] == "cancelled":
                        log_console("Cancelled", job["message"])
                    else:
                        log_console("Error", job["message"])
                    for warning in warnings:
                        log_console("Warning", warning)
                    job["done"] = True

        threading.Thread(target=collect, daemon=True).start()
        return job_id

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            return {key: (list(value) if isinstance(value, list) else value)
                    for key, value in job.items() if key not in ("root", "process")}

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job and not job["done"] and job["process"].poll() is None:
                (job["root"] / "cancel").touch()
                job["cancel_requested"] = True

    def release(self, job_id):
        with self.lock:
            if job_id in self.jobs and self.jobs[job_id]["done"]:
                self.jobs.pop(job_id)

"""JSON-lines worker protocol and cooperative cancellation."""

import argparse
import json
from pathlib import Path
import sys
import faulthandler

from .backend import capabilities
from .core import Request, convert, plan
from .diagnostics import exception_message
from .job_storage import JobLock


class Cancelled(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "convert", "capabilities"))
    parser.add_argument("--request", help="Path to a JSON Request")
    parser.add_argument("--cancel-file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--work-dir", help=argparse.SUPPRESS)
    parser.add_argument("--job-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Keep stdout exclusively for protocol events, including during library imports.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    # Native failures cannot raise a Python exception. Capture their current
    # stack in diagnostics, without relaying it to the Forge console.
    faulthandler.enable(file=sys.stderr)

    def emit(event):
        protocol.write(json.dumps(event, ensure_ascii=True) + "\n")
        protocol.flush()

    def cancelled():
        if args.cancel_file and Path(args.cancel_file).exists():
            raise Cancelled("Conversion cancelled; no incomplete checkpoint was published")

    job_lock = None
    try:
        if args.job_dir:
            job_lock = JobLock(Path(args.job_dir) / "worker.lock")
        if args.command == "capabilities":
            emit({"event": "capabilities", "result": capabilities(args.device)})
        else:
            if not args.request:
                raise ValueError("--request is required")
            request = Request(**json.loads(Path(args.request).read_text(encoding="utf-8")))
            cancelled()
            result = plan(request, emit=emit, cancelled=cancelled) if args.command == "plan" else convert(request, emit, cancelled, args.work_dir)
            emit({"event": "result", "result": result})
        return 0
    except Cancelled as error:
        if getattr(error, "conversion_report", None):
            error.conversion_report["status"] = "cancelled"
        emit({"event": "cancelled", "message": str(error), "report": getattr(error, "conversion_report", None)})
        return 2
    except Exception as error:
        if getattr(error, "conversion_report", None):
            error.conversion_report["status"] = "failed"
        emit({"event": "error", "type": type(error).__name__, "message": exception_message(error),
              "report": getattr(error, "conversion_report", None)})
        # Keep ordinary CLI errors readable, too. Library diagnostics may still
        # be captured on stderr, but we never print a Python traceback here.
        print(f"[Model Converter] Error: {exception_message(error)}", file=sys.stderr)
        return 1
    finally:
        if job_lock:
            job_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

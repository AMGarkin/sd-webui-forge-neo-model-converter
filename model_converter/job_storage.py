"""Owned job directories and crash-released locks, independent of the host."""

import errno
import os
from pathlib import Path
import shutil


class JobLock:
    def __init__(self, path):
        if path.is_symlink():
            raise ValueError("Unsafe job lock path")
        self.stream = path.open("a+b")
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise BlockingIOError("Job temporary files are in use") from None
            raise
        except BaseException:
            self.stream.close()
            raise

    def close(self):
        self.stream.close()


JOB_MARKER = '{"kind": "model-converter-job-v1"}'


def cleanup_stale_jobs(temp_dir):
    """Remove only marked jobs whose parent and worker have both released locks."""
    warnings = []
    temp_dir = Path(temp_dir).resolve()
    for root in temp_dir.glob("model-converter-job-*"):
        if (root.is_symlink() or getattr(root, "is_junction", lambda: False)()
                or root.resolve().parent != temp_dir or not root.is_dir()):
            continue
        locks = []
        try:
            marker = root / "owner.json"
            if marker.is_symlink() or marker.read_text(encoding="utf-8") != JOB_MARKER:
                continue
            try:
                for name in ("parent.lock", "worker.lock"):
                    locks.append(JobLock(root / name))
            except BlockingIOError:
                continue  # A live parent or worker still owns this directory.
            # A late-starting worker must fail before reading its request.
            (root / "request.json").unlink(missing_ok=True)
            for lock in locks:
                lock.close()
            locks.clear()
            shutil.rmtree(root)
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as error:
            warnings.append(f"Could not remove abandoned job {root}: {error}")
        finally:
            for lock in locks:
                lock.close()
    return warnings

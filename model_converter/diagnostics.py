"""Actionable errors without guessing that every native crash is out of memory."""

import errno


def exception_message(error):
    message = str(error)
    if isinstance(error, FileNotFoundError):
        return f"File or directory not found: {error.filename or message}"
    if isinstance(error, FileExistsError):
        return f"Output already exists. Choose a new output path. {error.filename or message}"
    if isinstance(error, PermissionError):
        return f"Access denied. Check file permissions or whether the file is in use. {error.filename or message}"
    if isinstance(error, OSError) and (error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112):
        return f"Not enough disk space. Free space on the output drive and try again. {message}"
    if isinstance(error, MemoryError) or "out of memory" in message.lower():
        return "Not enough RAM or GPU memory. Unload the active Forge model, close other GPU tasks, or select CPU in Advanced options. " + message
    return message


def exit_message(code, published=False):
    unsigned = code & 0xFFFFFFFF
    if unsigned == 0xC0000005:
        message = ("The conversion process crashed in native code (Windows access violation, 0xC0000005). "
                   "This code alone does not identify low memory or disk space. "
                   "The last operation and native stack are in Diagnostic details; include them in a bug report.")
    elif unsigned in (0xC0000017, 0xC000009A):
        message = "Windows could not provide enough memory or system resources. Close other workloads and try again."
    else:
        message = f"The conversion process stopped unexpectedly (exit code {code}, 0x{unsigned:08X}). See diagnostic details."
    if published:
        message += " The worker reported a completed checkpoint before stopping; the file was retained, but the run did not exit cleanly."
    else:
        message += " No successful conversion was confirmed."
    return message

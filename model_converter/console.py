"""Short Forge-console messages; worker protocol stdout remains untouched."""


def log_console(level, message):
    # Collapse multi-line exceptions/compiler diagnostics and control characters.
    # Full diagnostics stay in the UI, not in routine Forge console output.
    lines = str(message).splitlines()
    text = next((line.strip() for line in lines if line.strip()), "Unknown error")
    text = "".join(char if char.isprintable() else " " for char in text)
    text = text[:600]
    try:
        print(f"[Model Converter] {level}: {text}", flush=True)
    except (OSError, UnicodeError):
        # A detached console must not turn a valid checkpoint into a failed job.
        pass


def protection_summary(settings):
    """Describe actual user controls, including an unavailable extra-rule set."""
    if settings.get("applied") is False:
        precision = settings.get("output_precision", "original")
        return ("original precision; all canonical tensors retained without quantization" if precision == "original" else
                f"{precision.upper()} export; floating tensors use target dtype; non-floating buffers retained; no quantization or layer protections")
    flag = lambda value: "on" if value else "off"
    extra = settings.get("extra", {})
    if not extra.get("available"):
        extra_text = "not available"
    elif extra.get("enabled"):
        added = extra.get("additional_preserved_weights")
        extra_text = "on" + (f" (+{added} preserved)" if added is not None else " (count unavailable)")
    else:
        extra_text = "off"
    return (f"all layer protections={flag(settings.get('all', True))}, general protection={flag(settings.get('general'))}, model protection={flag(settings.get('model'))}, "
            f"profile format overrides={flag(settings.get('profile_formats'))}, extra={extra_text}")


def convrot_summary(selection):
    counts = selection.get("counts", {})
    if not counts:
        return ""
    mode = "dynamic minimum" if selection.get("dynamic") else "fixed"
    return (f"ConvRot {mode} {selection['requested_group_size']}; actual groups (weights): "
            + ", ".join(f"{group}: {count}" for group, count in counts.items()))


def optimization_summary(selection):
    if not selection or not selection.get("applied", True):
        return ""
    mode = selection.get("mode", "standard")
    if mode == "thorough":
        optimizer = ("Prodigy + Schedule-Free (Schedule-Free off)" if selection['optimizer'] == 'prodigy' else "AdamW")
        if selection.get('use_speed') and selection['optimizer'] == 'prodigy':
            optimizer += " + SPEED"
        stopping = (f"early stop on (patience {selection.get('patience', 100)}, improvement {selection.get('min_delta', .001):.2%})"
                    if selection.get('early_stop', True) else "early stop off")
        return f"Optimization: Thorough; {optimizer}, up to {selection['steps']} steps per tile; {stopping}; data-free SVD objective"
    return f"Optimization: {mode.title()}" + ("; bounded scale search" if mode == "balanced" else "")

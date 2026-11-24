"""Forge Neo model discovery and a simple conversion interface."""

import time
from copy import copy
from pathlib import Path

import gradio as gr
from modules import script_callbacks
from modules.paths_internal import models_path, script_path
from modules.ui_components import ToolButton

from model_converter.core import Request
from model_converter.formats import UNQUANTIZED_FORMATS
from model_converter.optimization import MODE_CHOICES, OPTIMIZER_CHOICES, THOROUGH_FORMATS
from model_converter.console import log_console
from model_converter.diagnostics import exception_message
from model_converter.jobs import Jobs
from model_converter.job_storage import cleanup_stale_jobs
from model_converter.paths import clean_path, resolve_source
from model_converter.profiles import profile_description, detect_profile, qwen3vl_identity
from model_converter.policies import PRECISION_CHOICES
from model_converter.profile_registry import model_profile_choices, canonical_ui_profile, profile_label
from model_converter.references import ReferenceProtection
from model_converter.io import inspect
from model_converter.layouts import inspect_canonical
from model_converter.huggingface import inventory as hf_inventory, provenance
from model_converter.ui_helpers import (FORMAT_CHOICES, discover_models, model_folders, files_in_folder,
                                        choice_value, selected_model_source, output_path, hf_output_path, progress_html, status_text, summary)


jobs = Jobs()
HF_CACHE = Path(models_path) / "ModelConverterSources"
SOURCE_CHOICES = [("Local disk", "local"), ("Hugging Face", "huggingface")]


def host_directories():
    roots = {}
    try:
        from modules import shared
        options = shared.cmd_opts
        for category, attr in (("Stable-diffusion", "ckpt_dir"), ("text_encoder", "text_encoder_dir")):
            configured = list(getattr(options, attr + "s", []) or [])
            if getattr(options, attr, None):
                configured.insert(0, getattr(options, attr))
            if configured:
                roots[category] = configured
    except (ImportError, AttributeError):
        pass
    return roots


def host_roots():
    # Forge supports multiple search roots. Use the first explicitly configured
    # directory as an automatic destination, and inventory every configured root.
    return {category: paths[0] for category, paths in host_directories().items()}


def model_choices():
    roots = [Path(base) / category for base in (models_path, Path(script_path) / "models")
             for category in ("Stable-diffusion", "text_encoder")]
    return discover_models([*roots, HF_CACHE,
                            *(path for paths in host_directories().values() for path in paths)])


def host_temp_dir():
    try:
        from modules import shared
        value = clean_path(shared.opts.temp_dir)
    except (ImportError, AttributeError):
        return None
    return resolve_source(value, script_path) if value else None


def cleanup_host_jobs():
    try:
        from modules import shared
        enabled = shared.opts.clean_temp_dir_at_start
    except (ImportError, AttributeError):
        return
    directory = host_temp_dir()
    if enabled and directory:
        for warning in cleanup_stale_jobs(directory):
            log_console("Warning", warning)


def build_tab():
    cleanup_host_jobs()
    inventory = model_choices()
    folders = model_folders(inventory)
    initial_folder = folders[0][1] if folders else None
    with gr.Blocks() as panel:
        with gr.Row(equal_height=False, elem_id="model_converter_main"):
            with gr.Column(scale=1, min_width=320, elem_id="model_converter_inputs"):
                # Forge UiLoadsave validates dropdown/radio defaults against labels.
                source_kind = gr.Radio(choices=[label for label, _ in SOURCE_CHOICES],
                                       value="Local disk", label="Source")
                with gr.Group() as local_inputs:
                    with gr.Row():
                        folder = gr.Dropdown(choices=folders, value=initial_folder, label="Model folder", scale=8, allow_custom_value=True)
                        refresh = ToolButton(value="\U0001f504", elem_id="model_converter_refresh",
                                             tooltip="Refresh model folders and files")
                    source = gr.Dropdown(choices=files_in_folder(inventory, initial_folder), allow_custom_value=True,
                                         label="Model file, or paste a full path", value=None)
                with gr.Group(visible=False) as hf_inputs:
                    hf_repo = gr.Textbox(label="Hugging Face repository", placeholder="OpenMOSS-Team/MOSS-Image-SFT")
                    hf_revision = gr.Textbox(label="Revision (optional)", placeholder="main, tag or immutable commit")
                    hf_list = gr.Button("List repository components")
                    hf_component = gr.Dropdown(choices=[], label="Component folder", value=None)
                    hf_set = gr.Dropdown(choices=[], label="Weight set / dtype variant", value=None)
                    hf_keep = gr.Checkbox(value=True, label="Keep downloaded source files after conversion",
                        info="Complete sources remain reusable from Local disk, including offline. Failed/partial downloads are preserved for retry.")
                    hf_status = gr.Markdown("Only the chosen weight set and its config/index metadata will be downloaded.")
                hf_catalog = gr.State({})
                format = gr.Dropdown(choices=[label for label, _ in FORMAT_CHOICES], value="FP8 e4m3fn", label="Output format",
                    info="BF16/FP16 cast all floating tensors, including norms and biases, on CPU. Non-floating buffers stay unchanged. These exports use Standard without quantization or layer protections. FP16 overflow fails explicitly.")
                optimization_mode = gr.Dropdown(choices=[label for label, _ in MODE_CHOICES], value="Standard", label="Optimization",
                    info="Standard: direct conversion. Balanced: scale search. Thorough: scale search and learned rounding; takes longer. Reconstruction metrics do not guarantee image quality.")
                gr.Markdown("Local outputs are saved beside the source. HF outputs use the host's diffusion/text-encoder model folder. "
                            "Sources and existing outputs are never overwritten.")
            with gr.Column(scale=1, min_width=320, elem_id="model_converter_run"):
                with gr.Row():
                    start = gr.Button("Quantize model", variant="primary", scale=4, min_width=180)
                    cancel = gr.Button("Cancel", scale=1, min_width=90)
                progress = gr.HTML(value=progress_html(show_status=False))
                status = gr.Textbox(label="Status", value="Ready", lines=1, max_lines=2, interactive=False)
                result = gr.Textbox(label="Result", lines=2, interactive=False)
        state = gr.State("")
        with gr.Accordion("Advanced options", open=False):
            output = gr.Textbox(label="Output path (optional)", placeholder="Folder or filename.safetensors; empty = automatic",
                info="A .safetensors filename selects a file; an existing folder, a path ending in / or \\, or a name without an extension selects a folder (created if needed). Relative folders use the Forge root; bare filenames are saved beside the source. Folder outputs get an automatic model/format filename.")
            device = gr.Dropdown(choices=["Automatic", "GPU (CUDA)", "CPU"],
                                 value="Automatic", label="Conversion device", allow_custom_value=True)
            with gr.Row():
                convrot_group_size = gr.Dropdown(choices=["4", "16", "64", "256", "1024"], value="256",
                    label="ConvRot group size", info="INT8 ConvRot and INT4 ConvRot only. With Dynamic on, this is the minimum group size.")
                dynamic_convrot = gr.Checkbox(value=False, label="Dynamic ConvRot groups",
                    info="Choose the largest compatible group per layer, from the selected minimum up to 1024. Use minimum 64 or 16 to include narrower widths. W4A8/W6A8 require 256 in the target CUDA loader.")
            with gr.Accordion("Thorough optimization settings", open=False):
                optimization_optimizer = gr.Dropdown(choices=[label for label, _ in OPTIMIZER_CHOICES],
                    value="Prodigy + Schedule-Free", label="Learned-rounding optimizer",
                    info="Prodigy + Schedule-Free requires optional version 2.0.1. Schedule-Free is disabled for this rounding algorithm. AdamW uses Torch. No packages are installed automatically.")
                optimization_speed = gr.Checkbox(value=False, label="Use Prodigy SPEED",
                    info="Changes step-size estimation; does not guarantee a faster conversion. Applies only to Prodigy + Schedule-Free.")
                # Gradio 4.40 omits None from initial props; Number then starts at zero.
                optimization_steps = gr.Number(value=0, minimum=0, maximum=4000, precision=0,
                    label="Maximum learned steps per tile",
                    info="0 or empty = automatic: 2000 steps for either optimizer. A positive number overrides the default.")
                optimization_seed = gr.Number(value=0, minimum=0, maximum=4294967295, precision=0, label="Optimization seed")
                optimization_early_stop = gr.Checkbox(value=True, label="Early stopping",
                    info="Stop a block when the best accepted quantized objective stops improving enough. Keeps the best accepted result; does not prove a global optimum.")
                optimization_patience = gr.Number(value=100, minimum=25, maximum=4000, precision=0,
                    label="Early-stop patience (steps)")
                optimization_min_gain = gr.Number(value=.1, minimum=0, maximum=99.9,
                    label="Early-stop minimum improvement (%)",
                    info="Cumulative relative improvement needed to reset patience. Checks every 25 steps after at least 100 steps; exact reconstruction can stop immediately.")
                gr.Markdown("AdamW is configured automatically: learning rate 0.05, betas (0.9, 0.99), weight decay 0, epsilon 1e-8. "
                            "These settings apply to rounding logits, not model training.")
                gr.Markdown("Thorough uses a bounded SVD objective derived from source weights, without prompt/image calibration. "
                            "It retains the best accepted discrete candidate and permits at most 5% higher weight MSE than Balanced. "
                            "Thorough is available for all nine quantized formats, including W4A8/W6A8.")
            with gr.Row():
                with gr.Column(min_width=320):
                    profile = gr.Dropdown(choices=model_profile_choices(),
                                          value="auto", label="Conversion policy")
                    profile_help = gr.Markdown(profile_description("auto"))
                with gr.Column(min_width=320):
                    precision_policy = gr.Dropdown(choices=[label for label, _ in PRECISION_CHOICES],
                        value=PRECISION_CHOICES[0][0], label="Precision preset")
                    gr.Markdown("Mixed presets change attention storage only. Fused attention/MLP projections use one format for the entire tensor. "
                                "Other eligible layers use the format selected above.")
            with gr.Accordion("Layer protection and profile rules", open=False):
                protect_layers = gr.Checkbox(value=True, label="Enable layer protections",
                    info="Turn off to bypass all preservation rules: comfy-model-tools, CTQ/Star/native rules, lookup tables, small matrices, custom keep rules and packed dimension probes. This can reduce quality or break target loading. Tensor-type and format-dimension requirements still apply.")
                protect_general = gr.Checkbox(value=True, label="General sensitive-layer protection",
                    info="Use general comfy-model-tools criteria and CTQ common exclusions.")
                protect_model = gr.Checkbox(value=True, label="Model-specific layer protection",
                    info="Use the selected model's combined CTQ, Star and native protection rules.")
                profile_formats = gr.Checkbox(value=False, label="Use profile format overrides", interactive=False,
                    info="Allow the profile to use different quantization formats for selected layers. Off uses the format chosen above, unless you set custom rules or a mixed preset.")
                pruned_protection = gr.Checkbox(value=False, label="Extra protection for pruned models", interactive=False,
                    info="Available only when this profile defines extra rules and model-specific protection is enabled. This does not prune weights.")
                gr.Markdown("The master switch overrides all layer preservation rules below. "
                            "Only supported floating-point 2-D weights with valid format dimensions can be quantized. "
                            "Input widths must fit the selected ConvRot group. INT4 also requires a multiple of 64; W4A8/W6A8 retain rotation group 256. "
                            "Biases, norms and convolutions keep their source dtypes; quantization scales can be FP32.")
            with gr.Row():
                first_blocks = gr.Number(value=0, minimum=0, maximum=10000, precision=0,
                                         label="Keep first blocks in original precision (per stack)")
                last_blocks = gr.Number(value=0, minimum=0, maximum=10000, precision=0,
                                        label="Keep last blocks in original precision (per stack)")
            with gr.Accordion("Custom layer rules (optional)", open=False):
                preserve_patterns = gr.Textbox(label="Keep layers in original precision", lines=3,
                                               placeholder="One full tensor-name glob per line, e.g. *.mlp.down_proj.weight")
                layer_rules = gr.Textbox(label="Layer format overrides (JSON)", lines=4,
                                         placeholder='[{"pattern": "*.self_attn.k_proj.weight", "format": "int8_convrot"}]')
                gr.Markdown("Patterns match full tensor names (case sensitive); `*` matches any characters. "
                            "JSON rules use `pattern` and `format` (a filename suffix or canonical format ID, or `preserve`). "
                            "CTQ/Star model protection, general safety, keep patterns and boundary blocks take precedence; otherwise the first matching rule wins. "
                            "Unmatched rules are reported in the preview/result. Rules cannot enable protected or non-linear weights.")
            save_report = gr.Checkbox(value=False, label="Save detailed JSON report beside the checkpoint")
            gr.Markdown("The protection switches control preservation of embeddings and sensitive layers, including matching CTQ and Star rules. "
                        "Tensor types and format dimensions always apply. "
                        "The target application must support comfy_quant for the converted modules.")
            with gr.Row():
                save_defaults = gr.Button("Save converter defaults", size="sm", scale=0, min_width=180)
                defaults_status = gr.Textbox(label="Settings save status", interactive=False, show_label=False,
                                            container=False, scale=1, lines=1)
            gr.Markdown("Save these settings as defaults in Forge's ui-config.json. You can also use Settings → Defaults. "
                        "The source model (local path or HF selection) and conversion policy are excluded; policy starts on automatic detection. "
                        "Saved defaults apply after restarting the UI.")
        with gr.Accordion("Diagnostic details", open=False):
            preview = gr.Button("Preview conversion (optional)")
            gr.Markdown("Preview shows which weights will be converted and the estimated file size. "
                        "The same checks run automatically when you quantize; preview is not required. "
                        "Time remaining is an estimate and includes writing and validation.")
            diagnostics = gr.Textbox(label="Technical details", lines=5, interactive=False)
            report = gr.JSON(label="Conversion details / preview")

        saved_components = [source_kind, hf_keep, format, optimization_mode, output, device,
                            convrot_group_size, dynamic_convrot, optimization_optimizer, optimization_speed,
                            optimization_steps, optimization_seed, optimization_early_stop, optimization_patience,
                            optimization_min_gain, precision_policy, protect_layers, protect_general, protect_model,
                            profile_formats, pruned_protection, first_blocks, last_blocks, preserve_patterns,
                            layer_rules, save_report]
        for component in (folder, source, hf_repo, hf_revision, hf_component, hf_set, profile,
                          status, result, diagnostics, defaults_status):
            component.do_not_save_to_config = True

        def save_converter_defaults(*values):
            try:
                from modules import shared
                defaults = copy(shared.demo.ui_loadsave)
                selected = dict(zip(saved_components, values))
                defaults.component_mapping = {path: component for path, component in defaults.component_mapping.items()
                                              if component in selected}
                if len(defaults.component_mapping) != len(saved_components):
                    raise ValueError("Forge's UI defaults registry is incomplete. Restart the UI and try again.")
                message = defaults.ui_apply(*(selected[component] for component in defaults.component_mapping.values()))
                return message + " Saved defaults apply after restarting the UI."
            except Exception as error:
                message = "Could not save converter defaults: " + exception_message(error)
                log_console("Error", message)
                return message

        save_defaults.click(save_converter_defaults, saved_components, defaults_status, queue=False)

        def launch(selected_folder, src, dst, fmt, dev, selected_profile, write_report, previous,
                   selected_precision="uniform", first=0, last=0, keep_patterns="", overrides="", pruned=False,
                   general=True, model=True, formats=False, kind="local", repo="", revision="", component=None,
                   weight_set=None, keep=True, catalog=None, protections=True, rotation_group=256, dynamic_rotation=False,
                   optimization="standard", optimizer="prodigy", speed=False, steps=None, seed=0,
                   early_stop=True, patience=100, min_gain=.1, command="convert"):
            if previous:
                try:
                    active = jobs.snapshot(previous)
                except KeyError:
                    active = None
                if active and not active["done"]:
                    message = "A conversion is already running. Cancel it or wait for it to finish."
                    log_console("Warning", message)
                    return (previous, progress_html(active, show_status=False), message, "", active["report"],
                            gr.update(interactive=False), gr.update(interactive=False), status_text(active))
                jobs.release(previous)
            launch_attempted = False
            try:
                fmt = choice_value(FORMAT_CHOICES, fmt)
                kind = choice_value(SOURCE_CHOICES, kind)
                selected_precision = choice_value(PRECISION_CHOICES, selected_precision)
                optimization = choice_value(MODE_CHOICES, optimization)
                optimizer = choice_value(OPTIMIZER_CHOICES, optimizer)
                origin = None
                hf_fields = {}
                if kind == "huggingface":
                    if not catalog or catalog.get("repo") != repo.strip() or catalog.get("requested_revision") != (revision.strip() or "main"):
                        raise ValueError("List the repository components for the current repository/revision first.")
                    if component is None or not weight_set:
                        raise ValueError("Choose a component folder and weight set first.")
                    origin = {"repo": repo.strip(), "component": "" if component == "." else component}
                    hf_fields = dict(source_kind=kind, hf_repo=repo.strip(), hf_revision=catalog["commit"],
                                     hf_component=origin["component"], hf_set=weight_set,
                                     hf_cache_dir=str(HF_CACHE.resolve()), hf_keep=keep)
                    src = repo.strip()
                else:
                    if not src or not clean_path(src):
                        raise ValueError("Select a model or paste its file path first.")
                    src = selected_model_source(src, selected_folder, model_choices(), script_path)
                    origin = provenance(src)
                target = (hf_output_path(origin, fmt, models_path, dst, write_report, base_dir=script_path, roots=host_roots())
                          if origin else output_path(src, fmt, dst, write_report, base_dir=script_path))
                if kind == "local" and clean_path(dst):
                    target = output_path(src, fmt, dst, write_report, base_dir=script_path)
                # Only automatic host routes are created; an explicit path keeps
                # the existing requirement that its parent already exists.
                if origin and not clean_path(dst):
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
                launch_attempted = True
                job_id = jobs.start(Request(src, target, fmt, dev, canonical_ui_profile(selected_profile),
                                            allow_experimental=True, save_report=write_report,
                                            precision_policy=selected_precision, keep_first_blocks=first,
                                            keep_last_blocks=last, preserve_patterns=keep_patterns or "",
                                            layer_rules=overrides or "", pruned_protection=pruned,
                                            protect_general=general, protect_model=model, use_profile_formats=formats,
                                            protect_layers=protections,
                                            convrot_group_size=int(rotation_group), dynamic_convrot=dynamic_rotation,
                                            optimization_mode=optimization, optimization_optimizer=optimizer,
                                            optimization_use_speed=speed, optimization_steps=None if steps in (None, 0) else int(steps), optimization_seed=int(seed),
                                            optimization_early_stop=early_stop, optimization_patience=int(patience),
                                            optimization_min_delta=float(min_gain) / 100,
                                            source_provenance=origin, **hf_fields), command, temp_dir=host_temp_dir())
            except Exception as error:
                message = exception_message(error)
                if not launch_attempted:
                    log_console("Error", message)
                # Gradio 4.40 prints tracebacks even for gr.Error. Return a UI
                # failure state for expected input/startup errors instead.
                return ("", progress_html({"status": "failed", "done": True, "message": message}, show_status=False),
                        message, "", None, gr.update(interactive=True), gr.update(interactive=True), "Could not start conversion")
            return (job_id, progress_html({"phase": "Starting conversion", "status": "running"}, show_status=False), "", "", None,
                    gr.update(interactive=False), gr.update(interactive=False), "Starting conversion")

        def monitor(job_id):
            while job_id:
                try:
                    snapshot = jobs.snapshot(job_id)
                except KeyError:
                    return
                details = snapshot.get("diagnostics", "")
                if snapshot.get("exit_code"):
                    details = f"Process exit code: {snapshot['exit_code']}\n" + details
                yield (progress_html(snapshot, show_status=False), summary(snapshot), details, snapshot["report"],
                       gr.update(interactive=snapshot["done"]), gr.update(interactive=snapshot["done"]), status_text(snapshot))
                if snapshot["done"]:
                    return
                time.sleep(0.3)

        inputs = [folder, source, output, format, device, profile, save_report, state,
                  precision_policy, first_blocks, last_blocks, preserve_patterns, layer_rules, pruned_protection,
                  protect_general, protect_model, profile_formats, source_kind, hf_repo, hf_revision,
                  hf_component, hf_set, hf_keep, hf_catalog, protect_layers, convrot_group_size, dynamic_convrot,
                  optimization_mode, optimization_optimizer, optimization_speed, optimization_steps, optimization_seed,
                  optimization_early_stop, optimization_patience, optimization_min_gain]
        for button, command in ((preview, "plan"), (start, "convert")):
            button.click(lambda *args, cmd=command: launch(*args, command=cmd), inputs,
                         [state, progress, result, diagnostics, report, start, preview, status], queue=False).then(
                             monitor, state, [progress, result, diagnostics, report, start, preview, status], concurrency_limit=None)

        def stop(job_id):
            if job_id:
                jobs.cancel(job_id)

        cancel.click(stop, state, [], queue=False)

        def select_folder(selected_folder):
            choices = model_choices()
            if selected_folder:
                selected_folder = resolve_source(selected_folder, script_path, model_folders(choices))
            if selected_folder and Path(clean_path(selected_folder)).is_dir():
                choices += discover_models([clean_path(selected_folder)])
            return gr.update(choices=files_in_folder(choices, clean_path(selected_folder)), value=None)

        def refresh_models(selected_folder, selected_file):
            models = model_choices()
            choices = model_folders(models)
            if selected_folder:
                selected_folder = resolve_source(selected_folder, script_path, choices)
            if selected_folder not in {path for _, path in choices}:
                selected_folder = choices[0][1] if choices else None
            files = files_in_folder(models, selected_folder)
            if selected_file not in {path for _, path in files}:
                selected_file = None
            return gr.update(choices=choices, value=selected_folder), gr.update(choices=files, value=selected_file)

        folder.input(select_folder, folder, source, queue=False)
        def update_start(fmt):
            return gr.update(value="Save model" if choice_value(FORMAT_CHOICES, fmt) in UNQUANTIZED_FORMATS else "Quantize model")
        format.change(update_start, format, start, queue=False)
        panel.load(update_start, format, start, queue=False)
        def update_optimization(fmt, mode):
            fmt = choice_value(FORMAT_CHOICES, fmt)
            mode = choice_value(MODE_CHOICES, mode)
            choices = MODE_CHOICES if fmt in THOROUGH_FORMATS else MODE_CHOICES[:2] if fmt not in UNQUANTIZED_FORMATS else MODE_CHOICES[:1]
            # Reset an unavailable selection visibly when the output format changes.
            selected = next((label for label, value in choices if value == mode), "Standard")
            return gr.update(choices=[label for label, _ in choices], value=selected,
                             interactive=fmt not in UNQUANTIZED_FORMATS)
        format.change(update_optimization, [format, optimization_mode], optimization_mode, queue=False)
        panel.load(update_optimization, [format, optimization_mode], optimization_mode, queue=False)
        refresh.click(refresh_models, [folder, source], [folder, source], queue=False)
        def update_source_kind(kind):
            kind = choice_value(SOURCE_CHOICES, kind)
            return gr.update(visible=kind == "local"), gr.update(visible=kind == "huggingface")
        source_kind.change(update_source_kind, source_kind, [local_inputs, hf_inputs], queue=False)
        panel.load(update_source_kind, source_kind, [local_inputs, hf_inputs], queue=False)

        def list_hf_components(repo, revision):
            try:
                catalog = hf_inventory(repo.strip(), revision.strip(), HF_CACHE)
                folders = sorted({g["folder"] for g in catalog["sets"]})
                choices = [(f or "Repository root", f or ".") for f in folders]
                component = choices[0][1] if len(choices) == 1 else None
                sets = [(g["stem"], g["stem"]) for g in catalog["sets"] if g["folder"] == ("" if component == "." else component)]
                message = f"Commit `{catalog['commit']}`. " + ("Offline retained metadata. " if catalog["offline"] else "")
                message += "Choose a component and weight set; conversion checks space before downloading."
                log_console("Source", f"Listed {repo.strip()} at {catalog['commit'][:12]}: {len(folders)} components")
                return catalog, gr.update(choices=choices, value=component), gr.update(choices=sets, value=sets[0][1] if len(sets) == 1 else None), message
            except Exception as error:
                message = exception_message(error)
                log_console("Error", message)
                return {}, gr.update(choices=[], value=None), gr.update(choices=[], value=None), message

        def select_hf_component(component, catalog):
            folder = "" if component == "." else component
            sets = [(g["stem"], g["stem"]) for g in catalog.get("sets", []) if g["folder"] == folder]
            return gr.update(choices=sets, value=sets[0][1] if len(sets) == 1 else None)

        hf_list.click(list_hf_components, [hf_repo, hf_revision], [hf_catalog, hf_component, hf_set, hf_status])
        hf_component.input(select_hf_component, [hf_component, hf_catalog], hf_set, queue=False)
        def update_profile_options(selected_profile, selected_folder, src, general, model, formats, extra, kind="local", protections=True, fmt="float8_e4m3fn"):
            selected_profile = canonical_ui_profile(selected_profile)
            kind = choice_value(SOURCE_CHOICES, kind)
            fmt = choice_value(FORMAT_CHOICES, fmt)
            if fmt in UNQUANTIZED_FORMATS:
                description = ("Original precision export retains tensor dtypes and values." if fmt == "original" else
                               f"{fmt.upper()} export casts all floating tensors, including protected layers, norms and biases; non-floating buffers retain their dtypes and values. FP16 overflow fails without publishing output.")
                return (description + " Canonical layout adaptation still applies. Quantization policies and precision rules are inactive.",
                        gr.update(interactive=False), gr.update(interactive=False))
            general = bool(general and protections)
            model = bool(model and protections)
            protection_status = ("All layer protections are off. " if not protections else
                                 f"General protection: {'on' if general else 'off'}. Model-specific protection: {'on' if model else 'off'}. ")
            try:
                if selected_profile == "auto":
                    if kind == "huggingface":
                        return (protection_status + "Automatic policy is resolved from canonical keys after source acquisition. Select a model policy manually to enable its optional rules before downloading.",
                                gr.update(interactive=False), gr.update(interactive=False))
                    if not src:
                        return ("Select a source model to show available profile rules.",
                                gr.update(interactive=False), gr.update(interactive=False))
                    actual = selected_model_source(src, selected_folder, model_choices(), script_path)
                    tensors = inspect_canonical(actual)[0]
                    selected_profile = detect_profile(tensors, fmt)
                    identity = qwen3vl_identity(tensors)
                else:
                    identity = None
                reference = ReferenceProtection(selected_profile, {})
            except (ValueError, OSError) as error:
                return ("Profile availability: " + exception_message(error),
                        gr.update(value=False, interactive=False), gr.update(value=False, interactive=False))
            extra_available = bool(model and reference.extra_available)
            format_available = bool(protections and reference.format_overrides_available)
            extra_status = "Available" if extra_available else "Not available for this profile" if model else "Model-specific protection is off"
            recipe_status = ("Off while layer protections are disabled" if not protections else
                             "On" if formats and format_available else
                             "Available; enable below to use it" if format_available else "No format overrides defined")
            label = (identity['label'] + ' · ' if identity else '') + profile_label(selected_profile)
            rules = (profile_description(selected_profile) if model else
                     "The identified profile's layer-preservation rules are not applied. Tensor types and format dimensions still apply.")
            description = (f"**{label}**: {protection_status}{rules}\n\n"
                           f"Extra protection: {extra_status}. Profile format recipe: {recipe_status}.")
            return (description, gr.update(value=bool(extra and extra_available), interactive=extra_available),
                    gr.update(value=bool(formats and format_available), interactive=format_available))

        control_inputs = [profile, folder, source, protect_general, protect_model, profile_formats, pruned_protection, source_kind, protect_layers, format]
        control_outputs = [profile_help, pruned_protection, profile_formats]
        profile.change(update_profile_options, control_inputs, control_outputs, queue=False)
        source.change(update_profile_options, control_inputs, control_outputs, queue=False)
        format.change(update_profile_options, control_inputs, control_outputs, queue=False)
        protect_model.input(update_profile_options, control_inputs, control_outputs, queue=False)
        protect_general.input(update_profile_options, control_inputs, control_outputs, queue=False)
        protect_layers.change(update_profile_options, control_inputs, control_outputs, queue=False)
        def update_protection_controls(enabled):
            return gr.update(interactive=enabled), gr.update(interactive=enabled)
        protect_layers.change(update_protection_controls, protect_layers, [protect_general, protect_model], queue=False)
        panel.load(update_protection_controls, protect_layers, [protect_general, protect_model], queue=False)
        source_kind.change(update_profile_options, control_inputs, control_outputs, queue=False)
        panel.load(update_profile_options, control_inputs, control_outputs, queue=False)
    return [(panel, "Model Converter", "model_converter")]


script_callbacks.on_ui_tabs(build_tab)

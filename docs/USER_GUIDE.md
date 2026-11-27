# User guide

[Back to README](../README.md)

- [Requirements](#requirements)
- [Choosing a source](#choosing-a-source)
- [Formats and compatibility](#formats-and-compatibility)
- [Optimization](#optimization)
- [Layer policies and precision rules](#layer-policies-and-precision-rules)
- [Output paths and saved defaults](#output-paths-and-saved-defaults)
- [Disk space, memory and cancellation](#disk-space-memory-and-cancellation)
- [Diagnostics and troubleshooting](#diagnostics-and-troubleshooting)
- [Command-line use](#command-line-use)

## Requirements

Install into Forge Neo as described in the [README](../README.md#installation).
Use the host's Python environment; do not install another Torch, CUDA or Kitchen
build just for this extension. ComfyUI, CTQ and StarNodes installations are not required.

The development environment used Windows, an NVIDIA RTX 4060, Python 3.13.17,
Torch 2.15.0 (CUDA 13.2 and later 13.4 builds), safetensors 0.8.0,
comfy-kitchen 0.2.41 and Gradio 4.40.0. HF acquisition was tested with
huggingface-hub 0.36.2; optional Prodigy must be exactly 2.0.1. These are tested
versions, not upgrade instructions or a minimum-version guarantee. Other operating
systems/hardware are not independently validated. Concrete API probes check the
selected format and optimization mode before writing.

Audited snapshots include Forge Neo `b0c9f6c3eecf1adaeb940e12a0c73ae40bfc758a`
for format/rotation checks, Forge `a057d9ede903eb51c38ceb20e186b6841665d250`
for saved defaults, and ComfyUI `860a7fdc7772bcb53f74a675d0592d89f94fa265`
for layer/adapter checks. These do not establish every future revision's compatibility.

**Conversion device** defaults to **Automatic**: CUDA when available, otherwise
CPU. Explicit CPU/CUDA conversion is also available. Errors do not trigger an
automatic device change. The extension does not unload an active Forge model;
free that memory yourself if needed. Original-precision and BF16/FP16 exports run on CPU.

## Choosing a source

### Local files and shards

Use **Model folder**, then **Model file, or paste a full path**. Both lists are
searchable. Refresh after adding files. Quoted Windows paths are accepted.

Discovery includes nonempty `Stable-diffusion` and `text_encoder` trees under
Forge's model roots, configured external diffusion/text-encoder directories, and
retained converter downloads. LoRA/LyCORIS and VAE folders are excluded by name.
Other files remain accessible by full path. Inventory is not a compatibility check.

Inputs can be unquantized BF16/FP16/FP32 safetensors weights, complete numbered
shard sets, explicit shard indexes, or component directories with one unambiguous
set. Selecting any numbered shard resolves the full set. An index, when present,
must agree with all files/tensors. Missing shards, duplicates, conflicting metadata
and already quantized inputs fail. Multiple variants/components require a choice;
they are not merged automatically. Pickle checkpoints, GGUF and LoRA merging are
outside this converter. Do not replace source files during conversion.

### Hugging Face

Enter `owner/repository`, optionally a branch/tag/commit under **Revision**, and
click **List repository components**. Select the component and weight set/dtype
variant. The resolved immutable commit is used for every file. Only the selected
set, metadata and required dependencies are acquired, not the whole repository.
LTX-2/2.5 also needs sibling connectors from that revision; original MiniMax H3
Image needs its PDD schedule. Repository Python is never executed.

Existing HF authentication is used. Accept gated-model terms and configure
authentication in the host environment beforehand; there is no token-management
UI. Authentication/invalid-revision errors do not silently select cached alternatives.

**Keep downloaded source files after conversion** defaults on. Sources live under
`<configured models root>/ModelConverterSources` and appear by repository/commit/
component in Local disk. Complete retained sources work offline there. HF offline
reuse of a retained catalog is reported. Partial downloads remain for retry;
size checks and available LFS checksums detect incomplete/corrupted transfers.

With Keep off, successful conversion removes only files acquired by that job,
after output publication. Pre-existing sources, local files, other jobs' inputs
and the global HF cache are preserved. Failed/cancelled jobs retain inputs.
Small metadata/manifests can remain. Close Forge and standalone workers before
manually removing retained sources.

HF preview fetches bounded metadata without downloading weights. With complete
retained weights, the normal detailed layer plan is available.

## Formats and compatibility

### Output formats

All nine quantized formats support Standard, Balanced and Thorough; individual
layers must still satisfy format dimensions and loader requirements.

| UI format | Filename suffix | Storage |
| --- | --- | --- |
| FP8 E4M3FN | `fp8_e4m3` | E4M3FN weights and a scalar scale |
| FP8 E5M2 | `fp8_e5m2` | E5M2 weights and a scalar scale |
| INT8 | `int8` | Tensorwise INT8 |
| INT8 ConvRot | `int8_convrot` | Rotated INT8 with row scales |
| INT4 ConvRot (W4A4) | `w4a4_convrot` | Packed INT4 with row scales |
| W4A8 | `w4a8_convrot` | Symmetric grouped 4-bit codes with INT8 reconstruction |
| W6A8 | `w6a8_convrot` | Symmetric grouped 6-bit codes with INT8 reconstruction |
| NVFP4 | `nvfp4` | Packed FP4, FP8 block scales and an FP32 global scale |
| MXFP8 | `mxfp8` | FP8 weights and E8M0 block scales |
| BF16 (dtype conversion) | `bf16` | Floating tensors cast to BF16; no quantization metadata |
| FP16 (dtype conversion) | `fp16` | Floating tensors cast to FP16; no quantization metadata |
| Original precision (no quantization) | `original` | Original dtypes; no quantization metadata |

There is no single best format for all models. Lower bit widths can reduce size
further but change generation more. Protected weights stay in source precision,
so size is not simply parameter count times bit width. Compare representative
prompts against the original before relying on a conversion.

Quantized exports use per-module `comfy_quant`, canonical names, packed weights and scales.
All request `full_precision_matrix_mult=true`: the host dequantizes for compute-dtype
multiplication. This does not force FP32 or establish accelerated low-bit inference,
speed gains or total VRAM savings. NVFP4/MXFP8 conversion explicitly uses Kitchen
eager on CPU/CUDA; export on an RTX 4060 does not establish Blackwell acceleration.
NVFP4 uses groups of 16 and pads weights to multiples of 16; MXFP8 uses groups
of 32 and pads to 32. Their scale storage uses Kitchen's blocked layout.

Original precision assembles shards and applies implemented layout adaptations
without quantizing or casting. It requires Standard, is not dequantization, and
cannot supply a missing target architecture.

**BF16 / FP16 (dtype conversion)** uses the same canonical adapters and shard
assembly, then casts every floating tensor (including norms, biases, embeddings
and buffers) to the chosen dtype. Integer/bool buffers and metadata are retained.
These exports require Standard and ignore quantization protections, precision
presets and per-layer format rules. Existing tensors in the target dtype retain
their values. This is lossy precision conversion, without scales or `comfy_quant`.

For example, select the `transformer` component from `Tongyi-MAI/Z-Image-Turbo`
and choose **BF16 (dtype conversion)** to save its FP32 weights as BF16. For
FP32 tensors the output payload uses half as many bytes; download size is unchanged.
FP16 has a narrower numeric range than BF16. Non-finite sources or cast overflow
fail explicitly before publication; no values are clamped or format substituted.
Preview estimates payload size but cannot detect overflow without reading weights.
Casting alone does not establish full-model loading, generation quality or speed.

### Compatibility

Conversion structurally validates output before publication. It does not execute
a complete model or generate a test image. Structural validation, quantized-layer
loading/forwards, full-model loading and generation are separate evidence levels.

BF16/FP16 dtype exports have synthetic file, strict Linear loading and forward
checks in both Forge Neo and ComfyUI on CPU/CUDA. Full Z-Image conversion and
generation from these new exports have not been independently verified.

These user-generated examples provide representative functional evidence, not a
complete compatibility matrix or a general quality ranking:

| Model / converted component | Format and modes | Generation evidence |
| --- | --- | --- |
| MOSS-Image SFT canonical diffusion checkpoint | FP8 E4M3FN/E5M2, INT8, INT8 ConvRot, W4A8, W6A8; original MOSS policy | Forge Neo, user-reported |
| Anima diffusion model | INT8 ConvRot; Standard, Balanced, Thorough AdamW/Prodigy; additional Prodigy SPEED examples | Forge Neo, matched user-generated examples |
| Z-Image Turbo diffusion model | INT4 ConvRot; Standard, Balanced, Thorough AdamW/Prodigy/SPEED | Forge Neo, three matched prompt/seed groups |
| Qwen3 4B text encoder in a Z-Image workflow | NVFP4; Standard, Balanced, Thorough AdamW/Prodigy | Forge Neo, three matched prompt/seed groups |
| FLUX.2 Klein 9B diffusion model | W4A8, include-MLP policy, general protection off; Standard, Balanced, Thorough AdamW/Prodigy/SPEED | ComfyUI, three matched prompt/seed groups; Forge generation of this combination is not established |

W6A8 Thorough has structural and target-layer evidence, but no recorded full-model
generation. These examples do not validate every mixed rule, source layout, model
version or jointly quantized component. Generation-time Git revisions were not
independently established for every user run; Klein PNG metadata records comfy-core
0.46.0. Treat the examples as case-specific evidence rather than release certification.

### Source layouts and host limits

Adapters cover audited variants of MOSS, FLUX.1/2, Z-Image, SD3, PixArt, AuraFlow,
Diffusers UNet, Wan (including VACE/Animate), Hunyuan Video 1/1.5, LTX Video and
LTX-2/2.5, Mochi, Lumina2, HiDream, Chroma, Cosmos/Predict2, Ideogram, Krea, Ming,
MiniMax H3, Fibo 2.0, RMBG-3.0 and YuE2. Native/Transformers namespaces are
retained where appropriate, including multimodal Qwen3-VL.

Automatic policy selection reads the selected component's keys and applicable
config after acquisition. This allows the transformer and text encoder from
one repository to use different policies. Known Fibo layouts select the Fibo
recipe. An official BRIA RMBG-3.0 HF source with those same keys selects
RMBG-3.0. Select the RMBG-3.0 policy manually for a standalone copy with no
retained HF source record.

Known limits:

- A host may support quantized Linear layers while lacking the complete model.
  Audited Forge snapshots lacked detectors for several added architectures,
  including Hunyuan Video, Mochi, LTX-2, HiDream, Cosmos and Ideogram. Check the
  current host's architecture support separately.
- Sana had no implementation in the inspected hosts. Alternate FLUX.3 layouts,
  unspecified Lens/NeRF architectures and arbitrary new Wan extensions
  are not established by catalog labels.
- Generic text/visual profiles do not prove component-loader support. The inspected
  SheetSage2 audio loader does not consume `comfy_quant`.
- VAE mapping, convolution quantization and VAE encode/decode validation are outside
  supported functionality. Convolution-only inputs have no eligible linear weights.
  VAE-labelled reference recipes do not remove this limit.
- PixArt sources can omit the unused eval-time `y_embedder.y_embedding` buffer;
  this is reported instead of replaced with invented values.
- LTX-2 requires matching companion text encoder/projection handling. MiniMax Image
  requires its validated PDD schedule. Layer success alone does not prove a pipeline.
- Unknown inputs keep names and receive an experimental/unverified notice. Known
  unsupported variants report their adapter gap. Universal cannot invent arbitrary
  key remapping or a missing architecture.

The model/recipe choices include pinned StarNodes/CTQ data and a separate manual
MiniMax Compact map. HQ, MaxQ and All Versions
are recipe names, not compatibility/quality guarantees. GGUF, AWQ, SVDQuant, CTQ
blockwise FP8/INT8 and hybrid MXFP8 are not aliases for these exporters.
See [layout sources](HF_LAYOUT_SOURCES.md) and the README's license notices.

## Optimization

Modes change offline effort and reconstructed weights, not bit width, enabled
protections or the host inference kernel.

| Mode | Work | Extra optimizer package |
| --- | --- | --- |
| Standard | Direct Kitchen conversion; default | None |
| Balanced | Searches scales/clipping; retains Standard when no better candidate is found | None |
| Thorough | Balanced plus adjacent-grid learned rounding using a weight-derived SVD objective | None with AdamW; Prodigy 2.0.1 when selected |

See [optional Prodigy installation](../README.md#optional-prodigy). Thorough defaults
to **Prodigy + Schedule-Free**; choose **AdamW** explicitly if Prodigy is absent.
Missing/incompatible Prodigy fails preflight rather than silently choosing AdamW.
Schedule-Free is disabled.

Under **Advanced options → Thorough optimization settings**:

- **Use Prodigy SPEED** defaults off. It changes step-size estimation, can increase
  or decrease conversion time, and affects neither AdamW nor inference.
- **Maximum learned steps per tile**: zero/empty means automatic **2000** for either
  optimizer. An explicit integer 1–4000 takes priority.
- **Optimization seed** defaults to 0; range 0–4294967295. It controls rounding,
  not image-generation seeds.
- **Early stopping** defaults on, with 100-step patience and 0.1% minimum cumulative
  relative improvement. Checks run every 25 steps after at least 100; exact
  reconstruction can stop immediately. Patience accepts 25–4000; the UI improvement
  threshold accepts 0–99.9 percent. Disable for a fixed maximum budget. An early
  stop is not a failure or proof of a global optimum.

The second progress bar shows the current block and preceding stop result. Higher
caps can increase runtime despite early stopping. AdamW is automatic: learning
rate 0.05, betas (0.9, 0.99), decay 0, epsilon 1e-8. Prodigy uses rate 1,
initial distance 1e-4, the same betas/decay/epsilon, factored FP32 moments and no
stochastic optimizer rounding. These optimize rounding decisions, not model training.

Balanced measures actual discrete reconstructions and stored scales. Thorough
uses source-weight tiles and up to 16 right singular vectors `V`:

```text
J = mean((D - W)^2) + mean(((D - W) @ V)^2)
```

`W` is the source tile and `D` the decoded candidate, in the rotated basis where
applicable. There is no prompt/image or captured-activation calibration. Thorough
keeps the best hard-rounded candidate only when this objective improves and
original-basis weight MSE is at most 105% of Balanced's. Ordinary MSE can therefore
be slightly higher than Balanced. A binary regularizer ramps over
`min(maximum steps, 400)`; larger caps extend that trajectory. W4A8/W6A8 fitting
includes their rounded/clamped INT8 decode grid.

Weight MSE, SVD objective and image quality are different measures. Lower objective
or longer conversion does not guarantee closer/better images. Retentions/fallbacks
include reasons. Aggregate counts are per-layer/reason entries, not failed-layer
counts; some tiles can retain Balanced while the layer improves overall. OOM,
missing APIs and invalid output fail rather than changing format/device/budget.

## Layer policies and precision rules

### Automatic and manual policies

Default **Detect model profile (otherwise universal)** uses an audited signature
or Universal. Native signatures include canonical MOSS SFT, FLUX.1, FLUX.2/Klein,
Z-Image, Anima and Qwen3; Qwen3-VL signatures select the Qwen VLM policy.
FLUX.2/Klein with W4A8/W6A8 selects include-MLP; other formats use the BFL policy.
A manual policy takes priority.

Universal selects eligible FP32/FP16/BF16 `*.weight` matrices with both dimensions
at least 64. Validated native policies allow dimensions from 8. Default protections
preserve lookup tables, sensitive names, biases, buffers, convolutions and incompatible
shapes. Names/shapes are heuristics, not proof of module type. Preserved tensors
keep values/dtypes; metadata is retained subject to canonical layout adaptations.

### Protection controls

Keep **Enable layer protections** on unless deliberately testing its effects.
Turning the master switch off bypasses all preservation heuristics: lookups,
small matrices, custom keep rules, boundary blocks, attention preservation and
packed architecture probes. It also disables profile format recipes. This can
reduce quality or produce a structurally valid file that the target cannot run.
Tensor-type, storage, format-dimension and no-overwrite checks still apply.

With the master on:

- **General sensitive-layer protection** defaults on: name exclusions informed by
  comfy-model-tools and CTQ.
- **Model-specific layer protection** defaults on: applicable native/CTQ/StarNodes
  family rules are combined.
- **Use profile format overrides** defaults off: opt in to a recipe's per-layer
  formats instead of only the main format/preset/custom rules.
- **Extra protection for pruned models** defaults off, available only for recipes
  with extra rules and enabled model protection. It does not prune weights. The
  additional-preserved count excludes redundant matches.

Turning off the two name switches alone does not bypass lookup/type/shape or
loader-probe protections. Packed Flux input projections, for example, stay in
source precision when storage shape would mislead architecture detection. Turning
the master off bypasses that safeguard too.

### Presets and custom rules

Default **Precision preset** uses the main format for all eligible weights.
Other presets keep attention in source precision or use FP8 E4M3FN/INT8 ConvRot.
Fused attention/MLP projections use one format for the entire tensor.

**Keep first/last blocks** protects a count per numerically ordered transformer
stack. Zero disables it; a count exceeding a stack protects all of it. Protecting
every candidate fails rather than labelling an unchanged checkpoint quantized.

**Keep layers in original precision** accepts one full tensor-name, case-sensitive
glob per line, such as `*.mlp.down_proj.weight`. **Layer format overrides (JSON)**
accepts an ordered list:

```json
[
  {"pattern": "*.self_attn.k_proj.weight", "format": "int8_convrot"},
  {"pattern": "*.mlp.down_proj.weight", "format": "fp8_e4m3"}
]
```

Use a filename suffix from the format table, canonical format ID, or `preserve`.
Enabled protections win, followed by preserve globs, boundary blocks, the first
matching JSON rule, attention preset, enabled profile rules and the main format.
Rules cannot force unsupported tensor types/dimensions. Unmatched rules and blocked
overrides are reported. Filenames identify the main format; inspect details for mixtures.

### MiniMax H3 Compact W4/W6/INT8

Select **MiniMax H3 Compact W4/W6/INT8 (LokkenJP map)** under Conversion policy,
then enable **Use profile format overrides**. Keep **Enable layer protections**
and **Model-specific layer protection** on, the uniform precision preset, zero
first/last-block keeps, and no custom keep/format rules for the exact map.
Use fixed **ConvRot group size 256**, with Dynamic off. **W4A8 ConvRot** is a
convenient main-format choice; the enabled recipe sets each target's actual format.
Choose Standard, Balanced or Thorough according to the existing optimization
settings. The profile remains manual and does not change automatic detection.

This reproduces the precision assignments from
[LokkenJP's ErosMax v1 Compact report](https://huggingface.co/LokkenJP/TenEros_Max_optimized_learned/blob/3415250db9e2cb89b886f5375c1306fc933e215f/10Eros_Max_8stepTurbo_v1.0_hybrid_activation_learned_compact_w4w6int8_14.50gb.safetensors.quantization.json):
64 W4A8 and 122 W6A8 main matrices, plus 14 main and eight token-refiner INT8
ConvRot matrices. Other tensors retain their dtypes and values after canonical
layout adaptation. Exactly 50 main blocks and two refiner blocks are required;
the recipe fails on incompatible layouts instead of extrapolating indices.
Advanced settings can change assignments and are reported as a modified recipe.

With profile format overrides enabled, Compact stores W4A8/W6A8 group scales in
**FP8 E4M3**, with FP32 channel scales. Other recipes keep their existing FP32
group scales. Standard quantizes directly with the selected scale precision;
Balanced and Thorough fit and measure candidates on that same stored FP8 grid.
The converter never casts already optimized FP32 scales afterward.

Group scales are fractional multipliers read directly by the W4A8/W6A8 loader,
not Linear weights. INT8 ConvRot quantizes and rotates weight matrices; it is
not a replacement scale dtype. Both grouped weight formats already use ConvRot.
Quantizing the scale buffers to integers would need another scale and a different
loader contract, not just another per-module `comfy_quant` entry. Compact retains
FP8 scales to reduce storage while representing their varying fractional values;
FP32 scales are the existing higher-precision alternative in other recipes.

For the audited fused source, 1,141,604,352 group scales account for 4.57 decimal
GB in FP32 and 1.14 GB in FP8. Compact's planned payload is now approximately
**14.51 GB**, saving 3.42 GB versus its earlier FP32-scale implementation.
This excludes the safetensors header. Source FP32 folded biases/F64 buffers
remain intact; the result need not match the original artifact byte for byte.

The author's checkpoint also uses activation-calibrated learned weights, an
optimized W4 codebook and discrete polishing. This converter uses symmetric
codebook-free grids and data-free optimization. Matching the precision map
and scale storage does not reproduce learned values or generation quality.
The reference's 64 small W4 codebooks total only 4096 bytes; their purpose is
reconstruction quality rather than the multi-gigabyte size saving. See the author's
[technical overview](https://huggingface.co/LokkenJP/TenEros_Max_optimized_learned/blob/3415250db9e2cb89b886f5375c1306fc933e215f/H3_EROSMAX_V1_TECHNICAL_OVERVIEW.md)
for the original method and source-specific evidence.

The HF adapter supports the audited rank-8 folded AdaLN variant in
`ibyteohdear/10Eros-Max-Transformer_8stepTURBO_v1.0_hybrid_VBVR_FUSED`.
Select its root component and `diffusion_pytorch_model` set; all five shards
form one input. It maps the timestep table, folded biases and QKV, and swaps
Diffusers SwiGLU halves from up/gate to native gate/up without executing
repository Python.
Earlier pruned-adapter exports retained the wrong SwiGLU half order and need
regeneration. Changing protection or quantization settings cannot repair them.
It retains FP32 folded biases and the F64 basis buffer. `adaln_basis` and
`adaln_mean` remain in the file for provenance/adapter use; the ordinary ComfyUI
model does not consume them. The existing **Extra protection for pruned models**
switch is unrelated and is unavailable for this recipe.

Actual source-header planning and ComfyUI constructor/shape checks passed, along
with synthetic mixed export and layer forwards on CPU/CUDA. The owner confirms
ComfyUI video generation with the corrected ErosMax INT8 artifact and a fresh
conversion after the SwiGLU fix. The fresh run's precise settings were not supplied;
this does not establish generation for every format, profile or MiniMax variant.
The audited Forge installation has no MiniMax model detector; supported quantized
layers do not supply it.

### ConvRot groups

**ConvRot group size** supports 4/16/64/256/1024 for INT8 ConvRot and INT4 ConvRot.
Default: fixed 256. **Dynamic ConvRot groups** chooses the largest compatible group
per layer between the selected minimum and 1024. Minimum 256 does not recover width
1152; minimum 64 can. INT8 widths such as 4304 can use minimum 16. Actual groups/
counts appear in results/reports.

INT4 independently requires width divisible by 64. W4A8/W6A8 retain rotation 256
because audited CUDA loaders reject others; scale groups are 16. W6A8 also needs
width divisible by 32. Other formats/original export ignore these controls.
Smaller/dynamic groups can make more matrices eligible but do not prove quality
improvement or bypass layer protections.

## Output paths and saved defaults

Local outputs default beside the source. HF diffusion/transformer/uncertain
components use Forge's diffusion directory; `text_encoder`/`text_encoder_N` use
its text-encoder directory. Configured external destinations are respected.
Retained HF inputs keep HF destination routing.

**Output path (optional)** accepts:

| Input | Result |
| --- | --- |
| Empty | Automatic destination and model/format filename |
| `custom.safetensors` | Explicit filename beside the local source |
| Absolute `.safetensors` filename | Exact file; parent must already exist |
| Existing folder, trailing `/` or `\`, or extensionless name | Folder, created if needed, with an automatic filename |
| `exports` | Folder under the Forge root |
| Relative path with directories | Resolved against the Forge root in the UI |

Automatic names replace a trailing BF16/FP16/FP32 token or append the format suffix.
HF names include repository/component identity. Automatic collisions receive
numeric suffixes, including report collisions. Occupied explicit destinations
and racing writers fail without replacement.

**Save converter defaults** at the end of Advanced options saves this tab's
preferences in Forge's `ui-config.json`. **Settings → Defaults → Apply** includes
them too. Restart the UI to apply them. Source folders/files, HF source identity
and conversion policy are excluded; policy always starts automatic. Format,
optimization, destination override, device, rules and remaining preferences are
retained. Other tabs' defaults are preserved.

## Disk space, memory and cancellation

The writer needs about twice the output payload in free temporary space, plus
the source. Before HF acquisition, missing source bytes and a conservative
two-times-source output/staging estimate are reserved, with a margin. Shared
volumes sum those requirements. Complete retained sources skip the source-sized
guess; the actual layer plan checks output/staging afterward. Pre-download guesses
can exceed what a particular quantized export needs.

A separate worker processes buffers depending on the largest layer and tiles,
not a whole output-model dictionary. Optimized modes/NVFP4 use bounded row
processing; other Standard paths can process a full layer. FP32 intermediates,
optimizer state, packed output and scales still need RAM/VRAM. Output size is not
a peak-memory estimate.

UI jobs use Forge's temporary directory where possible. Empty settings use system
TEMP for requests/diagnostics and staging beside the output. Across filesystems,
staging explicitly falls back beside the output. Atomic no-overwrite publication
needs hard-link support, which some network/removable filesystems lack.

**Cancel** is cooperative: the in-flight operation finishes first. Downloads may
wait for a network timeout. Final writing/validation can take time without an
ETA. Completion requires a clean worker exit and cleanup. An error after output
publication can leave a valid checkpoint while the job is failed; read the result
before retrying.

The parent cleans job files after success/cancellation/worker crashes. If Forge
itself dies, converter-owned abandoned directories under its configured temporary
root can be removed at the next startup with Forge's startup temp cleanup enabled.
Parent/worker locks protect live jobs. Beside-output staging and unsupervised CLI
leftovers may need manual cleanup after confirming no converter uses them.
Retained HF sources are separate from temporary cleanup.

## Diagnostics and troubleshooting

**Diagnostic details → Preview conversion** shows layer choices, preservation
reasons and estimated size. The same checks run automatically on conversion;
preview is optional. **Conversion details / preview** includes resolved settings,
versions, per-layer formats, final metrics, retentions/fallbacks and validation levels.

**Save detailed JSON report beside the checkpoint** retains details as
`<output>.report.json`; default is off. Report publication is separate from the
checkpoint. Report-write failure remains an error even if the checkpoint is saved.
Existing reports are never replaced.

| Symptom | Action |
| --- | --- |
| Missing/incompatible Prodigy | Install exactly 2.0.1 with the host interpreter, or explicitly choose AdamW/another mode. |
| Missing Kitchen API / unsupported format | Read the capability error and choose a supported format/environment. Do not replace host dependencies blindly. |
| No eligible weights | Check source linear weights and protection reasons. A convolution-only VAE is unsupported. |
| Output exists | Choose a new explicit filename or automatic naming. |
| Not enough disk | Free space on the source/output/staging volumes named in the message. |
| Out of memory | Unload the active Forge model, close other workloads, or use CPU where appropriate. CPU still needs RAM. |
| Model loads but generation fails | Check architecture/component support and protections; recreate from originals if packed dimension probes were quantized. |
| Windows `0xC0000005` | Native access violation: inspect last operation/diagnostics. The code alone does not establish OOM. |
| Cleanup/report failure after saving | The checkpoint may remain; inspect the result/path before retrying. |

For bug reports include extension revision, host/package versions, model repository/
revision or architecture, format/mode/device, settings, final error and diagnostic
tail. Review reports before sharing: they can include local paths and repository
identifiers. Do not post access tokens or model weights.

## Command-line use

Optional: run from the extension folder with the host interpreter. The tab does
not require JSON requests or CLI setup.

```powershell
$forgePython = 'D:/Forge/venv/Scripts/python.exe' # Replace with your host interpreter.
& $forgePython -m model_converter.worker capabilities --device cuda
& $forgePython -m model_converter.worker plan --request request.json
& $forgePython -m model_converter.worker convert --request request.json --cancel-file cancel.flag
```

Example local request:

```json
{
  "source": "D:/models/model_bf16.safetensors",
  "output": "D:/models/model_int8_convrot.safetensors",
  "profile": "auto",
  "format": "int8_convrot",
  "device": "auto",
  "allow_experimental": true,
  "save_report": true,
  "optimization_mode": "thorough",
  "optimization_optimizer": "adamw",
  "optimization_steps": null,
  "optimization_early_stop": true,
  "optimization_patience": 100,
  "optimization_min_delta": 0.001
}
```

Unlike the UI, the worker needs an explicit output filename and existing parent.
Relative directories use the invocation directory; bare local output filenames
remain source-relative. Format IDs: `float8_e4m3fn`, `float8_e5m2`,
`int8_tensorwise`, `int8_convrot`, `convrot_w4a4`, `asym_w4a8_int8`, `w6a8_int8`,
`nvfp4`, `mxfp8`, `bf16`, `fp16`, `original`.

Mode IDs: `standard`, `balanced`, `thorough`; optimizer IDs: `adamw`, `prodigy`.
Omitted/null steps mean 2000; numeric zero is rejected unlike the UI sentinel.
Use `optimization_use_speed` for SPEED and `optimization_seed` for seed. API gain
is a fraction: 0.001 means 0.1 percent. Additional fields are declared in
[Request](../model_converter/core.py).

HF requests set `source_kind: "huggingface"`, `hf_repo`, `hf_revision`,
`hf_component` (empty for root), `hf_set`, an application-owned `hf_cache_dir`,
and `hf_keep`. An explicit output path remains required.

Stdout is JSON lines, diagnostics use stderr. Exit codes: 0 success, 1 failure,
2 cancellation. Create the cancellation file to cancel; remove it before reusing
that path for another job.

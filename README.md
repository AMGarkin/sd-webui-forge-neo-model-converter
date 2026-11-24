# Forge Neo Model Converter

An experimental extension for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
that converts diffusion models and text encoders into smaller `.safetensors` files.
Use local models or download a selected component from Hugging Face. Source models
and existing outputs are never overwritten.

Available formats: **FP8 E4M3FN, FP8 E5M2, INT8, INT8 ConvRot, INT4 ConvRot,
W4A8, W6A8, NVFP4 and MXFP8**. Three optimization modes let you choose how much
time to spend on conversion. Original-precision export can also assemble shards
without quantization or dtype casting.

Outputs use the `comfy_quant` storage format understood by Forge Neo and ComfyUI.
Compatibility depends on the model, component loader and installed libraries;
a listed profile is not a guarantee that every model variant works. Smaller files
do not guarantee faster generation or unchanged image quality.

## Installation

The easiest way is to install directly from Forge Neo:

1. Open **Extensions → Install from URL**.
2. Enter this repository URL:

   ```text
   https://github.com/AMGarkin/sd-webui-forge-neo-model-converter
   ```

3. Click **Install** and wait for installation to finish.
4. Restart Forge Neo and open the **Model Converter** tab.

Alternatively, close Forge Neo and clone the repository from a terminal in its
installation folder:

```sh
git clone https://github.com/AMGarkin/sd-webui-forge-neo-model-converter.git extensions/sd-webui-forge-neo-model-converter
```

You can also download this repository as a ZIP, extract it, and place the folder
directly under `extensions` as `sd-webui-forge-neo-model-converter`. The folder
must contain `scripts`, `model_converter` and this README. Start Forge Neo afterward.

The extension uses Forge Neo's existing Python, Torch, safetensors and
comfy-kitchen packages. It does not install or upgrade them. No separate ComfyUI
installation is required. See the [user guide](docs/USER_GUIDE.md#requirements)
for the tested environment and compatibility limits.

### Optional: Prodigy

**Standard**, **Balanced**, and **Thorough with AdamW** work without Prodigy.
To use Thorough with **Prodigy + Schedule-Free**, install version **2.0.1** in
the Python environment used by Forge Neo. With Forge closed, run this PowerShell
command from its installation folder:

```powershell
& .\venv\Scripts\python.exe -m pip install --no-deps "prodigy-plus-schedule-free==2.0.1"
```

If your environment is named `.venv`, replace `venv` with `.venv`. For a custom
or portable installation, use its actual Python executable. `--no-deps` keeps
pip from changing Forge's dependencies. Restart Forge after installing.

Despite the package's name, Schedule-Free is disabled in this converter. Prodigy
is an optional optimizer for learned rounding, not another output format.

## Quick start

1. Open **Model Converter**. Select **Local disk**, choose a model folder and
   file, or paste a full path. For **Hugging Face**, enter `owner/repository`,
   click **List repository components**, and select a component and weight set.
2. Choose an **Output format** and **Optimization**:

   - **Standard**: direct conversion; the default.
   - **Balanced**: extra scale search.
   - **Thorough**: scale search plus learned rounding; takes longer. Without
     Prodigy, select **AdamW** under **Advanced options → Thorough optimization settings**.

3. Click **Quantize model** and wait for the final result. Local outputs are saved
   beside the source; HF outputs go to Forge's diffusion/text-encoder model folder.
   The result shows the saved path. Refresh Forge's model list before loading it.

Keep the automatic conversion policy and layer protections enabled to start with.
**Advanced options** lets you change the destination, conversion device and
precision rules. For shard assembly without quantization, choose
**Original precision (no quantization)** and click **Save model**.

See the [user guide](docs/USER_GUIDE.md) for formats, optimization settings, HF
downloads, disk requirements, compatibility and troubleshooting.

## Uninstall

1. Finish or cancel any conversion, then close Forge Neo.
2. Delete `extensions/sd-webui-forge-neo-model-converter` and restart Forge.

Converted models and retained HF sources remain available. If no longer needed,
remove retained downloads from `ModelConverterSources` under Forge's configured
models directory. Removing the extension does not require deleting model files,
the global Hugging Face cache or Forge's `ui-config.json`.

If you installed Prodigy only for this extension, you can optionally remove it
using the same Python executable used for installation:

```powershell
& .\venv\Scripts\python.exe -m pip uninstall prodigy-plus-schedule-free
```

Keep it installed if another tool in that environment uses it.

## Credits and license

- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) — host application and integration APIs.
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) — primary quantization library and tensor layouts.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) — quantized-loader contracts and tensor-layout reference data.
- [StarNodes Model Converter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter) — model recipe catalog and conversion references.
- [convert_to_quant](https://github.com/silveroxides/convert_to_quant) — model protection data and learned-rounding research references.
- [comfy-model-tools](https://github.com/Comfy-Org/comfy-model-tools) — reference for general layer selection and ConvRot export.
- [Diffusers](https://github.com/huggingface/diffusers) — architecture and tensor-layout references.
- [Prodigy + Schedule-Free](https://github.com/LoganBooker/prodigy-plus-schedule-free) — optional learned-rounding optimizer.
- [AdaRound](https://arxiv.org/abs/2004.10568) — research on learned rounding; this converter uses its own data-free SVD objective.

Original project code is licensed under [AGPL-3.0](LICENSE). Bundled third-party
data retains its own notices: [StarNodes (MIT)](docs/STARNODES_LICENSE.md),
[CTQ (MIT)](docs/CTQ_NOTICE.md), and [ComfyUI layout data (GPL-3.0)](docs/COMFY_LAYOUT_NOTICE.md).
Additional layout source references are recorded [here](docs/HF_LAYOUT_SOURCES.md).

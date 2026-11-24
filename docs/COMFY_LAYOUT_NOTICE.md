# ComfyUI tensor-layout contract data

`model_converter/layout_maps.json` contains tensor-name, offset and ordering data
extracted from `comfy/utils.py` in Comfy-Org/ComfyUI, commit
`ee34406f56f84f6bc89996b8b77a4bedec98a2b9`:
https://github.com/Comfy-Org/ComfyUI/blob/ee34406f56f84f6bc89996b8b77a4bedec98a2b9/comfy/utils.py

Upstream is licensed under GPL-3.0; its license is retained verbatim in
`COMFY_LAYOUT_LICENSE.md`. This contract data includes the FLUX, SD3 MMDiT,
PixArt, AuraFlow, Z-Image and UNet name maps. The snapshot contains no executable
host code. New standalone assembly, validation, I/O, transport and UI code belongs
to this AGPL-3.0 project. GPL-3.0 material is combined under AGPL-3.0 as permitted
by GPL-3.0 section 13. Preserve this attribution and both license texts.

Additional independently implemented name contracts were audited against the
installed Diffusers 0.39.0 `loaders/single_file_utils.py`, and the installed
ComfyUI/Forge Neo model classes. No Diffusers Python implementation was copied.
These audits establish tensor contracts for specific variants, not blanket
compatibility with every recipe, pipeline, future version or advertised format.

The completion pass's additional contracts, primary source commits and target
versions are documented in `HF_LAYOUT_SOURCES.md`. No additional executable
upstream converter code is bundled by that pass.

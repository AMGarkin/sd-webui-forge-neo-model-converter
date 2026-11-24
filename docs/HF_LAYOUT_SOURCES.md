# Additional HF/Diffusers layout sources

## MiniMax H3 pruned ErosMax adapter (2026-11-24)

Source repository: `ibyteohdear/10Eros-Max-Transformer_8stepTURBO_v1.0_hybrid_VBVR_FUSED`,
commit `9c0c2f88016fd1cc06c94e6e47da4fc225f767c3`. Audited root `config.json`,
all five safetensors headers and `modeling_minimax_h3_pruned.py` as text only.
No remote Python was imported/executed or copied into production. The independently
authored adapter renames the already folded timestep table/biases, fuses Q/K/V
and swaps Diffusers up/gate MLP halves to native gate/up. The `_10eros` ordering
flag describes provenance and does not override stored Diffusers layout. The
owner's actual source/export/reference row comparisons exposed and corrected
the earlier erroneous no-swap interpretation. FP32/F64 buffers retain source
dtypes. Both `adaln_basis` and `adaln_mean` remain, although ordinary ComfyUI does
not consume them. The source's baked LoRA and timestep approximation are retained;
the converter does not merge another LoRA or reconstruct the removed timestep MLP.

Target: ComfyUI `2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884`,
`comfy/model_detection.py` and `comfy/ldm/minimax/model.py` native
`adaln_t_table` path, with `apply_silu=False` and FP32 folded-bias projection.
The source computes a BF16 projection followed by FP32 bias addition; the target
curve projection computes in FP32, so bitwise source-forward parity is not claimed.
Meta constructor/shape agreement, sampled synthetic expressions and quantized
layer forwards are distinct from populated full-model loading/generation.
The owner subsequently confirms ComfyUI video generation with the repaired ErosMax
INT8 file and a fresh conversion after the SwiGLU fix. The fresh run's exact
settings were not supplied; this does not verify all formats or MiniMax variants.
The audited Forge `a057d9ede903eb51c38ceb20e186b6841665d250` has no MiniMax detector.

Separate precision-map reference: LokkenJP/TenEros_Max_optimized_learned at
`3415250db9e2cb89b886f5375c1306fc933e215f`, the ErosMax v1 Compact assembly report
and technical overview. The new profile implements the 208 published assignments
as facts; no upstream converter/learner Python, weights or activation data are
copied. The reference weights remain subject to their upstream MiniMax license.
Compact now uses FP8 group scales and FP32 channel scales, matching the
reference's scale storage precision. Its symmetric codebook-free grids still
differ from the reference learned W4 codebook. This map is not its activation
learner or allocator and does not inherit its generation evidence.

Updated: 2026-11-21. The new assembly/read/transport code is independently
implemented. No upstream converter Python is executed or imported in production.
The separate bundled Comfy contract tables retain their GPL notice and license in
`COMFY_LAYOUT_NOTICE.md` and `COMFY_LAYOUT_LICENSE.md`.

## Executable target and constructor references

Revalidation used installed Forge Neo commit
`5998fb39ea347dfe3a72a8ec88a971e909ed8738` and ComfyUI commit
`860a7fdc7772bcb53f74a675d0592d89f94fa265`, with installed Diffusers **0.39.0**.
No host dependencies were installed or upgraded. Target contracts were inspected
in ComfyUI's `comfy/ldm/{hunyuan_video,genmo,lumina,lightricks,hidream,chroma,cosmos,wan,ernie,ideogram4,krea2}`,
`comfy/model_detection.py`, and the corresponding Forge loader/detector modules.

- ComfyUI target snapshot: https://github.com/Comfy-Org/ComfyUI/tree/860a7fdc7772bcb53f74a675d0592d89f94fa265/comfy/ldm
- Forge target snapshot: https://github.com/Haoming02/sd-webui-forge-classic/tree/5998fb39ea347dfe3a72a8ec88a971e909ed8738
- Diffusers upstream audit `a3608b512ed7248499a44c61d954965ed9bdae4d`:
  https://github.com/huggingface/diffusers/tree/a3608b512ed7248499a44c61d954965ed9bdae4d/scripts
  (`convert_ltx2_to_diffusers.py`, `convert_cosmos_to_diffusers.py`,
  `convert_wan_to_diffusers.py`, including the Wan Animate fixed antialias kernel).
- MiniMax newer conversion audit `f53d552036a0d1bd5570782a39cd40cfabf112bc`:
  https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/scripts/convert_minimax_h3_to_diffusers.py

Diffusers is Apache-2.0; the installed host code retains its own licenses. Reading
these implementations establishes names, permutations and persistent-state
contracts; it is not copying their conversion algorithms into this project.

## Immutable model header references

The read-only audits used bounded headers/configs, not full-size weights. The
commits below identify the inspected sources; they do not establish generation.

| Repository | Audited commit |
| --- | --- |
| ACE-Step/ACE-Step-v1-3.5B | `82cd0d7b6322bd28cd4e830fe675ddb6180ce36c` |
| circlestone-labs/Anima | `6e0b2da2112573380abfb6a1435a60a994b254f2` |
| Boogu/Boogu-Image-0.1-Base | `334ad7e5c6343d33308b306127e047b133e86926` |
| baidu/ERNIE-Image | `5346b31d68c9c23758ba56ef8be5e9dc174c7f99` |
| inclusionAI/Ming-Image-0.1-Design | `c57c49689d12ef7d4e24a170014f1dc43d3ce258` |
| MiniMaxAI/MiniMax-H3 | `0919fb684102ef115126612f3fbb84d1027e7317` |
| MiniMaxAI/MiniMax-H3-Image | `8ced60068623def5b74b4d10558117cb61d6482d` |
| m-a-p/YuE2-3B | `8f24312f187763b9854adeec87e437e2b03bbff1` |
| Qwen/Qwen-Image-2.1 | `d26bb61231c349cf6b7896fa83353113880e1ba3` |
| lodestones/Chroma1-Radiance | `82d788e482ddcd8b0498952f446307ad0c96c953` |
| Comfy-Org/Fibo-2.0 | `4742d4bec410034142c8e65559e5f00c6f29eccb` |
| Comfy-Org/SeedVR2 | `df48879708206a403d2a61acd55578c2e80fd233` |
| Comfy-Org/MiniMax-H3-Image | `0bb149f51e3156ed4b90dd45d5620346bfe1df8f` |

Four 32-byte payload samples from original MiniMax H3 Image and its canonical
checkpoint confirmed the head-interleaved QKV row permutation. They do not
establish whole-file equality.
Image variants require their original PDD sigma schedule in `pdd_config.json`;
the converter validates and embeds it as target transformer config metadata.

Original Fibo, FLUX.3, Ideogram and Krea weight-header requests in this audit
returned HTTP 403. This is an access/evidence limit, not a successful conversion.
Fibo's public canonical header and Ideogram/Krea real constructor contracts were
checked separately; FLUX.3 alternate-key conversion remains unverified.

No model weights or repository Python from these publishers are redistributed.
Names/configuration facts and measured contract evidence do not grant model-use
rights or establish visual quality, acceleration or full-pipeline compatibility.

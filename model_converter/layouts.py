"""Lazy canonical tensor recipes. Architecture is never inferred from sharding.

Tensor-name contract data in layout_maps.json is attributed in
docs/COMFY_LAYOUT_NOTICE.md. No host modules are imported at runtime.
"""

import json
import math
from pathlib import Path
import re

from .io import SIZES, TensorReader, inspect
from .sources import component_config

SUPPORTED_CLASSES = {"MOSSImageTransformer", "FluxTransformer2DModel", "Flux2Transformer2DModel", "Flux2KleinTransformer2DModel",
                     "ZImageTransformer2DModel", "PixArtTransformer2DModel", "SD3Transformer2DModel", "AuraFlowTransformer2DModel",
                     "WanTransformer3DModel", "LTXVideoTransformer3DModel", "UNet2DConditionModel", "QwenImageTransformer2DModel",
                     "MiniMaxH3PrunedTransformer3DModel"}
LAYOUT_GAPS = {"SanaTransformer2DModel": "the inspected Forge Neo and ComfyUI commits have no Sana model implementation; remapping cannot supply an absent host architecture"}


def validate_config(config):
    cls = config.get("_class_name", "")
    if cls in LAYOUT_GAPS:
        raise ValueError(f"Known component {cls} has no audited mapping: {LAYOUT_GAPS[cls]}. Use a canonical target checkpoint; this Diffusers variant cannot yet be exported as compatible.")
    if cls.startswith("Autoencoder"):
        raise ValueError(f"{cls}: VAE component mapping/encode-decode and convolution quantization are not implemented")


def recipe(parts):
    """Parts are (source name, source spec, first row, exclusive last row)."""
    dtype = parts[0][1]["dtype"]
    tail = parts[0][1]["shape"][1:]
    for name, spec, start, stop in parts:
        if spec["dtype"] != dtype or spec["shape"][1:] != tail or not 0 <= start < stop <= spec["shape"][0]:
            raise ValueError(f"Incompatible mapping shapes/dtypes: {name}")
    shape = [sum(stop - start for _, _, start, stop in parts), *tail]
    return {"dtype": dtype, "shape": shape, "data_offsets": [0, math.prod(shape) * SIZES[dtype]],
            "file": parts[0][1]["file"], "parts": [
                {"name": name, "spec": spec, "start": start, "stop": stop} for name, spec, start, stop in parts]}


def source_part(name, tensors, start=0, stop=None):
    if name not in tensors:
        raise ValueError(f"Incomplete layout mapping: required tensor {name}")
    spec = tensors[name]
    if not spec["shape"]:
        raise ValueError(f"Cannot assemble a scalar tensor: {name}")
    return spec.get("source_name", name), spec, start, spec["shape"][0] if stop is None else stop


class CanonicalReader:
    """Materialize at most one logical tensor (or requested row tile)."""

    def __init__(self, entry):
        self.entry = entry

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_tensor(self, name, start=None, stop=None):
        import torch
        e = self.entry
        if "constant" in e:
            value = torch.tensor(e["constant"], dtype=torch.float32).reshape(e["shape"])
            return value if start is None else value[start:stop]
        if "reshape" in e:
            base = dict(e)
            del base["reshape"]
            base["shape"] = [sum(p["stop"] - p["start"] for p in e["parts"]), *e["parts"][0]["spec"]["shape"][1:]]
            with CanonicalReader(base) as reader:
                value = reader.get_tensor(name).reshape(e["reshape"])
            return value if start is None else value[start:stop]
        if "parts" not in e:
            source_name = e.get("source_name", name)
            with TensorReader(e["file"], {source_name: e}) as reader:
                return reader.get_tensor(source_name, start, stop)
        first, last = (0, e["shape"][0]) if start is None else (start, stop)
        if last is None or not 0 <= first <= last <= e["shape"][0]:
            raise ValueError("Invalid canonical row slice")
        output = None
        cursor = 0
        for part in e["parts"]:
            length = part["stop"] - part["start"]
            lo, hi = max(first, cursor), min(last, cursor + length)
            if lo < hi:
                with TensorReader(part["spec"]["file"], {part["name"]: part["spec"]}) as reader:
                    value = reader.get_tensor(part["name"], part["start"] + lo - cursor, part["start"] + hi - cursor)
                if output is None:
                    output = torch.empty([last - first, *e["shape"][1:]], dtype=value.dtype)
                output[lo - first:hi - first].copy_(value)
                del value
            cursor += length
        if output is None:
            raise ValueError("Empty canonical tensor request")
        return output


def input_files(entry):
    if "constant" in entry:
        return set()
    return {part["spec"]["file"] for part in entry["parts"]} if "parts" in entry else {entry["file"]}


def _blocks(tensors, prefix):
    indices = sorted({int(m[1]) for k in tensors if (m := re.match(re.escape(prefix) + r"\.(\d+)\.", k))})
    if indices and indices != list(range(indices[-1] + 1)):
        raise ValueError(f"Non-contiguous architecture blocks: {prefix}")
    return indices


def _check_config(tensors, config, stacks, width_key=None):
    for stack, field in stacks:
        if field in config:
            count = config[field]
            if type(count) is not int or count < 0 or len(_blocks(tensors, stack)) != count:
                raise ValueError(f"Component config {field} disagrees with the complete {stack} stack")
    if width_key and "num_attention_heads" in config and "attention_head_dim" in config:
        width = config["num_attention_heads"] * config["attention_head_dim"]
        if tensors.get(width_key, {}).get("shape", [None])[0] != width:
            raise ValueError(f"Component config hidden width disagrees with {width_key}")


def _contract_map(family, tensors):
    raw = json.loads(Path(__file__).with_name("layout_maps.json").read_text())["maps"][family + "_to_diffusers"]
    blocks = {"flux": ("transformer_blocks", "single_transformer_blocks"),
              "z_image": ("layers", "noise_refiner", "context_refiner"),
              "pixart": ("transformer_blocks",), "mmdit": ("transformer_blocks",),
              "auraflow": ("joint_transformer_blocks", "single_transformer_blocks")}[family]
    width_key = {"flux": "x_embedder.weight", "z_image": "layers.0.attention.to_q.weight",
                 "pixart": "transformer_blocks.0.attn1.to_q.weight", "mmdit": "pos_embed.proj.weight",
                 "auraflow": "pos_embed.proj.weight"}[family]
    width = tensors.get(width_key, {}).get("shape", [0])[0]
    if width <= 0:
        raise ValueError(f"{family}: missing hidden-width anchor {width_key}")
    result = {}
    for source, target in raw.items():
        stack = next((p for p in blocks if source.startswith(p + ".0.")), None)
        for i in _blocks(tensors, stack) if stack else (None,):
            src = source.replace(stack + ".0.", stack + f".{i}.", 1) if stack else source
            dst = target if isinstance(target, str) else target[0]
            if stack:
                dst = re.sub(r"\.0\.", f".{i}.", dst, count=1)
            offset = None if isinstance(target, str) else target[1]
            if offset is not None:
                unit = 64 if family == "mmdit" else 1
                offset = [offset[0], offset[1] // unit * width, offset[2] // unit * width]
            swap = not isinstance(target, str) and len(target) > 2
            # SD3 swaps the last context modulation only; the one-block template
            # includes that final block. Earlier blocks retain their original order.
            if family == "mmdit" and "norm1_context.linear" in src and i != len(_blocks(tensors, stack)) - 1:
                swap = False
            result[src] = (dst, offset, swap)
    return result


def _mapped(tensors, mapping):
    destinations = {}
    consumed = set()
    for name, (target, offset, swap) in mapping.items():
        if name not in tensors:
            continue
        spec = tensors[name]
        if offset is not None:
            if offset[0] != 0 or spec["shape"][0] != offset[2]:
                raise ValueError(f"Mapping width mismatch: {name}")
            position = offset[1]
        else:
            position = 0
        parts = [source_part(name, tensors)]
        if swap:
            rows = spec["shape"][0]
            if rows % 2:
                raise ValueError(f"Scale/shift mapping requires an even row count: {name}")
            parts = [source_part(name, tensors, rows // 2, rows), source_part(name, tensors, 0, rows // 2)]
        destinations.setdefault(target, []).append((position, parts))
        consumed.add(name)
    unknown = set(tensors) - consumed
    if unknown:
        raise ValueError("Known Diffusers layout has unmapped tensors; adapter cannot establish compatibility: " + ", ".join(sorted(unknown)[:8]))
    output = {}
    for target, chunks in destinations.items():
        cursor, parts = 0, []
        for position, values in sorted(chunks, key=lambda x: x[0]):
            if position != cursor:
                raise ValueError(f"Missing/overlapping fused tensor pieces: {target}")
            parts.extend(values)
            cursor += sum(p[3] - p[2] for p in values)
        # Q/K/V fusion must never succeed with a missing final piece.
        assembled = any(src in consumed and v[0] == target and v[1] is not None for src, v in mapping.items())
        expected = [v for v in mapping.values() if v[0] == target and v[1] is not None] if assembled else []
        if expected and cursor != max(v[1][1] + v[1][2] for v in expected):
            raise ValueError(f"Incomplete fused projection: {target}")
        output[target] = recipe(parts)
    return output


def _unet_map(tensors, config):
    data = json.loads(Path(__file__).with_name("layout_maps.json").read_text())["unet"]
    basic = {src: dst for dst, src in data["UNET_MAP_BASIC"]}
    resnet = {src: dst for dst, src in data["UNET_MAP_RESNET"].items()}
    blocks = _blocks(tensors, "down_blocks")
    if not blocks or not _blocks(tensors, "up_blocks"):
        raise ValueError("UNet component needs complete down/up block stacks")
    down, up = {}, {}
    n = 1
    for b in blocks:
        indices = _blocks(tensors, f"down_blocks.{b}.resnets")
        for i in indices:
            down[b, i] = n
            n += 1
        if any(k.startswith(f"down_blocks.{b}.downsamplers.") for k in tensors):
            down[b, "sample"] = n
            n += 1
    n = 0
    for b in _blocks(tensors, "up_blocks"):
        for i in _blocks(tensors, f"up_blocks.{b}.resnets"):
            up[b, i] = n
            n += 1
    mapping = {}
    for name in tensors:
        target = basic.get(name)
        m = re.match(r"(down|up)_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.(.+)", name)
        if m:
            side, b, kind, i, tail = m.groups()
            b, i = int(b), int(i)
            table = down if side == "down" else up
            if (b, i) not in table:
                raise ValueError(f"UNet attention/resnet block mismatch: {name}")
            root = "input_blocks" if side == "down" else "output_blocks"
            target = (f"{root}.{table[b, i]}." + ("0." + resnet[tail] if kind == "resnets" else "1." + tail)
                      if kind == "attentions" or tail in resnet else None)
        m = re.match(r"(down|up)_blocks\.(\d+)\.(?:down|up)samplers\.0\.conv\.(weight|bias)$", name)
        if m:
            side, b, param = m.groups()
            b = int(b)
            if side == "down":
                target = f"input_blocks.{down[b, 'sample']}.0.op.{param}"
            else:
                last = max(i for bb, i in up if bb == b)
                slot = 2 if any(k.startswith(f"up_blocks.{b}.attentions.{last}.") for k in tensors) else 1
                target = f"output_blocks.{up[b, last]}.{slot}.conv.{param}"
        m = re.match(r"mid_block.resnets.([01])\.(.+)", name)
        if m and m[2] in resnet:
            target = f"middle_block.{int(m[1]) * 2}.{resnet[m[2]]}"
        if name.startswith("mid_block.attentions.0."):
            target = "middle_block.1." + name.removeprefix("mid_block.attentions.0.")
        if target:
            mapping[name] = (target, None, False)
    return mapping


def _wan_map(tensors):
    mapping = {}
    roots = {"condition_embedder.time_embedder.linear_1": "time_embedding.0",
             "condition_embedder.time_embedder.linear_2": "time_embedding.2",
             "condition_embedder.text_embedder.linear_1": "text_embedding.0",
             "condition_embedder.text_embedder.linear_2": "text_embedding.2",
             "condition_embedder.time_proj": "time_projection.1",
             "condition_embedder.image_embedder.norm1": "img_emb.proj.0",
             "condition_embedder.image_embedder.ff.net.0.proj": "img_emb.proj.1",
             "condition_embedder.image_embedder.ff.net.2": "img_emb.proj.3",
             "condition_embedder.image_embedder.norm2": "img_emb.proj.4",
             "proj_out": "head.head", "scale_shift_table": "head.modulation"}
    for name in tensors:
        target = None
        for src, dst in roots.items():
            if name == src or name.startswith(src + "."):
                target = dst + name[len(src):]
                break
        if name.startswith("patch_embedding."):
            target = name
        if re.match(r"blocks\.\d+\.", name):
            target = name
            for src, dst in ((".attn1.", ".self_attn."), (".attn2.", ".cross_attn."),
                             (".to_out.0.", ".o."), (".to_q.", ".q."), (".to_k.", ".k."), (".to_v.", ".v."),
                             (".add_k_proj.", ".k_img."), (".add_v_proj.", ".v_img."), (".norm_added_k.", ".norm_k_img."),
                             (".ffn.net.0.proj.", ".ffn.0."), (".ffn.net.2.", ".ffn.2."),
                             (".scale_shift_table", ".modulation")):
                target = target.replace(src, dst)
            # Only the block's cross-attention and MLP norms are interchanged.
            target = re.sub(r"^(blocks\.\d+\.)norm([23])\.", lambda m: m[1] + "norm" + str(5 - int(m[2])) + ".", target)
            if not re.match(r"blocks\.\d+\.(self_attn|cross_attn|ffn|norm[123]|modulation)(\.|$)", target):
                target = None
        if target:
            mapping[name] = (target, None, False)
    return mapping


def canonicalize(tensors, config):
    original_tensors = tensors
    for prefix in ("model.diffusion_model.", "diffusion_model.", "net."):
        if tensors and all(n.startswith(prefix) for n in tensors):
            tensors = {n.removeprefix(prefix): {**s, "source_name": s.get("source_name", n)} for n,s in tensors.items()}
    cls = config.get("_class_name", "")
    if any(re.search(r"\.(?:lora_[AB]|lora_down|lora_up)\.", name, re.IGNORECASE) for name in tensors):
        raise ValueError("LoRA/adapter tensors are not standalone model components; merging adapters is outside this converter's scope")
    family = None
    from . import extended_layouts
    if config.get("model_type") == "yue2" and "model.layers.0.nar_self_attn.q_proj.weight" in tensors:
        return extended_layouts.yue2(tensors,config), {"adapter":"yue2-ar-nar", "status":"canonical diffusion/text bundle; separate component loader validation required"}
    if cls == "MiniMaxH3DiTModel" and "blocks.0.attn.qkv_proj.weight" in tensors:
        return extended_layouts.minimax_raw(tensors,config), {"adapter":"minimax-raw-qkv", "status":"original per-head interleave canonicalized; PDD schedule metadata is required for image variants"}
    native_signatures = {
        "ace-step": ("genre_embedder.weight", "lyric_embs.weight", "final_layer.linear.weight"),
        "boogu": ("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight", "x_embedder.weight", "norm_out.linear_2.weight"),
        "ernie-image": ("adaLN_modulation.1.weight", "layers.0.mlp.linear_fc2.weight", "layers.0.self_attention.to_q.weight"),
        "qwen-image-2.1": ("txt_in.text_norm.weight", "modulation.1.weight", "transformer_blocks.0.img_mlp.gate_layer.weight"),
        "fibo2": ("perceiver.latent_queries", "layers.0.attention.qkv.weight", "final_layer.linear.weight"),
        "seedvr2": ("negative_conditioning", "positive_conditioning", "blocks.0.ada.vid.attn_gate"),
        "flux3": ("emb_in.image.weight", "early_stream_modulations.image.lin.weight", "single_blocks.0.mlp_in.weight"),
        "minimax-native": ("video_patch_proj.weight", "condition_proj.weight", "rope.inv_freq"),
        "krea2": ("first.weight", "txtfusion.projector.weight", "blocks.0.attn.wq.weight"),
        "ltx2": ("patchify_proj.weight", "audio_patchify_proj.weight", "transformer_blocks.0.attn1.to_q.weight"),
    }
    for adapter, anchors in native_signatures.items():
        if all(k in tensors for k in anchors):
            return tensors, {"adapter":adapter+"-identity", "status":"complete original HF header/target namespace audited; generation is separate"}
    if all(k in tensors for k in ("distilled_guidance_layer.in_proj.weight", "double_blocks.0.img_attn.qkv.weight")):
        output={}
        for name,spec in tensors.items():
            target=re.sub(r"(\.norm\.(?:query_norm|key_norm)|distilled_guidance_layer\.norms\.\d+|nerf_blocks\.\d+\.norm|nerf_final_layer_conv\.norm)\.scale$",r"\1.weight",name)
            output[target]={**spec,"source_name":spec.get("source_name",name)} if target!=name else spec
        return output,{"adapter":"chroma-native", "status":"native Chroma/Radiance namespace with current target norm aliases; generation is separate"}
    if any(re.fullmatch(r"layers\.\d+\.proj\.q\.weight", k) for k in tensors) and (
            cls == "MOSSImageTransformer" or "rope.bands" in tensors):
        _check_config(tensors, config, (("layers", "num_layers"),))
        output = dict(tensors)
        for i in _blocks(tensors, "layers"):
            keys = [f"layers.{i}.proj.{p}.weight" for p in ("q", "k", "v")]
            fused = f"layers.{i}.proj.qkv.weight"
            if fused in output:
                raise ValueError("Mixed fused/unfused MOSS projections")
            output[fused] = recipe([source_part(k, tensors) for k in keys])
            shape = output[fused]["shape"]
            if len(shape) != 2 or shape[0] != shape[1] * 3 or ("dim" in config and shape[1] != config["dim"]):
                raise ValueError(f"MOSS QKV dimensions disagree with hidden width: {fused}")
            for k in keys:
                del output[k]
        return output, {"adapter": "moss-qkv", "status": "canonical mapping; full-model validation is separate"}
    if all(k in tensors for k in ("perceiver.latent_queries", "layers.0.attention.to_q.weight",
                                   "layers.0.feed_forward.w1.weight", "final_layer.linear.weight")):
        _check_config(tensors, config, (("layers", "n_layers"), ("noise_refiner", "n_refiner_layers"),
                                        ("perceiver.layers", "perceiver_num_layers")))
        output = _mapped(tensors, extended_layouts.fibo2(tensors))
        return output, {"adapter": "fibo2-split-mlp",
                        "status": "canonical attention layout with separate source MLP matrices; full-model validation is separate"}
    # Canonical keys take precedence over config declarations and storage grouping.
    from .profiles import detect_native_profile, qwen3vl_identity
    if detect_native_profile(tensors) or qwen3vl_identity(tensors):
        return tensors, {"adapter": "canonical", "status": "recognized canonical key layout"}
    extended = {"HunyuanVideoTransformer3DModel": ("hunyuan-video", extended_layouts.hunyuan),
                "HunyuanVideo15Transformer3DModel": ("hunyuan-video-1.5", extended_layouts.hunyuan),
                "MochiTransformer3DModel": ("mochi", extended_layouts.mochi),
                "Lumina2Transformer2DModel": ("lumina2", extended_layouts.lumina),
                "CosmosTransformer3DModel": ("cosmos", extended_layouts.cosmos),
                "LTX2VideoTransformer3DModel": ("ltx2", extended_layouts.ltx2)}
    extended["Krea2Transformer2DModel"] = ("krea2", extended_layouts.krea2)
    extended["MiniMaxH3Transformer3DModel"] = ("minimax-h3", extended_layouts.minimax)
    extended["MiniMaxH3PrunedTransformer3DModel"] = ("minimax-h3-pruned", extended_layouts.minimax_pruned)
    extended["Ideogram4Transformer2DModel"] = ("ideogram4", extended_layouts.ideogram)
    if cls in extended:
        adapter, builder = extended[cls]
        stack = "layers" if adapter in ("lumina2", "ideogram4") else "transformer_blocks"
        _check_config(tensors, config, ((stack, "num_layers"),))
        output = _mapped(tensors, builder(tensors, config))
        if adapter in ("minimax-h3", "minimax-h3-pruned"):
            output = extended_layouts.finish_minimax(output,tensors,config)
        if adapter == "krea2":
            for name,spec in list(output.items()):
                if re.fullmatch(r"blocks\.\d+\.mod.lin",name):
                    output[name]={**spec,"reshape":[math.prod(spec["shape"])],"shape":[math.prod(spec["shape"])]}
        layout = {"adapter": adapter, "status": "canonical mapping; full-model validation is separate"}
        if adapter == "minimax-h3-pruned":
            layout["warnings"] = ["Pruned MiniMax uses ComfyUI's native adaln_t_table curve loader. The audited Forge installation has no MiniMax model detector; layer-format support does not supply the architecture."]
        return output, layout
    if cls == "HiDreamImageTransformer2DModel" and all(k in tensors for k in (
            "x_embedder.proj.weight", "double_stream_blocks.0.block.adaLN_modulation.1.weight", "final_layer.linear.weight")):
        _check_config(tensors, config, (("double_stream_blocks", "num_layers"), ("single_stream_blocks", "num_single_layers")))
        return tensors, {"adapter": "hidream-identity", "status": "native routed-expert/caption namespace; full-model validation is separate"}
    if cls == "QwenImageTransformer2DModel" and all(k in tensors for k in (
            "img_in.weight", "txt_in.weight", "transformer_blocks.0.attn.to_q.weight", "norm_out.linear.weight")):
        return tensors, {"adapter": "qwen-image-identity", "status": "audited native Qwen-Image tensor namespace; full-model validation is separate"}
    if cls == "UNet2DConditionModel" or ("conv_in.weight" in tensors and "down_blocks.0.resnets.0.conv1.weight" in tensors):
        return _mapped(tensors, _unet_map(tensors, config)), {"adapter": "unet", "status": "canonical UNet mapping; full-model validation is separate"}
    if cls in ("WanTransformer3DModel", "WanVACETransformer3DModel", "WanAnimateTransformer3DModel") and "blocks.0.attn1.to_q.weight" in tensors:
        _check_config(tensors, config, (("blocks", "num_layers"),), "blocks.0.attn1.to_q.weight")
        mapping = _wan_map(tensors)
        if cls == "WanVACETransformer3DModel":
            subset = {k.replace("vace_blocks.", "blocks.", 1): v for k, v in tensors.items() if k.startswith("vace_blocks.")}
            for name, (target, offset, swap) in _wan_map(subset).items():
                mapping[name.replace("blocks.", "vace_blocks.", 1)] = (target.replace("blocks.", "vace_blocks.", 1), offset, swap)
            for name in tensors:
                if name.startswith("vace_patch_embedding."):
                    mapping[name] = (name, None, False)
                if re.match(r"vace_blocks\.\d+\.proj_(in|out)\.", name):
                    mapping[name] = (name.replace(".proj_in.", ".before_proj.").replace(".proj_out.", ".after_proj."), None, False)
        if cls == "WanAnimateTransformer3DModel":
            output = _mapped(tensors, extended_layouts.wan_animate(tensors, config))
            return extended_layouts.finish_wan_animate(output, tensors, config), {"adapter":"wan-animate", "status":"canonical mapping; full-model validation is separate", "warnings":["Restored fixed [1,3,3,1] antialias kernels omitted from the Diffusers persistent state; no learned weights were invented."]}
        return _mapped(tensors, mapping), {"adapter": "wan-vace" if cls == "WanVACETransformer3DModel" else "wan", "status": "canonical mapping; full-model validation is separate"}
    if cls == "LTXVideoTransformer3DModel" and "proj_in.weight" in tensors:
        _check_config(tensors, config, (("transformer_blocks", "num_layers"),), "proj_in.weight")
        mapping = {}
        for name in tensors:
            target = re.sub(r"^proj_in\.", "patchify_proj.", name)
            target = re.sub(r"^time_embed\.", "adaln_single.", target)
            target = target.replace(".norm_q.", ".q_norm.").replace(".norm_k.", ".k_norm.")
            mapping[name] = (target, None, False)
        return _mapped(tensors, mapping), {"adapter": "ltx-video", "status": "LTX Video 0.9 transformer mapping; LTX-2 requires a different adapter"}
    if "transformer_blocks.0.attn.to_q.weight" in tensors and "x_embedder.weight" in tensors:
        family = "flux"
    elif "noise_refiner.0.attention.to_q.weight" in tensors:
        family = "z_image"
    elif "adaln_single.emb.timestep_embedder.linear_1.weight" in tensors:
        family = "pixart"
    elif "joint_transformer_blocks.0.attn.add_q_proj.weight" in tensors:
        family = "auraflow"
    elif "transformer_blocks.0.attn.add_q_proj.weight" in tensors and "pos_embed.proj.weight" in tensors:
        family = "mmdit"
    if family:
        stacks = {"flux": (("transformer_blocks", "num_layers"), ("single_transformer_blocks", "num_single_layers")),
                  "z_image": (("layers", "n_layers"),), "pixart": (("transformer_blocks", "num_layers"),),
                  "mmdit": (("transformer_blocks", "num_layers"),),
                  "auraflow": (("joint_transformer_blocks", "num_mmdit_layers"), ("single_transformer_blocks", "num_single_dit_layers"))}[family]
        _check_config(tensors, config, stacks, "x_embedder.weight" if family == "flux" else None)
        mapping = _contract_map(family, tensors)
        if family == "flux":
            # FLUX.2 global modulations and SwiGLU have the same target contract.
            for name in tensors:
                if name.startswith(("double_stream_modulation_", "single_stream_modulation.")):
                    mapping[name] = (name.replace(".linear.", ".lin."), None, False)
                if name.startswith("time_guidance_embed."):
                    src = name.replace("time_guidance_embed.", "time_text_embed.")
                    if src in mapping:
                        mapping[name] = mapping[src]
            if cls == "ChromaTransformer2DModel":
                for name in tensors:
                    if name.startswith("distilled_guidance_layer."):
                        target = name.replace(".linear_1.", ".in_layer.").replace(".linear_2.", ".out_layer.")
                        mapping[name] = (target, None, False)
        info = {"adapter": family, "status": "canonical mapping; full-model validation is separate"}
        if family == "pixart" and "caption_projection.y_embedding" not in tensors:
            info["warnings"] = ["PixArt Diffusers omits y_embedder.y_embedding, an unused training/dropout buffer in the inspected target's eval mode. No replacement values are invented; strict full-network loading reports that missing buffer."]
        output = _mapped(tensors, mapping)
        if family == "z_image" and cls == "DiffusionTransformer" and config.get("dim") and "all_patch_size" in config:
            output["__ming_image__"] = {"shape":[0],"dtype":"F32","data_offsets":[0,0],"constant":[],"file":next(iter(tensors.values()))["file"]}
            info["adapter"] = "ming-image"
        return output, info
    # Transformers text layouts already use the native Qwen/T5/CLIP namespace.
    text_classes = ("Qwen", "T5", "UMT5", "CLIP", "Llama", "Gemma", "Mistral")
    if cls.startswith(text_classes) or any(str(a).startswith(text_classes) for a in config.get("architectures", [])):
        return tensors, {"adapter": "transformers-identity", "status": "native text keys; component quantized loader support must be verified"}
    # Other canonical namespaces still require concrete layout signatures.
    native_pairs = (("blocks.0.self_attn.q.weight", "patch_embedding.weight"),
                    ("patchify_proj.weight", "transformer_blocks.0.attn1.to_q.weight"),
                    ("input_blocks.0.0.weight", "out.2.weight"),
                    ("double_blocks.0.img_attn_qkv.weight", "img_in.proj.weight"))
    if any(all(k in tensors for k in pair) for pair in native_pairs):
        return tensors, {"adapter": "canonical-family", "status": "canonical family signature; full-model loading is unverified"}
    validate_config(config)
    if cls in SUPPORTED_CLASSES:
        raise ValueError(f"Known component {cls} has no audited mapping for these keys. Check its config, complete component and architecture variant.")
    return original_tensors, {"adapter": "universal-identity", "status": "EXPERIMENTAL / UNVERIFIED: unknown keys retained; no host compatibility established"}


def inspect_canonical(source, *, detect=False):
    tensors, metadata = inspect(source, with_metadata=True)
    config = component_config(source)
    tensors, layout = canonicalize(tensors, config)
    if config.get("_class_name") == "LTX2VideoTransformer3DModel" and not any(n.startswith("video_embeddings_connector.") for n in tensors):
        from .sources import ltx_connector_source
        from .extended_layouts import ltx_connectors
        handle = ltx_connector_source(source)
        extra = inspect(handle)
        extra = _mapped(extra, ltx_connectors(extra, component_config(handle)))
        if set(extra) & set(tensors):
            raise ValueError("LTX-2 connectors overlap transformer tensors")
        tensors = {**tensors, **extra}
        layout["dependencies"] = [str(handle)]
        layout["warnings"] = ["LTX-2 connector weights are included. text_embedding_projection is the companion text-encoder projection, retained in its canonical namespace; load it with the matching Gemma text component. Forge has no installed LTX audio/video implementation."]
    target_config = {}
    if config.get("_class_name") == "LTX2VideoTransformer3DModel":
        for key in ("in_channels", "cross_attention_dim", "attention_head_dim", "num_attention_heads", "caption_channels", "num_layers", "audio_in_channels", "audio_cross_attention_dim", "audio_attention_head_dim", "audio_num_attention_heads", "vae_scale_factors", "timestep_scale_multiplier", "rope_type"):
            if key in config:
                target_config[key] = config[key]
        target_config.update(positional_embedding_theta=config.get("rope_theta",10000.0),
            frequencies_precision="float64" if config.get("rope_double_precision",True) else "float32",
            apply_gated_attention=config.get("gated_attn",False), cross_attention_adaln=config.get("cross_attn_mod",False),
            use_prompt_adaln_single=config.get("use_prompt_embeddings",True))
        if "handle" in locals():
            cc = component_config(handle)
            for side in ("video", "audio"):
                prefix = "" if side == "video" else "audio_"
                for src,dst in (("num_attention_heads","num_attention_heads"),("attention_head_dim","attention_head_dim")):
                    if f"{side}_connector_{src}" in cc:
                        target_config[prefix+"connector_"+dst]=cc[f"{side}_connector_{src}"]
            target_config["connector_num_layers"]=cc.get("video_connector_num_layers",2)
            if cc.get("video_connector_num_layers",2)!=cc.get("audio_connector_num_layers",2):
                raise ValueError("Installed LTX target requires equal video/audio connector depths")
            target_config["connector_apply_gated_attention"]=cc.get("video_gated_attn",False)
    if config.get("_class_name") == "MiniMaxH3DiTModel" and config.get("audio_enabled") is False:
        from .sources import resolve_local
        local, _ = resolve_local(source)
        paths = [p for p in (local.parent / "pdd_config.json", local.parent.parent / "pdd_config.json") if p.is_file()]
        if not paths:
            raise ValueError("MiniMax H3 Image requires its pdd_config.json schedule; retain it beside the component or at the repository root")
        if paths[0].stat().st_size > 1024**2:
            raise ValueError("PDD config exceeds metadata size limit")
        pdd=json.loads(paths[0].read_text(encoding="utf-8"))
        sigmas=pdd.get("sigmas")
        if not isinstance(sigmas,list) or len(sigmas)<2 or not all(isinstance(s,(int,float)) and math.isfinite(s) and 0<=s<=1 for s in sigmas) or not all(a>b for a,b in zip(sigmas,sigmas[1:])):
            raise ValueError("MiniMax PDD sigmas must be a finite decreasing schedule")
        target_config.update(pdd_sigmas=sigmas,pdd_steps=config.get("pdd_steps",1))
    if target_config:
        existing=json.loads(metadata.get("config","{}"))
        if not isinstance(existing,dict) or not isinstance(existing.get("transformer",{}),dict):
            raise ValueError("Invalid existing target config metadata")
        transformer=existing.get("transformer",{})
        if any(k in transformer and transformer[k]!=v for k,v in target_config.items()):
            raise ValueError("Source/target config metadata conflict")
        metadata={**metadata,"config":json.dumps({**existing,"transformer":{**transformer,**target_config}})}
        layout["embedded_target_config"]=target_config
    if detect:
        from .profiles import detect_profile
        layout["detected_profile"] = detect_profile(tensors, config=config)
    return tensors, metadata, layout

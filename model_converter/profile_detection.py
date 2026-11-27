"""Recognize reference policies from component tensor namespaces."""


# These signatures identify policy families. Canonicalization and loader
# validation have their own contracts.
DIFFUSION_SIGNATURES = {
    "star:ERNIE-Image": (
        ("adaLN_modulation.1.weight", "layers.0.mlp.linear_fc2.weight", "layers.0.self_attention.to_q.weight"),
    ),
    "star:Flux 3 Klein": (
        ("emb_in.image.weight", "early_stream_modulations.image.lin.weight", "single_blocks.0.mlp_in.weight"),
        ("early_stream_modulations.image.linear.weight", "mode_blocks.image.0.ff.linear_in.weight", "context_embedder.weight"),
    ),
    "star:Krea-2": (
        ("first.weight", "txtfusion.projector.weight", "blocks.0.attn.wq.weight"),
        ("img_in.weight", "text_fusion.projector.weight", "time_mod_proj.weight"),
    ),
    "star:Wan (All Versions)": (
        ("patch_embedding.weight", "blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight"),
    ),
    "star:LTX-2.5": (
        ("keyframes_abs_pos_embedding", "audio_patchify_proj.weight", "transformer_blocks.0.attn1.to_q.weight"),
        ("keyframes_abs_pos_embedding", "audio_proj_in.weight", "transformer_blocks.0.attn1.to_q.weight"),
    ),
    "star:LTX-Video (All Versions)": (
        ("patchify_proj.weight", "transformer_blocks.0.attn1.to_q.weight", "scale_shift_table"),
    ),
    "star:minimax_h3_image": (
        ("pdd_proj_out.0.weight", "transformer_blocks.0.adaln_proj.linear.weight", "context_embedder.weight"),
    ),
    "star:Qwen-Image-2.1": (
        ("txt_in.text_norm.weight", "modulation.1.weight", "transformer_blocks.0.img_mlp.gate_layer.weight"),
    ),
    "star:Qwen-Image": (
        ("img_in.weight", "txt_in.weight", "transformer_blocks.0.attn.add_q_proj.weight",
         "transformer_blocks.0.img_mlp.net.0.proj.weight"),
    ),
    "star:SDXL (Not NVFP4)": (
        ("label_emb.0.0.weight", "input_blocks.0.0.weight", "middle_block.1.transformer_blocks.0.attn2.to_k.weight"),
    ),
    "star:Fibo-2.0-edit-turbo": (
        ("perceiver.latent_queries", "layers.0.attention.qkv.weight", "final_layer.linear.weight"),
        ("perceiver.latent_queries", "layers.0.attention.to_q.weight", "final_layer.linear.weight"),
    ),
    "star:Lens": (
        ("transformer_blocks.0.attn.img_qkv.weight", "transformer_blocks.0.attn.txt_qkv.weight",
         "transformer_blocks.0.txt_mlp.w3.weight"),
    ),
    "star:Boogu-Image": (
        ("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight", "x_embedder.weight",
         "norm_out.linear_2.weight"),
    ),
    "star:ACE-Step": (
        ("genre_embedder.weight", "lyric_embs.weight", "final_layer.linear.weight"),
    ),
    "star:SeedVR (All Versions)": (
        ("negative_conditioning", "positive_conditioning", "blocks.0.ada.vid.attn_gate"),
    ),
}


def detect_diffusion_profile(tensors):
    keys = {key.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.").removeprefix("net.")
            for key in tensors}
    if {"model.blocks.0.self_attn.q.weight", "model.patch_embedding.weight"} <= keys:
        # Causal Wan stores the native transformer under model.*.
        keys = {key.removeprefix("model.") for key in keys}
    if {"video_patch_proj.weight", "condition_proj.weight", "blocks.0.attn.qkv_proj.weight"} <= keys:
        return ("star:minimax_h3" if "audio_patch_proj.weight" in keys or "adaln_t_table" in keys
                else "star:minimax_h3_image")
    matches = {profile for profile, signatures in DIFFUSION_SIGNATURES.items()
               if any(keys.issuperset(signature) for signature in signatures)}
    if "star:LTX-2.5" in matches:
        matches.discard("star:LTX-Video (All Versions)")
    if len(matches) > 1:
        raise ValueError("Multiple model profiles match. Select a profile explicitly.")
    return next(iter(matches), None)


def detect_text_profile(keys, config):
    if {"language_model.model.layers.0.self_attn.q_proj.weight", "vision_tower.patch_conv.weight",
        "multi_modal_projector.patch_merger.merging_layer.weight"} <= keys:
        return "ctq:mistral"
    if {"text_embedding_projection.video_aggregate_embed.weight", "audio_projector.embedding_projection.weight",
        "model.layers.0.post_feedforward_layernorm.weight"} <= keys:
        return "star:ltx25_gemma4_te"
    for language, visual in (("model.language_model.", "model.visual."), ("language_model.", "visual."),
                             ("model.", "visual.")):
        if {language + "layers.0.linear_attn.in_proj_qkv.weight", language + "embed_tokens.weight",
            visual + "merger.linear_fc2.weight"} <= keys:
            return "ctq:qwen_vlm"
        if {language + "layers.0.self_attn.q_proj.weight", visual + "blocks.0.attn.qkv.weight",
            visual + "merger.mlp.2.weight"} <= keys:
            return "ctq:qwen_vlm"
    if {"shared.weight", "encoder.block.0.layer.0.SelfAttention.q.weight",
        "encoder.block.0.layer.1.DenseReluDense.wo.weight"} <= keys:
        return "ctq:t5xxl"
    if {"text_model.embeddings.token_embedding.weight", "text_model.embeddings.position_embedding.weight",
        "text_model.encoder.layers.0.self_attn.q_proj.weight"} <= keys:
        return "star:Text-Encoder"
    model_type = config.get("model_type")
    if model_type in ("gemma4", "gemma4_text", "gemma4_unified") and (
            "model.language_model.layers.0.post_feedforward_layernorm.weight" in keys
            or "model.layers.0.post_feedforward_layernorm.weight" in keys):
        return "ctq:gemma4"
    if {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight"} <= keys:
        if model_type in ("mistral", "mistral3", "ministral3"):
            return "ctq:mistral"
        return "star:Text-Encoder"
    if {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.router.weight", "model.layers.0.mlp.experts.down_proj_blocks"} <= keys:
        return "star:Text-Encoder"
    return None

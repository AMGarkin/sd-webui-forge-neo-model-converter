"""Additional host tensor contracts, independent of executable host packages.

Audited source versions and licenses are recorded in docs/HF_LAYOUT_SOURCES.md.
Every source tensor must be consumed; unsupported extensions fail closed.
"""

import re


def _renames(name, pairs):
    for old, new in pairs:
        name = name.replace(old, new)
    return name


def _fuse(mapping, tensors, names, target):
    """Require every piece whenever any piece of a fused projection exists."""
    if not any(n in tensors for n in names):
        return
    cursor = 0
    for name in names:
        if name not in tensors:
            raise ValueError(f"Incomplete fused projection {target}: missing {name}")
        rows = tensors[name]["shape"][0]
        mapping[name] = (target, [0, cursor, rows], False)
        cursor += rows


def hunyuan(tensors, config):
    mapping = {}
    roots = [("x_embedder.", "img_in."), ("proj_out.", "final_layer.linear."),
             ("norm_out.linear.", "final_layer.adaLN_modulation.1."),
             ("context_embedder.proj_in.", "txt_in.input_embedder."),
             ("context_embedder.time_text_embed.timestep_embedder.linear_1.", "txt_in.t_embedder.in_layer."),
             ("context_embedder.time_text_embed.timestep_embedder.linear_2.", "txt_in.t_embedder.out_layer."),
             ("context_embedder.time_text_embed.text_embedder.linear_1.", "txt_in.c_embedder.in_layer."),
             ("context_embedder.time_text_embed.text_embedder.linear_2.", "txt_in.c_embedder.out_layer."),
             ("time_text_embed.timestep_embedder.linear_1.", "time_in.in_layer."),
             ("time_text_embed.timestep_embedder.linear_2.", "time_in.out_layer."),
             ("time_text_embed.text_embedder.linear_1.", "vector_in.in_layer."),
             ("time_text_embed.text_embedder.linear_2.", "vector_in.out_layer."),
             ("time_text_embed.guidance_embedder.linear_1.", "guidance_in.in_layer."),
             ("time_text_embed.guidance_embedder.linear_2.", "guidance_in.out_layer."),
             ("time_embed.timestep_embedder.linear_1.", "time_in.in_layer."),
             ("time_embed.timestep_embedder.linear_2.", "time_in.out_layer."),
             ("time_r_embed.timestep_embedder.linear_1.", "time_r_in.in_layer."),
             ("time_r_embed.timestep_embedder.linear_2.", "time_r_in.out_layer."),
             ("cond_type_embed.", "cond_type_embedding."),
             ("context_embedder_2.norm.", "byt5_in.layernorm."),
             ("context_embedder_2.linear_1.", "byt5_in.fc1."),
             ("context_embedder_2.linear_2.", "byt5_in.fc2."),
             ("context_embedder_2.linear_3.", "byt5_in.fc3."),
             ("image_embedder.norm_in.", "vision_in.proj.0."),
             ("image_embedder.linear_1.", "vision_in.proj.1."),
             ("image_embedder.linear_2.", "vision_in.proj.3."),
             ("image_embedder.norm_out.", "vision_in.proj.4.")]
    from .layouts import _blocks
    for name in tensors:
        target = next((dst + name[len(src):] for src, dst in roots if name.startswith(src)), None)
        m = re.match(r"context_embedder.token_refiner.refiner_blocks\.(\d+)\.(.+)", name)
        if m:
            tail = _renames(m[2], [("attn.to_out.0.", "self_attn.proj."), ("ff.net.0.proj.", "mlp.0."),
                                  ("ff.net.2.", "mlp.2."), ("norm_out.linear.", "adaLN_modulation.1.")])
            if tail.startswith(("norm1.", "norm2.", "self_attn.proj.", "mlp.", "adaLN_modulation.")):
                target = f"txt_in.individual_token_refiner.blocks.{m[1]}.{tail}"
        m = re.match(r"(single_)?transformer_blocks\.(\d+)\.(.+)", name)
        if m:
            single, i, tail = m.groups()
            pairs = ([("attn.norm_q.", "norm.query_norm."), ("attn.norm_k.", "norm.key_norm."),
                      ("norm.linear.", "modulation.lin."), ("proj_out.", "linear2.")] if single else
                     [("norm1.linear.", "img_mod.lin."), ("norm1_context.linear.", "txt_mod.lin."),
                      ("attn.norm_q.", "img_attn.norm.query_norm."), ("attn.norm_k.", "img_attn.norm.key_norm."),
                      ("attn.norm_added_q.", "txt_attn.norm.query_norm."), ("attn.norm_added_k.", "txt_attn.norm.key_norm."),
                      ("attn.to_out.0.", "img_attn.proj."), ("attn.to_add_out.", "txt_attn.proj."),
                      ("ff.net.0.proj.", "img_mlp.0."), ("ff.net.2.", "img_mlp.2."),
                      ("ff_context.net.0.proj.", "txt_mlp.0."), ("ff_context.net.2.", "txt_mlp.2.")])
            converted = _renames(tail, pairs)
            if converted != tail:
                target = f"{'single' if single else 'double'}_blocks.{i}.{converted}"
        if target:
            mapping[name] = (target, None, name.startswith("norm_out.linear."))
    for stack in ("transformer_blocks", "single_transformer_blocks", "context_embedder.token_refiner.refiner_blocks"):
        for i in _blocks(tensors, stack):
            base = f"{stack}.{i}."
            for param in ("weight", "bias"):
                if stack == "transformer_blocks":
                    for source, dest in ((["to_q", "to_k", "to_v"], "img_attn.qkv"),
                                         (["add_q_proj", "add_k_proj", "add_v_proj"], "txt_attn.qkv")):
                        _fuse(mapping, tensors, [base + f"attn.{p}.{param}" for p in source], f"double_blocks.{i}.{dest}.{param}")
                elif stack == "single_transformer_blocks":
                    _fuse(mapping, tensors, [base + f"attn.to_{p}.{param}" for p in "qkv"] + [base + f"proj_mlp.{param}"], f"single_blocks.{i}.linear1.{param}")
                else:
                    _fuse(mapping, tensors, [base + f"attn.to_{p}.{param}" for p in "qkv"], f"txt_in.individual_token_refiner.blocks.{i}.self_attn.qkv.{param}")
    return mapping


def mochi(tensors, config):
    from .layouts import _blocks
    mapping = {}
    roots = [("patch_embed.", "x_embedder."), ("time_embed.timestep_embedder.linear_1.", "t_embedder.mlp.0."),
             ("time_embed.timestep_embedder.linear_2.", "t_embedder.mlp.2."), ("time_embed.pooler.", "t5_y_embedder."),
             ("time_embed.caption_proj.", "t5_yproj."), ("norm_out.linear.", "final_layer.mod."),
             ("proj_out.", "final_layer.linear.")]
    pairs = [("norm1.linear.", "mod_x."), ("norm1_context.linear_1.", "mod_y."), ("norm1_context.linear.", "mod_y."),
             ("attn1.norm_q.", "attn.q_norm_x."), ("attn1.norm_k.", "attn.k_norm_x."),
             ("attn1.norm_added_q.", "attn.q_norm_y."), ("attn1.norm_added_k.", "attn.k_norm_y."),
             ("attn1.to_out.0.", "attn.proj_x."), ("attn1.to_add_out.", "attn.proj_y."),
             ("ff.net.0.proj.", "mlp_x.w1."), ("ff.net.2.", "mlp_x.w2."),
             ("ff_context.net.0.proj.", "mlp_y.w1."), ("ff_context.net.2.", "mlp_y.w2.")]
    for name in tensors:
        target = next((d + name[len(s):] for s, d in roots if name.startswith(s)), None)
        if name == "pos_frequencies":
            target = name
        m = re.match(r"transformer_blocks\.(\d+)\.(.+)", name)
        if m and (tail := _renames(m[2], pairs)) != m[2]:
            target = f"blocks.{m[1]}.{tail}"
        if target:
            swap = name.startswith("norm_out.linear.") or name.endswith(".net.0.proj.weight")
            mapping[name] = (target, None, swap)
    for i in _blocks(tensors, "transformer_blocks"):
        for sources, target in ((["to_q", "to_k", "to_v"], "qkv_x"), (["add_q_proj", "add_k_proj", "add_v_proj"], "qkv_y")):
            _fuse(mapping, tensors, [f"transformer_blocks.{i}.attn1.{p}.weight" for p in sources], f"blocks.{i}.attn.{target}.weight")
    return mapping


def lumina(tensors, config):
    from .layouts import _blocks
    mapping = {}
    pairs = [("time_caption_embed.caption_embedder.", "cap_embedder."),
             ("time_caption_embed.timestep_embedder.linear_1.", "t_embedder.mlp.0."),
             ("time_caption_embed.timestep_embedder.linear_2.", "t_embedder.mlp.2."),
             ("norm_out.linear_1.", "final_layer.adaLN_modulation.1."), ("norm_out.linear_2.", "final_layer.linear."),
             (".attn.to_out.0.", ".attention.out."), (".attn.norm_q.", ".attention.q_norm."),
             (".attn.norm_k.", ".attention.k_norm."), (".feed_forward.linear_1.", ".feed_forward.w1."),
             (".feed_forward.linear_2.", ".feed_forward.w2."), (".feed_forward.linear_3.", ".feed_forward.w3."),
             (".norm1.linear.", ".adaLN_modulation.1."), (".norm1.norm.", ".attention_norm1."), (".norm2.", ".attention_norm2.")]
    for name in tensors:
        target = _renames(name, pairs)
        if name.startswith("context_refiner."):
            target = target.replace(".norm1.", ".attention_norm1.")
        if target != name or name.startswith("x_embedder.") or ".ffn_norm" in name:
            mapping[name] = (target, None, False)
    for stack in ("layers", "noise_refiner", "context_refiner"):
        for i in _blocks(tensors, stack):
            _fuse(mapping, tensors, [f"{stack}.{i}.attn.to_{p}.weight" for p in "qkv"], f"{stack}.{i}.attention.qkv.weight")
    return mapping


def ltx2(tensors, config):
    pairs = [("audio_proj_in.", "audio_patchify_proj."), ("proj_in.", "patchify_proj."),
             ("audio_time_embed.", "audio_adaln_single."), ("time_embed.", "adaln_single."),
             ("audio_prompt_time_embed.", "audio_prompt_adaln_single."),
             ("prompt_time_embed.", "prompt_adaln_single."),
             ("audio_prompt_adaln.", "audio_prompt_adaln_single."),
             ("prompt_adaln.", "prompt_adaln_single."),
             ("av_cross_attn_video_scale_shift.", "av_ca_video_scale_shift_adaln_single."),
             ("av_cross_attn_video_a2v_gate.", "av_ca_a2v_gate_adaln_single."),
             ("av_cross_attn_audio_scale_shift.", "av_ca_audio_scale_shift_adaln_single."),
             ("av_cross_attn_audio_v2a_gate.", "av_ca_v2a_gate_adaln_single."),
             ("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video"),
             ("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio"),
             (".norm_q.", ".q_norm."), (".norm_k.", ".k_norm.")]
    # Connector namespaces can be present in complete original checkpoints. Keep
    # them intact; unlike Diffusers' single-file importer we never drop them.
    return {n: (_renames(n, pairs), None, False) for n in tensors}


def ltx_connectors(tensors, config):
    pairs = [("video_connector.", "video_embeddings_connector."), ("audio_connector.", "audio_embeddings_connector."),
             (".transformer_blocks.", ".transformer_1d_blocks."), (".norm_q.", ".q_norm."), (".norm_k.", ".k_norm."),
             ("video_text_proj_in.", "text_embedding_projection.video_aggregate_embed."),
             ("audio_text_proj_in.", "text_embedding_projection.audio_aggregate_embed."),
             ("text_proj_in.", "text_embedding_projection.")]
    return {n: (_renames(n, pairs), None, False) for n in tensors}


def wan_animate(tensors, config):
    from .layouts import _wan_map, _blocks
    mapping = _wan_map(tensors)
    for name in tensors:
        target = None
        if name.startswith('pose_patch_embedding.'):
            target = name
        if name.startswith('motion_encoder.motion_network.'):
            target = name.replace('motion_encoder.motion_network.', 'motion_encoder.enc.fc.')
        if name == 'motion_encoder.motion_synthesis_weight':
            target = 'motion_encoder.dec.direction.weight'
        if name.startswith('face_encoder.'):
            target = re.sub(r'^(face_encoder\.(?:conv1_local|conv2|conv3))\.', r'\1.conv.', name)
        m = re.match(r'face_adapter\.(\d+)\.(norm_q|norm_k|to_q|to_out)\.(.+)', name)
        if m:
            layers = {'norm_q':'q_norm','norm_k':'k_norm','to_q':'linear1_q','to_out':'linear2'}
            target = f'face_adapter.fuser_blocks.{m[1]}.{layers[m[2]]}.{m[3]}'
        m = re.match(r'motion_encoder\.(conv_in|conv_out|res_blocks\.(\d+)\.(conv1|conv2|conv_skip))\.(weight|act_fn.bias)$', name)
        if m:
            layer, index, conv, param = m.groups()
            if layer == 'conv_in':
                root, slot = 'motion_encoder.enc.net_app.convs.0', 0 if param == 'weight' else 1
            elif layer == 'conv_out':
                root, slot = 'motion_encoder.enc.net_app.convs.8', None
            else:
                root = f'motion_encoder.enc.net_app.convs.{int(index)+1}.{"skip" if conv == "conv_skip" else conv}'
                slot = (0 if conv == 'conv1' else 1) + (param != 'weight')
            target = root + (f'.{slot}' if slot is not None else '') + ('.weight' if param == 'weight' else '.bias')
        if target:
            mapping[name] = (target, None, False)
    for i in _blocks(tensors, 'face_adapter'):
        for param in ('weight','bias'):
            _fuse(mapping,tensors,[f'face_adapter.{i}.to_{p}.{param}' for p in 'kv'],f'face_adapter.fuser_blocks.{i}.linear1_kv.{param}')
    return mapping


def finish_wan_animate(output, tensors, config):
    """Restore broadcast bias axes and fixed non-persistent antialias buffers."""
    from .layouts import _blocks
    if config.get('motion_encoder_size',512) != 512:
        raise ValueError('Wan Animate motion encoder size must match the installed target contract (512)')
    for name, spec in list(output.items()):
        if name.startswith('motion_encoder.enc.net_app.') and name.endswith('.bias'):
            rows = spec['shape'][0]
            output[name] = {**spec, 'reshape': [1,rows,1,1], 'shape': [1,rows,1,1]}
    kernel = [a*b/64 for a in (1,3,3,1) for b in (1,3,3,1)]
    for i in _blocks(tensors,'motion_encoder.res_blocks'):
        for conv in ('conv2','skip'):
            name = f'motion_encoder.enc.net_app.convs.{i+1}.{conv}.0.kernel'
            output[name] = {'dtype':'F32','shape':[4,4],'data_offsets':[0,64], 'constant':kernel,
                            'file':next(iter(tensors.values()))['file']}
    return output


def krea2(tensors, config):
    pairs = [('img_in.','first.'),('time_embed.linear_1.','tmlp.0.'),('time_embed.linear_2.','tmlp.2.'),
             ('time_mod_proj.','tproj.1.'),('txt_in.linear_1.','txtmlp.1.'),('txt_in.linear_2.','txtmlp.3.'),
             ('txt_in.norm.weight','txtmlp.0.scale'),('text_fusion.','txtfusion.'),('final_layer.linear.','last.linear.'),
             ('transformer_blocks.','blocks.'),('.attn.to_q.','.attn.wq.'),('.attn.to_k.','.attn.wk.'),
             ('.attn.to_v.','.attn.wv.'),('.attn.to_gate.','.attn.gate.'),('.attn.to_out.0.','.attn.wo.'),
             ('.attn.norm_q.weight','.attn.qknorm.qnorm.scale'),('.attn.norm_k.weight','.attn.qknorm.knorm.scale'),
             ('.ff.gate.','.mlp.gate.'),('.ff.up.','.mlp.up.'),('.ff.down.','.mlp.down.'),
             ('.prenorm.weight','.prenorm.scale'),('.postnorm.weight','.postnorm.scale')]
    pairs += [('final_layer.norm.weight','last.norm.scale'),('final_layer.scale_shift_table','last.modulation.lin'),
              ('.norm1.weight','.prenorm.scale'),('.norm2.weight','.postnorm.scale'),('.scale_shift_table','.mod.lin')]
    return {n: (_renames(n,pairs),None,False) for n in tensors}


def ideogram(tensors, config):
    from .layouts import _blocks
    mapping={n:(n.replace('.attention.to_out.0.','.attention.o.'),None,False)
             for n in tensors if '.attention.to_' not in n or '.attention.to_out.0.' in n}
    for i in _blocks(tensors,'layers'):
        _fuse(mapping,tensors,[f'layers.{i}.attention.to_{p}.weight' for p in 'qkv'],f'layers.{i}.attention.qkv.weight')
    return mapping


def yue2(tensors, config):
    from .layouts import _blocks, _mapped, recipe, source_part
    mapping={}
    diffusion='model.diffusion_model.'
    text='text_encoders.model.'
    for name in tensors:
        if name.startswith(('time_embedder.','latent_pos_embed.','vae2llm.','llm2vae.')) or name=='model.norm.weight':
            mapping[name]=(diffusion+name,None,False)
        elif name=='model.embed_tokens.weight':
            mapping[name]=(text+'embed_tokens.weight',None,False)
        elif name=='lm_head.weight':
            mapping[name]=(text+'lm_head.weight',None,False)
        elif name.startswith('model.layers.'):
            if '.nar_' in name:
                target=name.replace('.nar_self_attn.','.self_attn.').replace('.nar_mlp.','.mlp.').replace('.nar_input_layernorm.','.input_layernorm.').replace('.nar_pre_mlp_layernorm.','.post_attention_layernorm.')
                mapping[name]=(diffusion+target,None,False)
            else:
                mapping[name]=('text_encoders.'+name,None,False)
    for i in _blocks(tensors,'model.layers'):
        for src,dst in (('nar_',diffusion),('', 'text_encoders.')):
            base=f'model.layers.{i}.'
            for n in [base+src+'self_attn.'+p+'.weight' for p in ('q_proj','k_proj','v_proj')]+[base+src+'mlp.'+p+'.weight' for p in ('gate_proj','up_proj')]:
                mapping.pop(n,None)
            _fuse(mapping,tensors,[base+src+f'self_attn.{p}_proj.weight' for p in 'qkv'],dst+base+'self_attn.qkv_proj.weight')
            _fuse(mapping,tensors,[base+src+f'mlp.{p}_proj.weight' for p in ('gate','up')],dst+base+'mlp.gate_up_proj.weight')
    output=_mapped(tensors,mapping)
    output[text+'norm.weight']=recipe([source_part('model.norm.weight',tensors)])
    return output


def minimax_raw(tensors, config):
    from .layouts import recipe, source_part
    output=dict(tensors)
    heads,dim=config.get('num_attention_heads'),config.get('attention_head_dim')
    if type(heads) is not int or type(dim) is not int or not 0<heads<=256 or not 0<dim<=512:
        raise ValueError('Original MiniMax interleaved QKV requires explicit valid head counts/dimensions')
    for name,spec in tensors.items():
        if name.endswith('.attn.qkv_proj.weight'):
            if spec['shape'][0]!=3*heads*dim:
                raise ValueError('Original MiniMax QKV rows disagree with config')
            output[name]=recipe([source_part(name,tensors,(h*3+p)*dim,(h*3+p+1)*dim) for p in range(3) for h in range(heads)])
    return output


def minimax(tensors, config):
    from .layouts import _blocks
    pairs=[('audio_proj_in.','audio_patch_proj.'),('audio_proj_out.','final_layer.audio_out.'),
           ('proj_in.','video_patch_proj.'),('proj_out.','final_layer.video_out.'),
           ('context_embedder.','condition_proj.'),('time_embedder.linear_1.','time_embedder.proj_in.'),
           ('time_embedder.linear_2.','time_embedder.proj_out.'),('norm_out.linear.','final_layer.adaln_proj.linear.'),
           ('norm_out.norm.','final_layer.norm.'),('token_refiner.refiner_blocks.','token_refiner.blocks.'),
           ('transformer_blocks.','blocks.'),('.attn.norm_q.','.attn.q_norm.'),('.attn.norm_k.','.attn.k_norm.'),
           ('.attn.to_out.0.','.attn.out_proj.'),('.ff.net.0.proj.','.mlp.fc1.'),('.ff.net.2.','.mlp.fc2.')]
    mapping={}
    for name in tensors:
        target=_renames(name,pairs)
        if target!=name or name.startswith('token_refiner.final_norm.'):
            mapping[name]=(target,None,name.endswith('.ff.net.0.proj.weight'))
    for stack in ('transformer_blocks','token_refiner.refiner_blocks'):
        for i in _blocks(tensors,stack):
            target=('blocks' if stack=='transformer_blocks' else 'token_refiner.blocks')+f'.{i}.attn.qkv_proj.weight'
            _fuse(mapping,tensors,[f'{stack}.{i}.attn.to_{p}.weight' for p in 'qkv'],target)
    return mapping


def minimax_pruned(tensors, config):
    """Map already folded rank-k AdaLN coordinates to Comfy's native curve path."""
    from .layouts import _blocks
    rank, hidden, grid = config.get('adaln_rank'), config.get('hidden_size'), config.get('time_table_size')
    if any(type(value) is not int or value <= 0 for value in (rank, hidden, grid)) or grid < 2:
        raise ValueError('Pruned MiniMax requires explicit positive rank/hidden size and a time grid of at least two rows')
    if config.get('_10eros', {}).get('mlp_source_gate_first') is not True:
        raise ValueError('Pruned MiniMax requires the audited gate-first MLP declaration')
    expected = {'time_embedder.table': [grid, rank], 'adaln_basis': [rank, config.get('time_embed_dim')],
                'adaln_mean': [config.get('time_embed_dim')], 'norm_out.linear.weight': [2 * hidden, rank],
                'norm_out.folded_bias': [2 * hidden]}
    for block in _blocks(tensors, 'transformer_blocks'):
        expected[f'transformer_blocks.{block}.adaln_proj.linear.weight'] = [18 * hidden, rank]
        expected[f'transformer_blocks.{block}.adaln_proj.folded_bias'] = [18 * hidden]
    for name, shape in expected.items():
        if tensors.get(name, {}).get('shape') != shape:
            raise ValueError(f'Pruned MiniMax folded AdaLN shape mismatch: {name}')
    if len(_blocks(tensors, 'token_refiner.refiner_blocks')) != config.get('num_refiner_layers'):
        raise ValueError('Pruned MiniMax token-refiner depth disagrees with config')
    mapping = minimax(tensors, config)
    # This audited pruned source and Comfy both store SwiGLU gate before up.
    for name, (target, offset, swap) in list(mapping.items()):
        if name.endswith('.ff.net.0.proj.weight'):
            mapping[name] = (target, offset, False)
    for name in tensors:
        if name in ('adaln_basis', 'adaln_mean'):
            mapping[name] = (name, None, False)
        elif name == 'time_embedder.table':
            mapping[name] = ('adaln_t_table', None, False)
        elif name == 'norm_out.folded_bias':
            mapping[name] = ('final_layer.adaln_proj.linear.bias', None, False)
        elif name.endswith('.adaln_proj.folded_bias'):
            mapping[name] = (name.replace('transformer_blocks.', 'blocks.').replace('.folded_bias', '.linear.bias'), None, False)
    return mapping


def finish_minimax(output, tensors, config):
    import torch
    count=config.get('rope_freq_dim',16)
    theta=config.get('rope_theta',10000.0)
    if type(count) is not int or not 0<count<=128 or not isinstance(theta,(int,float)) or theta<=0:
        raise ValueError('Invalid MiniMax rotary frequency config')
    values=(1.0/(theta**(torch.arange(count,dtype=torch.float32)/count))).tolist()
    output['rope.inv_freq']={'shape':[count],'dtype':'F32','data_offsets':[0,count*4],'constant':values,
                            'file':next(iter(tensors.values()))['file']}
    return output


def cosmos(tensors, config):
    """Cosmos 1/Predict2 selected by concrete config, never fictitious V2 class."""
    predict2 = not bool(config.get("extra_pos_embed_type", "learnable"))
    roots = [("time_embed.t_embedder.", "t_embedder.1."),
             ("time_embed.norm.", "t_embedding_norm." if predict2 else "affline_norm."),
             ("patch_embed.proj.", "x_embedder.proj.1."), ("learnable_pos_embed.", "extra_pos_embedder."),
             ("proj_out.", "final_layer.linear."),
             ("norm_out.linear_1.", "final_layer.adaln_modulation.1." if predict2 else "final_layer.adaLN_modulation.1."),
             ("norm_out.linear_2.", "final_layer.adaln_modulation.2." if predict2 else "final_layer.adaLN_modulation.2.")]
    first = [("attn1.", "self_attn."), ("attn2.", "cross_attn."),
             ("norm1.linear_", "adaln_modulation_self_attn."), ("norm2.linear_", "adaln_modulation_cross_attn."),
             ("norm3.linear_", "adaln_modulation_mlp."), ("ff.net.0.proj.", "mlp.layer1."), ("ff.net.2.", "mlp.layer2.")]
    if not predict2:
        first = [("attn1.", "blocks.0.block.attn."), ("attn2.", "blocks.1.block.attn."),
                 ("norm1.linear_", "blocks.0.adaLN_modulation."), ("norm2.linear_", "blocks.1.adaLN_modulation."),
                 ("norm3.linear_", "blocks.2.adaLN_modulation."),
                 ("ff.net.0.proj.", "blocks.2.block.layer1."), ("ff.net.2.", "blocks.2.block.layer2.")]
    attn = ([(".to_q.", ".q_proj."), (".to_k.", ".k_proj."), (".to_v.", ".v_proj."),
             (".norm_q.", ".q_norm."), (".norm_k.", ".k_norm."), (".to_out.0.", ".output_proj.")] if predict2 else
            [(".to_q.", ".to_q.0."), (".to_k.", ".to_k.0."), (".to_v.", ".to_v.0."),
             (".norm_q.", ".to_q.1."), (".norm_k.", ".to_k.1.")])
    mapping = {}
    for name in tensors:
        target = next((d + name[len(s):] for s, d in roots if name.startswith(s)), None)
        m = re.match(r"transformer_blocks\.(\d+)\.(.+)", name)
        if m and (tail := _renames(m[2], first)) != m[2]:
            target = (f"blocks.{m[1]}." if predict2 else f"blocks.block{m[1]}.") + _renames(tail, attn)
        if target:
            mapping[name] = (target, None, False)
    return mapping

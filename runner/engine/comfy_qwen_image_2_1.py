#==================================================================================================
#
#   Copyright (C) 2026-2026 turboCLI authors. <https://omega.gg/turboCLI>
#
#   Author: Benjamin Arnaud. <https://bunjee.me> <bunjee@omega.gg>
#
#   This file is part of turboCLI.
#
#   - GNU Lesser General Public License Usage:
#   This file may be used under the terms of the GNU Lesser General Public License version 3 as
#   published by the Free Software Foundation and appearing in the LICENSE.md file included in the
#   packaging of this file. Please review the following information to ensure the GNU Lesser
#   General Public License requirements will be met: https://www.gnu.org/licenses/lgpl.html.
#
#   - Private License Usage:
#   turboCLI licensees holding valid private licenses may use this file in accordance with the
#   private license agreement provided with the Software or, alternatively, in accordance with the
#   terms contained in written agreement between you and turboCLI authors. For further information
#   contact us at contact@omega.gg.
#
#==================================================================================================

# comfy-qwen-image-2-1 engine: text2img + img2img on Qwen-Image 2.1, REUSING a ComfyUI install.
#
# One 7B model generates and edits with the same weights, so one engine declares both modes. It
# loads the files ComfyUI's three Qwen-Image 2.1 templates load: the int8 ConvRot transformer and
# the int8 ConvRot Qwen3-VL 8B text encoder, both streamed through the vendored comfy quant path
# (kept int8, run by comfy-kitchen's int8 kernels as ComfyUI does), plus the RGBA VAE. Only the
# configs, processor and scheduler come from the diffusers repo (the scaffold).
#
# OFFLOADER-ONLY: int8 needs the comfy quant path, so load() bails out for other offload modes.
# The transformer runs the way ComfyUI's model does (_comfy_forward), so the ComfyUI file binds
# as is, with no key conversion. So does the VAE: it is ComfyUI's own (comfy/ldm/wan/vae2_2.py,
# vendored by the offloader), which decodes in row strips rather than diffusers' tiles, see
# _build_vae. The sampling schedule is pinned to ComfyUI's (shift 0.69, simple), see load().
#
# The output is RGBA (the VAE decodes 4 channels), so save it as PNG.
#
# IMPORTANT (see engine/__init__.py): no torch/diffusers import at top level, discovery stays
# cheap. The heavy imports live inside load() and the helpers.

import os

from . import comfy_qwen_image_edit_2511 as qe  # cheap: no torch at top level

ID   = "comfy-qwen-image-2-1"
TYPE = "qwen-image-21"

PIPELINE    = "diffusers:QwenImage21Pipeline"
TRANSFORMER = "diffusers:QwenImage21Transformer2DModel"  # offloader disk-stream meta-load

MODES = ("text-to-image", "image-to-image")
CFG   = ("true_cfg_scale", 1.0)  # no negative prompt: no CFG, as ComfyUI's cfg 1

INFERENCE = 25  # ComfyUI's template (the diffusers default is 40)

# The pipeline sizes every reference image to ~1 MP at its own ratio, so they go as they are.
IMAGE_AREA = None

# The three files sit at plain paths of one Comfy-Org repo (no split_files/), so each names its
# `filename`. Revision pinned.
_REPO = {"repository": "Comfy-Org/Qwen-Image-2.1",
         "revision": "cb504a4090723e43f17ad01cec0359490e2de613"}

COMFY = {
    "revision": "main",
    "components": [
        dict(_REPO, role=role, path=path, filename=path)
        for role, path in (
            ("transformer", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
            ("text_encoder", "text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
            ("vae", "vae/qwen_image_2.1_vae_bf16.safetensors"),
        )
    ],
}

# The diffusers scaffolding: configs + processor (tokenizer and chat template) + scheduler.
SCAFFOLD = {
    "repository": "Qwen",
    "model": "Qwen-Image-2.1",
    "revision": "d26bb61231c349cf6b7896fa83353113880e1ba3",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "processor/*",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
    ],
}

def _transformer_meta(scaffold, dtype, comfy):
    """Meta-build (no weight RAM) the transformer from the scaffold config, set up to run as
    ComfyUI's model does (_comfy_forward, with the offloader's comfy_api())."""
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel

    cfg = QwenImage21Transformer2DModel.load_config(os.path.join(scaffold, "transformer"))
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(cfg)
        _comfy_forward(model, comfy)

    return model.to(dtype)


def _comfy_forward(model, comfy):
    """Run a meta-built QwenImage21Transformer2DModel the way ComfyUI's own model does
    (comfy/ldm/qwen_image21/model.py). diffusers computes the same thing with separate, often fp32
    ops; ComfyUI fuses them, which is its speed at 1024². Same fusions, through the offloader's
    comfy_api() (comfy_kitchen as ComfyUI configures it, comfy.ops, comfy.model_management):
      * MLP: the fused gate_up, its SiLU gate folded into the int8 down projection's input
        quantizer (so the ComfyUI file binds as is, no split);
      * attention: QK RMSNorm + RoPE in one rms_rope kernel;
      * block: LayerNorm * (1 + scale) by the adaln kernel and in-place gated residuals, the
        shared modulation split into the t=0 prefix row and the target rows (no per-token
        copies).
    Without comfy_kitchen's kernels only the MLP is fused (linear_input_act runs it eagerly).
    The classes live in here so that discovery never imports torch."""
    import torch
    import diffusers.models.transformers.transformer_qwenimage21 as tq

    ck, ops, mm = comfy.ck, comfy.ops, comfy.mm
    rope = {"key": None, "pe": None}
    prefix_len = {"key": None, "len": 0}
    prepare_qkv_orig = getattr(tq._qwenimage21_prepare_qkv, "_comfy_orig",
                               tq._qwenimage21_prepare_qkv)

    class FusedSwiGLU(torch.nn.Module):
        """ComfyUI's SwiGLUFeedForward(fused=True): [gate; up] in one GEMM."""

        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.gate_up = torch.nn.Linear(dim, 2 * hidden_dim, bias=False)
            self.out = torch.nn.Linear(hidden_dim, dim, bias=False)

        def forward(self, x):
            return ops.linear_input_act(self.out, self.gate_up(x), "swiglu")

    def pe(rotary_emb):
        # diffusers' complex [S, D/2] table -> comfy_kitchen's [1, S, 1, D/2, 2, 2] rotations,
        # once per forward (every block gets the same tensor).
        if rope["key"] is not rotary_emb:
            c, s = rotary_emb.real.float(), rotary_emb.imag.float()
            table = torch.stack([c, -s, s, c], dim=-1).reshape(*c.shape, 2, 2)
            rope.update(key=rotary_emb, pe=table[None, :, None])
        return rope["pe"]

    blocks = len(model.transformer_blocks)
    parked = {"run": None, "pinned": []}

    def park(t, run):
        # ComfyUI's "auto" prefix cache placement (QwenImage21.select_prefix_cache): the prefix
        # K/V stay on the device while it has 4x the whole cache free, else they go to host RAM
        # (an edit's ~2 GB would otherwise starve a small card's weight streaming), pinned within
        # ComfyUI's budget as PoseBranchCache.put does. A new run frees the last one's pins.
        # (ComfyUI also prefetches them on its offload stream; on a 4 GB card that stream is busy
        # streaming the weights, and a plain copy measured faster: 6.10 against 6.49 s/step.)
        if parked["run"] is not run:
            for host in parked["pinned"]:
                mm.unpin_memory(host)
            parked.update(run=run, pinned=[])
        if not t.is_cuda or torch.cuda.mem_get_info(t.device)[0] > 4 * 2 * blocks * t.nbytes:
            return t.clone()
        host = t.to("cpu", copy=True)
        if mm.pin_memory(host, evict_active=False):
            parked["pinned"].append(host)
        return host

    def prepare_qkv(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode,
                    cache_write_slice):
        # diffusers' _qwenimage21_prepare_qkv with ComfyUI's rms_rope for the QK norm + RoPE.
        if rotary_emb is None:
            return prepare_qkv_orig(attn, hidden_states, rotary_emb, layer_cache,
                                    kv_cache_mode, cache_write_slice)

        q = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        q, k = ck.rms_rope(q, k, pe(rotary_emb), attn.norm_q.weight.to(q.device, q.dtype),
                           attn.norm_k.weight.to(q.device, q.dtype), attn.norm_q.eps)

        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                layer_cache.store(park(k[:, cache_write_slice], cache_write_slice),
                                  park(v[:, cache_write_slice], cache_write_slice))
            elif kv_cache_mode == "cached":
                cached_k, cached_v = layer_cache.get()
                k = torch.cat([cached_k.to(k.device, non_blocking=True), k], dim=1)
                v = torch.cat([cached_v.to(v.device, non_blocking=True), v], dim=1)

        return q, k, v, q.shape[1]

    prepare_qkv._comfy_orig = prepare_qkv_orig

    def prefix(target_mask):
        # Text + reference rows ahead of the target, once per forward (counting on the host
        # syncs the device, as ComfyUI's prefix_len does once).
        if target_mask is None:
            return 0
        if prefix_len["key"] is not target_mask:
            prefix_len.update(key=target_mask, len=int((~target_mask).sum()))
        return prefix_len["len"]

    def rows(params, target_mask):
        # ComfyUI's _split_rows: (t=0 prefix row, target rows); the target rows sit last.
        if target_mask is None:
            return None, params.unsqueeze(1)
        return params[-1:].unsqueeze(1), params[:-1].unsqueeze(1)

    def modulated_norm(norm, x, scale, n, zero):
        s_prefix, s_target = scale
        out = ck.adaln(x, s_target, zero, norm.eps)
        if n:
            out[:, :n] = ck.adaln(x[:, :n], s_prefix, zero, norm.eps)
        return out

    def gated_residual(x, y, gate, n):
        g_prefix, g_target = gate
        x[:, n:].addcmul_(y[:, n:], g_target)
        if n:
            x[:, :n].addcmul_(y[:, :n], g_prefix)
        return x

    class Block(tq.QwenImage21TransformerBlock):
        """The diffusers block with ComfyUI's QwenImage21TransformerBlock.forward."""

        def forward(self, hidden_states, modulation, rotary_emb=None, attention_mask=None,
                    target_token_mask=None, layer_cache=None, kv_cache_mode=None,
                    cache_write_slice=None, segments=None, key_valid=None):
            mod1, mod2 = modulation.chunk(2, dim=-1)
            scale1, gate1 = mod1.chunk(2, dim=-1)
            scale2, gate2 = mod2.chunk(2, dim=-1)
            scale1, gate1, scale2, gate2 = (rows(p, target_token_mask)
                                            for p in (scale1, gate1.tanh(), scale2, gate2.tanh()))
            n = prefix(target_token_mask)
            zero = torch.zeros_like(scale1[1])

            y = self.attn(hidden_states=modulated_norm(self.img_norm1, hidden_states, scale1, n,
                                                       zero),
                          attention_mask=attention_mask, rotary_emb=rotary_emb,
                          layer_cache=layer_cache, kv_cache_mode=kv_cache_mode,
                          cache_write_slice=cache_write_slice, segments=segments,
                          key_valid=key_valid)
            hidden_states = gated_residual(hidden_states, y, gate1, n)

            y = self.img_mlp(modulated_norm(self.img_norm2, hidden_states, scale2, n, zero))
            hidden_states = gated_residual(hidden_states, y, gate2, n)

            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
            return hidden_states

    for block in model.transformer_blocks:
        mlp = block.img_mlp
        block.img_mlp = FusedSwiGLU(mlp.proj.in_features, mlp.proj.out_features)
        if ck is not None:
            block.__class__ = Block

    if ck is not None:
        tq._qwenimage21_prepare_qkv = prepare_qkv


def _text_encoder_meta(scaffold, dtype):
    """Meta-build Qwen3-VL (vision tower + language) from the scaffold config, headless. The
    pipeline only reads hidden states, and ComfyUI never runs the vocabulary head either, so
    lm_head is an identity rather than a 151936-wide GEMM per token."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(os.path.join(scaffold, "text_encoder"))
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(cfg)

    model.lm_head = torch.nn.Identity()
    return model


def _text_encoder_convert(sd):
    """ComfyUI's flat Qwen3-VL keys -> the nested transformers module: model.visual.* stays,
    model.* moves under model.language_model.*, the unused lm_head is dropped."""
    out = {}
    for k, v in sd.items():
        if k.startswith("lm_head."):
            continue
        if k.startswith("model.") and not k.startswith("model.visual."):
            k = "model.language_model." + k[len("model."):]
        out[k] = v
    return out


def _build_vae(scaffold, weight_file, dtype, backend):
    """ComfyUI's own VAE, opted in through the offloader's comfy_vae: the WanVAE comfy/sd.py
    builds for the Qwen Image 2.1 layout, with sd.py's settings for it. It decodes a single image
    in row strips, exact and in a quarter of diffusers' memory, so a 4 GB card decodes 1024²
    whole where diffusers' tiles bent the alpha along their seams."""
    import json
    from types import SimpleNamespace

    import safetensors.torch as safetensors_torch

    import comfy.ldm.wan.vae2_2 as wan
    import comfy.model_management as mm

    sd = safetensors_torch.load_file(weight_file)
    channels = sd["decoder.head.2.weight"].shape[0]
    # comfy/sd.py's Qwen Image 2.1 branch: the model and, below, its working-memory estimates.
    model = wan.WanVAE(dim=sd["encoder.conv1.weight"].shape[0],
                       dec_dim=sd["decoder.head.0.gamma"].shape[0], z_dim=64,
                       dim_mult=[1, 2, 4, 8, 8], num_res_blocks=2, attn_scales=[],
                       temperal_downsample=[False, True, True, True], dropout=0.0,
                       image_channels=channels, patch_size=1, temporal_kernel=1)
    model.load_state_dict(sd, strict=True)

    # z_dim and the latent statistics the pipeline reads.
    with open(os.path.join(scaffold, "vae", "config.json")) as f:
        config = SimpleNamespace(**json.load(f))

    vae = backend.comfy_vae(
        model, config,
        lambda shape, dtype: 900 * shape[2] * shape[3] * (16 * 16) * mm.dtype_size(dtype),
        lambda shape, dtype: 600 * shape[2] * shape[3] * mm.dtype_size(dtype),
        latent_channels=64, image_channels=channels, ratio=16)
    return vae.to(dtype).eval()


def load(ctx, params):
    """Assemble a QwenImage21Pipeline from ComfyUI's int8 single files via the disk-stream
    offloader. OFFLOADER-ONLY (int8 needs the comfy quant path)."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json)
    files    = qe._by_role(scaffold)

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader); both big models "
                           "are int8 and need the comfy quant path" % ID)

    from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLProcessor

    # ComfyUI samples with ModelSamplingFlux(shift 0.69) and the "simple" scheduler: the default
    # linspace(1, 1/N, N) under the exponential time shift at a FIXED mu of 0.69 (base == max, so
    # mu no longer depends on the resolution) and no terminal stretch.
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        scaffold, subfolder="scheduler", base_shift=0.69, max_shift=0.69, shift_terminal=None)
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(scaffold, "processor"))
    comfy = ctx.backend.comfy_api()  # brings up the vendored comfy package, see _build_vae
    vae = _build_vae(scaffold, files["vae"], ctx.dtype, ctx.backend)

    return ctx.backend.load_pipe_comfy(
        QwenImage21Pipeline,
        {"meta": lambda d: _transformer_meta(scaffold, d, comfy), "file": files["transformer"],
         "convert": None, "quant": True},
        {"meta": lambda d: _text_encoder_meta(scaffold, d), "file": files["text_encoder"],
         "convert": _text_encoder_convert, "quant": True},
        {"scheduler": scheduler, "processor": processor, "vae": vae},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

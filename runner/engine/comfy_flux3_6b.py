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

# comfy-flux3-6b engine: text2img + img2img on FLUX.3 Image Klein (5.6B, the 16-step distilled
# flow-map checkpoint, no CFG), REUSING a ComfyUI install file for file: BFL's int8 ConvRot
# transformer and Qwen3-VL-4B text encoder, both streamed through the vendored comfy quant path,
# and the FLUX.2 VAE (the same component as comfy-flux2-4b). Only configs and the Qwen3-VL
# processor come from the diffusers repo (the scaffold); the pipeline blocks build their own
# scheduler.
#
# GATED: both BFL repos need a Hugging Face token from an account that accepted the FLUX Kommunity
# License (install.sh takes it as its last argument).
#
# OFFLOADER-ONLY: int8 needs the comfy quant path, so load() bails out for other offload modes.
# The transformer runs the way ComfyUI's model does (_comfy_forward), the file's fused qkv kept.
#
# diffusers ships FLUX.3 as a modular pipeline only, so Flux3Pipe below gives it the classic face
# the runner and the offloader speak: an encode_prompt the offloader wraps (its encode bridge,
# node boundary and encode cache), a progress_bar the runner hooks, a step callback for cancel,
# and a result with .images.

import json
import os

from . import comfy_flux2_4b as k4  # cheap: no torch at top level

ID   = "comfy-flux3-6b"
TYPE = "flux3"

PIPELINE    = "diffusers:Flux3DistilledModularPipeline"
TRANSFORMER = "diffusers:Flux3Transformer2DModel"  # offloader disk-stream meta-load

MODES = ("text-to-image", "image-to-image")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", 0.0)              # distilled / CFG-free (Flux3Pipe ignores it)

# The flow-map checkpoint takes no guidance: the denoise passes it the timestep each step jumps
# to instead.
INFERENCE = 16

# References are resized to about one megapixel, as ComfyUI's template feeds them
# (ReferenceMultiLatent, resize 1.0), then center-cropped to the 16-pixel grid by the pipeline's
# image processor.
IMAGE_AREA = 1024 * 1024

# GATED (above): a host asks for the Hugging Face token before it installs this engine.
GATED = True

_REPO = "black-forest-labs/flux-3-image-klein-comfy"

# The "_sd_" files are the step-distilled ones, the others the 50-step base. Revisions pinned.
COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": _REPO,
         "revision": "5c9b714851dc072b04a23ef6d2d36421f5ccb3a0",
         "path": "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors",
         "filename": "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors"},
        {"role": "text_encoder", "repository": _REPO,
         "revision": "5c9b714851dc072b04a23ef6d2d36421f5ccb3a0",
         "path": "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
         "filename": "text_encoders/qwen3vl_4b_int8_convrot.safetensors"},
        {"role": "vae", "repository": "Comfy-Org/flux2-dev",
         "path": "vae/flux2-vae.safetensors"},
    ],
}

# The diffusers repo, for its configs and the Qwen3-VL processor (which sits in text_encoder/).
# Weight-free: the text encoder's index is the only other file its patterns catch.
SCAFFOLD = {
    "repository": "black-forest-labs",
    "model": "flux-3-image-klein",
    "revision": "337337621b8526e120f2275d250242f4a6bedd75",
    "allow_patterns": [
        "modular_model_index.json",
        "transformer/config.json",
        "vae/config.json",
        "text_encoder/*.json",
        "text_encoder/*.txt",
    ],
}

# Per-layer int8 companions that must follow their weight to its diffusers name.
_COMPANIONS = (".weight_scale", ".comfy_quant")


class Flux3Pipe:
    """diffusers' modular FLUX.3 distilled pipeline behind the classic pipeline face. The text
    encoder runs in encode_prompt (the offloader's wrappers sit on it), the rest of the modular
    blocks take the embeddings."""

    def __init__(self, transformer, text_encoder, tokenizer, vae):
        from diffusers.modular_pipelines.flux3 import Flux3DistilledAutoBlocks

        self.transformer  = transformer
        self.text_encoder = text_encoder
        self.tokenizer    = tokenizer
        self.vae          = vae

        self._execution_device = None
        self._interrupt = False

        self._callback = None
        self._step = 0

        blocks = Flux3DistilledAutoBlocks()
        blocks.sub_blocks.pop("text_encoder")

        # NOTE: Each workflow's denoise loop opens its bar on itself: it goes through ours, which
        #       the runner hooks for its step lines, and each update runs the step callback.
        for workflow in blocks.sub_blocks["denoise"].sub_blocks.values():
            workflow.sub_blocks["denoise"].progress_bar = lambda **kwargs: self._bar(**kwargs)

        self._pipe = blocks.init_pipeline()
        self._pipe.update_components(transformer=transformer, vae=vae)

        # NOTE: The modular pipeline reads its device from where its modules sit, which the
        #       offloader decides; it follows ours instead, as the classic pipelines do.
        outer = self
        cls = type(self._pipe)
        self._pipe.__class__ = type(cls.__name__, (cls,), {
            "_execution_device": property(lambda _: outer._execution_device)})

    def set_progress_bar_config(self, **kwargs):
        self._bar_config = kwargs

    def progress_bar(self, total=None):
        from tqdm.auto import tqdm

        return tqdm(total=total, **getattr(self, "_bar_config", {}))

    def _bar(self, total=None):
        bar = self.progress_bar(total=total)
        update = bar.update

        def stepped(n=1):
            result = update(n)

            if self._callback is not None:
                self._callback(self, self._step, None, {})

            self._step += 1

            if self._interrupt:
                raise _Stop()

            return result

        bar.update = stepped

        return bar

    def encode_prompt(self, prompt, device=None):
        from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep

        return Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
            self.text_encoder, self.tokenizer, prompt, device=device)

    def __call__(self, prompt, image=None, height=None, width=None, num_inference_steps=INFERENCE,
                 generator=None, callback_on_step_end=None, **_):
        from types import SimpleNamespace

        self._callback = callback_on_step_end
        self._step = 0

        prompt_embeds = self.encode_prompt(prompt, device=self._execution_device)

        try:
            images = self._pipe(prompt_embeds=prompt_embeds, image=image, height=height,
                                width=width, num_inference_steps=num_inference_steps,
                                generator=generator, output="images")
        except _Stop:
            images = []

        return SimpleNamespace(images=images)


class _Stop(Exception):
    """A cancel or a newer request, raised out of the denoise loop."""


def _transformer_convert(sd, names):
    """ComfyUI's FLUX.3 int8 transformer -> diffusers layout, filling `names` with the
    {ComfyUI key: diffusers key} of each tensor (for _lora_keys). The file is BFL's own layout:
    two modules stay as ComfyUI runs them (see _comfy_forward), the fused qkv (attn.to_qkv) and
    the final modulation in its (shift, scale) order, which diffusers' converter would swap.
    The converter renames the rest, and each companion follows its weight, a renamed weight being
    the same tensor. Its prefix map renames the blocks' comfy_attention.config keys as well, for
    the offloader's int8 attention."""
    from diffusers.loaders.single_file_utils import (
        convert_flux3_transformer_checkpoint_to_diffusers as convert)

    def rename(key):
        return (key.replace(".qkv.", ".attn.to_qkv.")
                .replace("final_layer.image.adaLN_modulation.1.", "norm_out.linear."))

    source = {id(v): k for k, v in sd.items()}
    sd = {rename(k): v for k, v in sd.items()}

    base = {k: v for k, v in sd.items() if not k.endswith(_COMPANIONS)}
    out = convert(dict(base))

    carried = set()
    for key, value in list(out.items()):
        src = source.get(id(value))

        if src is None:
            continue

        names[src] = key
        if not src.endswith(".weight"):
            continue

        stem = rename(src)[:-len(".weight")]

        for comp in _COMPANIONS:
            c = sd.get(stem + comp)

            if c is not None:
                out[key[:-len(".weight")] + comp] = c
                carried.add(stem + comp)

    lost = [k for k in sd if k.endswith(_COMPANIONS) and k not in carried]
    if lost:
        raise RuntimeError("int8 companions with no converted weight: %s" % lost[:5])
    return out


def _transformer_meta(scaffold, dtype, comfy):
    """Meta-build the FLUX.3 transformer from the scaffold config, set up to run as ComfyUI's
    model does (_comfy_forward, with the offloader's comfy_api())."""
    from accelerate import init_empty_weights
    from diffusers import Flux3Transformer2DModel

    cfg = Flux3Transformer2DModel.load_config(os.path.join(scaffold, "transformer"))
    with init_empty_weights():
        model = Flux3Transformer2DModel.from_config(cfg)
        _comfy_forward(model, comfy)

    return model.to(dtype)


def _comfy_forward(model, comfy):
    """Run a meta-built Flux3Transformer2DModel the way ComfyUI's own model does
    (comfy/ldm/flux3/model.py, Flux3._forward and Block.forward). diffusers computes the same
    thing with a modulation GEMM per token, split q/k/v and separate fp32 ops; ComfyUI's speed
    comes from, through the offloader's comfy_api() (comfy_kitchen as ComfyUI configures it and
    comfy.ops):
      * the modulation: one vector per mode, not one row per token;
      * the attention: the file's fused qkv (one GEMM, one activation quantization), QK RMSNorm
        + RoPE in one rms_rope kernel;
      * the MLP: the SwiGLU folded into the int8 mlp_out's input quantizer, the attention output
        added in its epilogue;
      * the block: adaln for LayerNorm * (1 + scale) + shift, in-place gated residuals per mode.
    The distilled flow map modulates the text and references with each step's target time, so
    ComfyUI runs their blocks every step too. The classes live in here so that discovery never
    imports torch."""
    import math

    import torch
    import torch.nn.functional as F
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    ck, ops = comfy.ck, comfy.ops

    if ck is None:
        raise RuntimeError("%s needs comfy-kitchen, which every turboCLI build installs" % ID)

    blocks = [*model.txt_mode_blocks, *model.single_transformer_blocks]
    for mode in model.mode_blocks.values():
        blocks += list(mode)

    for block in blocks:
        attn = block.attn
        attn.to_qkv = torch.nn.Linear(attn.query_dim, 3 * attn.inner_dim, bias=False)
        del attn.to_q, attn.to_k, attn.to_v

    def embedding(t):
        # comfy/ldm/flux/layers.py timestep_embedding(t, 256): [cos, sin] of t * 1000.
        t = 1000.0 * t
        half = model.config.timestep_channels // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32,
                                                          device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t)

    def rope(ids):
        # comfy/ldm/flux/math.py rope() per (t, h, w, l) axis, stacked as EmbedND does:
        # [1, S, 1, D/2, 2, 2] rotations.
        out = []
        for i, dim in enumerate(model.config.axes_dims_rope):
            scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64,
                                   device=ids.device)
            a = ids[:, i].float()[:, None] * (1.0 / (model.config.rope_theta ** scale))
            out.append(torch.stack([a.cos(), -a.sin(), a.sin(), a.cos()], dim=-1)
                       .reshape(*a.shape, 2, 2))
        return torch.cat(out, dim=-3).float()[None, :, None]

    def modulation(head, vec):
        # comfy Modulation(double=False): (shift, scale, gate) of silu(vec), each [B, 1, D].
        return head.linear(F.silu(vec[:, None])).chunk(3, dim=-1)

    def block_forward(self, x, spans, pe):
        # ComfyUI's Block.forward: spans are (start, end, (shift, scale, gate)) row ranges, the
        # first modulating every row, the others redoing theirs.
        B, N, _ = x.shape
        attn = self.attn
        shift, scale, _ = spans[0][2]
        x_mod = ck.adaln(x, scale, shift, self.norm.eps)
        for start, end, (shift, scale, _) in spans[1:]:
            x_mod[:, start:end] = ck.adaln(x[:, start:end], scale, shift, self.norm.eps)

        q, k, v = attn.to_qkv(x_mod).chunk(3, dim=-1)
        mlp = self.ff.linear_in(x_mod)
        del x_mod
        q, k = ck.rms_rope(q.view(B, N, attn.heads, -1), k.view(B, N, attn.heads, -1), pe,
                           attn.norm_q.weight.to(q.device, q.dtype),
                           attn.norm_k.weight.to(q.device, q.dtype), attn.norm_q.eps)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.view(B, N, attn.heads, -1).transpose(1, 2))
        out = attn.to_out[0](y.transpose(1, 2).reshape(B, N, -1))
        # The attention + MLP sum folded into the mlp_out GEMM epilogue.
        out = ops.linear_input_act(self.ff.linear_out, mlp, "swiglu", residual=out,
                                   residual_scale=out.new_ones(out.shape[-1]))
        del mlp

        for start, end, (_, _, gate) in spans:
            x[:, start:end].addcmul_(out[:, start:end], gate)
        return x

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids,
                image_cond_hidden_states=None, image_cond_ids=None, target_timestep=None,
                attention_kwargs=None, return_dict=True):
        # ComfyUI's Flux3._forward, preprocess_conds inlined. The joint sequence is
        # [image, references, text], as ComfyUI's.
        B, n_img, _ = hidden_states.shape
        dtype = hidden_states.dtype

        vec_target = 0
        if target_timestep is not None:
            vec_target = self.target_timestep_embedder(embedding(target_timestep[:1].to(dtype)))

        # The text and references at t = 0, in their own mode's early blocks.
        vec = self.timestep_embedder(embedding(hidden_states.new_zeros(1))) + vec_target
        txt = self.context_embedder(encoder_hidden_states)
        streams = [(txt, "txt", txt_ids, self.txt_mode_blocks)]
        refs = None
        if image_cond_hidden_states is not None:
            refs = self.x_embedders["image_cond"](image_cond_hidden_states)
            streams.append((refs, "image_cond", image_cond_ids, self.mode_blocks["image_cond"]))
        for tokens, mode, ids, mode_blocks in streams:
            mod = modulation(self.early_stream_modulations[mode], vec)
            pe = rope(ids)
            for block in mode_blocks:
                block(tokens, [(0, tokens.shape[1], mod)], pe)

        vec = self.timestep_embedder(embedding(torch.cat([timestep, timestep.new_zeros(1)])
                                               .to(dtype))) + vec_target
        vec_img, vec_zero = vec[:-1], vec[-1:]
        early = modulation(self.early_stream_modulations["image"], vec_img)

        img = self.x_embedders["image"](hidden_states)
        parts = [(img, "image", img_ids)]
        if refs is not None:
            parts.append((refs, "image_cond", image_cond_ids))
        parts.append((txt, "txt", txt_ids))
        pe = rope(torch.cat([ids for _, _, ids in parts]))
        spans, start = [], 0
        for tokens, mode, _ in parts:
            mod = modulation(self.single_stream_modulations[mode],
                             vec_img if mode == "image" else vec_zero)
            spans.append((start, start + tokens.shape[1], mod))
            start += tokens.shape[1]
        x = torch.empty(B, start, img.shape[-1], device=img.device, dtype=img.dtype)

        for block in self.mode_blocks["image"]:
            img = block(img, [(0, n_img, early)], pe[:, :n_img])

        torch.cat([img] + [tokens for tokens, _, _ in parts[1:]], dim=1, out=x)
        del img, parts
        for block in self.single_transformer_blocks:
            x = block(x, spans, pe)

        # comfy LastLayer: its modulation kept in the file's (shift, scale) order.
        shift, scale = self.norm_out.linear(F.silu(vec_img[:, None])).chunk(2, dim=-1)
        out = self.proj_out(ck.adaln(x[:, :n_img], scale, shift, self.norm_out.norm.eps))
        return Transformer2DModelOutput(sample=out) if return_dict else (out,)

    for block in blocks:
        cls = type(block)
        block.__class__ = type(cls.__name__, (cls,), {"forward": block_forward})

    model.__class__ = type(type(model).__name__, (type(model),), {"forward": forward})


def _lora_keys(scaffold, names):
    """LoRA key map, ComfyUI's model_lora_keys_unet for FLUX.3 (comfy/lora.py: its generic
    entries and its Flux3 branch) with `to` the diffusers key of each ComfyUI one (`names`, from
    _transformer_convert): ComfyUI's naming (diffusion_model. or lora_unet_), and q/k/v LoRAs on
    the fused qkv, each a third of its rows. diffusers' to_q/to_k/to_v names go to the same thirds
    (offloader.adapter.add_lora maps the rest of diffusers' naming)."""
    with open(os.path.join(scaffold, "transformer", "config.json")) as f:
        cfg = json.load(f)
    third = cfg["num_attention_heads"] * cfg["attention_head_dim"]

    key_map = {}
    for k, to in names.items():
        # ComfyUI's model names the QK norms' scales weights (Flux.process_unet_state_dict).
        if k.endswith("_norm.scale"):
            k = k[:-len(".scale")] + ".weight"

        if not k.endswith(".weight"):
            key_map["diffusion_model.{}".format(k)] = to
            continue

        key = k[:-len(".weight")]
        key_map["lora_unet_{}".format(key.replace(".", "_"))] = to
        key_map["diffusion_model.{}".format(key)] = to

        if key.endswith(".qkv"):
            for i, proj in enumerate(("q", "k", "v")):
                part = (to, (0, i * third, third))
                lora = "{}.{}_proj".format(key[:-len(".qkv")], proj)
                key_map["lora_unet_{}".format(lora.replace(".", "_"))] = part
                key_map["diffusion_model.{}".format(lora)] = part

                lora = "{}.to_{}".format(to[:-len(".to_qkv.weight")], proj)
                key_map[lora] = part
                key_map["transformer.{}".format(lora)] = part
    return key_map


def _text_encoder_meta(scaffold, dtype):
    """Meta-build Qwen3-VL-4B from the scaffold config, headless: the pipeline reads hidden
    states only and ComfyUI's file carries no lm_head, so the head is an identity. The file is
    already in transformers' layout (model.language_model.*, model.visual.*): no convert."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(os.path.join(scaffold, "text_encoder"))
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(cfg)

    model.lm_head = torch.nn.Identity()
    return model


def load(ctx, params):
    """Assemble Flux3Pipe from ComfyUI's int8 single files via the disk-stream offloader.
    OFFLOADER-ONLY (int8 needs the comfy quant path). The VAE and the processor are built from
    the scaffold."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json)
    files    = k4._by_role(scaffold)

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader); both big models "
                           "are int8 and need the comfy quant path" % ID)

    from transformers import Qwen3VLProcessor

    tokenizer = Qwen3VLProcessor.from_pretrained(os.path.join(scaffold, "text_encoder"))
    vae = k4._build_vae(scaffold, files["vae"], ctx.dtype)
    comfy = ctx.backend.comfy_api()
    names = {}  # filled by the convert, read by the LoRA key map (the offloader runs it after)

    return ctx.backend.load_pipe_comfy(
        Flux3Pipe,
        {"meta": lambda d: _transformer_meta(scaffold, d, comfy), "file": files["transformer"],
         "convert": lambda sd: _transformer_convert(sd, names), "quant": True,
         "lora_keys": lambda: _lora_keys(scaffold, names)},
        {"meta": lambda d: _text_encoder_meta(scaffold, d), "file": files["text_encoder"],
         "quant": True},
        {"tokenizer": tokenizer, "vae": vae},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

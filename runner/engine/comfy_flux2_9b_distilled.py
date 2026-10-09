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

# comfy-flux2-9b-distilled engine: text2img + img2img on FLUX.2-klein-9B, REUSING a ComfyUI
# install, file for file as ComfyUI's distilled Klein 9B template
# (image_flux2_klein_image_edit_9b_distilled): the fp8 transformer (flux-2-klein-9b-fp8, ~9.4GB)
# and the fp8-mixed Qwen3-8B text encoder (qwen_3_8b_fp8mixed, ~8.7GB), both streamed through the
# vendored comfy quant path (kept fp8, dequantized per forward as ComfyUI does), plus BFL's small
# decoder VAE (full_encoder_small_decoder: same encoder and latents, a narrower ~28M decoder,
# ~1.4x faster and lighter to decode). Only configs, tokenizer and scheduler come from the
# diffusers repo (the scaffold). comfy-flux2-9b is the bf16 sibling with flux2-9b's weights.
#
# GATED: the transformer and the scaffold sit in black-forest-labs repos that need a Hugging Face
# token from an account that accepted the FLUX Non-Commercial License (install.sh takes it as its
# last argument). The text encoder and VAE repos are open (the VAE is Apache-2.0).
#
# OFFLOADER-ONLY: fp8 needs the comfy quant path, so load() bails out for other offload modes.
# Everything but the files and load() is comfy-flux2-9b's (BASE: same scaffold, same pipeline).
# The meta-builders are comfy-flux2-4b's (scaffold-driven, size-agnostic). Unlike 4B, ComfyUI's
# fp8-mixed Qwen3-8B file carries no lm_head at all: the head is an identity, see
# _text_encoder_meta (no _tie_lm_head either).

import os

from . import comfy_flux2_4b as k4  # cheap: no torch at top level
from . import comfy_flux2_9b as base

ID   = "comfy-flux2-9b-distilled"
BASE = base.ID

# GATED (above): a host asks for the Hugging Face token before it installs this engine.
GATED = True

# The transformer and the VAE sit at the ROOT of their BFL repos, so they name `filename`; the text
# encoder follows Comfy-Org's split_files/ layout. Revisions pinned.
COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": "black-forest-labs/FLUX.2-klein-9b-fp8",
         "revision": "902d9d510b51533e07729f19211414a3648b77d2",
         "path": "diffusion_models/flux-2-klein-9b-fp8.safetensors",
         "filename": "flux-2-klein-9b-fp8.safetensors"},
        {"role": "text_encoder", "repository": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
         "revision": "3f62d9d8ae1fec33c6e91453d5c712855b096b55",
         "path": "text_encoders/qwen_3_8b_fp8mixed.safetensors"},
        {"role": "vae", "repository": "black-forest-labs/FLUX.2-small-decoder",
         "revision": "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a",
         "path": "vae/full_encoder_small_decoder.safetensors",
         "filename": "full_encoder_small_decoder.safetensors"},
    ],
}

# The small decoder's channel widths (FLUX.2-small-decoder's config.json); everything else in the
# VAE config is the standard FLUX.2 one from the scaffold.
SMALL_DECODER = [96, 192, 384, 384]

# Per-layer fp8 companions (injected or kept by comfy's convert_old_quants) that must follow their
# weight to its diffusers name.
_COMPANIONS = (".weight_scale", ".input_scale", ".comfy_quant")


def _transformer_convert(sd):
    """ComfyUI FLUX.2 fp8 transformer -> diffusers layout. diffusers' own converter remaps the
    base keys (renames + the fused qkv torch.chunk); each layer's fp8 companions follow its weight:
    a renamed weight is the same tensor and a split one a view into it, so every converted weight
    is traced back to its source by address. The file's scales are all per-tensor scalars, so the
    split q/k/v parts share theirs. A companion left without its weight is an error."""
    from diffusers.loaders.single_file_utils import (
        convert_flux2_transformer_checkpoint_to_diffusers as convert)

    base = {k: v for k, v in sd.items() if not k.endswith(_COMPANIONS)}
    spans = [(v.data_ptr(), v.data_ptr() + v.numel() * v.element_size(), k)
             for k, v in base.items() if v.numel()]
    out = convert(dict(base))

    carried = set()
    for nk, v in list(out.items()):
        if not nk.endswith(".weight") or not v.numel():
            continue
        src = next((k for lo, hi, k in spans if lo <= v.data_ptr() < hi), None)
        if src is None or not src.endswith(".weight"):
            continue
        for comp in _COMPANIONS:
            c = sd.get(src[:-len(".weight")] + comp)
            if c is not None:
                out[nk[:-len(".weight")] + comp] = c
                carried.add(src[:-len(".weight")] + comp)

    lost = [k for k in sd if k.endswith(_COMPANIONS) and k not in carried]
    if lost:
        raise RuntimeError("fp8 companions with no converted weight: %s" % lost[:5])
    return out


def _build_vae(scaffold, weight_file, dtype):
    """The FLUX.2 VAE with the small decoder: the scaffold's config with the narrower decoder
    widths. Unlike flux2-vae, ComfyUI's file is in the original BFL layout: diffusers' LDM VAE
    converter remaps it once the quant convs nested in encoder/decoder are lifted to the root,
    and the latent BatchNorm (which the converter drops) is carried as is. The result equals the
    repo's own diffusers file tensor for tensor (251/251, verified)."""
    import safetensors.torch as safetensors_torch
    from diffusers import AutoencoderKLFlux2
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    cfg = AutoencoderKLFlux2.load_config(os.path.join(scaffold, "vae"))
    cfg = dict(cfg, decoder_block_out_channels=SMALL_DECODER)

    sd = safetensors_torch.load_file(weight_file)
    lifted = {k.split(".", 1)[1] if k.startswith(("encoder.quant_conv.",
                                                  "decoder.post_quant_conv.")) else k: v
              for k, v in sd.items()}

    out = convert_ldm_vae_checkpoint(lifted, cfg)
    out.update({k: v for k, v in sd.items() if k.startswith("bn.")})

    vae = AutoencoderKLFlux2.from_config(cfg)
    vae.load_state_dict(out, strict=True)

    return vae.to(dtype).eval()


def _text_encoder_meta(scaffold, dtype):
    """comfy-flux2-4b's Qwen3 meta-build, headless. ComfyUI's fp8-mixed Qwen3-8B file carries no
    lm_head (its text encoder only returns hidden states), and the klein pipeline reads hidden
    states too, so the head becomes an identity rather than a layer left without weights."""
    import torch

    model = k4._text_encoder_meta(scaffold, dtype)
    model.lm_head = torch.nn.Identity()
    return model


def load(ctx, params):
    """Assemble a Flux2KleinPipeline from ComfyUI's fp8 single files via the disk-stream
    offloader. OFFLOADER-ONLY (fp8 needs the comfy quant path). The VAE, tokenizer and scheduler
    are built from the scaffold."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json)
    files    = k4._by_role(scaffold)

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader); both big models "
                           "are fp8 and need the comfy quant path" % ID)

    from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(scaffold, "tokenizer"))
    vae = _build_vae(scaffold, files["vae"], ctx.dtype)

    return ctx.backend.load_pipe_comfy(
        Flux2KleinPipeline,
        {"meta": lambda d: k4._transformer_meta(scaffold, d), "file": files["transformer"],
         "convert": _transformer_convert, "quant": True},
        {"meta": lambda d: _text_encoder_meta(scaffold, d), "file": files["text_encoder"],
         "quant": True},
        {"scheduler": scheduler, "tokenizer": tokenizer, "vae": vae, "is_distilled": True},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

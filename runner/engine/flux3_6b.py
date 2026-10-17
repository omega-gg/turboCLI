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

# flux3-6b engine: text2img + img2img on FLUX.3 Image Klein (5.6B, the 16-step distilled
# flow-map checkpoint, no CFG) from BFL's diffusers repo, its weights as published (bf16).
# comfy-flux3-6b is the same model from BFL's int8 ComfyUI files; this engine shares its pipe,
# its speed layer (run with the repo's split q/k/v, as ComfyUI runs such a file) and its LoRA
# key map.
#
# GATED: BFL's repo needs a Hugging Face token from an account that accepted the FLUX Kommunity
# License (install.sh takes it as its last argument).
#
# OFFLOADER-ONLY, as comfy-flux3-6b: the speed layer runs on the offloader's comfy_api().

import glob
import os

from . import comfy_flux3_6b as cf  # cheap: no torch at top level

ID = "flux3-6b"

TYPE        = cf.TYPE
PIPELINE    = cf.PIPELINE
TRANSFORMER = cf.TRANSFORMER

MODES      = cf.MODES
CFG        = cf.CFG
INFERENCE  = cf.INFERENCE
IMAGE_AREA = cf.IMAGE_AREA

# GATED (above): a host asks for the Hugging Face token before it installs this engine.
GATED = True

# The diffusers layout only: the repo also carries BFL's own copies of the transformer and VAE at
# its root, and fp8 / nvfp4 variants, which this engine does not read.
MODEL = {
    "repository": "black-forest-labs",
    "model": "flux-3-image-klein",
    "revision": "337337621b8526e120f2275d250242f4a6bedd75",
    "allow_patterns": [
        "modular_model_index.json",
        "transformer/*",
        "text_encoder/*",
        "vae/*",
    ],
}

# BFL's name for each diffusers key (the repo's root model.safetensors, ComfyUI's naming), the
# inverse of diffusers' convert_flux3_transformer_checkpoint_to_diffusers: _lora_keys reads
# ComfyUI's LoRA names off them.
_BFL_PARTS = (
    (".attn.to_q.", ".q_proj."),
    (".attn.to_k.", ".k_proj."),
    (".attn.to_v.", ".v_proj."),
    (".attn.to_out.0.", ".attn_out."),
    (".attn.norm_q.weight", ".norm.query_norm.scale"),
    (".attn.norm_k.weight", ".norm.key_norm.scale"),
    (".ff.linear_in.", ".mlp_in."),
    (".ff.linear_out.", ".mlp_out."),
    (".linear_1.", ".in_layer."),
    (".linear_2.", ".out_layer."),
)

_BFL_PREFIXES = (
    ("mode_blocks.", "content_mode_blocks."),
    ("single_transformer_blocks.", "single_blocks."),
    ("context_embedder.", "txt_in."),
    ("x_embedders.", "emb_in."),
    ("target_timestep_embedder.", "ref_time_in."),
    ("timestep_embedder.", "time_in."),
    ("norm_out.linear.", "final_layer.image.adaLN_modulation.1."),
    ("proj_out.", "final_layer.image.linear."),
)


def _bfl_key(key):
    for diffusers, bfl in _BFL_PARTS:
        key = key.replace(diffusers, bfl)

    if "_modulations." in key:
        key = key.replace(".linear.", ".lin.")

    for diffusers, bfl in _BFL_PREFIXES:
        if key.startswith(diffusers):
            return bfl + key[len(diffusers):]
    return key


def _transformer_convert(sd, names):
    """The repo's transformer as the speed layer runs it, filling `names` with
    {BFL key: diffusers key} for _lora_keys. Only the final modulation changes: the speed layer
    reads it in BFL's (shift, scale) order, as ComfyUI's LastLayer does, and diffusers' converter
    swapped it (the swap is its own inverse)."""
    from diffusers.loaders.single_file_utils import swap_scale_shift

    names.update({_bfl_key(k): k for k in sd})

    sd["norm_out.linear.weight"] = swap_scale_shift(sd["norm_out.linear.weight"], dim=0)
    return sd


def load(ctx, params):
    """Assemble comfy-flux3-6b's Flux3Pipe from the repo's sharded diffusers weights via the
    disk-stream offloader. OFFLOADER-ONLY. The VAE and the processor are the repo's own."""
    model = ctx.model  # model/flux-3-image-klein

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader); its speed layer "
                           "runs on the offloader's comfy_api()" % ID)

    from diffusers import AutoencoderKLFlux2
    from transformers import Qwen3VLProcessor

    def shards(part):
        return sorted(glob.glob(os.path.join(model, part, "*.safetensors")))

    tokenizer = Qwen3VLProcessor.from_pretrained(os.path.join(model, "text_encoder"))
    vae = AutoencoderKLFlux2.from_pretrained(model, subfolder="vae").to(ctx.dtype).eval()
    comfy = ctx.backend.comfy_api()
    names = {}  # filled by the convert, read by the LoRA key map (the offloader runs it after)

    return ctx.backend.load_pipe_comfy(
        cf.Flux3Pipe,
        {"meta": lambda d: cf._transformer_meta(model, d, comfy, fused_qkv=False),
         "file": shards("transformer"), "convert": lambda sd: _transformer_convert(sd, names),
         "lora_keys": lambda: cf._lora_keys(model, names)},
        {"meta": lambda d: cf._text_encoder_meta(model, d), "file": shards("text_encoder")},
        {"tokenizer": tokenizer, "vae": vae},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

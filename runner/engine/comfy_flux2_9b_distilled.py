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
# and the fp8-mixed Qwen3-8B text encoder (qwen_3_8b_fp8mixed, ~8.7GB), both kept fp8 and
# dequantized per forward as ComfyUI does, plus BFL's small decoder VAE
# (full_encoder_small_decoder: same encoder and latents, a narrower ~28M decoder, ~1.4x faster and
# lighter to decode). Only configs and scheduler come from the diffusers repo (the scaffold).
# comfy-flux2-9b is the bf16 sibling with flux2-9b's weights.
#
# GATED: the transformer and the scaffold sit in black-forest-labs repos that need a Hugging Face
# token from an account that accepted the FLUX Non-Commercial License (install.sh takes it as its
# last argument). The text encoder and VAE repos are open (the VAE is Apache-2.0).
#
# OFFLOADER-ONLY, as comfy-flux2-4b, whose load() this engine runs with its own VAE. Everything
# but the files and the VAE is comfy-flux2-9b's (BASE: same scaffold, same pipeline).

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


def load(ctx, params):
    """comfy-flux2-4b's assembly with the small decoder VAE."""
    return k4.load(ctx, params, build_vae=_build_vae)

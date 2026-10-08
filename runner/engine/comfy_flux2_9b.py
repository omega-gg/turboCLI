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

# comfy-flux2-9b engine: text2img + img2img on FLUX.2-klein-9B, REUSING a ComfyUI install.
#
# The same weights as the flux2-9b engine (same repo, same pinned revision) as ComfyUI single
# files: the bf16 klein 9B transformer (~18.2GB), the bf16 Qwen3-8B text encoder (qwen_3_8b,
# ~16.4GB) and the FLUX.2 VAE shared with comfy-flux2-4b. comfy-flux2-9b-distilled is the lighter
# sibling (ComfyUI's distilled template: fp8 weights and the small decoder VAE).
#
# GATED: the transformer and the scaffold sit in a black-forest-labs repo that needs a Hugging Face
# token from an account that accepted the FLUX Non-Commercial License (install.sh takes it as its
# last argument). The text encoder and VAE repos are open.
#
# Everything but the files is comfy-flux2-4b's (BASE): the 9B transformer is the same plain bf16
# single file layout, and Qwen3-8B's file carries its own untied lm_head, which _tie_lm_head keeps.

from . import comfy_flux2_4b as base

ID   = "comfy-flux2-9b"
BASE = base.ID

COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": "black-forest-labs/FLUX.2-klein-9B",
         "revision": "92196c8e11f7b6cf2b7493e037d8c5345c559216",
         "path": "diffusion_models/flux-2-klein-9b.safetensors",
         "filename": "flux-2-klein-9b.safetensors"},
        {"role": "text_encoder", "repository": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
         "revision": "3f62d9d8ae1fec33c6e91453d5c712855b096b55",
         "path": "text_encoders/qwen_3_8b.safetensors"},
        {"role": "vae", "repository": "Comfy-Org/flux2-dev",
         "path": "vae/flux2-vae.safetensors"},
    ],
}

SCAFFOLD = {
    "repository": "black-forest-labs",
    "model": "FLUX.2-klein-9B",
    "revision": "92196c8e11f7b6cf2b7493e037d8c5345c559216",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
    ],
}

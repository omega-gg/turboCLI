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

# qwen-image-edit-2511-lightning -- the qwen-image-edit-2511 model with the "lightning" 4-step LoRA
# always applied (no angles, no prompt dependence). Same model/pipeline as qwen-image-edit-2511;
# TYPE stays "qwen-image-edit" so the backend seam treats it as a normal qwen-image-edit.
# Standalone by design.

NAME     = "qwen-image-edit-2511-lightning"
TYPE     = "qwen-image-edit"
PIPELINE = "diffusers:QwenImageEditPlusPipeline"
MODES    = ("edit",)
CFG      = ("true_cfg_scale", 1.0)

# Install (python -m runner.install): the model + the lightning LoRA, into the model folder.
# "revision" pins the HF commit (mutable repos -> reproducible installs); check validates it.
MODEL = {"repository": "Qwen", "model": "Qwen-Image-Edit-2511",
         "revision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"}

# LoRA applied on load, found inside the model folder (the same file the install fetches).
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

LORAS = [
    {"repository": "lightx2v/Qwen-Image-Edit-2511-Lightning", "file": LIGHTNING,
     "revision": "d74eba145674fd7e31b949324e148e21e7118abd"},
]


def loras(params):
    # The lightning LoRA at full strength; core resolves it in the model folder and adds ctx.loras.
    return [(LIGHTNING, 1.0)]

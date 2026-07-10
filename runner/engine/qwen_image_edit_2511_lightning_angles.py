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

# qwen-image-edit-2511-lightning-angles -- the qwen-image-edit-2511 model with "lightning"
# (always) + "angles" (only for <sks> prompts). The angles toggle is part of the cache key
# (extra_key), so flipping it reloads the pipe. Same model/pipeline as qwen-image-edit-2511; TYPE
# stays "qwen-image-edit" so the backend seam treats it as a normal qwen-image-edit. Standalone by
# design.

ID   = "qwen-image-edit-2511-lightning-angles"
TYPE = "qwen-image-edit"

PIPELINE    = "diffusers:QwenImageEditPlusPipeline"
TRANSFORMER = "diffusers:QwenImageTransformer2DModel" # offloader disk-stream meta-load

MODES = ("image-to-image",)
CFG   = ("true_cfg_scale", 1.0)

INFERENCE = 4

# Install (python -m runner.install): the model + the lightning and angles LoRAs, into the model
# folder.
# "revision" pins the HF commit (mutable repos -> reproducible installs); check validates it.
MODEL = {"repository": "Qwen", "model": "Qwen-Image-Edit-2511",
         "revision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"}

# LoRAs applied on load, found inside the model folder (the same files the install fetches).
# "angles" only for <sks> prompts.
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
ANGLES    = "qwen-image-edit-2511-multiple-angles-lora.safetensors"

LORAS = [
    {"repository": "lightx2v/Qwen-Image-Edit-2511-Lightning", "file": LIGHTNING,
     "revision": "d74eba145674fd7e31b949324e148e21e7118abd"},
    {"repository": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", "file": ANGLES,
     "revision": "e3066224ab74263f4a5b6179cd1a3b0a15577e44"},
]


def extra_key(params):
    # Whether angles is active is part of the pipe identity, so flipping it forces a reload.
    return (params["prompt"].startswith("<sks>"),)


def loras(params):
    # Lightning always; angles (@0.9) only for <sks> prompts. Core resolves paths and adds
    # ctx.loras.
    files = [(LIGHTNING, 1.0)]

    if params["prompt"].startswith("<sks>"):
        files.append((ANGLES, 0.9))

    return files

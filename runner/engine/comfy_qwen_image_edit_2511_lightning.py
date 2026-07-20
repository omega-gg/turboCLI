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

# comfy-qwen-image-edit-2511-lightning -- comfy-qwen-image-edit-2511 with the "lightning" 4-step
# LoRA always applied (mirrors the stock qwen-image-edit-2511-lightning, every weight reused from a
# ComfyUI install). Inherits the whole engine from the comfy base via BASE -- the fp8 quant text
# encoder, the WAN-VAE reuse, the six meta-builders, and load(). The only delta is one more reused
# ComfyUI file: the lightning LoRA in models/loras/, added as a 4th COMFY component. The base
# load() applies any "lora" role component to the transformer at full strength (see
# comfy_qwen_image_edit_2511.load), so no load() override is needed here.

from . import comfy_qwen_image_edit_2511 as base  # cheap: base imports no torch at top level

ID   = "comfy-qwen-image-edit-2511-lightning"
BASE = base.ID

INFERENCE = 4  # the lightning LoRA's design step count (vs the base engine's 40)

# The lightning LoRA reused from ComfyUI's models/loras/. It sits in a plain HF repo -- the file is
# at the repo root, not under split_files/ -- so this component carries an explicit `filename` (the
# in-repo path) distinct from its `path` (the models/-relative destination). Revision pins the same
# commit the stock qwen-image-edit-2511-lightning uses.
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# Extend the base's three reused single files with the lightning LoRA (4th component). The base
# load() applies the "lora" role to the streamed transformer. dict(base.COMFY, ...) is a shallow
# copy that overrides `components` while inheriting `revision` -- it never mutates base.COMFY.
COMFY = dict(base.COMFY, components=base.COMFY["components"] + [
    {"role": "lora", "repository": "lightx2v/Qwen-Image-Edit-2511-Lightning",
     "revision": "d74eba145674fd7e31b949324e148e21e7118abd",
     "filename": LIGHTNING, "path": "loras/" + LIGHTNING},
])

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

# comfy-qwen-image-2-1-turbo: comfy-qwen-image-2-1 on Qwen-Image-2.1-Turbo, Qwen's 8-step
# distillation of the same model (CFG 1). Only the transformer weights differ: the text encoder,
# the VAE and the scaffold (the transformer config is identical) are the base's. Inherits the whole
# engine via BASE; the deltas are Comfy-Org's int8 ConvRot Turbo file in place of the base
# transformer, and 8 steps. The schedule stays the base's: ComfyUI samples Qwen-Image 2.1 at its
# model's shift 0.69 with "simple" (supported_models.py), Turbo weights included.

from . import comfy_qwen_image_2_1 as base  # cheap: base imports no torch at top level

ID   = "comfy-qwen-image-2-1-turbo"
BASE = base.ID

INFERENCE = 8  # Turbo's step count (Comfy-Org's README)

TURBO = "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"

# The base's three files with the transformer swapped for Turbo's, in the same Comfy-Org repo at
# the commit that added it. dict(base.COMFY, ...) never mutates base.COMFY.
COMFY = dict(base.COMFY, components=[
    dict(c, revision="df94239739eef7205973a145ffa6f441f7b64e84", path=TURBO, filename=TURBO)
    if c["role"] == "transformer" else c
    for c in base.COMFY["components"]
])

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

# comfy-z-image-turbo-upscale engine: ComfyUI's Z-Image Turbo upscale blueprint on the
# comfy-z-image-turbo weights. comfy-esrgan (UPSCALER) brings the input to `scale` on the 16 grid,
# then a short img2img pass redraws the detail: strength 0.33, 5 steps, ComfyUI's "masterpiece, 8k"
# when no prompt is given (the chain of z_image_turbo_upscale.py, on ComfyUI's files).

from . import comfy_z_image_turbo as base  # cheap: no torch at top level

ID   = "comfy-z-image-turbo-upscale"
BASE = base.ID

PIPELINE = "diffusers:ZImageImg2ImgPipeline"

MODES = ("image-upscale",)

INFERENCE = 5     # the steps that run: core asks INFERENCE / STRENGTH and the strength cuts it
STRENGTH  = 0.33
PROMPT    = "masterpiece, 8k"

# Installed first, required by check, run by core before the img2img pass.
UPSCALER = "comfy-esrgan"


def load(ctx, params):
    return base._build(ctx, base._pipeline_class("ZImageImg2ImgPipeline"))

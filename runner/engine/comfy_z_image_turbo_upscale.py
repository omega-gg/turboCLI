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

# comfy-z-image-turbo-upscale engine: z-image-turbo-upscale on ComfyUI's files, the
# comfy-z-image-turbo weights with comfy-esrgan as UPSCALER. Same chain (see
# z_image_turbo_upscale.py).

from . import comfy_z_image_turbo as base  # cheap: no torch at top level
from . import z_image_turbo_upscale as zu

ID   = "comfy-z-image-turbo-upscale"
BASE = base.ID

PIPELINE = zu.PIPELINE
MODES    = zu.MODES

INFERENCE = zu.INFERENCE
STRENGTH  = zu.STRENGTH
PROMPT    = zu.PROMPT

UPSCALER = "comfy-esrgan"


def load(ctx, params):
    return base._build(ctx, base._pipeline_class("ZImageImg2ImgPipeline"))

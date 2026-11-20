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

# comfy-seedvr2-7b-upscale engine: comfy-seedvr2-3b-upscale with SeedVR2 7B, as ComfyUI's
# utility_seedvr2_7b_int8_upscale_image template runs it: the 7B int8 ConvRot transformer and the
# same fp16 VAE (8.84 GB). Apache-2.0.

from . import comfy_seedvr2_3b_upscale as base  # cheap: no torch at top level

ID   = "comfy-seedvr2-7b-upscale"
BASE = base.ID

COMFY = dict(base.COMFY, components=[
    {"role": "transformer", "path": "diffusion_models/seedvr2_7b_int8_convrot.safetensors",
     "filename": "diffusion_models/seedvr2_7b_int8_convrot.safetensors"},
    base.COMFY["components"][1],
])

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

# comfy-esrgan-upscale engine: the esrgan-upscale engine on ComfyUI's Real-ESRGAN x4plus file,
# reused from (or downloaded into) ComfyUI's models/upscale_models. esrgan-upscale's run, whose
# weights() reads this engine's engine.json.
#
# Install (runner.install --comfy <ComfyUI dir>): one COMFY component, no scaffold (spandrel reads
# the file as is).

from . import esrgan_upscale as base  # cheap: no torch at top level

ID    = "comfy-esrgan-upscale"
MODES = base.MODES

UPSCALE = base.UPSCALE

run     = base.run
weights = base.weights

COMFY = {
    "components": [
        {"role": "upscaler", "repository": "Comfy-Org/Real-ESRGAN",
         "revision": "5fd49b7b278836f48af63ecd314d0f98ab336105",
         "path": "upscale_models/RealESRGAN_x4plus.safetensors",
         "filename": "RealESRGAN_x4plus.safetensors"},
    ],
}

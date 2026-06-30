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

# z-image-turbo engine -- text2img only on Z-Image-Turbo. Pure declaration; core's default load +
# run path cover it.

NAME     = "z-image-turbo"
TYPE     = "z-image"
PIPELINE = "diffusers:ZImagePipeline"
MODES    = ("generate",)
CFG      = ("guidance_scale", 0.0)

# Install (python -m runner.install): base model HF repo = "<repository>/<model>". No LoRAs.
# "revision" pins the HF commit (mutable repos -> reproducible installs); check validates it.
MODEL    = {"repository": "Tongyi-MAI", "model": "Z-Image-Turbo",
            "revision": "04cc4abb7c5069926f75c9bfde9ef43d49423021"}

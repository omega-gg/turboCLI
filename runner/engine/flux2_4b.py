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

# flux2-4b engine -- text2img + img2img on FLUX.2-klein-4B. Pure declaration: core's default load
# (from_pretrained + apply_offload) and run path cover it, so there is no code here. See
# PLAN-engine.md.

NAME     = "flux2-4b"
TYPE     = "flux2"
PIPELINE = "diffusers:Flux2KleinPipeline"   # resolved lazily by core on first load
MODES    = ("generate", "edit")             # wire values, same as the HTTP API
CFG      = ("guidance_scale", 0.0)

# Install (python -m runner.install): base model HF repo = "<repository>/<model>". No LoRAs.
MODEL    = {"repository": "black-forest-labs", "model": "FLUX.2-klein-4B"}

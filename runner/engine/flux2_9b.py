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

# flux2-9b engine: text2img + img2img on FLUX.2-klein-9B. Same pipeline as flux2-4b (inherited via
# BASE), only the model differs: a 9B transformer and a Qwen3-8B text encoder (~35 GB bf16).
# GATED: the repo needs a Hugging Face token from an account that accepted the FLUX Non-Commercial
# License (install.sh takes it as its last argument).

from . import flux2_4b as base

ID   = "flux2-9b"
BASE = base.ID

MODEL = {"repository": "black-forest-labs", "model": "FLUX.2-klein-9B",
         "revision": "92196c8e11f7b6cf2b7493e037d8c5345c559216"}

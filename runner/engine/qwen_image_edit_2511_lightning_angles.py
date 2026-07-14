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

# qwen-image-edit-2511-lightning-angles -- the lightning variant + "angles" (only for <sks>
# prompts). The angles toggle is part of the cache key (extra_key), so flipping it reloads the
# pipe. Inherits from qwen-image-edit-2511-lightning via BASE (TYPE / PIPELINE / TRANSFORMER /
# MODES / CFG / MODEL / INFERENCE); declares only the angles delta.

from . import qwen_image_edit_2511_lightning as base

ID   = "qwen-image-edit-2511-lightning-angles"
BASE = base.ID

# lightning (inherited) + angles. "angles" only applied for <sks> prompts (see loras()).
LIGHTNING = base.LIGHTNING
ANGLES    = "qwen-image-edit-2511-multiple-angles-lora.safetensors"

# Extend the base's install list with the angles LoRA (so LORAS is declared here, not inherited).
LORAS = base.LORAS + [
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

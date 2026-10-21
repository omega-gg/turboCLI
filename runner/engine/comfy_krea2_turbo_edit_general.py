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

# comfy-krea2-turbo-edit-general: comfy-krea2-turbo-edit-identity with conradlocke's Krea 2
# General Edit V2 LoRA (free for personal and non-commercial use) instead, for edits from a plain
# instruction: remove, recolor, relight, swap a background, add an object, rewrite text, change an
# outfit, restyle. Same conditioning, at its presets: REF_BOOST 4 and the reference aligned, so
# what the instruction does not touch keeps its place.

from . import comfy_krea2_turbo_edit_identity as base  # cheap: base imports no torch at top level

ID   = "comfy-krea2-turbo-edit-general"
BASE = base.ID

GATED = True  # the base's scaffold repo; a variant does not inherit GATED

LORA = "krea2_general_edit_v2.safetensors"

# The node pack's General Edit presets.
REF_BOOST    = 4.0
REF_POSITION = "aligned"

# The base's files with this LoRA in place of the identity one, same repo and revision.
COMFY = dict(base.COMFY, components=[
    dict(c, filename=LORA, path="loras/" + LORA) if c["role"] == "lora" else c
    for c in base.COMFY["components"]
])


def load(ctx, params):
    """The identity engine's assembly at the General Edit presets."""
    return base.load(ctx, params, REF_BOOST, REF_POSITION)

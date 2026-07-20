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

# comfy-krea2-turbo-realism -- comfy-krea2-turbo with the Krea2-realism-V2 LoRA always applied at
# 1.5. Inherits the whole engine (TYPE / PIPELINE / TRANSFORMER / MODES / CFG / INFERENCE / COMFY /
# SCAFFOLD, the meta-builders and the streaming load) from the base via BASE. The only delta:
# load() prepends the bundled realism LoRA to ctx.loras, so the base assembly streams it onto the
# fp8 transformer exactly like a user --loras entry (256 patches at weight 1.5). Its registry dir
# engine/comfy-krea2-turbo-realism/ clones the base install's engine.json + scaffold and bundles
# the LoRA under lora/.

import os

from . import comfy_krea2_turbo as base  # cheap: base imports no torch at top level

ID   = "comfy-krea2-turbo-realism"
BASE = base.ID

# The bundled LoRA (LoKr; every Krea2 key naming loads via the base's _lora_keys map) and the
# author's recommended default strength.
REALISM  = "Krea2-realism-V2.safetensors"
STRENGTH = 1.5


def load(ctx, params):
    """Reuse the base Krea2 assembly, but always stream the realism LoRA first at STRENGTH. It is
    prepended to ctx.loras (any user --loras still stack after), then base.load hands the merged
    list to load_pipe_comfy, applied on-cast onto the fp8 transformer."""
    lora = os.path.join(ctx.model, "lora", REALISM)
    ctx.loras = [(lora, STRENGTH)] + list(ctx.loras)

    return base.load(ctx, params)

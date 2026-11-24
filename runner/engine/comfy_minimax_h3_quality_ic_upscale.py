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

# comfy-minimax-h3-quality-ic-upscale engine: comfy-minimax-h3-ic-upscale on
# comfy-minimax-h3-quality's files, for its takes: the full bf16 ref2va, the text encoder in bf16.

from . import comfy_minimax_h3 as h3  # cheap: no torch at top level
from . import comfy_minimax_h3_ic_upscale as ic
from . import comfy_minimax_h3_quality as base

ID   = "comfy-minimax-h3-quality-ic-upscale"
BASE = base.ID

MODES     = ic.MODES
UPSCALE   = ic.UPSCALE
INFERENCE = ic.INFERENCE
CRF       = ic.CRF
OPTIONS   = ic.OPTIONS

COMFY = dict(base.COMFY, components=ic.components(
    base.COMFY["components"], "diffusion_models/minimax_h3_ref2va_bf16.safetensors"))

extra_key = ic.extra_key
check     = ic.check


def load(ctx, params):
    return h3.load(ctx, params, text_dtype="bfloat16", ic=dict(ic.IC, sharp=ic._sharp(params)))

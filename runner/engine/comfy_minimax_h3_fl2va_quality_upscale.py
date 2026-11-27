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

# comfy-minimax-h3-fl2va-quality-upscale engine: comfy-minimax-h3-upscale on
# comfy-minimax-h3-fl2va-quality's files, for its takes: refined on the fl2va transformer,
# references included.

from . import comfy_minimax_h3_fl2va_quality as base  # cheap: no torch at top level
from . import comfy_minimax_h3_upscale as upscale

ID   = "comfy-minimax-h3-fl2va-quality-upscale"
BASE = base.ID

MODES     = upscale.MODES
INFERENCE = upscale.INFERENCE
UPSCALE   = upscale.UPSCALE

COMFY = dict(base.COMFY, components=base.COMFY["components"] + [upscale.UPSCALER])

check = upscale.check

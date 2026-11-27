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

# comfy-minimax-h3-fl2va-quality engine: comfy-minimax-h3-quality with its references on the
# fl2va transformer, its one transformer: the same files but ref2va's and the same graphs (with
# references MiniMaxH3ReferenceToVideo, the frames pinned by MiniMaxH3AddGuide, beta). Not
# MiniMax's split, which runs references on ref2va: crisper takes, whose pinned first frame no
# longer stands out, but glitchier. comfy-minimax-h3-quality stays the default with references:
# judge a take on both by eye.

from . import comfy_minimax_h3_quality as base  # cheap: no torch at top level

ID   = "comfy-minimax-h3-fl2va-quality"
BASE = base.ID

# NOTE: Without transformer_ref, comfy_minimax_h3.load runs references on the transformer.
COMFY = dict(base.COMFY, components=[c for c in base.COMFY["components"]
                                     if c["role"] != "transformer_ref"])

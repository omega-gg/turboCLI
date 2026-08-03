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

# mask-lucida engine -- a subject matte via Lucida (egeorcun/lucida, a BiRefNet fine-tune for
# glass / camouflage / text / print). BASE = mask-birefnet: MODES + run() are inherited (run reads
# ctx.model, so it loads model/lucida); only the MODEL (a different snapshot repo) differs.

from . import mask_birefnet as base

ID   = "mask-lucida"
BASE = base.ID

MODEL = {"kind": "snapshot", "repository": "egeorcun", "model": "lucida",
         "revision": "6ee11122534c8de59402a589d2293c198cfbf848"}

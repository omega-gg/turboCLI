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

# Per-engine modules. Each file declares ID / PIPELINE / MODES / CFG (and optional extra_key() /
# loras() / load() hooks) and is auto-discovered by core. IMPORTANT: an engine module must NOT
# import diffusers/torch at top level -- name the pipeline by string and import lazily inside
# hooks, so discovery stays cheap and an unused engine costs nothing.
#
# INHERITANCE: a variant may declare `BASE = "<base engine ID>"` and write only its delta -- it
# inherits every contract symbol it does not itself define (TYPE / PIPELINE / TRANSFORMER / MODES /
# CFG / INFERENCE / MODEL / COMFY / SCAFFOLD / LORAS / load / loras / extra_key) from the base,
# folded once at discovery (see _inherit.py; zero runtime cost). An inherited load()/loras() keeps
# the base module's globals, so it still resolves the base's private helpers. To EXTEND a value
# (rather than inherit/replace), import the base module (cheap -- no torch at top) and compute it:
#   from . import qwen_image_edit_2511_lightning as base
#   BASE  = base.ID
#   LORAS = base.LORAS + [ ... ]                       # or: dict(base.COMFY, components=...)
# See qwen_image_edit_2511_lightning[_angles].py and comfy_qwen_image_edit_2511_lightning.py.

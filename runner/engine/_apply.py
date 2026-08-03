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

# Mask APPLICATION for the mask-apply engine. Torch-free (PIL only), a helper (underscore) so
# discovery skips it. Takes a precomputed mask (from a mask / mask-* engine) and lays it onto an
# image two ways:
#   composite  paste the source's masked region onto a reference (the original scene, or a new
#              backdrop); byte-exact reference where the mask is black
#   putalpha   write the mask as source's alpha channel -> an RGBA cutout, transparent elsewhere

from PIL import Image

LR = Image.Resampling.LANCZOS


def apply(source, mask, mode, reference=None):
    """Apply `mask` to `source`. `composite` upscales source + mask to the reference res and pastes
    the masked region onto the reference. `putalpha` writes the mask as source's alpha (a cutout);
    the mask is resized to source if needed."""
    if mode == "composite":
        up = source.resize(reference.size, LR)
        m  = mask.resize(reference.size, LR)

        return Image.composite(up, reference, m)

    out = source.copy()

    if mask.size != out.size:
        mask = mask.resize(out.size, LR)

    out.putalpha(mask)                                     # promotes RGB -> RGBA

    return out

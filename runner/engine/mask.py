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

# mask engine -- a soft pixel-diff mask of where an edit differs from its reference, best for
# adding an object / recoloring. A "compute" engine: core's run() seam calls run() directly (no
# diffusion, no offloader, torch-free). images = "reference,input"; options threshold=N (default
# 24). Apply the mask with image-mask-apply.

ID    = "mask"
MODES = ("image-to-mask",)


def run(ctx, params, emit):
    import numpy as np
    from PIL import Image

    from ._mask import build_mask, THR
    from ._options import parse_options

    thr  = int(parse_options(params.get("options", "")).get("threshold", THR))
    imgs = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

    if len(imgs) < 2:
        raise ValueError("mask needs images=reference,input")

    ref  = Image.open(imgs[0]).convert("RGB")
    edit = Image.open(imgs[1]).convert("RGB")

    mask = build_mask(ref, edit, "mask", thr)

    emit("mask[mask thr=%d]: masked %.0f%%" % (thr, np.asarray(mask).mean() / 255 * 100))

    return mask

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

# mask engine -- a mask of where an edit differs from its reference. A "compute" engine: core's
# run() seam calls it directly (no diffusion, torch-free). images = "input,reference".
# options: mode=default (soft pixel-diff, best for adding / recoloring) | region (grown boxes, best
# for removal / replace); tolerance=N (0-255, more = more pixels kept, default 231). Apply the mask
# with image-apply-mask.

ID    = "mask"
MODES = ("image-to-mask",)


def run(ctx, params, emit):
    import numpy as np
    from PIL import Image

    from ._mask import build_mask, TOLERANCE
    from ._options import parse_options

    opts = parse_options(params.get("options", ""))
    sub  = opts.get("mode", "default")
    tol  = int(opts.get("tolerance", TOLERANCE))
    imgs = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

    if sub not in ("default", "region"):
        raise ValueError("mode must be default or region")

    if len(imgs) < 2:
        raise ValueError("mask needs images=input,reference")

    edit = Image.open(imgs[0]).convert("RGB")
    ref  = Image.open(imgs[1]).convert("RGB")

    mask = build_mask(ref, edit, "region" if sub == "region" else "mask", 255 - tol)

    emit("mask[%s tol=%d]: masked %.0f%%" % (sub, tol, np.asarray(mask).mean() / 255 * 100))

    return mask

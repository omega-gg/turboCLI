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
# for removal / replace); cutoff=N (0-255, higher = fewer pixels, default 24); glow=N (pixels,
# default 0, mode=default only) reaches that far past the change, faded, which covers the outline a
# removal leaves behind; fade=<sides> (any of "ltrb", none by default, mode=region only) softens
# the mask on the borders the caller cut, for a mask made of part of a picture. Apply the mask
# with image-apply-mask.

ID    = "mask"
MODES = ("image-to-mask",)


def run(ctx, params, emit):
    import numpy as np
    from PIL import Image

    from ._mask import build_mask, CUTOFF
    from ._options import parse_options

    opts = parse_options(params.get("options", ""))
    sub  = opts.get("mode", "default")
    cut  = int(opts.get("cutoff", CUTOFF))
    glow = int(opts.get("glow", 0))
    fade = opts.get("fade", "")
    imgs = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

    if sub not in ("default", "region"):
        raise ValueError("mode must be default or region")

    if set(fade) - set("ltrb"):
        raise ValueError("fade takes the sides to soften, any of ltrb")

    if len(imgs) < 2:
        raise ValueError("mask needs images=input,reference")

    edit = Image.open(imgs[0]).convert("RGB")
    ref  = Image.open(imgs[1]).convert("RGB")

    mask = build_mask(ref, edit, "region" if sub == "region" else "mask", cut, glow, fade)

    emit("mask[%s cut=%d glow=%d]: masked %.0f%%"
         % (sub, cut, glow, np.asarray(mask).mean() / 255 * 100))

    return mask

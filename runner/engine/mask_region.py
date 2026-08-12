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

# mask-region engine -- grown bounding boxes around changed blobs, best for removal / replace
# (ghost-free, where a diff mask would leave an outline). A "compute" engine, torch-free. Same as
# `mask` but the region mode; a separate module because the mode literal cannot be inherited.
# images = "input,reference"; options threshold=N (default 24). Apply with image-apply-mask.

ID    = "mask-region"
MODES = ("image-to-mask",)


def run(ctx, params, emit):
    import numpy as np
    from PIL import Image

    from ._mask import build_mask, THR
    from ._options import parse_options

    thr  = int(parse_options(params.get("options", "")).get("threshold", THR))
    imgs = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

    if len(imgs) < 2:
        raise ValueError("mask-region needs images=input,reference")

    edit = Image.open(imgs[0]).convert("RGB")
    ref  = Image.open(imgs[1]).convert("RGB")

    mask = build_mask(ref, edit, "region", thr)

    emit("mask[region thr=%d]: masked %.0f%%" % (thr, np.asarray(mask).mean() / 255 * 100))

    return mask

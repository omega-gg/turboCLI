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

# mask-apply engine -- lay a precomputed mask (from any mask / mask-* engine) onto an image. A
# "compute" engine, torch-free. options mode=composite|putalpha|trim (default composite). composite
# pastes the source's masked region onto a reference (images = "input,mask,reference"); putalpha
# writes the mask as the source's alpha -> an RGBA cutout (images = "input,mask"). trim takes no
# mask: it crops the input to what shows, options pad=N the most transparency it keeps around
# (pixels, 32 by default) and speck=N the size under which a separate blob is left out (pixels,
# 0 by default: off), images = "input".

ID    = "mask-apply"
MODES = ("image-apply-mask",)


def run(ctx, params, emit):
    from PIL import Image, ImageStat

    from ._apply import TRIM_PAD, apply, trim
    from ._options import parse_options

    options = parse_options(params.get("options", ""))

    mode = options.get("mode", "composite")
    imgs = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

    if mode not in ("composite", "putalpha", "trim"):
        raise ValueError("mode must be composite, putalpha or trim")

    if mode == "trim":
        if len(imgs) < 1:
            raise ValueError("trim needs images=input")

        out = trim(Image.open(imgs[0]), int(options.get("pad", TRIM_PAD)),
                   int(options.get("speck", 0)))

        emit("apply[trim]: %dx%d" % out.size)

        return out

    if len(imgs) < 2:
        raise ValueError("mask-apply needs images=input,mask[,reference]")

    source = Image.open(imgs[0]).convert("RGB")
    mask   = Image.open(imgs[1]).convert("L")

    reference = None

    if mode == "composite":
        if len(imgs) < 3:
            raise ValueError("composite requires images=input,mask,reference")

        reference = Image.open(imgs[2]).convert("RGB")

    out = apply(source, mask, mode, reference)

    emit("apply[%s]: mask %.0f%%" % (mode, ImageStat.Stat(mask).mean[0] / 255 * 100))

    return out

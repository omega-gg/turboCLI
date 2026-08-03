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
# "compute" engine, torch-free. options mode=composite|putalpha (default composite). composite
# pastes the source's masked region onto a reference (images = "input,mask,reference"); putalpha
# writes the mask as the source's alpha -> an RGBA cutout (images = "input,mask").

ID    = "mask-apply"
MODES = ("image-mask-apply",)


def run(ctx, params, emit):
    from PIL import Image, ImageStat

    from ._apply import apply
    from ._options import parse_options

    mode = parse_options(params.get("options", "")).get("mode", "composite")
    imgs = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

    if mode not in ("composite", "putalpha"):
        raise ValueError("mode must be composite or putalpha")

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

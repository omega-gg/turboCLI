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

# mask-inspyrenet engine -- a subject matte via InSPyReNet (transparent-background). A "compute"
# engine; the transparent_background stack loads only inside run(). images = "input[,plate]";
# options threshold=N (shadow floor, default 12). Apply the matte with image-apply-mask.
#
# Install (python -m runner.install): kind "url" -> the checkpoint (a GitHub release asset) into
# model/inspyrenet/ckpt_base.pth. revision is the release tag (1.2.12), not an HF commit.

ID    = "mask-inspyrenet"
MODES = ("image-to-mask",)

MODEL = {"kind": "url", "model": "inspyrenet", "revision": "1.2.12", "file": "ckpt_base.pth",
         "url": "https://github.com/plemeri/transparent-background/releases/download/1.2.12/"
                "ckpt_base.pth"}


def run(ctx, params, emit):
    import os
    from PIL import Image, ImageStat

    from ._segment import pick_device, inspyrenet_alpha, build_matte, SHAD_THR
    from ._options import parse_options

    thr  = float(parse_options(params.get("options", "")).get("threshold", SHAD_THR))
    imgs = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

    if not imgs:
        raise ValueError("mask-inspyrenet needs images=input[,plate]")

    device = pick_device(ctx.renderer)
    image  = Image.open(imgs[0]).convert("RGB")
    plate  = Image.open(imgs[1]) if len(imgs) > 1 else None

    ckpt  = os.path.join(ctx.model, "ckpt_base.pth")
    matte = build_matte(image, inspyrenet_alpha(image, device, ckpt), plate, thr)

    pct = ImageStat.Stat(matte).mean[0] / 255 * 100
    emit("matte[inspyrenet/%s]: alpha %.0f%%" % (device, pct))

    return matte

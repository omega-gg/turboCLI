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

# mask-birefnet engine -- a subject matte via BiRefNet (ZhengPeng7/BiRefNet). A "compute" engine:
# core's run() seam calls run() directly (no diffusion, no offloader). torch/transformers load only
# inside run(). images = "input[,plate]" (a plate keeps the cast shadow); options cutoff=N
# (0-255, the plate-shadow floor -- higher = less shadow, default 12; inert without a plate). Apply
# the matte with image-apply-mask.
#
# Install (python -m runner.install): kind "snapshot" -> the whole HF repo verbatim (weights + the
# trust_remote_code birefnet.py) into model/<model>. revision pins the HF commit.

ID    = "mask-birefnet"
MODES = ("image-to-mask",)

MODEL = {"kind": "snapshot", "repository": "ZhengPeng7", "model": "BiRefNet",
         "revision": "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"}


def run(ctx, params, emit):
    import os
    from PIL import Image, ImageStat

    from ._segment import pick_device, birefnet_alpha, build_matte, SHADOW_CUTOFF
    from ._options import parse_options

    thr  = int(parse_options(params.get("options", "")).get("cutoff", SHADOW_CUTOFF))
    imgs = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

    if not imgs:
        raise ValueError("mask-birefnet needs images=input[,plate]")

    device = pick_device(ctx.renderer)
    image  = Image.open(imgs[0]).convert("RGB")
    plate  = Image.open(imgs[1]) if len(imgs) > 1 else None

    matte = build_matte(image, birefnet_alpha(image, device, ctx.model), plate, thr)

    emit("matte[%s/%s]: alpha %.0f%%"
         % (os.path.basename(ctx.model), device, ImageStat.Stat(matte).mean[0] / 255 * 100))

    return matte

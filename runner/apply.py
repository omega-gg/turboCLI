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

# Standalone mask APPLIER for image-mask-apply: take a precomputed mask (from image-mask or
# image-mask-background) and lay it onto an image. NOT generation -- no engine, no torch (only
# PIL), so it starts instantly and needs no GPU. Two modes:
#   composite  paste the source's masked region onto a reference (both upscaled to the reference
#              res, mask included). Restores the original scene outside an edit, or drops a subject
#              onto a new backdrop. Byte-exact reference where the mask is black.
#   putalpha   write the mask as the source's alpha channel -> an RGBA cutout, transparent where
#              the mask is black. No reference: the background is transparency.
#
# The mask is an 8-bit grayscale image; both generators emit that, so either feeds either mode.
#
# Run as: python -m runner.apply --mode composite --input edit.png --mask mask.png --output out.png
#         --reference orig.png        (putalpha: drop --reference)

import sys
import argparse
import traceback

from PIL import Image, ImageStat

LR = Image.Resampling.LANCZOS


def apply(source, mask, mode, reference=None):
    """Apply `mask` to `source`. `composite` upscales source + mask to the reference res and pastes
    the masked region onto the reference (reproducing image-mask's old fused merge). `putalpha`
    writes the mask as source's alpha (a cutout); the mask is resized to source if needed."""
    if mode == "composite":
        up = source.resize(reference.size, LR)
        m  = mask.resize(reference.size, LR)

        return Image.composite(up, reference, m)

    out = source.copy()

    if mask.size != out.size:
        mask = mask.resize(out.size, LR)

    out.putalpha(mask)                                     # promotes RGB -> RGBA

    return out


def main():
    p = argparse.ArgumentParser(prog="runner.apply")

    p.add_argument("--mode",      required=True)           # composite | putalpha
    p.add_argument("--input",     required=True)           # the source image the mask is for
    p.add_argument("--mask",      required=True)           # 8-bit grayscale mask / matte
    p.add_argument("--output",    required=True)
    p.add_argument("--reference", default=None)            # base canvas, required for composite

    args = p.parse_args()

    try:
        if args.mode not in ("composite", "putalpha"):
            raise ValueError("mode must be composite or putalpha")

        if args.mode == "composite" and not args.reference:
            raise ValueError("composite requires --reference")

        source = Image.open(args.input).convert("RGB")
        mask   = Image.open(args.mask).convert("L")

        reference = Image.open(args.reference).convert("RGB") if args.reference else None

        result = apply(source, mask, args.mode, reference)
        result.save(args.output)

        pct = ImageStat.Stat(mask).mean[0] / 255 * 100
        print("apply[%s]: mask %.0f%%" % (args.mode, pct), flush=True)
        print("Saved: %s" % args.output, flush=True)
    except Exception:
        print("ERROR: " + traceback.format_exc(), flush=True)

        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

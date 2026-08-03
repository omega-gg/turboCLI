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

# Diff/region mask GENERATION for the mask + mask-region engines. Torch-free (PIL + numpy only), a
# helper (underscore) so discovery skips it. `build_mask` emits a soft [0..255] `L` mask of where
# an edit differs from its reference; image-apply-mask then composites or cuts it out.
#
# The edit pipeline redraws + VAE-decodes the whole frame, so every pixel drifts; this diffs the
# edit against the reference and keeps only where it really changed. Two modes:
#   mask   soft pixel-diff mask -- tight, best for adding an object / recoloring
#   region grown bounding boxes -- ghost-free, best for removal / replace (a diff mask leaves a
#          removed object's low-contrast edges behind as an outline; the box replaces the whole
#          footprint wholesale)

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LR = Image.Resampling.LANCZOS

# THR is the default change threshold (options threshold=N): a pixel differing from the reference
# by more than THR is changed (kept from the input). Higher -> tighter, keeps more reference;
# lower -> keeps more, including whole-frame edit drift. MIN_AREA + the per-mode
# margins below stay fixed (not exposed).
THR, MIN_AREA          = 24, 400
DILATE, FEATHER_MASK   = 5, 3
GROW,   FEATHER_REGION = 60, 18


def _diff_mask(generated, ref, thr, dilate, feather):
    """Soft [0..255] mask (255 = changed) of where `generated` differs from `ref` beyond `thr`.
    Closes interior holes so a flat drawn object stays solid, dilates a margin for soft edges /
    contact shadows, then feathers the seam."""
    g = np.asarray(generated, dtype=np.int16)
    o = np.asarray(ref,       dtype=np.int16)

    # max abs diff over channels, so a chroma-only shift still registers as a change.
    diff = np.abs(g - o).max(axis=2).astype(np.uint8)

    mask = Image.fromarray(np.where(diff > thr, 255, 0).astype(np.uint8), "L")

    # open (despeckle): a star or highlight the model redrew a hair off is a tiny high-contrast
    # blob that trips the threshold; drop those so they come from the reference, not the edit.
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))

    mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))  # close holes

    if dilate >= 3:
        mask = mask.filter(ImageFilter.MaxFilter(dilate if dilate % 2 else dilate + 1))

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    return mask


def _blob_boxes(mask, min_area):
    """(x0,y0,x1,y1) bounding boxes of 4-connected blobs in a binary L mask; drop < min_area.
    Iterative flood fill (no scipy). Ported from merge_region.py `boxes`, single-threshold."""
    a = np.asarray(mask) > 127

    seen = np.zeros(a.shape, bool)
    h, w = a.shape
    out  = []

    for y0, x0 in zip(*np.nonzero(a)):

        if seen[y0, x0]:
            continue

        stack, pix = [(y0, x0)], []
        seen[y0, x0] = True

        while stack:                                       # 4-connected flood, iterative
            y, x = stack.pop()
            pix.append((y, x))

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))

        if len(pix) < min_area:
            continue

        p = np.array(pix)
        out.append((p[:, 1].min(), p[:, 0].min(), p[:, 1].max(), p[:, 0].max()))

    return out


def _region_mask(generated, ref, thr, grow, feather, min_area):
    """Solid grown bounding boxes around changed blobs (255 = replace). For removal/replace: the
    box is filled solid so nothing of the old object can ghost inside it. A pixel diff mask can't
    do this -- a dark object over a matching background falls below the threshold and survives as
    an outline; the box replaces the whole footprint wholesale. See merge_region.py."""
    g = np.asarray(generated, np.int16)
    o = np.asarray(ref,       np.int16)

    diff = np.abs(g - o).max(2).astype(np.uint8)           # max over channels: chroma counts too

    m = Image.fromarray(np.where(diff > thr, 255, 0).astype(np.uint8), "L")

    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))   # open: despeckle
    m = m.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))   # close: holes

    out = Image.new("L", generated.size, 0)
    d   = ImageDraw.Draw(out)

    # grow: the box comes from the thresholded blob, which stops at the object's dark edges and so
    # lands inside its true extent; growing the rectangle clears the clipped edge.
    for x0, y0, x1, y1 in _blob_boxes(m, min_area):
        d.rectangle((x0 - grow, y0 - grow, x1 + grow, y1 + grow), fill=255)

    if feather > 0:
        out = out.filter(ImageFilter.GaussianBlur(feather))

    return out


def build_mask(reference, edit, mode, thr=THR):
    """Soft [0..255] `L` mask (255 = changed) of where `edit` differs from `reference`, at the
    edit's own resolution. The reference is downscaled to the edit's canvas first, so a full-res
    reference + a smaller edit yields a mask at the edit res; image-apply-mask composite upscales
    it back to the reference at apply time. Same-size inputs => that resize is an identity."""
    ref = reference.resize(edit.size, LR)                  # reference as the edit's own canvas

    if mode == "region":
        return _region_mask(edit, ref, thr, GROW, FEATHER_REGION, MIN_AREA)

    return _diff_mask(edit, ref, thr, DILATE, FEATHER_MASK)

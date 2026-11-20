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

# The color option of image-to-image: the tones an engine moves while it keeps the picture
# (MiniMax-H3-Image lifts the shadows and adds red, FLUX.2 warms or cools the frame), matched back
# to one of its inputs. One curve per channel, fitted pixel against pixel on what the edit left
# alone, so an edit's own colors (a shirt made red) neither steer the curve nor get undone, and
# the detail stays the result's. Torch free. Design and measures in doc/image-adjust-plan.md.

import numpy as np

from PIL import Image, ImageFilter

# NOTE: The fit runs on copies this long on their long side, blurred a little: a tone curve needs
#       no detail, and the blur forgives what align left.
SIDE = 256

# NOTE: A drift moves a pixel a little, an edit a lot: the first fit takes the pixels nearer
#       their reference than this, in levels, as if nothing had drifted.
TOLERANCE = 48

# NOTE: Each pass keeps the pixels the last curve explains, the edit drops out.
ROUNDS = 3

BINS = 16

# NOTE: A bin with fewer pixels than this has no median to trust.
MIN_BIN = 20

# NOTE: A pixel the curve misses by more than this many times the median miss, and this many
#       levels at least, is the edit's.
SPREAD = 3
FLOOR  = 8.0

# NOTE: With fewer pixels than this left alone there is nothing to fit.
MIN_KEPT = 100

# NOTE: Under this mean move, in levels, the curves would rewrite the image for nothing.
MIN_CHANGE = 1.0

# NOTE: Past this mean move, in levels, the change was asked for, "make it night", and stays. An
#       engine drifts by less: 11.7 at most on the bench, a night edit 20 at least. The default of
#       color alone, color=<levels> gives another.
MAX_CHANGE = 16.0


def _pixels(image, color, size):
    """The image as rows of levels, on a small blurred copy (made small first: converting the
    whole image would copy it)."""
    image = image.resize(size, Image.BOX).convert(color).filter(ImageFilter.GaussianBlur(1.5))

    return np.asarray(image, np.float64).reshape(size[0] * size[1], -1)


def _fit(source, target, keep):
    """A table per channel from the source's levels to the target's: the median of each through
    the kept pixels in bins of the source, made to rise, and the shift at either end carried on
    past them."""
    levels = np.arange(256)

    tables = []

    for x, y in zip(source[keep].T, target[keep].T):
        index = np.minimum(x * BINS // 256, BINS - 1)

        points = [(np.median(x[index == b]), np.median(y[index == b]))
                  for b in range(BINS) if np.count_nonzero(index == b) >= MIN_BIN]

        if len(points) < 2:
            tables.append(levels.astype(np.float64))

            continue

        px, py = np.array(points).T

        py = np.maximum.accumulate(py)

        tables.append(np.clip(levels + np.interp(levels, px, py - px), 0, 255))

    return np.array(tables)


def run(image, reference, emit, limit=None):
    """The result's colors matched to the reference, or as they were, with one line saying
    which. A mean move over limit levels (MAX_CHANGE by default) was asked for and stays."""
    limit = limit or MAX_CHANGE

    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        emit("color: skipped, %s images are not matched" % image.mode)

        return image

    color = "L" if image.mode[0] == "L" else "RGB"

    scale = SIDE / max(image.size)

    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))

    source = _pixels(image, color, size)
    target = _pixels(reference, color, size)

    shown = np.ones(len(source), bool)

    # NOTE: A cutout's transparent pixels say nothing about its colors.
    if image.has_transparency_data:
        shown = np.asarray(image.getchannel("A").resize(size, Image.BOX)).ravel() > 0

    levels = np.clip(np.rint(source), 0, 255).astype(int)

    keep = shown & (np.abs(source - target).max(1) < TOLERANCE)

    for index in range(ROUNDS + 1):
        if np.count_nonzero(keep) < MIN_KEPT:
            emit("color: skipped, nothing is left as it was")

            return image

        tables = _fit(source, target, keep)

        if index == ROUNDS:
            break

        residual = np.abs(np.take_along_axis(tables.T, levels, 0) - target).max(1)

        keep = shown & (residual < max(FLOOR, SPREAD * np.median(residual[keep])))

    tables = np.rint(tables)

    counts = np.array([np.bincount(c, minlength=256) for c in levels[shown].T], np.float64)

    moves = (tables - np.arange(256)) * counts

    change = np.abs(moves).sum() / counts.sum()

    shifts = moves.sum(1) / counts.sum(1)

    text = "color: %s" % " ".join("%s %+.1f" % pair for pair in zip(color, shifts))

    if change > limit:
        emit(text + ", over %g levels, left as is" % limit)

        return image

    if change < MIN_CHANGE:
        emit(text + ", under %g level, left as is" % MIN_CHANGE)

        return image

    emit(text + ", matched")

    lut = tables.astype(np.uint8).ravel().tolist()

    # NOTE: The alpha goes through as it is.
    if image.mode in ("RGBA", "LA"):
        lut += list(range(256))

    return image.point(lut)

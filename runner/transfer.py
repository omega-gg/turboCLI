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

# The transfer option of image-adjust: an image graded toward another picture, a shot toward the
# one it must cut with. Unlike color, which fits the same pixels of an edit and its input, the two
# pictures need not show the same thing: each L*a*b* channel takes the reference's mean and
# spread (Reinhard et al., 2001), so the balance, the saturation and the contrast come along, not
# the content. transfer=<strength> goes part of the way, 1 by default. Torch free, in OpenCV's
# float L*a*b*: Pillow's is 8 bit, and its round trip alone greys a dark picture's shadows.

import cv2
import numpy as np

from PIL import Image

# NOTE: The statistics are taken on copies this long on their long side: a mean and a spread
#       need no detail.
SIDE = 512


def _levels(image):
    """The image as float L*a*b* levels (L 0 to 100, a and b about -128 to 127), rows of three."""
    rgb = np.asarray(image.convert("RGB"), np.float32) / 255

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def _statistics(image):
    """The mean and the spread of each L*a*b* channel, on a small copy."""
    scale = min(1, SIDE / max(image.size))

    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))

    levels = _levels(image.resize(size, Image.BOX)).astype(np.float64)

    return levels.mean(0), np.maximum(levels.std(0), 1e-6)


def run(image, reference, emit, limit=None):
    """The image graded toward the reference by limit, the strength (1 by default, the whole
    way), with one line saying how far each channel moved."""
    strength = 1.0 if limit is None else limit

    if strength > 1:
        raise ValueError("transfer=%g is over 1, the whole way" % strength)

    mean, spread = _statistics(image)

    target, target_spread = _statistics(reference)

    levels = _levels(image)

    graded = (levels - mean) * (target_spread / spread) + target

    graded = (levels + (graded - levels) * strength).astype(np.float32)

    rgb = cv2.cvtColor(graded.reshape(image.height, image.width, 3), cv2.COLOR_LAB2RGB)

    result = Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8), "RGB")

    moves = (target - mean) * strength

    emit("transfer: L %+.1f a %+.1f b %+.1f, strength %g" % (*moves, strength))

    # NOTE: The alpha goes through as it is.
    if image.has_transparency_data:
        result.putalpha(image.getchannel("A"))

    return result

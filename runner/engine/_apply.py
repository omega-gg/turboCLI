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

# Mask APPLICATION for the mask-apply engine. Torch-free (PIL only), a helper (underscore) so
# discovery skips it. Takes a precomputed mask (from a mask / mask-* engine) and lays it onto an
# image two ways:
#   composite  paste the source's masked region onto a reference (the original scene, or a new
#              backdrop); byte-exact reference where the mask is black
#   putalpha   write the mask as source's alpha channel -> an RGBA cutout, transparent elsewhere
#              and blank white there, see _clear
# and, with no mask, trims an image down to what shows:
#   trim       crop to the pixels that show, with at most `pad` pixels of transparency around,
#              and when asked, specks under `speck` pixels left out

from PIL import Image

LR = Image.Resampling.LANCZOS

# The most transparency a trim keeps around what shows, on each side, when not told otherwise.
TRIM_PAD = 32

# The alpha under which a pixel counts as not showing: a feathered mask leaves a faint haze far
# past its subject.
TRIM_ALPHA = 8

# A trim that leaves specks out looks for blobs on a mask this many times smaller, which keeps
# it fast.
TRIM_SCALE = 4


def apply(source, mask, mode, reference=None):
    """Apply `mask` to `source`. `composite` upscales source + mask to the reference res and pastes
    the masked region onto the reference. `putalpha` writes the mask as source's alpha (a cutout);
    the mask is resized to source if needed."""
    if mode == "composite":
        up = source.resize(reference.size, LR)
        m  = mask.resize(reference.size, LR)

        return Image.composite(up, reference, m)

    out = source.copy()

    if mask.size != out.size:
        mask = mask.resize(out.size, LR)

    out.putalpha(mask)                                     # promotes RGB -> RGBA

    return _clear(out)


def trim(image, pad=TRIM_PAD, speck=0):
    """Crop `image` to its pixels with an alpha over TRIM_ALPHA, and at most `pad` pixels more on
    every side, never past its edges. With `speck`, a separate blob of fewer pixels is a speck and
    left out, which a noisy mask needs and a particle around a subject would not survive. An
    image with nothing showing comes back whole, and one without alpha is all showing."""
    image = image.convert("RGBA")

    shown = image.getchannel("A").point(lambda a: 255 if a > TRIM_ALPHA else 0)

    box = _speck_box(shown, speck) if speck > 0 else shown.getbbox()

    if box is None:
        return _clear(image)

    left, top, right, bottom = box

    return _clear(image.crop((max(0, left - pad), max(0, top - pad),
                              min(image.width, right + pad), min(image.height, bottom + pad))))


def _clear(image):
    """`image` with its fully transparent pixels blank, (255, 255, 255, 0): a cutout keeps no color
    it does not show, and whatever drops its alpha sees the subject on white rather than the
    picture it was cut from."""
    shown = image.getchannel("A").point(lambda a: 255 if a else 0)

    return Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), shown)


def _speck_box(shown, speck):
    """The box of the blobs of `shown` with `speck` pixels or more, None when there are none."""
    from ._mask import _blob_boxes                         # numpy, only when specks are asked

    # A cell shows when any pixel of it does, so a thin part stays joined to its subject.
    small = shown.reduce(TRIM_SCALE).point(lambda a: 255 if a else 0)

    boxes = _blob_boxes(small, max(1, speck // (TRIM_SCALE * TRIM_SCALE)))

    if not boxes:
        return None

    return (int(min(b[0] for b in boxes)) * TRIM_SCALE,
            int(min(b[1] for b in boxes)) * TRIM_SCALE,
            int(max(b[2] for b in boxes) + 1) * TRIM_SCALE,
            int(max(b[3] for b in boxes) + 1) * TRIM_SCALE)

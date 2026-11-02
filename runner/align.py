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

# The align option of image-to-image: the zoom and shift an engine adds to its result (Qwen-Image
# 2.1 does, by a different amount for each seed), measured against one of its inputs and undone.
# Torch free. Design and measures in doc/image-align-plan.md.

import cv2
import numpy as np

from PIL import Image

TILE = 64
STEP = 32

# NOTE: The fit runs on a copy of at most this many pixels: each tile costs a phase correlation.
AREA = 1024 * 1024

# NOTE: Phase correlation reads large shifts short, each round adds what the last one left.
ROUNDS = 3

# NOTE: Fewer tiles than this gives a fit with nothing to stand on.
MIN_TILES = 16

# NOTE: A frame under this many tiles on a side is not measured: the zoom of its short axis would
#       come out of a few tiles.
MIN_SIDE = 4

# NOTE: Past this the zoom was asked for, "zoom out" or a camera that moves, and stays. An engine
#       drifts by less: Qwen-Image 2.1 by under 3% on the bench.
MAX_ZOOM = 0.03

# NOTE: Under this at the corners, in output pixels, the warp would soften about as much as it
#       moves.
MIN_SHIFT = 0.75

# NOTE: A warp that takes less than a fifth off the shifts does not explain them: the engine
#       redrew the frame rather than zooming it.
MIN_GAIN = 0.2


def _edges(gray):
    """Edge strength over its local mean: where things are, whatever the light."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)

    magnitude = cv2.magnitude(cv2.Sobel(blur, cv2.CV_32F, 1, 0),
                              cv2.Sobel(blur, cv2.CV_32F, 0, 1))

    return magnitude / (cv2.GaussianBlur(magnitude, (0, 0), 16) + 1e-3)


def _tiles(reference, result):
    """The center and the shift of each tile of two edge maps that matches."""
    window = cv2.createHanningWindow((TILE, TILE), cv2.CV_32F)

    height, width = reference.shape

    points = []
    shifts = []

    for y in range(0, height - TILE + 1, STEP):
        for x in range(0, width - TILE + 1, STEP):
            tile = reference[y:y + TILE, x:x + TILE]

            # NOTE: A flat tile has nothing to lock on, its edge map stays near its mean.
            if tile.std() < 0.05:
                continue

            # NOTE: phaseCorrelate applies the window to its inputs in place, and the tiles
            #       overlap, so it takes copies.
            other = result[y:y + TILE, x:x + TILE].copy()

            shift, response = cv2.phaseCorrelate(tile.copy(), other, window)

            # NOTE: Under this peak the two tiles do not show the same thing.
            if response > 0.1:
                points.append((x + TILE / 2, y + TILE / 2))
                shifts.append(shift)

    return np.array(points), np.array(shifts)


def _warp(image, zoom_x, zoom_y, x, y, flags):
    """The image taken back by a zoom and shift from the reference to it, the edge repeated."""
    matrix = np.float32([[zoom_x, 0, x], [0, zoom_y, y]])

    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=flags | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def _lines(points, shifts, keep):
    """The shift along each axis as a line of the position, through the tiles kept, or None."""
    if keep.sum() < MIN_TILES:
        return None

    return [np.polyfit(points[keep, i], shifts[keep, i], 1) for i in (0, 1)]


def _fit(reference, result, edges):
    """The zoom and shift per axis from the reference to the result, the reference's edges and
    the result's gray and edges given, or None when too few tiles match."""
    zoom_x = zoom_y = 1.0

    x = y = 0.0

    for index in range(ROUNDS):
        if index:
            edges = _edges(_warp(result, zoom_x, zoom_y, x, y, cv2.INTER_LINEAR))

        points, shifts = _tiles(reference, edges)

        keep = np.ones(len(points), bool)

        # NOTE: The tiles the edit redrew disagree with the rest, they go over three passes.
        for _ in range(3):
            lines = _lines(points, shifts, keep)

            if lines is None:
                return None

            residual = np.hypot(*(np.polyval(lines[i], points[:, i]) - shifts[:, i]
                                  for i in (0, 1)))

            keep = residual < max(0.75, 2.5 * np.median(residual[keep]))

        lines = _lines(points, shifts, keep)

        if lines is None:
            return None

        (slope_x, shift_x), (slope_y, shift_y) = lines

        x += zoom_x * shift_x
        y += zoom_y * shift_y

        zoom_x *= 1 + slope_x
        zoom_y *= 1 + slope_y

    return zoom_x, zoom_y, x, y


def _offset(reference, result):
    """The median shift of the tiles of two edge maps that match."""
    points, shifts = _tiles(reference, result)

    if len(points) == 0:
        return float("inf")

    return float(np.median(np.hypot(shifts[:, 0], shifts[:, 1])))


def run(image, reference, emit):
    """The result aligned on the reference, or as it was, with one line saying which."""
    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        emit("align: skipped, %s images are not aligned" % image.mode)

        return image

    ratio = image.width / image.height

    if abs(reference.width / reference.height - ratio) > 0.01 * ratio:
        emit("align: skipped, the input has another shape")

        return image

    scale = min(1.0, (AREA / (image.width * image.height)) ** 0.5)

    if min(image.size) * scale < MIN_SIDE * TILE:
        emit("align: skipped, the image is too small")

        return image

    def gray(picture):
        pixels = cv2.cvtColor(np.asarray(picture.convert("RGB")), cv2.COLOR_RGB2GRAY)

        pixels = pixels.astype(np.float32)

        if scale == 1.0:
            return pixels

        return cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    reference_edges = _edges(gray(reference.resize(image.size, Image.LANCZOS)))

    result = gray(image)
    edges  = _edges(result)

    fit = _fit(reference_edges, result, edges)

    if fit is None:
        emit("align: skipped, too few edges match")

        return image

    zoom_x, zoom_y, x, y = fit

    x /= scale
    y /= scale

    text = "align: zoom %+.2f%% x %+.2f%%" % ((zoom_x - 1) * 100, (zoom_y - 1) * 100)

    if max(abs(zoom_x - 1), abs(zoom_y - 1)) > MAX_ZOOM:
        emit(text + ", over %d%%, left as is" % round(MAX_ZOOM * 100))

        return image

    corners = [(0, 0), (image.width, 0), (0, image.height), (image.width, image.height)]

    shift = max(np.hypot((zoom_x - 1) * cx + x, (zoom_y - 1) * cy + y) for cx, cy in corners)

    if shift < MIN_SHIFT:
        emit(text + ", under %.2f px, left as is" % MIN_SHIFT)

        return image

    pixels = _warp(np.asarray(image).astype(np.float32), zoom_x, zoom_y, x, y,
                   cv2.INTER_LANCZOS4)

    aligned = Image.fromarray(pixels.round().clip(0, 255).astype(np.uint8))

    before = _offset(reference_edges, edges)
    after  = _offset(reference_edges, _edges(gray(aligned)))

    if after >= (1 - MIN_GAIN) * before:
        emit(text + ", does not explain the change, left as is")

        return image

    emit(text + ", shift %+.2f %+.2f px, aligned" % (x, y))

    return aligned

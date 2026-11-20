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

# adjust engine -- the fixes of an engine's result against the picture it edited: align
# (runner/align.py, the zoom and shift undone) and color (runner/color.py, the tones matched).
# core runs them after an image-to-image render, against its first input; as a "compute" engine,
# torch free, they run on an existing image: images = "base,reference". Options align and/or
# color, each alone for its default limit or with its own: align=<percent>, the zoom past which
# it was asked for; color=<levels>, the mean move past which it was. See
# doc/image-adjust-plan.md.

from importlib import import_module

ID    = "adjust"
MODES = ("image-adjust",)

# NOTE: The order they run in: the geometry first.
STEPS = ("align", "color")


def _limit(name, value):
    """The limit a step's value gives, None for its default (the key alone). ValueError when it
    is not a positive number."""
    if not value:
        return None

    try:
        limit = float(value)
    except ValueError:
        limit = float("nan")

    # NOTE: Written this way round, nan fails too.
    if not limit > 0:
        raise ValueError("%s=%s is not a positive number" % (name, value))

    return limit


def parse(options):
    """[(step, limit), ...] for each step the options ask, in STEPS order, the limit None for the
    step's default. ValueError when a limit is not a positive number."""
    return [(name, _limit(name, options[name])) for name in STEPS if name in options]


def step(name, image, reference, limit, emit):
    """The image through one step, each imported when asked: color alone never loads cv2."""
    return import_module(".." + name, __package__).run(image, reference, emit, limit)


def run(ctx, params, emit):
    from PIL import Image

    from ._options import parse_options

    images = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

    if len(images) != 2:
        raise ValueError("adjust needs images=base,reference")

    steps = parse(parse_options(params.get("options", "")))

    if not steps:
        raise ValueError("adjust needs align and/or color")

    image = Image.open(images[0])

    # NOTE: A palette file goes through as the colors it shows.
    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        image = image.convert("RGBA" if image.has_transparency_data else "RGB")

    reference = Image.open(images[1])

    for name, limit in steps:
        image = step(name, image, reference, limit, emit)

    return image

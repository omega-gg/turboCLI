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

# esrgan-upscale engine: image-to-image that upscales its first input to the output size with
# Real-ESRGAN x4plus. A "compute" engine (core's run() seam, no diffusion, no offloader); the model
# runs at its native x4 in tiles, then lanczos brings it to the size (see _upscale.py). No prompt.
#
# Install (python -m runner.install): kind "url" -> the checkpoint (a GitHub release asset) into
# model/RealESRGAN/RealESRGAN_x4plus.pth. revision is the release tag, not an HF commit.

import os

from . import _comfy  # cheap: no torch at top level

ID    = "esrgan-upscale"
MODES = ("image-to-image",)

MODEL = {"kind": "url", "model": "RealESRGAN", "revision": "v0.1.0",
         "file": "RealESRGAN_x4plus.pth",
         "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
                "RealESRGAN_x4plus.pth"}


def weights(model_dir):
    """The model file: the checkpoint in model_dir, or for comfy-esrgan-upscale, whose model_dir is
    its registry entry, the ComfyUI file its engine.json names."""
    if os.path.isfile(os.path.join(model_dir, "engine.json")):
        return _comfy.by_role(model_dir)["upscaler"]

    return os.path.join(model_dir, MODEL["file"])


def run(ctx, params, emit):
    from ._upscale import run

    return run(ctx, params, emit, weights(ctx.model))

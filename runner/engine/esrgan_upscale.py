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
# Real-ESRGAN x4plus. A "compute" engine (core's run() seam, no diffusion, no offloader). No prompt.
#
# The model runs at its native x4 in overlapping tiles blended by a linear ramp, then lanczos brings
# the result to the output size. Tile numbers are ComfyUI's ImageUpscaleWithModel: 512 with a 32
# overlap. On out of memory the result moves to the cpu (as ComfyUI keeps it), then the tile
# halves, down to 128. Top level is torch free; torch and spandrel load inside the functions.
#
# Install (python -m runner.install): kind "url" -> the checkpoint (a GitHub release asset) into
# model/RealESRGAN/RealESRGAN_x4plus.pth. revision is the release tag, not an HF commit.

import os

from . import _comfy  # cheap: no torch at top level

ID    = "esrgan-upscale"
MODES = ("image-to-image",)

UPSCALE = True

MODEL = {"kind": "url", "model": "RealESRGAN", "revision": "v0.1.0",
         "file": "RealESRGAN_x4plus.pth",
         "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
                "RealESRGAN_x4plus.pth"}

TILE, OVERLAP, TILE_MIN = 512, 32, 128

# The last loaded model, (file, device, descriptor): a warm call pays no load (67 MB).
_resident = None


def weights(model_dir):
    """The model file: the checkpoint in model_dir, or for comfy-esrgan-upscale, whose model_dir is
    its registry entry, the ComfyUI file its engine.json names."""
    if os.path.isfile(os.path.join(model_dir, "engine.json")):
        return _comfy.by_role(model_dir)["upscaler"]

    return os.path.join(model_dir, MODEL["file"])


def run(ctx, params, emit):
    """Upscale the first input to the output size, its alpha channel following at that size."""
    from PIL import Image

    path  = [s.strip() for s in params.get("images", "").split(",") if s.strip()][0]
    image = Image.open(path)
    size  = (int(params["width"]), int(params["height"]))

    result = _upscale(image.convert("RGB"), size, ctx.renderer, weights(ctx.model), emit)

    if "A" in image.getbands():
        result.putalpha(image.getchannel("A").resize(size, Image.Resampling.LANCZOS))

    return result


def _load(file, device):
    global _resident

    import torch
    from spandrel import ModelLoader, ImageModelDescriptor

    if _resident is not None and _resident[:2] == (file, device):
        return _resident[2]

    _resident = None

    model = ModelLoader().load_from_file(file)

    if not isinstance(model, ImageModelDescriptor):
        raise ValueError("%s is not a single image model" % file)

    model.to(device).eval()

    # NOTE: Half on a GPU, fp32 on the cpu, where half convolutions are slow or missing.
    if device != "cpu" and model.supports_half:
        model.half()

    model.model.to(memory_format=torch.channels_last)

    _resident = (file, device, model)

    return model


def _starts(size, tile, step):
    """Tile origins along one side: every `step`, the last one flush with the far edge."""
    if size <= tile:
        return [0]

    return list(range(0, size - tile, step)) + [size - tile]


def _ramp(size, feather, device):
    """1 inside, rising linearly over `feather` pixels at both ends (never 0)."""
    import torch

    ramp  = torch.ones(size, device=device)
    rise  = torch.arange(1, feather + 1, device=device) / feather
    count = min(feather, size)

    ramp[:count]  = torch.minimum(ramp[:count],  rise[:count])
    ramp[-count:] = torch.minimum(ramp[-count:], rise[:count].flip(0))

    return ramp


def _tiled(model, x, tile, place, emit):
    """Run `model` over x (1x3xHxW) in tiles of `tile` overlapping by OVERLAP, the result on
    `place`. Returns the 3x(H*s)x(W*s) float result.

    The tiles form a grid and each one's weight is a row ramp times a column ramp, so their sum is
    the sum of the row ramps times the sum of the column ramps: two lines, not a weight image."""
    import torch

    _, _, height, width = x.shape

    s    = model.scale
    step = tile - OVERLAP

    rows = _starts(height, tile, step)
    cols = _starts(width,  tile, step)

    h, w = min(tile, height) * s, min(tile, width) * s

    ramp_h = _ramp(h, OVERLAP * s, place)
    ramp_w = _ramp(w, OVERLAP * s, place)

    mask = ramp_h[:, None] * ramp_w[None, :]

    out    = torch.zeros((3, height * s, width * s), device=place)
    weight = torch.zeros(height * s, device=place), torch.zeros(width * s, device=place)

    for top in rows:
        weight[0][top * s:top * s + h] += ramp_h

    for left in cols:
        weight[1][left * s:left * s + w] += ramp_w

    count = len(rows) * len(cols)

    for i, (top, left) in enumerate(((top, left) for top in rows for left in cols), 1):
        y = model(x[:, :, top:top + tile, left:left + tile])[0].float().to(place)

        out[:, top * s:top * s + h, left * s:left * s + w] += y * mask

        emit("%3d%%|step %d/%d" % (i * 100 // count, i, count))

    return out.div_(weight[0][:, None] * weight[1][None, :])


def _upscale(image, size, renderer, file, emit):
    """Upscale the RGB PIL `image` with the model in `file`, then lanczos to `size`."""
    import numpy as np
    import torch
    from PIL import Image

    from ._segment import pick_device

    device = pick_device(renderer)
    model  = _load(file, device)

    with torch.inference_mode():
        x = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None]
        x = x.to(model.dtype).div_(255).contiguous(memory_format=torch.channels_last)

        tile  = TILE
        place = device

        while True:
            try:
                y = _tiled(model, x, tile, place, emit)
                break
            except torch.OutOfMemoryError:
                if place != "cpu":
                    place = "cpu"
                elif tile // 2 >= TILE_MIN:
                    tile //= 2
                else:
                    raise

                emit("out of memory, result on the cpu, tile %d" % tile)

                torch.cuda.empty_cache()

        y = y.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0)

        result = Image.fromarray(y.cpu().numpy())

    emit("esrgan[%s/%s]: %dx%d -> %dx%d" % (device, str(model.dtype).split(".")[-1],
                                            image.width, image.height, size[0], size[1]))

    if result.size == size:
        return result

    return result.resize(size, Image.Resampling.LANCZOS)

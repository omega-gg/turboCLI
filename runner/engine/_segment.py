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

# Subject-matte generation for the mask-birefnet / mask-lucida / mask-inspyrenet engines. A helper
# (underscore name) so engine discovery skips it. Top level is torch-free (numpy + PIL only); the
# heavy stack (torch / torchvision / transformers / transparent_background) is imported INSIDE the
# functions, so it never loads on the diffusion path or during discovery.
#
# birefnet_alpha / inspyrenet_alpha give a subject alpha in [0,1] at the input size; build_matte
# turns it into an 8-bit L matte, fusing the clean-plate cast shadow when a plate is given: the
# model covers the subject only, and a shadow darkens the ground, so where the input is darker than
# the plate (by more than the threshold) is kept as soft alpha.

import numpy as np
from PIL import Image, ImageFilter

# Shadow tuning. SHADOW_TOLERANCE is the default (options tolerance=N, 0-255): how much cast
# shadow to keep -- the engine passes 255 - tolerance as the darkening floor `thr`. SHAD_NORM
# (normalisation) and SHAD_MAX (max opacity, never fully opaque) stay fixed. Default 243 = 255 - 12
# (the previous shadow threshold).
SHADOW_TOLERANCE, SHAD_NORM, SHAD_MAX = 243, 70.0, 0.7


def pick_device(want):
    """Honour the requested device, falling back to cpu when it is not available in this build."""
    import torch

    if want == "cuda" and torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)

    if want == "mps" and mps and mps.is_available():
        return "mps"

    return "cpu"


def birefnet_alpha(image, device, model_dir):
    """BiRefNet matte (birefnet | lucida, from model_dir) at the input size in [0, 1]; `device`
    pre-resolved."""
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    net = AutoModelForImageSegmentation.from_pretrained(model_dir, trust_remote_code=True)
    net.eval()

    half = device == "cuda"                                # half fits the 885 MB model on 4 GB

    net.to(device)

    if half:
        net.half()
    else:
        # NOTE: The checkpoint carries its own dtype and BiRefNet ships fp16, which a cpu conv
        # cannot mix with a float input -- "Input type (float) and bias type (c10::Half)".
        net.float()

    pre = transforms.Compose([
        transforms.Resize((1024, 1024)),                   # BiRefNet's trained resolution
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    x = pre(image).unsqueeze(0).to(device)

    if half:
        x = x.half()

    with torch.inference_mode():
        pred = net(x)[-1].sigmoid().float().cpu()[0, 0]    # final map -> [0, 1] at 1024x1024

    alpha = Image.fromarray((pred.numpy() * 255).astype(np.uint8), "L").resize(image.size)

    return np.asarray(alpha, np.float32) / 255.0


def inspyrenet_alpha(image, device, ckpt_path):
    """InSPyReNet matte via transparent-background (base mode) at the input size in [0, 1]."""
    from transparent_background import Remover

    dev = "cuda:0" if device == "cuda" else device         # cpu | mps pass through

    remover = Remover(mode="base", ckpt=ckpt_path, device=dev)

    rgba = remover.process(image, type="rgba")             # PIL RGBA at the input size

    return np.asarray(rgba.split()[3], np.float32) / 255.0


def _plate_shadow(image, plate, subj, thr):
    """Cast shadow from a clean-plate diff: where the input is darker than the empty plate by more
    than `thr` (raise it to reject a drifted plate ghosting the background)."""
    def lum(im):
        a = np.asarray(im.convert("RGB"), np.float32)
        return a @ np.array([0.299, 0.587, 0.114], np.float32)

    plate = plate.convert("RGB").resize(image.size)

    dark   = np.clip(lum(plate) - lum(image), 0, None)     # a shadow darkens the ground
    shadow = np.clip((dark - thr) / SHAD_NORM, 0, SHAD_MAX)

    m = Image.fromarray((shadow / SHAD_MAX * 255).astype(np.uint8), "L")
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))   # despeckle
    shadow = np.asarray(m, np.float32) / 255.0 * SHAD_MAX

    return shadow * (1.0 - subj)                           # add only outside the subject


def build_matte(image, alpha, plate, thr):
    """8-bit L matte from a subject alpha [0, 1]. With a `plate` (the same scene without the
    subject), the cast shadow is recovered from the luminance diff and merged in; without one it is
    subject only (no wasted work)."""
    if plate is not None:
        shadow = _plate_shadow(image, plate, alpha, thr)
        alpha  = np.clip(np.maximum(alpha, shadow), 0, 1)

    return Image.fromarray((alpha * 255).astype(np.uint8), "L")

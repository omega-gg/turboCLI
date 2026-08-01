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

# Lucida background removal for the image-mask `extract` mode. Lucida is a MIT fine-tune of
# BiRefNet_HR (egeorcun/lucida) that produces a soft alpha matte of the subject. Runs in this
# tool's own venv (torch + transformers + timm/einops/kornia) under gg.omega/lucida; the model is
# beside this file in ./model (saved offline at build time).
#
# BiRefNet's matte covers the SUBJECT only, not its cast ground shadow. --plate is optional: give a
# clean background (same scene, no subject); where the input is darker than the plate is the cast
# shadow, recovered as soft alpha so it is kept. Without --plate it is subject only.
#
# Output: RGBA PNG, same size/placement as the input, transparent outside the subject (+ shadow).
# Run as: python extract.py --input in.png --output out.png [--plate plate.png] [--device cuda]

import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Fixed shadow tuning (not exposed): darkening threshold, normalisation, and the max shadow opacity
# (a cast shadow is semi-transparent, never fully opaque).
SHAD_THR, SHAD_NORM, SHAD_MAX = 12.0, 70.0, 0.7


def _pick_device(want):
    """Honour the requested device, falling back to cpu when it is not available in this build."""
    import torch

    if want == "cuda" and torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)

    if want == "mps" and mps and mps.is_available():
        return "mps"

    return "cpu"


def _subject_alpha(image, device):
    """Subject matte (BiRefNet/Lucida) at the input size in [0, 1], plus the device used."""
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    model_dir = Path(__file__).resolve().parent / "model"

    model = AutoModelForImageSegmentation.from_pretrained(str(model_dir), trust_remote_code=True)
    model.eval()

    device = _pick_device(device)
    half   = device == "cuda"                              # half fits the 885 MB model on 4 GB

    model.to(device)

    if half:
        model.half()

    pre = transforms.Compose([
        transforms.Resize((1024, 1024)),                  # Lucida's trained resolution
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    x = pre(image).unsqueeze(0).to(device)

    if half:
        x = x.half()

    with torch.inference_mode():
        pred = model(x)[-1].sigmoid().float().cpu()[0, 0]  # final map -> [0, 1] at 1024x1024

    alpha = Image.fromarray((pred.numpy() * 255).astype(np.uint8), "L").resize(image.size)

    return np.asarray(alpha, np.float32) / 255.0, device


def _plate_shadow(image, plate, subj):
    """Cast shadow from a clean-plate diff: where the input is darker than the empty plate."""
    def lum(im):
        a = np.asarray(im.convert("RGB"), np.float32)
        return a @ np.array([0.299, 0.587, 0.114], np.float32)

    plate = plate.convert("RGB").resize(image.size)

    dark   = np.clip(lum(plate) - lum(image), 0, None)     # a shadow darkens the ground
    shadow = np.clip((dark - SHAD_THR) / SHAD_NORM, 0, SHAD_MAX)

    m = Image.fromarray((shadow / SHAD_MAX * 255).astype(np.uint8), "L")
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))   # despeckle
    shadow = np.asarray(m, np.float32) / 255.0 * SHAD_MAX

    return shadow * (1.0 - subj)                           # add only outside the subject


def main():
    p = argparse.ArgumentParser(prog="extract")

    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--plate",  default=None)               # optional clean background: keep shadow
    p.add_argument("--device", default="cpu")              # cpu | cuda | mps (falls back to cpu)

    args = p.parse_args()

    try:
        image = Image.open(args.input).convert("RGB")

        alpha, device = _subject_alpha(image, args.device)

        # With a clean plate, add the cast shadow; without one it is subject only (no wasted work).
        if args.plate:
            shadow = _plate_shadow(image, Image.open(args.plate), alpha)
            alpha  = np.clip(np.maximum(alpha, shadow), 0, 1)

        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray((alpha * 255).astype(np.uint8), "L"))
        rgba.save(args.output)

        print("extract[%s]: alpha %.0f%%" % (device, alpha.mean() * 100), flush=True)
        print("Saved: %s" % args.output, flush=True)
    except Exception:
        print("ERROR: " + traceback.format_exc(), flush=True)

        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

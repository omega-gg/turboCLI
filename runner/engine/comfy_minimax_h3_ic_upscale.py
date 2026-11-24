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

# comfy-minimax-h3-ic-upscale engine: a comfy-minimax-h3 take rendered again at a higher
# resolution from its saved latents (latent=save) by NVIDIA's Open-H3-Regenerate-2K IC (NVlabs/Sana,
# sol-engine, models/Open-H3-Regenerate-2K/IC): the take kept in the context as a clean reference
# video on H3's Ref2VA model, IC's LoRA on HyperFlow's 8-step adapter and LMS Sharp, its video
# latent enlarged by the same community upscaler as comfy-minimax-h3-upscale's and refined in 3
# Euler steps (IC's up3; inference 4 is its up4). The soundtrack stays the take's. Twice the
# take's size, or the width and height given above it (core._take). image-to-video only, on
# latent=<stem>.video.latent alone; a take of 124 frames at most, IC's trained length.
#
# LICENSE: the IC LoRA, HyperFlow and the full ref2va file are under the MiniMax H3 Community
# License (as H3), LMS Sharp under its author's terms.

from . import comfy_minimax_h3 as base  # cheap: no torch at top level
from . import comfy_minimax_h3_upscale as upscale

ID   = "comfy-minimax-h3-ic-upscale"
BASE = base.ID

MODES   = upscale.MODES
UPSCALE = upscale.UPSCALE

INFERENCE = 3  # IC's default sampler, up3

CRF = 12.0  # as IC writes its videos

# IC's files (Efficient-Large-Model/Open-H3-Regenerate-2K-IC, its config.json's components).
IC_FILES = [{"role": role, "repository": "Efficient-Large-Model/Open-H3-Regenerate-2K-IC",
             "revision": "54a7104b204fb46486cf85c2f600942118186d57", "filename": filename,
             "path": path} for role, filename, path in (
    ("lora_ic", "lora/open_h3_regenerate_2k_ic_lora.safetensors",
     "loras/open_h3_regenerate_2k_ic_lora.safetensors"),
    ("lora_hyperflow", "lora/minimax_h3_hyperflow_8step_v1.0.safetensors",
     "loras/minimax_h3_hyperflow_8step_v1.0.safetensors"),
    ("lora_sharp", "lora/minimax_h3_lms_v1.0_r64.safetensors",
     "loras/minimax_h3_lms_v1.0_r64.safetensors"),
    ("silence", "assets/silence_audio_124f.safetensors",
     "latents/open_h3_regenerate_2k_ic_silence_audio_124f.safetensors"))]


def components(base_components, transformer_ic):
    """An IC engine's files: its base's but the fl2va transformer (the pruned ref2va stays, Sharp's
    curve table), the full-form ref2va (time embedder and full adaln, which HyperFlow needs), the
    latent upscaler and IC's files."""
    return ([c for c in base_components if c["role"] != "transformer"]
            + [{"role": "transformer_ic", "path": transformer_ic, "filename": transformer_ic},
               upscale.UPSCALER] + IC_FILES)


COMFY = dict(base.COMFY, components=components(
    base.COMFY["components"], "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"))

# HyperFlow's raw 8-step grid (its file's hyperflow_sigmas), shifted at the video's 12 as
# ref2va_ic/hyperflow.py's sigmas: up3 runs 0.834942, 0.696852, 0.468753, 0.
RAW_SIGMAS = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)

# IC's task before the take's prompt (ref2va_ic/encode.py), and the grid a sampler takes the last
# steps of.
IC = {"task": "Enhance <Video 1> into a high-quality version of the same video. Preserve the "
              "scene, objects, camera motion, timing, and composition of every frame. Restore "
              "fine details and remove degradation. Do not continue or change the scene.",
      "sigmas": tuple(12.0 * r / (1.0 + 11.0 * r) for r in RAW_SIGMAS)}

FRAMES = 124  # IC's draft length (ref2va_ic/media.py)

# LMS Sharp's strength (infer.py --sharp): IC's 1 by default, 0 leaves it out.
OPTIONS = dict(base.OPTIONS, sharp=("0", "0.5", "1"))


def _sharp(params):
    from ._options import parse_options

    return float(parse_options(params.get("options", "")).get("sharp", 1))


def extra_key(params):
    return (_sharp(params),)  # one transformer, whatever the take's inputs, per Sharp strength


def check(params):
    """A take's latents, of IC's length at most."""
    upscale.check(params)

    if base.frames(float(params.get("duration") or 5)) > FRAMES:
        raise ValueError("%s renders takes of %d frames at most (IC's), about 5 s"
                         % (params["engine"], FRAMES))


def load(ctx, params):
    return base.load(ctx, params, ic=dict(IC, sharp=_sharp(params)))

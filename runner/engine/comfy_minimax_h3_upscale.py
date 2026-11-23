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

# comfy-minimax-h3-upscale engine: a comfy-minimax-h3 take rendered again at a higher
# resolution from its saved latents (latent=save), as the experimental H3's --hires does
# (msvc/experimental/h3, remote/gen.py --from-latent): the video latent enlarged by the community
# MiniMax H3 Latent Upscaler (3D) node (offloader/minimax_h3_upscaler.py), then refined by H3 at
# twice the take's size (8 steps, denoise 0.45) with its inputs re-applied; the soundtrack stays
# the take's. image-to-video only, on latent=<stem>.video.latent alone: core makes the rest of
# the request of the take's settings, saved with its latents (core._take).

from . import comfy_minimax_h3 as base  # cheap: no torch at top level

ID   = "comfy-minimax-h3-upscale"
BASE = base.ID

MODES = ("image-to-video",)  # wire values, same as the HTTP API

# The upscaler's file, pinned as the experimental H3 pins it (h3_modal.py HIRES_HF).
UPSCALER = {"role": "upscaler", "repository": "LBH-123-AI/Minimax_h3_latent_Upscaler",
            "revision": "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6",
            "filename": "minimax_h3_latent_upscaler_3d_conv_v1/"
                        "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
            "path": "latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"}

COMFY = dict(base.COMFY, components=base.COMFY["components"] + [UPSCALER])

INFERENCE = 8  # the refine's steps (gen.py --hires-steps)

UPSCALE = True  # it enlarges its input, as the image upscalers do (check --modes)


def check(params):
    """The take's latents are what it renders again."""
    from ._options import parse_references

    if parse_references(params.get("images", ""))["latent"] is None:
        raise ValueError("%s renders a take again: give its latent=<stem>.video.latent"
                         % params["engine"])

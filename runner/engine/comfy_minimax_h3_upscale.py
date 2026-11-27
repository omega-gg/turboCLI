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

# The upscaler's file, the same 3D file as at the experimental H3's revision (3f941d5).
UPSCALER = {"role": "upscaler", "repository": "LBH-123-AI/Minimax_h3_latent_Upscaler",
            "revision": "85a0187096046badc0a71ab6bd6be182ab06c0f4",
            "filename": "minimax_h3_latent_upscaler_3d_conv_v1/"
                        "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
            "path": "latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"}

COMFY = dict(base.COMFY, components=base.COMFY["components"] + [UPSCALER])

INFERENCE = 8  # the refine's steps (gen.py --hires-steps)

# NOTE: The refine's strength in place of REFINE_DENOISE (0.45), any number from 0 (no refine:
#       the take as the upscaler enlarged it) to 1 (a new video from pure noise). Lower keeps more
#       of the take's fine detail (its rain), redrawn less. H3's shift starts 0.45 at sigma 0.91,
#       0.3 at 0.84 and 0.25 at 0.8 (beta, with references: 0.79 and 0.72): it tells from 0.25
#       down, and above 0.45 barely (0.6 at 0.94).
OPTIONS = dict(base.OPTIONS, denoise=(0.0, 1.0))

UPSCALE = True  # it enlarges its input, as the image upscalers do (check --modes)


def check(params):
    """The take's latents are what it renders again."""
    from ._options import parse_references

    if parse_references(params.get("images", ""))["latent"] is None:
        raise ValueError("%s renders a take again: give its latent=<stem>.video.latent"
                         % params["engine"])

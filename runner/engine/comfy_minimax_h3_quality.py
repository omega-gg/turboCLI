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

# comfy-minimax-h3-quality engine: comfy-minimax-h3 on Comfy-Org's highest precision files, the
# bf16 transformer and text encoder and the fp16 video VAE, as the base graph of the experimental
# H3 runs them (msvc/experimental/h3, remote/gen.py): the text encoder in bf16 (ComfyUI with
# --bf16-text-enc), SaveVideo at crf 12. With references, its Ref2VA graph: the bf16 ref2va
# transformer and the beta scheduler, the frames pinned by MiniMaxH3AddGuide. Its keyframes go to
# the node as given, as the template's do: gen.py's ImageScale center crop is left to the source.

from . import comfy_minimax_h3 as base  # cheap: no torch at top level

ID   = "comfy-minimax-h3-quality"
BASE = base.ID

COMFY = dict(base.COMFY, components=[
    {"role": role, "path": path, "filename": path} for role, path in (
        ("transformer",  "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"),
        ("transformer_ref", "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors"),
        ("text_encoder", "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"),
        ("vae",          "vae/minimax_h3_video_vae_fp16.safetensors"),
        ("audio_vae",    "vae/minimax_h3_audio_vae_fp32.safetensors"),
    )])

CRF = 12.0  # SaveVideo's crf input, a float


def load(ctx, params):
    return base.load(ctx, params, text_dtype="bfloat16", ref_scheduler="beta")

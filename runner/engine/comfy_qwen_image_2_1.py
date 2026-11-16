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

# comfy-qwen-image-2-1 engine: text2img + img2img on Qwen-Image 2.1, REUSING a ComfyUI install.
#
# One 7B model generates and edits with the same weights, so one engine declares both modes. It
# loads the files ComfyUI's three Qwen-Image 2.1 templates load: the int8 ConvRot transformer and
# the int8 ConvRot Qwen3-VL 8B text encoder (kept int8, run by comfy-kitchen's int8 kernels as
# ComfyUI does), plus the RGBA VAE. Only the configs, processor and scheduler come from the
# diffusers repo (the scaffold).
#
# OFFLOADER-ONLY: int8 needs the comfy quant path, so load() bails out for other offload modes.
# The transformer and the text encoder are ComfyUI's own models, which the offloader runs behind
# the pipeline's calls (its "qwen_image21" model), prefix K/V cache included. So is the VAE (the
# offloader's comfy_vae, see comfy_qwen_image_edit_2511._build_vae), which decodes a single image
# in row strips, exact and in a quarter of diffusers' memory, where diffusers' tiles bent the alpha
# along their seams. The sampling schedule is pinned to ComfyUI's (shift 0.69, simple), see load().
#
# The output is RGBA (the VAE decodes 4 channels), so save it as PNG.
#
# IMPORTANT (see engine/__init__.py): no torch/diffusers import at top level, discovery stays
# cheap. The heavy imports live inside load() and the helpers.

import math
import os

from . import comfy_qwen_image_edit_2511 as qe  # cheap: no torch at top level

ID   = "comfy-qwen-image-2-1"
TYPE = "qwen-image-21"

PIPELINE    = "diffusers:QwenImage21Pipeline"
TRANSFORMER = "diffusers:QwenImage21Transformer2DModel"  # ComfyUI's runs in its place

MODES = ("text-to-image", "image-to-image")
CFG   = ("true_cfg_scale", 1.0)  # no negative prompt: no CFG, as ComfyUI's cfg 1

INFERENCE = 25  # ComfyUI's template (the diffusers default is 40)

# The three files sit at plain paths of one Comfy-Org repo (no split_files/), so each names its
# `filename`. Revision pinned.
_REPO = {"repository": "Comfy-Org/Qwen-Image-2.1",
         "revision": "cb504a4090723e43f17ad01cec0359490e2de613"}

COMFY = {
    "revision": "main",
    "components": [
        dict(_REPO, role=role, path=path, filename=path)
        for role, path in (
            ("transformer", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
            ("text_encoder", "text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
            ("vae", "vae/qwen_image_2.1_vae_bf16.safetensors"),
        )
    ],
}

# The diffusers scaffolding: configs + processor (which the pipeline reads at construction) +
# scheduler.
SCAFFOLD = {
    "repository": "Qwen",
    "model": "Qwen-Image-2.1",
    "revision": "d26bb61231c349cf6b7896fa83353113880e1ba3",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "processor/*",
        "transformer/config.json",
        "vae/config.json",
    ],
}

def load(ctx, params):
    """Assemble a QwenImage21Pipeline from ComfyUI's int8 single files via the disk-stream
    offloader. OFFLOADER-ONLY (int8 needs the comfy quant path)."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json)
    files    = qe._by_role(scaffold)

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader); both big models "
                           "are int8 and need the comfy quant path" % ID)

    from diffusers import (QwenImage21Pipeline, QwenImage21Transformer2DModel,
                           FlowMatchEulerDiscreteScheduler)
    from transformers import Qwen3VLProcessor

    # ComfyUI samples with ModelSamplingFlux(shift 0.69) and the "simple" scheduler: the default
    # linspace(1, 1/N, N) under the exponential time shift at a FIXED mu of 0.69 (base == max, so
    # mu no longer depends on the resolution) and no terminal stretch.
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        scaffold, subfolder="scheduler", base_shift=0.69, max_shift=0.69, shift_terminal=None)
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(scaffold, "processor"))
    vae = qe._build_vae(scaffold, files["vae"], ctx.backend)

    # The pipeline sizes each reference to about output_resolution squared (1024 by default),
    # and an output of another size shifts the edit (ComfyUI's TextEncodeQwenImage21 says so).
    # At the output's area, a reference of its size on the 32 grid (the canvas) stays as it is.
    # Under 288 squared it goes twice the size or more, a whole factor keeping its shape: the
    # vision processor enlarges an image under 256 squared, which then no longer matches its
    # latents, and another reference rounds on the 32 grid.
    # The prompt and the references are encoded by ComfyUI's text encoder, as its
    # TextEncodeQwenImage21 does.
    class Pipe(QwenImage21Pipeline):
        def __call__(self, *args, width=None, height=None, **kwargs):
            side = math.sqrt(width * height)

            return super().__call__(*args, width=width, height=height,
                                    output_resolution=side * math.ceil(288 / side), **kwargs)

        def encode_prompt(self, prompt, image=None, device=None, num_images_per_prompt=1,
                          prompt_embeds=None, prompt_embeds_mask=None, image_pad_mask=None):
            if prompt_embeds is None:
                prompt_embeds, prompt_embeds_mask, image_pad_mask = self.text_encoder.encode(
                    prompt, images=image)

            device = device or self._execution_device

            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.to(device)

            return prompt_embeds.to(device), prompt_embeds_mask, image_pad_mask.to(device)

    return ctx.backend.load_pipe_comfy(
        Pipe,
        {"model": "qwen_image21", "file": files["transformer"],
         "config": QwenImage21Transformer2DModel.load_config(
             os.path.join(scaffold, "transformer"))},
        {"model": "qwen_image21", "file": files["text_encoder"]},
        {"scheduler": scheduler, "processor": processor, "vae": vae},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

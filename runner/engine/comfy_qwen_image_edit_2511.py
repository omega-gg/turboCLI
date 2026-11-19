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

# comfy-qwen-image-edit-2511: image-edit on Qwen-Image-Edit-2511, REUSING a ComfyUI install.
#
# Same model as qwen-image-edit-2511, built from ComfyUI's split single files: the transformer,
# the scaled-fp8 Qwen2.5-VL text encoder and the VAE.
#
# OFFLOADER-ONLY: the transformer and the text encoder are ComfyUI's own models, which the
# offloader runs behind the pipeline's calls (its "qwen_image" model), the files loaded as they
# are. The edit is ComfyUI's TextEncodeQwenImageEditPlus: each image seen by the text encoder at
# about 384x384 with the prompt, and VAE-encoded at about 1024x1024 as a reference for the
# transformer (its `edit`); the denoise, the schedule and the decode are the stock text-to-image
# pipeline's (_pipeline_class). The VAE is ComfyUI's own (through the offloader's comfy_vae).
#
# install:  runner.install --comfy <ComfyUI dir> reuses the three single files (downloading only
#           the missing ones, each from its own Comfy-Org repo) + the scaffold into engine/<id>.
# check:    runner.check treats a COMFY engine installed when engine.json + its files + scaffold.
#
# IMPORTANT (see engine/__init__.py): no torch/diffusers import at top level, discovery stays
# cheap. The heavy imports live inside load().

import os
from . import _comfy  # cheap: no torch at top level

ID   = "comfy-qwen-image-edit-2511"
TYPE = "qwen-image-edit"  # offloader seam vocabulary (same family as qwen-image-edit-2511)

PIPELINE    = "diffusers:QwenImagePipeline"
TRANSFORMER = "diffusers:QwenImageTransformer2DModel"  # ComfyUI's runs in its place

MODES = ("image-to-image",)
CFG   = ("true_cfg_scale", 1.0)

INFERENCE = 40

# ComfyUI split single files reused by this engine. The three live in DIFFERENT Comfy-Org repos, so
# each carries its own `repository` (install downloads a missing one from there).
COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
         "path": "diffusion_models/qwen_image_edit_2511_bf16.safetensors"},
        {"role": "text_encoder", "repository": "Comfy-Org/HunyuanVideo_1.5_repackaged",
         "path": "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"},
        {"role": "vae", "repository": "Comfy-Org/Qwen-Image_ComfyUI",
         "path": "vae/qwen_image_vae.safetensors"},
    ],
}

# The diffusers scaffolding fetched from the Qwen repo: the scheduler and the configs the pipeline
# reads, no weights. Revision pinned to the stock engine's.
SCAFFOLD = {
    "repository": "Qwen",
    "model": "Qwen-Image-Edit-2511",
    "revision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "vae/config.json",
        "transformer/config.json",
    ],
}


def _pipeline_class():
    """QwenImagePipeline with ComfyUI's TextEncodeQwenImageEditPlus: the prompt encoded with the
    images, and the images as the transformer's references. Built here so diffusers stays out of
    engine discovery."""
    import torch
    from diffusers import QwenImagePipeline

    class QwenImageEditPipe(QwenImagePipeline):
        def encode_prompt(self, prompt=None, device=None, num_images_per_prompt=1,
                          prompt_embeds=None, prompt_embeds_mask=None, max_sequence_length=1024,
                          image=None):
            if prompt_embeds is None:
                prompt_embeds, prompt_embeds_mask = self.text_encoder.encode(prompt, image or ())

            device = device or self._execution_device

            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.to(device)

            # The stock encode drops a mask with nothing masked, as ComfyUI does.
            return super().encode_prompt(None, device, num_images_per_prompt,
                                         prompt_embeds.to(device), prompt_embeds_mask,
                                         max_sequence_length)

        def __call__(self, prompt=None, image=None, **kwargs):
            images = list(image or [])

            # Encoded through encode_prompt, so the offloader's cache keys it on the images too.
            # The stock call wants a mask beside the embeds, even one with nothing masked.
            embeds, mask = self.encode_prompt(prompt, image=images)
            if mask is None:
                mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)

            with self.transformer.edit(images, self.vae):
                return super().__call__(prompt_embeds=embeds, prompt_embeds_mask=mask, **kwargs)

    return QwenImageEditPipe


def load(ctx, params):
    """Build the pipeline of _pipeline_class from ComfyUI's single files, the transformer and the
    text encoder as ComfyUI's own models on the offloader. OFFLOADER-ONLY. The VAE is ComfyUI's
    own; the scheduler comes from the scaffold."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json); shared by inheriting variants
    files    = _comfy.by_role(scaffold)

    if ctx.backend is None:
        engine_id = os.path.basename(os.path.normpath(scaffold))  # engine/<id> -> <id>
        raise RuntimeError("%s requires an offload backend (offload=offloader): ComfyUI's models "
                           "run on it" % engine_id)

    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")

    vae = _comfy.comfy_vae(scaffold, files["vae"], ctx.backend)  # ComfyUI's own

    # An inheriting variant may add a reused "lora" component (see comfy-...-lightning); apply it
    # to the transformer at full strength before user-supplied ctx.loras.
    lora_files = list(ctx.loras)
    if "lora" in files:
        lora_files = [(files["lora"], 1.0)] + lora_files

    return ctx.backend.load_pipe_comfy(
        _pipeline_class(),
        {"model": "qwen_image", "file": files["transformer"],
         "config": QwenImageTransformer2DModel.load_config(os.path.join(scaffold, "transformer"))},
        {"model": "qwen_image", "file": files["text_encoder"]},
        {"scheduler": scheduler, "tokenizer": None, "vae": vae},
        ctx.dtype, device=ctx.device, lora_files=lora_files or None)

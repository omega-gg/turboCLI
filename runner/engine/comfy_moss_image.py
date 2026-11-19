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

# comfy-moss-image engine: text2img + img2img on MOSS-Image SFT (~5B), REUSING a ComfyUI install
# file for file, as ComfyUI's two templates run it: the int8 ConvRot transformer, the Qwen3.5 9B
# w4a8 text encoder and the FLUX.2 small VAE; image to image adds Comfy-Org's edit LoRA (rank 64,
# from chinoll/Moss-Image-Edit). Only configs and the official pipeline file come from OpenMOSS's
# repo (the scaffold).
#
# Each mode is its template (image_moss_image_t2i / image_moss_image_edit), without the optional
# prompt upsampler:
#
#   text to image   25 steps, CFG 2.5, er_sde + simple, uncond skips block 22 for the first half
#   image to image  30 steps, CFG 4, euler + normal, uncond skips block 22 throughout; the source
#                   at ~1 MP is the reference (its keys and values computed once), a fixed
#                   negative prompt, the edit LoRA
#
# The pipe is reloaded when the mode changes (extra_key), since only edits carry the LoRA.
#
# OFFLOADER-ONLY: the transformer, the text encoder and the VAE are ComfyUI's own models, which the
# offloader runs behind the pipeline's calls (its "moss_image" model, comfy_vae), and so is the
# sampler (comfy_sampler). The pipeline is moss-image's MOSSPipe on them. An edit amplifies the
# smallest difference in its inputs into another picture (the VAE's fp16 against bf16 is enough),
# so the reference must be ComfyUI's encode exactly, as the decode is.

import os

from . import moss_image as stock   # cheap: no torch at top level
from . import _comfy  # cheap: no torch at top level

ID   = "comfy-moss-image"
TYPE = "moss-image"

PIPELINE    = stock.PIPELINE
TRANSFORMER = stock.TRANSFORMER  # ComfyUI's runs in its place

MODES = ("text-to-image", "image-to-image")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", {"text-to-image": 2.5, "image-to-image": 4.0})

INFERENCE = {"text-to-image": 25, "image-to-image": 30}

# Each template's KSampler, its negative prompt, and the share of the run
# SkipLayerGuidanceDiTSimple skips block 22 in (start_percent, end_percent).
TEMPLATE = {
    "text-to-image": {"sampler": "er_sde", "scheduler": "simple", "negative": "",
                      "skip": (0.0, 0.5)},
    "image-to-image": {"sampler": "euler", "scheduler": "normal",
                       "negative": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact",
                       "skip": (0.0, 1.0)},
}

# ComfyUI's files as its templates name them, at the root of Comfy-Org's repo (no split_files/).
COMFY = {
    "repository": "Comfy-Org/MOSS-Image",
    "revision": "main",
    "components": [{"role": role, "path": path, "filename": path} for role, path in (
        ("transformer",  "diffusion_models/moss_image_sft_int8_convrot.safetensors"),
        ("text_encoder", "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"),
        ("vae",          "vae/flux2-vae_small.safetensors"),
        ("lora",         "loras/moss_image_edit_lora_rank_64_bf16.safetensors"),
    )],
}

# The official repo, for the pipeline file and the transformer's config. Weight-free.
SCAFFOLD = dict(stock.MODEL, allow_patterns=[
    "model_index.json",
    "pipeline_moss_image.py",
    "transformer/config.json",
])


def _pipeline_class(scaffold, mode, sampler):
    """moss-image's MOSSPipe in `mode`'s template: encoding through ComfyUI's text encoder as
    CLIPTextEncode does, decoding through ComfyUI's VAE as VAEDecode does, the uncond skipping
    block 22 within the template's sigmas, the KSampler's seed and, for an edit, the source as
    the reference."""
    template = TEMPLATE[mode]
    start, end = template["skip"]

    class ComfyMOSSPipe(stock._pipeline_class(scaffold)):
        negative_prompt = template["negative"]
        skip_sigmas     = (sampler.percent_to_sigma(end), sampler.percent_to_sigma(start))

        def encode_prompt(self, prompt, device=None):
            return self.text_encoder.encode(prompt).to(device or self._execution_device), None

        def _decode(self, latents):
            images = self.vae.decode(latents, return_dict=False)[0]
            return self.image_processor.postprocess(images, output_type="pil")

        def _before_sampling(self, device):
            self.transformer.extra_conds(device)

        def __call__(self, prompt, image=None, generator=None, **kwargs):
            self.scheduler.seed = None if generator is None else generator.initial_seed()

            if not image:
                return super().__call__(prompt, generator=generator, **kwargs)

            with self.transformer.edit(image, self.vae):
                return super().__call__(prompt, generator=generator, **kwargs)

    return ComfyMOSSPipe


def extra_key(params):
    return (params["mode"],)


def load(ctx, params):
    """Assemble the pipeline of _pipeline_class from ComfyUI's single files via the offloader.
    OFFLOADER-ONLY."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json)
    files    = _comfy.by_role(scaffold)
    mode     = params["mode"]

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader): ComfyUI's models "
                           "run on it" % ID)

    sampler = ctx.backend.comfy_sampler("moss_image", TEMPLATE[mode]["sampler"],
                                        TEMPLATE[mode]["scheduler"])
    pipeline_cls = _pipeline_class(scaffold, mode, sampler)

    loras = list(ctx.loras)
    if mode == "image-to-image":
        loras.insert(0, (files["lora"], 1.0))

    return ctx.backend.load_pipe_comfy(
        pipeline_cls,
        {"model": "moss_image", "file": files["transformer"],
         "config": pipeline_cls.transformer_class.load_config(os.path.join(scaffold,
                                                                           "transformer"))},
        {"model": "moss_image", "file": files["text_encoder"]},
        {"scheduler": sampler, "tokenizer": None,
         "vae": ctx.backend.comfy_vae(files["vae"], {})},
        ctx.dtype, device=ctx.device, lora_files=loras or None)

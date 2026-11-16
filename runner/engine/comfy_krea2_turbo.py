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

# comfy-krea2-turbo -- text-to-image on Krea 2 Turbo, REUSING a ComfyUI install.
#
# Krea 2 Turbo is a distilled single-stream MMDiT (~12B): few-step (8), CFG-free (guidance 0). This
# engine runs it on ComfyUI's split single files, both SCALED-FP8 and kept fp8, dequantized per
# forward as ComfyUI does (emulated on a GPU without fp8 tensor cores): the ~13GB Krea2
# transformer and the ~9GB Qwen3-VL-4B text encoder. The VAE is ComfyUI's own, the same file as
# comfy-qwen (the offloader's comfy_vae, see comfy_qwen_image_edit_2511._build_vae). Only the
# scheduler + configs come from the diffusers repo.
#
# GATED: that scaffold sits in krea/Krea-2-Turbo, which needs a Hugging Face token from an account
# that accepted Krea's license (install.sh takes it as its last argument). The repos of the
# weights (Comfy-Org) are open.
#
# The transformer and the text encoder are ComfyUI's own models, which the offloader runs behind
# the pipeline's calls (its "krea2" model), the files loaded as they are: the pipeline encodes as
# ComfyUI's CLIPTextEncode does (_pipeline_class).
#
# OFFLOADER-ONLY: the fp8 quant path needs comfy ops, so load() bails out for other offload modes.
#
# install:  runner.install --comfy <ComfyUI dir> reuses the three single files (downloading only
#           the missing ones from their Comfy-Org repos) + the scaffold into engine/<id>.
# check:    runner.check treats a COMFY engine installed when engine.json + its files + scaffold.
#
# IMPORTANT (see engine/__init__.py): no torch/diffusers import at top level -- discovery stays
# cheap. The heavy imports live inside load() and the meta-builders.

import os
import json

from . import comfy_qwen_image_edit_2511 as qe  # cheap: no torch at top level

ID   = "comfy-krea2-turbo"
TYPE = "krea2"  # offloader seam vocabulary (single-stream Krea2 MMDiT family)

PIPELINE    = "diffusers:Krea2Pipeline"
TRANSFORMER = "diffusers:Krea2Transformer2DModel"  # ComfyUI's runs in its place

MODES = ("text-to-image",)
CFG   = ("guidance_scale", 0.0)  # distilled / CFG-free

INFERENCE = 8

# GATED (above): a host asks for the Hugging Face token before it installs this engine.
GATED = True

# ComfyUI split single files reused by this engine. The transformer + text encoder live in
# Comfy-Org/Krea-2 at plain paths (not split_files/), so each sets `filename`; the VAE is the same
# file comfy-qwen reuses (Comfy-Org/Qwen-Image_ComfyUI, split_files/ layout -> filename default).
COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": "Comfy-Org/Krea-2",
         "path": "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
         "filename": "diffusion_models/krea2_turbo_fp8_scaled.safetensors"},
        {"role": "text_encoder", "repository": "Comfy-Org/Krea-2",
         "path": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
         "filename": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"},
        {"role": "vae", "repository": "Comfy-Org/Qwen-Image_ComfyUI",
         "path": "vae/qwen_image_vae.safetensors"},
    ],
}

# The diffusers scaffolding fetched from the Krea repo: the scheduler and the configs the pipeline
# reads (the VAE weights are reused from ComfyUI). Revision pinned.
SCAFFOLD = {
    "repository": "krea",
    "model": "Krea-2-Turbo",
    "revision": "1161245028ef398cd0a951101b2bbf486464f841",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "vae/config.json",
        "transformer/config.json",
    ],
}


def _by_role(engine_dir):
    """{role: absolute path} for each reused ComfyUI file. Reads the engine.json comfy record
    (written by install: {root, external, components, scaffold}) and resolves each component's
    root-relative `path` (e.g. models/vae/qwen_image_vae.safetensors) against `root`."""
    with open(os.path.join(engine_dir, "engine.json")) as f:
        comfy = json.load(f)["comfy"]

    return {c["role"]: os.path.join(comfy["root"], *c["path"].split("/"))
            for c in comfy["components"]}


def _pipeline_class():
    """Krea2Pipeline encoding through ComfyUI's text encoder, as ComfyUI's CLIPTextEncode does; an
    engine building on this one passes reference images too. Built here so diffusers stays out of
    engine discovery."""
    import torch
    from diffusers import Krea2Pipeline

    class Krea2Pipe(Krea2Pipeline):
        def encode_prompt(self, prompt=None, device=None, num_images_per_prompt=1,
                          prompt_embeds=None, prompt_embeds_mask=None, max_sequence_length=512,
                          image=None):
            if prompt_embeds is None:
                prompt_embeds, prompt_embeds_mask = self.text_encoder.encode(prompt, images=image)

            # The offloader drops a mask with nothing masked; the stock call wants one.
            if prompt_embeds_mask is None:
                prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.bool)

            device = device or self._execution_device

            return prompt_embeds.to(device), prompt_embeds_mask.to(device)

    return Krea2Pipe


def load(ctx, params, pipeline_cls=None):
    """Assemble the Krea2Pipeline of _pipeline_class (or `pipeline_cls`, a subclass of it: an
    engine building on this one) from ComfyUI's single files via the disk-stream offloader.
    OFFLOADER-ONLY: both big models are scaled-fp8 and need the comfy quant path, so bail out for
    other offload modes. Build the small resident components (VAE, scheduler) from the scaffold
    here, then hand the backend the transformer and the text encoder by their ComfyUI model."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json)
    files    = _by_role(scaffold)

    if ctx.backend is None:
        engine_id = os.path.basename(os.path.normpath(scaffold))  # engine/<id> -> <id>
        raise RuntimeError(
            "%s requires an offload backend (offload=offloader); both big models are fp8 and need "
            "the comfy quant path" % engine_id)

    from diffusers import Krea2Transformer2DModel, FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")
    vae = qe._build_vae(scaffold, files["vae"], ctx.backend)  # ComfyUI's own

    # The distilled/patch config, read straight from the scaffold's model_index.json.
    with open(os.path.join(scaffold, "model_index.json")) as f:
        index = json.load(f)

    return ctx.backend.load_pipe_comfy(
        pipeline_cls or _pipeline_class(),
        {"model": "krea2", "file": files["transformer"],
         "config": Krea2Transformer2DModel.load_config(os.path.join(scaffold, "transformer"))},
        {"model": "krea2", "file": files["text_encoder"]},
        {"scheduler": scheduler, "tokenizer": None, "vae": vae,
         "is_distilled": index["is_distilled"], "patch_size": index["patch_size"]},
        ctx.dtype, device=ctx.device, lora_files=list(ctx.loras) or None)

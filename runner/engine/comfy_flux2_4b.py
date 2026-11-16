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

# comfy-flux2-4b engine: text2img + img2img on FLUX.2-klein-4B, REUSING a ComfyUI install.
#
# Same weights as the flux2-4b engine (same repo, same pinned revision), but loaded from ComfyUI's
# split single files instead of the diffusers repo, so a user who already runs ComfyUI does not
# re-download ~15GB: the bf16 klein transformer and the Qwen3-4B text encoder, the same file
# ComfyUI holds for Z-Image (text_encoders/qwen_3_4b.safetensors).
#
# OFFLOADER-ONLY: the transformer and the text encoder are ComfyUI's own models, which the
# offloader runs behind the pipeline's calls (its "flux2" model), the files loaded as they are; the
# pipeline encodes as ComfyUI's CLIPTextEncode does (_pipeline_class).
#
# The VAE is reused too. AutoencoderKLFlux2 has no from_single_file mapping in diffusers, but it
# needs none: ComfyUI's flux2-vae.safetensors is ALREADY in diffusers key layout (251/251 keys,
# identical shapes, verified), so _build_vae just instantiates from the scaffold config and
# load_state_dict's the file.
#
# install:  runner.install --comfy <ComfyUI dir> resolves/downloads the components and writes
#           engine.json + scaffold into engine/<id> (dispatched on this COMFY declaration).
# check:    runner.check treats a COMFY engine as installed when engine.json + its files + the
#           scaffold are present (no HF revision marker).
#
# IMPORTANT (see engine/__init__.py): no torch/diffusers import at top level, discovery stays
# cheap. The heavy imports live inside load().

import os
import json

ID   = "comfy-flux2-4b"
TYPE = "flux2"  # speak the offloader seam's vocabulary (same pipeline family as flux2-4b)

PIPELINE    = "diffusers:Flux2KleinPipeline"
TRANSFORMER = "diffusers:Flux2Transformer2DModel"  # ComfyUI's runs in its place

MODES = ("text-to-image", "image-to-image")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", 0.0)              # distilled / CFG-free

INFERENCE = 4

# ComfyUI split single files reused by this engine. `path` is the file's location under ComfyUI's
# models/, `role` the diffusers component it feeds. The transformer sits at the ROOT of the BFL
# repo (not under split_files/), so it names `filename` explicitly; the text encoder comes from
# Comfy-Org's split_files/ layout, so its filename defaults. Fetched only when missing.
COMFY = {
    "revision": "main",
    "components": [
        {"role": "transformer", "repository": "black-forest-labs/FLUX.2-klein-4B",
         "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
         "path": "diffusion_models/flux-2-klein-4b.safetensors",
         "filename": "flux-2-klein-4b.safetensors"},
        {"role": "text_encoder", "repository": "Comfy-Org/z_image_turbo",
         "path": "text_encoders/qwen_3_4b.safetensors"},
        {"role": "vae", "repository": "Comfy-Org/flux2-dev",
         "path": "vae/flux2-vae.safetensors"},
    ],
}

# The diffusers scaffolding fetched from the same repo (and revision) flux2-4b pins: configs +
# scheduler. Weight-free: vae/config.json only, never the VAE weights (reused from ComfyUI).
SCAFFOLD = {
    "repository": "black-forest-labs",
    "model": "FLUX.2-klein-4B",
    "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "transformer/config.json",
        "vae/config.json",
    ],
}


def _by_role(engine_dir):
    """{role: absolute path} for each reused ComfyUI file. Reads the engine.json comfy record
    (written by install: {root, external, components, scaffold}) and resolves each component's
    root-relative `path` (e.g. models/diffusion_models/flux-2-klein-4b.safetensors) against
    `root`."""
    with open(os.path.join(engine_dir, "engine.json")) as f:
        comfy = json.load(f)["comfy"]

    return {c["role"]: os.path.join(comfy["root"], *c["path"].split("/"))
            for c in comfy["components"]}


def _build_vae(scaffold, weight_file, dtype):
    """Instantiate the Flux2 VAE from the scaffold config and load ComfyUI's single file.
    AutoencoderKLFlux2 has no from_single_file mapping in diffusers, but needs none: the ComfyUI
    file is already in diffusers key layout (251/251 keys, identical shapes, verified), so it
    binds as-is. `.to(dtype)` leaves the integer bn.num_batches_tracked buffer alone."""
    import safetensors.torch as safetensors_torch
    from diffusers import AutoencoderKLFlux2

    cfg = AutoencoderKLFlux2.load_config(os.path.join(scaffold, "vae"))
    vae = AutoencoderKLFlux2.from_config(cfg)

    vae.load_state_dict(safetensors_torch.load_file(weight_file), strict=True)

    return vae.to(dtype).eval()


def _pipeline_class():
    """Flux2KleinPipeline encoding through ComfyUI's text encoder, as ComfyUI's CLIPTextEncode
    does. Built here so diffusers stays out of engine discovery."""
    from diffusers import Flux2KleinPipeline

    class Flux2Pipe(Flux2KleinPipeline):
        def encode_prompt(self, prompt, device=None, num_images_per_prompt=1, prompt_embeds=None,
                          max_sequence_length=512, text_encoder_out_layers=(9, 18, 27)):
            if prompt_embeds is None:
                prompt_embeds = self.text_encoder.encode(prompt)

            prompt_embeds = prompt_embeds.to(device or self._execution_device)

            return prompt_embeds, self._prepare_text_ids(prompt_embeds)

    return Flux2Pipe


def load(ctx, params, build_vae=_build_vae):
    """Build the Flux2KleinPipeline of _pipeline_class from ComfyUI's single files, the
    transformer and the text encoder as ComfyUI's own models on the offloader. OFFLOADER-ONLY.
    The VAE (`build_vae`, an engine building on this one passes its own) and the scheduler come
    from the scaffold."""
    scaffold = ctx.model  # engine/<id>/ (scaffolding + engine.json)
    files    = _by_role(scaffold)

    if ctx.backend is None:
        engine_id = os.path.basename(os.path.normpath(scaffold))  # engine/<id> -> <id>
        raise RuntimeError("%s requires an offload backend (offload=offloader): ComfyUI's models "
                           "run on it" % engine_id)

    from diffusers import Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")
    vae = build_vae(scaffold, files["vae"], ctx.dtype)

    return ctx.backend.load_pipe_comfy(
        _pipeline_class(),
        {"model": "flux2", "file": files["transformer"],
         "config": Flux2Transformer2DModel.load_config(os.path.join(scaffold, "transformer"))},
        {"model": "flux2", "file": files["text_encoder"]},
        {"scheduler": scheduler, "tokenizer": None, "vae": vae, "is_distilled": True},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

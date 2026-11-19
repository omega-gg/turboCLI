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

# comfy-z-image-turbo engine: text2img on Z-Image-Turbo, REUSING a ComfyUI install's model files.
#
# Same weights as the z-image-turbo engine, but ComfyUI stores them as three split single-file
# safetensors (a diffusion model, a Qwen3-4B text encoder, a VAE) rather than a diffusers repo, so
# a user who already runs ComfyUI does not re-download ~20GB.
#
# OFFLOADER-ONLY: the transformer and the text encoder are ComfyUI's own models, which the
# offloader runs behind the pipeline's calls (its "lumina2" model), the files loaded as they are;
# the pipeline encodes as ComfyUI's CLIPTextEncode does (_pipeline_class). The VAE is diffusers'
# AutoencoderKL on ComfyUI's ae.safetensors.
#
# install:  runner.install --comfy <ComfyUI dir> resolves/downloads the components and writes
#           engine.json + scaffold into engine/<id> (dispatched on this COMFY declaration).
# check:    runner.check treats a COMFY engine as installed when engine.json + its files + the
#           scaffold are present (no HF revision marker).
#
# IMPORTANT (see engine/__init__.py): no torch/diffusers import at top level, discovery stays
# cheap. The heavy imports live inside load().

from . import _comfy  # cheap: no torch at top level

ID   = "comfy-z-image-turbo"
TYPE = "z-image"  # speak the offloader seam's vocabulary (same pipeline family as z-image-turbo)

PIPELINE    = "diffusers:ZImagePipeline"
TRANSFORMER = "diffusers:ZImageTransformer2DModel"  # ComfyUI's runs in its place

MODES = ("text-to-image",)
CFG   = ("guidance_scale", 0.0)

INFERENCE = 8

# ComfyUI split single files reused by this engine. `path` is the file's location under ComfyUI's
# models/ (and, under split_files/, its in-repo path too), `role` the diffusers component it feeds.
# Fetched only when missing, from ComfyUI's own published repo.
COMFY = {
    "repository": "Comfy-Org/z_image_turbo",
    "revision": "main",
    "components": [
        {"role": "transformer",  "path": "diffusion_models/z_image_turbo_bf16.safetensors"},
        {"role": "text_encoder", "path": "text_encoders/qwen_3_4b.safetensors"},
        {"role": "vae",          "path": "vae/ae.safetensors"},
    ],
}

# The diffusers scaffolding fetched from the repo z-image-turbo pins: the scheduler and the VAE's
# config, no weights.
SCAFFOLD = {
    "repository": "Tongyi-MAI",
    "model": "Z-Image-Turbo",
    "revision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
    "allow_patterns": [
        "model_index.json",
        "scheduler/*",
        "vae/config.json",
    ],
}


def _pipeline_class(base="ZImagePipeline"):
    """The diffusers Z-Image `base` pipeline encoding through ComfyUI's text encoder, as ComfyUI's
    CLIPTextEncode does. Built here so diffusers stays out of engine discovery."""
    import diffusers

    class ZImagePipe(getattr(diffusers, base)):
        def _encode_prompt(self, prompt, device=None, prompt_embeds=None,
                           max_sequence_length=512):
            if prompt_embeds is not None:
                return prompt_embeds

            prompts = [prompt] if isinstance(prompt, str) else prompt
            device = device or self._execution_device

            return [self.text_encoder.encode(p)[0].to(device) for p in prompts]

        if base == "ZImageImg2ImgPipeline":
            def prepare_latents(self, image, timestep, batch_size, channels, height, width,
                                dtype, device, generator, latents=None):
                # ComfyUI's VAEEncode takes the latent's mean and leaves the seed to the noise;
                # diffusers samples the latent, drawing on the generator first. The pipeline takes
                # a latent as it is.
                vae = self.vae
                mean = vae.encode(image.to(device, dtype)).latent_dist.mode()
                image = (mean - vae.config.shift_factor) * vae.config.scaling_factor

                return super().prepare_latents(image, timestep, batch_size, channels, height,
                                               width, dtype, device, generator, latents)

    return ZImagePipe


def load(ctx, params):
    return _build(ctx, _pipeline_class())


def _build(ctx, pipeline):
    """Build `pipeline` (a _pipeline_class) from ComfyUI's single files, the transformer and the
    text encoder as ComfyUI's own models on the offloader. OFFLOADER-ONLY. The VAE and the
    scheduler come from the scaffold."""
    scaffold = ctx.model  # engine/comfy-z-image-turbo/ (scaffolding + engine.json)
    files    = _comfy.by_role(scaffold)

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader): ComfyUI's models "
                           "run on it" % ID)

    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")
    vae = AutoencoderKL.from_single_file(files["vae"], config=scaffold, subfolder="vae",
                                         torch_dtype=ctx.dtype, local_files_only=True)

    return ctx.backend.load_pipe_comfy(
        pipeline,
        {"model": "lumina2", "file": files["transformer"], "config": {}},
        {"model": "lumina2", "file": files["text_encoder"]},
        {"scheduler": scheduler, "tokenizer": None, "vae": vae},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

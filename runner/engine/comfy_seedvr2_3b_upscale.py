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

# comfy-seedvr2-3b-upscale engine: image-to-image that upscales its first input to the output size
# with SeedVR2 3B, a one-step diffusion restorer, REUSING a ComfyUI install file for file, as
# ComfyUI's SeedVR2 template runs it: the int8 ConvRot transformer and the fp16 VAE (3.96 GB).
# Apache-2.0. No prompt: the text is two embeddings the transformer's file holds.
#
# The template (utility_seedvr2_3b_int8_upscale_image): the input resized to the output size
# (lanczos), SeedVR2Preprocess (RGB, zero padded right and bottom to the 16 grid), VAEEncodeTiled
# 512/128, SeedVR2Conditioning (the latent and a ones mask), KSampler at 1 step, cfg 1, euler +
# simple, VAEDecodeTiled 512/128, SeedVR2PostProcessing (cropped back, no color correction). Core
# puts the input's alpha back, as the template joins it.
#
# OFFLOADER-ONLY: the transformer and the VAE are ComfyUI's own models, which the offloader runs
# behind the pipeline's calls (its "seedvr2" model, comfy_vae), and so is the sampler
# (comfy_sampler). The pipeline is a short loop of its own: diffusers has no SeedVR2.

from . import _comfy  # cheap: no torch at top level

ID   = "comfy-seedvr2-3b-upscale"
TYPE = "seedvr2"

TRANSFORMER = "seedvr2"  # ComfyUI's NaDiT, no diffusers class: marks the engine offload-wired

MODES = ("image-to-image",)       # wire values, same as the HTTP API
CFG   = ("guidance_scale", 1.0)   # one step at cfg 1 (the pipe ignores it)

INFERENCE = 1
UPSCALE   = True

# ComfyUI's files as its template names them, at the root of Comfy-Org's repo (no split_files/).
COMFY = {
    "repository": "Comfy-Org/SeedVR2",
    "revision": "df48879708206a403d2a61acd55578c2e80fd233",
    "components": [{"role": role, "path": path, "filename": path} for role, path in (
        ("transformer", "diffusion_models/seedvr2_3b_int8_convrot.safetensors"),
        ("vae",         "vae/seedvr2_ema_vae_fp16.safetensors"),
    )],
}

# VAEEncodeTiled and VAEDecodeTiled's tile size and overlap in the template, in pixels.
TILED = (512, 128)

# SeedVR2Preprocess pads to it (nodes_seedvr.py:62-73).
GRID = 16


def _pipeline_class():
    """ComfySeedVR2Pipe: the template's chain behind the classic pipeline face. Built here so
    diffusers stays out of engine discovery."""
    from types import SimpleNamespace

    import numpy as np
    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image

    class ComfySeedVR2Pipe(DiffusionPipeline):
        model_cpu_offload_seq = "transformer->vae"

        def __init__(self, transformer, scheduler, vae):
            super().__init__()

            self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)

            self._interrupt = False

        def _encode(self, image, width, height, device):
            """ImageScale's lanczos to the output size (comfy.utils.lanczos is PIL's on the 8 bit
            picture), SeedVR2Preprocess's padding, then VAEEncodeTiled: the latent, scaled."""
            pixels = np.asarray(image.convert("RGB").resize((width, height), Image.LANCZOS))

            x = torch.from_numpy(pixels.astype(np.float32) / 255.0).movedim(-1, 0)
            x = torch.nn.functional.pad(x, (0, -width % GRID, 0, -height % GRID))

            return self.vae.encode(x[None, :, None].to(device) * 2.0 - 1.0).latent_dist.mode()

        def _decode(self, latents, width, height):
            """VAEDecodeTiled and its process_output, cropped back to the output size, then
            SaveImage's 8 bits: each value of the [0, 1] picture times 255 and truncated."""
            images = self.vae.decode(latents, return_dict=False)[0][:, :, 0, :height, :width]
            images = images.add(1.0).div(2.0).clamp(0.0, 1.0)

            pixels = 255.0 * images.movedim(1, -1).float().cpu().numpy()

            return [Image.fromarray(p) for p in pixels.clip(0, 255).astype("uint8")]

        def prepare_latents(self, shape, seed, device):
            """KSampler's noise (sample.py:22-38): the seed's CPU float32 draw, scaled by a first
            sigma of 1, which leaves nothing of the latent at denoise 1."""
            return torch.randn(shape, generator=torch.Generator().manual_seed(seed),
                               dtype=torch.float32).to(device)

        @torch.no_grad()
        def __call__(self, image, height=1024, width=1024, num_inference_steps=INFERENCE,
                     generator=None, callback_on_step_end=None, **_):
            device = self._execution_device

            if generator is None:
                generator = torch.Generator()
                generator.seed()
            seed = generator.initial_seed()

            latent = self._encode(image[0], width, height, device)

            # SeedVR2Conditioning (nodes_seedvr.py:395-414): the latent and a ones mask.
            condition = torch.cat([latent, torch.ones_like(latent[:, :1])], dim=1)

            scheduler = self.scheduler
            scheduler.seed = seed
            scheduler.set_timesteps(num_inference_steps, device=device)

            latents = self.prepare_latents(latent.shape, seed, device)

            with self.progress_bar(total=len(scheduler.timesteps)) as bar:
                for i, t in enumerate(scheduler.timesteps):
                    v = self.transformer(latents, t, condition)[0]

                    latents = scheduler.step(v, t, latents)[0]

                    if callback_on_step_end is not None:
                        callback_on_step_end(self, i, t, {})

                    bar.update()

                    if self._interrupt:
                        return SimpleNamespace(images=[])

            return SimpleNamespace(images=self._decode(latents, width, height))

    return ComfySeedVR2Pipe


def load(ctx, params):
    """Assemble the pipeline of _pipeline_class from ComfyUI's single files via the offloader.
    OFFLOADER-ONLY."""
    files = _comfy.by_role(ctx.model)  # engine/<id>/ (engine.json)

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader): ComfyUI's models "
                           "run on it" % params["engine"])

    return ctx.backend.load_pipe_comfy(
        _pipeline_class(),
        {"model": "seedvr2", "file": files["transformer"], "config": {}},
        None,
        {"scheduler": ctx.backend.comfy_sampler("seedvr2", "euler", "simple",
                                                files["transformer"]),
         "vae": ctx.backend.comfy_vae(files["vae"], {}, TILED)},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

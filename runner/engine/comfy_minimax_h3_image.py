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

# comfy-minimax-h3-image engine: text2img + img2img on MiniMax-H3-Image (33.1B DiT, 6 steps, no
# CFG), REUSING a ComfyUI install file for file, as ComfyUI's two templates run it: the int8
# ConvRot transformer, H3's Qwen3-VL 32B int8 ConvRot text encoder and the fp16 image VAE
# (49.5 GB). Only the official repo's index comes from MiniMax's repo (the scaffold).
#
# LICENSE: the MiniMax-H3-Image Community License. It excludes the EU, the UK, South Korea and the
# US, and asks for MiniMax's written authorization above $1M of annual revenue or for a hosted
# service.
#
# Each mode is its template (image_minimax_h3_image_t2i / _edit), without the prompt expander:
# MiniMaxH3ImageConditioning, then KSampler at 6 steps, CFG 1, euler + simple, on the PDD grid the
# file declares. An edit's source, at 4 MP on the 32 grid, is <Picture 1>: the text encoder sees
# it, and its latent is the reference. As for MOSS, an edit renders at the requested size (the
# template takes the scaled source's).
#
# OFFLOADER-ONLY: the transformer, the text encoder and the VAE are ComfyUI's own models, which the
# offloader runs behind the pipeline's calls (its "minimax_h3_image" model, comfy_vae), and so is
# the sampler (comfy_sampler). The pipeline is a short loop of its own: diffusers' H3 transformer
# takes packed rows and position ids, ComfyUI's packs them itself.

from . import minimax_h3_image as stock  # cheap: no torch at top level
from . import _comfy  # cheap: no torch at top level

ID   = "comfy-minimax-h3-image"
TYPE = "minimax-h3-image"

PIPELINE    = stock.PIPELINE
TRANSFORMER = stock.TRANSFORMER  # ComfyUI's runs in its place

MODES = ("text-to-image", "image-to-image")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", 1.0)              # guidance-distilled (the pipe ignores it)

# One output head per step: the file's PDD grid has 6 intervals.
INFERENCE = 6

# ComfyUI's files as its templates name them, at the root of Comfy-Org's repo (no split_files/).
# The templates fetch the text encoder from Comfy-Org/MiniMax-H3, which holds the same file.
COMFY = {
    "repository": "Comfy-Org/MiniMax-H3-Image",
    "revision": "89c635085f539d03ad35485ed542c5a286e3c6d6",
    "components": [{"role": role, "path": path, "filename": path} for role, path in (
        ("transformer",  "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"),
        ("text_encoder", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
        ("vae",          "vae/minimax_h3_image_vae_fp16.safetensors"),
    )],
}

# The official repo, for its index. Weight-free: ComfyUI reads everything else off its files.
SCAFFOLD = dict(stock.MODEL, allow_patterns=["model_index.json"])


def _pipeline_class():
    """ComfyH3Pipe: MiniMaxH3ImageConditioning, then KSampler and VAEDecode, as the templates chain
    them, behind the classic pipeline face. Built here so diffusers stays out of engine
    discovery."""
    from types import SimpleNamespace

    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image

    class ComfyH3Pipe(DiffusionPipeline):
        model_cpu_offload_seq = "text_encoder->transformer->vae"

        def __init__(self, transformer, text_encoder, scheduler, vae):
            super().__init__()

            self.register_modules(transformer=transformer, text_encoder=text_encoder,
                                  scheduler=scheduler, vae=vae)

            self._interrupt = False

        def encode_prompt(self, prompt, sources=(), device=None):
            """The conditioning and its token tags; an edit's sources are part of them."""
            context, tags = self.text_encoder.encode(prompt, sources)

            return context.to(device or self._execution_device), tags

        def _decode(self, latents):
            """VAEDecode, then SaveImage's 8 bits: ComfyUI's VAE decodes the frame straight to
            [0, 1], each value times 255 and truncated (nodes.py's SaveImage)."""
            images = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

            pixels = (255.0 * images.movedim(1, -1).float().cpu().numpy()).clip(0, 255)

            return [Image.fromarray(p) for p in pixels.astype("uint8")]

        @torch.no_grad()
        def __call__(self, prompt, image=None, height=1024, width=1024,
                     num_inference_steps=INFERENCE, generator=None, callback_on_step_end=None,
                     **_):
            device = self._execution_device

            if generator is None:
                generator = torch.Generator()
                generator.seed()
            seed = generator.initial_seed()

            scheduler = self.scheduler
            scheduler.seed = seed
            scheduler.set_timesteps(num_inference_steps, device=device)

            # KSampler's noise (sample.py:22-38): the seed's CPU float32 draw over the empty
            # latent MiniMaxH3ImageConditioning returns, scaled by a first sigma of 1.
            shape = (1, 24, 1, height // 16, width // 16)
            latents = torch.randn(shape, generator=torch.Generator().manual_seed(seed),
                                  dtype=torch.float32).to(device)

            # MiniMaxH3ImageConditioning in its order: an edit's references, then the text with
            # their pictures (no image: the text alone).
            with self.transformer.edit(image or (), self.vae) as sources:
                context, tags = self.encode_prompt(prompt, sources, device=device)
                conds = self.transformer.extra_conds(context, tags, shape, seed, device)

                with self.progress_bar(total=len(scheduler.timesteps)) as bar:
                    for i, t in enumerate(scheduler.timesteps):
                        v = self.transformer(latents, t, sample_sigmas=scheduler.sigmas,
                                             **conds)[0]

                        latents = scheduler.step(v, t, latents)[0]

                        if callback_on_step_end is not None:
                            callback_on_step_end(self, i, t, {})

                        bar.update()

                        if self._interrupt:
                            return SimpleNamespace(images=[])

            return SimpleNamespace(images=self._decode(latents))

    return ComfyH3Pipe


def load(ctx, params):
    """Assemble the pipeline of _pipeline_class from ComfyUI's single files via the offloader.
    OFFLOADER-ONLY."""
    files = _comfy.by_role(ctx.model)  # engine/<id>/ (scaffolding + engine.json)

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader): ComfyUI's models "
                           "run on it" % ID)

    return ctx.backend.load_pipe_comfy(
        _pipeline_class(),
        {"model": "minimax_h3_image", "file": files["transformer"], "config": {}},
        {"model": "minimax_h3_image", "file": files["text_encoder"]},
        {"scheduler": ctx.backend.comfy_sampler("minimax_h3_image", "euler", "simple",
                                                files["transformer"]),
         "vae": ctx.backend.comfy_vae(files["vae"], {})},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

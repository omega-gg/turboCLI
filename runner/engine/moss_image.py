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

# moss-image engine: text2img on MOSS-Image SFT (~5B, Qwen3.5 9B text encoder, the FLUX.2 small
# VAE) from OpenMOSS's repo, its weights as published (bf16), run by the official pipeline with or
# without the offloader. comfy-moss-image is the same model from ComfyUI's files; it builds on this
# engine's pipe.
#
# Text to image only: OpenMOSS publishes no edit model, and its pipeline takes no reference
# images.
#
# The pipeline is the repo's own pipeline_moss_image.py (Apache-2.0), loaded from the pinned
# install. MOSSPipe below keeps its components and encode, and gives it the loop the runner drives:
# a progress_bar it hooks, a step callback for cancel. The loop is the official one (CFG in
# float32, the uncond pass skipping block 22), and comfy-moss-image sets its seams: the uncond's
# skip window (by sigma), the negative prompt, the decode and what to prepare before sampling.

import os

ID   = "moss-image"
TYPE = "moss-image"

PIPELINE    = "pipeline_moss_image:MOSSImagePipeline"     # the repo's own, behind MOSSPipe
TRANSFORMER = "pipeline_moss_image:MOSSImageTransformer"  # offloader disk-stream meta-load

MODES = ("text-to-image",)        # wire values, same as the HTTP API
CFG   = ("guidance_scale", 3.0)   # the official default

INFERENCE = 50

MODEL = {
    "repository": "OpenMOSS-Team",
    "model": "MOSS-Image-SFT",
    "revision": "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9",
    "allow_patterns": [
        "model_index.json",
        "pipeline_moss_image.py",
        "scheduler/*",
        "transformer/*",
        "text_encoder/*",
        "tokenizer/*",
        "vae/*",
    ],
}


def official(folder):
    """The repo's pipeline_moss_image module, from `folder` (the install, or a scaffold)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline_moss_image",
                                                  os.path.join(folder, "pipeline_moss_image.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pipeline_class(folder):
    """MOSSPipe: the official MOSSImagePipeline of `folder` with the runner's loop. Built here so
    diffusers stays out of engine discovery."""
    from types import SimpleNamespace

    import torch
    from diffusers import AutoencoderKLFlux2
    from diffusers.utils.torch_utils import randn_tensor
    from transformers import AutoModel, AutoTokenizer

    moss = official(folder)

    class MOSSPipe(moss.MOSSImagePipeline):
        transformer_class = moss.MOSSImageTransformer

        negative_prompt = ""      # the official defaults: an empty negative prompt,
        skip_layers     = (22,)   # block 22 skipped by the uncond pass,
        skip_sigmas     = (0, 1)  # over the whole run

        @classmethod
        def from_pretrained(cls, model, transformer=None, torch_dtype=None,
                            local_files_only=True, **kwargs):
            """The repo's components as its from_components_pretrained loads them, the VAE in
            float32 (a streamed `transformer` may come in place of its own)."""
            if transformer is None:
                transformer = cls.transformer_class.from_pretrained(
                    model, subfolder="transformer", torch_dtype=torch_dtype,
                    local_files_only=local_files_only, **kwargs)

            text_encoder = AutoModel.from_pretrained(model, subfolder="text_encoder",
                                                     dtype=torch_dtype,
                                                     local_files_only=local_files_only)
            tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer",
                                                      local_files_only=local_files_only)
            vae = AutoencoderKLFlux2.from_pretrained(model, subfolder="vae",
                                                     torch_dtype=torch.float32,
                                                     local_files_only=local_files_only, **kwargs)
            scheduler = moss.MOSSImageScheduler.from_pretrained(model, subfolder="scheduler")

            return cls(transformer.eval(), scheduler, text_encoder.eval(), tokenizer, vae.eval())

        def encode_prompt(self, prompt, device=None):
            return super().encode_prompt(prompt)

        def _decode(self, latents):
            """The official decode: the FLUX.2 VAE's statistics back on, its 2x2 patches
            unfolded."""
            z = latents.to(dtype=self.vae.dtype)
            bn = self.vae.bn
            eps = self.vae.config.batch_norm_eps
            z = (z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1)
                 + bn.running_mean.to(z).view(1, -1, 1, 1))
            b, c, h, w = z.shape
            z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
            images = self.vae.decode(z.reshape(b, c // 4, 2 * h, 2 * w), return_dict=False)[0]
            return self.image_processor.postprocess(images.clamp(-1, 1), output_type="pil")

        def _before_sampling(self, device):
            """Nothing here; comfy-moss-image computes an edit's reference keys and values, before
            the progress bar as ComfyUI does, so sampling times compare."""

        @torch.no_grad()
        def __call__(self, prompt, height=1024, width=1024, num_inference_steps=INFERENCE,
                     guidance_scale=CFG[1], generator=None, callback_on_step_end=None, **_):
            device = self._execution_device

            positive = self.encode_prompt(prompt)[0]
            negative = None
            if guidance_scale != 1.0:
                negative = self.encode_prompt(self.negative_prompt)[0]

            # The noise in float32, as the official pipeline draws it.
            shape = (1, self.transformer.config.latent_channels, height // 16, width // 16)
            latents = randn_tensor(shape, generator=generator, device=device,
                                   dtype=torch.float32)

            scheduler = self.scheduler
            scheduler.set_timesteps(num_inference_steps, device=device)
            low, high = self.skip_sigmas

            self._before_sampling(device)

            with self.progress_bar(total=len(scheduler.timesteps)) as bar:
                for i, t in enumerate(scheduler.timesteps):
                    v = self.transformer(latents, t, positive, return_dict=False)[0].float()

                    if negative is not None:
                        sigma = float(scheduler.sigmas[i])
                        skip = self.skip_layers if low <= sigma <= high else ()
                        uncond = self.transformer(latents, t, negative, skip_layers=skip,
                                                  return_dict=False)[0].float()
                        v = uncond + guidance_scale * (v - uncond)

                    latents = scheduler.step(v, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_on_step_end(self, i, t, {})

                    bar.update()

                    if getattr(self, "_interrupt", False):
                        return SimpleNamespace(images=[])

            return SimpleNamespace(images=self._decode(latents))

    return MOSSPipe


def load(ctx, params):
    """MOSSPipe from the repo: through the offloader when one is chosen (load_pipe, as the other
    stock engines), else placed by the native offload modes."""
    pipeline_cls = _pipeline_class(ctx.model)

    if ctx.backend is not None:
        return ctx.backend.load_pipe(ctx.model, ctx.dtype, pipeline_cls,
                                     pipeline_cls.transformer_class, device=ctx.device,
                                     lora_files=ctx.loras or None)

    p = pipeline_cls.from_pretrained(ctx.model, torch_dtype=ctx.dtype, use_safetensors=True,
                                     low_cpu_mem_usage=True)

    ctx.apply_loras(p, ctx.loras)

    return ctx.apply_offload(p)

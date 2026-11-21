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

# comfy-minimax-h3 engine: text-to-video and image-to-video on MiniMax H3, a video with its
# soundtrack, REUSING a ComfyUI install file for file, as ComfyUI's templates run it
# (video_minimax_h3_t2v / _i2v): the int8 ConvRot transformer, H3's Qwen3-VL 32B NVFP4 text
# encoder, the int8 ConvRot video VAE and the fp32 audio VAE.
#
# LICENSE: the MiniMax H3 Community License. It excludes the EU, the UK, South Korea and the US,
# asks for MiniMax's written authorization above $20M of yearly revenue, and for "MiniMax H3" on
# the interface of a commercial product.
#
# Each mode is its template with the Turbo LoRA off: MiniMaxH3ImageToVideo (the first and last
# frames), or with picture or audio references video_minimax_h3_r2v's MiniMaxH3ReferenceToVideo
# on the ref2va transformer, the frames then pinned by MiniMaxH3AddGuide; then 20 steps of
# res_multistep + simple at cfg 1 (BasicGuider), VAEDecode and VAEDecodeAudio, CreateVideo at
# 24 fps and SaveVideo (crf 18). An audio guide is a MiniMaxH3AddGuide in either workflow.
# The duration is 24 fps snapped up to the model's 17k+5 frame grid, as the template computes it.
#
# OFFLOADER-ONLY: the transformer, the text encoder and the VAEs are ComfyUI's own models, which
# the offloader runs behind the pipeline's calls (its "minimax_h3" model, comfy_vae), and so is the
# sampler (comfy_sampler). The pipeline is a short loop of its own, on the flat pack of the video
# and audio latents ComfyUI samples.

from . import minimax_h3 as stock  # cheap: no torch at top level
from . import _comfy  # cheap: no torch at top level

ID   = "comfy-minimax-h3"
TYPE = "minimax-h3"

PIPELINE    = stock.PIPELINE
TRANSFORMER = stock.TRANSFORMER  # ComfyUI's runs in its place

MODES = ("text-to-video", "image-to-video")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", 1.0)              # BasicGuider: no CFG (the pipe ignores it)

INFERENCE = 20

# The options a video takes (core checks them), each in place of the engine's for that video:
# cuDNN SDPA first, as the experimental H3 ran ComfyUI, the scheduler, and the keyframes' fit
# (an ImageScale with center crop before the node, or none).
OPTIONS = {"attention": ("cudnn",), "scheduler": ("simple", "beta"), "crop": ("none", "center")}

# ComfyUI's files as its templates name them, at the root of Comfy-Org's repo (no split_files/).
COMFY = {
    "repository": "Comfy-Org/MiniMax-H3",
    "revision": "4b53a35191437de05370da548e59aeabe2a5a09a",
    "components": [{"role": role, "path": path, "filename": path} for role, path in (
        ("transformer",  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
        ("transformer_ref", "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
        ("text_encoder", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
        ("vae",          "vae/minimax_h3_video_vae_int8_convrot.safetensors"),
        ("audio_vae",    "vae/minimax_h3_audio_vae_fp32.safetensors"),
    )],
}

# MiniMaxH3ImageToVideo's sizes are on this grid (its width and height inputs' step).
GRID = 32

FPS = 24

AUDIO_LATENT_FPS = 40


def frames(duration):
    """The template's length (24 fps, at least 5 frames), snapped up to the 17k+5 grid as
    align_frame_count does (nodes_minimax_h3.py:50-53)."""
    n = max(5, round(duration * FPS))

    return n + (5 - n % 17) % 17


extra_key = stock.extra_key  # the workflow, which picks the transformer


def _scale(image, width, height, crop):
    """common_upscale with lanczos (comfy/utils.py:1130-1168) on a PIL picture: `crop` "center"
    cuts it to the target's ratio first. comfy.utils.lanczos resizes the 8 bit picture with PIL."""
    from PIL import Image

    if crop == "center":
        old_width, old_height = image.size
        old_aspect, new_aspect = old_width / old_height, width / height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        image = image.crop((x, y, old_width - x, old_height - y))

    return image.resize((width, height), Image.Resampling.LANCZOS)


def _pipeline_class(default_crop="none"):
    """ComfyH3VideoPipe: MiniMaxH3ImageToVideo, SamplerCustomAdvanced, VAEDecode and
    VAEDecodeAudio, as the templates chain them, behind the classic pipeline face.
    `default_crop` is the engine's keyframe fit, which crop= overrides: "center" first fits each
    keyframe to the canvas as an ImageScale with center crop would (comfy-minimax-h3-hq),
    "none" leaves them to the node. Built here so diffusers stays out of engine discovery."""
    from types import SimpleNamespace

    import numpy as np
    import torch
    from diffusers import DiffusionPipeline

    class ComfyH3VideoPipe(DiffusionPipeline):
        model_cpu_offload_seq = "text_encoder->transformer->vae"

        def __init__(self, transformer, text_encoder, scheduler, vae, audio_vae):
            super().__init__()

            self.register_modules(transformer=transformer, text_encoder=text_encoder,
                                  scheduler=scheduler, vae=vae, audio_vae=audio_vae)

            self._schedule = scheduler.scheduler  # the engine's, which scheduler= overrides

            self._interrupt = False

        def encode_prompt(self, prompt, images=(), device=None, items=None):
            """The conditioning and its token tags; the keyframes, or the references' items, are
            part of them."""
            context, tags = self.text_encoder.encode(prompt, images, items)

            return context.to(device or self._execution_device), tags

        def _keyframe(self, image, width, height, node_crop, crop):
            """A keyframe as MiniMaxH3ImageToVideo sizes it (nodes_minimax_h3.py:80-84, 159-170),
            as an IMAGE [1, H, W, 3] in [0, 1]: stretched for the first frame, cover-cropped for
            the last, after the ImageScale with center crop that `crop` "center" asks for."""
            if crop == "center":
                image = _scale(image, width, height, "center")

            image = _scale(image, width, height, node_crop)

            return torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0)[None]

        def prepare_latents(self, shapes, seed, device):
            """RandomNoise (sample.py:22-38): the seed's CPU float32 draws, the video's then the
            audio's, packed flat (comfy/utils.py:1445-1453) and scaled by a first sigma of 1."""
            generator = torch.Generator().manual_seed(seed)

            noise = [torch.randn(shape, generator=generator, dtype=torch.float32)
                     for shape in shapes]

            return torch.cat([n.reshape(n.shape[0], 1, -1) for n in noise], dim=-1).to(device)

        def __call__(self, *args, attention=None, **kwargs):
            # NOTE: The attention option: cuDNN SDPA first for this generation (attention=cudnn).
            with self.transformer.attention(attention):
                return self._generate(*args, **kwargs)

        @torch.no_grad()
        def _generate(self, prompt, image=None, height=768, width=1344, duration=5.0,
                     num_inference_steps=INFERENCE, generator=None, callback_on_step_end=None,
                     references=(), guide=None, crop=default_crop, scheduler=None, **_):
            # NOTE: The scheduler and crop options replace the engine's for this generation:
            #       scheduler=simple|beta, crop=none|center (the keyframes' fit).
            if width % GRID or height % GRID:
                raise ValueError("MiniMax H3 renders sizes on the %d pixel grid" % GRID)

            device = self._execution_device

            if generator is None:
                generator = torch.Generator()
                generator.seed()
            seed = generator.initial_seed()

            # _empty_av_latent (nodes_minimax_h3.py:56-63, 97-102): the video's and the audio's.
            frame_count = frames(duration)
            latent_t = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
            audio_t = round(frame_count / FPS * AUDIO_LATENT_FPS)
            shapes = [torch.Size([1, 24, latent_t, height // 16, width // 16]),
                      torch.Size([1, 32, 2, audio_t])]

            tf, pinned = self.transformer, list(image or ()) + [None, None]
            refs = None

            if references:
                # MiniMaxH3ReferenceToVideo (nodes_minimax_h3.py:259-381): the references
                # encoded, the prompt with their items, then each frame pinned by AddGuide.
                items, refs = tf.reference_to_video(references, self.vae, self.audio_vae,
                                                    width, height)
                context, tags = self.encode_prompt(prompt, device=device, items=items)
                keyframes = []
                for source, index in zip(pinned, (0, -1)):
                    if source is not None:
                        tf.add_guide(keyframes, shapes, index, vae=self.vae, image=source)
            else:
                # MiniMaxH3ImageToVideo (nodes_minimax_h3.py:150-179): the keyframes, the prompt
                # with them, then each keyframe's latent through the video VAE (one frame).
                keyframes = [{"resolved_frame_index": index,
                              "image": self._keyframe(source, width, height, node_crop, crop)}
                             for source, index, node_crop in zip(pinned, (0, frame_count - 1),
                                                                 ("disabled", "center"))
                             if source is not None]

                context, tags = self.encode_prompt(prompt, [kf["image"] for kf in keyframes],
                                                   device)

                for kf in keyframes:
                    pixels = kf.pop("image").movedim(-1, 1).unsqueeze(2)
                    kf["latent"] = self.vae.encode(pixels * 2.0 - 1.0).latent_dist.mode()

            # MiniMaxH3AddGuide's audio: a recording pinned at a time of the soundtrack.
            if guide is not None:
                tf.add_guide(keyframes, shapes, round(guide[1] * FPS),
                             audio_vae=self.audio_vae, audio=guide[0])

            sampler = self.scheduler
            sampler.scheduler = scheduler or self._schedule
            sampler.seed = seed
            sampler.set_timesteps(num_inference_steps, device=device)

            # NOTE: The noise first: the transformer loads with it, as KSampler's does, then
            #       extra_conds runs on it (samplers.py:1240-1260, 1220-1226).
            latents = self.prepare_latents(shapes, seed, device)

            conds = tf.extra_conds(context, tags, keyframes, shapes, seed, device, refs)

            with self.progress_bar(total=len(sampler.timesteps)) as bar:
                for i, t in enumerate(sampler.timesteps):
                    v = self.transformer(latents, t, sample_sigmas=sampler.sigmas, **conds)[0]

                    latents = sampler.step(v, t, latents)[0]

                    if callback_on_step_end is not None:
                        callback_on_step_end(self, i, t, {})

                    bar.update()

                    if self._interrupt:
                        return SimpleNamespace(videos=[], audio=None, fps=FPS)

            # process_latent_out (model_base.py:2240-2252): the audio back off the video's
            # schedule, then the pack split into its two latents.
            n = shapes[0][1:].numel()
            latents = latents.float()
            latents[..., n:] *= 1.0 / self.transformer.audio_scale
            video = latents[:, :, :n].reshape(shapes[0])
            audio = latents[:, :, n:].reshape(shapes[1])

            # VAEDecode, then save_to's 8 bits: each [0, 1] value times 255, truncated.
            pixels = self.vae.decode(video.cpu(), return_dict=False)[0][0].movedim(0, -1)
            pixels = pixels.mul_(255).clamp_(0, 255).byte().numpy()

            # VAEDecodeAudio (nodes_audio.py:99-113): the waveform, its loudness capped.
            waveform = self.audio_vae.decode(audio.cpu(), return_dict=False)[0]
            std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
            std[std < 1.0] = 1.0
            waveform /= std

            return SimpleNamespace(videos=[pixels],
                                   audio=(waveform[0].numpy(), self.audio_vae.audio_sample_rate),
                                   fps=FPS)

    return ComfyH3VideoPipe


def load(ctx, params, crop="none", text_dtype=None, ref_scheduler="simple"):
    """Assemble the pipeline of _pipeline_class from ComfyUI's single files via the offloader:
    the ref2va transformer and `ref_scheduler` for a request with references. `text_dtype` (a
    torch dtype's name) runs the text encoder in it, as ComfyUI's --bf16-text-enc does.
    OFFLOADER-ONLY."""
    import torch

    files = _comfy.by_role(ctx.model)  # engine/<id>/ (engine.json)

    if ctx.backend is None:
        raise RuntimeError("%s requires an offload backend (offload=offloader): ComfyUI's models "
                           "run on it" % params["engine"])

    ref = extra_key(params)[0]
    transformer = files["transformer_ref" if ref else "transformer"]

    text_encoder = {"model": "minimax_h3", "file": files["text_encoder"]}

    if text_dtype is not None:
        text_encoder["dtype"] = getattr(torch, text_dtype)

    return ctx.backend.load_pipe_comfy(
        _pipeline_class(crop),
        {"model": "minimax_h3", "file": transformer, "config": {}},
        text_encoder,
        {"scheduler": ctx.backend.comfy_sampler("minimax_h3", "res_multistep",
                                                ref_scheduler if ref else "simple", transformer),
         "vae": ctx.backend.comfy_vae(files["vae"], {}),
         "audio_vae": ctx.backend.comfy_vae(files["audio_vae"], {})},
        ctx.dtype, device=ctx.device, lora_files=ctx.loras or None)

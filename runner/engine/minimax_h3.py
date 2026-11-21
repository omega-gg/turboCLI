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

# minimax-h3 engine: text-to-video and image-to-video on MiniMax H3, a video with its soundtrack
# (a 33B audio-video DiT, H3's Qwen3-VL 32B text encoder, its video and audio VAEs; no CFG) from
# MiniMax's repo, its weights as published, run by diffusers' own models with or without the
# offloader. comfy-minimax-h3 is the same model from ComfyUI's files.
#
# LICENSE: the MiniMax H3 Community License. It excludes the EU, the UK, South Korea and the US,
# asks for MiniMax's written authorization above $20M of yearly revenue, and for "MiniMax H3" on
# the interface of a commercial product.
#
# diffusers ships it as modular blocks only, so H3VideoPipe below gives it the classic face the
# runner and the offloader speak, as minimax-h3-image's H3Pipe does: from_pretrained, an
# encode_prompt the offloader wraps (the keyframes go to the text encoder too, so they are part of
# the call), a progress_bar the runner hooks, a step callback for cancel, device placement, and a
# result with .videos and .audio. The blocks keep diffusers' defaults and limits: 50 steps, a
# video of 5 to 15 seconds at 24 fps, the first frame stretched onto the canvas, the last
# cover-cropped. With picture or audio references, its ref2va workflow on the transformer_ref/
# partition: references only (at most 9 pictures and 3 audios, an audio never alone), no frame
# and no audio guide.

ID   = "minimax-h3"
TYPE = "minimax-h3"

PIPELINE    = "diffusers:MiniMaxH3ModularPipeline"     # behind H3VideoPipe
TRANSFORMER = "diffusers:MiniMaxH3Transformer3DModel"  # offloader disk-stream meta-load

MODES = ("text-to-video", "image-to-video")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", 1.0)              # guidance-distilled (H3VideoPipe ignores it)

INFERENCE = 50

# Both partitions: transformer/ for text and keyframes, transformer_ref/ for references.
MODEL = {
    "repository": "MiniMaxAI",
    "model": "MiniMax-H3",
    "revision": "1cc7549f2cf0af8ff2f09b24f3c57303c8aa6317",
    "allow_patterns": [
        "modular_model_index.json",
        "transformer/*",
        "transformer_ref/*",
        "vae/*",
        "audio_vae/*",
        "text_encoder/*",
        "tokenizer/*",
        "processor/*",
        "scheduler/*",
        "audio_scheduler/*",
    ],
}

FPS = 24


def extra_key(params):
    """The workflow, which picks the transformer: the resident pipe reloads when it changes."""
    from ._options import parse_references

    return (bool(parse_references(params.get("images", ""))["references"]),)


def _pipeline_class(ref=False):
    """H3VideoPipe: diffusers' modular MiniMax-H3 blocks behind the classic pipeline face, on the
    ref2va partition when `ref`. The text encoder runs in encode_prompt, the rest of the blocks
    take its embeddings. Built here so diffusers stays out of engine discovery."""
    from types import SimpleNamespace

    import numpy as np
    from diffusers import (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, DiffusionPipeline,
                           MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel)
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
        MiniMaxH3AutoBeforeEncodeStep, MiniMaxH3AutoTextEncoderStep, MiniMaxH3Blocks)
    from diffusers.modular_pipelines.minimax_h3.references import (MiniMaxH3AudioReference,
                                                                    MiniMaxH3ImageReference)
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    class Stop(Exception):
        """A cancel or a newer request, raised out of the denoise loop."""

    class H3VideoPipe(DiffusionPipeline):
        model_cpu_offload_seq = "text_encoder->transformer->vae"

        # NOTE: The partition the offloader streams.
        transformer_folder = "transformer_ref" if ref else "transformer"

        def __init__(self, transformer, text_encoder, tokenizer, processor, vae, audio_vae,
                     scheduler, audio_scheduler):
            super().__init__()

            self.register_modules(transformer=transformer, text_encoder=text_encoder,
                                  tokenizer=tokenizer, processor=processor, vae=vae,
                                  audio_vae=audio_vae, scheduler=scheduler,
                                  audio_scheduler=audio_scheduler)

            self._interrupt = False

            self._callback = None
            self._step = 0

            blocks = MiniMaxH3Blocks()
            blocks.sub_blocks.pop("text_encoder")

            # NOTE: Each workflow's denoise loop opens its bar on itself: it goes through ours,
            #       which the runner hooks for its step lines, and each update runs the step
            #       callback.
            for workflow in blocks.sub_blocks["denoise"].sub_blocks.values():
                workflow.sub_blocks["denoise"].progress_bar = lambda **kwargs: self._bar(**kwargs)

            # The text encoder's own blocks, the keyframes put on the canvas for it first.
            encode = SequentialPipelineBlocks.from_blocks_dict({
                "before_encode": MiniMaxH3AutoBeforeEncodeStep(),
                "text_encoder": MiniMaxH3AutoTextEncoderStep()})

            self._pipe = blocks.init_pipeline()
            self._pipe.update_components(**{self.transformer_folder: transformer}, vae=vae,
                                         audio_vae=audio_vae, scheduler=scheduler,
                                         audio_scheduler=audio_scheduler)

            self._encode = encode.init_pipeline()
            self._encode.update_components(text_encoder=text_encoder, tokenizer=tokenizer,
                                           processor=processor)

            # NOTE: The modular pipelines read their device from where their modules sit, which an
            #       offload decides; they follow ours instead.
            outer = self
            for pipe in (self._pipe, self._encode):
                cls = type(pipe)
                pipe.__class__ = type(cls.__name__, (cls,), {
                    "_execution_device": property(lambda _: outer._execution_device)})

        @classmethod
        def from_pretrained(cls, model, transformer=None, torch_dtype=None,
                            local_files_only=True, **kwargs):
            """The components the repo's modular index names, each from its subfolder (a streamed
            `transformer` may come in place of its own)."""
            def diffusers_part(part_cls, subfolder):
                return part_cls.from_pretrained(model, subfolder=subfolder,
                                                torch_dtype=torch_dtype,
                                                local_files_only=local_files_only, **kwargs)

            def scheduler(subfolder):
                return MiniMaxH3Scheduler.from_pretrained(model, subfolder=subfolder,
                                                          local_files_only=local_files_only)

            if transformer is None:
                transformer = diffusers_part(MiniMaxH3Transformer3DModel, cls.transformer_folder)

            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model, subfolder="text_encoder", dtype=torch_dtype,
                local_files_only=local_files_only)
            tokenizer = Qwen2TokenizerFast.from_pretrained(model, subfolder="tokenizer",
                                                           local_files_only=local_files_only)
            processor = Qwen3VLProcessor.from_pretrained(model, subfolder="processor",
                                                         local_files_only=local_files_only)

            return cls(transformer, text_encoder, tokenizer, processor,
                       diffusers_part(AutoencoderKLMiniMaxH3, "vae"),
                       diffusers_part(AutoencoderKLMiniMaxH3Audio, "audio_vae"),
                       scheduler("scheduler"), scheduler("audio_scheduler"))

        def _bar(self, total=None):
            bar = self.progress_bar(total=total)
            update = bar.update

            def stepped(n=1):
                result = update(n)

                if self._callback is not None:
                    self._callback(self, self._step, None, {})

                self._step += 1

                if self._interrupt:
                    raise Stop()

                return result

            bar.update = stepped

            return bar

        @staticmethod
        def _inputs(image, references, guide):
            """The workflow's inputs: the references (MiniMaxH3ImageReference,
            MiniMaxH3AudioReference), or the first and last frames."""
            first, last = (list(image or ()) + [None, None])[:2]

            if guide is not None:
                raise ValueError("diffusers' MiniMax H3 takes no audio guide")
            if references and (first or last):
                raise ValueError("diffusers' MiniMax H3 pins no frame with references")
            if references:
                return {"references": [MiniMaxH3AudioReference.from_file(r) if isinstance(r, str)
                                       else MiniMaxH3ImageReference(image=r) for r in references]}

            return {"image": first, "last_image": last}

        def encode_prompt(self, prompt, inputs, height=None, width=None, num_frames=None,
                          device=None):
            """The embeddings and their token tags; the frames or references are part of them."""
            out = self._encode(prompt=prompt, height=height, width=width, num_frames=num_frames,
                               **inputs, output=["prompt_embeds", "text_token_tags"])

            return out["prompt_embeds"], out["text_token_tags"]

        def __call__(self, prompt, image=None, height=None, width=None, duration=5.0,
                     num_inference_steps=INFERENCE, generator=None, callback_on_step_end=None,
                     references=(), guide=None, attention=None, scheduler=None, crop=None,
                     **_):
            if (attention, scheduler, crop) != (None, None, None):
                raise ValueError("minimax-h3 runs as diffusers does: no video option")

            self._callback = callback_on_step_end
            self._step = 0

            inputs = self._inputs(image, references, guide)
            num_frames = max(5, round(duration * FPS))

            prompt_embeds, tags = self.encode_prompt(prompt, inputs, height, width, num_frames,
                                                     device=self._execution_device)

            try:
                out = self._pipe(prompt_embeds=prompt_embeds, text_token_tags=tags,
                                 height=height, width=width, num_frames=num_frames,
                                 num_inference_steps=num_inference_steps, generator=generator,
                                 output_type="np", **inputs,
                                 output=["videos", "audio", "sampling_rate"])
            except Stop:
                return SimpleNamespace(videos=[], audio=None, fps=FPS)

            # The frames in [0, 1], to 8 bits as ComfyUI's save_to does: times 255, truncated.
            frames = (np.asarray(out["videos"])[0] * 255).clip(0, 255).astype(np.uint8)

            audio = out["audio"][0].float().cpu().numpy()

            return SimpleNamespace(videos=[frames], audio=(audio, out["sampling_rate"]), fps=FPS)

    return H3VideoPipe


def load(ctx, params):
    """H3VideoPipe from the repo: through the offloader when one is chosen (load_pipe, as the
    other stock engines), else placed by the native offload modes."""
    pipeline_cls = _pipeline_class(extra_key(params)[0])

    if ctx.backend is not None:
        from diffusers import MiniMaxH3Transformer3DModel

        return ctx.backend.load_pipe(ctx.model, ctx.dtype, pipeline_cls,
                                     MiniMaxH3Transformer3DModel, device=ctx.device,
                                     lora_files=ctx.loras or None)

    p = pipeline_cls.from_pretrained(ctx.model, torch_dtype=ctx.dtype, use_safetensors=True,
                                     low_cpu_mem_usage=True)

    ctx.apply_loras(p, ctx.loras)

    return ctx.apply_offload(p)

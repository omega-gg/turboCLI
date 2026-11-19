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

# minimax-h3-image engine: text2img + img2img on MiniMax-H3-Image (33.1B DiT, H3's Qwen3-VL 32B
# text encoder, its 24-channel image VAE; 6 steps, one output head per step, no CFG) from
# MiniMax's repo, its weights as published (bf16, ~135 GB), run by diffusers' own models with or
# without the offloader. comfy-minimax-h3-image is the same model from ComfyUI's files.
#
# LICENSE: the MiniMax-H3-Image Community License. It excludes the EU, the UK, South Korea and the
# US, and asks for MiniMax's written authorization above $1M of annual revenue or for a hosted
# service.
#
# diffusers ships it as modular blocks only, so H3Pipe below gives it the classic face the runner
# and the offloader speak, as flux3-6b's Flux3Pipe does: from_pretrained, an encode_prompt the
# offloader wraps (an edit's references go to the text encoder too, so they are part of the
# call), a progress_bar the runner hooks, a step callback for cancel, device placement, and a
# result with .images. The blocks keep diffusers' defaults: a size under 4 MP renders on a 4 MP
# canvas of its aspect ratio, resized back (min_generation_area).

ID   = "minimax-h3-image"
TYPE = "minimax-h3-image"

PIPELINE    = "diffusers:MiniMaxH3ImageModularPipeline"  # behind H3Pipe
TRANSFORMER = "diffusers:MiniMaxH3ImageTransformer2DModel"  # offloader disk-stream meta-load

MODES = ("text-to-image", "image-to-image")  # wire values, same as the HTTP API
CFG   = ("guidance_scale", 1.0)              # guidance-distilled (H3Pipe ignores it)

# One output head per step: the checkpoint samples its 6 sigmas, whatever the step count.
INFERENCE = 6

# The diffusers layout only: the repo also carries MiniMax's own transformer and VAE at its root,
# and their PDD heads file, which this engine does not read.
MODEL = {
    "repository": "MiniMaxAI",
    "model": "MiniMax-H3-Image",
    "revision": "8ced60068623def5b74b4d10558117cb61d6482d",
    "allow_patterns": [
        "modular_model_index.json",
        "transformer_diffusers/*",
        "vae_diffusers/*",
        "text_encoder/*",
        "tokenizer/*",
        "processor/*",
        "scheduler/*",
    ],
}


def _pipeline_class():
    """H3Pipe: diffusers' modular MiniMax-H3-Image blocks behind the classic pipeline face. The
    text encoder runs in encode_prompt, the rest of the blocks take its embeddings. Built here so
    diffusers stays out of engine discovery."""
    from types import SimpleNamespace

    from diffusers import (AutoencoderKLMiniMaxH3Image, DiffusionPipeline,
                           MiniMaxH3ImageTransformer2DModel, MiniMaxH3Scheduler)
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.minimax_h3_image.modular_blocks_minimax_h3_image import (
        MiniMaxH3ImageAutoBeforeEncodeStep, MiniMaxH3ImageAutoTextEncoderStep,
        MiniMaxH3ImageBlocks)
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    class Stop(Exception):
        """A cancel or a newer request, raised out of the denoise loop."""

    class H3Pipe(DiffusionPipeline):
        model_cpu_offload_seq = "text_encoder->transformer->vae"

        transformer_folder = "transformer_diffusers"  # the offloader streams it from there

        def __init__(self, transformer, text_encoder, tokenizer, processor, vae, scheduler):
            super().__init__()

            self.register_modules(transformer=transformer, text_encoder=text_encoder,
                                  tokenizer=tokenizer, processor=processor, vae=vae,
                                  scheduler=scheduler)

            self._interrupt = False

            self._callback = None
            self._step = 0

            blocks = MiniMaxH3ImageBlocks()
            blocks.sub_blocks.pop("text_encoder")

            # NOTE: Each workflow's denoise loop opens its bar on itself: it goes through ours,
            #       which the runner hooks for its step lines, and each update runs the step
            #       callback.
            for workflow in blocks.sub_blocks["denoise"].sub_blocks.values():
                workflow.sub_blocks["denoise"].progress_bar = lambda **kwargs: self._bar(**kwargs)

            # The text encoder's own blocks, an edit's references resized for it first.
            encode = SequentialPipelineBlocks.from_blocks_dict({
                "before_encode": MiniMaxH3ImageAutoBeforeEncodeStep(),
                "text_encoder": MiniMaxH3ImageAutoTextEncoderStep()})

            self._pipe = blocks.init_pipeline()
            self._pipe.update_components(transformer=transformer, vae=vae, scheduler=scheduler)

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

            if transformer is None:
                transformer = diffusers_part(MiniMaxH3ImageTransformer2DModel,
                                             cls.transformer_folder)

            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model, subfolder="text_encoder", dtype=torch_dtype,
                local_files_only=local_files_only)
            tokenizer = Qwen2TokenizerFast.from_pretrained(model, subfolder="tokenizer",
                                                           local_files_only=local_files_only)
            processor = Qwen3VLProcessor.from_pretrained(model, subfolder="processor",
                                                         local_files_only=local_files_only)
            scheduler = MiniMaxH3Scheduler.from_pretrained(model, subfolder="scheduler",
                                                           local_files_only=local_files_only)

            return cls(transformer, text_encoder, tokenizer, processor,
                       diffusers_part(AutoencoderKLMiniMaxH3Image, "vae_diffusers"), scheduler)

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

        def encode_prompt(self, prompt, image=None, device=None):
            """The embeddings and their token tags; an edit's references are part of them."""
            out = self._encode(prompt=prompt, image=image,
                               output=["prompt_embeds", "text_token_tags"])

            return out["prompt_embeds"], out["text_token_tags"]

        def __call__(self, prompt, image=None, height=None, width=None, generator=None,
                     callback_on_step_end=None, **_):
            self._callback = callback_on_step_end
            self._step = 0

            prompt_embeds, tags = self.encode_prompt(prompt, image, device=self._execution_device)

            try:
                images = self._pipe(prompt_embeds=prompt_embeds, text_token_tags=tags,
                                    image=image, height=height, width=width,
                                    generator=generator, output="images")
            except Stop:
                images = []

            return SimpleNamespace(images=images)

    return H3Pipe


def load(ctx, params):
    """H3Pipe from the repo: through the offloader when one is chosen (load_pipe, as the other
    stock engines), else placed by the native offload modes."""
    pipeline_cls = _pipeline_class()

    if ctx.backend is not None:
        from diffusers import MiniMaxH3ImageTransformer2DModel

        return ctx.backend.load_pipe(ctx.model, ctx.dtype, pipeline_cls,
                                     MiniMaxH3ImageTransformer2DModel, device=ctx.device,
                                     lora_files=ctx.loras or None)

    p = pipeline_cls.from_pretrained(ctx.model, torch_dtype=ctx.dtype, use_safetensors=True,
                                     low_cpu_mem_usage=True)

    ctx.apply_loras(p, ctx.loras)

    return ctx.apply_offload(p)

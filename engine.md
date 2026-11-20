# turboCLI engines

Every engine turboCLI can install, by task, to help you pick one. Install an engine with
`install.sh <engine> <renderer> ...` (see [bash/turbo/README.md](bash/turbo/README.md)).

- **Size** counts every file the engine needs. Engines of the same model share files, so once one
  is installed the next one only adds what differs (`comfy-qwen-image-2-1-turbo` adds 7.3 GB to
  `comfy-qwen-image-2-1`).
- **Steps** is the default number of inference steps: fewer is faster.
- **Requirements**: *gated* needs a Hugging Face token from an account that accepted the model's
  license; *offloader* runs only with `offload=offloader`
  ([turbo-offloader](https://omega.gg/turbo-offloader)), because its quantized weights need
  ComfyUI's ops; *MiniMax license*: the
  [MiniMax-H3-Image Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)
  excludes the EU, the UK, South Korea and the US, and asks for MiniMax's written authorization
  above $1M of annual revenue or for a hosted service.
- `comfy-` engines use the model files of a ComfyUI install, the way ComfyUI runs them. Give
  `install.sh` your ComfyUI folder to reuse its files; without one they are downloaded.

## Text-to-image

| Engine | Description | Size | Steps | Requirements |
|---|---|---|---|---|
| `flux2-4b` | FLUX.2 Klein 4B, a small, fast generator (diffusers weights) | 15 GB | 4 | |
| `comfy-flux2-4b` | FLUX.2 Klein 4B from ComfyUI's files | 16 GB | 4 | offloader |
| `flux2-9b` | FLUX.2 Klein 9B, the larger Klein (diffusers weights) | 35 GB | 4 | gated |
| `comfy-flux2-9b` | FLUX.2 Klein 9B from ComfyUI's bf16 files | 35 GB | 4 | gated, offloader |
| `comfy-flux2-9b-distilled` | FLUX.2 Klein 9B as ComfyUI's distilled template: fp8 with a small decoder, lighter and faster | 18 GB | 4 | gated, offloader |
| `flux3-6b` | FLUX.3 Image Klein 5.6B, the 16-step distilled model (diffusers weights, bf16) | 20 GB | 16 | gated |
| `comfy-flux3-6b` | FLUX.3 Image Klein 5.6B, the 16-step distilled model in int8 | 11 GB | 16 | gated, offloader |
| `z-image-turbo` | Z-Image-Turbo, a fast distilled generator (diffusers weights) | 20 GB | 8 | |
| `comfy-z-image-turbo` | Z-Image-Turbo from ComfyUI's files | 21 GB | 8 | offloader |
| `comfy-krea2-turbo` | Krea 2 Turbo, a 12B distilled generator in fp8 | 19 GB | 8 | gated, offloader |
| `comfy-qwen-image-2-1` | Qwen-Image 2.1, a 7B model in int8 that generates and edits, with RGBA output | 17 GB | 25 | offloader |
| `comfy-qwen-image-2-1-turbo` | Qwen-Image-2.1-Turbo, its 8-step distillation, with RGBA output | 17 GB | 8 | offloader |
| `moss-image` | MOSS-Image SFT, a 5B generator strong at text and layouts (OpenMOSS's weights, bf16) | 27 GB | 50 | |
| `comfy-moss-image` | MOSS-Image SFT in int8, as ComfyUI's template runs it | 13 GB | 25 | offloader |
| `minimax-h3-image` | MiniMax-H3-Image, a 33B distilled generator made for 4 to 20 MP, best at 16 MP (MiniMax's weights, bf16) | 135 GB | 6 | MiniMax license |
| `comfy-minimax-h3-image` | MiniMax-H3-Image in int8, as ComfyUI's template runs it | 49 GB | 6 | offloader, MiniMax license |

## Image editing

Image-to-image: the prompt describes the change, applied to one or more reference images.

| Engine | Description | Size | Steps | Requirements |
|---|---|---|---|---|
| `flux2-4b` | FLUX.2 Klein 4B, edits from reference images (diffusers weights) | 15 GB | 4 | |
| `comfy-flux2-4b` | FLUX.2 Klein 4B from ComfyUI's files | 16 GB | 4 | offloader |
| `flux2-9b` | FLUX.2 Klein 9B, the larger Klein (diffusers weights) | 35 GB | 4 | gated |
| `comfy-flux2-9b` | FLUX.2 Klein 9B from ComfyUI's bf16 files | 35 GB | 4 | gated, offloader |
| `comfy-flux2-9b-distilled` | FLUX.2 Klein 9B as ComfyUI's distilled template: fp8 with a small decoder, lighter and faster | 18 GB | 4 | gated, offloader |
| `flux3-6b` | FLUX.3 Image Klein 5.6B, edits from up to 10 reference images (diffusers weights, bf16) | 20 GB | 16 | gated |
| `comfy-flux3-6b` | FLUX.3 Image Klein 5.6B, edits from up to 10 reference images, 16-step distilled, in int8 | 11 GB | 16 | gated, offloader |
| `comfy-krea2-turbo-edit-identity` | Krea 2 Turbo restaging a person from a reference photo, conradlocke's Identity Edit V2 LoRA (personal and non-commercial use) | 21 GB | 8 | gated, offloader |
| `comfy-krea2-turbo-edit-general` | Krea 2 Turbo editing from a plain instruction (remove, recolor, relight, swap a background, add an object, rewrite text, restyle), conradlocke's General Edit V2 LoRA (personal and non-commercial use) | 21 GB | 8 | gated, offloader |
| `comfy-qwen-image-2-1` | Qwen-Image 2.1, edits from up to 10 reference images, with RGBA output | 17 GB | 25 | offloader |
| `comfy-qwen-image-2-1-turbo` | Qwen-Image-2.1-Turbo, its 8-step distillation, with RGBA output | 17 GB | 8 | offloader |
| `comfy-moss-image` | MOSS-Image SFT in int8 editing from an instruction, Comfy-Org's edit LoRA (from chinoll/Moss-Image-Edit) | 13 GB | 30 | offloader |
| `minimax-h3-image` | MiniMax-H3-Image, edits from reference images, `<Picture 1>` in the prompt (MiniMax's weights, bf16) | 135 GB | 6 | MiniMax license |
| `comfy-minimax-h3-image` | MiniMax-H3-Image in int8, edits from reference images, as ComfyUI's template runs it; also a strong upscaler, a small image cleaned up to 16 MP (it invents the fine detail the source lacks) | 49 GB | 6 | offloader, MiniMax license |
| `qwen-image-edit-2511` | Qwen-Image-Edit-2511, a 20B model made for editing (diffusers weights) | 58 GB | 40 | |
| `qwen-image-edit-2511-lightning` | Qwen-Image-Edit-2511 with the Lightning LoRA | 59 GB | 4 | |
| `qwen-image-edit-2511-lightning-angles` | Lightning plus fal's Multiple-Angles LoRA, for prompts starting with `<sks>` | 59 GB | 4 | |
| `comfy-qwen-image-edit-2511` | Qwen-Image-Edit-2511 from ComfyUI's files | 51 GB | 40 | offloader |
| `comfy-qwen-image-edit-2511-lightning` | Qwen-Image-Edit-2511 from ComfyUI's files, with the Lightning LoRA | 51 GB | 4 | offloader |

## Masking and matting

| Engine | Description | Size | Steps | Requirements |
|---|---|---|---|---|
| `mask` | A mask of where an edit differs from its reference, as a pixel difference or grown boxes | none | | no GPU |
| `mask-apply` | Applies a mask: composite onto a reference, alpha cutout, or trim | none | | no GPU |
| `adjust` | Fixes an existing image against its reference: the zoom and shift undone, the colors matched (`image-adjust.sh`) | none | | no GPU |
| `mask-birefnet` | A subject matte (background removal) with BiRefNet | 0.4 GB | | |
| `mask-lucida` | A BiRefNet fine-tune for glass, camouflage, text and print | 0.8 GB | | |
| `mask-inspyrenet` | A subject matte with InSPyReNet | 0.4 GB | | |

## Upscaling

| Engine | Description | Size | Steps | Requirements |
|---|---|---|---|---|
| `esrgan` | Real-ESRGAN x4plus: sharp and fast, adds no detail the source lacks; no prompt | 0.07 GB | | |
| `comfy-esrgan` | Real-ESRGAN x4plus from ComfyUI's file | 0.07 GB | | |
| `z-image-turbo-upscale` | `esrgan`, then a short Z-Image-Turbo pass that redraws the fine detail (ComfyUI's upscale template); optional prompt | 20 GB | 5 | |
| `comfy-z-image-turbo-upscale` | The same from ComfyUI's files, with `comfy-esrgan` | 21 GB | 5 | offloader |

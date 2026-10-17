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
  ComfyUI's ops.
- `comfy-` engines use the model files of a ComfyUI install, the way ComfyUI runs them. Give
  `install.sh` your ComfyUI folder to reuse its files; without one they are downloaded.

## Text-to-image

| Engine | Description | Size | Steps | Requirements |
|---|---|---|---|---|
| `flux2-4b` | FLUX.2 Klein 4B, a small, fast generator (diffusers weights) | 15 GB | 4 | |
| `comfy-flux2-4b` | FLUX.2 Klein 4B from ComfyUI's files | 16 GB | 4 | |
| `flux2-9b` | FLUX.2 Klein 9B, the larger Klein (diffusers weights) | 35 GB | 4 | gated |
| `comfy-flux2-9b` | FLUX.2 Klein 9B from ComfyUI's bf16 files | 35 GB | 4 | gated |
| `comfy-flux2-9b-distilled` | FLUX.2 Klein 9B as ComfyUI's distilled template: fp8 with a small decoder, lighter and faster | 18 GB | 4 | gated, offloader |
| `comfy-flux3-6b` | FLUX.3 Image Klein 5.6B, the 16-step distilled model in int8 | 11 GB | 16 | gated, offloader |
| `z-image-turbo` | Z-Image-Turbo, a fast distilled generator (diffusers weights) | 20 GB | 8 | |
| `comfy-z-image-turbo` | Z-Image-Turbo from ComfyUI's files | 21 GB | 8 | |
| `comfy-krea2-turbo` | Krea 2 Turbo, a 12B distilled generator in fp8 | 19 GB | 8 | offloader |
| `comfy-krea2-turbo-realism` | Krea 2 Turbo with the Krea2-realism-V2 LoRA | 20 GB | 8 | offloader |
| `comfy-qwen-image-2-1` | Qwen-Image 2.1, a 7B model in int8 that generates and edits, with RGBA output | 17 GB | 25 | offloader |
| `comfy-qwen-image-2-1-turbo` | Qwen-Image-2.1-Turbo, its 8-step distillation, with RGBA output | 17 GB | 8 | offloader |

## Image editing

Image-to-image: the prompt describes the change, applied to one or more reference images.

| Engine | Description | Size | Steps | Requirements |
|---|---|---|---|---|
| `flux2-4b` | FLUX.2 Klein 4B, edits from reference images (diffusers weights) | 15 GB | 4 | |
| `comfy-flux2-4b` | FLUX.2 Klein 4B from ComfyUI's files | 16 GB | 4 | |
| `flux2-9b` | FLUX.2 Klein 9B, the larger Klein (diffusers weights) | 35 GB | 4 | gated |
| `comfy-flux2-9b` | FLUX.2 Klein 9B from ComfyUI's bf16 files | 35 GB | 4 | gated |
| `comfy-flux2-9b-distilled` | FLUX.2 Klein 9B as ComfyUI's distilled template: fp8 with a small decoder, lighter and faster | 18 GB | 4 | gated, offloader |
| `comfy-flux3-6b` | FLUX.3 Image Klein 5.6B, edits from up to 10 reference images, 16-step distilled, in int8 | 11 GB | 16 | gated, offloader |
| `comfy-qwen-image-2-1` | Qwen-Image 2.1, edits from up to 10 reference images, with RGBA output | 17 GB | 25 | offloader |
| `comfy-qwen-image-2-1-turbo` | Qwen-Image-2.1-Turbo, its 8-step distillation, with RGBA output | 17 GB | 8 | offloader |
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
| `mask-birefnet` | A subject matte (background removal) with BiRefNet | 0.4 GB | | |
| `mask-lucida` | A BiRefNet fine-tune for glass, camouflage, text and print | 0.8 GB | | |
| `mask-inspyrenet` | A subject matte with InSPyReNet | 0.4 GB | | |

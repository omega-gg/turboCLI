# Plan: MiniMax H3 IC upscale engines (Open-H3-Regenerate-2K IC) in turboCLI

## Context

turboCLI's H3 takes keep their latents (`latent=save`), and `comfy-minimax-h3-upscale` /
`comfy-minimax-h3-quality-upscale` render a take again at 2x from them (LBH-123-AI latent upscaler
3D, then 8 H3 refine steps at denoise 0.45): about 12 min and $1.25 for a 10 s 1376x576 take on a
B200.

NVIDIA's Open-H3-Regenerate-2K **IC** — reference: https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K/IC,
weights `Efficient-Large-Model/Open-H3-Regenerate-2K-IC` @ `54a7104b204fb46486cf85c2f600942118186d57`,
MiniMax H3 Community License — refines an H3 draft on H3's Ref2VA model: the draft stays in the
context as a clean reference video, and a LoRA stack (IC rank 384 + HyperFlow 8-step + LMS Sharp)
makes 3 Euler steps a faithful, sharper re-render. Reported better than MiniMax's API 2K stage.

Goal, in turboCLI (with turbo-offloader): two engines that re-render a saved take from its latents
with IC, **2x by default** (the test target, estimated about half today's 2x cost), any larger size
(2K, 4K later) from the width and height `image-to-video.sh` already takes:
- `comfy-minimax-h3-quality-ic-upscale` (BASE comfy-minimax-h3-quality, bf16), built and verified
  first;
- `comfy-minimax-h3-ic-upscale` (BASE comfy-minimax-h3, int8 convrot).

Efficient, clean, simple; **Modal cost as low as possible**: installs and checks on CPU
containers, GPU only for two short B200 sessions, nothing left running. First implementation step:
copy this plan to `turboCLI/doc/minimax-h3-ic-upscale-plan.md`.

## What IC does (mirrored, not its CLI)

- Sequence `[Qwen3-VL context: draft frames as <Video 1> + text | reference rows: the clean draft
  latent at its own size, timestep max(1-sigma, 0.999), anchor .999*ref + .001*noise | audio rows:
  encoded silence, t = 1 | target rows]`, CFG 1, one transformer call a step.
- Prompt: IC's fixed `TASK` (`Enhance <Video 1> into a high-quality version of the same video. ...
  Do not continue or change the scene.`) + `\n\nScene description: ` + the take's prompt.
- Target: the draft latent enlarged by the same LBH-123-AI 3d_conv_v1 upscaler, noised to
  0.834942, Euler on `[0.834942, 0.696852, 0.468753, 0]` (`up3`: HyperFlow's grid at shift 12).
- LoRAs: HyperFlow (plus its two-time embedder: target and text rows get r = 1 - sigma_next,
  gate from the file's `hyperflow_gate`), LMS Sharp (plus 50 AdaLN curve branches on the
  [1025, 8] `adaln_t_table`), IC; all at strength 1.
- Audio: the draft's track kept.

Deliberate differences: the saved take latent is the reference (already VAE-normalized, the same
`(z - mean) / std` as IC's encode; no re-encode, no 16:9 crop, the take's ratio kept); any output size;
ComfyUI's files through turboCLI. The take's own LoRAs are dropped for the IC stack (trained for the
base model, not IC).

## Design

### Model files
The IC engines need Comfy-Org's **full-form** Ref2VA files (the pruned ones the base engines use
have no time embedder, so no HyperFlow), at the pinned `Comfy-Org/MiniMax-H3` @ 4b53a35:
`diffusion_models/minimax_h3_ref2va_bf16.safetensors` (66.3 GB, quality) and
`minimax_h3_ref2va_int8_convrot.safetensors` (34.0 GB). Sharp's curve table is the base pruned
ref2va file's `adaln_t_table` (sha256 c02a6c11…a8e7, IC's own source): no extra download. The three
IC LoRAs (8.3 + 2.8 + 1.2 GB) under `loras/`.

### turbo-offloader (`C:\dev\workspace\turbo-offloader\offloader`)
- `adapter.py`: `FlowSampler.set_sigmas(sigmas, device)` (sigmas, timesteps = sigmas[:-1] x
  multiplier, step state reset); `set_timesteps` ends by calling it. Euler unchanged: with the DiT
  returning -v it equals IC's `x + (sigma - nxt) * v`.
- `minimax_h3.py`:
  - `class HyperFlowModel(h3.MiniMaxH3Model)`, built by `load()` when `config["hyperflow"]`:
    `install(sd, hyperflow, curve, strength)` keeps its extras in a plain holder (no submodule):
    the endpoint embedder (the file's `time_embedder` weights + HyperFlow's
    `endpoint_time_embedder` pairs, merged once in fp32) and Sharp's folded branches
    (S_i = B_i @ A_i, fp32 [96768, 8] x 50); `_forward` stores (t_v, r = 1 - sigma_next) from
    `sample_sigmas` (as FinalLayer's PDD rule, `comfy/ldm/minimax/model.py:358-359`);
    `_time_embedding` blends `e + gate * (e_r - e)` for rows at t_v, keeps r = t elsewhere, and
    stores Sharp's `coords = lerp(table, t)`; a forward hook on each `blocks[i].adaln_proj.linear`
    adds `coords @ S_i^T`. Extras moved to the device in `extra_conds` (not lazily in forward).
  - `lora_keys()`, the one place for H3's LoRA names: Diffusers keys onto ComfyUI's fused weights
    with ComfyUI's `(key, offset, function)` patches (`to_q/k/v` -> `attn.qkv_proj.weight (0,
    j*7168, 7168)`, `ff.net.0.proj` -> `mlp.fc1.weight` with `diff.roll(14336, 0)` ([value, gate]
    -> [gate, value]), `to_out.0` -> `attn.out_proj`, `ff.net.2` -> `mlp.fc2`, the 2 refiner
    blocks, the time embedder); HyperFlow's endpoint pairs and Sharp's 8-input adaln branches
    routed to names outside the model (consumed by the extras, skipped by `add_patches`). Sharp's
    other keys are ComfyUI's already. No file conversion.
  - `video_reference(latent, frames)`: `{"kind": "video", latent_t/h/w, "ref_audio_t": 0,
    "latent": take}` and the Qwen video item (frames every 12th = 2 fps, timestamps i/2), as
    upstream `MiniMaxH3ReferenceToVideo`'s `ref_videos` branch.
  - `silence(audio_vae, T)`; `extra_conds(..., fixed_audio=True)` (audio_scale 1, an
    `audio_denoise_mask` of zeros: audio rows pinned at t = 1, velocity zeroed); `forward` takes
    `audio_denoise_mask`.
- No vendored-file edit; `implementation.md` documents the subclass.

### turboCLI (`C:\dev\workspace\turboCLI\runner`)
- `core.py:_take`:
  - the take's engine checked against the engine module's `BASE` (today `params["engine"] ==
    take["engine"] + "-upscale"`, which refuses `...-quality-ic-upscale`); existing upscale
    engines unchanged (their BASE is their take's engine);
  - the size from the request's width and height (the script's own arguments): at least the
    take's size, they are the target (its ratio kept within the 32 grid, refused otherwise; e.g.
    `3456 1440` for 2K from 1376x576); smaller, as the script's 512x512 default, twice the take's.
    Every upscale engine gains it.
- `engine/comfy_minimax_h3.py`, the existing pipe and load extended, no new pipe class:
  - `load(..., ic=False)`: with `ic`, the full ref2va (`transformer_ic`), Euler, the IC LoRAs
    `[(hyperflow,1), (sharp,1), (ic,1)]` in place of the take's, `config={"hyperflow", "curve",
    "strength"}`; otherwise unchanged;
  - `_generate`'s take branch, when the pipe is IC: the decoded take as the video reference (in
    place of the keyframes), `TASK` + the take's prompt, `set_sigmas(SIGMAS)`, silence audio rows;
    the upscale, noise mix, loop, take audio out and decode reused as they are.
- `engine/comfy_minimax_h3_ic_upscale.py` (new, a declaration, BASE comfy-minimax-h3): `COMFY` =
  base components + the full int8 ref2va (role `transformer_ic`) + `upscale.UPSCALER` + the three
  IC LoRAs (pinned); `INFERENCE = 3`, `SIGMAS = (0.834942, 0.696852, 0.468753, 0.0)`, `TASK`;
  MODES / UPSCALE / `check` from the upscale engine (plus: a take of at most 124 frames, IC's
  trained length); `load = lambda ctx, params: base.load(ctx, params, ic=True)`.
- `engine/comfy_minimax_h3_quality_ic_upscale.py` (new, a declaration, BASE
  comfy-minimax-h3-quality): the same with the full bf16 ref2va, `text_dtype="bfloat16"`; both
  IC engines `CRF = 12`.

Size of the change: about 250 lines across the five files, most of it `HyperFlowModel` and the
key map; no vendored-file edit, no duplicated pipeline.

### Quality first (efficient, never at the cost of quality)
- IC's own settings, unchanged: 3 steps are IC's default `up3`, not a cut; `inference=4` gives
  IC's `up4` (the grid's last 4 points) for more quality, with no extra code (`SIGMAS` taken from
  the grid's tail by the step count).
- Full precision where IC keeps it: the LoRA deltas, the endpoint embedder and Sharp's folded
  branches computed in fp32 and baked once; the HyperFlow time embedding kept in fp32 up to the
  adaln input (ComfyUI casts it to bf16 before `silu`, Diffusers doesn't: our override keeps
  fp32).
- The quality engine first and as the reference result: the full bf16 Ref2VA file, the bf16 text
  encoder, the standard VAE decode (IC's fast VAE is an approximation, left out), crf 12 on both
  IC engines (the -quality setting).
- The take's own latent as the reference (exactly what was rendered, not a re-encoded copy).
- Measured, not assumed: step 3's parity at fp32 rounding, step 5's fidelity to the take (PSNR /
  SSIM back at the take's size) and the side by side decide; the int8 engine is kept only if it
  holds up there.

### Speed (the cost on Modal is B200 time)
- 3 steps (IC's `up3`), one transformer call a step (CFG 1): against 8 for today's 2x.
- `attention=cudnn` by default on the IC engines (about twice as fast on a B200, as measured
  for H3).
- The LoRAs baked into the weights once at load, not patched on every cast and step: on a B200
  the full bf16 file is resident, so ComfyUI's patcher applies them when the model loads (checked
  in step 3: no per-step `LowVramPatch`). The Sharp branches are folded once (S_i), the endpoint
  embedder merged once.
- No VAE re-encode: the take latent is the reference as it is; the take is decoded once at its
  low size for the Qwen frames, and the text encoder is freed before sampling.
- Load time is most of a short run: several takes upscaled in one Modal session share one load
  (turboCLI's existing server, `server.sh`); the int8 engine streams 34 GB instead of 66.
- None of this adds code beyond the defaults above: the speed comes from fewer steps, the
  attention option and ComfyUI's own patcher. Out of scope, to keep the code simple: IC's Sol-Attn
  MXFP8 engine and fast VAE (FlashAttention-4, quack, their own runtime), about 3.9x on their
  timings; noted for later.

### As built (where it differs from the above)
- GPU agnostic: nothing assumes a B200, which is only what the verification runs on.
- The LoRAs are not baked at load: turbo-offloader's `add_lora` applies every LoRA on cast
  (LowVramPatch), on any card. The cost is the rank products per weight cast (about 28 TFLOP a
  step for the three LoRAs, a fraction of a step's time), measured in step 4.
- The time embedding is cast to the compute dtype before the adaln projections, as ComfyUI's
  `_time_embedding` does (Diffusers' bf16 model runs it in bf16 too): keeping it in fp32 would
  dequantize every adaln weight to fp32 each step.
- IC's silence is its own asset (`assets/silence_audio_124f.safetensors`, a `silence` component,
  26 KB), not encoded: exactly IC's audio rows. No `silence()` in the offloader.
- No `TAKE` attribute: `_take` checks the take's engine against the engine's `BASE`. The size
  rule: larger than the take's (both sides, not equal) and on its ratio within 1%.
- `SIGMAS` is HyperFlow's whole 9-point grid, computed from its raw grid at shift 12 as IC does
  (`RAW_SIGMAS`); a sampler takes its last `inference` steps. The engines declare `IC` (the task
  and the grid), `FRAMES = 124`, `extra_key` constant, and drop the base's fl2va file.
- `attention=cudnn` comes from the take's options (carried by `_take`), not an engine default.
- The test take is a `comfy-minimax-h3-quality` take: the int8 engine renders a copy labeled
  `comfy-minimax-h3` (the same latent space) in step 4.

## Verification (cost first)

1. **Local** (free): lint; `_take` stubs (BASE check, 512x512 gives 2x, 3456x1440 kept, a wrong
   ratio refused); `check-model.sh` lists the engines.
2. **Modal CPU** (no GPU billed): `test/modal.sh install comfy-minimax-h3-quality-ic-upscale`
   (files to the Volume), then one `modal.sh exec` reading only headers: every LoRA tensor maps
   onto a ComfyUI weight of matching shape (IC 312/312, HyperFlow 314 + 2 endpoint pairs, Sharp
   208 + 50 curves, none dropped), and HyperFlow's metadata grid equals `SIGMAS`. Everything else
   is covered by step 3's forward check and step 4's video.
3. **Tiny-model parity** (free: a CPU container, or the first seconds of step 4's session if
   ComfyUI's H3 kernels need the GPU): a random small model (hidden 256, 2 layers) in both
   layouts, tiny random LoRAs in the three formats; IC's own `hyperflow/sharp/lora.install` +
   forward against our key map + `HyperFlowModel`, fp32, at the 3 sigmas. Pass: max difference at
   fp32 rounding. No second big model on the GPU.
4. **One B200 session** (~$1 in all, the only GPU spend): the 5 s 1376x576 take of the 2x upscale's verification
   (`h3in/indy-cudnn.video.latent`, a `comfy-minimax-h3-quality` take, 37 latent frames = IC's
   124) at 2x, 2752x1152, `attention=cudnn`, timed launch to saved video (the real price), then the
   int8 engine, both in one `modal.sh exec`. Checked there too: the LoRAs baked at load (no
   per-step patch). Compared by eye with that take's existing turbo and ComfyUI 2x in
   `C:\Users\barnaud\Desktop\image\minimax-h3-video\` (`indy_turbo-quality-upscale-2x.mp4`,
   `indy_comfyui-upscale-2x.mp4`), the new results saved there beside them
   (`indy_turbo-quality-ic-upscale-2x.mp4`, `indy_turbo-ic-upscale-2x.mp4`).
5. **Which upscaler keeps the take's details best** (local ffmpeg, free), on the four 2x of the
   same take: turbo 2x, ComfyUI 2x, quality IC 2x, int8 IC 2x:
   - fidelity: each 2x scaled back to the take's 1376x576 (lanczos) against the take itself,
     PSNR and SSIM per frame (mean and worst): the closer, the more existing detail kept and the
     less invented;
   - side by side: one video stacking the four 2x with their names, and frame crops at 1:1 (a
     face, text, foliage) of the take enlarged 2x (lanczos) beside each upscaler, in
     `C:\Users\barnaud\Desktop\image\minimax-h3-video\` (`indy_compare-upscalers-2x.mp4`,
     `indy_compare-upscalers-frame<N>-crop.png`);
   - the measures and the verdict in the plan's Results.
6. Later, on request: 2K (3456x1440), a 10 s take (beyond IC's 124 frames).

Risks: IC was trained on 1344x768 drafts (the test take is 1376x576, at 2x: only step 4 shows it); the int8
engine is no faster than bf16 (LoRA patches turn off the int8 matmul), its gain is memory (34 vs
66 GB); small rounding gaps (on-cast bf16 LoRAs vs IC's fp32) are measured in step 3.

## Docs
`turboCLI/doc/minimax-h3-ic-upscale-plan.md` (this plan, then Results), `implementation.md`
(engine rows, the upscale size rule, `_take`'s BASE check), `engine.md`, `MANUAL.md` (Upscaling a
take: the IC engines, a width and height for 2K), `bash/turbo/image-to-video.sh` / `install.sh` /
README usage lists, `turboCLI.pro`, `test/comfy.py` if a graph is added; turbo-offloader
`implementation.md` (HyperFlowModel, set_sigmas). Commits only on request, no Claude
attribution, no pin bump.

## Results

- **Step 1, local**: both engines discovered (BASE, 3 steps, crf 12, the base's options, the
  fl2va file dropped); up3 = [0.834942, 0.696852, 0.468753, 0] and up4 as IC's; `_take`: 512x512
  gives 2752x1152, 3456x1440 kept, 3456x1600 refused (ratio), a quality take refused by the int8
  engine, the 2x engines unchanged; a 10 s take refused (124 frames).
- **Step 3, tiny-model parity** (local CPU, fp32, IC's own `hyperflow.py` / `sharp.py` and
  Diffusers' `Timesteps` / `TimestepEmbedding`): LoRA names (qkv slices, fc1 halves, out, fc2)
  0 difference through ComfyUI's `calculate_weight`; HyperFlow's two-time embedding 0 difference;
  Sharp's curve coordinates and branch 1.2e-7.
- **Step 2, Modal CPU**: both engines installed (the full bf16 and int8 ref2va, the IC files;
  the rest reused from the Volume); the header-only check on the real files: IC 312/312
  patched, HyperFlow 314 patched + 2 endpoint pairs consumed by HyperFlowModel, Sharp 208
  patched + 50 curve branches consumed, none dropped or misshaped; HyperFlow's hyperflow_sigmas
  equal the engine's RAW_SIGMAS (gate 0.25, rank = alpha = 256). The Modal CLI ran through the
  certificate wrapper (implementation.md, Tests).
- **Step 4, B200, the quality engine alone** (the 5 s 1376x576 take, `h3in/indy-cudnn`, to
  2752x1152, `attention=cudnn` from the take): 159.8 s from launch to saved video (today's 2x of
  the same take: 201.2 s), 3 steps at about 30-40 s (the first with the warm-up), 886 LoRA
  patches applied. Three runs failed first, about $0.50 together: a name clash (`frames`, the
  module's function, shadowed), then the take's decode before sampling ended the sampling for
  the offloader (`_end_sampling` dropped `current_patcher`, so the transformer never loaded):
  fixed by `ComfyVAE.decode_input`, a decode bounded as an input node.
- **Step 5, fidelity to the take** (each 2x back to 1376x576, lanczos, against the take, 124
  frames): turbo 2x 33.85 dB / SSIM 0.9399 (worst 0.9266), ComfyUI 2x 33.85 dB / 0.9399 (worst
  0.9266), **quality IC 2x 37.92 dB / 0.9653 (worst 0.9375)**: IC keeps about 4 dB more of the
  take. By eye at 1:1 (`C:\Users\barnaud\Desktop\image\minimax-h3-video\
  indy_compare-upscalers-frame60-crop*.png`, `indy_compare-upscalers-2x.mp4`): IC holds the take's
  shapes, shading and colors closest, with the sharpest eye, lids and stubble, the jacket's own
  leather and seams; turbo's and ComfyUI's 2x (identical) are smoother, slightly waxy, their
  fine texture re-invented. IC's flaw: concentric ring ("fingerprint") patterns on bright,
  smooth skin highlights, a texture the take doesn't have.
- **Verdict (by eye, the user's)**: turbo's original `comfy-minimax-h3-quality-upscale` looks
  better than the IC 2x on this take: IC's ring patterns and sharpened skin outweigh its closer
  fidelity (the measures favour IC, the eye doesn't).
- **Sharp off** (`sharp=0`, an option on both IC engines: IC's `--sharp`, 1 by default, 0 leaves
  Sharp's LoRA and curve branches out; 628 patches): 186.6 s from launch to saved, 38.00 dB /
  SSIM 0.9673 against the take, and the same ring patterns on the skin
  (`indy_compare-upscalers-frame60-crop-sharp.png`): Sharp isn't their cause. Left: IC's LoRA
  itself on a draft off its training (576 rows at 2x, against 768-row drafts to 2K), or a gap of
  the port past its tested parts (the Qwen video item, the full sequence).
- **Decision**: both IC engines and `sharp=` kept in turboCLI as an alternative, documented
  (MANUAL: the measures and the rings); `comfy-minimax-h3-quality-upscale` stays the 2x to use.
  No more GPU spend for now. Not run: the int8 engine, IC's 4-step sampler, a 768-row take, a
  reference run of NVIDIA's own CLI on the same take (it would tell the port from IC itself).
  GPU spent in all: about $1.10 (five B200 runs, three of them failing within a minute).

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

#==================================================================================================
#  LGPL shared diffusion engine. What both front-ends (cli.py one-shot, server.py HTTP) share:
#  offload-backend discovery, engine discovery, the resident-pipe cache, and one generation.
#
#  Engine-specific knowledge lives in engine/<name>.py (NAME/PIPELINE/MODES/CFG + optional
#  extra_key()/loras()/load()); placement strategy lives behind the ../backend/<mode>/ seam
#  (external, may be GPL) -- reached here only by string name + the fixed seam methods, never by
#  importing a backend.
#  See PLAN-engine.md.
#==================================================================================================

import os
import gc
import glob
import importlib
import traceback

#--------------------------------------------------------------------------------------------------
# Offload backend discovery -- MUST run before `import torch`
#--------------------------------------------------------------------------------------------------

# Optional weight-streaming plug-ins discovered as ../backend/<mode>/ subpackages (relative to
# cwd, which the wrappers cd into -- the deployed diffusion dir). Each exposes the seam
# (pre_torch_init/available/supports/load_pipe + optional prepare/reclaim/release);
# pre_torch_init() must precede torch, and a failed init is skipped. This names no specific
# backend.

OFFLOAD_MODES = set()     # every backend/<mode>/ present (whether or not it initialised)
OFFLOAD_BACKENDS = {}     # mode -> ready backend module (pre_torch_init succeeded + available())

# Built-in (non-backend) placement strategies handled by Context.apply_offload().
NATIVE_OFFLOADS = ("none", "model_cpu", "sequential_cpu")

for _path in sorted(glob.glob(os.path.join("backend", "*", "__init__.py"))):
    _mode = os.path.basename(os.path.dirname(_path))

    OFFLOAD_MODES.add(_mode)

    try:
        _mod = importlib.import_module("backend." + _mode)

        _mod.pre_torch_init()

        if _mod.available():
            OFFLOAD_BACKENDS[_mode] = _mod
    except Exception:
        pass

import torch

#--------------------------------------------------------------------------------------------------
# Engine discovery -- cheap: engine modules import no torch/diffusers at top level
#--------------------------------------------------------------------------------------------------

ENGINES = {}   # name -> engine module

for _path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "engine", "*.py"))):
    _file = os.path.basename(_path)

    if _file.startswith("_"):
        continue

    _emod = importlib.import_module("%s.engine.%s" % (__package__, _file[:-3]))

    ENGINES[_emod.NAME] = _emod


def log(message):
    print(message, flush=True)


#--------------------------------------------------------------------------------------------------
# Resident pipeline cache
#--------------------------------------------------------------------------------------------------

# The currently loaded pipeline and the key it was loaded with:
# (name, model, renderer, cuda_offload, slicing) + engine.extra_key(params).
pipe = None
pipe_key = None


def _resolve(spec):
    """'diffusers:Flux2KleinPipeline' -> the class (imported lazily, only when an engine loads)."""
    module_name, cls_name = spec.split(":")

    return getattr(importlib.import_module(module_name), cls_name)


def _device_dtype(renderer):
    if renderer == "cuda":
        return "cuda", torch.bfloat16

    if renderer == "mps":
        return "mps", torch.float16

    return "cpu", torch.float32


def parse_loras(spec):
    """Parse the optional `loras` param into [(path, weight), ...].

    Wire format: comma-separated entries, each "<path>@<weight>" (weight 0.0-1.0). The weight is
    taken after the LAST '@' so Windows paths (which contain ':' but never '@') parse cleanly. A
    missing weight -- no '@', or a trailing '@' with nothing after it -- defaults to 1.0.
    Empty/blank -> []."""
    out = []

    for item in (spec or "").split(","):
        item = item.strip()

        if not item:
            continue

        path, sep, weight = item.rpartition("@")

        if sep:
            weight = weight.strip()
            # Clamp to [0.0, 1.0]; a missing weight (trailing '@') defaults to 1.0.
            out.append((path.strip(), max(0.0, min(1.0, float(weight))) if weight else 1.0))
        else:
            # No '@' -> rpartition put the whole string in `weight`; the path is it, weight
            # defaults.
            out.append((weight.strip(), 1.0))

    return out


class Ctx:
    """The firewall handed to an engine's load(): core helpers only, never backend internals.

    backend is the resolved seam module (or None). apply_offload()/from_pretrained() are the shared
    mechanics so an engine module never re-implements placement or the diffusers load. loras is the
    user-supplied LoRA list [(path, weight), ...] (empty if none); an engine merges with its own.
    """

    def __init__(self, renderer, cuda_offload, backend, loras=()):
        self.renderer = renderer
        self.cuda_offload = cuda_offload
        self.backend = backend
        self.loras = list(loras)
        self.device, self.dtype = _device_dtype(renderer)

    def apply_loras(self, p, lora_list):
        """Load + activate a list of (path, weight) LoRAs on a NON-backend pipe (the diffusers
        path). No-op for an empty list. The backend path passes the same list to
        load_pipe(lora_files=...) instead, which applies them on-cast. Used by _default_load and by
        the LoRA-preset engines, so the load_lora_weights + set_adapters dance is written once."""
        if not lora_list:
            return p

        names, weights = [], []

        for i, (path, weight) in enumerate(lora_list):
            name = "lora%d" % i
            folder, filename = os.path.split(path)

            p.load_lora_weights(folder or ".", weight_name=filename, adapter_name=name)

            names.append(name)
            weights.append(weight)

        p.set_adapters(names, adapter_weights=weights)

        return p

    def from_pretrained(self, spec, params):
        cls = _resolve(spec)

        return cls.from_pretrained(
            params["model"],
            torch_dtype=self.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

    def apply_offload(self, p):
        """The shared non-backend placement (none / model_cpu / sequential_cpu). No-op intent for a
        backend pipe, which the backend already placed."""
        device, offload = self.device, self.cuda_offload

        if device == "cuda":
            if offload == "model_cpu":
                p.enable_model_cpu_offload()
            elif offload == "sequential_cpu":
                p.enable_sequential_cpu_offload()
            else:
                p.to("cuda")
        elif device == "mps":
            if offload == "model_cpu":
                p.enable_model_cpu_offload(device="mps")
            elif offload == "sequential_cpu":
                p.enable_sequential_cpu_offload(device="mps")
            else:
                p.to("mps")

        return p


def engine_type(mod):
    """The engine identity the backend seam understands (it keys the pipeline class + supports()
    off this string). Defaults to the registry NAME; a module variant -- e.g. a LoRA preset that
    registers under its own NAME but reuses a base engine -- overrides TYPE to keep speaking the
    seam's vocabulary while staying selectable as a distinct engine."""
    return getattr(mod, "TYPE", mod.NAME)


def _default_load(ctx, mod, params):
    """Default load for engines with no load() hook: build + place via the standard diffusers path.
    A module's optional loras(params) -> [(filename, weight), ...] is resolved against the model
    folder and applied first, then any user-supplied LoRAs (ctx.loras)."""
    model = params["model"]

    preset = mod.loras(params) if hasattr(mod, "loras") else []
    lora_files = [(os.path.join(model, name), weight) for name, weight in preset] + ctx.loras

    if ctx.backend:
        return ctx.backend.load_pipe(
            model, ctx.dtype, engine_type(mod), device=ctx.device, lora_files=lora_files or None
        )

    p = ctx.from_pretrained(mod.PIPELINE, params)
    ctx.apply_loras(p, lora_files)

    return ctx.apply_offload(p)


def _engine_key(mod, params):
    extra = ()

    if hasattr(mod, "extra_key"):
        extra = tuple(mod.extra_key(params))

    # User LoRAs are part of the pipe identity, so changing them reloads (== a different cached
    # pipe).
    loras = tuple(parse_loras(params.get("loras", "")))

    return (mod.NAME, params["model"], params["renderer"],
            params["cuda_offload"], params["slicing"]) + extra + (loras,)


def release_pipe(reason):
    """Drop the resident pipeline (config change / idle / explicit clear). Tears down any offload
    backend first so host-pinned weights + file handles + offloader<->module cycle are freed."""
    global pipe, pipe_key

    if pipe is None:
        return False

    log("Releasing current model (%s)..." % reason)

    renderer = pipe_key[2]

    backend = getattr(pipe, "_offload_backend", None)

    if backend is not None:
        backend.release(pipe)

    del pipe

    pipe = None
    pipe_key = None

    gc.collect()

    if renderer == "cuda":
        torch.cuda.empty_cache()
    elif renderer == "mps":
        torch.mps.empty_cache()

    return True


def install_encode_cache(pipe):
    """Cache the text encoder's output by prompt: key each result on the encode call's full inputs,
    hold it on CPU, and evict under host-RAM pressure. Host RAM is read via psutil --
    psutil.virtual_memory().available below min(10GB, max(2GB, 10% of RAM)) triggers eviction, worst
    entry first by 1.3**generation_age * cached_bytes with an LRU tiebreak. The embedding is kept on
    the CPU and moved back to the compute device on a hit, which also frees the VRAM it would otherwise
    pin and lets it count as host RAM for eviction.

    diffusers encodes inside __call__ via self.encode_prompt; wrapping that one method means a repeated
    prompt skips the encoder's forward entirely -- so with an idempotent model loader the un-run encoder
    is never streamed in, leaving the diffusion model resident (a warm run spends ~0s on a cached
    encode). The key is the call's FULL (args, kwargs): an argument that isn't cleanly hashable -- a
    tensor/image, e.g. an image-conditioned edit encode -- makes the call uncacheable, so it never gets
    a false hit. No-op if the pipe has no encode_prompt or the cache is already installed."""
    real = getattr(pipe, "encode_prompt", None)

    if real is None or getattr(pipe, "_encode_cache_installed", False):
        return

    import time
    import bisect
    import psutil

    OLD_OOM_MULT = 1.3   # older generations weighted heavier for eviction
    BASE_USAGE = 0.05    # floor per entry -- keeps zero-size entries LRU-ordered
    cpu = torch.device("cpu")
    target = min(10 * 1024 ** 3, max(2 * 1024 ** 3, psutil.virtual_memory().total * 0.10))

    uncacheable = object()

    def norm(v):
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, (torch.device, torch.dtype)):
            return str(v)
        if isinstance(v, (list, tuple)):
            parts = tuple(norm(x) for x in v)
            return uncacheable if any(p is uncacheable for p in parts) else parts
        return uncacheable

    def key_of(args, kwargs):
        a = tuple(norm(v) for v in args)
        kw = tuple((k, norm(kwargs[k])) for k in sorted(kwargs))
        if any(x is uncacheable for x in a) or any(v is uncacheable for _, v in kw):
            return None
        return (a, kw)

    def move(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(move(o, device) for o in obj)
        return obj

    def nbytes(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(nbytes(o) for o in obj)
        return 0

    cache = {}   # key -> [cpu_value, bytes, timestamp, generation]
    gen = [0]

    def ram_release():
        if psutil.virtual_memory().available >= target:
            return
        scored = []
        for k, (_, sz, ts, g) in cache.items():
            if g == gen[0]:                    # never evict this generation's own entries
                continue
            bisect.insort(scored, ((OLD_OOM_MULT ** (gen[0] - g)) * (sz + BASE_USAGE), ts, k))
        while scored and psutil.virtual_memory().available < target:
            cache.pop(scored.pop()[2], None)   # highest oom_score first

    def cached_encode(*args, **kwargs):
        key = key_of(args, kwargs)

        if key is None:                        # tensor/image arg -> not safely cacheable
            return real(*args, **kwargs)

        gen[0] += 1
        hit = cache.get(key)

        if hit is not None:
            cpu_value, sz, _, _ = hit
            cache[key] = [cpu_value, sz, time.time(), gen[0]]
            device = getattr(pipe, "_execution_device", None) or pipe.device
            # embeds are read-only downstream (diffusers never mutates them), so hand back the stored
            # tensor directly -- on a same-device hit this aliases it, deliberately, to skip a copy
            return move(cpu_value, device)

        out = real(*args, **kwargs)
        cache[key] = [move(out, cpu), nbytes(out), time.time(), gen[0]]
        ram_release()

        return out

    pipe.encode_prompt = cached_encode
    pipe._encode_cache_installed = True


def get_pipe(mod, params, emit):
    """Load-or-reuse the resident pipeline for this engine+config. Reuses when the key is
    unchanged, else releases and reloads (== server.sh get_pipe)."""
    global pipe, pipe_key

    key = _engine_key(mod, params)

    if pipe is not None and pipe_key == key:
        return pipe

    reload = pipe is not None

    if reload:
        release_pipe("config changed")

    emit("loading %s model (%s / %s)..." % (mod.NAME, params["renderer"], params["cuda_offload"]))

    renderer = params["renderer"]

    if renderer == "cuda":
        log("GPU: " + torch.cuda.get_device_name(0))
    elif renderer == "mps":
        log("GPU: Apple MPS")

    backend = OFFLOAD_BACKENDS.get(params["cuda_offload"])

    ctx = Ctx(
        renderer, params["cuda_offload"], backend, loras=parse_loras(params.get("loras", "")),
    )

    loader = getattr(mod, "load", None)

    if loader is None:
        p = _default_load(ctx, mod, params)
    else:
        p = loader(ctx, params)

    # Keep the backend reachable for prepare/reclaim/release; None for an unbacked pipe.
    if backend is not None:
        p._offload_backend = backend

    # NOTE: This might improve performances.
    p.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    if params["slicing"] == "slice":
        # NOTE: These are useful for low VRAM gpu(s).
        p.enable_attention_slicing()
        p.vae.enable_slicing()
        p.vae.enable_tiling()

    # Prompt-encode cache for the modes with no backend (none / model_cpu / sequential_cpu); a backend
    # may install its own in prepare() instead.
    if backend is None:
        install_encode_cache(p)

    pipe = p
    pipe_key = key

    emit("model ready")

    return pipe


def generate(params, emit, should_stop=None):
    """Run one generation. Returns True if an image was saved, False otherwise (validation error,
    cancelled, or superseded). Unexpected errors propagate to the caller (cli/server wraps them).

    params: the same plain dict the HTTP API builds (engine, mode, model, prompt, output, images,
    width, height, seed, inference, renderer, cuda_offload, slicing).
    emit: a line sink (print for cli, the socket stream for server).
    should_stop: optional callable -> None | "cancel" | "supersede" (server preemption; None for
    cli).
    """
    engine = params["engine"]
    mode = params["mode"]

    mod = ENGINES.get(engine)

    if mod is None:
        emit("ERROR: unknown engine '%s'" % engine)

        return False

    if mode not in mod.MODES:
        emit("ERROR: engine '%s' does not support mode '%s'" % (engine, mode))

        return False

    cuda_offload = params["cuda_offload"]

    # Offload backends are validated generically so this names no specific backend.
    if cuda_offload in OFFLOAD_MODES:
        backend = OFFLOAD_BACKENDS.get(cuda_offload)

        if backend is None:
            emit("ERROR: %s offload unavailable (backend not installed)" % cuda_offload)

            return False

        if not backend.supports(engine_type(mod)):
            emit("ERROR: %s offload does not support engine '%s'" % (cuda_offload, engine))

            return False

    elif cuda_offload not in NATIVE_OFFLOADS:
        available = ", ".join(sorted(set(NATIVE_OFFLOADS) | OFFLOAD_MODES))

        emit("ERROR: unknown cuda_offload '%s' (no backend/%s/ found; available: %s)"
             % (cuda_offload, cuda_offload, available))

        return False

    p = get_pipe(mod, params, emit)

    device, _ = _device_dtype(params["renderer"])

    seed = int(params["seed"])

    if seed == -1:
        generator = None
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    width = int(params["width"])
    height = int(params["height"])

    inference = int(params["inference"])

    prompt = params["prompt"]

    kwargs = dict(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=inference,
        generator=generator,
    )

    cfg_name, cfg_value = mod.CFG
    kwargs[cfg_name] = cfg_value

    if mode == "edit":
        from PIL import Image, ImageOps

        image_list = [s.strip() for s in params.get("images", "").split(",") if s.strip()]

        prompt_images = []

        for ip in image_list:
            emit("loading input: %s" % ip)

            img = Image.open(ip).convert("RGB")

            scale = min(img.width / width, img.height / height, 1.0)

            image_width = max(1, round(width * scale))
            image_height = max(1, round(height * scale))

            img = ImageOps.fit(img, (image_width, image_height), Image.Resampling.LANCZOS)

            prompt_images.append(img)

        kwargs["image"] = prompt_images

        # Clear memory before the heavy lifting.
        gc.collect()

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # Returns None to keep going, or why we should stop: "cancel" / "supersede".
    def stop_reason():
        return should_stop() if should_stop else None

    preview = prompt if len(prompt) <= 60 else prompt[:60] + "..."

    emit('generating "%s" (%d steps)...' % (preview, inference))

    # This callback handles interruption. If a newer request has arrived, we ask the pipeline to
    # interrupt so the GPU is freed for it.
    def step_end(pipeline, step, timestep, callback_kwargs):
        reason = stop_reason()

        if reason:
            pipeline._interrupt = True

            if reason == "cancel":
                emit("cancelling on request...")
            else:
                emit("stopping for a newer request...")

        return callback_kwargs

    try:
        p._interrupt = False
    except Exception:
        pass

    # We want one clean line per step carrying the SAME elapsed/rate as diffusers own tqdm bar. A
    # disabled bar computes no stats, so we keep the bar fully alive but route its rendering
    # to a sink, then read its format_dict and mirror the native figures.
    class _Sink:
        def write(self, *_):
            pass

        def flush(self):
            pass

    try:
        p.set_progress_bar_config(file=_Sink(), leave=False)
    except Exception:
        pass

    def emit_step(bar):
        fd      = bar.format_dict
        total   = fd.get("total") or inference
        done    = fd.get("n") or 0
        secs    = fd.get("elapsed") or 0.0
        rate    = fd.get("rate")

        percent = int(round(done * 100.0 / total)) if total else 0

        # Format elapsed as MM:SS with tqdm's own helper so it matches the native bar exactly.
        clock = bar.format_interval(secs)

        # tqdm's own convention: s/it once it passes 1s/it, it/s while faster. rate is None until
        # the first timed update.
        if rate:
            inv   = 1.0 / rate
            speed = ("%.2fs/it" % inv) if inv > 1.0 else ("%.2fit/s" % rate)
        else:
            speed = "?it/s"

        emit("%3d%%|step %d/%d (%s, %s)" % (percent, done, total, clock, speed))

    original_progress_bar = type(p).progress_bar

    # Emit our line right AFTER each tqdm update so format_dict reflects the step just completed.
    # The step_end callback above fires BEFORE the update, which would lag the figures by one step.
    def hooked_progress_bar(*args, **kwargs_):
        bar = original_progress_bar(p, *args, **kwargs_)

        # Heartbeat emitted here (at bar creation = real loop start, after text-encoding) so its
        # 00:00 is truthful and not inflated by encode/warm-up time.
        emit("  0%%|step 0/%d (00:00)" % inference)

        original_update = bar.update

        def update(n=1):
            result = original_update(n)

            try:
                emit_step(bar)
            except Exception:
                pass

            return result

        bar.update = update

        return bar

    p.progress_bar = hooked_progress_bar

    try:
        # Per-generation load boundary for the offload backend (e.g. reload a managed text encoder
        # to GPU before the pipeline reads its execution device). No-op for a pipe with no backend.
        backend = getattr(p, "_offload_backend", None)

        if backend is not None:
            backend.prepare(p)

        with torch.inference_mode():
            result = p(callback_on_step_end=step_end, **kwargs)
    finally:
        # Restore the cached pipeline's original method.
        try:
            del p.progress_bar
        except Exception:
            pass

        # Per-generation offload-backend housekeeping (return torch's retained allocator pool).
        # No-op for a pipe with no backend.
        backend = getattr(p, "_offload_backend", None)

        if backend is not None:
            try:
                backend.reclaim(p)
            except Exception:
                log(traceback.format_exc())

    # Interrupted mid-run: discard the partial result.
    reason = stop_reason()

    if getattr(p, "_interrupt", False) or reason:
        if reason == "cancel":
            emit("CANCELLED: stopped on request, server is idle")
        else:
            emit("SUPERSEDED: a newer request took over, this one was cancelled")

        return False

    image = result.images[0]

    image.save(params["output"])

    # NOTE: Send "Saved:" so the client gets the result as early as possible.
    emit("Saved: " + params["output"])

    return True

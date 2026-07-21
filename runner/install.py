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

# Model install front-end -- the install-side mirror of the run engines. One command installs
# everything an engine needs: its base model plus any LoRAs it declares, into the model dir.
#
#   python -m runner.install --engine <name> [--model <M>] [--dtype default]
#   python -m runner.install --engine <name> --remove
#
# The base-model revision is pinned in the engine's MODEL["revision"] (not a CLI flag), and the
# model is saved into "<install>/model/<model>" (default_folder(), deduced from the runner's path).
#
# `--model` is optional for engines that fix their own model at the module level (e.g.
# qwen-image-edit-2511 and flux2-4b each declare MODEL["model"]); an engine that omits it leaves
# the choice to `--model`.
#
# So `flux2-4b` installs FLUX.2-klein-4B; each qwen engine installs the Qwen model plus exactly the
# LoRAs it uses (none / lightning / lightning + angles). Each engine module under runner/engine
# declares its install needs (MODEL repo + optional LORAS list).
#
# Installs are selective via the engine registry (engine/<id>/engine.json records each install's
# revision): a base already at the pinned revision -- per any engine record for that model -- is
# kept (no re-download, no torch import), and only missing LoRAs are fetched, so installing
# qwen-image-edit-2511-lightning over an existing qwen-image-edit-2511 just pulls the lightning
# LoRA.
#
# Each install also writes a per-engine registry entry engine/<id>/engine.json (its model + LoRAs +
# revisions; for a comfy engine its component refs + loader scaffold). This is the source of truth
# for "installed", separate from the canonical (shared) weights under model/. Removal is
# reference-counted: --remove drops the registry entry, then deletes the model / LoRAs / comfy
# components only when no other installed engine still references them (see _remove).
#
# Separate from the generation cli/server: install is ONLINE and needs no GPU, so this does NOT
# import core (no offload-backend discovery / CUDA init). The wrapper install.sh keeps the env
# (HF_HOME, hf-transfer). Run from the deployed diffusion dir so `engine` is importable.

import os
import sys
import json
import glob
import shutil
import argparse
import importlib

# LoRAs install into a "lora/" subfolder of the model dir (apart from the base checkpoint files).
LORA_DIR = "lora"


def default_folder():
    """The model folder: `model` inside the install dir, side by side with runner/
    (<install>/model; the runner lives in <install>/runner/). Deduced from this file's path -- no
    --folder needed. Duplicated from core.default_folder() so install/check stay torch-free (core
    imports torch). Holds only canonical weights, shared across engines."""
    install = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(install, "model")


def engine_folder():
    """The engine registry: `engine` inside the install (beside model/, runner/). One dir per
    installed engine holds its engine.json manifest and, for a comfy engine, its loader scaffold.
    Duplicated from core.engine_folder() so install/check stay torch-free."""
    install = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(install, "engine")


def _engine_dir(engine_id):
    return os.path.join(engine_folder(), engine_id)


def _read_engine(engine_id):
    """Read engine/<id>/engine.json -> the registry record, or None when absent."""
    path = os.path.join(_engine_dir(engine_id), "engine.json")

    if not os.path.isfile(path):
        return None

    with open(path) as f:
        return json.load(f)


def _write_engine(record):
    """Write engine/<record["id"]>/engine.json -- the per-engine registry manifest. Replacing an
    existing record also GCs what the old one referenced and the new one no longer does (a
    component dropped from COMFY, a LoRA dropped from LORAS, a switched model): the record is the
    only map from an engine to its files, so this is the last moment anything can find them."""
    out = _engine_dir(record["id"])
    old = _read_engine(record["id"])

    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, "engine.json"), "w") as f:
        json.dump(record, f, indent=2)

    # After the write, so the new record counts as a live reference -- otherwise a plain reinstall
    # would GC the files it just downloaded.
    if old is not None:
        _gc(old, _referenced())


def _registry():
    """Every installed engine's registry record (each engine/<id>/engine.json)."""
    root = engine_folder()

    if not os.path.isdir(root):
        return []

    records = []

    for eid in sorted(os.listdir(root)):
        record = _read_engine(eid)

        if record is not None:
            records.append(record)

    return records


def _under(path, root):
    """True when `path` is `root` or lives inside it (normalized, case-insensitive on Windows)."""
    path = os.path.normcase(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(root))

    return path == root or path.startswith(root + os.sep)


def _stock_record(engine_id, model, revision, loras):
    """The registry record for a stock engine: the model dir it uses + its own declared LoRAs."""
    return {"id": engine_id, "model": model, "revision": revision,
            "loras": [{"file": lora["file"], "revision": lora.get("revision")} for lora in loras]}


def _discover():
    # name -> engine module (cheap: modules hold constants only, no torch/diffusers at top).
    engines = {}

    for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "engine", "*.py"))):
        f = os.path.basename(path)

        if f.startswith("_"):
            continue

        m = importlib.import_module("%s.engine.%s" % (__package__, f[:-3]))
        engines[m.ID] = m

    # Resolve BASE inheritance (variants inherit contract symbols they omit -- engine/_inherit.py).
    importlib.import_module("%s.engine._inherit" % __package__).resolve(engines)

    return engines


def _resolve(spec):
    """'diffusers:Flux2KleinPipeline' -> the class."""
    module_name, cls_name = spec.split(":")

    return getattr(importlib.import_module(module_name), cls_name)


def _model_revision(model):
    """The revision recorded for a shared model dir by any installed engine (they agree -- engines
    sharing a model pin the same MODEL["revision"]); None when no engine records it. Lets install /
    check derive the physical dir's revision from the registry, so model dirs need no .commit."""
    for record in _registry():
        if record.get("model") == model:
            return record.get("revision")

    return None


def _engine_installed(mod):
    """True when this engine has a registry entry (engine/<id>/engine.json) whose referenced files
    are all present -- the source of truth for check.py. Comfy: the scaffold's model_index.json +
    every component path exist. Stock: the model dir's model_index.json + every recorded LoRA file
    exist (the record itself carries the installed revision)."""
    record = _read_engine(mod.ID)

    if record is None:
        return False

    comfy = record.get("comfy")

    if comfy is not None:
        if not os.path.isfile(os.path.join(_engine_dir(mod.ID), "model_index.json")):
            return False

        components = comfy.get("components", [])

        return bool(components) and all(os.path.isfile(_comp_path(comfy, c)) for c in components)

    model_dir = os.path.join(default_folder(), record["model"])

    if not os.path.isfile(os.path.join(model_dir, "model_index.json")):
        return False

    return all(os.path.isfile(os.path.join(model_dir, LORA_DIR, lora["file"]))
               for lora in record.get("loras", []))


def _models_root(comfy):
    """ComfyUI's models/ dir: <comfy>/ComfyUI/models (portable build) or <comfy>/models."""
    nested = os.path.join(comfy, "ComfyUI", "models")

    return nested if os.path.isdir(nested) else os.path.join(comfy, "models")


def _comp_path(comfy, comp):
    """Absolute path of a comfy component = its record's `root` (the reused ComfyUI install) joined
    with the component's root-relative `path` (e.g. models/vae/qwen_image_vae.safetensors)."""
    return os.path.join(comfy["root"], *comp["path"].split("/"))


def _comp_key(path):
    """A component file's identity for reference counting: absolute + normcased, so two records
    naming the same file with different spellings (a relative --comfy root vs an absolute one)
    compare equal -- a mismatch here would GC a file another engine still uses."""
    return os.path.normcase(os.path.abspath(path))


def _install_comfy(mod, comfy, out):
    """Install a ComfyUI-reuse engine (--comfy): keep each COMFY component already in the ComfyUI
    install, download only the missing ones there, fetch the tiny SCAFFOLD (configs/tokenizer/
    scheduler, no weights) into the engine registry dir `out`, and write out/engine.json naming the
    reused files. No base repo, no GPU -- so a user who already runs ComfyUI does not re-download
    the big weights. `out` is the engine registry dir (engine/<id>), not a model dir."""
    from huggingface_hub import scan_cache_dir, hf_hub_download, snapshot_download
    from huggingface_hub.errors import CacheNotFound

    comfy_cfg = mod.COMFY
    models = _models_root(comfy)
    root = os.path.dirname(models)  # the ComfyUI install dir; components are stored relative to it

    repositories = []
    manifest = []

    # Components: reuse each file already under ComfyUI's models/, download the rest there.
    for c in comfy_cfg["components"]:
        dest = os.path.join(models, *c["path"].split("/"))

        if os.path.isfile(dest):
            print("Component present: %s" % c["path"], flush=True)
        else:
            print("Downloading component: %s" % c["path"], flush=True)

            # A component may name its own repo (engines whose files live in different Comfy-Org
            # repos); else the engine's top-level COMFY repo. In-repo path defaults to
            # split_files/<path> (Comfy-Org repackaged layout) but a component may set `filename`
            # explicitly (e.g. a LoRA whose file sits at a plain repo's root). Land it in models/
            # then move to its subdir.
            repo = c.get("repository") or comfy_cfg.get("repository")
            src = hf_hub_download(repo_id=repo,
                                  filename=c.get("filename") or ("split_files/" + c["path"]),
                                  revision=c.get("revision") or comfy_cfg.get("revision") or None,
                                  local_dir=models)

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(src, dest)

            repositories.append(repo)

        rel = os.path.relpath(dest, root).replace(os.sep, "/")  # e.g. models/vae/qwen_image_vae...
        manifest.append({"role": c["role"], "path": rel})

    shutil.rmtree(os.path.join(models, "split_files"), ignore_errors=True)

    # Scaffold: the small diffusers configs/tokenizer/scheduler (no weights) for offline load.
    scaffold = mod.SCAFFOLD
    repo = scaffold["repository"] + "/" + scaffold["model"]

    print("Fetching scaffold: %s" % repo, flush=True)

    snapshot_download(repo_id=repo, revision=scaffold.get("revision") or None,
                      allow_patterns=scaffold["allow_patterns"], local_dir=out)

    # snapshot_download leaves a .cache/ bookkeeping dir in local_dir; drop it.
    shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)

    _prune_scaffold(out, scaffold["allow_patterns"])

    repositories.append(repo)

    # engine.json: the registry record -- component refs (for load + reference-counted GC) and the
    # loader scaffold's source. `root` is the reused ComfyUI install dir; components store their
    # path relative to it. `external` = weights outside our model folder (a user's ComfyUI); it is
    # diagnostic only -- what actually protects them is _gc's live _under(path, default_folder())
    # guard, recomputed from the real path rather than trusting this install-time snapshot.
    _write_engine({
        "id": mod.ID, "model": None, "revision": None, "loras": [],
        "comfy": {"root": root, "external": not _under(models, default_folder()),
                  "components": manifest,
                  "scaffold": {"repository": scaffold["repository"], "model": scaffold["model"],
                               "revision": scaffold.get("revision")}},
    })

    # Trim the HF cache for anything pulled (a pure reuse pulls nothing, so it may not exist).
    try:
        cache = scan_cache_dir()

        for repo_entry in cache.repos:
            if repo_entry.repo_id in repositories:
                cache.delete_revisions(
                    *[rev.commit_hash for rev in repo_entry.revisions]).execute()
    except CacheNotFound:
        pass

    print("Done.", flush=True)


def _prune_scaffold(out, allow_patterns):
    """Drop scaffold files a previous install left behind after SCAFFOLD's allow_patterns were
    narrowed. A file the patterns no longer match is one snapshot_download would never place here,
    so it is stale whatever the fetch did -- which also makes this safe when snapshot_download
    falls back to "returning existing local_dir" offline. A scaffold is per-engine, never shared,
    so `out` is ours alone to trim (external ComfyUI or not). Runs after the download rather than
    wiping before it: `out` holds engine.json, the GC's only record of what this engine owns, so a
    failed fetch must leave it intact. engine.json is not part of the repo -- keep it."""
    from huggingface_hub.utils import filter_repo_objects

    files = []

    for dirpath, _dirnames, filenames in os.walk(out):
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), out).replace(os.sep, "/")

            if rel != "engine.json":
                files.append(rel)

    keep = set(filter_repo_objects(files, allow_patterns=allow_patterns))

    for rel in files:
        if rel not in keep:
            os.remove(os.path.join(out, *rel.split("/")))
            print("Removed stale scaffold file %s" % rel, flush=True)

    _prune_empty(out)


def _prune_empty(top):
    """Remove empty directories under `top` bottom-up (including `top` if it ends up empty)."""
    if not os.path.isdir(top):
        return

    for dirpath, _dirnames, _filenames in os.walk(top, topdown=False):
        if not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def _referenced():
    """What the registry as it stands right now still points at: model names, (model, lora file)
    pairs and component keys. A caller decides what "still referenced" means purely by when it asks
    -- after dropping an entry, the survivors; after writing a new one, the survivors plus it."""
    models, loras, comps = set(), set(), set()

    for record in _registry():
        if record.get("model"):
            models.add(record["model"])

        for lora in record.get("loras", []):
            loras.add((record["model"], lora["file"]))

        comfy = record.get("comfy") or {}
        for comp in comfy.get("components", []):
            comps.add(_comp_key(_comp_path(comfy, comp)))

    return models, loras, comps


def _gc(record, kept):
    """Delete everything `record` referenced that the live set `kept` no longer names -- its model
    dir (or, when the base survives, just its orphaned LoRA files) and its comfy components. The
    single expression of the deletion policy: only files under our own model folder are ever
    touched, so a reused ComfyUI install elsewhere on disk is read-only to us."""
    models_kept, loras_kept, comps_kept = kept

    # Stock model + LoRA GC.
    model = record.get("model")

    if model:
        model_dir = os.path.join(default_folder(), model)

        if model not in models_kept:
            shutil.rmtree(model_dir, ignore_errors=True)
            print("Removed model %s" % model, flush=True)
        else:
            # Base still used -> drop only this engine's now-orphaned LoRA files.
            for lora in record.get("loras", []):
                if (model, lora["file"]) in loras_kept:
                    continue

                path = os.path.join(model_dir, LORA_DIR, lora["file"])

                if os.path.isfile(path):
                    os.remove(path)
                    print("Removed LoRA %s" % lora["file"], flush=True)

    # Comfy component GC -- only files under our model folder; never a user's external ComfyUI.
    root = default_folder()

    rec_comfy = record.get("comfy") or {}
    for comp in rec_comfy.get("components", []):
        path = _comp_path(rec_comfy, comp)

        if _comp_key(path) in comps_kept:
            continue

        if _under(path, root) and os.path.isfile(path):
            os.remove(path)
            print("Removed component %s" % os.path.basename(path), flush=True)

    _prune_empty(os.path.join(root, "ComfyUI"))


def _remove(engine_id):
    """Remove an engine: delete its registry entry (engine/<id>), then GC the model / LoRAs / comfy
    components it referenced that no *remaining* installed engine still needs -- and only files
    under our model folder (a user's external ComfyUI is never touched)."""
    record = _read_engine(engine_id)
    edir = _engine_dir(engine_id)

    if record is None and not os.path.isdir(edir):
        print("%s is not installed" % engine_id, flush=True)
        return

    # Drop the registry entry first, so the survivor scan below excludes this engine.
    shutil.rmtree(edir, ignore_errors=True)

    if record is None:
        print("Removed %s" % engine_id, flush=True)
        return

    # Entry dropped above, so the registry now holds exactly the survivors.
    _gc(record, _referenced())

    print("Removed %s" % engine_id, flush=True)


def main():
    parser = argparse.ArgumentParser(prog="runner.install")

    parser.add_argument("--engine", required=True)
    # --model: model name, e.g. FLUX.2-klein-4B; optional when the engine declares its own
    parser.add_argument("--model", default=None)
    # --dtype: "default" keeps the checkpoint's native dtype (no cast on save); a concrete dtype
    # (bfloat16/float16/float32) re-casts the weights on disk. Run-time dtype is independent of
    # this (core._device_dtype picks it from the renderer), so "default" is the right install
    # choice unless you specifically want a smaller/larger on-disk copy.
    parser.add_argument("--dtype", default="default")
    # --remove: delete the engine's model directory (base model + any LoRAs in it) instead of
    # installing.
    parser.add_argument("--remove", action="store_true")
    # --comfy: optional ComfyUI install dir to reuse (COMFY engines only). Omitted -> a self-
    # contained model/ComfyUI/ layout in our model dir (see the COMFY dispatch / _install_comfy).
    parser.add_argument("--comfy", default=None)

    args = parser.parse_args()

    mod = _discover().get(args.engine)

    if mod is None:
        print("ERROR: unknown engine '%s'" % args.engine)
        sys.exit(1)

    # A COMFY engine needs no MODEL: its registry entry + scaffold key off ID (see the COMFY
    # dispatch below), so only a stock engine must name the canonical repo it installs.
    model = getattr(mod, "MODEL", None)

    if model is None and not hasattr(mod, "COMFY"):
        print("ERROR: engine '%s' is not installable (no MODEL)" % args.engine)
        sys.exit(1)

    # --remove: reference-counted -- drop the registry entry, then GC any model/LoRAs/comfy
    # components no other installed engine still references (see _remove).
    if args.remove:
        _remove(mod.ID)
        return

    # ComfyUI-reuse engines reuse a ComfyUI install's split single files: with --comfy an existing
    # install, else a self-contained model/ComfyUI/ layout. The scaffold + registry entry live in
    # engine/<id>, referencing the (canonical, shared) weights under model/.
    if hasattr(mod, "COMFY"):
        # Absolute, so every record's `root` spells its components the same way -- _comp_key
        # compares them across records and a mismatch would GC a file another engine still uses.
        comfy = os.path.abspath(args.comfy or os.path.join(default_folder(), "ComfyUI"))

        _install_comfy(mod, comfy, _engine_dir(mod.ID))
        return

    if args.comfy:
        print("ERROR: engine '%s' does not reuse a ComfyUI install (--comfy not supported)"
              % args.engine)
        sys.exit(1)

    # --- stock install: canonical diffusers repo into model/<name> (shared across engines) ---
    # The model name comes from --model, or from the engine's own MODEL["model"] when it declares
    # one (e.g. qwen-image-edit-2511, a single fixed model). One of the two must be present.
    name = args.model or model.get("model")

    if name is None:
        print("ERROR: engine '%s' needs --model (it declares no default model)" % args.engine)
        sys.exit(1)

    loras = getattr(mod, "LORAS", [])

    # The base-model revision is pinned in the engine's MODEL (mutable HF repos -> reproducible
    # installs). The model is saved into "<install>/model/<model>" (default_folder()).
    revision = model.get("revision")

    out = os.path.join(default_folder(), name)

    # base_ok: the shared model dir is present and already at the pinned revision, per the registry
    # (any engine record for this model). If so, skip the heavy base re-download and only add
    # missing LoRAs. No per-dir .commit -- the registry carries the revision.
    base_ok = (bool(revision)
               and os.path.isfile(os.path.join(out, "model_index.json"))
               and _model_revision(name) == revision)

    loras_present = all(os.path.isfile(os.path.join(out, LORA_DIR, lora["file"]))
                        for lora in loras)

    # Fully installed already -> just ensure the registry entry (this is also the migration path:
    # re-running install over a present model writes engine.json without re-downloading).
    if base_ok and loras_present:
        print("%s already installed at %s" % (name, revision), flush=True)
        _write_engine(_stock_record(mod.ID, name, revision, loras))
        return

    # hf_hub_download is all we need to add a LoRA; torch/diffusers only for a base (re)install.
    from huggingface_hub import scan_cache_dir, hf_hub_download

    repositories = []

    if base_ok:
        print("Base model present; installing missing LoRAs only.", flush=True)
    else:
        # Clean base (re)install: drop any previous copy, then ensure the base dir exists.
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(default_folder(), exist_ok=True)

        # Heavy imports happen here (post argv-parse), so --help stays instant and bad args fail
        # fast.
        import gc
        import torch

        base_repo = model["repository"] + "/" + name

        PipelineCls = _resolve(mod.PIPELINE)

        print("Prefetching model: %s" % name, flush=True)

        # "default" -> None: diffusers loads each weight in its saved dtype (no cast). NOTE:
        # diffusers coerces a non-torch.dtype value (e.g. the string "auto") to float32, so None --
        # not "auto" -- is what keeps the stock dtype here.
        torch_dtype = None if args.dtype == "default" else getattr(torch, args.dtype)

        # `or None` so a blank revision (e.g. an engine that omits the pin) means latest, not ref
        # "".
        pipe = PipelineCls.from_pretrained(
            base_repo,
            revision=revision or None,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )

        pipe.save_pretrained(out, safe_serialization=True)

        # NOTE: drop the pipe before touching the cache (avoids permission-denied on Windows).
        del pipe
        gc.collect()

        repositories.append(base_repo)

    # Fetch each LoRA the engine needs, unless its file is already present (a fresh base install
    # above wiped the dir, so all its LoRAs are missing and get pulled).
    for lora in loras:
        file = lora["file"]

        if os.path.isfile(os.path.join(out, LORA_DIR, file)):
            print("LoRA already present: %s" % file, flush=True)
        else:
            print("Downloading LoRA: %s" % file, flush=True)

            hf_hub_download(repo_id=lora["repository"], filename=file,
                            revision=lora.get("revision") or None,
                            local_dir=os.path.join(out, LORA_DIR))

            repositories.append(lora["repository"])

    # Trim the HF cache for everything we just pulled (the saved copies are self-contained). The
    # cache dir may not exist (nothing cached, or already pruned) -> nothing to trim.
    from huggingface_hub.errors import CacheNotFound

    try:
        cache = scan_cache_dir()

        for repo in cache.repos:
            if repo.repo_id in repositories:
                cache.delete_revisions(*[rev.commit_hash for rev in repo.revisions]).execute()
    except CacheNotFound:
        pass

    # Register this engine (its model + revision + own declared LoRAs) for listing + reference-
    # counted remove. This record carries the revision, so the model dir needs no .commit marker.
    _write_engine(_stock_record(mod.ID, name, revision, loras))

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

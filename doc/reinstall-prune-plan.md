# Prune stale files on engine reinstall

## Context

Reinstalling an engine used to leave stale files on disk forever.

`engine.json` is the only map from an engine to the files it owns — nothing ever scans disk for
unowned files. `_write_engine` replaces that record wholesale, so when a component is dropped from
`COMFY["components"]` (or a LoRA from `LORAS`) between versions, the difference between old and new
is lost at the moment of the write — and that is the only moment anything could have acted on it.

A path becomes unreachable once **no remaining record names it** — which is exactly the condition
under which it should have been deleted. That is the bug: the moment a file becomes garbage is the
same moment it becomes invisible. With a file only one engine owns:

```
comfy-z-image-turbo v1:  [transformer, text_encoder, vae]   <- vae/ae.safetensors
comfy-z-image-turbo v2:  [transformer, text_encoder]
```

After the reinstall nothing named `ae.safetensors`, so `_remove` never considered it and it stayed
on disk forever. (A *shared* file such as `qwen_image_vae.safetensors`, named by four engines, is
not permanently orphaned — the other records still point at it and the last removal would GC it.
Uniquely-owned files are the ones that leaked.)

A second, milder instance: nothing wiped `engine/<id>/` before `snapshot_download`, so narrowing
`SCAFFOLD.allow_patterns` left stale scaffold files. Observed concretely — adding `comfy-flux2-4b`,
then switching it from a scaffold-bundled VAE to ComfyUI's own, left a 161 MB orphaned
`vae/diffusion_pytorch_model.safetensors` that `check` reported as a healthy install.

Reinstall-over is not an edge case: COMFY engines have no "already installed" short-circuit, so
reinstalling is the supported way to repoint at a different ComfyUI or pick up added components.

**Outcome:** after any reinstall, `engine/<id>/` and the model trees we own contain only what the
current installation needs. A user's own ComfyUI install is never touched.

## Key insight

Remove and reinstall-prune are the *same* operation: delete what the OLD record referenced that the
CURRENT registry no longer names. The only difference is *when* you ask:

- `_remove` — entry already `rmtree`'d, so `_registry()` = the survivors.
- reinstall — new record already written, so `_registry()` = the survivors **plus it**.

So `_referenced()` needs no `exclude_id` parameter. Sequencing does the excluding, and one GC
function serves both callers.

## Design

### `_referenced()` / `_gc(record, kept)`

`_referenced()` is the survivor scan lifted out of `_remove` (model names, `(model, lora file)`
pairs, component keys). `_gc` is `_remove`'s former deletion body: it drops the model dir (or, when
the base survives, just the orphaned LoRA files) and the comfy components that `kept` no longer
names. `_remove` shrinks to its bookkeeping plus `_gc(record, _referenced())`.

`_gc` is the **single expression of the deletion policy** — its `_under(path, default_folder())`
guard is the only place that decides what is ours to delete. There is deliberately no second copy
of that rule.

### The hook: inside `_write_engine`

All three install paths already call `_write_engine`, so putting the prune there costs four lines
and a future install path cannot forget it:

```python
old = _read_engine(record["id"])
... write the new record ...
if old is not None:
    _gc(old, _referenced())
```

**Write-then-GC ordering is load-bearing.** Reversed, every ordinary reinstall would GC the files
it just downloaded.

### `_comp_key`

`_comp_path` does no `abspath`, and the old scan only `normcase`d. Two records naming the same file
with different spellings (a relative `--comfy` root vs an absolute one) would hash differently, so
the old record's path would miss `kept` and be deleted **while another engine still needed it**.
`_comp_key` normalizes to absolute + normcased for reference counting only; `_comp_path` is left
alone so engine-side `_by_role()` loaders are unaffected. The `--comfy` root is also made absolute
at install time.

This is the main correctness risk: `qwen_image_vae.safetensors` is shared by 4 engines and
`qwen_3_4b.safetensors` by 2.

### `_prune_scaffold(out, allow_patterns)`

Matches the files *on disk* against the patterns with huggingface_hub's own `filter_repo_objects`
(it accepts any iterable of path strings), so no network call and no hand-rolled glob semantics —
`vae/*` in fnmatch also matches `/`, and getting that subtly wrong is how a pruner deletes
something it shouldn't.

Runs **after** the download rather than wiping before it. Wiping first would delete `engine.json`,
the GC's only record of what the engine owns — turning a mid-install network failure into the
permanent-orphan bug this fixes. This is not hypothetical: an observed run had `snapshot_download`
fail on TLS and silently fall back to "returning existing local_dir", which wipe-before would have
turned into an unrecoverable install.

## What is and isn't pruned

| surface | pruned? | by |
|---|---|---|
| `engine/<id>/` scaffold | yes, always — ours regardless of where ComfyUI lives | `_prune_scaffold` |
| comfy components under `models/` | yes, ref-counted, only under `default_folder()` | `_gc` |
| stock model dirs | yes, whole dir when no record names it | `_gc` |
| stock LoRA files | yes, ref-counted on `(model, file)` | `_gc` |

Deliberately not pruned:

- **Anything outside `default_folder()`** — a user's own ComfyUI. Enforced by the one `_under`
  guard in `_gc`. Note `external` in the record is *not* consulted: it is a stale install-time
  snapshot and honoring it would create a second, drift-prone copy of the rule. It stays written
  for diagnostics only.
- Non-LoRA files inside a stock model dir — the stock path already `rmtree`s before a base
  re-download, and the `base_ok` skip only happens when the pinned revision already matches.
- Other engines' `engine/<id>` dirs; the HF cache and `models/split_files` (already handled).
- **Record-diff only, never a whole-tree sweep.** Only files the previous record named are
  candidates, so models placed in the default ComfyUI by hand are safe. Accepted limit: orphans
  created *before* this fix already lost their registry entry and still need manual cleanup.

## Verification

Logic is covered by a temp-tree harness (the real install cannot exercise component GC, because a
custom ComfyUI is external and the `_under` guard short-circuits it):

| case | expectation |
|---|---|
| engine drops a component another engine shares | file **kept** |
| engine drops a uniquely-owned component | file deleted |
| plain reinstall, nothing changed | nothing deleted |
| component under an external ComfyUI dropped | never deleted |
| same file, root spelled with a `..` segment | still counts as a reference |
| LoRA dropped from one of two engines sharing it | shared kept, orphan deleted |
| scaffold with narrowed `allow_patterns` | orphan deleted, allowed files + engine.json kept |

End-to-end on a real external-ComfyUI install: plant stale files in `engine/<id>`, reinstall, then
assert the ComfyUI tree's file count and the shared text encoder's mtime are unchanged while the
planted files are gone and both engines still report installed.

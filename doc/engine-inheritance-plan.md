# Engine inheritance (`BASE`)

Lets a variant engine inherit from an existing one and declare **only its delta**, instead of
copying the base module wholesale. E.g. `qwen-image-edit-2511-lightning` inherits from
`qwen-image-edit-2511` (adds a LoRA + changes `INFERENCE`), and `..._lightning_angles` inherits from
`..._lightning` (adds a second LoRA + an `extra_key`). Users can derive a custom engine from any
existing one the same way.

## Using it

Add `BASE = "<base engine ID>"` to a variant module and write only what differs; every contract
symbol the variant does **not** define is inherited from the base:

```python
from . import qwen_image_edit_2511 as base   # cheap: base imports no torch at top level

ID   = "qwen-image-edit-2511-lightning"
BASE = base.ID                               # inherit TYPE/PIPELINE/TRANSFORMER/MODES/CFG/MODEL

INFERENCE = 4                                # override
LORAS = [ ... ]                              # add
def loras(params): return [(LIGHTNING, 1.0)] # add
```

Inheritable symbols: `TYPE, PIPELINE, TRANSFORMER, MODES, CFG, INFERENCE, MODEL, COMFY, SCAFFOLD,
LORAS, load, loras, extra_key`. `ID` is always the variant's own.

**Override** = redefine the symbol in the variant. **Inherit** = omit it. **Extend** a collection
= import the base module and compute from it (the base is imported anyway for `BASE = base.ID`):

```python
LORAS = base.LORAS + [angles_entry]                     # extend a list
COMFY = dict(base.COMFY, components=base.COMFY["components"] + [lora_component])  # extend a dict
```

An inherited `load()`/`loras()` keeps the **base module's** globals, so it still resolves the base's
private helpers (`_by_role`, `_build_vae`, …) — the variant inherits behaviour without copying any
helper. This is what lets `comfy-qwen-image-edit-2511-lightning` drop all six meta-builders + `load()`
and add only its 4th `"lora"` COMFY component (the base `load()` applies any `"lora"` role at full
strength).

## How it works

- Both discovery loops (`core.py`, `install.py::_discover`) already import every engine module at
  startup (cheap discovery — no torch at top level). Right after building the `{ID: module}` dict,
  they call `engine/_inherit.py::resolve(engines)`.
- `resolve` folds each `BASE` chain: for a variant, it resolves the base first (recursive,
  memoized, cycle-guarded), then `setattr`s every inheritable symbol the variant lacks from the
  base. **Copied once, at discovery** → after the fold, consumers read plain module attributes with
  **zero per-access overhead**; no wrapper, no `__getattr__`.
- Function hooks are copied as function **objects**, so `__globals__` still points at the base
  module (helpers resolve correctly without being copied).
- An unknown `BASE` id raises `KeyError` and a cycle raises `ValueError` — both at discovery
  (import time for the run side, `_discover()` for install/check), the correct blast radius.

## Notes / safety

- No consumer changes: `generate`, `resolve_model`, `_default_load`, `_engine_key`,
  `_install_comfy`, `_stock_record` all keep reading the same named symbols — now present on the
  variant.
- A stock variant can never accidentally gain `COMFY` (which would reroute it into the comfy
  install/load path): a stock variant extends only a stock base, which has no `COMFY`, so it is
  never in the copy source.
- `check.py` needs no change — it calls `install._discover()`, which folds before returning.

## Files

- `runner/engine/_inherit.py` — the resolver (torch-free; leading `_` so it isn't discovered).
- `runner/core.py`, `runner/install.py` — one `_inherit.resolve(...)` call each after discovery.
- `runner/engine/__init__.py` — the contract comment documents `BASE`.
- Refactored variants (delta only): `qwen_image_edit_2511_lightning.py`,
  `qwen_image_edit_2511_lightning_angles.py`, `comfy_qwen_image_edit_2511_lightning.py`.
- `runner/engine/comfy_qwen_image_edit_2511.py` — base `load()` routes an optional `"lora"`
  component so inheriting variants add it declaratively.

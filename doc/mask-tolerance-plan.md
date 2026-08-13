# Mask engines: tolerance option + merge mask / mask-region

## Context

Two ergonomics changes to the mask engines:

1. **Replace `threshold` with `tolerance`** (0-255) across ALL mask engines. Previously the option
   was a strictness threshold (higher = FEWER pixels/shadow kept: `diff > thr`, `dark - thr`).
   `tolerance` is the **same effect, reversed** -- more tolerance = more pixels/shadow, less =
   fewer -- so the knob reads intuitively. Implemented as `internal_threshold = 255 - tolerance`;
   the low-level ops are unchanged, and the default tolerance is picked to preserve prior behavior
   byte-for-byte.
2. **Merge `mask` + `mask-region` into one `mask` engine**, selecting the sub-mode via a
   `mode=default` (diff mask, the default) / `mode=region` option -- the same `mode=` options
   pattern `mask-apply` already uses (`parse_options(...).get("mode", ...)`).

For the matte engines (birefnet / lucida / inspyrenet), `tolerance` controls the **plate-shadow
floor only** -- the soft subject matte is untouched, and it is inert without a plate.

## Design

- Uniform transform in each engine's `run()`: `thr = 255 - int(opts.get("tolerance", <default>))`;
  the helper functions (`_mask.build_mask`, `_segment.build_matte` / `_plate_shadow`) stay
  threshold-based, so internals are unchanged and default output is byte-exact.
- Defaults preserve prior behavior: diff / region default tolerance **231** (= 255 - 24, the old
  `THR`); shadow default tolerance **243** (= 255 - 12, the old `SHAD_THR`).
- Input-first image convention throughout: `mask` takes `input,reference`; the matte engines take
  `input` or `input,plate`; `mask-apply` takes `input,mask[,reference]`.

## Changes

### `runner/engine/_mask.py`
- Replace `THR, MIN_AREA = 24, 400` with `TOLERANCE = 231` + keep `MIN_AREA = 400` (and the fixed
  `DILATE / FEATHER_MASK / GROW / FEATHER_REGION`). The constant comment documents tolerance 0-255,
  0 = strict/few, 255 = loose/more; the engine passes `255 - tolerance` as the internal threshold.
- `build_mask(reference, edit, mode, thr)` -- drop the `=THR` default (engine always passes `thr`);
  `mode == "region"` -> region, anything else (incl. `"default"`) -> diff.

### `runner/engine/mask.py` (the merged engine)
- `run()`: `opts = parse_options(...)`; `sub = opts.get("mode", "default")` (validate in
  `default|region`); `tol = int(opts.get("tolerance", TOLERANCE))`; `edit = imgs[0]`,
  `ref = imgs[1]`; `mask = build_mask(ref, edit, "region" if sub == "region" else "mask",
  255 - tol)`; emit `mask[<sub> tol=%d]`. Header documents both sub-modes + `tolerance` / `mode`.

### Delete `runner/engine/mask_region.py`.

### `runner/engine/_segment.py`
- Replace `SHAD_THR = 12.0` with `SHADOW_TOLERANCE = 243` (keep `SHAD_NORM`, `SHAD_MAX`); update the
  comment. `_plate_shadow(image, plate, subj, thr)` + `build_matte(image, alpha, plate, thr)`
  unchanged (`thr` = the darkening floor the engine passes as `255 - tolerance`).

### `runner/engine/mask_birefnet.py` + `mask_inspyrenet.py`
- Import `SHADOW_TOLERANCE`; `thr = 255 - int(opts.get("tolerance", SHADOW_TOLERANCE))`;
  `build_matte(..., thr)`. Header: `options tolerance=N` (plate-shadow floor; inert without a plate).
  `mask_lucida.py` inherits birefnet's `run()` -- no change.

### Wrappers + satellites (drop mask-region; threshold -> tolerance)
- `bash/turbo/image-to-mask.sh`: engine list drops `mask-region`; options line ->
  `tolerance=N (0-255, more = more pixels/shadow), mode=default|region`; updated examples.
- `bash/turbo/install.sh` + `bash/turbo/README.md`: drop the `mask-region` line; same options/example
  edits.
- `turboCLI.pro`: remove the `runner/engine/mask_region.py` entry.
- `implementation.md`: layout tree (drop `mask-region`), mask subsection (one `mask` engine,
  `mode=default|region`, tolerance semantics), engine table (merge the mask/mask-region row; options
  `tolerance=N` + `mode=default|region`; matte rows -> `tolerance=N` shadow floor), satellites.
- `test/README.md`: engine list (drop `mask-region`), the removal example ->
  `image-to-mask mask ... mode=region`, thresholds -> `tolerance=`.
- `runner/cli.py` + `runner/install.py` comments (threshold -> tolerance; drop mask-region).
- Historical docs (`doc/image-mask-plan.md`, `image-mask-split-plan.md`) left as records.

## Verification

- **Byte-exact vs a baseline** (map tolerance = 255 - old_threshold): baseline the committed engine
  code via `git show HEAD:...`, generate masks, then the working-tree code, compare with `cmp`.
  - `mask` default (tolerance 231) == baseline `mask`; `tolerance=215` (old thr 40) == baseline
    `mask40`; `mode=region` tolerance 231 == baseline `region`.
- **Merge**: `install mask` (register-only) works; `check-model` no longer lists `mask-region`;
  `image-to-mask mask ... mode=region` and `mode=default` both run.
- **Knob direction**: a tolerance sweep (e.g. 255 / 231 / 128 / 0) shows masked% rising with
  tolerance.
- Static: `awk 'length>99'` empty on all code, `sh -n` wrappers, `python -m compileall runner`.
- Sync the mirror (copy changed runner/engine + wrappers + docs; delete `mask_region.py` there).

## Risks
- The `mode` option key is shared with `mask-apply` (`composite|putalpha`) vs `mask`
  (`default|region`) -- different engines, no clash, but keep docs unambiguous.
- Emit label `thr=` -> `tol=` (informational; nothing parses it).
- Default tolerance differs by engine (231 diff / 243 shadow) to preserve behavior -- documented.

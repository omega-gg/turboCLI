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

# Engine inheritance -- NOT an engine (leading "_" -> skipped by both discovery loops). A variant
# engine may declare `BASE = "<base engine ID>"` and write only its delta; this post-import fold
# copies every contract symbol the variant does NOT itself define from its base. Zero per-access
# cost: symbols are copied ONCE (setattr) right after discovery, so consumers read plain module
# attributes exactly as before. Torch-free (runs during cheap discovery).

# The fixed, known contract symbols a variant may inherit -- ID is excluded (a variant always
# declares its own). Function hooks are copied as function OBJECTS, so an inherited load()/loras()
# keeps the BASE module's __globals__ and still resolves the base's private helpers (_by_role,
# _build_vae, ...) -- so those helpers need not be copied. Private names / imported modules are not
# in this set and are never copied.
_INHERITED = (
    "TYPE", "PIPELINE", "TRANSFORMER", "MODES", "CFG", "INFERENCE",
    "MODEL", "COMFY", "SCAFFOLD", "LORAS",
    "load", "loras", "extra_key",
)


def resolve(engines):
    """Fold each BASE chain in the discovered {ID: module} dict: copy every contract symbol a
    variant does not itself define from its (fully resolved) base. Idempotent, order-independent.
    Raises on an unknown base ID or an inheritance cycle -- both surface loudly at discovery."""
    done = set()

    def visit(engine_id, stack):
        if engine_id in done:
            return

        mod = engines[engine_id]
        base_id = getattr(mod, "BASE", None)

        if base_id is not None:
            if base_id not in engines:
                raise KeyError("engine '%s' BASE unknown engine '%s'" % (engine_id, base_id))
            if base_id in stack:
                raise ValueError("engine inheritance cycle: %s"
                                 % " -> ".join(stack + [engine_id, base_id]))

            visit(base_id, stack + [engine_id])  # fully fold the base first (transitive chains)

            base = engines[base_id]
            for name in _INHERITED:
                if not hasattr(mod, name) and hasattr(base, name):
                    setattr(mod, name, getattr(base, name))

        done.add(engine_id)

    for engine_id in list(engines):
        visit(engine_id, [])

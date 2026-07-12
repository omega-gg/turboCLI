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

# Model install check -- the verify-side mirror of runner.install. It reads the engine registry
# (engine/<id>/engine.json, the source of truth for "installed") and confirms every file that
# record references is present on disk:
#
#   python -m runner.check --engine <name>   # check one engine
#   python -m runner.check                   # list installed engine ids
#
# An engine is "installed" when it has a registry entry AND its referenced files exist: for a stock
# engine the model dir carries the recorded revision (.commit) and every recorded LoRA; for a comfy
# engine the scaffold (model_index.json) + every reused single file exist (see
# install._engine_installed). Anything else reports "not installed" and exits 1, so a bumped
# revision or a removed file triggers a rebuild. Light by design: no torch/diffusers import, so it
# runs under the bundled python without the venv. Run from the deployed dir so `engine` imports.

import sys
import argparse

from runner.install import _discover, _engine_installed


def main():
    parser = argparse.ArgumentParser(prog="runner.check")

    parser.add_argument("--engine", default=None)

    args = parser.parse_args()

    engines = _discover()

    # No engine: print the id of every installed engine (one per line), from the registry.
    if args.engine is None:
        for eid in sorted(engines):
            if _engine_installed(engines[eid]):
                print(eid)

        sys.exit(0)

    mod = engines.get(args.engine)

    if mod is None:
        print("unknown engine '%s'" % args.engine)
        sys.exit(1)

    # Installed = the engine has a registry entry (engine/<id>/engine.json) and every file it
    # references is present (model .commit revision + LoRAs, or comfy scaffold + components).
    if _engine_installed(mod):
        print("%s is installed" % args.engine)
        sys.exit(0)

    print("%s is not installed" % args.engine)
    sys.exit(1)


if __name__ == "__main__":
    main()

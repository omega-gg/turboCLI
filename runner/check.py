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

# Model install check -- the verify-side mirror of runner.install. Given an engine, it resolves the
# model + pinned revision from the engine module (the single source of truth) and confirms the
# install on disk is current:
#
#   python -m runner.check --engine <name> --output <base-dir>
#
# A model is "installed" when "<base-dir>/<model>" exists, its ".commit" marker matches the engine's
# MODEL["revision"], and every LoRA the engine declares is present. Anything else (missing, stale
# revision, missing LoRA) reports "not installed" and exits 1, so a bumped revision triggers a
# rebuild. Light by design: no torch/diffusers import, so it runs under the bundled python without
# the venv. Run from the deployed diffusion dir so `engine` is importable.

import os
import sys
import argparse

from runner.install import _discover, _installed


def main():
    parser = argparse.ArgumentParser(prog="runner.check")

    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)            # base model dir; <output>/<model>

    args = parser.parse_args()

    mod = _discover().get(args.engine)

    if mod is None:
        print("unknown engine '%s'" % args.engine)
        sys.exit(1)

    model = getattr(mod, "MODEL", None)

    if model is None:
        print("%s is not installable" % args.engine)
        sys.exit(1)

    name = model.get("model")
    revision = model.get("revision")

    path = os.path.join(args.output, name)

    # Installed = model folder carries the expected revision (.commit) and every declared LoRA file.
    if _installed(path, revision, getattr(mod, "LORAS", [])):
        print("%s is installed" % args.engine)
        sys.exit(0)

    print("%s is not installed" % args.engine)
    sys.exit(1)


if __name__ == "__main__":
    main()

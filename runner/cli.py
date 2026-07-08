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

# One-shot front-end: build the params dict from argv and run a single generation, then exit.
# Replaces the inlined heredoc in the run*.sh wrappers; shares core.generate with the server.
#
# Every input arrives via argv flags (--prompt=..., --width=..., etc.), so arbitrary text reaches
# Python untouched -- no shell/source interpolation. (The wrappers pass --prompt="$1" with the
# equals form so a prompt starting with '-' is not read as a flag.)
#
# Run as: python -m runner.cli --engine flux2-4b --mode generate --folder ... (from the deployed
# diffusion dir, which the wrapper cd's into so `engine` and `backend` are importable).

import sys
import argparse
import traceback


def main():
    parser = argparse.ArgumentParser(prog="runner.cli")

    parser.add_argument("--engine", required=True)
    parser.add_argument("--mode", default="generate")
    parser.add_argument("--folder", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--images", default="")
    # "<path>@<weight>,..." (weight 0.0-1.0, default 1.0)
    parser.add_argument("--loras", default="")
    parser.add_argument("--width", default="512")
    parser.add_argument("--height", default="512")
    parser.add_argument("--seed", default="-1")
    parser.add_argument("--inference", default="-1")
    parser.add_argument("--renderer", default="cpu")
    parser.add_argument("--offload", default="offloader")
    parser.add_argument("--slicing", default="none")

    args = parser.parse_args()

    params = {
        "engine": args.engine,
        "mode": args.mode,
        "folder": args.folder,
        "prompt": args.prompt,
        "images": args.images,
        "loras": args.loras,
        "output": args.output,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "inference": args.inference,
        "renderer": args.renderer,
        "offload": args.offload,
        "slicing": args.slicing,
    }

    def emit(line):
        print(line, flush=True)

    # NOTE: Importing core here (after argv parsing) runs offload-backend discovery before torch
    #       and keeps `--help` instant. core.generate returns True only when the image was saved.
    from runner import core

    try:
        ok = core.generate(params, emit)
    except Exception:
        emit("ERROR: " + traceback.format_exc())

        sys.exit(1)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

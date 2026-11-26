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

# The Modal side of test/modal.sh, which starts it with `modal run`. It only carries a bash script
# into a container and runs it there: on CPU, or on an L40S GPU (48 GB: every engine fits), with
# 64 GiB of RAM or, for the biggest models, 224 GiB (gpu_big).
#
# - The image is turboCLI as its own bash/python/build.sh and bash/turbo/build.sh build it, in
#   /opt/sky (SKY_PATH_BIN), and ComfyUI as test/comfyui.sh installs it, in /opt/ComfyUI; Modal
#   rebuilds it when those scripts change, and caches it.
# - The models and engine records live on the turbo-models Volume, mounted at /models (a Volume
#   cannot hold the build: uv's cache needs file operations it refuses).
# - The sources (turboCLI and turbo-offloader, as their working trees are) arrive as a tar,
#   unpacked to /src, and the files the script leaves in /src/out come back to the caller.

import io
import os
import subprocess
import sys
import tarfile

import modal

app = modal.App("turbo-test")

test = os.path.dirname(os.path.abspath(__file__))

# Lines a Windows working tree ends in CRLF, which bash refuses.
lf = "find /opt/turboCLI -name '*.sh' -exec sed -i 's/\\r$//' {} +"

image = (modal.Image.from_registry("buildpack-deps:bookworm-scm", add_python="3.12")
         .env({"SKY_PATH_BIN": "/opt/sky"})
         .add_local_dir(os.path.join(test, "..", "bash"), "/opt/turboCLI/bash", copy=True)
         .run_commands(lf, "cd /opt/turboCLI/bash/python && bash build.sh default",
                       "cd /opt/turboCLI/bash/turbo && bash build.sh cuda")
         .add_local_file(os.path.join(test, "comfyui.sh"), "/opt/turboCLI/test/comfyui.sh",
                         copy=True)
         .run_commands(lf, "bash /opt/turboCLI/test/comfyui.sh /opt/ComfyUI")
         # Triton, which both sides' torch carry on Linux, builds its launcher with a C compiler;
         # OpenCV (the matte engines) needs libGL.
         .apt_install("gcc", "libgl1"))

volume = modal.Volume.from_name("turbo-models")

options = dict(image=image, volumes={"/models": volume},
               secrets=[modal.Secret.from_name("hf-token")], timeout=4 * 3600)


def shell(script, sources):
    """Runs the bash script on the unpacked sources, then returns its exit status and /src/out."""
    tarfile.open(fileobj=io.BytesIO(sources)).extractall("/src", filter="data")
    os.makedirs("/src/out", exist_ok=True)

    status = subprocess.run(["bash", "-c", script], cwd="/src").returncode
    volume.commit()

    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        tar.add("/src/out", arcname=".")
    return status, out.getvalue()


# Installs, and the suite on CPU (--renderer cpu): ComfyUI keeps a model in fp32 there when it
# fits, twice its file.
@app.function(cpu=8, memory=49152, **options)
def cpu(script, sources):
    return shell(script, sources)


# The RAM holds the weights read off the Volume (a 9B bf16 model too), so a run streams them from
# memory, not from it.
@app.function(gpu="L40S", memory=65536, **options)
def gpu(script, sources):
    return shell(script, sources)


# The same L40S with the RAM of a model past 64 GB (MiniMax-H3-Image's stock 135 GB), which would
# otherwise stream off the Volume every step. ComfyUI's 49.5 GB runs on the regular one, sampling
# 16% slower: here for timings against the L40S references, recorded on it.
@app.function(gpu="L40S", memory=229376, **options)
def gpu_big(script, sources):
    return shell(script, sources)


# A B200 (180 GB): MiniMax H3's bf16 files resident on the card, as the experimental H3 ran them.
# The card holds the whole model, so turboCLI runs without ComfyUI's pinned host copy of it
# (--disable-pinned-memory: about 100 GiB at most for comfy-minimax-h3-quality, 190 with the copy).
@app.function(gpu="B200", memory=106496, **options)
def gpu_b200(script, sources):
    return shell("export OFFLOADER_DISABLE_PINNED_MEMORY=1\n" + script, sources)


@app.local_entrypoint()
def main(target: str, script: str, sources: str, out: str):
    function = {"gpu": gpu, "gpu_big": gpu_big, "gpu_b200": gpu_b200}.get(target, cpu)

    with open(sources, "rb") as f:
        status, files = function.remote(script, f.read())

    tarfile.open(fileobj=io.BytesIO(files)).extractall(out, filter="data")
    sys.exit(status)

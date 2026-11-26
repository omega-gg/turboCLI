#!/bin/bash
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


set -e

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ "$1" != "install" -a "$1" != "run" -a "$1" != "exec" ]; then

    echo "Usage: modal.sh install [engine ...]"
    echo "       modal.sh run [--big | --video] [run.sh options] [engine ...]"
    echo "       modal.sh exec [--gpu | --big | --b200] <command>"
    echo ""
    echo "Runs the test suite on Modal, on L40S GPUs, with turboCLI's own scripts: the working"
    echo "trees of turboCLI and turbo-offloader (../turbo-offloader, or TURBO_OFFLOADER) go up as"
    echo "they are. Needs the Modal CLI logged in, a 'turbo-models' Volume and an 'hf-token'"
    echo "Secret. The image is turboCLI as build.sh builds it, and ComfyUI in /opt/ComfyUI as"
    echo "test/comfyui.sh installs it, made again when those scripts change."
    echo ""
    echo "install: on CPU, installs each engine (install.sh), one after the other, its models"
    echo "         staying on the Volume"
    echo "run:     test/run.sh with these options, one L40S per engine, all at once (MODAL_JOBS"
    echo "         at a time, 8 by default); no engine: every installed one. The reference, the"
    echo "         summaries, the logs and images come back into test/. With --renderer cpu, on"
    echo "         CPU containers instead (slow: a small --size and a few engines). With --big,"
    echo "         on L40S containers with 224 GiB of RAM instead of 64: minimax-h3-image, and"
    echo "         comfy-minimax-h3-image's timings (its references). With --video, the"
    echo "         video suite (run.sh --video) on B200 containers: MiniMax H3 resident"
    echo "exec:    on CPU (an L40S with --gpu, one with 224 GiB of RAM with --big, a B200 with"
    echo "         --b200), a bash command in the same setup, from the sources' folder; what it"
    echo "         writes to out/"
    echo "         comes back into test/results/modal/exec"
    echo ""
    echo "examples:"
    echo "    modal.sh install comfy-flux2-4b z-image-turbo mask"
    echo "    modal.sh run --list"
    echo "    modal.sh run --record --repeat 2 --comfy /opt/ComfyUI"
    echo "    modal.sh run --renderer cpu --size 256 mask comfy-flux2-4b"
    echo "    modal.sh run --big --size 4096 --comfy /opt/ComfyUI comfy-minimax-h3-image"
    echo "    modal.sh run --video --size 512 --comfy /opt/ComfyUI"
    echo "    modal.sh exec 'bash turboCLI/bash/turbo/check-model.sh'"

    exit 1
fi

command="$1"

shift

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

test="$(cd "$(dirname "$0")" && pwd)"

root="$(dirname "$test")"

offloader="${TURBO_OFFLOADER:-$root/../turbo-offloader}"

# Containers at once: within a GPU limit (10 on a starter Modal account), any over it waiting.
parallel="${MODAL_JOBS:-8}"

stage="$(mktemp -d)"

# An interrupted run stops the containers it started, a finished one its staging folder.
trap 'kill $(jobs -p) 2> /dev/null; exit 1' INT TERM

trap 'rm -rf "$stage"' EXIT

#--------------------------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------------------------

# A local path as the Modal CLI reads it (a Windows one under Git Bash).
native()
{
    if command -v cygpath > /dev/null; then cygpath -w "$1"; else echo "$1"; fi
}

# The working tree of a repository, its files as they are, ignored ones and caches left out.
copy()
{
    mkdir -p "$stage/src/$2"

    git -C "$1" ls-files -co --exclude-standard | grep -v __pycache__ \
        | (cd "$1" && tar -cf - -T -) | tar -xf - -C "$stage/src/$2"
}

# The script to run in the container, after the setup every run shares: the install on the Volume
# (built the first time), turboCLI and the offloader of the working trees over it.
script()
{
    cat << 'EOF'
set -e

# The image's build, whatever a Secret holds.
export SKY_PATH_BIN=/opt/sky

turbo="$SKY_PATH_BIN/gg.omega/turbo"

# A Windows working tree's scripts end their lines in CRLF, which bash refuses here.
find . -name "*.sh" -exec sed -i 's/\r$//' {} +

# The image holds the build, the Volume the models and engine records. Comfy engines record their
# files under /turbo, where an earlier layout mounted it.
for name in model engine; do

    mkdir -p /models/gg.omega/turbo/$name

    rm -rf $turbo/$name

    ln -s /models/gg.omega/turbo/$name $turbo/$name
done

ln -sfn $turbo /turbo

# ComfyUI's models are the comfy engines' files, on the Volume too.
mkdir -p /models/gg.omega/turbo/model/ComfyUI/models

rm -rf /opt/ComfyUI/models

ln -s /models/gg.omega/turbo/model/ComfyUI/models /opt/ComfyUI/models

# The working trees over the build.
cp -r turboCLI/runner turboCLI/bash $turbo/
cp -r turbo-offloader/offloader/. $turbo/backend/offloader/

EOF
    echo "$1"
}

# Runs a script in a container: <cpu|gpu|gpu_big|gpu_b200> <script> <folder its out/ lands in>.
remote()
{
    mkdir -p "$3"

    # NOTE: Git Bash would turn the script's /sky paths into Windows ones: the paths go as
    #       native() makes them, the rest untouched. The CLI prints its marks in UTF-8, which a
    #       Windows console's codepage refuses once piped.
    MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 modal run "$(native "$test/modal_app.py")" \
        --target $1 --script "$(script "$2")" --sources "$(native "$stage/sources.tar")" \
        --out "$(native "$3")"
}

# The script of one engine's run: run.sh, then its reference, summary, logs and images into out/.
suite()
{
    echo "bash turboCLI/test/run.sh $renderer $arguments $1 || status=\$?"

    cat << 'EOF'

mkdir -p out/logs
cp -r turboCLI/test/reference out/
cp turboCLI/test/results/*/summary.txt out/ 2> /dev/null || true
cp turboCLI/test/results/*/*.log* out/logs/ 2> /dev/null || true
cp turboCLI/test/results/*/*.png out/logs/ 2> /dev/null || true
cp turboCLI/test/results/*/*.mp4 out/logs/ 2> /dev/null || true

exit ${status:-0}
EOF
}

#--------------------------------------------------------------------------------------------------
# Run
#--------------------------------------------------------------------------------------------------

if ! command -v modal > /dev/null; then

    echo "the Modal CLI is not installed (pip install modal, then modal setup)" >&2

    exit 1
fi

copy "$root" turboCLI

copy "$offloader" turbo-offloader

tar -cf "$stage/sources.tar" -C "$stage/src" .

if [ "$command" = "install" ]; then

    lines=""

    # One after the other: engines share files, which two downloads must not write at once.
    for engine in "$@"; do

        lines="$lines(cd turboCLI/bash/turbo && bash install.sh $engine cuda)"$'\n'
    done

    remote cpu "$lines" "$stage/out"

    exit
fi

if [ "$command" = "exec" ]; then

    target="cpu"

    if [ "$1" = "--gpu" ]; then

        target="gpu"

        shift

    elif [ "$1" = "--big" ]; then

        target="gpu_big"

        shift

    elif [ "$1" = "--b200" ]; then

        target="gpu_b200"

        shift
    fi

    remote $target "$*" "$test/results/modal/exec"

    exit
fi

# run: run.sh's options apart from the engines, and --big, which is ours; --video, run.sh's,
# also takes the B200.
arguments=""
engines=""
gpu="gpu"

while [ $# -gt 0 ]; do

    case "$1" in
        --big)   gpu="gpu_big";;
        --video) gpu="gpu_b200"; arguments="$arguments --video";;
        --renderer|--size|--repeat|--tolerance|--comfy|--comfy-only|--mode)
            arguments="$arguments $(printf '%q %q' "$1" "$2")"; shift;;
        -*) arguments="$arguments $1";;
        *)  engines="$engines $1";;
    esac

    shift
done

# The L40S (or the GPU asked for), unless the run asks for the CPU, which CPU containers serve
# (no GPU billed).
target="$gpu"
renderer="--renderer cuda"

if [[ "$arguments " == *" --renderer cpu "* ]]; then

    target="cpu"
    renderer=""
fi

# No engine named: every one run.sh would test there (run.sh --list).
if [ -z "$engines" ]; then

    remote cpu "bash turboCLI/test/run.sh --list $arguments > out/engines.txt" "$stage/list" \
        > /dev/null

    engines="$(cat "$stage/list/engines.txt")"
fi

# --list: those engines, and no run.
if [[ "$arguments " == *" --list "* ]]; then

    echo "$engines" | tr ' ' '\n' | grep -v "^$"

    exit 0
fi

# One container per engine, all at once (up to $parallel).
mkdir -p "$stage/out"

for engine in $engines; do

    while [ "$(jobs -rp | wc -l)" -ge "$parallel" ]; do sleep 2; done

    echo "started $engine"

    (
        status=0

        remote $target "$(suite $engine)" "$stage/out/$engine" > "$stage/out/$engine.log" 2>&1 \
            || status=$?

        echo $status > "$stage/out/$engine.status"
    ) &
done

wait

# Each engine's reference lines from its own run, the others' kept as they were.
for file in $(for path in "$stage/out/"*/reference/*.txt; do
                  if [ -f "$path" ]; then basename "$path"; fi
              done | sort -u); do

    header="$(grep -h "^#" "$stage/out/"*/reference/"$file" | head -n 2)"

    {
        if [ -f "$test/reference/$file" ]; then

            awk -v run=" $(echo $engines) " '!/^#/ && index(run, " " $2 " ") == 0' \
                "$test/reference/$file"
        fi

        for engine in $engines; do

            if [ -f "$stage/out/$engine/reference/$file" ]; then

                awk -v e=$engine '!/^#/ && $2 == e' "$stage/out/$engine/reference/$file"
            fi
        done
    } | sort > "$stage/merged"

    { echo "$header"; cat "$stage/merged"; } > "$test/reference/$file"
done

# The logs and images into test/results/modal, the summaries here, a failed container named.
mkdir -p "$test/results/modal"

status=0

echo ""

for engine in $engines; do

    cp "$stage/out/$engine/logs/"* "$test/results/modal/" 2> /dev/null || true

    cat "$stage/out/$engine/summary.txt" 2> /dev/null || true

    if [ "$(cat "$stage/out/$engine.status" 2> /dev/null)" != "0" ]; then

        cp "$stage/out/$engine.log" "$test/results/modal/" 2> /dev/null || true

        echo "$engine: FAILED, see test/results/modal/$engine.log"

        status=1
    fi
done

exit $status

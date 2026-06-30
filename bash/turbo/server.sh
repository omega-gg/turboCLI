#!/bin/sh
set -e
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

#--------------------------------------------------------------------------------------------------
# Settings
#--------------------------------------------------------------------------------------------------

host="127.0.0.1"

port="8080"

scan="0"

range="20"

timeout="600"

#--------------------------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------------------------

getSky()
{
    if [ -z "$SKY_PATH_BIN" ]; then

        echo "SKY_PATH_BIN is not set" >&2

        return
    fi

    case `uname` in
        MINGW*|MSYS*|CYGWIN*)
            cygpath -u "$SKY_PATH_BIN/gg.omega";;
        *)
            echo "$SKY_PATH_BIN/gg.omega";;
    esac
}

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# -lt 1 -o $# -gt 3 ] \
   || \
   [ "$1" != "start" -a "$1" != "stop" -a "$1" != "cancel" -a "$1" != "clear" ] \
   || \
   [ $# = 3 -a "$3" != "scan" ]; then

    echo "Usage: server <action> [port = $port] [scan]"
    echo ""
    echo "actions:"
    echo "    start:  start the server"
    echo "    stop:   stop the server"
    echo "    cancel: stop the current task"
    echo "    clear:  stop the current task and release the loaded model"
    echo ""
    echo "scan: with 'start', bind the first free port in [port, port + $((range - 1))]"
    echo ""
    echo "examples:"
    echo "    server start"
    echo "    server start  9000"
    echo "    server start  9000 scan"
    echo "    server stop   9000"
    echo "    server cancel 9000"
    echo "    server clear  9000"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_TURBOCLI:-$sky/turbo}"

python="${SKY_PATH_PYTHON:-$sky/python}"

action="$1"

if [ "$2" = "scan" ]; then

    scan="1"

elif [ $# -ge 2 ]; then

    port="$2"

    if [ $# -ge 3 ]; then scan="1"; fi
fi

#--------------------------------------------------------------------------------------------------
# Actions
#--------------------------------------------------------------------------------------------------

base="http://$host:$port"

if [ "$action" = "stop" ]; then

    echo "Stopping the server on $base"

    curl -s --max-time 5 -X POST "$base/shutdown" || true

    echo ""

    exit 0
fi

if [ "$action" = "cancel" ]; then

    echo "Cancelling the current task on $base"

    curl -s --max-time 5 -X POST "$base/cancel" || true

    echo ""

    exit 0
fi

if [ "$action" = "clear" ]; then

    echo "Clearing the loaded model on $base"

    curl -s --max-time 30 -X POST "$base/clear" || true

    echo ""

    exit 0
fi

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Use CUDA's stream ordered allocator so large VAE decodes fit and avoid the WDDM RAM spill.
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

case `uname` in
    Darwin*)
        # Fallback on CPU if needed.
        export PYTORCH_ENABLE_MPS_FALLBACK=1

        # Disable the memory cap to avoid allocation failures on large models.
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0;;
esac

cd "$bin"

if [ -f ".venv/Scripts/activate" ]; then

    # Windows / Git Bash
    . ".venv/Scripts/activate"
else
    . ".venv/bin/activate"
fi

#--------------------------------------------------------------------------------------------------
# Server
#--------------------------------------------------------------------------------------------------

if [ "$scan" = "1" ]; then

    scan="--scan"
else
    scan=""
fi

python -m runner.server \
    --host "$host" --port "$port" --range "$range" --timeout "$timeout" $scan

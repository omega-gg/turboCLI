#!/bin/bash
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

getOs()
{
    case `uname` in
    MINGW*|MSYS*|CYGWIN*) os="windows";;
    Darwin*)              os="macOS";;
    Linux*)               os="linux";;
    *)                    os="other";;
    esac

    type=`uname -m`

    if [ $type = "x86_64" ]; then

        if [ $os = "windows" ]; then

            echo win64
        else
            echo $os
        fi

    elif [ $os = "windows" ]; then

        echo win32
    else
        echo $os
    fi
}

getPath()
{
    path="$1"

    if [ "${path#/}" = "$path" ] && [ "${path#?:[\\/]}" = "$path" ]; then

        path="$PWD/$path"
    fi

    if [ "$os" = "windows" ]; then

        # NOTE: Python does not handle backslash.
        cygpath -w "$path" | sed 's|\\|/|g'
    else
        echo "$path"
    fi
}

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

fixes="$4"

temp=$IFS

IFS=","

for fix in $fixes; do

    case "$fix" in
        align|color|transfer|align=*|color=*|transfer=*) ;;
        *)                           fixes="";;
    esac
done

IFS=$temp

if [ $# -lt 4 -o $# -gt 5 ] || [ -z "$fixes" ]; then

    echo "Usage: image-adjust <base image> <reference image> <output image> <fixes> [server]"
    echo ""
    echo "Fix an existing image against its reference, an engine's result against its input."
    echo "Torch-free, no GPU."
    echo ""
    echo "fixes: align        the zoom and shift an engine added are measured and undone,"
    echo "                    unless over 3% (align=<percent> for another limit)"
    echo "       color        the colors are matched to the reference, the detail kept,"
    echo "                    unless they moved over 16 levels (color=<levels> for another)"
    echo "       align,color  both, in that order"
    echo "       transfer     graded toward the reference, which need not show the same"
    echo "                    picture: each L*a*b* channel takes its mean and spread"
    echo "                    (transfer=<strength> from 0 to 1, part of the way)"
    echo ""
    echo "server: host:port (or port for 127.0.0.1) of a rendering server"
    echo ""
    echo "examples:"
    echo "    image-adjust clean.png frame.png fixed.png color"
    echo "    image-adjust edit.png frame.png fixed.png align,color"
    echo "    image-adjust edit.png frame.png fixed.png align,color=24 8080"
    echo "    image-adjust shot.png other-shot.png graded.png transfer=0.5"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_TURBOCLI:-$sky/turbo}"

python="${SKY_PATH_PYTHON:-$sky/python}"

options="$fixes"

if [ $# -ge 5 ]; then server="$5"; fi

host=$(getOs)

if [ $host = "win32" -o $host = "win64" ]; then

    os="windows"
else
    os="default"
fi

images="$(getPath "$1"),$(getPath "$2")"

path=$(getPath "$3")

#--------------------------------------------------------------------------------------------------
# Server
#--------------------------------------------------------------------------------------------------

if [ -n "$server" ]; then

    case "$server" in
        *:*) host="${server%:*}"; port="${server##*:}";;
        *)   host="127.0.0.1";    port="$server";;
    esac

    base="http://$host:$port"

    echo "Using server at $base"

    stream=$(mktemp)

    curl -sS -N --max-time "3600" \
                --data-urlencode "engine=adjust" \
                --data-urlencode "mode=image-adjust" \
                --data-urlencode "images=$images" \
                --data-urlencode "output=$path" \
                --data-urlencode "options=$options" \
                --data-urlencode "renderer=cpu" \
                --data-urlencode "offload=none" \
                "$base/generate" | tee "$stream"

    if grep -q '^Saved: ' "$stream"; then

        rm -f "$stream"

        exit 0
    fi

    echo "Server request failed"

    rm -f "$stream"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

cd "$bin"

if [ -f ".venv/Scripts/activate" ]; then

    # Windows / Git Bash
    . ".venv/Scripts/activate"
else
    . ".venv/bin/activate"
fi

#--------------------------------------------------------------------------------------------------
# Run
#--------------------------------------------------------------------------------------------------

python -m runner.cli \
       --engine "adjust" \
       --mode "image-adjust" \
       --images "$images" \
       --output "$path" \
       --options "$options" \
       --renderer "cpu" \
       --offload none

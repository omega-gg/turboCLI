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

if [ $# != 4 ] \
   || \
   [ "$1" != "mask" -a "$1" != "region" ]; then

    echo "Usage: image-mask <mode> <reference image> <input image> <output image>"
    echo ""
    echo "Merge an edited image back onto its reference: keep the changed region from the input,"
    echo "restore the byte-exact reference everywhere else. No generation, pure image processing"
    echo "(PIL + numpy, no GPU). Run it after an image-to-image edit to undo the whole-frame"
    echo "color/tone drift outside the part you actually changed."
    echo ""
    echo "mode: mask   soft pixel diff, best for adding an object / recoloring"
    echo "      region grown bounding boxes, best for removal / replace (ghost-free)"
    echo ""
    echo "reference: the base canvas (the original scene)"
    echo ""
    echo "input: the edited / generated image"
    echo ""
    echo "The output is at the reference resolution: a full-res reference with a smaller input"
    echo "merges back at full resolution."
    echo ""
    echo "To cut a subject onto a transparent background instead, see image-remove-background."
    echo ""
    echo "examples:"
    echo "    image-mask mask   original.png edited.png output.png"
    echo "    image-mask region original.png edited.png output.png"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_TURBOCLI:-$sky/turbo}"

python="${SKY_PATH_PYTHON:-$sky/python}"

host=$(getOs)

if [ $host = "win32" -o $host = "win64" ]; then

    os="windows"
else
    os="default"
fi

mode="$1"

reference=$(getPath "$2")

input=$(getPath "$3")

output=$(getPath "$4")

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

#--------------------------------------------------------------------------------------------------
# Run
#--------------------------------------------------------------------------------------------------

cd "$bin"

if [ -f ".venv/Scripts/activate" ]; then

    # Windows / Git Bash
    . ".venv/Scripts/activate"
else
    . ".venv/bin/activate"
fi

python -m runner.mask \
       --reference "$reference" \
       --input     "$input" \
       --output    "$output" \
       --mode      "$mode"

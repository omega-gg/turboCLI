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

if [ $# != 5 ] \
   || \
   [ "$1" != "mask" -a "$1" != "region" -a "$1" != "extract" -a "$1" != "extract-full" ] \
   || \
   [ "$2" != "cpu" -a "$2" != "cuda" -a "$2" != "mps" ]; then

    echo "Usage: image-mask <mode> <renderer> <reference image> <input image> <output image>"
    echo ""
    echo "mask / region: merge an edited image back onto its reference -- keep the changed region"
    echo "from the input, restore the exact reference everywhere else. Pure image processing."
    echo ""
    echo "extract / extract-full: cut the subject out of the input onto a transparent background"
    echo "(RGBA PNG, same size/placement). Delegates to the lucida tool (see bash/lucida)."
    echo ""
    echo "mode: mask         soft pixel diff, best for adding an object / recoloring"
    echo "      region       grown bounding boxes, best for removal / replace (ghost-free)"
    echo "      extract      background removal (lucida / BiRefNet), subject only"
    echo "      extract-full extract plus the cast shadow (reference = clean background plate)"
    echo ""
    echo "renderer: cpu, cuda or mps -- used by extract only (cuda/mps fall back to cpu if the"
    echo "          lucida build lacks them); mask / region ignore it (pure CPU processing)"
    echo ""
    echo "reference: mask/region the base canvas; extract-full the clean background plate whose"
    echo "           cast shadow (input darker than the plate) is kept. Unused by extract."
    echo ""
    echo "input: the edited / generated image"
    echo ""
    echo "examples:"
    echo "    image-mask mask         cpu  original.png edited.png output.png"
    echo "    image-mask region       cpu  original.png edited.png output.png"
    echo "    image-mask extract      cuda photo.png    photo.png  cutout.png"
    echo "    image-mask extract-full cuda plate.png    photo.png  cutout.png"

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

renderer="$2"

reference=$(getPath "$3")

input=$(getPath "$4")

output=$(getPath "$5")

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

# NOTE: extract delegates to the lucida tool's own run.sh (its venv + model live under
#       gg.omega/lucida). extract-full passes the reference as the clean plate so the shadow is
#       kept. The raw paths ($3..$5) go through -- run.sh resolves them itself.
if [ "$mode" = "extract" -o "$mode" = "extract-full" ]; then

    run="$(cd "$(dirname "$0")" && pwd)/../lucida/run.sh"

    if [ "$mode" = "extract-full" ]; then
        sh "$run" "$renderer" "$4" "$5" "$3"
    else
        sh "$run" "$renderer" "$4" "$5"
    fi

    exit $?
fi

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

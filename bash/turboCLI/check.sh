#!/bin/sh
set -e
#==================================================================================================
#
#   Copyright (C) 2026-2026 turboCLI authors. <https://omega.gg/Sky>
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

version="3.14.2"

commit="784fa62652fb2719d415830f918fc32a49ecc7a1"

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
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

bin="${SKY_PATH_PYTHON:-$sky/python}"

python="${SKY_PATH_PYTHON:-$sky/python}"

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

#--------------------------------------------------------------------------------------------------
# Python
#--------------------------------------------------------------------------------------------------

if [ ! -d "$bin" ]; then

    echo "Python $version is not installed"

    exit 1
fi

cd "$bin"

current="$(python - << EOF
import sys
print("{}.{}.{}".format(*sys.version_info[:3]))
EOF
)"

if [ "$version" != "$current" ]; then

    echo "Python $version is not installed"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Check
#--------------------------------------------------------------------------------------------------

path="${SKY_PATH_TURBOCLI:-$sky/turboCLI}"

json=$(ls "$path"/.venv/Lib/site-packages/diffusers-*.dist-info/direct_url.json \
          "$path"/.venv/lib/python*/site-packages/diffusers-*.dist-info/direct_url.json \
          2>/dev/null | head -n 1)

if [ ! -f "$json" ]; then

    echo "turboCLI is not installed"

    exit 1
fi

install=$(sed -n 's/.*"commit_id":"\([^"]*\)".*/\1/p' "$json")

if [ "$install" != "$commit" ]; then

    echo "turboCLI is not installed"

    exit 1
fi

echo "turboCLI is installed"

exit 0

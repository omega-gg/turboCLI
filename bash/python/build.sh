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

name="python"

version="3.14.2"

version_uv="0.11.14"

release="20251205"

url="https://github.com/astral-sh/python-build-standalone/releases/download/$release"

url_uv="https://github.com/astral-sh/uv/releases/download/$version_uv"

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

getPath()
{
    if [ $os = "windows" ]; then

        cygpath -w "$1"
    else
        echo "$1"
    fi
}

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# != 1 ] || [ $# = 1 -a "$1" != "default" -a "$1" != "clean" ]; then

    echo "Usage: build <default | clean>"
    echo ""
    echo "example:"
    echo "    build default"

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

sky="$(getSky)"

#--------------------------------------------------------------------------------------------------
# Clean
#--------------------------------------------------------------------------------------------------

if [ "$1" = "clean" ]; then

    echo "CLEANING"

    rm -rf "$sky/$name"

    exit 0
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

case `uname` in
MINGW*|MSYS*|CYGWIN*) os="windows";;
Darwin*)              os="macOS";;
Linux*)               os="linux";;
*)                    os="other";;
esac

if [ $os = "other" ]; then

    echo "build: Unsupported OS."

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Clean
#--------------------------------------------------------------------------------------------------

mkdir -p "$sky"
cd       "$sky"

rm -rf "$name"

#--------------------------------------------------------------------------------------------------
# Download
#--------------------------------------------------------------------------------------------------

mkdir -p "$name"

cd "$name"

if [ $os = "windows" ]; then

    case "$(uname -m)" in

        x86_64|amd64)  arch="x86_64-pc-windows-msvc";;
        i686|x86)      arch="i686-pc-windows-msvc";;
        aarch64|arm64) arch="aarch64-pc-windows-msvc";;
        *)             arch="x86_64-pc-windows-msvc";;
    esac

elif [ $os = "macOS" ]; then

    # NOTE macOS: Detect the hardware architecture, not the caller's. Under Rosetta uname
    #             reports x86_64 but we want the native arm64 build.
    if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then

        arch="aarch64-apple-darwin"
    else
        arch="x86_64-apple-darwin"
    fi
else # [ $os = "linux" ]; then

    case "$(uname -m)" in

        x86_64|amd64)  arch="x86_64-unknown-linux-gnu";;
        aarch64|arm64) arch="aarch64-unknown-linux-gnu";;
        *)             arch="x86_64-unknown-linux-gnu";;
    esac
fi

setup="cpython-$version+$release-$arch-install_only.tar.gz"

url="$url/$setup"

curl --retry 3 -L -o "$setup" "$url"

#--------------------------------------------------------------------------------------------------
# Extract
#--------------------------------------------------------------------------------------------------

tar -xf "$setup" --strip-components=1

if [ $os != "windows" ]; then

    ln -sf python3 bin/python
fi

rm "$setup"

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$sky/$name:$PATH";;
    *)                    export PATH="$sky/$name/bin:$PATH";;
esac

#--------------------------------------------------------------------------------------------------
# uv
#--------------------------------------------------------------------------------------------------

if [ $os = "windows" ]; then

    setup="uv-$arch.zip"
else
    setup="uv-$arch.tar.gz"
fi

curl --retry 3 -L -o "$setup" "$url_uv/$setup"

if [ $os = "windows" ]; then

    unzip -jq "$setup" -d .
else
    tar -xf "$setup" --strip-components=1 -C bin
fi

rm "$setup"

#--------------------------------------------------------------------------------------------------
# Versions
#--------------------------------------------------------------------------------------------------

python --version
uv     --version

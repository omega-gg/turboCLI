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
# Settings
#--------------------------------------------------------------------------------------------------

# The ComfyUI turbo-offloader vendors, and the Krea 2 edit node pack its forward mirrors.
comfyui="v0.45.4"

krea2edit="13ef60f20828949becada815d3cbc371eb5ccaab"

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# != 1 ]; then

    echo "Usage: comfyui.sh <folder>"
    echo ""
    echo "Installs ComfyUI $comfyui into <folder>, for test/run.sh --comfy, on the torch turboCLI"
    echo "builds with (bash/turbo/build.sh), with the Krea 2 edit node pack. It uses the python"
    echo "and uv of turboCLI's build (SKY_PATH_BIN). Linux and macOS."

    exit 1
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

folder="$1"

build="$(cd "$(dirname "$0")" && pwd)/../bash/turbo/build.sh"

export PATH="${SKY_PATH_PYTHON:-$SKY_PATH_BIN/gg.omega/python}/bin:$PATH"

# A pin of turboCLI's build.sh, e.g. torch_version.
pin()
{
    grep -m 1 "^$1=" "$build" | cut -d '"' -f 2
}

#--------------------------------------------------------------------------------------------------
# Install
#--------------------------------------------------------------------------------------------------

git clone --depth 1 --branch $comfyui https://github.com/Comfy-Org/ComfyUI.git "$folder"

git clone https://github.com/lbouaraba/comfyui-krea2edit.git \
    "$folder/custom_nodes/comfyui-krea2edit"

git -C "$folder/custom_nodes/comfyui-krea2edit" checkout -q $krea2edit

uv venv "$folder/venv" --python python

. "$folder/venv/bin/activate"

# turboCLI's torch first, so ComfyUI's requirements keep it.
uv pip install torch==$(pin torch_version) torchvision==$(pin torchvision_version) \
               torchaudio==$(pin torchaudio_version) \
               --index-url https://download.pytorch.org/whl/$(pin torch_cuda)

uv pip install -r "$folder/requirements.txt"

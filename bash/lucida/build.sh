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

# Standalone Lucida (BiRefNet_HR fine-tune) background remover for image-mask extract. Installs its
# own venv + model under gg.omega/lucida, isolated from the turbo venv.

#--------------------------------------------------------------------------------------------------
# Settings
#--------------------------------------------------------------------------------------------------

name="lucida"

# Both BiRefNet models are installed side by side (extract picks one). general is the default.
general="ZhengPeng7/BiRefNet"
general_revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"

lucida="egeorcun/lucida"
lucida_revision="6ee11122534c8de59402a589d2293c198cfbf848"

# NOTE: torch/torchvision/transformers match turbo's build.sh; the segmentation extras are pinned.
torch_version="2.12.1"
torchvision_version="0.27.1"
torch_cuda="cu130"

transformers_version="5.12.1"
huggingface_hub_version="1.21.0"
hf_transfer_version="0.1.9"
safetensors_version="0.8.0"

timm_version="1.0.28"
einops_version="0.8.2"
kornia_version="0.8.3"

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

require()
{
    if [ "$latest" = 1 ]; then

        echo "$1"
    else
        echo "$1==$2"
    fi
}

#--------------------------------------------------------------------------------------------------
# Syntax
#--------------------------------------------------------------------------------------------------

if [ $# -lt 1 -o $# -gt 2 ] \
   || [ "$1" != "cpu" -a "$1" != "cuda" -a "$1" != "mps" ] \
   || [ "$2" != "" -a "$2" != "latest" ]; then

    echo "Usage: build <cpu | cuda | mps> [latest]"
    echo ""
    echo "latest: install the newest releases + models, ignoring the pins (not reproducible)"
    echo ""
    echo "example:"
    echo "    build cuda"
    echo "    build cuda latest"

    exit 1
fi

if [ "$2" = "latest" ]; then

    latest=1

    general_ref="main"
    lucida_ref="main"

    echo "WARNING: building with 'latest' -- ignoring pinned versions, not reproducible."
else
    latest=0

    general_ref="$general_revision"
    lucida_ref="$lucida_revision"
fi

#--------------------------------------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------------------------------------

# NOTE: Absolute script dir, captured before any cd, so extract.py can be copied into place.
source="$(cd "$(dirname "$0")" && pwd)"

sky="$(getSky)"

python="${SKY_PATH_PYTHON:-$sky/python}"

#--------------------------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------------------------

case `uname` in
    MINGW*|MSYS*|CYGWIN*) export PATH="$python:$PATH";;
    *)                    export PATH="$python/bin:$PATH";;
esac

export UV_CACHE_DIR="$sky/cache/uv"

#--------------------------------------------------------------------------------------------------
# Clean
#--------------------------------------------------------------------------------------------------

mkdir -p "$sky"
cd       "$sky"

rm -rf "$name"

mkdir "$name"
cd    "$name"

#--------------------------------------------------------------------------------------------------
# Activate
#--------------------------------------------------------------------------------------------------

# NOTE: A relative python path keeps the venv portable.
case `uname` in
    MINGW*|MSYS*|CYGWIN*) uv venv .venv --relocatable --python "../python/python.exe";;
    *)                    uv venv .venv --relocatable --python "../python/bin/python";;
esac

if [ -f ".venv/Scripts/activate" ]; then

    # Windows / Git Bash
    . ".venv/Scripts/activate"
else
    . ".venv/bin/activate"
fi

#--------------------------------------------------------------------------------------------------
# Install
#--------------------------------------------------------------------------------------------------

if [ "$1" = "cuda" ]; then

    uv pip install \
        "$(require torch $torch_version)" \
        "$(require torchvision $torchvision_version)" \
        --index-url https://download.pytorch.org/whl/$torch_cuda

elif [ "$1" = "mps" ]; then

    uv pip install \
        "$(require torch $torch_version)" \
        "$(require torchvision $torchvision_version)"
else
    uv pip install \
        "$(require torch $torch_version)" \
        "$(require torchvision $torchvision_version)" \
        --index-url https://download.pytorch.org/whl/cpu
fi

uv pip install \
    "$(require transformers $transformers_version)" \
    "$(require huggingface_hub $huggingface_hub_version)" \
    "$(require hf_transfer $hf_transfer_version)" \
    "$(require safetensors $safetensors_version)" \
    "$(require timm $timm_version)" \
    "$(require einops $einops_version)" \
    "$(require kornia $kornia_version)"

#--------------------------------------------------------------------------------------------------
# Runner
#--------------------------------------------------------------------------------------------------

cp "$source/extract.py" "extract.py"

#--------------------------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------------------------

export HF_HOME="$sky/cache/huggingface"

# NOTE: snapshot_download fetches each repo verbatim into model/<name> -- weights + the trusted
#       remote code (birefnet.py, config auto_map already local) -- so runtime loads fully offline.
#       It does not instantiate the model (no torch/timm here), keeping the install robust. Each
#       revision is pinned (mutable HF repo -> reproducible install); 'latest' uses main.
download()
{
    echo "Downloading $1 ($2) -> model/$3 ..."

    python - "$1" "$2" "$3" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir="model/" + sys.argv[3])
EOF
}

download "$general" "$general_ref" "general"
download "$lucida"  "$lucida_ref"  "lucida"

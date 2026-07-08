# turboCLI

[![Discord](https://img.shields.io/discord/705770212485496852)](https://omega.gg/discord)
[![LGPLv3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl.html)

turboCLI is a high performance runner for generative models, with a focus on efficiency,
maintainability and simplicity. It can be used via CLI or embedded in a python application. It
works sequentially or behind a server with fast starting times and inference speeds, cold or warm.
Adding a new model is a few lines of code, in particular when it has a similar structure to what we
already support.

turboCLI is a good baseline for client based applications like [turbopixel](https://omega.gg/turbopixel/sources).
That's the original motivation for it. However, licensing this under LGPL sounds like the right
thing to do.

- [Bash scripts](bash/README.md)

## Contribute

PR(s) are welcomed

## Custom backends

- [turbo-offloader](https://omega.gg/turbo-offloader) - High performance offloader

## Supported engines

- FLUX.2
- Z-Image-Turbo
- Qwen-Image-Edit

## Platforms

- Windows 10 and later
- macOS 64 bit
- Linux 64 bit

## Requirements

macOS and Linux should work out of the box.

On Windows:
- [Git for Windows](https://git-for-windows.github.io)

## Quickstart

### CPU

```
# Install folder
export SKY_PATH_BIN="$PWD/bin"

cd bash/python
sh build.sh default

cd ../turbo
sh build.sh cpu

sh install.sh z-image-turbo bfloat16 fast

sh text-to-image.sh z-image-turbo cpu "a beautiful knight" out.png
```

### CUDA

```
# Install folder
export SKY_PATH_BIN="$PWD/bin"

cd bash/python
sh build.sh default

cd ../turbo
sh build.sh cuda

sh install.sh z-image-turbo bfloat16 fast

sh text-to-image.sh z-image-turbo cuda "a beautiful knight" out.png
```

### Apple MPS

```
# Install folder
export SKY_PATH_BIN="$PWD/bin"

cd bash/python
sh build.sh default

cd ../turbo
sh build.sh mps

# MPS prefers float16 (bfloat16 support is patchy)
sh install.sh z-image-turbo float16 fast

sh text-to-image.sh z-image-turbo mps "a beautiful knight" out.png
```

## Sample model

Here is the z-image recipe:

```
NAME     = "z-image-turbo"
TYPE     = "z-image"
PIPELINE = "diffusers:ZImagePipeline"
MODES    = ("generate",)
CFG      = ("guidance_scale", 0.0)
MODEL    = {"repository": "Tongyi-MAI", "model": "Z-Image-Turbo",
            "revision": "04cc4abb7c5069926f75c9bfde9ef43d49423021"}
```

## This is great, how can I support you ?

Purchase turbopixel, activate auto-renew: https://get.omega.gg/turbopixel

## License

Copyright (C) 2026-2026 turboCLI authors | https://omega.gg/turboCLI

### Authors

- Benjamin Arnaud aka [bunjee](https://bunjee.me) | <bunjee@omega.gg>

### GNU Lesser General Public License Usage

turboCLI may be used under the terms of the GNU Lesser General Public License version 3 as published
by the Free Software Foundation and appearing in the LICENSE.md file included in the packaging of
this file. Please review the following information to ensure the GNU Lesser General Public License
requirements will be met: https://www.gnu.org/licenses/lgpl.html.

### Private License Usage

turboCLI licensees holding valid private licenses may use this file in accordance with the private
license agreement provided with the Software or, alternatively, in accordance with the terms
contained in written agreement between you and turboCLI authors. For further information contact us
at contact@omega.gg.

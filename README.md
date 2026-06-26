# turboCLI

[![Discord](https://img.shields.io/discord/705770212485496852)](https://omega.gg/discord)
[![LGPLv3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl.html)

turboCLI is a high-performance runner for generative models, with a focus on efficiency,
maintainability and simplicity.

It can be used via bash CLI or embedded in a python application. It works sequentially or behind a
server with fast starting times, cold or warm.

Adding a new model should be just a few lines of code, in particular when it has a similar
structure to what we already support.

Its LGPL license makes it a good baseline for client based applications like [turbopixel](https://omega.gg/turbopixel/sources).
That's actually the original motivation for doing so. However, licensing this under LGPL sounds
like the right thing to do.

- [Bash scripts](bash/README.md)

## Supported engines

- FLUX.2
- Z-Image-Turbo
- Qwen-Image-Edit

## Platforms

- Windows 10 and later.
- macOS 64 bit.
- Linux 32 bit and 64 bit.

## Sample model

Here is the z-image recipe:

```
NAME     = "z-image-turbo"
TYPE     = "z-image"
PIPELINE = "diffusers:ZImagePipeline"
MODES    = ("generate",)
CFG      = ("guidance_scale", 0.0)
MODEL    = {"repository": "Tongyi-MAI", "model": "Z-Image-Turbo"}
```

## Quickstart

### CPU

```
export SKY_PATH_BIN="$PWD/bin"
cd bash/python
sh build.sh default
cd ../turboCLI
sh build.sh cpu
cd ../z-image
sh build.sh z-image-turbo bfloat16 fast
sh run.sh "a beautiful knight" out.png 512 512 cpu
```

### CUDA

```
export SKY_PATH_BIN="$PWD/bin"
cd bash/python
sh build.sh default
cd ../turboCLI
sh build.sh cuda
cd ../z-image
sh build.sh z-image-turbo bfloat16 fast
sh run.sh "a beautiful knight" out.png 512 512 cuda
```

### Apple MPS

```
export SKY_PATH_BIN="$PWD/bin"
cd bash/python
sh build.sh default
cd ../turboCLI
sh build.sh mps
cd ../z-image
sh build.sh z-image-turbo float16 fast
sh run.sh "a beautiful knight" out.png 512 512 mps
```

## License

Copyright (C) 2026-2026 turboCLI authors | https://omega.gg/Sky

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

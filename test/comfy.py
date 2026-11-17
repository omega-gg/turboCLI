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

# The ComfyUI side of test/run.sh, run with ComfyUI's own python. Each comfy engine checked
# against ComfyUI has its template's graph here, its model files named by the engine's COMFY
# declaration (runner/engine/<id>.py), so they follow the engine.
#
#   comfy.py cases                                   the (engine, mode) pairs with a graph
#   comfy.py files <engine>                          its model files, relative to models/
#   comfy.py run <engine> <mode> <prompt> <seed> <N|WxH> <input> <output> <url>
#   comfy.py psnr <image> <image>                    their PSNR in dB

import importlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def declaration(engine):
    """The engine module (cheap: no torch at top level)."""
    sys.path.insert(0, ROOT)
    return importlib.import_module("runner.engine." + engine.replace("-", "_"))


def files(engine):
    """{role: path relative to ComfyUI's models/}."""
    return {c["role"]: c["path"] for c in declaration(engine).COMFY["components"]}


# Each graph builder takes (engine, files, prompt, seed, width, height, image) and returns
# {id: (class, inputs)}, its decoded image at node "12". Nodes 1, 2 and 3 load the model, the
# text encoder and the VAE.

def loaders(f, clip_type):
    return {
        "1": ("UNETLoader", {"unet_name": f["transformer"], "weight_dtype": "default"}),
        "2": ("CLIPLoader", {"clip_name": f["text_encoder"], "type": clip_type,
                             "device": "default"}),
        "3": ("VAELoader", {"vae_name": f["vae"]}),
    }


def ksampler(seed, steps, model, positive, negative, latent):
    return ("KSampler", {"seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0, "model": model,
                         "positive": positive, "negative": negative, "latent_image": latent})


def flux2(engine, f, prompt, seed, width, height, image):
    # image_flux2_klein_text_to_image.json
    return dict(loaders(f, "flux2"), **{
        "4": ("CLIPTextEncode", {"text": prompt, "clip": ["2", 0]}),
        "5": ("ConditioningZeroOut", {"conditioning": ["4", 0]}),
        "6": ("CFGGuider", {"cfg": 1.0, "model": ["1", 0], "positive": ["4", 0],
                            "negative": ["5", 0]}),
        "7": ("KSamplerSelect", {"sampler_name": "euler"}),
        "8": ("Flux2Scheduler", {"steps": 4, "width": width, "height": height}),
        "9": ("RandomNoise", {"noise_seed": seed}),
        "10": ("EmptyFlux2LatentImage", {"width": width, "height": height, "batch_size": 1}),
        "11": ("SamplerCustomAdvanced", {"noise": ["9", 0], "guider": ["6", 0],
                                         "sampler": ["7", 0], "sigmas": ["8", 0],
                                         "latent_image": ["10", 0]}),
        "12": ("VAEDecode", {"samples": ["11", 0], "vae": ["3", 0]}),
    })


def flux3(engine, f, prompt, seed, width, height, image):
    # image_flux3_klein_distilled_t2i.json, its subgraph, the prompt upsampler off
    return dict(loaders(f, "flux3"), **{
        "4": ("CLIPTextEncode", {"text": prompt, "clip": ["2", 0]}),
        "5": ("CLIPTextEncode", {"text": "", "clip": ["2", 0]}),
        "6": ("EmptyFlux2LatentImage", {"width": width, "height": height, "batch_size": 1}),
        "7": ksampler(seed, 16, ["1", 0], ["4", 0], ["5", 0], ["6", 0]),
        "12": ("VAEDecode", {"samples": ["7", 0], "vae": ["3", 0]}),
    })


def z_image(engine, f, prompt, seed, width, height, image):
    # image_z_image_turbo.json, euler in place of res_multistep (diffusers' scheduler's sampler)
    return dict(loaders(f, "lumina2"), **{
        "4": ("ModelSamplingAuraFlow", {"model": ["1", 0], "shift": 3.0}),
        "5": ("CLIPTextEncode", {"text": prompt, "clip": ["2", 0]}),
        "6": ("ConditioningZeroOut", {"conditioning": ["5", 0]}),
        "7": ("EmptySD3LatentImage", {"width": width, "height": height, "batch_size": 1}),
        "8": ksampler(seed, 8, ["4", 0], ["5", 0], ["6", 0], ["7", 0]),
        "12": ("VAEDecode", {"samples": ["8", 0], "vae": ["3", 0]}),
    })


def krea2(engine, f, prompt, seed, width, height, image):
    # image_krea2_turbo_t2i.json
    return dict(loaders(f, "krea2"), **{
        "4": ("CLIPTextEncode", {"text": prompt, "clip": ["2", 0]}),
        "5": ("ConditioningZeroOut", {"conditioning": ["4", 0]}),
        "6": ("EmptyLatentImage", {"width": width, "height": height, "batch_size": 1}),
        "7": ksampler(seed, 8, ["1", 0], ["4", 0], ["5", 0], ["6", 0]),
        "12": ("VAEDecode", {"samples": ["7", 0], "vae": ["3", 0]}),
    })


def krea2_edit(engine, f, prompt, seed, width, height, image):
    # comfyui-krea2edit's krea2_identity_edit.json, one reference, the engine's own boost and
    # placement
    module = declaration(engine)
    position = {"beside": "beside (right of target)",
                "aligned": "aligned (trained)"}[module.REF_POSITION]

    def encode(text):
        return ("Krea2EditGroundedEncode", {"clip": ["2", 0], "prompt": text, "image": ["5", 0],
                                            "grounding_px": 768, "system_prompt": ""})

    return dict(loaders(f, "krea2"), **{
        "4": ("LoraLoaderModelOnly", {"lora_name": f["lora"], "strength_model": 1.0,
                                      "model": ["1", 0]}),
        "5": ("LoadImage", {"image": image}),
        "6": ("VAEEncode", {"pixels": ["5", 0], "vae": ["3", 0]}),
        "7": ("EmptySD3LatentImage", {"width": width, "height": height, "batch_size": 1}),
        "8": ("Krea2EditModelPatch", {"model": ["4", 0], "source_latent": ["6", 0],
                                      "ref_boost": module.REF_BOOST, "ref_boost_a": 1.0,
                                      "fit_mode": "fit", "vae": ["3", 0],
                                      "source_image": ["5", 0], "target_latent": ["7", 0],
                                      "ref_position": position,
                                      "ref_position_a": "aligned (trained)"}),
        "9": encode(prompt),
        "10": encode(""),
        "11": ksampler(seed, 8, ["8", 0], ["9", 0], ["10", 0], ["7", 0]),
        "12": ("VAEDecode", {"samples": ["11", 0], "vae": ["3", 0]}),
    })


def qwen21(engine, f, prompt, seed, width, height, image):
    # image_qwen_image_2_1_turbo_t2i.json and _image_edit.json, their subgraphs, the prompt
    # enhancer off: the turbo's own sigmas. A reference goes to the encode, which then hands the
    # latent to sample on.
    encode = {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "",
              "resolution": max(width, height)}
    latent = ["10", 0]
    graph = loaders(f, "qwen_image")
    if image:
        encode.update({"vae": ["3", 0], "images.image_1": ["13", 0]})
        latent = ["4", 2]
        graph["13"] = ("LoadImage", {"image": image})
    else:
        graph["10"] = ("EmptyLatentImage", {"width": width, "height": height, "batch_size": 1})

    return dict(graph, **{
        "4": ("TextEncodeQwenImage21", encode),
        "5": ("QwenImage21Cache", {"model": ["1", 0], "device": "auto", "dtype": "default"}),
        "6": ("CFGGuider", {"cfg": 1.0, "model": ["5", 0], "positive": ["4", 0],
                            "negative": ["4", 1]}),
        "7": ("KSamplerSelect", {"sampler_name": "euler"}),
        "8": ("ManualSigmas", {"sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, "
                                         "0.704534, 0.414568, 0"}),
        "9": ("RandomNoise", {"noise_seed": seed}),
        "11": ("SamplerCustomAdvanced", {"noise": ["9", 0], "guider": ["6", 0],
                                         "sampler": ["7", 0], "sigmas": ["8", 0],
                                         "latent_image": latent}),
        "12": ("VAEDecode", {"samples": ["11", 0], "vae": ["3", 0]}),
    })


GRAPHS = {
    ("comfy-flux2-4b", "text-to-image"): flux2,
    ("comfy-flux2-9b-distilled", "text-to-image"): flux2,
    ("comfy-flux3-6b", "text-to-image"): flux3,
    ("comfy-z-image-turbo", "text-to-image"): z_image,
    ("comfy-krea2-turbo", "text-to-image"): krea2,
    ("comfy-krea2-turbo-edit-identity", "image-to-image"): krea2_edit,
    ("comfy-krea2-turbo-edit-general", "image-to-image"): krea2_edit,
    ("comfy-qwen-image-2-1-turbo", "text-to-image"): qwen21,
    ("comfy-qwen-image-2-1-turbo", "image-to-image"): qwen21,
}


def request(url, data=None, headers={}):
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers))


def upload(url, path):
    """/upload/image: the input, under ComfyUI's input/; returns its name there."""
    boundary = b"turbotest"
    name = b"turbotest_" + os.path.basename(path).encode()
    with open(path, "rb") as f:
        body = (b"--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\n"
                b"Content-Type: image/png\r\n\r\n%s\r\n--%s\r\nContent-Disposition: form-data;"
                b" name=\"overwrite\"\r\n\r\ntrue\r\n--%s--\r\n"
                % (boundary, name, f.read(), boundary, boundary))
    headers = {"Content-Type": "multipart/form-data; boundary=" + boundary.decode()}
    return json.load(request(url + "/upload/image", body, headers))["name"]


def run(engine, mode, prompt, seed, size, input, output, url):
    image = upload(url, input) if input != "-" else None

    names = {role: path.split("/", 1)[1] for role, path in files(engine).items()}
    width, height = (int(v) for v in (size.split("x") * 2)[:2])
    nodes = GRAPHS[(engine, mode)](engine, names, prompt, int(seed), width, height, image)
    nodes["99"] = ("SaveImage", {"filename_prefix": "turbotest/" + engine, "images": ["12", 0]})
    graph = {k: {"class_type": c, "inputs": i} for k, (c, i) in nodes.items()}

    body = json.dumps({"prompt": graph}).encode()
    prompt_id = json.load(request(url + "/prompt", body,
                                  {"Content-Type": "application/json"}))["prompt_id"]
    while True:
        history = json.load(request(url + "/history/" + prompt_id))
        if prompt_id in history:
            break
        time.sleep(0.5)

    entry = history[prompt_id]
    if entry["status"]["status_str"] != "success":
        sys.exit("ComfyUI failed: " + json.dumps(entry["status"])[:1000])

    saved = entry["outputs"]["99"]["images"][0]
    with open(output, "wb") as f:
        f.write(request(url + "/view?" + urllib.parse.urlencode(saved)).read())


def psnr(a, b):
    import numpy as np
    from PIL import Image

    a, b = (np.asarray(Image.open(p).convert("RGB")).astype(float) for p in (a, b))
    if a.shape != b.shape:
        return "size differs"
    mse = ((a - b) ** 2).mean()
    return "identical" if mse == 0 else "%.1f dB" % (10 * np.log10(255 ** 2 / mse))


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    if command == "cases":
        for engine, mode in GRAPHS:
            print(engine, mode)
    elif command == "files":
        print("\n".join(files(args[0]).values()))
    elif command == "run":
        run(*args)
    elif command == "psnr":
        print(psnr(*args))
    else:
        sys.exit("comfy.py: unknown command " + command)

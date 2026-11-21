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
#   comfy.py args <engine>                           the flags ComfyUI starts with for it
#   comfy.py run <engine> <mode> <prompt> <seed> <N|WxH> <input> <output> <url> [duration]
#   comfy.py psnr <file> <file>                      two images' or videos' PSNR in dB

import functools
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
# text encoder and the VAE. A video's builder takes the duration too, and saves its video at
# node "99" itself.

def loaders(f, clip_type):
    return {
        "1": ("UNETLoader", {"unet_name": f["transformer"], "weight_dtype": "default"}),
        "2": ("CLIPLoader", {"clip_name": f["text_encoder"], "type": clip_type,
                             "device": "default"}),
        "3": ("VAELoader", {"vae_name": f["vae"]}),
    }


def ksampler(seed, steps, model, positive, negative, latent, cfg=1.0, sampler="euler",
             scheduler="simple"):
    return ("KSampler", {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
                         "scheduler": scheduler, "denoise": 1.0, "model": model,
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


def esrgan(engine, f, prompt, seed, width, height, image):
    # Upscale Image (using Model), lanczos to the output size
    return {
        "9": ("UpscaleModelLoader", {"model_name": f["upscaler"]}),
        "10": ("LoadImage", {"image": image}),
        "11": ("ImageUpscaleWithModel", {"upscale_model": ["9", 0], "image": ["10", 0]}),
        "12": ("ImageScale", {"image": ["11", 0], "upscale_method": "lanczos", "width": width,
                              "height": height, "crop": "disabled"}),
    }


def seedvr2(engine, f, prompt, seed, width, height, image):
    # utility_seedvr2_{3b,7b}_int8_upscale_image.json at the output size: ImageScale's lanczos in
    # place of the template's scale by multiplier, no color correction, the alpha left out
    tiled = {"tile_size": 512, "overlap": 128, "temporal_size": 4096, "temporal_overlap": 8}
    return {
        "1": ("UNETLoader", {"unet_name": f["transformer"], "weight_dtype": "default"}),
        "3": ("VAELoader", {"vae_name": f["vae"]}),
        "13": ("LoadImage", {"image": image}),
        "14": ("ImageScale", {"image": ["13", 0], "upscale_method": "lanczos", "width": width,
                              "height": height, "crop": "disabled"}),
        "15": ("SeedVR2Preprocess", {"resized_images": ["14", 0]}),
        "16": ("VAEEncodeTiled", dict(tiled, pixels=["15", 0], vae=["3", 0])),
        "17": ("SeedVR2Conditioning", {"model": ["1", 0], "vae_conditioning": ["16", 0]}),
        "8": ksampler(seed, 1, ["1", 0], ["17", 0], ["17", 1], ["16", 0]),
        "11": ("VAEDecodeTiled", dict(tiled, samples=["8", 0], vae=["3", 0])),
        "12": ("SeedVR2PostProcessing", {"images": ["11", 0], "original_resized_images": ["14", 0],
                                         "color_correction_method": "none"}),
    }


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
    # enhancer off: the turbo's own sigmas. A reference goes to the encode; the latent is the
    # output's size, as turbo's (the template's takes the reference's from the encode).
    encode = {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "",
              "resolution": max(width, height)}
    graph = loaders(f, "qwen_image")
    if image:
        encode.update({"vae": ["3", 0], "images.image_1": ["13", 0]})
        graph["13"] = ("LoadImage", {"image": image})

    return dict(graph, **{
        "10": ("EmptyLatentImage", {"width": width, "height": height, "batch_size": 1}),
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
                                         "latent_image": ["10", 0]}),
        "12": ("VAEDecode", {"samples": ["11", 0], "vae": ["3", 0]}),
    })


def moss(engine, f, prompt, seed, width, height, image):
    # image_moss_image_t2i.json and _edit.json, the prompt upsampler off, with the engine's own
    # settings for each (TEMPLATE). The latent is the output's size, as turbo's (the edit
    # template's takes the scaled source's).
    module = declaration(engine)
    mode = "image-to-image" if image else "text-to-image"
    template = module.TEMPLATE[mode]
    start, end = template["skip"]

    graph = loaders(f, "moss_image")
    model = ["1", 0]

    if image:
        graph.update({
            "5": ("LoraLoaderModelOnly", {"lora_name": f["lora"], "strength_model": 1.0,
                                          "model": model}),
            "13": ("LoadImage", {"image": image}),
            "14": ("ImageScaleToTotalPixels", {"image": ["13", 0], "upscale_method": "lanczos",
                                               "megapixels": 1.0, "resolution_steps": 16}),
        })
        model = ["5", 0]

        def encode(text):
            return ("TextEncodeMOSSImageEdit", {"clip": ["2", 0], "prompt": text,
                                                "edit_format": "Moss-Image-Edit",
                                                "vae": ["3", 0], "images.image_0": ["14", 0],
                                                "image_strength": 1.0})
    else:
        def encode(text):
            return ("CLIPTextEncode", {"text": text, "clip": ["2", 0]})

    return dict(graph, **{
        "4": ("SkipLayerGuidanceDiTSimple", {"model": model, "double_layers": "",
                                             "single_layers": "22", "start_percent": start,
                                             "end_percent": end}),
        "6": encode(prompt),
        "7": encode(template["negative"]),
        "8": ("EmptyFlux2LatentImage", {"width": width, "height": height, "batch_size": 1}),
        "9": ksampler(seed, module.INFERENCE[mode], ["4", 0], ["6", 0], ["7", 0], ["8", 0],
                      module.CFG[1][mode], template["sampler"], template["scheduler"]),
        "12": ("VAEDecode", {"samples": ["9", 0], "vae": ["3", 0]}),
    })


def h3_image(engine, f, prompt, seed, width, height, image):
    # image_minimax_h3_image_t2i.json and _edit.json, the prompt expander off. The conditioning
    # gives the latent, at the output's size as turbo's (the edit template's takes the scaled
    # source's); an edit's source goes in at 4 MP on the 32 grid, as <Picture 1>.
    module = declaration(engine)

    graph = loaders(f, "minimax")
    conditioning = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": width,
                    "height": height}

    if image:
        graph.update({
            "13": ("LoadImage", {"image": image}),
            "14": ("ResizeImageMaskNode", {"input": ["13", 0], "scale_method": "area",
                                           "resize_type": "scale total pixels",
                                           "resize_type.megapixels": 4}),
            "15": ("ResizeImageMaskNode", {"input": ["14", 0], "scale_method": "area",
                                           "resize_type": "scale to multiple",
                                           "resize_type.multiple": 32}),
        })
        conditioning["images.image_1"] = ["15", 0]

    return dict(graph, **{
        "6": ("MiniMaxH3ImageConditioning", conditioning),
        "7": ("ConditioningZeroOut", {"conditioning": ["6", 0]}),
        "9": ksampler(seed, module.INFERENCE, ["1", 0], ["6", 0], ["7", 0], ["6", 1],
                      module.CFG[1]),
        "12": ("VAEDecode", {"samples": ["9", 0], "vae": ["3", 0]}),
    })


def h3_video(engine, f, prompt, seed, width, height, image, duration, ref=False):
    # video_minimax_h3_t2v.json and _i2v.json, the Turbo LoRA off (comfy-minimax-h3), or the base
    # graph of the experimental H3, remote/gen.py (comfy-minimax-h3-hq, its keyframe fitted by
    # an ImageScale with center crop first, SaveVideo at its crf): MiniMaxH3ImageToVideo, then
    # SamplerCustomAdvanced (res_multistep, simple, BasicGuider), VAEDecode and VAEDecodeAudio,
    # CreateVideo at 24 fps, SaveVideo. The video at node "99". With `ref`, the image is a picture
    # reference: video_minimax_h3_r2v.json's MiniMaxH3ReferenceToVideo on the ref2va transformer
    # (simple), or gen.py's Ref2VA (beta).
    module = declaration(engine)
    base = declaration(module.BASE) if hasattr(module, "BASE") else module  # frames, INFERENCE
    experiment = engine.endswith("-hq")

    conditioning = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": width,
                    "height": height, "length": base.frames(duration)}
    graph = {
        "1": ("UNETLoader", {"unet_name": f["transformer_ref" if ref else "transformer"],
                             "weight_dtype": "default"}),
        "2": ("CLIPLoader", {"clip_name": f["text_encoder"], "type": "minimax",
                             "device": "default"}),
        "3": ("VAELoader", {"vae_name": f["vae"]}),
        "4": ("VAELoader", {"vae_name": f["audio_vae"]}),
        "5": ("RandomNoise", {"noise_seed": seed}),
        "6": ("MiniMaxH3ImageToVideo", conditioning),
        "7": ("KSamplerSelect", {"sampler_name": "res_multistep"}),
        "8": ("BasicScheduler", {"model": ["1", 0],
                                 "scheduler": "beta" if ref and experiment else "simple",
                                 "steps": base.INFERENCE, "denoise": 1.0}),
        "9": ("BasicGuider", {"model": ["1", 0], "conditioning": ["6", 0]}),
        "10": ("SamplerCustomAdvanced", {"noise": ["5", 0], "guider": ["9", 0],
                                         "sampler": ["7", 0], "sigmas": ["8", 0],
                                         "latent_image": ["6", 1]}),
        "12": ("VAEDecode", {"samples": ["10", 0], "vae": ["3", 0]}),
        "15": ("VAEDecodeAudio", {"samples": ["10", 0], "vae": ["4", 0]}),
        "16": ("CreateVideo", {"images": ["12", 0], "audio": ["15", 0], "fps": 24}),
    }

    if ref:
        graph["13"] = ("LoadImage", {"image": image})
        graph["6"] = ("MiniMaxH3ReferenceToVideo", dict(
            conditioning, audio_vae=["4", 0], ref_image_size="match",
            **{"ref_images.ref_image_0": ["13", 0]}))
    elif image:
        graph["13"] = ("LoadImage", {"image": image})
        conditioning["first_frame"] = ["13", 0]
        if experiment:
            graph["14"] = ("ImageScale", {"image": ["13", 0], "upscale_method": "lanczos",
                                          "width": width, "height": height, "crop": "center"})
            conditioning["first_frame"] = ["14", 0]

    save = {"video": ["16", 0], "filename_prefix": "turbotest/" + engine, "format": "auto",
            "format.codec": "auto"}
    if experiment:
        save.update({"format": "mp4", "format.codec": "h264",
                     "format.codec.encoding": "re-encode",
                     "format.codec.encoding.crf": module.CRF})
    graph["99"] = ("SaveVideo", save)

    return graph


GRAPHS = {
    ("comfy-flux2-4b", "text-to-image"): flux2,
    ("comfy-flux2-9b-distilled", "text-to-image"): flux2,
    ("comfy-flux3-6b", "text-to-image"): flux3,
    ("comfy-z-image-turbo", "text-to-image"): z_image,
    ("comfy-esrgan-upscale", "image-to-image"): esrgan,
    ("comfy-seedvr2-3b-upscale", "image-to-image"): seedvr2,
    ("comfy-seedvr2-7b-upscale", "image-to-image"): seedvr2,
    ("comfy-krea2-turbo", "text-to-image"): krea2,
    ("comfy-krea2-turbo-edit-identity", "image-to-image"): krea2_edit,
    ("comfy-krea2-turbo-edit-general", "image-to-image"): krea2_edit,
    ("comfy-qwen-image-2-1-turbo", "text-to-image"): qwen21,
    ("comfy-qwen-image-2-1-turbo", "image-to-image"): qwen21,
    ("comfy-moss-image", "text-to-image"): moss,
    ("comfy-moss-image", "image-to-image"): moss,
    ("comfy-minimax-h3-image", "text-to-image"): h3_image,
    ("comfy-minimax-h3-image", "image-to-image"): h3_image,
    ("comfy-minimax-h3", "text-to-video"): h3_video,
    ("comfy-minimax-h3", "image-to-video"): h3_video,
    ("comfy-minimax-h3-hq", "text-to-video"): h3_video,
    ("comfy-minimax-h3-hq", "image-to-video"): h3_video,
    ("comfy-minimax-h3", "image-to-video-ref"): functools.partial(h3_video, ref=True),
    ("comfy-minimax-h3-hq", "image-to-video-ref"): functools.partial(h3_video, ref=True),
}

# The flags ComfyUI starts with for an engine's graph: the experiment ran its bf16 text encoder in
# bf16 (ComfyUI casts it to fp16 otherwise).
ARGS = {
    "comfy-minimax-h3-hq": ["--bf16-text-enc"],
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


def run(engine, mode, prompt, seed, size, input, output, url, duration="5"):
    image = upload(url, input) if input != "-" else None

    names = {role: path.split("/", 1)[1] for role, path in files(engine).items()}
    width, height = (int(v) for v in (size.split("x") * 2)[:2])
    video = {"duration": float(duration)} if "-video" in mode else {}
    nodes = GRAPHS[(engine, mode)](engine, names, prompt, int(seed), width, height, image,
                                   **video)
    nodes.setdefault("99", ("SaveImage", {"filename_prefix": "turbotest/" + engine,
                                          "images": ["12", 0]}))
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

    # SaveVideo lists its file under images, as an animated one, or under videos.
    out = entry["outputs"]["99"]
    saved = (out.get("images") or out.get("videos"))[0]
    with open(output, "wb") as f:
        f.write(request(url + "/view?" + urllib.parse.urlencode(saved)).read())


def decode(path):
    """A video's frames (RGB, [T, H, W, 3]) and its sound ([channels, samples], or None)."""
    import av
    import numpy as np

    with av.open(path) as container:
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    with av.open(path) as container:
        sound = (np.concatenate([f.to_ndarray() for f in container.decode(audio=0)], axis=-1)
                 if container.streams.audio else None)
    return frames, sound


def psnr(a, b):
    """The PSNR of two images, or of two videos' frames, their sound compared too (decoded: a
    file's metadata may differ)."""
    import numpy as np

    sound = ""
    if a.endswith(".mp4"):
        (a, sound_a), (b, sound_b) = decode(a), decode(b)
        same = (sound_a is None) == (sound_b is None) and (
            sound_a is None or (sound_a.shape == sound_b.shape and (sound_a == sound_b).all()))
        sound = "" if same else ", sound differs"
    else:
        from PIL import Image

        a, b = (np.asarray(Image.open(p).convert("RGB")) for p in (a, b))
    a, b = a.astype(float), b.astype(float)
    if a.shape != b.shape:
        return "size differs"
    mse = ((a - b) ** 2).mean()
    return ("identical" if mse == 0 else "%.1f dB" % (10 * np.log10(255 ** 2 / mse))) + sound


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    if command == "cases":
        for engine, mode in GRAPHS:
            print(engine, mode)
    elif command == "files":
        print("\n".join(files(args[0]).values()))
    elif command == "args":
        print(" ".join(ARGS.get(args[0], [])))
    elif command == "run":
        run(*args)
    elif command == "psnr":
        print(psnr(*args))
    else:
        sys.exit("comfy.py: unknown command " + command)

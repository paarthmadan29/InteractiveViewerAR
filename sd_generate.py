import os
from PIL import Image
import numpy as np
import torch
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from transformers import pipeline
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()


def txt2img(text_prompt:str):

    depth_estimator = pipeline('depth-estimation')

    image = load_image("data/scene_depth.png")
    print("----------------")
    print(image.size)
    print("----------------")
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    output_images = pipe(
            prompt=text_prompt,
            negative_prompt="people, blur",
            num_images_per_prompt=2,
            guidance_scale=8,
            num_inference_steps=30,
            height=1536,
            width=2048,
            image=image
        ).images

    return output_images


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

prmpt_list = [
                "a photo of a luxurious room interior",
                "a garden"
                "office room"
             ]
counter = 3
for prompt in prmpt_list:
    print(prompt)
    output_images = txt2img(prompt)
    for i, im in enumerate(output_images):
        mkdir(f"avataar/outputs/prompt_{counter}")
        im.save(f"avataar/outputs/prompt_{counter}/im_{i}.png")
    counter += 1
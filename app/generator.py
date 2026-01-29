import base64
import io
import os
from io import BytesIO
from typing import Any
from PIL import Image
import torch
from diffusers.models import QwenImageTransformer2DModel
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image


transformer = QwenImageTransformer2DModel.from_pretrained(
    "linoyts/Qwen-Image-Edit-Rapid-AIO",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    cache_dir=os.environ.get("MODEL_PATH"),
)

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    cache_dir=os.environ.get("MODEL_PATH"),
)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)


def image_from_base64(image: str) -> Image.Image:
    return Image.open(
        io.BytesIO(
            base64.b64decode(
                image.split(",", 1)[1]
            )
        )
    ).convert("RGB")


def process(job: dict[str, Any]) -> dict[str, str]:
    input_ = job["input"]
    images = input_.get("images")
    prompt = input_.get("prompt")
    negative_prompt = input_.get("negative_prompt", None)
    guidance_scale = input_.get("guidance_scale", 1.0)
    steps = input_.get("steps", 4)
    input_images = [image_from_base64(image=image) for image in images]
    inputs = {
        "image": [load_image(input_image) for input_image in input_images],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 1.0,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        output = pipeline(**inputs).frames[0]
        buf = BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)
        image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{image_data}"}

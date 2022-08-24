import gc
import io
import os

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from hivemind.moe.server.layers.custom_experts import register_expert_class
from hivemind.utils.logging import get_logger
from PIL import Image
from PIL.ImageFilter import GaussianBlur
from tsd import run_stable_diffusion

from csd.nsfw import censor_image, load_clip
from csd.utils import clean_gpu, load_from_plasma, push_model_to_plasma

logger = get_logger(__name__)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..")
NSFW_PATH = os.path.join(os.path.dirname(__file__), "..")


MAX_PROMPT_LENGTH = 512
CHANNELS = 3
HEIGHT = WIDTH = 512

WEBP_QUALITY = 60
NSFW_THRESHOLD = 0.9


def get_input_example(batch_size: int, *_unused):
    prompts = torch.empty((1, MAX_PROMPT_LENGTH), dtype=torch.uint8)
    return (prompts,)


def encode_image(image: Image.Image, quality: int = 50):
    byte_array = io.BytesIO()
    image.save(byte_array, format="WEBP", quality=quality)
    return torch.frombuffer(byte_array.getvalue(), dtype=torch.uint8)


def apply_censorship(
    image: Image.Image,
    apply: bool = False,
    blur_radius: int = 20,

):
    return image.filter(GaussianBlur(blur_radius)) if apply else image


@register_expert_class("DiffusionModule", get_input_example)
class DiffusionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.plasma = {}

        # clip = load_clip()
        self.nsfw_filter = torch.jit.load("nsfw_filter.ts")
        logger.info("Loaded nsfw filter")

        diffusion = torch.load("sd_model.pt", map_location="cpu")
        self.plasma["diffusion"] = push_model_to_plasma(diffusion)
        # clean_gpu(diffusion)
        logger.info("Loaded diffusion model")

    def forward(self, prompts: torch.ByteTensor, **kwargs):
        from tsd.utils import seed_everything
        seed = seed_everything(kwargs.get("seed"))
        print(seed)

        diffusion = load_from_plasma(self.plasma["diffusion"])
        decoded_prompts = [bytes(tensor).rstrip(b"\0").decode(errors="ignore") for tensor in prompts]
        output_images = run_stable_diffusion(diffusion, decoded_prompts[0], batch_size=len(prompts))
        # clean_gpu(diffusion)

        preprocess = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])
        nsfw_scores = [self.nsfw_filter(preprocess(image)).tolist()[0] for image in output_images]
        print(nsfw_scores)

        encoded_images = [encode_image(apply_censorship(image, apply=score>0.8)) for image, score in zip(output_images, nsfw_scores)]

        max_buf_len = max(len(buf) for buf in encoded_images)
        stacked_images = torch.stack([F.pad(buf, (0, max_buf_len - len(buf))) for buf in encoded_images])

        logger.info("Inference done")

        return stacked_images

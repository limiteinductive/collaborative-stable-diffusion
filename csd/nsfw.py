from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TypeVar

import open_clip
import tomli
import torch
from PIL import Image

from .utils import get_model_device, performance


def load_clip(
    device="cuda",
) -> open_clip.model.CLIP:
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion400m_e32", precision='fp16', device=torch.device(device)
    )
    model.preprocess = preprocess

    return model


@performance
def encode_image(model: open_clip.model.CLIP, image: Image.Image) -> torch.Tensor:
    device = get_model_device(model)
    image_features = model.encode_image(model.preprocess(image).unsqueeze(0).to(device))

    return image_features


@performance
def encode_text(model: open_clip.model.CLIP, text: str) -> torch.Tensor:
    device = get_model_device(model)
    text_features = model.encode_text(open_clip.tokenize(text).to(device))

    return text_features


@performance
def compute_clip_similarity(
    model: open_clip.model.CLIP, features: torch.Tensor, tokens: List[str]
) -> Dict[str, float]:
    device = get_model_device(model)
    tokens_features = model.encode_text(open_clip.tokenize(tokens).to(device))
    normalize = lambda x: x / x.norm(dim=1, keepdim=True)

    scores = (
        (100.0 * normalize(features) @ normalize(tokens_features).T)
        .softmax(dim=1)
        .tolist()[0]
    )

    return {token: score for token, score in zip(tokens, scores)}


@dataclass
class NSFWScores:
    concept: Dict[str, float]
    special: Dict[str, float]


def compute_nsfw_scores(
    model: open_clip.model.CLIP,
    features: torch.Tensor,
    nsfw_config_path="nsfw.toml",
    adjustment: float = 0.05,
) -> NSFWScores:
    with open(nsfw_config_path, "rb") as f:
        scores = tomli.load(f)["scores"]

    concept_similarities = compute_clip_similarity(
        model, features, list(scores["concept"].keys())
    )
    special_similarities = compute_clip_similarity(
        model, features, list(scores["special"].keys())
    )

    compute_scores = lambda score, similarities: {
        token: round(sim - score[token] + adjustment, 3)
        for token, sim in similarities.items()
    }

    return NSFWScores(
        concept=compute_scores(scores["concept"], concept_similarities),
        special=compute_scores(scores["special"], special_similarities),
    )


def test_nsfw_scores(
    scores: NSFWScores, threshold: float = 0.8, special_threshold: float = 0.6
) -> bool:

    exist_positive = lambda x, t: bool(list(filter(lambda y: (y - t) > 0, x.values())))

    return exist_positive(scores.special, special_threshold) and exist_positive(
        scores.concept, threshold
    )


def censor_image(
    model: open_clip.model.CLIP,
    image: Image.Image,
    nsfw_config_path="nsfw.toml",
    threshold: float = 0.8,
    special_threshold: float = 0.6,
    verbose=False,
) -> bool:
    features = encode_image(model, image)
    scores = compute_nsfw_scores(model, features, nsfw_config_path)
    result = test_nsfw_scores(
        scores, threshold=threshold, special_threshold=special_threshold
    )

    if verbose:
        print(scores)

    return result


def censor_text(
    model: open_clip.model.CLIP,
    text: str,
    nsfw_config_path="nsfw.toml",
    threshold: float = 0.6,
    special_threshold: float = 0.8,
    verbose=False,
) -> bool:
    features = encode_text(model, text)
    scores = compute_nsfw_scores(model, features, nsfw_config_path)
    result = test_nsfw_scores(
        scores, threshold=threshold, special_threshold=special_threshold
    )

    if verbose:
        print(scores)

    return result

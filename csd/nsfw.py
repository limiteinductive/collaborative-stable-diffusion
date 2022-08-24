from dataclasses import dataclass
from functools import partial
from statistics import mean
from typing import Callable, Dict, List, Tuple, TypeVar

import open_clip
import tomli
import torch
import optuna
from PIL import Image

from .utils import get_model_device, lmap, performance


def load_clip(
    device="cuda",
) -> open_clip.model.CLIP:
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        precision="fp16",
        device=torch.device(device),
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
    model: open_clip.model.CLIP,
    features: torch.Tensor,
    tokens_features: List[torch.Tensor],
    tokens: List[str],
) -> Dict[str, float]:

    normalize = lambda x: x / x.norm(dim=1, keepdim=True)
    scores = (normalize(features) @ normalize(tokens_features).T).tolist()[0]

    return {token: score for token, score in zip(tokens, scores)}


@dataclass
class NSFWScores:
    clip_model: str
    threshold: float
    concept: Dict[str, float]
    special: Dict[str, float]


def compute_nsfw_scores(
    model: open_clip.model.CLIP,
    features: torch.Tensor,
    nsfw_config_path="nsfw.toml",
) -> NSFWScores:
    with open(nsfw_config_path, "rb") as f:
        scores = tomli.load(f)["scores"]

    device = get_model_device(model)

    concept_features = model.encode_text(open_clip.tokenize(list(scores["concept"].keys()).to(device)))
    special_features = model.encode_text(open_clip.tokenize(list(scores["special"].keys()).to(device)))

    concept_similarities = compute_clip_similarity(model, features, concept_features)
    special_similarities = compute_clip_similarity(model, features, special_features)

    compute_scores = lambda score, similarities, adjustment: {
        token: round(sim - score[token] + adjustment, 3) for token, sim in similarities.items()
    }

    return NSFWScores(
        concept=compute_scores(scores["concept"], concept_similarities, 0.01),
        special=compute_scores(scores["special"], special_similarities, 0.05),
    )


def test_nsfw_scores(scores: NSFWScores, threshold: float = 0.015) -> bool:

    exist_positive = lambda x, t: bool(list(filter(lambda y: (y - t) > 0, x.values())))

    return exist_positive(scores.special, threshold) or exist_positive(scores.concept, threshold)


def censor_image(
    model: open_clip.model.CLIP,
    image: Image.Image,
    nsfw_config_path="nsfw.toml",
    threshold=0.01,
    verbose=False,
) -> bool:
    features = encode_image(model, image)

    scores = compute_nsfw_scores(model, features, nsfw_config_path)
    result = test_nsfw_scores(scores, threshold=threshold)

    if verbose:
        print(scores)

    return result


def censor_text(
    model: open_clip.model.CLIP,
    text: str,
    nsfw_config_path="nsfw.toml",
    threshold: float = 0.02,
    verbose=False,
) -> bool:
    features = encode_text(model, text)
    scores = compute_nsfw_scores(model, features, nsfw_config_path)
    result = test_nsfw_scores(scores, threshold=threshold)

    if verbose:
        print(scores)

    return result


@performance
def discover_nsfw_scores(
    model: open_clip.model.CLIP,
    tokens: List[str],
    sfw_images: List[Image.Image],
    nsfw_images: List[Image.Image],
    n_trials: int = 100,
):
    device = get_model_device(model)

    tokens_features = model.encode_text(open_clip.tokenize(tokens).to(device))
    sfw_features = lmap(partial(encode_image, model), sfw_images)
    nsfw_features = lmap(partial(encode_image, model), nsfw_images)
    sfw_similarities = [compute_clip_similarity(model, feature, tokens_features, tokens) for feature in sfw_features]
    nsfw_similarities = [compute_clip_similarity(model, feature, tokens_features, tokens) for feature in nsfw_features]


    def objective(trial: optuna.Trial):
        thresholds = {token: trial.suggest_float(token, low=0, high=1) for token in tokens}


        sfw_results = [1 - bool(list(filter(lambda t: similarity[t] > thresholds[t], tokens)))  for similarity in sfw_similarities]
        nsfw_results = [1 - bool(list(filter(lambda t: similarity[t] < thresholds[t], tokens)))  for similarity in nsfw_similarities]

        success_rate = mean(sfw_results + nsfw_results)

        return success_rate

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study

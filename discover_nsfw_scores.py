from csd.nsfw import load_clip, discover_nsfw_scores
from pathlib import Path
from PIL import Image
import tomli

if __name__ == "__main__":
    clip = load_clip()

    # with open("nsfw.toml", "rb") as f:
    #     tokens = [concept for concept in tomli.load(f)["scores"].keys()]

    tokens = ["sex", "nude", "boobs", "explicit", "vagina", "18+", "nsfw", "porn", "xxx", "dick"]

    sfw_images = [Image.open(p) for p in Path("sfw_images").iterdir()]
    nsfw_images = [Image.open(p) for p in Path("nsfw_images").iterdir()]

    study = discover_nsfw_scores(clip, tokens, sfw_images, nsfw_images, n_trials=20)

    print(study.best_trial)

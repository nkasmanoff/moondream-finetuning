"""
Inference helper for the fine-tuned Moondream text LoRA.

Loads the base Moondream 2 model, injects LoRA adapters, loads the
fine-tuned weights, and exposes a simple ``is_race(image)`` function
that returns ``True`` / ``False``.

Usage
-----
    from inference import load_race_detector, is_race

    model = load_race_detector()          # one-time setup
    result = is_race(model, some_pil_img) # True / False
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from safetensors.torch import load_file

from moondream2.moondream import MoondreamModel, MoondreamConfig
from trainer_helpers import inject_lora_into_model
from kart_dataset import SCENE_QUESTION

DEVICE = "cuda" if torch.cuda.is_available() else "mps"

DEFAULT_BASE_MODEL = "moondream2/model.safetensors"
DEFAULT_LORA_PATH = (
    "model_artifacts/"
    "moondream-finetuning_model_artifacts_moondream_lora_text_best_step_215.safetensors"
)


def load_race_detector(
    base_model_path: str = DEFAULT_BASE_MODEL,
    lora_path: str = DEFAULT_LORA_PATH,
    *,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: list[str] | None = None,
    device: str = DEVICE,
) -> MoondreamModel:
    """Build the Moondream model with the fine-tuned LoRA adapter loaded.

    Returns a ready-to-use ``MoondreamModel`` in eval mode.
    """
    if lora_target_modules is None:
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"]

    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    model.load_state_dict(load_file(base_model_path))
    model.to(device)
    for _, buf in model.named_buffers():
        buf.data = buf.data.to(device)

    inject_lora_into_model(
        model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=lora_target_modules,
    )

    lora_state = load_file(lora_path)
    model.load_state_dict(lora_state, strict=False)
    model.eval()
    return model


def is_race(
    model: MoondreamModel,
    image: Union[str, Path, Image.Image],
    *,
    question: str = SCENE_QUESTION,
) -> bool:
    """Return ``True`` if the image depicts an active Mario Kart race.

    Parameters
    ----------
    model : MoondreamModel
        Model returned by :func:`load_race_detector`.
    image : str, Path, or PIL.Image.Image
        A file path or an already-opened PIL image.
    question : str
        Override the default scene prompt if needed.

    Returns
    -------
    bool
        ``True`` when the model answers "yes", ``False`` otherwise.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    with torch.no_grad():
        result = model.query(
            image=image,
            question=question,
            stream=False,
            settings={"max_tokens": 16, "temperature": 0.0},
        )

    answer = result["answer"].strip().lower()
    print(answer)
    return answer.startswith("yes")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [image_path ...]")
        sys.exit(1)

    model = load_race_detector()
    for path in sys.argv[1:]:
        race = is_race(model, path)
        print(f"{path}: {'race' if race else 'not a race'}")

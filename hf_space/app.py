"""
Gradio app for the fine-tuned Moondream Mario Kart race detector.

Hosts the model as both a web UI and a REST API on Hugging Face Spaces.
Compatible with ZeroGPU (the model lives on CPU and is moved to GPU
on-demand via the @spaces.GPU decorator).

API usage (Python):
    from gradio_client import Client
    client = Client("YOUR_SPACE_URL")
    result = client.predict("path/to/image.jpg", api_name="/predict")

API usage (curl):
    curl -X POST YOUR_SPACE_URL/api/predict \
      -H "Content-Type: application/json" \
      -d '{"data": ["data:image/jpeg;base64,<BASE64_IMAGE>"]}'
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from moondream2.moondream import MoondreamModel, MoondreamConfig
from trainer_helpers import inject_lora_into_model
from kart_dataset import SCENE_QUESTION

BASE_MODEL_REPO = os.environ.get("BASE_MODEL_REPO", "moondream/starmie-v1")
BASE_MODEL_FILE = os.environ.get("BASE_MODEL_FILE", "model.safetensors")
LORA_WEIGHTS_PATH = os.environ.get("LORA_WEIGHTS_PATH", "lora_weights/lora.safetensors")

LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "64"))


def load_model() -> MoondreamModel:
    """Download weights (if needed) and build the model on CPU.

    ZeroGPU will move it to a real GPU when @spaces.GPU fires.
    """
    base_path = Path(BASE_MODEL_FILE)
    if not base_path.exists():
        base_path = Path(
            hf_hub_download(repo_id=BASE_MODEL_REPO, filename=BASE_MODEL_FILE)
        )

    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    model.load_state_dict(load_file(str(base_path)))

    inject_lora_into_model(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=0.0,
        target_modules=["qkv", "proj", "fc1", "fc2"],
    )

    lora_state = load_file(LORA_WEIGHTS_PATH)
    model.load_state_dict(lora_state, strict=False)
    model.eval()
    return model


print("Loading model …")
MODEL = load_model()
print("Model ready (CPU). GPU will be allocated per-request by ZeroGPU.")


@spaces.GPU
def predict(image: Image.Image, question: str = SCENE_QUESTION) -> dict:
    """Run inference on a single image.

    The @spaces.GPU decorator ensures a GPU is allocated for this call.
    The model and buffers are moved to cuda, inference runs, and the GPU
    is released automatically when the function returns.
    """
    if image is None:
        return {"answer": "", "is_race": False}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL.to(device)
    for _, buf in MODEL.named_buffers():
        buf.data = buf.data.to(device)

    image = image.convert("RGB")

    with torch.no_grad():
        result = MODEL.query(
            image=image,
            question=question,
            stream=False,
            settings={"max_tokens": 16, "temperature": 0.0},
        )

    answer = result["answer"].strip()
    is_race = answer.lower().startswith("yes")
    return {"answer": answer, "is_race": is_race}


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(
            value=SCENE_QUESTION,
            label="Question",
            info="Override the default prompt if needed.",
        ),
    ],
    outputs=gr.JSON(label="Result"),
    title="Mario Kart Race Detector",
    description=(
        "Upload a screenshot and the fine-tuned Moondream model will tell you "
        "whether it depicts an active Mario Kart race."
    ),
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

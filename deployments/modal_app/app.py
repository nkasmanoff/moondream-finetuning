"""
Modal app for the fine-tuned Moondream Mario Kart race detector.

Deploy:
    modal deploy app.py

Dev (live-reload):
    modal serve app.py

Test locally:
    python app.py image.jpg

Call the endpoint:
    curl -X POST https://YOUR_WORKSPACE--mario-kart-detector-predict.modal.run \
      -H "Content-Type: application/json" \
      -d '{"image": "<BASE64_IMAGE>"}'
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("mario-kart-detector")

SCENE_QUESTION = "Is this an active mario kart race? Respond yes, no, or unsure."
LORA_RANK = 32
LORA_ALPHA = 64
BASE_MODEL_PATH = "/root/app/model.safetensors"
LORA_WEIGHTS_PATH = "/root/app/lora_weights/lora.safetensors"


def download_and_convert_base_model():
    """Image build step: download moondream2 from HF and save as safetensors."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    import os, glob

    local_dir = snapshot_download(
        repo_id="vikhyatk/moondream2",
        revision="2025-06-21",
    )

    safetensor_files = glob.glob(os.path.join(local_dir, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors found in {local_dir}")

    hf_state = {}
    for f in sorted(safetensor_files):
        hf_state.update(load_file(f))

    # HF safetensors keys are prefixed with "model." (e.g. "model.text.blocks.0..."),
    # but MoondreamModel expects keys without that prefix ("text.blocks.0...").
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in hf_state.items()
    }

    os.makedirs(os.path.dirname(BASE_MODEL_PATH), exist_ok=True)

    import sys
    sys.path.insert(0, "/root/app")
    from moondream2.moondream import MoondreamModel, MoondreamConfig

    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded base model — missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print(f"  First few missing keys: {missing[:5]}")
    save_file(model.state_dict(), BASE_MODEL_PATH)
    print(f"Base model saved to {BASE_MODEL_PATH}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7",
        "torchvision>=0.22",
        "einops",
        "transformers==4.51.3",
        "safetensors",
        "huggingface_hub",
        "tokenizers",
        "Pillow",
        "accelerate",
        "fastapi[standard]",
    )
    .add_local_file(
        "trainer_helpers.py",
        "/root/app/trainer_helpers.py",
        copy=True,
    )
    .add_local_file(
        "kart_dataset.py",
        "/root/app/kart_dataset.py",
        copy=True,
    )
    .add_local_dir(
        "moondream2",
        "/root/app/moondream2",
        copy=True,
    )
    .add_local_file(
        "lora_weights/lora.safetensors",
        "/root/app/lora_weights/lora.safetensors",
        copy=True,
    )
    .env({"PYTHONPATH": "/root/app"})
    .run_function(download_and_convert_base_model, gpu="A10G")
)


@app.cls(
    image=image,
    gpu="A10G",
    container_idle_timeout=120,
    max_containers=5,
    allow_concurrent_inputs=15,
    retries=2,
)
class RaceDetector:
    """Runs the fine-tuned Moondream model on an A10G GPU.

    Scales to up to 5 containers, each handling 15 concurrent inputs.
    The model is loaded once when the container starts (@modal.enter)
    and reused across requests until the container idles out.
    """

    @modal.enter()
    def setup(self):
        import torch
        from safetensors.torch import load_file

        from moondream2.moondream import MoondreamConfig, MoondreamModel
        from trainer_helpers import inject_lora_into_model

        self.model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
        self.model.load_state_dict(load_file(BASE_MODEL_PATH))
        self.model.to("cuda")
        for _, buf in self.model.named_buffers():
            buf.data = buf.data.to("cuda")

        inject_lora_into_model(
            self.model,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=0.0,
            target_modules=["qkv", "proj", "fc1", "fc2"],
        )

        lora_state = load_file(LORA_WEIGHTS_PATH)
        self.model.load_state_dict(lora_state, strict=False)
        self.model.eval()
        print("Model loaded on GPU.")

    @modal.fastapi_endpoint(method="POST")
    def predict(self, body: dict):
        """Classify an image.

        Expects JSON: {"image": "<base64>", "question": "optional override"}
        Returns JSON:  {"answer": "yes", "is_race": true}
        """
        import base64
        import io

        import torch
        from PIL import Image

        image_b64 = body.get("image")
        question = body.get("question", SCENE_QUESTION)

        if not image_b64:
            return {"error": "Missing 'image' field (base64-encoded)"}

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with torch.no_grad():
            result = self.model.query(
                image=image,
                question=question,
                stream=False,
                settings={"max_tokens": 16, "temperature": 0.0},
            )

        answer = result["answer"].strip()
        return {"answer": answer, "is_race": answer.lower().startswith("yes")}

    @modal.method()
    def classify(self, image_bytes: bytes, question: str = SCENE_QUESTION) -> dict:
        """Direct Python-to-Python call (no HTTP overhead).

        Usage:
            detector = modal.Cls.from_name("mario-kart-detector", "RaceDetector")()
            result = detector.classify.remote(open("img.jpg","rb").read())
        """
        import io

        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with torch.no_grad():
            result = self.model.query(
                image=image,
                question=question,
                stream=False,
                settings={"max_tokens": 16, "temperature": 0.0},
            )

        answer = result["answer"].strip()
        return {"answer": answer, "is_race": answer.lower().startswith("yes")}


@app.local_entrypoint()
def main(image_path: str = "", question: str = SCENE_QUESTION):
    """Quick test:
        modal run app.py --image-path photo.jpg
    Batch test (same image 10x to exercise concurrency):
        modal run app.py --image-path photo.jpg --copies 10
    """
    if not image_path:
        print("Usage: modal run app.py --image-path <path> [--question <q>] [--copies <n>]")
        return

    data = Path(image_path).read_bytes()
    detector = RaceDetector()

    import sys

    copies = 1
    if "--copies" in sys.argv:
        idx = sys.argv.index("--copies")
        copies = int(sys.argv[idx + 1])

    if copies == 1:
        result = detector.classify.remote(data, question)
        print(result)
    else:
        print(f"Sending {copies} concurrent requests…")
        results = list(detector.classify.map([data] * copies, [question] * copies))
        for i, r in enumerate(results):
            print(f"  [{i}] {r}")

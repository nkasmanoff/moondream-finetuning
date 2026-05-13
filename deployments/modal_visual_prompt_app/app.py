"""
Modal + Gradio demo for the fine-tuned Moondream visual-prompt detector.

Deploy:
    modal deploy app.py

Dev (live-reload, opens a tunnel to your laptop):
    modal serve app.py

The app exposes ONE URL serving both:
- a Gradio web UI with two tabs
    * "Paint to find"  — upload an image, paint over a single instance of the
      object you want to find, and the model uses your painted region as the
      visual prompt to detect every other instance in the same image.
    * "LVIS examples"  — clickable gallery of (query, target) pairs from the
      eval set so visitors can try the model without their own images.
- Gradio's auto-generated REST API at the same hostname (``/api/predict``).

Why this setup is interesting
-----------------------------
Detection in Moondream usually starts from a **text** label::

    [BOS, image patches] [detect prefix] [tokenize(" cat")] [detect suffix]

The fine-tune in this repo replaces the tokenized class name with a
mean-pooled vision embedding of a query image::

    [BOS, image patches] [detect prefix] [* mean_pool(vis_enc(query)) *] [detect suffix]
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          a single splice slot, fed by the
                                          LoRA-trained vision.proj_mlp

So the model learns to detect "things that look like *this* image" instead of
"things that match *this word*". That's what this UI demonstrates.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Source of truth for the Gradio UI lives in `gradio_app.py` so the same
# Blocks can be served locally (`local_app.py`) or wrapped under ASGI here.
# We don't import it at module top because the import only needs to succeed
# *inside* the Modal container — running `modal deploy app.py` from a dev
# box without `gradio` installed shouldn't be an error.


# ============================================================================
# Modal app + image
# ============================================================================

app = modal.App("moondream-visual-prompt-demo")

# We cache the converted base weights in a Modal Volume rather than baking
# them into the image. The build-time alternative (`.run_function(...)` in
# the image) wedged a free CPU sandbox slot for tens of minutes on multiple
# attempts; doing the download once inside a GPU container (with fast
# network) and persisting the result to a Volume bypasses that queue
# entirely. The first cold start pays ~2–3 min for the HF snapshot; every
# subsequent container starts from the already-converted safetensors.
BASE_WEIGHTS_VOLUME = modal.Volume.from_name(
    "moondream-base-weights", create_if_missing=True
)
BASE_WEIGHTS_DIR = "/cache/moondream2"
BASE_MODEL_PATH = f"{BASE_WEIGHTS_DIR}/model.safetensors"

LORA_WEIGHTS_PATH = "/root/app/lora_weights/lora.safetensors"
EXAMPLES_DIR = "/root/app/examples"

# These must match the LoRA shapes the adapter was *trained* with. The
# defaults below match the only checkpoint currently in model_artifacts/
# (moondream_visual_prompt_lora_step_340.safetensors), which was saved with
# the same hparams as scripts/run_visual_prompt_sweep.sh::wide_both.
#
# IMPORTANT: bumping these without re-pointing lora_weights/lora.safetensors
# at a matching checkpoint will silently produce garbage detections, because
# `load_state_dict(..., strict=False)` drops every rank-mismatched LoRA
# tensor without complaining. `inference.py` cross-checks shapes at load
# time so a real mismatch raises instead of serving an untrained model.
TEXT_LORA_RANK = 64
TEXT_LORA_ALPHA = 128
PROJ_MLP_LORA_RANK = 32
PROJ_MLP_LORA_ALPHA = 64


def _ensure_base_model_on_volume():
    """Idempotent: pull moondream2 from HF and re-save it in the key layout
    ``MoondreamModel`` expects (no ``model.`` prefix), onto the mounted Volume.

    Called from ``@modal.enter()`` rather than at build time — the build-time
    variant repeatedly stalled in Modal's CPU-build queue. Running this in a
    GPU container with the Volume mounted gives us fast network *and* a
    persistent cache, so we pay the ~2–3 min conversion exactly once across
    every cold start of every container.
    """
    import glob
    import os
    import sys

    if os.path.exists(BASE_MODEL_PATH):
        print(f"Base model already on volume at {BASE_MODEL_PATH}, skipping download.")
        return

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    print("Base model not on volume — downloading from HF…")
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

    # HF ships keys as "model.text.blocks…", MoondreamModel wants "text.blocks…".
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in hf_state.items()
    }

    os.makedirs(os.path.dirname(BASE_MODEL_PATH), exist_ok=True)

    sys.path.insert(0, "/root/app")
    from moondream2.moondream import MoondreamConfig, MoondreamModel

    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded base model — missing: {len(missing)}, unexpected: {len(unexpected)}")
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
        "numpy",
        "gradio==5.50.0",
        "fastapi[standard]",
    )
    .add_local_file(
        "trainer_helpers.py",
        "/root/app/trainer_helpers.py",
        copy=True,
    )
    .add_local_file(
        "inference.py",
        "/root/app/inference.py",
        copy=True,
    )
    .add_local_file(
        "gradio_app.py",
        "/root/app/gradio_app.py",
        copy=True,
    )
    .add_local_dir(
        "moondream2",
        "/root/app/moondream2",
        copy=True,
        # Skip the local 4 GB base weights — `_ensure_base_model_on_volume`
        # pulls them from HF at container start and persists them on the
        # mounted Modal Volume, so uploading them from the dev box would
        # just bloat every image rebuild for nothing.
        ignore=["model.safetensors", "__pycache__", "*.pyc", "wandb"],
    )
    .add_local_file(
        "lora_weights/lora.safetensors",
        LORA_WEIGHTS_PATH,
        copy=True,
    )
    .add_local_dir(
        "examples",
        EXAMPLES_DIR,
        copy=True,
        ignore=["*.md", ".gitkeep"],
    )
    .env(
        {
            "PYTHONPATH": "/root/app",
            "BASE_MODEL_PATH": BASE_MODEL_PATH,
            "LORA_WEIGHTS_PATH": LORA_WEIGHTS_PATH,
            "TEXT_LORA_RANK": str(TEXT_LORA_RANK),
            "TEXT_LORA_ALPHA": str(TEXT_LORA_ALPHA),
            "PROJ_MLP_LORA_RANK": str(PROJ_MLP_LORA_RANK),
            "PROJ_MLP_LORA_ALPHA": str(PROJ_MLP_LORA_ALPHA),
        }
    )
)


# ============================================================================
# Modal class — model loaded once per container, served via Gradio over ASGI
# ============================================================================


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=180,
    max_containers=2,
    retries=1,
    volumes={BASE_WEIGHTS_DIR: BASE_WEIGHTS_VOLUME},
    timeout=900,
)
@modal.concurrent(max_inputs=8)
class VisualPromptDemo:
    """Hosts the fine-tuned visual-prompt detector behind a Gradio UI.

    Inference path:
        target image → Moondream encode_image (frozen ViT)
        query crop   → ViT → vision.proj_mlp (LoRA) → mean-pool → (1,1,D) splice
        [prefix] [splice] [suffix] → text decoder (LoRA) → region head → boxes
    """

    @modal.enter()
    def setup(self):
        # First container to ever start fills the Volume; later containers
        # skip straight to load. Either way `inference.load_model_from_env`
        # then reads from BASE_MODEL_PATH on the mounted Volume.
        _ensure_base_model_on_volume()
        BASE_WEIGHTS_VOLUME.commit()

        from inference import detect_with_visual_prompt, load_model_from_env

        self.model = load_model_from_env()
        self._detect = detect_with_visual_prompt
        print("Visual-prompt model loaded on GPU.")

    # ---- Programmatic entrypoint (handy for `modal run app.py`) -------------

    @modal.method()
    def detect(
        self,
        target_bytes: bytes,
        query_bytes: bytes,
        max_objects: int = 25,
    ) -> list[dict]:
        """Direct Python-to-Python call. Returns boxes in normalized [0, 1]."""
        import io
        from PIL import Image

        target = Image.open(io.BytesIO(target_bytes)).convert("RGB")
        query = Image.open(io.BytesIO(query_bytes)).convert("RGB")
        return self._detect(self.model, target, query, max_objects=max_objects)

    # ---- Gradio UI mounted as ASGI endpoint --------------------------------

    @modal.asgi_app()
    def ui(self):
        from fastapi import FastAPI
        from gradio.routes import mount_gradio_app

        from gradio_app import build_gradio_app

        # Building the Blocks here (rather than at module top level) lets the
        # closures capture ``self.model`` so we never reload weights per request.
        blocks = build_gradio_app(
            model=self.model,
            detect_fn=self._detect,
            examples_dir=EXAMPLES_DIR,
        )

        web_app = FastAPI(title="Moondream Visual-Prompt Demo")
        return mount_gradio_app(app=web_app, blocks=blocks, path="/")


# ============================================================================
# Local entrypoint (smoke test from the CLI)
# ============================================================================


@app.local_entrypoint()
def main(target: str = "", query: str = "", max_objects: int = 25):
    """Quick programmatic test:

        modal run app.py --target target.jpg --query query.jpg
    """
    if not target or not query:
        print("Usage: modal run app.py --target <target.jpg> --query <query.jpg>")
        return

    detector = VisualPromptDemo()
    boxes = detector.detect.remote(
        Path(target).read_bytes(),
        Path(query).read_bytes(),
        max_objects=max_objects,
    )
    print(f"Detected {len(boxes)} object(s):")
    for i, b in enumerate(boxes):
        print(f"  [{i}] {b}")

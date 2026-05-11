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


# ============================================================================
# Modal app + image
# ============================================================================

app = modal.App("moondream-visual-prompt-demo")

BASE_MODEL_PATH = "/root/app/moondream2/model.safetensors"
LORA_WEIGHTS_PATH = "/root/app/lora_weights/lora.safetensors"
EXAMPLES_DIR = "/root/app/examples"

# These must match the LoRA shapes the adapter was *trained* with. Defaults
# below match scripts/run_visual_prompt_sweep.sh::wide_proj_mlp.
TEXT_LORA_RANK = 32
TEXT_LORA_ALPHA = 64
PROJ_MLP_LORA_RANK = 64
PROJ_MLP_LORA_ALPHA = 128


def _download_and_convert_base_model():
    """Image build step: pull moondream2 from HF and re-save it in the key
    layout ``MoondreamModel`` expects (no ``model.`` prefix).

    Same approach as ``deployments/modal_app/app.py`` but writes to a path
    inside ``/root/app/moondream2/`` so the trainer-style relative paths
    (``moondream2/model.safetensors``) keep working at run time.
    """
    import glob
    import os
    import sys

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

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
    .add_local_dir(
        "moondream2",
        "/root/app/moondream2",
        copy=True,
        # Skip the local 4 GB base weights — `_download_and_convert_base_model`
        # below pulls them from HF inside the image build, so uploading them
        # from the dev box just doubles the first-deploy time for nothing.
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
    # Build step is pure CPU work (HF snapshot download + safetensors key
    # rename + save). Asking for a GPU here just means we wait in the A10G
    # queue while doing nothing GPU-y; build this layer on CPU instead.
    .run_function(_download_and_convert_base_model, cpu=4.0, memory=16384)
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

        # Building the Blocks here (rather than at module top level) lets the
        # closures capture ``self.model`` so we never reload weights per request.
        blocks = _build_gradio_app(
            model=self.model,
            detect_fn=self._detect,
            examples_dir=EXAMPLES_DIR,
        )

        web_app = FastAPI(title="Moondream Visual-Prompt Demo")
        return mount_gradio_app(app=web_app, blocks=blocks, path="/")


# ============================================================================
# Gradio Blocks
# ============================================================================


def _build_gradio_app(model, detect_fn, examples_dir: str):
    """Construct the two-tab Gradio Blocks.

    Kept as a free function so it can also be imported by a local entrypoint
    or unit test without touching Modal.
    """
    import gradio as gr
    import numpy as np
    from PIL import Image

    HEADER = """
    # Visual-prompt detection — Moondream 2 fine-tune

    This is *image-as-prompt* detection: instead of asking the model in words
    ("find all the dogs"), you give it a **picture** of one example, and it
    finds every other instance of that thing in your target image.

    The fine-tune wires a single mean-pooled vision embedding of the query
    image into the same prompt slot the tokenizer would normally fill with a
    class name.
    """

    # --- helpers -----------------------------------------------------------

    def _bbox_from_painted_layer(layer_rgba: Image.Image) -> tuple[int, int, int, int] | None:
        """Find the tight bbox of nonzero alpha in a Gradio brush layer.

        Returns ``(x_min, y_min, x_max, y_max)`` in pixel coords or ``None``
        when the user didn't paint anything.
        """
        if layer_rgba.mode != "RGBA":
            layer_rgba = layer_rgba.convert("RGBA")
        alpha = np.array(layer_rgba.split()[-1])
        if alpha.max() == 0:
            return None
        ys, xs = np.where(alpha > 0)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _norm_to_pixel_box(box: dict, w: int, h: int) -> tuple[int, int, int, int]:
        return (
            max(0, int(round(box["x_min"] * w))),
            max(0, int(round(box["y_min"] * h))),
            min(w, int(round(box["x_max"] * w))),
            min(h, int(round(box["y_max"] * h))),
        )

    # --- Tab 1: paint-to-find ---------------------------------------------

    def paint_to_find(editor_value, max_objects):
        if editor_value is None:
            raise gr.Error("Upload an image first.")

        background = editor_value.get("background")
        layers = editor_value.get("layers") or []
        if background is None:
            raise gr.Error("No image found in the editor.")
        if not layers:
            raise gr.Error(
                "Paint over one example of the object you want to find, "
                "then click Detect."
            )

        background = background.convert("RGB")
        bbox = _bbox_from_painted_layer(layers[0])
        if bbox is None:
            raise gr.Error(
                "I couldn't find any painted pixels — try painting more clearly "
                "over the example object."
            )

        query_crop = background.crop(bbox)
        preds = detect_fn(model, background, query_crop, max_objects=int(max_objects))

        W, H = background.size
        annotations = [
            (_norm_to_pixel_box(p, W, H), f"match {i + 1}")
            for i, p in enumerate(preds)
        ]
        status = (
            f"Found **{len(preds)}** instance(s). "
            f"Query crop was {bbox[2] - bbox[0]} x {bbox[3] - bbox[1]} px."
        )
        return (background, annotations), query_crop, status

    # --- Tab 2: examples gallery ------------------------------------------

    def detect_pair(query_image, target_image, max_objects):
        if query_image is None or target_image is None:
            raise gr.Error("Pick a query image and a target image.")
        target_image = target_image.convert("RGB")
        query_image = query_image.convert("RGB")
        preds = detect_fn(model, target_image, query_image, max_objects=int(max_objects))
        W, H = target_image.size
        annotations = [
            (_norm_to_pixel_box(p, W, H), f"match {i + 1}")
            for i, p in enumerate(preds)
        ]
        return (target_image, annotations), f"Found **{len(preds)}** instance(s)."

    examples = _discover_examples(examples_dir)

    # --- assemble ----------------------------------------------------------

    with gr.Blocks(
        title="Moondream Visual-Prompt Demo",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet"),
    ) as demo:
        gr.Markdown(HEADER)

        with gr.Tabs():
            # ---------- Tab 1 ----------
            with gr.Tab("Paint to find"):
                gr.Markdown(
                    "Upload an image, paint over **one** example object with "
                    "the brush, then hit **Detect**. The painted region is "
                    "auto-cropped and used as the visual prompt — the model "
                    "will draw boxes around every other matching instance "
                    "in the same image."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        editor = gr.ImageEditor(
                            type="pil",
                            label="Target image — paint one example object",
                            sources=["upload", "clipboard"],
                            brush=gr.Brush(
                                default_size=28,
                                colors=["#7c3aed"],
                                default_color="#7c3aed",
                                color_mode="fixed",
                            ),
                            eraser=gr.Eraser(),
                            transforms=(),
                            layers=False,
                            crop_size=None,
                            height=520,
                        )
                        max_obj_paint = gr.Slider(
                            1, 50, value=25, step=1,
                            label="Max objects to detect",
                        )
                        detect_btn = gr.Button("Detect", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        annotated_paint = gr.AnnotatedImage(
                            label="Detections",
                            color_map={f"match {i}": "#7c3aed" for i in range(1, 51)},
                            height=520,
                        )
                        with gr.Row():
                            query_preview = gr.Image(
                                type="pil",
                                label="Auto-extracted query crop",
                                height=180,
                                interactive=False,
                            )
                            status_paint = gr.Markdown()

                detect_btn.click(
                    fn=paint_to_find,
                    inputs=[editor, max_obj_paint],
                    outputs=[annotated_paint, query_preview, status_paint],
                    api_name="paint_to_find",
                )

            # ---------- Tab 2 ----------
            with gr.Tab("LVIS examples"):
                gr.Markdown(
                    "These are (query crop, target image) pairs sampled from "
                    "the LVIS validation split. Click an example to load it, "
                    "then hit **Detect** to run the model."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gallery_query = gr.Image(
                            type="pil",
                            label="Query crop",
                            height=240,
                        )
                    with gr.Column(scale=2):
                        gallery_target = gr.Image(
                            type="pil",
                            label="Target image",
                            height=420,
                        )
                with gr.Row():
                    max_obj_pair = gr.Slider(
                        1, 50, value=25, step=1,
                        label="Max objects to detect",
                    )
                    detect_pair_btn = gr.Button("Detect", variant="primary", size="lg")

                annotated_pair = gr.AnnotatedImage(
                    label="Detections on target",
                    color_map={f"match {i}": "#7c3aed" for i in range(1, 51)},
                    height=520,
                )
                status_pair = gr.Markdown()

                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[gallery_query, gallery_target],
                        label="Click an example to load it",
                        examples_per_page=8,
                    )
                else:
                    gr.Markdown(
                        "_No examples bundled — run `python prepare_examples.py` "
                        "before deploying to populate this tab._"
                    )

                detect_pair_btn.click(
                    fn=detect_pair,
                    inputs=[gallery_query, gallery_target, max_obj_pair],
                    outputs=[annotated_pair, status_pair],
                    api_name="detect_pair",
                )

        gr.Markdown(
            "---\n"
            "**Tips:** the model was trained on LVIS objects — it generalizes "
            "best when the query crop is a *clean, well-cropped* example. "
            "Heavy occlusion or unusual viewpoints in the query lower recall."
        )

    return demo


def _discover_examples(examples_dir: str) -> list[list[str]]:
    """Find ``examples/<id>/{query.png,target.png}`` pairs.

    Tolerates a missing or empty examples dir (returns ``[]``); ``app.py``
    still works, the Examples block just won't render.
    """
    base = Path(examples_dir)
    if not base.exists():
        return []
    pairs: list[list[str]] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        q = sub / "query.png"
        t = sub / "target.png"
        if q.exists() and t.exists():
            pairs.append([str(q), str(t)])
    return pairs


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

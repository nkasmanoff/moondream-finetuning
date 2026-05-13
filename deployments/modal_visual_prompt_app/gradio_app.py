"""
Gradio Blocks for the Moondream visual-prompt detector.

This module is intentionally **Modal-free** so it can be reused by:

- ``app.py``       — the Modal deployment (mounts these Blocks under ASGI)
- ``local_app.py`` — a plain ``demo.launch()`` for local dev / HF Spaces

It exposes two callables:

- ``build_gradio_app(model, detect_fn, examples_dir)``: constructs the
  ``gr.Blocks`` UI. ``model`` is a loaded ``MoondreamModel`` (we close over
  it so we never reload weights per request); ``detect_fn`` is the
  ``detect_with_visual_prompt`` from ``inference.py``.
- ``discover_examples(examples_dir)``: scans ``examples/<id>/{query,target}.png``
  pairs for the gallery tab.
"""

from __future__ import annotations

import inspect
from pathlib import Path


HEADER = """
# Visual-prompt detection — Moondream 2 fine-tune

This is *image-as-prompt* detection: instead of asking the model in words
("find all the dogs"), you give it a **picture** of one example, and it
finds every other instance of that thing in your target image.

The fine-tune wires a single mean-pooled vision embedding of the query
image into the same prompt slot the tokenizer would normally fill with a
class name.
"""

FOOTER = (
    "---\n"
    "**Tips:** the model was trained on LVIS objects — it generalizes "
    "best when the query crop is a *clean, well-cropped* example. "
    "Heavy occlusion or unusual viewpoints in the query lower recall."
)


# ---------------------------------------------------------------------------
# Examples discovery
# ---------------------------------------------------------------------------


def discover_examples(examples_dir: str) -> list[list[str]]:
    """Find ``examples/<id>/{query.png,target.png}`` pairs.

    Tolerates a missing or empty examples dir (returns ``[]``); the app
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


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def build_gradio_app(
    model,
    detect_fn,
    examples_dir: str,
    inference_decorator=None,
):
    """Construct the two-tab Gradio Blocks for the visual-prompt demo.

    ``inference_decorator`` is an optional callable applied to the per-tab
    click handlers (``paint_to_find`` / ``detect_pair``). The HF Spaces
    ZeroGPU launcher passes ``spaces.GPU(duration=...)`` here so the model
    forward pass is what actually requests a real GPU; local & Modal
    launchers pass ``None`` (a real GPU is already attached).
    """
    import gradio as gr
    import numpy as np
    from PIL import Image

    decorate = inference_decorator if inference_decorator is not None else (lambda f: f)

    # `crop_size` was dropped from `gr.ImageEditor` in Gradio 6.x in favour
    # of `canvas_size` / `fixed_canvas`. We don't actually want to constrain
    # the canvas at all here — the user may upload arbitrarily-sized photos
    # — so we just omit the kwarg on versions that no longer accept it.
    _editor_extra_kwargs: dict = {}
    if "crop_size" in inspect.signature(gr.ImageEditor.__init__).parameters:
        _editor_extra_kwargs["crop_size"] = None

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

    examples = discover_examples(examples_dir)

    # --- assemble ----------------------------------------------------------

    # `theme=` moved from `gr.Blocks(...)` to `demo.launch(theme=...)` in
    # Gradio 6.0; we set it on the Blocks here for v5 compatibility (Modal
    # image pins gradio==5.50.0) and the v6 path emits a one-off warning
    # but still picks the theme up via the Blocks attribute.
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
                            height=520,
                            **_editor_extra_kwargs,
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
                    fn=decorate(paint_to_find),
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
                        "to populate this tab._"
                    )

                detect_pair_btn.click(
                    fn=decorate(detect_pair),
                    inputs=[gallery_query, gallery_target, max_obj_pair],
                    outputs=[annotated_pair, status_pair],
                    api_name="detect_pair",
                )

        gr.Markdown(FOOTER)

    return demo

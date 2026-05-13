"""
Local Gradio launcher for the Moondream visual-prompt demo.

Run from this directory::

    python local_app.py
    python local_app.py --share                              # public tunnel
    python local_app.py --base-model ../../moondream2/model.safetensors \\
                        --lora       ../../model_artifacts/moondream_visual_prompt_lora_step_340.safetensors

Defaults assume the layout this repo ships with:

- Base weights:  ``<repo>/moondream2/model.safetensors``
- LoRA weights:  ``deployments/modal_visual_prompt_app/lora_weights/lora.safetensors``
                  (a symlink into ``<repo>/model_artifacts/``)
- Examples dir:  ``deployments/modal_visual_prompt_app/examples``

Device selection mirrors ``visual_prompt_trainer.py``:
    cuda → mps → cpu
You can force one with the ``--device`` flag or ``MOONDREAM_DEVICE`` env var.

Why this is a separate entrypoint
---------------------------------
The Modal deployment in ``app.py`` wraps the same Gradio Blocks under ASGI,
mounts a Volume of converted base weights, and runs on an A10G. None of
that machinery makes sense locally — here we just load the model in the
host process and call ``demo.launch()``. The shared UI lives in
``gradio_app.py`` so both paths render the exact same app.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Make `inference`, `gradio_app`, `moondream2`, `trainer_helpers` importable
# from this directory regardless of where the user runs `python` from.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _default_paths() -> tuple[Path, Path, Path]:
    """Resolve sensible defaults for base / LoRA / examples paths.

    Walks upward from this file to find the repo root (heuristic: the dir
    containing ``moondream2/``), then picks paths inside it.
    """
    repo_root = _HERE
    for _ in range(6):  # bounded walk-up
        if (repo_root / "moondream2").exists():
            break
        repo_root = repo_root.parent

    base = repo_root / "moondream2" / "model.safetensors"
    lora = _HERE / "lora_weights" / "lora.safetensors"
    examples = _HERE / "examples"
    return base, lora, examples


def parse_args() -> argparse.Namespace:
    base_default, lora_default, examples_default = _default_paths()

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL_PATH", str(base_default)),
        help=f"Path to base Moondream safetensors (default: {base_default})",
    )
    p.add_argument(
        "--lora",
        default=os.environ.get("LORA_WEIGHTS_PATH", str(lora_default)),
        help=f"Path to fine-tuned LoRA safetensors (default: {lora_default})",
    )
    p.add_argument(
        "--examples-dir",
        default=os.environ.get("EXAMPLES_DIR", str(examples_default)),
        help=f"Directory of <id>/{{query,target}}.png pairs for the gallery tab "
             f"(default: {examples_default})",
    )
    # Match the only checkpoint currently in model_artifacts/.
    p.add_argument("--text-lora-rank", type=int,
                   default=int(os.environ.get("TEXT_LORA_RANK", "64")))
    p.add_argument("--text-lora-alpha", type=float,
                   default=float(os.environ.get("TEXT_LORA_ALPHA", "128")))
    p.add_argument("--proj-mlp-lora-rank", type=int,
                   default=int(os.environ.get("PROJ_MLP_LORA_RANK", "32")))
    p.add_argument("--proj-mlp-lora-alpha", type=float,
                   default=float(os.environ.get("PROJ_MLP_LORA_ALPHA", "64")))
    p.add_argument(
        "--device",
        default=os.environ.get("MOONDREAM_DEVICE"),
        help="Override device: cuda | mps | cpu. Default: cuda > mps > cpu.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true",
                   help="Open a public Gradio tunnel (handy for sharing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base = Path(args.base_model)
    lora = Path(args.lora)
    if not base.exists():
        raise SystemExit(
            f"Base model not found: {base}\n"
            "Pass --base-model or set BASE_MODEL_PATH."
        )
    if not lora.exists():
        raise SystemExit(
            f"LoRA weights not found: {lora}\n"
            "Pass --lora or set LORA_WEIGHTS_PATH."
        )

    # Imported lazily so `--help` is fast and so an import error in the
    # heavy deps surfaces *after* arg parsing rather than before it.
    from inference import detect_with_visual_prompt, load_model
    from gradio_app import build_gradio_app

    print(f"Loading model on device={args.device or 'auto'}…")
    print(f"  base: {base}")
    print(f"  lora: {lora}")
    model = load_model(
        base_model_path=str(base),
        lora_weights_path=str(lora),
        text_lora_rank=args.text_lora_rank,
        text_lora_alpha=args.text_lora_alpha,
        proj_mlp_lora_rank=args.proj_mlp_lora_rank,
        proj_mlp_lora_alpha=args.proj_mlp_lora_alpha,
        device=args.device,
    )
    print(f"Model loaded on {next(model.parameters()).device}.")

    demo = build_gradio_app(
        model=model,
        detect_fn=detect_with_visual_prompt,
        examples_dir=str(args.examples_dir),
    )

    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

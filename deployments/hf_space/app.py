"""
Hugging Face Space entrypoint (ZeroGPU) for the Moondream visual-prompt demo.

Why this looks different from ``deployments/modal_visual_prompt_app/local_app.py``
even though both render the same UI:

- ZeroGPU containers boot on CPU. A real GPU is only allocated *for the
  duration of an `@spaces.GPU`-decorated function call*. To make this work
  without paying CUDA-transfer cost on every request, the model must be
  placed on ``cuda`` at module import time — PyTorch CUDA emulation makes
  that legal on the CPU-only build container; the spaces runtime then
  migrates the already-on-cuda parameters into the real GPU on first call.
- The base Moondream weights are 4 GB. We don't ship them — we pull them
  from the official ``vikhyatk/moondream2`` repo at startup and rewrite
  the keys into the layout ``MoondreamModel`` wants. ``snapshot_download``
  caches under ``~/.cache/huggingface``, which persists across container
  restarts on Spaces, so this is a cold-start-only cost.
- Click handlers from ``gradio_app.py`` are wrapped in ``spaces.GPU(...)``
  via the ``inference_decorator`` hook. That's the only Spaces-specific
  surgery; the rest of the UI is the same module the local launcher uses.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import spaces
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# We can't write next to app.py on Spaces (the app dir is read-only after
# build). /tmp is writable but ephemeral; the heavy file (the HF snapshot
# itself) lives in the persistent HF cache, so re-converting on cold start
# is cheap (just a key rename + re-save).
CONVERTED_BASE_PATH = "/tmp/moondream2_converted.safetensors"
LORA_WEIGHTS_PATH = str(HERE / "lora_weights" / "lora.safetensors")
EXAMPLES_DIR = str(HERE / "examples")

# Match the only checkpoint shipped in this Space — the same hparams that
# `scripts/run_visual_prompt_sweep.sh::wide_both` was trained with.
TEXT_LORA_RANK = 64
TEXT_LORA_ALPHA = 128
PROJ_MLP_LORA_RANK = 32
PROJ_MLP_LORA_ALPHA = 64

# Cap any single inference at 60 s of GPU time. On the H200 backing
# ZeroGPU, a request takes <2 s end-to-end, so this is a generous cap that
# also covers the first call's CUDA migration.
GPU_DURATION_S = 60


def _ensure_base_model() -> str:
    """Download moondream2 weights and re-save in MoondreamModel's key layout.

    HF ships the keys as ``model.text.blocks.…`` (transformers wrapper);
    ``MoondreamModel.load_state_dict`` wants ``text.blocks.…``. The source
    download is cached, so this is cheap to re-run on cold starts.
    """
    if os.path.exists(CONVERTED_BASE_PATH):
        print(f"Converted base model present at {CONVERTED_BASE_PATH}, skipping.")
        return CONVERTED_BASE_PATH

    print("Downloading vikhyatk/moondream2 (safetensors only) from HF…")
    # Restrict to weights — config / tokenizer / GGUF / modeling_phi.py are
    # not needed for `MoondreamModel.load_state_dict` (we ship our own
    # config + tokenizer in moondream2/). Saves ~1 GB of cold-start IO.
    local_dir = snapshot_download(
        repo_id="vikhyatk/moondream2",
        revision="2025-06-21",
        allow_patterns=["*.safetensors"],
    )
    files = glob.glob(os.path.join(local_dir, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {local_dir}")

    state: dict = {}
    for f in sorted(files):
        state.update(load_file(f))
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state.items()
    }

    # Build the model purely to get the canonical key set, then re-save.
    # This re-save is what every subsequent cold start loads instead of
    # repeating the HF→MoondreamModel rename dance.
    from moondream2.moondream import MoondreamConfig, MoondreamModel

    print("Converting to MoondreamModel key layout…")
    skeleton = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    missing, unexpected = skeleton.load_state_dict(state, strict=False)
    print(f"  load_state_dict — missing: {len(missing)}, unexpected: {len(unexpected)}")
    save_file(skeleton.state_dict(), CONVERTED_BASE_PATH)
    print(f"Saved converted weights to {CONVERTED_BASE_PATH}")
    del skeleton
    return CONVERTED_BASE_PATH


# ---------------------------------------------------------------------------
# Module-level model load (intentionally — see file docstring).
# ---------------------------------------------------------------------------

base_path = _ensure_base_model()

from inference import detect_with_visual_prompt, load_model  # noqa: E402
from gradio_app import build_gradio_app  # noqa: E402

print("Loading Moondream + LoRA on cuda (CUDA emulation outside @spaces.GPU)…")
model = load_model(
    base_model_path=base_path,
    lora_weights_path=LORA_WEIGHTS_PATH,
    text_lora_rank=TEXT_LORA_RANK,
    text_lora_alpha=TEXT_LORA_ALPHA,
    proj_mlp_lora_rank=PROJ_MLP_LORA_RANK,
    proj_mlp_lora_alpha=PROJ_MLP_LORA_ALPHA,
    device="cuda",
)
print(f"Model placed on {next(model.parameters()).device}.")


# ---------------------------------------------------------------------------
# Build the UI. The actual GPU work happens inside the click handlers, so
# `spaces.GPU(...)` is applied per-handler via the decorator hook.
# ---------------------------------------------------------------------------

demo = build_gradio_app(
    model=model,
    detect_fn=detect_with_visual_prompt,
    examples_dir=EXAMPLES_DIR,
    inference_decorator=spaces.GPU(duration=GPU_DURATION_S),
)

# HF Spaces auto-runs `app.py` and discovers `demo`. `.queue()` enables the
# request queue (recommended for ZeroGPU so concurrent visitors don't all
# race for the same allocation).
demo.queue().launch()

# Moondream Visual-Prompt Demo (Modal + Gradio)

A web demo for the visual-prompt LoRA fine-tune trained by
[`visual_prompt_trainer.py`](../../visual_prompt_trainer.py).

The model takes an *image* as the prompt instead of a text label and detects
every other instance of that thing in a target image. The Gradio UI exposes
two ways to try it:

- **Paint to find** — drop one image, paint over a single example object, the
  app auto-crops what you painted, uses it as the visual prompt, and draws
  boxes around every other matching instance in the same image.
- **LVIS examples** — clickable gallery of (query, target) pairs sampled from
  the LVIS held-out test split, so visitors can play without their own images.

The whole thing is one Modal app: a single URL serves the UI and a programmatic
REST API (Gradio's auto-generated `/api/predict`).

```
deployments/modal_visual_prompt_app/
├── app.py                # Modal entrypoint (mounts the Blocks under ASGI)
├── local_app.py          # Local launcher (`python local_app.py`)
├── gradio_app.py         # Shared Gradio Blocks (no Modal dep)
├── inference.py          # Trim of visual_prompt_trainer for serving
├── prepare_examples.py   # Helper to populate examples/ from LVIS
├── moondream2/           # symlink → ../../moondream2
├── trainer_helpers.py    # symlink → ../../trainer_helpers.py
├── lora_weights/
│   └── lora.safetensors  # symlink → your chosen sweep checkpoint
├── examples/             # populated by prepare_examples.py
├── requirements.txt      # local dev parity with Modal image
└── README.md
```

## Quick start — run it locally

The Gradio Blocks have zero Modal dependency, so you can run them straight on
your laptop (CUDA, Apple Silicon MPS, or CPU). This is also what you'd push to
a Hugging Face Space.

```bash
cd deployments/modal_visual_prompt_app

# 1. (Optional) Populate the LVIS examples gallery (once, local-only).
python prepare_examples.py --num_pairs=8

# 2. Launch — uses ../../moondream2/model.safetensors and lora_weights/lora.safetensors
#    by default, picks cuda > mps > cpu, serves on http://127.0.0.1:7860/.
python local_app.py

# Useful flags:
python local_app.py --device cpu                      # force CPU
python local_app.py --port 7861                       # change port
python local_app.py --share                           # public Gradio tunnel
python local_app.py --base-model /abs/path/to/m.safetensors \
                    --lora       /abs/path/to/lora.safetensors
```

`local_app.py` defaults to the LoRA hparams of the checkpoint shipped in
`model_artifacts/` (`text_lora_rank=64 / alpha=128`,
`proj_mlp_lora_rank=32 / alpha=64`). If you point `--lora` at a different
sweep checkpoint, pass the matching `--text-lora-rank` etc. — `inference.py`
cross-checks shapes and will refuse to silently load a mismatched adapter.

### Gradio version note

Both Gradio 5.x and 6.x work. The Modal image pins `gradio==5.50.0`; on a
local 6.x install you'll see a one-off warning about `theme=` having moved
from `gr.Blocks(...)` to `demo.launch(...)` — harmless.

## Quick start — deploy on Modal

```bash
cd deployments/modal_visual_prompt_app

# 1. (Optional) Populate the LVIS examples gallery.
python prepare_examples.py --num_pairs=8

# 2. Live-reload dev (creates an https tunnel; reloads on file save).
modal serve app.py

# 3. Or deploy a stable URL.
modal deploy app.py
```

After `modal serve`, open the printed URL in your browser. The first request
takes ~60s (cold start: container boots, model loads onto an A10G).
Subsequent requests are sub-second until the container idles out.

## Picking a different LoRA checkpoint

The default `lora_weights/lora.safetensors` is a symlink to the
`vp_sweep/wide_proj_mlp/step_100` checkpoint, trained with these hparams:

| param                  | value |
|------------------------|------:|
| `text_lora_rank`       |   32  |
| `text_lora_alpha`      |   64  |
| `proj_mlp_lora_rank`   |   64  |
| `proj_mlp_lora_alpha`  |  128  |

To swap to a different sweep checkpoint, point the symlink at it AND make sure
the LoRA hparams match what that run used (look at
[`scripts/run_visual_prompt_sweep.sh`](../../scripts/run_visual_prompt_sweep.sh)
or the matching wandb run config). Override either via env vars at the top of
`app.py` or by re-running:

```bash
# Repoint the weights …
ln -sf ../../../model_artifacts/vp_sweep/wide_both_high_lr/moondream_visual_prompt_lora_step_50.safetensors \
       lora_weights/lora.safetensors

# … and update these constants in app.py if hparams differ:
#   TEXT_LORA_RANK / TEXT_LORA_ALPHA
#   PROJ_MLP_LORA_RANK / PROJ_MLP_LORA_ALPHA
modal deploy app.py
```

> **Why does the rank matter at load time?** LoRA wraps the original linear
> layers with a low-rank `(A @ B)` adapter. The checkpoint stores `A` and `B`
> with shapes that depend on the rank — so the model has to be built with the
> *same* rank before `load_state_dict(strict=False)` will accept the weights.

## API usage

The Gradio app auto-exposes a REST endpoint per `api_name=` block:

```python
from gradio_client import Client

client = Client("https://YOUR_WORKSPACE--moondream-visual-prompt-demo-visualpromptdemo-ui.modal.run")

# "Paint to find" tab requires the editor's dict shape — easier to use the
# pair endpoint from a script:
result = client.predict(
    "query.jpg",     # query crop
    "target.jpg",    # target image
    25,              # max_objects
    api_name="/detect_pair",
)
print(result)
```

For Python-only callers (no HTTP), use the Modal class directly:

```python
import modal
demo = modal.Cls.from_name("moondream-visual-prompt-demo", "VisualPromptDemo")()
boxes = demo.detect.remote(open("target.jpg", "rb").read(),
                           open("query.jpg", "rb").read())
```

Or via the local entrypoint:

```bash
modal run app.py --target target.jpg --query query.jpg
```

## Environment variables

All are optional — defaults match the bundled `wide_proj_mlp` checkpoint.

| name                    | default                              |
|-------------------------|--------------------------------------|
| `BASE_MODEL_PATH`       | `/root/app/moondream2/model.safetensors` |
| `LORA_WEIGHTS_PATH`     | `/root/app/lora_weights/lora.safetensors` |
| `TEXT_LORA_RANK`        | `32`                                 |
| `TEXT_LORA_ALPHA`       | `64`                                 |
| `PROJ_MLP_LORA_RANK`    | `64`                                 |
| `PROJ_MLP_LORA_ALPHA`   | `128`                                |

Set them with `app = modal.App(...).env(...)` in `app.py` or pass per-deploy
via `modal deploy app.py --env BASE_MODEL_PATH=...`.

## Notes & gotchas

- **Symlinks**: `moondream2/`, `trainer_helpers.py`, and `lora_weights/lora.safetensors`
  are symlinked into the deployment dir to keep one source of truth. Modal's
  `add_local_dir` / `add_local_file` follows symlinks, so this works for
  build steps. If you ever move the deployment folder *out* of the repo,
  replace the symlinks with `cp -r` copies.
- **GPU**: `gpu="A10G"` is the cheapest GPU that comfortably runs Moondream 2
  with both LoRAs at fp32. Bump to `L4` or `A100` if you want lower per-request
  latency.
- **Cold start**: ~60s the first time a container spins up (container boot +
  weight load). After that, requests take ~0.5–1s. Set `max_containers=N` and
  `min_containers=1` (a.k.a. `keep_warm`) on the `@app.cls(...)` decorator if
  you need always-warm capacity.
- **Privacy**: `modal serve` and `modal deploy` create *public* URLs by
  default. Use Modal's `auth_token` decorator or run behind your own
  authenticating proxy if the demo shouldn't be open to the world.

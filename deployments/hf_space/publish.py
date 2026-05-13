"""
Publish the visual-prompt demo to a Hugging Face Space (ZeroGPU).

Why not just push this directory directly?
------------------------------------------
The runtime needs ``inference.py``, ``gradio_app.py``, ``trainer_helpers.py``,
the ``moondream2/`` module, the LoRA safetensors, and the LVIS examples —
all of which live in different places in this repo (we keep one source of
truth and symlink in the Modal deployment dir). HF Spaces is a single git
repo, so this script *stages* a self-contained tree into a temp dir and
then uploads that tree.

Usage::

    HF_TOKEN=hf_xxx python publish.py
    HF_TOKEN=hf_xxx python publish.py --repo-id nkasmanoff/my-space --private
    HF_TOKEN=hf_xxx python publish.py --hardware cpu-basic        # for testing

Idempotent: re-runs upload (and updates hardware if it changed). The
underlying ``upload_folder`` is a single commit per run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # deployments/hf_space → repo root
MODAL_DIR = REPO_ROOT / "deployments" / "modal_visual_prompt_app"


def stage(stage_dir: Path, lora_src: Path) -> None:
    """Copy everything the Space needs into ``stage_dir``.

    Layout produced (mirrors what app.py expects to import):

        stage_dir/
        ├── app.py
        ├── gradio_app.py
        ├── inference.py
        ├── trainer_helpers.py
        ├── README.md
        ├── requirements.txt
        ├── moondream2/         (Python files only — no model.safetensors)
        ├── lora_weights/lora.safetensors
        └── examples/<id>/{query,target}.png
    """
    print(f"Staging into {stage_dir}…")
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Top-level files from this directory
    for name in ("app.py", "README.md", "requirements.txt", ".gitattributes"):
        src = HERE / name
        if src.exists():
            shutil.copy2(src, stage_dir / name)

    # Shared modules from the Modal deployment dir + repo root
    for src, dst in (
        (MODAL_DIR / "gradio_app.py", stage_dir / "gradio_app.py"),
        (MODAL_DIR / "inference.py", stage_dir / "inference.py"),
        (REPO_ROOT / "trainer_helpers.py", stage_dir / "trainer_helpers.py"),
    ):
        if not src.exists():
            raise FileNotFoundError(f"Missing required source file: {src}")
        shutil.copy2(src, stage_dir / dst.name)

    # moondream2/ Python module — explicitly skip the 4 GB model.safetensors
    # (we download it fresh on the Space at startup) and any cache cruft.
    moondream_src = REPO_ROOT / "moondream2"
    moondream_dst = stage_dir / "moondream2"
    moondream_dst.mkdir(exist_ok=True)
    for entry in moondream_src.iterdir():
        if entry.name in {"model.safetensors", "__pycache__", "wandb"}:
            continue
        if entry.suffix == ".pyc":
            continue
        target = moondream_dst / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)

    # LoRA weights — single file ~220 MB, will be uploaded via LFS.
    lora_dst_dir = stage_dir / "lora_weights"
    lora_dst_dir.mkdir(exist_ok=True)
    if not lora_src.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_src}")
    shutil.copy2(lora_src, lora_dst_dir / "lora.safetensors")

    # Examples gallery — small PNG pairs.
    examples_src = MODAL_DIR / "examples"
    examples_dst = stage_dir / "examples"
    if examples_src.exists():
        examples_dst.mkdir(exist_ok=True)
        for sub in examples_src.iterdir():
            if not sub.is_dir():
                continue
            shutil.copytree(sub, examples_dst / sub.name, dirs_exist_ok=True)

    # `.gitattributes` so HF tracks the LoRA file via LFS. Anything matching
    # `*.safetensors` is already LFS-tracked by default on Spaces, but being
    # explicit avoids a "file too large" rejection in edge cases.
    gitattrs = stage_dir / ".gitattributes"
    if not gitattrs.exists():
        gitattrs.write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
            "*.bin filter=lfs diff=lfs merge=lfs -text\n"
            "*.png filter=lfs diff=lfs merge=lfs -text\n"
        )

    print("Staged contents:")
    for p in sorted(stage_dir.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            rel = p.relative_to(stage_dir)
            print(f"  {rel}  ({size:,} bytes)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--repo-id",
        default=os.environ.get("HF_SPACE_REPO_ID", "nkasmanoff/moondream-visual-prompt"),
        help="Space repo id (e.g. user/space-name).",
    )
    p.add_argument("--private", action="store_true", help="Create as a private Space.")
    p.add_argument(
        "--hardware",
        default="zero-a10g",
        help="Space hardware. ZeroGPU = 'zero-a10g' (legacy name; backed by H200). "
             "Use 'cpu-basic' for free CPU testing.",
    )
    p.add_argument(
        "--lora",
        default=str(REPO_ROOT / "model_artifacts" / "moondream_visual_prompt_lora_step_340.safetensors"),
        help="Path to the LoRA safetensors to bundle.",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Stage only — print the staged tree and exit. Useful for debugging.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.no_upload:
        sys.exit("HF_TOKEN env var required (e.g. `HF_TOKEN=hf_xxx python publish.py`).")

    with tempfile.TemporaryDirectory(prefix="moondream-hf-space-") as tmp:
        stage_dir = Path(tmp)
        stage(stage_dir, Path(args.lora))

        if args.no_upload:
            print("--no-upload set; staged tree only.")
            return 0

        api = HfApi(token=token)

        print(f"\nCreating / updating Space {args.repo_id} (hardware={args.hardware})…")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=args.hardware,
            private=args.private,
            exist_ok=True,
        )

        # `create_repo` only sets hardware on first creation. If the space
        # already existed on a different tier (e.g. cpu-basic from a prior
        # run), force it onto the requested hardware now.
        try:
            api.request_space_hardware(repo_id=args.repo_id, hardware=args.hardware)
        except Exception as e:
            # PRO/Pro-org check or transient hub issue — don't kill the
            # upload over it; the user can flip hardware in the UI.
            print(f"  (request_space_hardware skipped: {e})")

        print(f"Uploading staged tree to {args.repo_id}…")
        api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=args.repo_id,
            repo_type="space",
            commit_message="Publish Moondream visual-prompt demo",
        )

        url = f"https://huggingface.co/spaces/{args.repo_id}"
        print(f"\nDone. Space: {url}")
        print(f"Build logs: {url}?logs=build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Generate a small static gallery of (query_crop, target_image) pairs for the
"LVIS examples" tab of the Gradio demo.

Run this **once** locally before ``modal deploy app.py``. Output layout::

    examples/
      0001_basketball/
        query.png
        target.png
        meta.json
      0002_zebra/
        query.png
        target.png
        meta.json
      ...

The script picks ``--num_pairs`` triplets from the LVIS held-out *test* split
(novel classes, by default) so the gallery shows the model generalizing rather
than recalling training examples.

Usage::

    python prepare_examples.py
    python prepare_examples.py --num_pairs=12 --max_samples=2000
    python prepare_examples.py --hold_out_classes=False  # familiar classes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

# Make the parent repo importable so ``_datasets`` resolves when this file is
# run directly from inside ``deployments/modal_visual_prompt_app/``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _datasets.lvis_visual_prompt_dataset import LVISVisualPrompting, load_lvis_fo  # noqa: E402


def main(
    num_pairs: int = 8,
    out_dir: str = "examples",
    dataset_name: str = "Voxel51/LVIS",
    max_samples: int = 5000,
    hold_out_classes: bool = True,
    split: str = "test",
    split_seed: int = 0,
    sampling_seed: int = 42,
    min_target_boxes: int = 2,
):
    """Dump ``num_pairs`` triplets to ``out_dir/``.

    Args:
        num_pairs: how many (query, target) pairs to save.
        out_dir: directory to write into; created if missing.
        dataset_name: FiftyOne / Hub name.
        max_samples: cap on samples pulled from the Hub.
        hold_out_classes: if True, sample from the held-out test split's
            *novel* classes (the more interesting demo).
        split: which split to draw from ('train' / 'val' / 'test').
        split_seed: must match training to keep the train/val/test partition.
        sampling_seed: this script's own seed for picking which triplets.
        min_target_boxes: skip triplets where the target image only has 1 box —
            we want examples that demonstrate detecting *multiple* instances.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset_name} (max_samples={max_samples}) …")
    fo_dataset = load_lvis_fo(dataset_name, max_samples=max_samples)

    ds = LVISVisualPrompting(
        split=split,
        dataset_name=dataset_name,
        max_samples=max_samples,
        hold_out_classes=hold_out_classes,
        num_triplets_per_epoch=max(num_pairs * 8, 256),
        split_seed=split_seed,
        sampling_seed=sampling_seed,
        fo_dataset=fo_dataset,
    )

    saved = 0
    seen_classes: set[str] = set()
    for i in range(len(ds)):
        if saved >= num_pairs:
            break

        sample = ds[i]
        boxes = sample["boxes"]
        cls = sample["class_name"]

        if boxes.shape[0] < min_target_boxes:
            continue
        # Spread across distinct classes for visual variety.
        if cls in seen_classes:
            continue
        seen_classes.add(cls)

        slug = "".join(c if c.isalnum() else "_" for c in cls)[:40].strip("_")
        pair_dir = out_path / f"{saved + 1:04d}_{slug}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        sample["query_crop"].save(pair_dir / "query.png")
        sample["target_image"].save(pair_dir / "target.png")
        meta = {
            "class_name": cls,
            "num_gt_boxes": int(boxes.shape[0]),
            "boxes_xywh_normalized": boxes.tolist(),
            "split": split,
            "hold_out_classes": hold_out_classes,
        }
        (pair_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  [{saved + 1:>2}/{num_pairs}] {cls!r:>30} → {pair_dir}")
        saved += 1

    if saved == 0:
        raise RuntimeError(
            "No examples written. Try lowering min_target_boxes or raising "
            "max_samples / num_pairs."
        )

    print(f"\nDone. Wrote {saved} examples to {out_path.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)

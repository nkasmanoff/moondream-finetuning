"""
LVIS-based visual-prompting dataset.

Each ``__getitem__`` samples a triplet on the fly:

- ``query_crop``: PIL.Image cropped to one bounding box of a chosen class
- ``target_image``: a *different* image that contains >=1 instance of the same class
- ``boxes``: tensor (N, 4) of all GT [x_min, y_min, w, h] boxes for that class
  in ``target_image``, normalized to [0, 1]
- ``class_name``: str (used for logging only; the model never sees the label)

Two split modes via ``hold_out_classes``:

- ``False``: split images of the same class pool — vanilla "new images, seen classes"
- ``True``:  split *classes* themselves — the interesting "novel-class" eval

The dataset uses FiftyOne to fetch ``Voxel51/LVIS``, but only during ``__init__``;
``__getitem__`` reads file paths and pre-extracted bboxes from in-memory dicts.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

import fiftyone as fo
import fiftyone.utils.huggingface as fouh


def load_lvis_fo(
    dataset_name: str = "Voxel51/LVIS",
    max_samples: int = 5000,
) -> "fo.Dataset":
    """Idempotent loader: re-uses an existing FiftyOne dataset of the same name
    if one is already registered, otherwise downloads from the Hub.

    FiftyOne refuses to create two datasets with the same name and only
    silently falls back to the cached one when ``overwrite=False`` AND the
    download dir is reused — instantiating ``LVISVisualPrompting`` multiple
    times in a single process would otherwise crash with::

        ValueError: Dataset name 'Voxel51/LVIS' is not available
    """
    if fo.dataset_exists(dataset_name):
        return fo.load_dataset(dataset_name)
    return fouh.load_from_hub(
        dataset_name, max_samples=max_samples, overwrite=False,
    )


class LVISVisualPrompting(Dataset):
    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "Voxel51/LVIS",
        max_samples: int = 5000,
        num_triplets_per_epoch: int = 10_000,
        hold_out_classes: bool = True,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        split_seed: int = 0,
        sampling_seed: int = 0,
        min_distinct_images_per_class: int = 2,
        fo_dataset: Optional["fo.Dataset"] = None,
    ):
        """If ``fo_dataset`` is provided, it is used directly (avoiding repeated
        Hub loads when constructing multiple splits). Otherwise we look up the
        named dataset in the local FiftyOne registry, falling back to a Hub
        download. Both paths share the same underlying samples — only the
        in-memory split indices differ between train/val/test.
        """
        assert split in {"train", "val", "test"}, f"unknown split: {split}"

        if fo_dataset is None:
            fo_dataset = load_lvis_fo(dataset_name, max_samples=max_samples)

        # Build an in-memory index so __getitem__ never touches FiftyOne.
        # images_by_sid[sid] = {"filepath": str, "dets": [(label, [x,y,w,h]), ...]}
        images_by_sid: dict[str, dict] = {}
        for sample in fo_dataset.iter_samples():
            dets = sample.detections.detections if sample.detections is not None else None
            if not dets:
                continue
            images_by_sid[sample.id] = {
                "filepath": sample.filepath,
                "dets": [(d.label, list(d.bounding_box)) for d in dets],
            }

        if hold_out_classes:
            # First gather all classes across all images.
            class_to_locs_full: dict[str, list[tuple[str, int]]] = {}
            for sid, info in images_by_sid.items():
                for j, (label, _) in enumerate(info["dets"]):
                    class_to_locs_full.setdefault(label, []).append((sid, j))

            eligible = sorted(
                c for c, locs in class_to_locs_full.items()
                if len({sid for sid, _ in locs}) >= min_distinct_images_per_class
            )

            rng = random.Random(split_seed)
            shuffled = list(eligible)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_tr = int(n * train_ratio)
            n_va = int(n * val_ratio)
            class_slices = {
                "train": shuffled[:n_tr],
                "val": shuffled[n_tr : n_tr + n_va],
                "test": shuffled[n_tr + n_va :],
            }
            keep_classes = set(class_slices[split])

            self.class_to_locs = {
                c: locs for c, locs in class_to_locs_full.items() if c in keep_classes
            }
            # Keep only images that contain at least one kept-class detection.
            self.images_by_sid = {
                sid: info
                for sid, info in images_by_sid.items()
                if any(label in keep_classes for label, _ in info["dets"])
            }
        else:
            # Split by image id; each image lives in exactly one of train/val/test.
            rng = random.Random(split_seed)
            sids = sorted(images_by_sid)
            rng.shuffle(sids)
            n = len(sids)
            n_tr = int(n * train_ratio)
            n_va = int(n * val_ratio)
            sid_slices = {
                "train": sids[:n_tr],
                "val": sids[n_tr : n_tr + n_va],
                "test": sids[n_tr + n_va :],
            }
            keep_sids = set(sid_slices[split])

            self.images_by_sid = {
                sid: info for sid, info in images_by_sid.items() if sid in keep_sids
            }
            class_to_locs: dict[str, list[tuple[str, int]]] = {}
            for sid, info in self.images_by_sid.items():
                for j, (label, _) in enumerate(info["dets"]):
                    class_to_locs.setdefault(label, []).append((sid, j))
            # Re-filter: need >=2 distinct images of each class in this split.
            self.class_to_locs = {
                c: locs for c, locs in class_to_locs.items()
                if len({sid for sid, _ in locs}) >= min_distinct_images_per_class
            }

        self.classes = sorted(self.class_to_locs)
        if not self.classes:
            raise RuntimeError(
                f"No eligible classes in split={split!r}. "
                f"Try lowering min_distinct_images_per_class or raising max_samples."
            )

        self.num_triplets = num_triplets_per_epoch
        self.split = split
        self.hold_out_classes = hold_out_classes
        # Seed used to spice up per-index rngs so different epochs see different draws.
        self._epoch_salt = sampling_seed

    def __len__(self) -> int:
        return self.num_triplets

    def _sample_triplet(self, idx: int) -> dict:
        # Per-index deterministic rng — reproducible & DataLoader-worker-safe.
        rng = random.Random((idx + 1) * 2654435761 ^ self._epoch_salt)

        cls = rng.choice(self.classes)
        locs = self.class_to_locs[cls]
        distinct_sids = list({sid for sid, _ in locs})
        q_sid, t_sid = rng.sample(distinct_sids, 2)
        q_j = rng.choice([j for sid, j in locs if sid == q_sid])

        q_info = self.images_by_sid[q_sid]
        q_img = Image.open(q_info["filepath"]).convert("RGB")
        qx, qy, qw, qh = q_info["dets"][q_j][1]
        W, H = q_img.size
        query_crop = q_img.crop(
            (qx * W, qy * H, (qx + qw) * W, (qy + qh) * H)
        )

        t_info = self.images_by_sid[t_sid]
        t_img = Image.open(t_info["filepath"]).convert("RGB")
        boxes = [bb for label, bb in t_info["dets"] if label == cls]

        return {
            "query_crop": query_crop,
            "target_image": t_img,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "class_name": cls,
        }

    def __getitem__(self, idx: int) -> dict:
        return self._sample_triplet(idx)

"""
Categories available in the basketball-player-detection-3.v1i.coco dataset:
ball	

ball-in-basket	

number	

player	

player-in-possession	

player-jump-shot	

player-layup-dunk	

player-shot-block	

referee	

rim
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class BasketballCocoDataset(Dataset):
    def __init__(self, root_dir, split="train", categories_to_use=None):
        """
        Basketball COCO-style dataset for object detection.

        Args:
            root_dir: Path to basketball-player-detection-3.v1i.coco directory
            split: One of "train", "valid", or "test"
            categories_to_use: List of category names to filter by (e.g., ["ball", "player"])
                              If None, uses all categories
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split

        # Load COCO annotations
        annotations_path = self.split_dir / "_annotations.coco.json"
        with open(annotations_path, "r") as f:
            self.coco_data = json.load(f)

        # Create category mapping
        self.categories = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }

        # Create image_id to filename mapping
        self.images = {img["id"]: img for img in self.coco_data["images"]}

        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)

        # Filter by categories if specified
        if categories_to_use is not None:
            # Get category IDs for the requested categories
            category_ids_to_use = [
                cat_id
                for cat_id, cat_name in self.categories.items()
                if cat_name in categories_to_use
            ]

            # Filter annotations to only include the desired categories
            filtered_annotations = {}
            for img_id, anns in self.annotations_by_image.items():
                filtered_anns = [
                    ann for ann in anns if ann["category_id"] in category_ids_to_use
                ]
                if (
                    filtered_anns
                ):  # Only include images that have at least one annotation of the desired category
                    filtered_annotations[img_id] = filtered_anns

            self.annotations_by_image = filtered_annotations

        # Get list of image_ids that have annotations
        self.image_ids = list(self.annotations_by_image.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]

        # Load image
        image_path = self.split_dir / image_info["file_name"]
        image = Image.open(image_path)

        # Convert the image to RGB if it's not already
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get annotations for this image
        annotations = self.annotations_by_image[image_id]

        # Return the first annotation (similar to refcoco_dataset)
        # You can modify this to return all annotations if needed
        ann = annotations[0]

        # Get category label
        label = self.categories[ann["category_id"]]

        # Get bounding box (COCO format: x, y, width, height)
        x, y, w, h = ann["bbox"]

        # Normalize to 0-1 (similar to refcoco_dataset)
        boxes = {
            "x_min": x / image.width,
            "y_min": y / image.height,
            "x_max": (x + w) / image.width,
            "y_max": (y + h) / image.height,
        }

        return image, label, [boxes]


class BasketballDetection(Dataset):
    def __init__(self, split: str = "train", categories_to_use=["player"]):
        """Wrapper for basketball dataset to match SFT trainer format"""
        self.categories_to_use = categories_to_use
        dataset_root = "datasets/basketball-player-detection-3.v1i.coco"

        if split == "train":
            self.dataset = BasketballCocoDataset(
                dataset_root, split="train", categories_to_use=categories_to_use
            )
        elif split == "val":
            self.dataset = BasketballCocoDataset(
                dataset_root, split="valid", categories_to_use=categories_to_use
            )
        else:
            self.dataset = BasketballCocoDataset(
                dataset_root, split="test", categories_to_use=categories_to_use
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, boxes = self.dataset[idx]

        # Convert from normalized [x_min, y_min, x_max, y_max] format
        # to [x, y, w, h] format expected by SFT trainer
        flat_boxes = []
        class_names = []
        for box in boxes:
            x = box["x_min"]
            y = box["y_min"]
            w = box["x_max"] - box["x_min"]
            h = box["y_max"] - box["y_min"]
            flat_boxes.append([x, y, w, h])
            class_names.append(label.replace("-", " "))

        # Use float32 for better numerical stability and compatibility
        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)

        return {
            "image": image,
            "boxes": flat_boxes,
            "class_names": class_names,
            "image_id": image_id,
        }

    def get_sample_for_validation(self, idx):
        """Get sample in format compatible with validation functions"""
        # Just use the underlying dataset's format which is already correct
        return self.dataset[idx]


def load_object_detection_dataset(split):
    """
    Load basketball dataset for training/validation.

    Args:
        split: Either "train" or "val"

    Returns:
        BasketballCocoDataset instance
    """
    dataset_root = "datasets/basketball-player-detection-3.v1i.coco"

    if split == "train":
        return BasketballCocoDataset(
            dataset_root,
            split="train",
            categories_to_use=["ball"],  # Filter to only "ball" category
        )
    elif split == "val":
        return BasketballCocoDataset(
            dataset_root,
            split="valid",  # Note: COCO dataset uses "valid" not "val"
            categories_to_use=["ball"],  # Filter to only "ball" category
        )
    elif split == "test":
        return BasketballCocoDataset(
            dataset_root,
            split="test",
            categories_to_use=["ball"],  # Filter to only "ball" category
        )
    else:
        raise ValueError(f"Invalid split: {split}")

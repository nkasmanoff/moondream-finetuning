from torch.utils.data import Dataset

from datasets import load_dataset


class ObjectDetectionDataset(Dataset):
    def __init__(self, split, ds_name):
        self.dataset = load_dataset(ds_name)[split]

        # shuffle the dataset
        self.dataset.shuffle(seed=11)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # from the dataset  get image, convert to xmin, ymin, xmax, ymax
        sample = self.dataset[idx]
        image = sample["image"]
        # Convert the image to RGB if it's not already
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = sample["answer"][0]
        box = sample["bbox"]  # coco bbox is x, y, w, h
        x, y, w, h = box

        # normalize to 0-1
        boxes = {
            "x_min": (x) / image.width,
            "y_min": (y) / image.height,
            "x_max": (x + w) / image.width,
            "y_max": (y + h) / image.height,
        }
        # Extract the image, query, and bounding boxes
        return image, label, [boxes]


def load_object_detection_dataset(split):
    if split == "train":
        return ObjectDetectionDataset("val", "lmms-lab/RefCOCO")
    elif split == "val":
        return ObjectDetectionDataset("test", "lmms-lab/RefCOCO")
    else:
        raise ValueError(f"Invalid split: {split}")

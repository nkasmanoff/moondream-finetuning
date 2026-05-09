from datasets import load_dataset

from torch.utils.data import Dataset

ds_name = "moondream/waste_detection"  # "nkasmanoff/retail_detector_flattened"  # or "moondream/waste_detection"

# ds = load_dataset("moondream/waste_detection")
centered_coords = True if "nkasmanoff" in ds_name else False


class ObjectDetectionDataset(Dataset):
    def __init__(self, split, ds_name, centered_coords=centered_coords):
        if "nkasmanoff" in ds_name:
            if split == "train":
                dataset = load_dataset(ds_name)["train"]
                # take the first 95%
                dataset = dataset.select(range(int(len(dataset) * 0.95)))
                self.dataset = dataset
            elif split == "val":
                # take the last 5%
                dataset = load_dataset(ds_name)["train"]
                dataset = dataset.select(range(int(len(dataset) * 0.95), len(dataset)))
                self.dataset = dataset
        else:
            self.dataset = load_dataset(ds_name)[split]

        # shuffle the dataset
        self.dataset.shuffle(seed=11)
        self.centered_coords = centered_coords
        # use a single sample

    # self.dataset = self.dataset.select(range(1))  # Use only the first sample for testing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # from the dataset  get image, convert to xmin, ymin, xmax, ymax
        sample = self.dataset[idx]
        image = sample["image"]
        # Convert the image to RGB if it's not already
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = sample["labels"][0]
        boxes = []
        for box in sample["boxes"]:
            x, y, w, h = box
            if self.centered_coords:
                x = x + w / 2
                y = y + h / 2
            boxes.append(
                {
                    "x_min": x - w / 2,
                    "y_min": y - h / 2,
                    "x_max": x + w / 2,
                    "y_max": y + h / 2,
                }
            )

        # Extract the image, query, and bounding boxes
        return image, label, boxes


def load_object_detection_dataset(split):
    if split == "val":
        split = "test"
    return ObjectDetectionDataset(split, ds_name, centered_coords=centered_coords)

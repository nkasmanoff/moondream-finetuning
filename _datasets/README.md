# Datasets

Contains all the datasets for training object detection models.

## Available Datasets

### Basketball COCO Dataset

-   **Location**: `basketball-player-detection-3.v1i.coco/`
-   **Format**: COCO-style annotations
-   **Splits**: train, valid, test
-   **Categories**: basketball, ball, ball-in-basket, number, player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block, referee, rim

**Usage**:

```python
from datasets.basketball_dataset import load_object_detection_dataset

# Load datasets (currently filtered to "ball" category only)
train_ds = load_object_detection_dataset("train")
val_ds = load_object_detection_dataset("val")
```

### RefCOCO Dataset

-   **Module**: `refcoco_dataset.py`
-   **Source**: Hugging Face `lmms-lab/RefCOCO`
-   **Format**: Referring expression object detection

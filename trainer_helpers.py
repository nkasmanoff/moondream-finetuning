"""
Shared helper functions for trainer scripts.
Contains validation, region loss, learning rate scheduling, and LoRA utilities.
"""

import math
import logging

import torch
import torch.nn.functional as F

# `matplotlib` and `wandb` are training-only deps used by `validate()`. We
# import them lazily inside that function so that inference-only consumers
# (e.g. the HF Spaces deployment, which only pulls in `LoRALinear` and
# `inject_lora_into_model` from this module) don't need to install them.

# Constants that may be used by multiple trainers
MAX_PLOT_SAMPLES = 25


# ============================================================================
# Learning Rate Scheduling
# ============================================================================


def lr_schedule(
    step: int, max_steps: int, base_lr: float, constant: bool = False
) -> float:
    """
    Learning rate schedule with warmup and cosine annealing.

    Args:
        step: Current training step
        max_steps: Total number of training steps
        base_lr: Base learning rate
        constant: If True, return constant learning rate

    Returns:
        Learning rate for the current step
    """
    if constant:
        return base_lr

    x = step / max_steps
    if x < 0.1:
        return 0.1 * base_lr + 0.9 * base_lr * x / 0.1
    else:
        return 0.1 * base_lr + 0.9 * base_lr * (1 + math.cos(math.pi * (x - 0.1))) / 2


# ============================================================================
# Validation Functions
# ============================================================================


def validate(model, val_ds, step, max_samples=250, max_plot_samples=MAX_PLOT_SAMPLES):
    """
    Validate the model on the validation set.

    Args:
        model: The model to validate
        val_ds: Validation dataset (must have get_sample_for_validation method or be indexable)
        step: Current training step for logging
        max_samples: Maximum number of samples to validate on
        max_plot_samples: Maximum number of samples to plot

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    import matplotlib.pyplot as plt
    import wandb

    from moondream2.rl_utils import match_boxes_score
    from moondream2.visualization_utils import plot_prediction
    from moondream2.moondream_functions import detect

    model.eval()
    TP = FP = FN = 0

    # Use the minimum of max_samples and the actual dataset size
    num_samples = min(max_samples, len(val_ds))

    images = []
    with torch.no_grad():
        for i in range(num_samples):
            # Handle different dataset formats
            if hasattr(val_ds, "get_sample_for_validation"):
                sample = val_ds.get_sample_for_validation(i)
            else:
                sample = val_ds[i]

            detections = detect(model, sample[0], sample[1], None, temperature=0)

            # Handle different sample formats
            if hasattr(val_ds, "get_sample_for_validation"):
                # Format: (image, label, normalized_boxes)
                tp, fp, fn = match_boxes_score(detections["objects"], sample[2])
            else:
                # Format: (image, label, boxes) - direct tuple
                tp, fp, fn = match_boxes_score(detections["objects"], sample[2])

            # plot sample
            if i < max_plot_samples:
                try:
                    fig = plot_prediction(detections, sample)
                    fig_path = f"predictions/prediction_{i}.png"
                    fig.savefig(fig_path, dpi=100, bbox_inches="tight")
                    plt.close(fig)  # Free up memory
                    images.append(wandb.Image(fig_path))
                except Exception as e:
                    logging.warning(f"Failed to save prediction image {i}: {e}")
            TP += tp
            FP += fp
            FN += fn

    # Only log images if we have any
    if images:
        try:
            wandb.log({"predictions": images[-max_plot_samples:]}, step=step)
        except Exception as e:
            logging.warning(f"Failed to log prediction images to wandb: {e}")

    model.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if TP + FP == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def validate_with_gt(val_ds, max_samples=250):
    """
    Validate with ground truth (perfect detection).

    Args:
        val_ds: Validation dataset (must have get_sample_for_validation method or be indexable)
        max_samples: Maximum number of samples to validate on

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    from moondream2.rl_utils import match_boxes_score

    num_samples = min(max_samples, len(val_ds))

    TP = FP = FN = 0
    for i in range(num_samples):
        # Handle different dataset formats
        if hasattr(val_ds, "get_sample_for_validation"):
            sample = val_ds.get_sample_for_validation(i)
            detections = {"objects": sample[2]}
            tp, fp, fn = match_boxes_score(detections["objects"], sample[2])
        else:
            sample = val_ds[i]
            detections = {"objects": sample[2]}
            tp, fp, fn = match_boxes_score(detections["objects"], sample[2])

        TP += tp
        FP += fp
        FN += fn

    if TP + FP == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================================
# Region Loss Functions
# ============================================================================


def region_loss(
    hidden_states: torch.Tensor,
    w,
    labels: torch.Tensor,
    c_idx: torch.Tensor,
    s_idx: torch.Tensor,
):
    """
    Compute region loss for coordinate and size predictions.

    Args:
        hidden_states: Hidden states from the text model
        w: Region model weights/module
        labels: Ground truth labels (coordinate and size bins)
        c_idx: Indices for coordinate tokens
        s_idx: Indices for size tokens

    Returns:
        Combined coordinate and size loss
    """
    from moondream2.region import decode_coordinate, decode_size

    l_idx = torch.arange(len(labels))

    c_idx = c_idx - 1
    c_hidden = hidden_states[:, c_idx, :]
    c_logits = decode_coordinate(c_hidden, w)
    c_labels = labels[(l_idx % 4) < 2]

    c_loss = F.cross_entropy(
        c_logits.view(-1, c_logits.size(-1)),
        c_labels,
    )

    s_idx = s_idx - 1
    s_hidden = hidden_states[:, s_idx, :]
    s_logits = decode_size(s_hidden, w).view(-1, 1024)
    s_labels = labels[(l_idx % 4) >= 2]

    s_loss = F.cross_entropy(s_logits, s_labels)

    return c_loss + s_loss


# ============================================================================
# Binning Helper Functions
# ============================================================================


def coord_to_bin(center: float, n_bins: int = 1024) -> int:
    """
    Map normalized coord in [0,1] to discrete bin [0, n_bins-1].

    Args:
        center: Normalized coordinate value in [0, 1]
        n_bins: Number of bins (default 1024)

    Returns:
        Bin index in [0, n_bins-1]
    """
    return int(min(max(round(center * (n_bins - 1)), 0), n_bins - 1))


def size_to_bin(s: float, n_bins: int = 1024) -> int:
    """
    Map normalized size to log-scale bin as in the region head:
    bin = (log2(size) + 10.0) / 10.0 * 1023.0, clamped to [0, 1023].

    Args:
        s: Normalized size value
        n_bins: Number of bins (default 1024)

    Returns:
        Bin index in [0, n_bins-1]
    """
    s_clamped = max(float(s), 1.0 / 1024.0)
    s_log2 = math.log2(s_clamped)
    mapped = (s_log2 + 10.0) / 10.0 * (n_bins - 1)
    b = int(round(mapped))
    return int(min(max(b, 0), n_bins - 1))


def bin_to_size(bin_idx: int, n_bins: int = 1024) -> float:
    """
    Inverse mapping from bin index back to size value.

    Args:
        bin_idx: Bin index in [0, n_bins-1]
        n_bins: Number of bins (default 1024)

    Returns:
        Size value corresponding to the bin index
    """
    return float(2.0 ** ((bin_idx / (n_bins - 1.0)) * 10.0 - 10.0))


# ============================================================================
# LoRA Functions
# ============================================================================


class LoRALinear(torch.nn.Module):
    """
    LoRA-enhanced Linear layer that wraps an existing Linear layer.
    During training, the original weights are frozen and only LoRA weights are trained.
    """

    def __init__(
        self,
        original_layer: torch.nn.Module,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA matrices (match dtype/device of the wrapped layer)
        weight = getattr(original_layer, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            lora_dtype = weight.dtype
            lora_device = weight.device
        else:
            lora_dtype = torch.get_default_dtype()
            lora_device = None

        self.lora_A = torch.nn.Parameter(
            torch.zeros(in_features, rank, dtype=lora_dtype, device=lora_device)
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(rank, out_features, dtype=lora_dtype, device=lora_device)
        )
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

        # Initialize A with kaiming uniform and B with zeros
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA."""
        original_output = self.original_layer(x)

        # Apply LoRA
        if self.dropout is not None:
            x_lora = self.dropout(x)
        else:
            x_lora = x

        # Ensure dtype compatibility for matmul
        if x_lora.dtype != self.lora_A.dtype:
            x_lora = x_lora.to(self.lora_A.dtype)

        lora_output = (x_lora @ self.lora_A @ self.lora_B) * self.scaling

        # Match output dtype to the original layer output
        if lora_output.dtype != original_output.dtype:
            lora_output = lora_output.to(original_output.dtype)

        return original_output + lora_output


def inject_lora_into_model(
    model: torch.nn.Module,
    rank: int = 16,
    alpha: float = 32,
    dropout: float = 0.1,
    target_modules: list = None,
):
    """
    Replace target Linear layers in the model with LoRA-enhanced versions.
    This modifies the model in-place.

    Args:
        model: The model to inject LoRA into
        rank: Rank of LoRA matrices
        alpha: Scaling factor for LoRA (typically 2x rank)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to

    Returns:
        List of LoRA parameters
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2"]

    # Get the device of the model
    model_device = next(model.parameters()).device
    lora_params = []

    # Inject LoRA into text model layers
    for i, block in enumerate(model.text.blocks):
        # Attention layers
        if "qkv" in target_modules and hasattr(block.attn, "qkv"):
            original = block.attn.qkv
            lora_layer = LoRALinear(original, rank, alpha, dropout).to(model_device)
            block.attn.qkv = lora_layer
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

        if "proj" in target_modules and hasattr(block.attn, "proj"):
            original = block.attn.proj
            lora_layer = LoRALinear(original, rank, alpha, dropout).to(model_device)
            block.attn.proj = lora_layer
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

        # MLP layers
        if "fc1" in target_modules and hasattr(block.mlp, "fc1"):
            original = block.mlp.fc1
            lora_layer = LoRALinear(original, rank, alpha, dropout).to(model_device)
            block.mlp.fc1 = lora_layer
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

        if "fc2" in target_modules and hasattr(block.mlp, "fc2"):
            original = block.mlp.fc2
            lora_layer = LoRALinear(original, rank, alpha, dropout).to(model_device)
            block.mlp.fc2 = lora_layer
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    logging.info(f"Injected LoRA into {len(lora_params) // 2} layers")
    return lora_params


def get_lora_state_dict(model: torch.nn.Module, include_region: bool = True) -> dict:
    """
    Extract LoRA parameters and optionally region model parameters from the model.

    Args:
        model: The model containing LoRA layers
        include_region: Whether to include region model parameters (default: True)

    Returns:
        Dictionary containing LoRA parameters and optionally region parameters
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B

    # Include region model parameters if requested
    if include_region and hasattr(model, "region"):
        for name, param in model.region.named_parameters():
            lora_state_dict[f"region.{name}"] = param

    return lora_state_dict

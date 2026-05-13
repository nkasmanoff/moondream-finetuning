"""
Minimal subset of trainer_helpers for inference: LoRA injection only.
"""

import math
import logging

import torch


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

        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_layer(x)

        if self.dropout is not None:
            x_lora = self.dropout(x)
        else:
            x_lora = x

        if x_lora.dtype != self.lora_A.dtype:
            x_lora = x_lora.to(self.lora_A.dtype)

        lora_output = (x_lora @ self.lora_A @ self.lora_B) * self.scaling

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
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2"]

    model_device = next(model.parameters()).device
    lora_params = []

    for i, block in enumerate(model.text.blocks):
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

"""
Visual-prompt inference for the fine-tuned Moondream 2 detector.

This is a *deployment-time* trim of ``visual_prompt_trainer.py`` — it keeps the
LoRA injection (text decoder + ``vision.proj_mlp``) and the inference path
``detect_with_visual_prompt`` but drops every training-only dependency
(``wandb``, ``fire``, ``fiftyone``, ``matplotlib``, dataset loaders, …).

The forward pass mirrors training exactly::

    [BOS, image patches] [detect prefix] [* mean_pool(vis_enc(query_crop)) *] [detect suffix]
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          single splice slot, replaces the
                                          tokenized class name in the text path

so a server built on this file produces the same predictions as
``validate_visual_prompt`` produces during training.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image
from safetensors.torch import load_file

from moondream2.image_crops import reconstruct_from_crops
from moondream2.moondream import MoondreamConfig, MoondreamModel
from moondream2.moondream_functions import (
    _prefill,
    _vis_enc,
    _vis_proj,
)
from moondream2.text import lm_head, text_encoder
from moondream2.vision import prepare_crops
from trainer_helpers import LoRALinear, inject_lora_into_model


# ============================================================================
# LoRA on vision.proj_mlp (mirrors visual_prompt_trainer.inject_lora_into_proj_mlp)
# ============================================================================


def inject_lora_into_proj_mlp(
    model: MoondreamModel,
    rank: int = 64,
    alpha: float = 128,
    dropout: float = 0.0,
) -> None:
    """Wrap ``vision.proj_mlp.fc1`` and ``.fc2`` with ``LoRALinear`` in-place.

    The same module path used at training time. Adapter weights for these layers
    are loaded later from the saved safetensors via ``load_state_dict(strict=False)``.
    """
    model_device = next(model.parameters()).device
    proj = model.vision.proj_mlp
    for name in ("fc1", "fc2"):
        original = getattr(proj, name)
        lora_layer = LoRALinear(original, rank, alpha, dropout).to(model_device)
        setattr(proj, name, lora_layer)


# ============================================================================
# Query-crop encoding
# ============================================================================


@torch.inference_mode()
def _run_query_encoder(model: MoondreamModel, image: Image.Image) -> torch.Tensor:
    """Project a (small) query image into the text decoder's input space.

    Returns ``(729, D)`` tokens — the LoRA on ``proj_mlp`` shapes how a query
    image gets mapped into the same space the tokenizer would have produced
    for a class-name string.
    """
    all_crops, tiling = prepare_crops(image, model.config.vision, device=model.device)
    outputs = _vis_enc(model, all_crops)
    global_features = outputs[0]
    local_features = outputs[1:].view(
        -1,
        model.config.vision.enc_n_layers,
        model.config.vision.enc_n_layers,
        model.config.vision.enc_dim,
    )
    reconstructed = reconstruct_from_crops(
        local_features,
        tiling,
        patch_size=1,
        overlap_margin=model.config.vision.overlap_margin,
    )
    return _vis_proj(model, global_features, reconstructed)


@torch.inference_mode()
def _query_emb_pooled(model: MoondreamModel, query_image: Image.Image) -> torch.Tensor:
    """Mean-pool the query's projected patch tokens to a ``(1, 1, D)`` prompt slot."""
    tokens = _run_query_encoder(model, query_image)            # (729, D)
    return tokens.mean(dim=0, keepdim=True).unsqueeze(0)        # (1, 1, D)


# ============================================================================
# Detection
# ============================================================================


@torch.inference_mode()
def detect_with_visual_prompt(
    model: MoondreamModel,
    target_image: Image.Image,
    query_image: Image.Image,
    max_objects: int = 25,
) -> list[dict]:
    """Run detection on ``target_image`` using ``query_image`` as the prompt.

    Returns a list of ``{x_min, y_min, x_max, y_max}`` dicts in ``[0, 1]`` coords.
    """
    encoded = model.encode_image(target_image)
    model.load_encoded_image(encoded)

    detect_template = model.config.tokenizer.templates["detect"]
    pre = text_encoder(
        torch.tensor([detect_template["prefix"]], device=model.device), model.text
    )
    suf = text_encoder(
        torch.tensor([detect_template["suffix"]], device=model.device), model.text
    )
    q_pooled = _query_emb_pooled(model, query_image)
    prompt_emb = torch.cat([pre, q_pooled, suf], dim=1)
    L = prompt_emb.size(1)

    pos = encoded.pos
    mask = model.attn_mask[:, :, pos : pos + L, :]
    pos_ids = torch.arange(pos, pos + L, device=model.device, dtype=torch.long)
    hidden = _prefill(model, prompt_emb, mask, pos_ids)
    logits = lm_head(hidden, model.text)
    next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
    return model._generate_points(
        hidden[:, -1:, :], next_token, pos + L,
        include_size=True, max_objects=max_objects,
    )


# ============================================================================
# Model loader
# ============================================================================


def load_model(
    base_model_path: str,
    lora_weights_path: str,
    *,
    text_lora_rank: int = 32,
    text_lora_alpha: float = 64,
    text_lora_targets: Optional[list[str]] = None,
    proj_mlp_lora_rank: int = 64,
    proj_mlp_lora_alpha: float = 128,
    device: Optional[str] = None,
) -> MoondreamModel:
    """Build the Moondream 2 model with both LoRA adapters injected and the
    fine-tuned visual-prompt safetensors loaded on top.

    Defaults match the ``vp_sweep/wide_proj_mlp`` config in
    ``scripts/run_visual_prompt_sweep.sh``.
    """
    if text_lora_targets is None:
        text_lora_targets = ["qkv", "proj", "fc1", "fc2"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    model.load_state_dict(load_file(base_model_path))
    model.to(device)
    for _, buf in model.named_buffers():
        buf.data = buf.data.to(device)

    inject_lora_into_model(
        model,
        rank=text_lora_rank,
        alpha=text_lora_alpha,
        dropout=0.0,
        target_modules=text_lora_targets,
    )
    inject_lora_into_proj_mlp(
        model,
        rank=proj_mlp_lora_rank,
        alpha=proj_mlp_lora_alpha,
        dropout=0.0,
    )

    # The trainer saves LoRA-only A/B + the full region head into one file, so
    # strict=False is intentional — the base model keys aren't in this file.
    adapter_state = load_file(lora_weights_path)
    model.load_state_dict(adapter_state, strict=False)
    model.eval()
    return model


# ============================================================================
# Convenience: env-var driven default loader (used by app.py)
# ============================================================================


def load_model_from_env() -> MoondreamModel:
    """Build the model using the same env-var contract used in ``app.py``.

    Useful for local smoke tests::

        BASE_MODEL_PATH=moondream2/model.safetensors \\
        LORA_WEIGHTS_PATH=lora_weights/lora.safetensors \\
        python -c "from inference import load_model_from_env; m = load_model_from_env(); print('ok')"
    """
    return load_model(
        base_model_path=os.environ.get(
            "BASE_MODEL_PATH", "moondream2/model.safetensors"
        ),
        lora_weights_path=os.environ.get(
            "LORA_WEIGHTS_PATH", "lora_weights/lora.safetensors"
        ),
        text_lora_rank=int(os.environ.get("TEXT_LORA_RANK", "32")),
        text_lora_alpha=float(os.environ.get("TEXT_LORA_ALPHA", "64")),
        proj_mlp_lora_rank=int(os.environ.get("PROJ_MLP_LORA_RANK", "64")),
        proj_mlp_lora_alpha=float(os.environ.get("PROJ_MLP_LORA_ALPHA", "128")),
    )

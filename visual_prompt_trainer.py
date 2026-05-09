"""
Teacher-forced **visual-prompt** fine-tuning.

This mirrors ``sft_trainer.py`` but replaces the text "object" string in the
``detect`` template with a query image whose mean-pooled vision embedding gets
spliced into the same prompt slot::

    [BOS, image patches] [detect prefix] [* mean_pool(vis_enc(query_crop)) *] [detect suffix]
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       text path : tokenizer.encode(" " + class_name)
                       visual path: this trainer

Training-time supervision is identical to ``sft_trainer.py``: teacher-forced
cross-entropy on the region head's coordinate / size bins for each GT box.

Trainable parameter set (v1):
- LoRA on the text decoder layers (qkv, proj, fc1, fc2)
- LoRA on the vision projection MLP (vision.proj_mlp.fc1, vision.proj_mlp.fc2)
  — this is the only adapter on the path query_crop -> splice_position
- Full fine-tune of model.region

The base ViT backbone is run under ``torch.no_grad()`` for both the target
image (frozen) and the query crop (we only need gradients through the
``proj_mlp`` LoRA, so the encoder activations don't need to be retained).

Usage::

    python visual_prompt_trainer.py
    python visual_prompt_trainer.py --hold_out_classes=False --num_triplets_per_epoch=5000
    python visual_prompt_trainer.py --overfit_batch_size=8 --epochs=10 --eval_interval=1
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import fire
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from matplotlib.patches import Rectangle
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from tqdm import tqdm

from _datasets.lvis_visual_prompt_dataset import (
    LVISVisualPrompting,
    load_lvis_fo,
)
from moondream2.image_crops import reconstruct_from_crops
from moondream2.moondream import MoondreamModel, MoondreamConfig
from moondream2.moondream_functions import (
    _decode_one_tok,
    _prefill,
    _vis_enc,
    _vis_proj,
    encode_image_grad,
)
from moondream2.region import (
    decode_coordinate,
    decode_size,
    encode_coordinate,
    encode_size,
)
from moondream2.rl_utils import match_boxes_score
from moondream2.text import lm_head, text_encoder
from moondream2.vision import prepare_crops
from trainer_helpers import (
    LoRALinear,
    bin_to_size,
    coord_to_bin,
    get_lora_state_dict,
    inject_lora_into_model,
    lr_schedule,
    size_to_bin,
)

device = "cuda" if torch.cuda.is_available() else "mps"


# ============================================================================
# LoRA on vision.proj_mlp
# ============================================================================


def inject_lora_into_proj_mlp(
    model: MoondreamModel,
    rank: int = 16,
    alpha: float = 32,
    dropout: float = 0.0,
) -> list[torch.nn.Parameter]:
    """Wrap ``vision.proj_mlp.fc1`` and ``.fc2`` with ``LoRALinear`` in-place.

    The proj_mlp is the only piece of the model on the path
    ``query_crop -> splice_position`` whose params get gradients in this trainer.
    """
    model_device = next(model.parameters()).device
    lora_params: list[torch.nn.Parameter] = []
    proj = model.vision.proj_mlp
    for name in ("fc1", "fc2"):
        original = getattr(proj, name)
        lora_layer = LoRALinear(original, rank, alpha, dropout).to(model_device)
        setattr(proj, name, lora_layer)
        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
    logging.info("Injected LoRA into vision.proj_mlp.fc1 / fc2")
    return lora_params


# ============================================================================
# Query-crop encoding (no_grad backbone, grad-enabled proj_mlp)
# ============================================================================


def _run_query_encoder(model: MoondreamModel, image) -> torch.Tensor:
    """Project a (small) query image into the text decoder's input space.

    Equivalent to ``_run_vision_encoder`` but the heavy ViT backbone runs under
    ``torch.no_grad()``. Gradients still flow back through ``proj_mlp`` (LoRA),
    which is the only trainable path from the query.
    """
    with torch.no_grad():
        all_crops, tiling = prepare_crops(image, model.config.vision, device=model.device)
        outputs = _vis_enc(model, all_crops)
        global_features = outputs[0].detach()
        local_features = outputs[1:].detach().view(
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
    return _vis_proj(model, global_features, reconstructed)  # (729, D), grads on proj_mlp


def _query_emb_pooled(model: MoondreamModel, query_image) -> torch.Tensor:
    """Mean-pool the query's projected patch tokens to a (1, 1, D) prompt slot."""
    tokens = _run_query_encoder(model, query_image)        # (729, D)
    pooled = tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)
    return pooled


# ============================================================================
# Per-sample loss (mirrors teacher_forced_region_loss in sft_trainer.py)
# ============================================================================


def teacher_forced_visual_prompt_loss(
    model: MoondreamModel,
    query_crop,
    target_image,
    boxes: torch.Tensor,
    max_objects: Optional[int] = None,
) -> torch.Tensor:
    """Teacher-forced detection loss with the visual-prompt slot.

    Args:
        model: Moondream 2 model.
        query_crop: PIL image — single instance of the class to detect.
        target_image: PIL image — image to detect that class in.
        boxes: tensor of shape (N, 4), each [x_min, y_min, w, h] in [0, 1].
        max_objects: cap on number of supervised GT boxes per call.
    """
    if boxes.numel() == 0:
        return torch.zeros([], device=model.device)

    detect_template = model.config.tokenizer.templates["detect"]
    prefix_ids = detect_template["prefix"]
    suffix_ids = detect_template["suffix"]

    # Target image is frozen — same as in sft_trainer.py.
    with torch.no_grad():
        encoded_target = encode_image_grad(model, target_image, settings=None)
    model.load_encoded_image(encoded_target)
    pos = encoded_target.pos

    # Build the spliced prompt embedding: [prefix_text, query_pooled, suffix_text].
    pre = text_encoder(
        torch.tensor([prefix_ids], device=model.device, dtype=torch.long), model.text
    )
    suf = text_encoder(
        torch.tensor([suffix_ids], device=model.device, dtype=torch.long), model.text
    )
    q_pooled = _query_emb_pooled(model, query_crop)            # (1, 1, D), grads on
    prompt_emb = torch.cat([pre, q_pooled, suf], dim=1)         # (1, L, D)
    L = prompt_emb.size(1)

    mask = model.attn_mask[:, :, pos : pos + L, :]
    pos_ids = torch.arange(pos, pos + L, device=model.device, dtype=torch.long)
    hidden_BC = _prefill(model, prompt_emb, mask, pos_ids)      # (1, L, D)
    pos = pos + L
    hidden = hidden_BC[:, -1:, :]

    # Subsequent one-token decode steps use a 2048-wide causal mask.
    step_mask = torch.zeros(1, 1, 2048, device=model.device, dtype=torch.bool)
    step_mask[:, :, :pos] = 1

    n = boxes.size(0) if max_objects is None else min(boxes.size(0), max_objects)
    total_loss = torch.zeros([], device=model.device)
    n_terms = 0

    for obj_idx in range(n):
        bb = boxes[obj_idx].to(model.device)
        x_min, y_min, w_box, h_box = bb
        x_center = torch.clamp(x_min + w_box / 2.0, 0.0, 1.0)
        y_center = torch.clamp(y_min + h_box / 2.0, 0.0, 1.0)

        # ---- X bin CE ----
        x_logits = decode_coordinate(hidden, model.region)
        x_target = torch.tensor(
            [coord_to_bin(float(x_center))], device=model.device, dtype=torch.long,
        )
        total_loss = total_loss + F.cross_entropy(
            x_logits.view(-1, x_logits.size(-1)), x_target
        )
        n_terms += 1

        x_t = (
            x_center.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(dtype=x_logits.dtype)
        )
        next_emb = encode_coordinate(x_t, model.region)
        step_mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = _decode_one_tok(model, next_emb, step_mask, pos_ids)
        pos += 1

        # ---- Y bin CE ----
        y_logits = decode_coordinate(hidden, model.region)
        y_target = torch.tensor(
            [coord_to_bin(float(y_center))], device=model.device, dtype=torch.long,
        )
        total_loss = total_loss + F.cross_entropy(
            y_logits.view(-1, y_logits.size(-1)), y_target
        )
        n_terms += 1

        y_t = (
            y_center.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(dtype=y_logits.dtype)
        )
        next_emb = encode_coordinate(y_t, model.region)
        step_mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = _decode_one_tok(model, next_emb, step_mask, pos_ids)
        pos += 1

        # ---- Size (w, h) bin CE ----
        size_logits = decode_size(hidden, model.region)         # (2, 1024)
        w_bin = size_to_bin(float(w_box))
        h_bin = size_to_bin(float(h_box))
        total_loss = total_loss + F.cross_entropy(
            size_logits[0].unsqueeze(0),
            torch.tensor([w_bin], device=model.device, dtype=torch.long),
        )
        total_loss = total_loss + F.cross_entropy(
            size_logits[1].unsqueeze(0),
            torch.tensor([h_bin], device=model.device, dtype=torch.long),
        )
        n_terms += 2

        size_t = torch.tensor(
            [bin_to_size(w_bin), bin_to_size(h_bin)],
            device=model.device, dtype=size_logits.dtype,
        )
        next_emb = encode_size(size_t, model.region).unsqueeze(0).unsqueeze(0)
        step_mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = _decode_one_tok(model, next_emb, step_mask, pos_ids)
        pos += 1

    return total_loss / max(n_terms, 1)


# ============================================================================
# Inference helper (visual-prompt detect, used during validation)
# ============================================================================


@torch.inference_mode()
def detect_with_visual_prompt(
    model: MoondreamModel,
    target_image,
    query_image,
    max_objects: int = 25,
) -> list[dict]:
    """Run detection using a query image as the prompt instead of a text label.

    Returns a list of {x_min, y_min, x_max, y_max} dicts in [0, 1] coords.
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
# Validation
# ============================================================================


def _xywh_to_corner_dict(box) -> dict:
    x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h}


def _plot_vp_prediction(
    query_crop, target_image, gt_dicts, pred_dicts, class_name, tp, fp, fn,
):
    """Side-by-side figure: query crop on left, target with GT (green) + preds (magenta) on right."""
    fig, (ax_q, ax_t) = plt.subplots(1, 2, figsize=(12, 6))
    ax_q.imshow(query_crop)
    ax_q.set_title(f"query: {class_name}")
    ax_q.axis("off")

    ax_t.imshow(target_image)
    W, H = target_image.size
    for b in gt_dicts:
        ax_t.add_patch(Rectangle(
            (b["x_min"] * W, b["y_min"] * H),
            (b["x_max"] - b["x_min"]) * W, (b["y_max"] - b["y_min"]) * H,
            fill=False, edgecolor="lime", linewidth=2, label="GT",
        ))
    for b in pred_dicts:
        ax_t.add_patch(Rectangle(
            (b["x_min"] * W, b["y_min"] * H),
            (b["x_max"] - b["x_min"]) * W, (b["y_max"] - b["y_min"]) * H,
            fill=False, edgecolor="magenta", linewidth=2, linestyle="--", label="pred",
        ))
    ax_t.set_title(
        f"GT={len(gt_dicts)}  pred={len(pred_dicts)}  TP={tp}  FP={fp}  FN={fn}"
    )
    ax_t.axis("off")
    plt.tight_layout()
    return fig


def validate_visual_prompt(
    model: MoondreamModel,
    val_ds: LVISVisualPrompting,
    step: int,
    max_samples: int = 250,
    iou_threshold: float = 0.5,
    max_plot_samples: int = 16,
    log_to_wandb: bool = True,
) -> dict:
    """F1 / precision / recall against GT, using detect_with_visual_prompt.

    Also saves up to ``max_plot_samples`` side-by-side prediction figures and
    logs them to wandb under the ``predictions/vp`` key (when ``log_to_wandb``).
    """
    model.eval()
    n = min(max_samples, len(val_ds))
    TP = FP = FN = 0
    images: list = []
    with torch.no_grad():
        for i in tqdm(range(n), desc=f"validate@step={step}", leave=False):
            sample = val_ds[i]
            preds = detect_with_visual_prompt(
                model, sample["target_image"], sample["query_crop"],
            )
            gt_dicts = [_xywh_to_corner_dict(b) for b in sample["boxes"]]
            tp, fp, fn = match_boxes_score(preds, gt_dicts, iou_threshold=iou_threshold)
            TP += tp; FP += fp; FN += fn

            if i < max_plot_samples:
                try:
                    fig = _plot_vp_prediction(
                        sample["query_crop"], sample["target_image"],
                        gt_dicts, preds, sample["class_name"], tp, fp, fn,
                    )
                    fig_path = f"predictions/vp_step{step}_{i:03d}.png"
                    fig.savefig(fig_path, dpi=90, bbox_inches="tight")
                    plt.close(fig)
                    if log_to_wandb:
                        images.append(wandb.Image(fig_path, caption=sample["class_name"]))
                except Exception as e:
                    logging.warning(f"failed to save prediction figure {i}: {e}")

    if log_to_wandb and images:
        try:
            wandb.log({"predictions/vp": images}, step=step)
        except Exception as e:
            logging.warning(f"failed to log prediction images to wandb: {e}")

    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": TP, "fp": FP, "fn": FN}


# ============================================================================
# Main
# ============================================================================


def main(
    lr: float = 5e-5,
    epochs: int = 5,
    grad_accum_steps: int = 64,
    validation_samples: int = 250,
    eval_interval: int = 5,
    val_plot_samples: int = 16,
    overfit_batch_size: Optional[int] = None,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    text_lora_targets: Optional[list] = None,
    proj_mlp_lora_rank: int = 8,
    proj_mlp_lora_alpha: int = 16,
    proj_mlp_lora_dropout: float = 0.0,
    max_objects_per_sample: int = 10,
    # Dataset
    dataset_name: str = "Voxel51/LVIS",
    max_samples: int = 5000,
    num_triplets_per_epoch: int = 10_000,
    hold_out_classes: bool = True,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    split_seed: int = 0,
    # IO
    model_path: str = "moondream2/model.safetensors",
    wandb_project: str = "moondream-visual-prompt-ft",
):
    """Train Moondream 2 with visual prompts via LoRA on text decoder + proj_mlp."""
    if text_lora_targets is None:
        text_lora_targets = ["qkv", "proj", "fc1", "fc2"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Quiet noisy third-party loggers — httpx in particular spams a line per
    # HTTP request during the first FiftyOne download, swamping training logs.
    for noisy in ("httpx", "httpcore", "urllib3", "fiftyone"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    os.makedirs("predictions", exist_ok=True)
    os.makedirs("model_artifacts", exist_ok=True)

    wandb.init(
        project=wandb_project,
        config=dict(
            EPOCHS=epochs, GRAD_ACCUM_STEPS=grad_accum_steps, LR=lr,
            VALIDATION_SAMPLES=validation_samples, EVAL_INTERVAL=eval_interval,
            OVERFIT_BATCH_SIZE=overfit_batch_size,
            LORA_RANK=lora_rank, LORA_ALPHA=lora_alpha, LORA_DROPOUT=lora_dropout,
            TEXT_LORA_TARGETS=text_lora_targets,
            PROJ_MLP_LORA_RANK=proj_mlp_lora_rank,
            PROJ_MLP_LORA_ALPHA=proj_mlp_lora_alpha,
            MAX_OBJECTS_PER_SAMPLE=max_objects_per_sample,
            DATASET=dataset_name, MAX_SAMPLES=max_samples,
            NUM_TRIPLETS_PER_EPOCH=num_triplets_per_epoch,
            HOLD_OUT_CLASSES=hold_out_classes,
            md_version="2", trainer="visual_prompt_teacher_forced",
        ),
    )

    # Build and load the base model.
    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    model.load_state_dict(load_file(model_path))
    model.to(device)
    for _, buf in model.named_buffers():
        buf.data = buf.data.to(device)

    # Inject LoRA on the text decoder + on vision.proj_mlp.
    inject_lora_into_model(
        model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
        target_modules=text_lora_targets,
    )
    inject_lora_into_proj_mlp(
        model, rank=proj_mlp_lora_rank, alpha=proj_mlp_lora_alpha,
        dropout=proj_mlp_lora_dropout,
    )

    # Freeze everything, then re-enable: LoRA params + region head.
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    for p in model.region.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters:    {total_params:,}")
    logging.info(f"Trainable params:    {trainable_params:,}")
    logging.info(f"Trainable ratio:     {100 * trainable_params / total_params:.3f}%")

    optimizer = AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad]}], lr=lr,
    )

    # Datasets — load the FiftyOne dataset *once* and share it across splits.
    fo_dataset = load_lvis_fo(dataset_name, max_samples=max_samples)
    common = dict(
        dataset_name=dataset_name, max_samples=max_samples,
        hold_out_classes=hold_out_classes, train_ratio=train_ratio,
        val_ratio=val_ratio, split_seed=split_seed,
        fo_dataset=fo_dataset,
    )
    train_ds = LVISVisualPrompting(
        split="train", num_triplets_per_epoch=num_triplets_per_epoch, **common,
    )
    val_ds = LVISVisualPrompting(
        split="val", num_triplets_per_epoch=validation_samples, **common,
    )
    test_ds = LVISVisualPrompting(
        split="test", num_triplets_per_epoch=validation_samples, **common,
    )

    if overfit_batch_size is not None and overfit_batch_size > 0:
        logging.info(
            f"Overfit mode: training and validating on the first "
            f"{overfit_batch_size} sampled triplets."
        )
        train_ds.num_triplets = overfit_batch_size
        val_ds.num_triplets = overfit_batch_size
        test_ds.num_triplets = overfit_batch_size

    logging.info(f"Train classes: {len(train_ds.classes)} (split={train_ds.split})")
    logging.info(f"Val   classes: {len(val_ds.classes)} (split={val_ds.split})")
    logging.info(f"Test  classes: {len(test_ds.classes)} (split={test_ds.split})")
    if hold_out_classes:
        overlap = set(train_ds.classes) & set(val_ds.classes)
        if overlap:
            logging.warning(f"train/val class overlap (should be empty): {len(overlap)}")

    # Initial validation (on novel classes if hold_out_classes=True).
    initial = validate_visual_prompt(
        model, val_ds, step=0, max_samples=validation_samples,
        max_plot_samples=val_plot_samples,
    )
    best_f1 = initial["f1"]
    best_step = 0
    logging.info(f"Initial val f1: {initial['f1']:.4f}  (P={initial['precision']:.3f}, R={initial['recall']:.3f})")
    wandb.log(
        {f"initial/{k}": v for k, v in initial.items()},
        step=0,
    )

    total_steps = max(1, epochs * len(train_ds) // grad_accum_steps)
    pbar = tqdm(total=total_steps)

    model.train()
    i = 0
    accum_loss_sum = 0.0
    accum_loss_n = 0
    for epoch in range(epochs):
        for sample in train_ds:
            i += 1
            loss = teacher_forced_visual_prompt_loss(
                model,
                sample["query_crop"],
                sample["target_image"],
                sample["boxes"],
                max_objects=max_objects_per_sample,
            )
            (loss / grad_accum_steps).backward()
            accum_loss_sum += float(loss.item())
            accum_loss_n += 1

            if i % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                step = i // grad_accum_steps
                lr_val = lr_schedule(step, total_steps, base_lr=lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_val

                avg_loss = accum_loss_sum / max(accum_loss_n, 1)
                accum_loss_sum, accum_loss_n = 0.0, 0
                pbar.set_postfix({"step": step, "loss": f"{avg_loss:.4f}"})
                pbar.update(1)
                wandb.log(
                    {"loss/train": avg_loss, "lr": lr_val, "epoch": epoch},
                    step=step,
                )

                if step % eval_interval == 0 and step > 0:
                    val = validate_visual_prompt(
                        model, val_ds, step=step, max_samples=validation_samples,
                        max_plot_samples=val_plot_samples,
                    )
                    logging.info(
                        f"step={step}  val f1={val['f1']:.4f}  "
                        f"P={val['precision']:.3f}  R={val['recall']:.3f}"
                    )
                    wandb.log(
                        {f"val/{k}": v for k, v in val.items()},
                        step=step,
                    )

                    if val["f1"] > best_f1:
                        best_f1 = val["f1"]
                        best_step = step
                        save_file(
                            get_lora_state_dict(model, include_region=True),
                            f"model_artifacts/moondream_visual_prompt_lora_step_{step}.safetensors",
                        )
                        logging.info(
                            f"saved best LoRA+region adapter at step {step} (f1={best_f1:.4f})"
                        )
    pbar.close()

    # Reload best, run on test split.
    if best_step > 0:
        path = f"model_artifacts/moondream_visual_prompt_lora_step_{best_step}.safetensors"
        model.load_state_dict(load_file(path), strict=False)
        logging.info(f"loaded best adapter from {path}")

    model.eval()
    test = validate_visual_prompt(
        model, test_ds, step=best_step, max_samples=validation_samples,
        max_plot_samples=val_plot_samples,
    )
    logging.info(f"test f1 (best step {best_step}): {test['f1']:.4f}")
    wandb.log({f"test/{k}": v for k, v in test.items()}, step=best_step + 1, commit=True)
    wandb.run.summary["test/f1"] = test["f1"]
    wandb.run.summary["test/precision"] = test["precision"]
    wandb.run.summary["test/recall"] = test["recall"]
    wandb.run.summary["best_validation_step"] = best_step

    save_file(
        get_lora_state_dict(model, include_region=True),
        "model_artifacts/moondream_visual_prompt_lora_finetune.safetensors",
    )
    logging.info("saved final adapter to model_artifacts/moondream_visual_prompt_lora_finetune.safetensors")
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

"""
Teacher-forced region fine-tuning that follows the generative detection path.

Instead of supervising the region head on hidden states from a synthetic sequence
(`_produce_hidden`), this trainer:
- Uses `encode_image_grad` + `_prefill_prompt_grad` to set up caches exactly
  like `detect` / `detect_grad`.
- Steps the decoder with teacher-forced ground-truth centers and sizes,
  and applies cross-entropy on the coordinate / size logits at the same
  points where inference uses them.

This is designed for Moondream 2 and the basketball detection dataset, and
supports LoRA-only fine-tuning.

Usage Examples:
---------------
Basic training with default hyperparameters:
    python sft_trainer.py

Training with custom learning rate and epochs:
    python sft_trainer.py --lr=1e-5 --epochs=5

Training with LoRA enabled:
    python sft_trainer.py --use_lora=True --lora_rank=32

Training with custom gradient accumulation:
    python sft_trainer.py --grad_accum_steps=16 --eval_interval=10

Training with overfitting mode:
    python sft_trainer.py --overfit_batch_size=4 --epochs=5 --use_lora=True --grad_accum_steps=4 --eval_interval=1
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm
from safetensors.torch import load_file, save_file
import wandb
import fire

from datasets.basketball_dataset import BasketballDetection
from trainer_helpers import (
    LoRALinear,
    inject_lora_into_model,
    get_lora_state_dict,
    lr_schedule,
    coord_to_bin,
    size_to_bin,
    bin_to_size,
    validate,
    validate_with_gt,
)

from moondream2.moondream import MoondreamModel, MoondreamConfig
from moondream2.moondream_functions import encode_image_grad, _prefill_prompt_grad
from moondream2.region import (
    decode_coordinate,
    decode_size,
    encode_coordinate,
    encode_size,
)

device = "cuda" if torch.cuda.is_available() else "mps"


def teacher_forced_region_loss(
    model: MoondreamModel,
    image,
    class_name: str,
    boxes: torch.Tensor,
    max_objects: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute region loss by following the same decode path as `detect` but with
    teacher-forced ground-truth centers and sizes.

    Args:
        model: MoondreamModel (Moondream 2).
        image: PIL.Image for the sample.
        class_name: Object string used for the detect prompt.
        boxes: Tensor of shape (N, 4) with [x_min, y_min, w, h] in [0, 1].
        max_objects: Optional cap on number of GT boxes to supervise.
    """
    if boxes.numel() == 0:
        return torch.zeros([], device=model.device)

    # Use the same detect template as inference
    detect_template = model.config.tokenizer.templates["detect"]
    object_tokens = model.tokenizer.encode(" " + class_name).ids
    instruction_token_ids = (
        detect_template["prefix"] + object_tokens + detect_template["suffix"]
    )

    # Encode image and prefill caches with BOS + image
    with torch.no_grad():
        encoded_image = encode_image_grad(model, image, settings=None)
    model.load_encoded_image(encoded_image)
    pos = encoded_image.pos

    # Prefill detect prompt (no teacher forcing yet)
    prompt_tokens = torch.tensor(
        [instruction_token_ids],
        dtype=torch.long,
        device=model.device,
    )
    logits_BV, hidden_BC, next_token, pos = _prefill_prompt_grad(
        model,
        prompt_tokens,
        pos,
        temperature=0.0,
        top_p=0.0,
        spatial_refs=None,
        attn_mask=None,
    )
    # Hidden state corresponding to the last prompt token
    hidden = hidden_BC[:, -1:, :]

    # Attention mask used for subsequent one-token decode steps
    mask = torch.zeros(1, 1, 2048, device=model.device, dtype=torch.bool)
    mask[:, :, :pos] = 1

    if max_objects is None:
        max_objects = boxes.size(0)
    n_objects = min(boxes.size(0), max_objects)

    total_loss = torch.zeros([], device=model.device)
    n_terms = 0

    for obj_idx in range(n_objects):
        bb = boxes[obj_idx].to(model.device)  # [x_min, y_min, w, h]
        x_min, y_min, w_box, h_box = bb

        # Convert to center coordinates
        x_center = torch.clamp(x_min + w_box / 2.0, 0.0, 1.0)
        y_center = torch.clamp(y_min + h_box / 2.0, 0.0, 1.0)

        # ----- X coordinate step -----
        x_logits = decode_coordinate(hidden, model.region)  # [1, 1, 1024]
        x_bin = coord_to_bin(float(x_center))
        x_target = torch.tensor([x_bin], dtype=torch.long, device=model.device)
        loss_x = F.cross_entropy(
            x_logits.view(-1, x_logits.size(-1)),
            x_target,
        )
        total_loss = total_loss + loss_x
        n_terms += 1

        # Teacher-force x embedding and advance decoder (y will see this)
        # Shape here should match Moondream's own `_generate_points_grad`:
        # x_center.unsqueeze(-1) -> (1, 1, 1), then encode_coordinate -> (1, 1, dim)
        x_center_tensor = (
            x_center.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(dtype=x_logits.dtype, device=model.device)
        )
        next_emb = encode_coordinate(x_center_tensor, model.region)
        mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = model._decode_one_tok(next_emb, mask, pos_ids)
        pos += 1

        # ----- Y coordinate step -----
        y_logits = decode_coordinate(hidden, model.region)
        y_bin = coord_to_bin(float(y_center))
        y_target = torch.tensor([y_bin], dtype=torch.long, device=model.device)
        loss_y = F.cross_entropy(
            y_logits.view(-1, y_logits.size(-1)),
            y_target,
        )
        total_loss = total_loss + loss_y
        n_terms += 1

        # Teacher-force y embedding and advance decoder (size will see this)
        y_center_tensor = (
            y_center.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(dtype=y_logits.dtype, device=model.device)
        )
        next_emb = encode_coordinate(y_center_tensor, model.region)
        mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = model._decode_one_tok(next_emb, mask, pos_ids)
        pos += 1

        # ----- Size step (log-scale bins for w, h) -----
        size_logits = decode_size(hidden, model.region)  # (2, 1024)
        w_bin = size_to_bin(float(w_box))
        h_bin = size_to_bin(float(h_box))
        w_target = torch.tensor([w_bin], dtype=torch.long, device=model.device)
        h_target = torch.tensor([h_bin], dtype=torch.long, device=model.device)

        loss_w = F.cross_entropy(size_logits[0].unsqueeze(0), w_target)
        loss_h = F.cross_entropy(size_logits[1].unsqueeze(0), h_target)
        total_loss = total_loss + loss_w + loss_h
        n_terms += 2

        # Teacher-force size embedding and take one LM step to start next object
        w_val = bin_to_size(w_bin)
        h_val = bin_to_size(h_bin)
        size_tensor = torch.tensor(
            [w_val, h_val],
            device=model.device,
            dtype=size_logits.dtype,
        )
        next_emb = encode_size(size_tensor, model.region).unsqueeze(0).unsqueeze(0)
        mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = model._decode_one_tok(next_emb, mask, pos_ids)
        pos += 1

    if n_terms == 0:
        return torch.zeros([], device=model.device)
    return total_loss / n_terms


def main(
    lr: float = 5e-5,
    epochs: int = 5,
    grad_accum_steps: int = 64,
    validation_samples: int = 250,
    eval_interval: int = 5,
    overfit_batch_size: Optional[int] = None,
    use_lora: bool = True,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    lora_target_modules: list = None,
    model_path: str = "moondream2/model.safetensors",
    wandb_project: str = "moondream-basketball-ft",
    dataset_name: str = "basketball-player-detection",
):
    """
    Main training function with configurable hyperparameters via Fire CLI.

    Args:
        lr: Learning rate (default: 5e-5, optimal range: 5e-5 to 1e-4)
        epochs: Number of training epochs (default: 5, optimal range: 3-5)
        grad_accum_steps: Gradient accumulation steps (default: 64)
        validation_samples: Number of samples to use for validation (default: 250)
        eval_interval: Evaluate every N gradient accumulation steps (default: 5)
        overfit_batch_size: Set to > 0 to overfit on a tiny subset (default: None)
        use_lora: Whether to use LoRA instead of full fine-tuning (default: True)
        lora_rank: Rank of LoRA matrices (default: 32, optimal: 32 or 64)
        lora_alpha: Scaling factor for LoRA, typically 2x rank (default: 64)
        lora_dropout: Dropout for LoRA layers (default: 0.1, optimal: 0.0 or 0.1)
        lora_target_modules: Which layers to apply LoRA to (default: ["qkv", "proj", "fc1", "fc2"])
        model_path: Path to model safetensors file (default: "moondream2/model.safetensors")
        wandb_project: Weights & Biases project name (default: "moondream-basketball-ft")
        dataset_name: Dataset name for wandb logging (default: "basketball-player-detection")
    """
    # Set default lora_target_modules if None
    if lora_target_modules is None:
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("model_artifacts", exist_ok=True)

    wandb.init(
        project=wandb_project,
        config={
            "EPOCHS": epochs,
            "GRAD_ACCUM_STEPS": grad_accum_steps,
            "LR": lr,
            "VALIDATION_SAMPLES": validation_samples,
            "EVAL_INTERVAL": eval_interval,
            "OVERFIT_BATCH_SIZE": overfit_batch_size,
            "USE_LORA": use_lora,
            "LORA_RANK": lora_rank if use_lora else None,
            "LORA_ALPHA": lora_alpha if use_lora else None,
            "LORA_DROPOUT": lora_dropout if use_lora else None,
            "dataset": dataset_name,
            "md_version": "2",
            "trainer": "teacher_forced",
        },
    )

    # Build and load Moondream 2
    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.to(device)

    # Ensure all buffers are on the right device
    for _, buffer in model.named_buffers():
        buffer.data = buffer.data.to(device)

    # Apply LoRA if enabled
    if use_lora:
        logging.info("Applying LoRA adapters to text model...")
        inject_lora_into_model(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )

        # Freeze all base parameters, then unfreeze only LoRA weights and region parameters
        for param in model.parameters():
            param.requires_grad = False
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
        for param in model.region.parameters():
            param.requires_grad = True
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable LoRA and region parameters: {trainable_params:,}")
        logging.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        optimizer = AdamW(
            [{"params": [p for p in model.parameters() if p.requires_grad]}],
            lr=lr,
        )
    else:
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Total trainable parameters (full fine-tuning): {total_params:,}")
        optimizer = AdamW(model.parameters(), lr=lr)

    # Datasets (basketball player detection)
    dataset = BasketballDetection(split="train")
    val_dataset = BasketballDetection(split="val")
    test_dataset = BasketballDetection(split="test")

    # Overfit subset handling
    if overfit_batch_size is not None and overfit_batch_size > 0:
        logging.info(
            f"Overfitting mode: using first {overfit_batch_size} training samples for both train and val"
        )
        train_full = BasketballDetection(split="train")
        train_full.dataset.image_ids = train_full.dataset.image_ids[:overfit_batch_size]
        dataset = train_full

        val_full = BasketballDetection(split="train")
        val_full.dataset.image_ids = val_full.dataset.image_ids[:overfit_batch_size]
        val_dataset = val_full

        test_full = BasketballDetection(split="train")
        test_full.dataset.image_ids = test_full.dataset.image_ids[:overfit_batch_size]
        test_dataset = test_full

    logging.info(f"Train dataset size: {len(dataset)}")
    logging.info(f"Val dataset size: {len(val_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    # Initial validation
    gt_validation_score = validate_with_gt(val_dataset, max_samples=validation_samples)
    logging.info(f"GT validation f1: {round(gt_validation_score['f1'], 4)}")

    initial_validation_score = validate(
        model, val_dataset, step=0, max_samples=validation_samples
    )

    best_validation_score = initial_validation_score["f1"]
    best_validation_step = 0
    logging.info(f"Initial validation f1: {round(initial_validation_score['f1'], 4)}")

    initial_test_score = validate(
        model, test_dataset, step=0, max_samples=validation_samples
    )

    wandb.log(
        {
            "gt_validation_f1": gt_validation_score["f1"],
            "gt_validation_precision": gt_validation_score["precision"],
            "gt_validation_recall": gt_validation_score["recall"],
            "initial_validation_f1": initial_validation_score["f1"],
            "initial_validation_precision": initial_validation_score["precision"],
            "initial_validation_recall": initial_validation_score["recall"],
            "initial_test_f1": initial_test_score["f1"],
            "initial_test_precision": initial_test_score["precision"],
            "initial_test_recall": initial_test_score["recall"],
        },
        step=0,
    )

    total_steps = epochs * len(dataset) // grad_accum_steps
    pbar = tqdm(total=total_steps)

    model.train()
    i = 0
    for epoch in range(epochs):
        for sample in dataset:
            i += 1

            boxes_by_class = {}
            for box, cls in zip(sample["boxes"], sample["class_names"]):
                boxes_by_class.setdefault(cls, []).append(box)

            total_loss = None
            for class_name, boxes_list in boxes_by_class.items():
                boxes_tensor = torch.stack(
                    [bb.to(dtype=torch.float32) for bb in boxes_list]
                )
                loss = teacher_forced_region_loss(
                    model,
                    sample["image"],
                    class_name,
                    boxes_tensor,
                )
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss

            if total_loss is not None:
                (total_loss / max(len(boxes_by_class), 1)).backward()

            if i % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i // grad_accum_steps, total_steps, base_lr=lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val

                current_step = i // grad_accum_steps
                loss_val = (
                    total_loss.item() / max(len(boxes_by_class), 1)
                    if total_loss is not None
                    else 0.0
                )
                pbar.set_postfix({"step": current_step, "loss": loss_val})
                pbar.update(1)

                wandb.log(
                    {
                        "loss/train": loss_val,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=current_step,
                )

                # Periodic validation
                if current_step % eval_interval == 0 and current_step > 0:
                    logging.info(f"Evaluating at step {current_step}")
                    validation_score = validate(
                        model,
                        val_dataset,
                        step=current_step,
                        max_samples=validation_samples,
                    )
                    logging.info(f"Validation f1: {round(validation_score['f1'], 4)}")

                    wandb.log(
                        {
                            "validation_f1": validation_score["f1"],
                            "validation_precision": validation_score["precision"],
                            "validation_recall": validation_score["recall"],
                        },
                        step=current_step,
                    )

                    # Save best model (LoRA + region if enabled, otherwise full model)
                    if validation_score["f1"] > best_validation_score:
                        best_validation_score = validation_score["f1"]
                        best_validation_step = current_step
                        if use_lora:
                            save_file(
                                get_lora_state_dict(model, include_region=True),
                                f"model_artifacts/moondream_lora_best_step_{current_step}.safetensors",
                            )
                            logging.info(
                                f"Saved best LoRA adapter + region model at step {current_step} with F1: {round(best_validation_score, 4)}"
                            )
                        else:
                            save_file(
                                model.state_dict(),
                                f"model_artifacts/moondream_best_step_{current_step}.safetensors",
                            )
                            logging.info(
                                f"Saved best full model at step {current_step} with F1: {round(best_validation_score, 4)}"
                            )
    pbar.close()

    # Load and test the best model
    if best_validation_step > 0:
        logging.info(f"Loading best model from step {best_validation_step}")
        if use_lora:
            # Load best LoRA weights + region model
            best_state_dict = load_file(
                f"model_artifacts/moondream_lora_best_step_{best_validation_step}.safetensors"
            )
            model.load_state_dict(best_state_dict, strict=False)
            logging.info(
                f"Loaded best LoRA adapter + region model from step {best_validation_step}"
            )
        else:
            # Load best full model
            best_state_dict = load_file(
                f"model_artifacts/moondream_best_step_{best_validation_step}.safetensors"
            )
            model.load_state_dict(best_state_dict)
            logging.info(f"Loaded best full model from step {best_validation_step}")
    else:
        logging.info(
            "Skipping best model load: best_validation_step is 0 (no improvement over initial validation)"
        )

    # Run final test on the best model
    model.eval()
    test_score = validate(
        model,
        test_dataset,
        step=best_validation_step,
        max_samples=validation_samples,
    )
    logging.info(
        f"Test f1 (best model from step {best_validation_step}): {round(test_score['f1'], 4)}"
    )

    # Log final test results to both step log and summary
    # Using "test/" prefix so they appear in their own graph panel
    final_step = best_validation_step + 1
    wandb.log(
        {
            "test/f1": test_score["f1"],
            "test/precision": test_score["precision"],
            "test/recall": test_score["recall"],
        },
        step=final_step,
        commit=True,
    )

    # Also save to run summary for easy access
    wandb.run.summary["test/f1"] = test_score["f1"]
    wandb.run.summary["test/precision"] = test_score["precision"]
    wandb.run.summary["test/recall"] = test_score["recall"]
    wandb.run.summary["best_validation_step"] = best_validation_step

    wandb.finish()

    # Final checkpoint
    if use_lora:
        save_file(
            get_lora_state_dict(model, include_region=True),
            "model_artifacts/moondream_lora_finetune.safetensors",
        )
        logging.info(
            "Saved final LoRA adapter + region model to model_artifacts/moondream_lora_finetune.safetensors"
        )
    else:
        save_file(
            model.state_dict(),
            "model_artifacts/moondream_finetune.safetensors",
        )
        logging.info(
            "Saved final model to model_artifacts/moondream_finetune.safetensors"
        )


if __name__ == "__main__":
    """
    Run with Fire CLI. Examples:
        python sft_trainer.py
        python sft_trainer.py --lr=1e-5 --epochs=5
        python sft_trainer.py --use_lora=True
    """
    fire.Fire(main)

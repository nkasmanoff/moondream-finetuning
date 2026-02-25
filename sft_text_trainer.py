"""
Teacher-forced text fine-tuning for Moondream's query (VQA) capability
on Mario Kart frame data (scene / position / coins tasks).

Instead of supervising the region head on coordinate/size bins (like the
detection trainer), this trainer:
- Uses `encode_image_grad` to encode the image into KV caches.
- Builds the full query prompt + ground-truth answer as a single token sequence.
- Runs the entire sequence through the decoder in one forward pass.
- Applies cross-entropy loss only on the answer token positions.

This is designed for Moondream 2 and supports LoRA-only fine-tuning.

Usage Examples:
---------------
Basic training (requires Supabase env vars):
    python sft_text_trainer.py

Training specific tasks:
    python sft_text_trainer.py --tasks='["scene","position"]'

Training with custom learning rate and epochs:
    python sft_text_trainer.py --lr=1e-5 --epochs=5

Training with overfitting mode:
    python sft_text_trainer.py --overfit_batch_size=4 --epochs=5 --use_lora=True --grad_accum_steps=4 --eval_interval=1
"""

import logging
import os
import random
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from safetensors.torch import load_file, save_file
import wandb
import fire

from trainer_helpers import (
    LoRALinear,
    inject_lora_into_model,
    get_lora_state_dict,
    lr_schedule,
)

from moondream2.moondream import MoondreamModel, MoondreamConfig
from moondream2.moondream_functions import encode_image_grad, _prefill
from moondream2.text import text_encoder, _lm_head
from kart_dataset import KartSceneDataset

device = "cuda" if torch.cuda.is_available() else "mps"


def train_val_test_split(
    full_dataset: KartSceneDataset,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Split a KartSceneDataset into train / val / test by shuffling samples."""
    samples = list(full_dataset.samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_samples = samples[:n_test]
    val_samples = samples[n_test : n_test + n_val]
    train_samples = samples[n_test + n_val :]

    return (
        KartSceneDataset(train_samples),
        KartSceneDataset(val_samples),
        KartSceneDataset(test_samples),
    )


def teacher_forced_text_loss(
    model: MoondreamModel,
    image,
    question: str,
    answer: str,
) -> torch.Tensor:
    """
    Compute teacher-forced cross-entropy loss on answer tokens.

    The full sequence (query prompt + ground-truth answer + EOS) is processed
    in a single forward pass, and loss is computed only on answer positions.

    Args:
        model: MoondreamModel (Moondream 2).
        image: PIL.Image for the sample.
        question: The question string.
        answer: The ground-truth answer string.
    """
    query_template = model.config.tokenizer.templates["query"]
    question_tokens = model.tokenizer.encode(question).ids
    answer_tokens = model.tokenizer.encode(answer).ids
    eos_id = model.config.tokenizer.eos_id

    prompt_ids = (
        query_template["prefix"]
        + question_tokens
        + query_template["suffix"]
        + query_template["suffix"]
    )
    full_ids = prompt_ids + answer_tokens + [eos_id]

    with torch.no_grad():
        encoded_image = encode_image_grad(model, image, settings=None)
    model.load_encoded_image(encoded_image)
    pos = encoded_image.pos

    full_tokens = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    prompt_emb = text_encoder(full_tokens, model.text)

    mask = model.attn_mask[:, :, pos : pos + prompt_emb.size(1), :]
    pos_ids = torch.arange(pos, pos + prompt_emb.size(1), dtype=torch.long)
    hidden = _prefill(model, prompt_emb, mask, pos_ids)

    logits = _lm_head(hidden, model.text)  # (1, seq_len, vocab_size)

    n_prompt = len(prompt_ids)
    answer_logits = logits[:, n_prompt - 1 : -1, :]
    answer_targets = full_tokens[:, n_prompt:]

    loss = F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.size(-1)),
        answer_targets.reshape(-1),
    )
    return loss


def validate_text(model, val_ds, step, max_samples=250):
    """
    Validate by generating answers with model.query() and computing
    exact-match accuracy against ground truth.

    Args:
        model: MoondreamModel to evaluate.
        val_ds: Validation dataset returning dicts with "image", "question", "answer".
        step: Current training step (for logging).
        max_samples: Maximum number of samples to evaluate.

    Returns:
        Dict with "accuracy", "correct", "total".
    """
    model.eval()
    correct = 0
    total = 0
    num_samples = min(max_samples, len(val_ds))

    with torch.no_grad():
        for i in range(num_samples):
            sample = val_ds[i]
            result = model.query(
                image=sample["image"],
                question=sample["question"],
                stream=False,
                settings={"max_tokens": 256, "temperature": 0.0},
            )
            predicted = result["answer"].strip().lower()
            gt = sample["answer"].strip().lower()

            if predicted == gt:
                correct += 1
            total += 1

    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


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
    wandb_project: str = "moondream-kart-text-ft",
    dataset_name: str = "kart-scene",
    tasks: Union[str, List[str]] = "scene",
    supabase_email: Optional[str] = None,
    supabase_password: Optional[str] = None,
    session_ids: Optional[List[str]] = None,
    image_kind: str = "hires",
    cache_dir: str = "./frame_cache",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 42,
):
    """
    Main training function for query/text fine-tuning with Fire CLI.

    Args:
        lr: Learning rate (default: 5e-5)
        epochs: Number of training epochs (default: 5)
        grad_accum_steps: Gradient accumulation steps (default: 64)
        validation_samples: Number of samples to use for validation (default: 250)
        eval_interval: Evaluate every N gradient accumulation steps (default: 5)
        overfit_batch_size: Set to > 0 to overfit on a tiny subset (default: None)
        use_lora: Whether to use LoRA instead of full fine-tuning (default: True)
        lora_rank: Rank of LoRA matrices (default: 32)
        lora_alpha: Scaling factor for LoRA, typically 2x rank (default: 64)
        lora_dropout: Dropout for LoRA layers (default: 0.1)
        lora_target_modules: Which layers to apply LoRA to (default: ["qkv", "proj", "fc1", "fc2"])
        model_path: Path to model safetensors file
        wandb_project: Weights & Biases project name
        dataset_name: Dataset name for wandb logging
        tasks: Kart task(s) — "scene", "position", "coins", or a list (default: "scene")
        supabase_email: Supabase auth email (or SUPABASE_USER_EMAIL env var)
        supabase_password: Supabase auth password (or SUPABASE_USER_PASSWORD env var)
        session_ids: Limit to specific session UUIDs (default: all)
        image_kind: "hires" or "thumb" (default: "hires")
        cache_dir: Local directory to cache downloaded images (default: "./frame_cache")
        val_frac: Fraction of data for validation (default: 0.1)
        test_frac: Fraction of data for test (default: 0.1)
        split_seed: Random seed for train/val/test split (default: 42)
    """
    if lora_target_modules is None:
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
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
            "tasks": tasks,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "split_seed": split_seed,
            "md_version": "2",
            "trainer": "teacher_forced_text",
        },
    )

    # Build and load Moondream 2
    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.to(device)

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

        for param in model.parameters():
            param.requires_grad = False
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable LoRA parameters: {trainable_params:,}")
        logging.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        optimizer = AdamW(
            [{"params": [p for p in model.parameters() if p.requires_grad]}],
            lr=lr,
        )
    else:
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Total trainable parameters (full fine-tuning): {total_params:,}")
        optimizer = AdamW(model.parameters(), lr=lr)

    # Load Kart dataset from Supabase
    logging.info(f"Loading KartSceneDataset (tasks={tasks}, image_kind={image_kind})...")
    full_dataset = KartSceneDataset.from_supabase(
        email=supabase_email,
        password=supabase_password,
        tasks=tasks,
        image_kind=image_kind,
        cache_dir=cache_dir,
        session_ids=session_ids,
        skip_unlabeled=True,
        verbose=True,
    )
    logging.info(f"Total samples loaded: {len(full_dataset)}")

    dataset, val_dataset, test_dataset = train_val_test_split(
        full_dataset, val_frac=val_frac, test_frac=test_frac, seed=split_seed,
    )

    if overfit_batch_size is not None and overfit_batch_size > 0:
        logging.info(
            f"Overfitting mode: using first {overfit_batch_size} training samples"
        )
        dataset = KartSceneDataset(dataset.samples[:overfit_batch_size])
        val_dataset = KartSceneDataset(val_dataset.samples[:overfit_batch_size])
        test_dataset = KartSceneDataset(test_dataset.samples[:overfit_batch_size])

    logging.info(f"Train dataset size: {len(dataset)}")
    logging.info(f"Val dataset size: {len(val_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    # Initial validation
    initial_val = validate_text(
        model, val_dataset, step=0, max_samples=validation_samples
    )
    best_val_accuracy = initial_val["accuracy"]
    best_validation_step = 0
    logging.info(f"Initial validation accuracy: {best_val_accuracy:.4f}")

    initial_test = validate_text(
        model, test_dataset, step=0, max_samples=validation_samples
    )

    wandb.log(
        {
            "initial_val_accuracy": initial_val["accuracy"],
            "initial_test_accuracy": initial_test["accuracy"],
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

            loss = teacher_forced_text_loss(
                model,
                sample["image"],
                sample["question"],
                sample["answer"],
            )
            (loss / grad_accum_steps).backward()

            if i % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i // grad_accum_steps, total_steps, base_lr=lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val

                current_step = i // grad_accum_steps
                pbar.set_postfix({"step": current_step, "loss": loss.item()})
                pbar.update(1)

                wandb.log(
                    {
                        "loss/train": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=current_step,
                )

                if current_step % eval_interval == 0 and current_step > 0:
                    logging.info(f"Evaluating at step {current_step}")
                    val_score = validate_text(
                        model,
                        val_dataset,
                        step=current_step,
                        max_samples=validation_samples,
                    )
                    logging.info(
                        f"Validation accuracy: {val_score['accuracy']:.4f}"
                    )

                    wandb.log(
                        {"val_accuracy": val_score["accuracy"]},
                        step=current_step,
                    )

                    if val_score["accuracy"] > best_val_accuracy:
                        best_val_accuracy = val_score["accuracy"]
                        best_validation_step = current_step
                        if use_lora:
                            save_file(
                                get_lora_state_dict(model, include_region=False),
                                f"model_artifacts/moondream_lora_text_best_step_{current_step}.safetensors",
                            )
                            logging.info(
                                f"Saved best LoRA adapter at step {current_step} "
                                f"with accuracy: {best_val_accuracy:.4f}"
                            )
                        else:
                            save_file(
                                model.state_dict(),
                                f"model_artifacts/moondream_text_best_step_{current_step}.safetensors",
                            )
                            logging.info(
                                f"Saved best full model at step {current_step} "
                                f"with accuracy: {best_val_accuracy:.4f}"
                            )
    pbar.close()

    # Load and test the best model
    if best_validation_step > 0:
        logging.info(f"Loading best model from step {best_validation_step}")
        if use_lora:
            best_state_dict = load_file(
                f"model_artifacts/moondream_lora_text_best_step_{best_validation_step}.safetensors"
            )
            model.load_state_dict(best_state_dict, strict=False)
            logging.info(
                f"Loaded best LoRA adapter from step {best_validation_step}"
            )
        else:
            best_state_dict = load_file(
                f"model_artifacts/moondream_text_best_step_{best_validation_step}.safetensors"
            )
            model.load_state_dict(best_state_dict)
            logging.info(f"Loaded best full model from step {best_validation_step}")
    else:
        logging.info(
            "Skipping best model load: no improvement over initial validation"
        )

    # Final test
    model.eval()
    test_score = validate_text(
        model,
        test_dataset,
        step=best_validation_step,
        max_samples=validation_samples,
    )
    logging.info(
        f"Test accuracy (best model from step {best_validation_step}): "
        f"{test_score['accuracy']:.4f}"
    )

    final_step = best_validation_step + 1
    wandb.log(
        {"test/accuracy": test_score["accuracy"]},
        step=final_step,
        commit=True,
    )

    wandb.run.summary["test/accuracy"] = test_score["accuracy"]
    wandb.run.summary["best_validation_step"] = best_validation_step

    wandb.finish()

    # Final checkpoint
    if use_lora:
        save_file(
            get_lora_state_dict(model, include_region=False),
            "model_artifacts/moondream_lora_text_finetune.safetensors",
        )
        logging.info(
            "Saved final LoRA adapter to model_artifacts/moondream_lora_text_finetune.safetensors"
        )
    else:
        save_file(
            model.state_dict(),
            "model_artifacts/moondream_text_finetune.safetensors",
        )
        logging.info(
            "Saved final model to model_artifacts/moondream_text_finetune.safetensors"
        )


if __name__ == "__main__":
    """
    Run with Fire CLI. Examples:
        python sft_text_trainer.py
        python sft_text_trainer.py --lr=1e-5 --epochs=5
        python sft_text_trainer.py --use_lora=True
    """
    fire.Fire(main)

#!/bin/bash

# Hyperparameter sweep for sft_text_trainer.py on the scene task.
# Mix of LoRA and full fine-tuning runs.

set -e

WANDB_PROJECT="moondream-kart-text-sweep"
COMMON="--validation_samples=250 --eval_interval=5 --tasks=scene --wandb_project=${WANDB_PROJECT}"

echo "Starting scene text trainer sweep (${WANDB_PROJECT})"
echo "=============================================================="

# --- LoRA runs ---

echo ""
echo "[1/8] LoRA — LR=5e-5, rank=32"
python sft_text_trainer.py \
    --lr=5e-5 --epochs=5 --grad_accum_steps=64 \
    --use_lora=True --lora_rank=32 --lora_alpha=64 --lora_dropout=0.1 \
    ${COMMON}

echo ""
echo "[2/8] LoRA — LR=1e-4, rank=32"
python sft_text_trainer.py \
    --lr=1e-4 --epochs=5 --grad_accum_steps=64 \
    --use_lora=True --lora_rank=32 --lora_alpha=64 --lora_dropout=0.1 \
    ${COMMON}

echo ""
echo "[3/8] LoRA — LR=5e-5, rank=64"
python sft_text_trainer.py \
    --lr=5e-5 --epochs=5 --grad_accum_steps=64 \
    --use_lora=True --lora_rank=64 --lora_alpha=128 --lora_dropout=0.1 \
    ${COMMON}

echo ""
echo "[4/8] LoRA — LR=1e-4, rank=64, no dropout"
python sft_text_trainer.py \
    --lr=1e-4 --epochs=5 --grad_accum_steps=64 \
    --use_lora=True --lora_rank=64 --lora_alpha=128 --lora_dropout=0.0 \
    ${COMMON}

# --- Full fine-tuning runs ---

echo ""
echo "[5/8] Full FT — LR=1e-5"
python sft_text_trainer.py \
    --lr=1e-5 --epochs=3 --grad_accum_steps=64 \
    --use_lora=False \
    ${COMMON}

echo ""
echo "[6/8] Full FT — LR=3e-5"
python sft_text_trainer.py \
    --lr=3e-5 --epochs=3 --grad_accum_steps=64 \
    --use_lora=False \
    ${COMMON}

echo ""
echo "[7/8] Full FT — LR=5e-5"
python sft_text_trainer.py \
    --lr=5e-5 --epochs=3 --grad_accum_steps=64 \
    --use_lora=False \
    ${COMMON}

echo ""
echo "[8/8] Full FT — LR=1e-5, longer (5 epochs)"
python sft_text_trainer.py \
    --lr=1e-5 --epochs=5 --grad_accum_steps=64 \
    --use_lora=False \
    ${COMMON}

echo ""
echo "=============================================================="
echo "Sweep done! (8 runs) — project: ${WANDB_PROJECT}"
echo "=============================================================="

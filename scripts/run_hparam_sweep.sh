#!/bin/bash

# Hyperparameter sweep script for sft_trainer.py
# Run configurations sequentially to test different hyperparameter combinations

# Set to exit on error
set -e

echo "Starting hyperparameter sweep..."
echo "=================================="

# Configuration 1: Default settings
echo ""
echo "Config 1: Default settings"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 2: Lower learning rate
echo ""
echo "Config 2: Lower learning rate (1e-5)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 3: Higher learning rate
echo ""
echo "Config 3: Higher learning rate (1e-4)"
python sft_trainer.py \
    --lr=1e-4 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 4: Very low learning rate
echo ""
echo "Config 4: Very low learning rate (5e-6)"
python sft_trainer.py \
    --lr=5e-6 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 5: Lower LoRA rank
echo ""
echo "Config 5: Lower LoRA rank (16)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 6: Higher LoRA rank
echo ""
echo "Config 6: Higher LoRA rank (64)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 7: Very high LoRA rank
echo ""
echo "Config 7: Very high LoRA rank (128)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=128 \
    --lora_alpha=256 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 8: More epochs
echo ""
echo "Config 8: More epochs (10)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=10 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 9: Fewer epochs, higher LR
echo ""
echo "Config 9: Fewer epochs (3), higher LR (1e-4)"
python sft_trainer.py \
    --lr=1e-4 \
    --epochs=3 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 10: Lower gradient accumulation
echo ""
echo "Config 10: Lower gradient accumulation (32)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=32 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 11: Higher gradient accumulation
echo ""
echo "Config 11: Higher gradient accumulation (128)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=128 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 12: No LoRA dropout
echo ""
echo "Config 12: No LoRA dropout (0.0)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.0 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 13: Higher LoRA dropout
echo ""
echo "Config 13: Higher LoRA dropout (0.2)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.2 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 14: Lower LR + Higher rank
echo ""
echo "Config 14: Lower LR (1e-5) + Higher rank (64)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

# Configuration 15: Higher LR + Lower rank
echo ""
echo "Config 15: Higher LR (1e-4) + Lower rank (16)"
python sft_trainer.py \
    --lr=1e-4 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-player-detection

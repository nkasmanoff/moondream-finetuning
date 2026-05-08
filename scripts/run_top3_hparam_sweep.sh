#!/bin/bash

# Hyperparameter sweep script for sft_trainer.py
# Based on top 3 performing configurations from initial sweep
# Run configurations sequentially to test variations around the best performers

# Set to exit on error
set -e

echo "Starting hyperparameter sweep based on top 3 configurations..."
echo "=============================================================="

# Top 1: tough-snowball-9 - test/f1: 0.1991
# EPOCHS: 3, GRAD_ACCUM_STEPS: 64, LORA_ALPHA: 64, LORA_DROPOUT: 0.1, LORA_RANK: 32, LR: 0.0001
echo ""
echo "Config 1: Top performer - tough-snowball-9 (test/f1: 0.1991)"
echo "Variations around: EPOCHS=3, LR=1e-4, LORA_RANK=32, LORA_ALPHA=64, LORA_DROPOUT=0.1, GRAD_ACCUM=64"
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
    --wandb_project=moondream-basketball-ft-sweep-top3

# Variation 1a: More epochs (4) with same LR
echo ""
echo "Config 1a: Top 1 variation - More epochs (4)"
python sft_trainer.py \
    --lr=1e-4 \
    --epochs=4 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Variation 1b: Slightly lower LR (8e-5) with same epochs
echo ""
echo "Config 1b: Top 1 variation - Slightly lower LR (8e-5)"
python sft_trainer.py \
    --lr=8e-5 \
    --epochs=3 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Variation 1c: Higher LR (1.2e-4) with same epochs
echo ""
echo "Config 1c: Top 1 variation - Higher LR (1.2e-4)"
python sft_trainer.py \
    --lr=1.2e-4 \
    --epochs=3 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Top 2: vital-cherry-12 - test/f1: 0.1978
# EPOCHS: 5, GRAD_ACCUM_STEPS: 64, LORA_ALPHA: 64, LORA_DROPOUT: 0, LORA_RANK: 32, LR: 0.00005
echo ""
echo "Config 2: Second best - vital-cherry-12 (test/f1: 0.1978)"
echo "Variations around: EPOCHS=5, LR=5e-5, LORA_RANK=32, LORA_ALPHA=64, LORA_DROPOUT=0.0, GRAD_ACCUM=64"
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
    --wandb_project=moondream-basketball-ft-sweep-top3

# Variation 2a: More epochs (6) with same settings
echo ""
echo "Config 2a: Top 2 variation - More epochs (6)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=6 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.0 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Variation 2b: Slightly higher LR (6e-5) with same epochs
echo ""
echo "Config 2b: Top 2 variation - Slightly higher LR (6e-5)"
python sft_trainer.py \
    --lr=6e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.0 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Variation 2c: Add small dropout (0.05) with same settings
echo ""
echo "Config 2c: Top 2 variation - Small dropout (0.05)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.05 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Top 3: snowy-sunset-6 - test/f1: 0.1972
# EPOCHS: 5, GRAD_ACCUM_STEPS: 64, LORA_ALPHA: 128, LORA_DROPOUT: 0.1, LORA_RANK: 64, LR: 0.00005
echo ""
echo "Config 3: Third best - snowy-sunset-6 (test/f1: 0.1972)"
echo "Variations around: EPOCHS=5, LR=5e-5, LORA_RANK=64, LORA_ALPHA=128, LORA_DROPOUT=0.1, GRAD_ACCUM=64"
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
    --wandb_project=moondream-basketball-ft-sweep-top3

# Variation 3a: More epochs (6) with same settings
echo ""
echo "Config 3a: Top 3 variation - More epochs (6)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=6 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Variation 3b: Lower LR (4e-5) with same epochs
echo ""
echo "Config 3b: Top 3 variation - Lower LR (4e-5)"
python sft_trainer.py \
    --lr=4e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Variation 3c: Higher LR (6e-5) with same epochs
echo ""
echo "Config 3c: Top 3 variation - Higher LR (6e-5)"
python sft_trainer.py \
    --lr=6e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Cross-pollination experiments: Combining best aspects
echo ""
echo "=============================================================="
echo "Cross-pollination experiments: Combining best aspects"
echo "=============================================================="

# Hybrid 1: Top 1's LR (1e-4) + Top 2's epochs (5) + Top 2's no dropout
echo ""
echo "Hybrid 1: Top 1 LR (1e-4) + Top 2 epochs (5) + Top 2 no dropout"
python sft_trainer.py \
    --lr=1e-4 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.0 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Hybrid 2: Top 1's LR (1e-4) + Top 3's higher rank (64)
echo ""
echo "Hybrid 2: Top 1 LR (1e-4) + Top 3 rank (64) + Top 1 epochs (3)"
python sft_trainer.py \
    --lr=1e-4 \
    --epochs=3 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep-top3

# Hybrid 3: Top 2's settings + Top 3's higher rank
echo ""
echo "Hybrid 3: Top 2 settings + Top 3 rank (64)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.0 \
    --wandb_project=moondream-basketball-ft-sweep-top3

echo ""
echo "=============================================================="
echo "Hyperparameter sweep completed!"
echo "=============================================================="

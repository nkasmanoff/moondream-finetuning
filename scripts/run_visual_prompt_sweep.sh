#!/bin/bash

# ============================================================================
# Visual-prompt fine-tuning sweep.
#
# Runs `visual_prompt_trainer.py` with several hparam configurations, each
# tuned to make good use of the GPU (bigger LoRA adapters, more boxes per
# sample, more frequent optimizer steps). Every config writes to its own
# `model_artifacts/vp_sweep/<name>/` and `predictions/vp_sweep/<name>/` dir,
# so adapters and prediction figures don't overwrite across runs.
#
# Configs vary the most informative axes:
#   1. wide_proj_mlp        - emphasize the visual-prompt projection adapter
#                             (the ONLY trainable path from query_crop -> splice)
#   2. wide_text            - emphasize the text-decoder adapter
#   3. wide_both            - both adapters big (high capacity baseline)
#   4. wide_both_high_lr    - same as 3 with lr=1e-4 (faster optimization?)
#   5. fat_objects          - more boxes per sample (more KV-cache + per-step
#                             compute), medium adapters
#
# These are sequential. With `set -e`, a failure in one config aborts the
# sweep. Comment out configs you don't want, or copy-paste one config to run
# it standalone.
#
# Each config takes roughly the same wall clock as the baseline default run
# (~10h on an L40S at the time of writing). Expect ~2 days for the full sweep.
#
# IMPORTANT: this script assumes no other training jobs are already on the
# GPU. The high-capacity configs alone push memory well past 25 GB.
# ============================================================================

set -e

WANDB_PROJECT=moondream-visual-prompt-ft-sweep
ROOT_ARTIFACTS=model_artifacts/vp_sweep
ROOT_PREDS=predictions/vp_sweep

mkdir -p "${ROOT_ARTIFACTS}" "${ROOT_PREDS}"

echo "Starting visual-prompt sweep..."
echo "Artifacts root:   ${ROOT_ARTIFACTS}"
echo "Predictions root: ${ROOT_PREDS}"
echo "Wandb project:    ${WANDB_PROJECT}"
echo "=========================================="

run_config() {
    local name=$1
    shift
    echo ""
    echo "=========================================="
    echo "Config: ${name}"
    echo "Started:  $(date)"
    echo "=========================================="
    python visual_prompt_trainer.py \
        --run_name="${name}" \
        --artifact_dir="${ROOT_ARTIFACTS}/${name}" \
        --predictions_dir="${ROOT_PREDS}/${name}" \
        --wandb_project="${WANDB_PROJECT}" \
        "$@"
    echo "Finished: $(date)  (config: ${name})"
}

# ----------------------------------------------------------------------------
# Config 1: wide_proj_mlp
# Push the proj_mlp LoRA hard - the only path that shapes how query_crop
# embeddings get spliced into the prompt slot. Default-sized text adapter.
# ----------------------------------------------------------------------------
run_config wide_proj_mlp \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=32 \
    --eval_interval=10 \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.05 \
    --proj_mlp_lora_rank=64 \
    --proj_mlp_lora_alpha=128 \
    --proj_mlp_lora_dropout=0.0 \
    --max_objects_per_sample=20 \
    --num_triplets_per_epoch=10000

# ----------------------------------------------------------------------------
# Config 2: wide_text
# Push the text-decoder LoRA. Default-sized proj_mlp adapter. Lets us
# attribute gains to the text side vs the visual-prompt projection side.
# ----------------------------------------------------------------------------
run_config wide_text \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=32 \
    --eval_interval=10 \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.05 \
    --proj_mlp_lora_rank=8 \
    --proj_mlp_lora_alpha=16 \
    --proj_mlp_lora_dropout=0.0 \
    --max_objects_per_sample=20 \
    --num_triplets_per_epoch=10000

# ----------------------------------------------------------------------------
# Config 3: wide_both
# Both adapters big. High-capacity baseline; a likely winner if the model
# just needs more parameters.
# ----------------------------------------------------------------------------
run_config wide_both \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=32 \
    --eval_interval=10 \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.05 \
    --proj_mlp_lora_rank=32 \
    --proj_mlp_lora_alpha=64 \
    --proj_mlp_lora_dropout=0.0 \
    --max_objects_per_sample=20 \
    --num_triplets_per_epoch=10000

# ----------------------------------------------------------------------------
# Config 4: wide_both_high_lr
# Same capacity as wide_both with lr=1e-4. LoRA usually likes higher LR than
# full fine-tuning; tests whether the baseline 5e-5 is leaving juice on the
# table.
# ----------------------------------------------------------------------------
run_config wide_both_high_lr \
    --lr=1e-4 \
    --epochs=5 \
    --grad_accum_steps=32 \
    --eval_interval=10 \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.05 \
    --proj_mlp_lora_rank=32 \
    --proj_mlp_lora_alpha=64 \
    --proj_mlp_lora_dropout=0.0 \
    --max_objects_per_sample=20 \
    --num_triplets_per_epoch=10000

# ----------------------------------------------------------------------------
# Config 5: fat_objects
# Trade adapter width for per-sample compute. 25 boxes/sample = 75 extra
# decode steps per training example, exercising the KV cache and giving the
# region head a lot of supervision per image.
# ----------------------------------------------------------------------------
run_config fat_objects \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=32 \
    --eval_interval=10 \
    --lora_rank=48 \
    --lora_alpha=96 \
    --lora_dropout=0.05 \
    --proj_mlp_lora_rank=24 \
    --proj_mlp_lora_alpha=48 \
    --proj_mlp_lora_dropout=0.0 \
    --max_objects_per_sample=25 \
    --num_triplets_per_epoch=10000

echo ""
echo "=========================================="
echo "Sweep complete."
echo "Adapters under: ${ROOT_ARTIFACTS}"
echo "Figures under:  ${ROOT_PREDS}"
echo "=========================================="

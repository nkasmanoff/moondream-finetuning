# Model Artifacts Directory

This directory contains all saved model checkpoints from fine-tuning runs.

## Directory Structure

```
model_artifacts/
├── moondream_lora_best_step_<N>.safetensors    # Best LoRA + region checkpoint at step N
├── moondream_lora_finetune.safetensors          # Final LoRA + region checkpoint
├── moondream_best_step_<N>.safetensors          # Best full model checkpoint at step N (if not using LoRA)
└── moondream_finetune.safetensors               # Final full model checkpoint (if not using LoRA)
```

## File Types

### LoRA + Region Model Checkpoints

When training with `use_lora=True`, the saved checkpoints contain:

-   **LoRA adapter weights**: Low-rank adaptations applied to the text model's attention and MLP layers
-   **Region model weights**: Full weights of the region detection head (coordinate and size encoders/decoders)

These checkpoints are much smaller than full model saves and require the base model to be loaded first.

### Full Model Checkpoints

When training with `use_lora=False`, the saved checkpoints contain:

-   Complete model state including vision encoder, text model, and region model
-   Can be loaded directly without needing the base model

## Loading Saved Models

### Method 1: Using the utility script

```bash
# Load LoRA + region checkpoint (default)
python load_finetuned_model.py --checkpoint=model_artifacts/moondream_lora_best_step_100.safetensors

# Load full model checkpoint
python load_finetuned_model.py --checkpoint=model_artifacts/moondream_best_step_100.safetensors --use_lora=False
```

### Method 2: Programmatically

```python
from safetensors.torch import load_file
from moondream2.moondream import MoondreamModel, MoondreamConfig
from trainer_helpers import inject_lora_into_model

# For LoRA + region model
model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
state_dict = load_file("moondream2/model.safetensors")
model.load_state_dict(state_dict)
model.to("cuda")

# Inject LoRA structure (must match training config)
inject_lora_into_model(model, rank=32, alpha=64, dropout=0.1)

# Load fine-tuned weights
finetuned_state = load_file("model_artifacts/moondream_lora_best_step_100.safetensors")
model.load_state_dict(finetuned_state, strict=False)
model.eval()
```

## Important Notes

1. **Region Model**: The region model weights are ALWAYS saved with the checkpoint, whether using LoRA or not. This is critical because the region detection head is fine-tuned during training.

2. **LoRA Parameters**: When loading LoRA checkpoints, you must inject the LoRA structure with the same hyperparameters used during training (rank, alpha, dropout, target_modules).

3. **Strict Loading**:

    - For LoRA checkpoints, use `strict=False` since not all model parameters are in the checkpoint
    - For full model checkpoints, use `strict=True` (default) to ensure all weights are loaded

4. **Best vs Final**:
    - `*_best_step_N.safetensors`: Checkpoint with the best validation F1 score
    - `*_finetune.safetensors`: Final checkpoint at end of training

## Troubleshooting

### Missing region model parameters

If you see errors about missing region model parameters when loading:

-   Make sure you're using a checkpoint saved with the updated trainer that includes region weights
-   Checkpoints saved before the fix only contain LoRA weights, not region weights

### LoRA structure mismatch

If you see errors about mismatched tensor shapes:

-   Verify that the LoRA hyperparameters (rank, alpha, target_modules) match those used during training
-   Check the wandb run config or training logs for the correct hyperparameters

### Device mismatch

If you see CUDA/device errors:

-   Ensure all model components are on the same device after loading
-   Use `model.to(device)` and ensure buffers are moved with a loop over `named_buffers()`

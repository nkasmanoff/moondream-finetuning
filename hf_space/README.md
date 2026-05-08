---
title: Mario Kart Race Detector
emoji: 🏎️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
hardware: zero-a10g
---

# Mario Kart Race Detector API

Fine-tuned Moondream model that detects whether a screenshot shows an active Mario Kart race.

## API Usage

### Python (gradio_client)

```python
from gradio_client import Client

client = Client("YOUR_SPACE_URL")
result = client.predict(
    "path/to/screenshot.jpg",   # image filepath or URL
    "Is this an active mario kart race? Respond yes, no, or unsure.",
    api_name="/predict",
)
print(result)  # {"answer": "yes", "is_race": True}
```

### Python (requests)

```python
import base64, requests

with open("screenshot.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "YOUR_SPACE_URL/api/predict",
    json={"data": [f"data:image/jpeg;base64,{b64}", "Is this an active mario kart race? Respond yes, no, or unsure."]},
)
print(resp.json())
```

### curl

```bash
curl -X POST YOUR_SPACE_URL/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["data:image/jpeg;base64,<BASE64>", "Is this an active mario kart race? Respond yes, no, or unsure."]}'
```

## Setup

1. Create a new HF Space (Gradio SDK, ZeroGPU hardware).
2. Push this entire folder as the Space repo.
3. Copy your LoRA weights to `lora_weights/lora.safetensors`.
4. Set environment variables in the Space settings if the defaults don't match:
   - `BASE_MODEL_REPO` – HF repo for the base model (default: `moondream/starmie-v1`)
   - `BASE_MODEL_FILE` – filename within the repo (default: `model.safetensors`)
   - `LORA_WEIGHTS_PATH` – path to LoRA weights in the Space (default: `lora_weights/lora.safetensors`)
   - `LORA_RANK` – LoRA rank (default: `32`)
   - `LORA_ALPHA` – LoRA alpha (default: `64`)

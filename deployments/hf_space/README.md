---
title: Moondream Visual-Prompt Demo
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: "3.10"
app_file: app.py
hardware: zero-a10g
pinned: false
license: apache-2.0
short_description: Image-as-prompt object detection (Moondream 2 LoRA)
---

# Moondream Visual-Prompt Demo

Image-as-prompt object detection: instead of asking the model in *words*
("find all the dogs"), give it a **picture** of one example object — the
fine-tuned Moondream 2 will draw boxes around every other matching
instance in your target image.

## How it works

The base [Moondream 2](https://huggingface.co/vikhyatk/moondream2) detect
template is

    [BOS, image patches] [detect prefix] [tokenize(" cat")] [detect suffix]

The fine-tune in this Space replaces the tokenized class name with a
mean-pooled vision embedding of a query image:

    [BOS, image patches] [detect prefix] [* mean_pool(vis_enc(query)) *] [detect suffix]
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          a single splice slot, fed by the
                                          LoRA-trained vision.proj_mlp

The model learns to detect "things that look like *this* image" instead of
"things that match *this word*".

Trainable parameters were:

- LoRA on the text decoder (`qkv`, `proj`, `fc1`, `fc2`) — rank 64
- LoRA on `vision.proj_mlp.{fc1,fc2}` — rank 32
- Full fine-tune of the region head

Trained on LVIS triplets (query crop from one image, target image with one
or more instances of the same class). Source code:
[github.com/nkasmanoff/moondream-finetuning](https://github.com/nkasmanoff/moondream-finetuning)
(if public).

## Two ways to try it

- **Paint to find** — upload an image, paint over one example object with
  the brush. The painted region is auto-cropped and used as the visual
  prompt; the model draws boxes around every other matching instance in
  the same image.
- **LVIS examples** — clickable gallery of (query, target) pairs sampled
  from the LVIS validation split.

## Hardware

This Space runs on **ZeroGPU** (NVIDIA H200, dynamically allocated). A
typical detection takes <2 s of GPU time. Cold starts pay ~30 s for the
first download of the base Moondream weights from
[`vikhyatk/moondream2`](https://huggingface.co/vikhyatk/moondream2);
subsequent restarts hit the HF cache and start in seconds.

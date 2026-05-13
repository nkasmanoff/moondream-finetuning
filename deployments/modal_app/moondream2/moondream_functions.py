from typing import Union, Optional, Dict

import torch
from PIL import Image

from moondream2 import (
    MoondreamModel,
    EncodedImage,
    ObjectSamplingSettings,
    DEFAULT_MAX_OBJECTS,
    SpatialRefs,
)
from .region import decode_coordinate, encode_coordinate, decode_size, encode_size
from .text import text_encoder, lm_head, text_decoder
from .image_crops import reconstruct_from_crops
from .vision import vision_encoder, vision_projection, prepare_crops


def _vis_enc(model: MoondreamModel, x: torch.Tensor):
    return vision_encoder(x, model.vision, model.config.vision)


def _vis_proj(model: MoondreamModel, g: torch.Tensor, r: torch.Tensor):
    return vision_projection(g, r, model.vision, model.config.vision)


def _prefill(
    model: MoondreamModel,
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    pos_ids: torch.Tensor,
):
    return text_decoder(x, model.text, attn_mask, pos_ids, model.config.text, None)


def _decode_one_tok(
    model: MoondreamModel,
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    pos_ids: torch.Tensor,
):
    hidden = text_decoder(x, model.text, attn_mask, pos_ids, model.config.text, None)
    logits = lm_head(hidden, model.text)
    return logits, hidden


def _run_vision_encoder(model: MoondreamModel, image: Image.Image) -> torch.Tensor:
    all_crops, tiling = prepare_crops(image, model.config.vision, device=model.device)
    torch._dynamo.mark_dynamic(all_crops, 0)
    outputs = _vis_enc(model, all_crops)
    global_features = outputs[0]
    local_features = outputs[1:].view(
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
    return _vis_proj(model, global_features, reconstructed)


def _apply_top_p(probs: torch.Tensor, top_p: float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_probs = torch.zeros_like(probs)
    next_probs.scatter_(dim=-1, index=probs_idx, src=probs_sort)
    return next_probs


def _prefill_prompt(
    model: MoondreamModel,
    prompt_tokens: torch.Tensor,
    pos: int,
    temperature: float,
    top_p: float,
    spatial_refs: Optional[SpatialRefs] = None,
    attn_mask: Optional[torch.Tensor] = None,
):
    with torch.inference_mode():
        prompt_emb = text_encoder(prompt_tokens, model.text)
        torch._dynamo.mark_dynamic(prompt_emb, 1)

        if attn_mask is None:
            attn_mask = model.attn_mask

        mask = attn_mask[:, :, pos : pos + prompt_emb.size(1), :]
        pos_ids = torch.arange(pos, pos + prompt_emb.size(1), dtype=torch.long)
        hidden_BC = _prefill(model, prompt_emb, mask, pos_ids)
        logits_BV = lm_head(hidden_BC, model.text)

        if temperature == 0:
            next_token = torch.argmax(logits_BV, dim=-1).unsqueeze(1)
        else:
            probs = torch.softmax(logits_BV / temperature, dim=-1)
            probs = _apply_top_p(probs, top_p)
            next_token = torch.multinomial(probs, num_samples=1)

    pos = pos + prompt_emb.size(1)
    return logits_BV, hidden_BC, next_token, pos


def _generate_points(
    model: MoondreamModel,
    hidden: torch.Tensor,
    next_token: torch.Tensor,
    pos: int,
    include_size: bool = True,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    temperature: float = 0.0,
):
    out = []
    mask = torch.zeros(1, 1, 2048, device=model.device, dtype=torch.bool)
    mask[:, :, :pos] = 1
    pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)

    with torch.inference_mode():
        while (
            next_token.item() != model.config.tokenizer.eos_id
            and len(out) < max_objects
        ):
            x_logits = decode_coordinate(hidden, model.region)

            if temperature > 0:
                x_probs = torch.softmax(x_logits.squeeze(0) / temperature, dim=-1)
                x_bin = torch.multinomial(x_probs, num_samples=1)
            else:
                x_bin = torch.argmax(x_logits, dim=-1)

            x_logit = torch.gather(x_logits.squeeze(0), -1, x_bin)
            x_logprob = torch.log_softmax(x_logits.squeeze(0), dim=-1)
            x_logprob = torch.gather(x_logprob, -1, x_bin).squeeze()
            x_center = x_bin.float() / x_logits.size(-1)
            next_emb = encode_coordinate(
                x_center.unsqueeze(-1).to(dtype=x_logits.dtype), model.region
            )

            mask[:, :, pos], pos_ids[0] = 1, pos
            _, hidden = _decode_one_tok(model, next_emb, mask, pos_ids)
            pos += 1
            y_logits = decode_coordinate(hidden, model.region)

            if temperature > 0:
                y_probs = torch.softmax(y_logits.squeeze(0) / temperature, dim=-1)
                y_bin = torch.multinomial(y_probs, num_samples=1)
            else:
                y_bin = torch.argmax(y_logits, dim=-1)

            y_logit = torch.gather(y_logits.squeeze(0), -1, y_bin)
            y_logprob = torch.log_softmax(y_logits.squeeze(0), dim=-1)
            y_logprob = torch.gather(y_logprob, -1, y_bin).squeeze()
            y_center = y_bin.float() / y_logits.size(-1)
            next_emb = encode_coordinate(
                y_center.unsqueeze(-1).to(dtype=y_logits.dtype), model.region
            )

            if include_size:
                mask[:, :, pos], pos_ids[0] = 1, pos
                _, hidden = _decode_one_tok(model, next_emb, mask, pos_ids)
                pos += 1
                size_logits = decode_size(hidden, model.region)

                if temperature > 0:
                    w_probs = torch.softmax(size_logits[0] / temperature, dim=-1)
                    w_bin = torch.multinomial(w_probs, num_samples=1)[0]
                    h_probs = torch.softmax(size_logits[1] / temperature, dim=-1)
                    h_bin = torch.multinomial(h_probs, num_samples=1)[0]
                else:
                    w_bin = torch.argmax(size_logits[0], dim=-1)
                    h_bin = torch.argmax(size_logits[1], dim=-1)

                w_logit = torch.gather(
                    size_logits[0], -1, w_bin.unsqueeze(-1)
                ).squeeze()
                h_logit = torch.gather(
                    size_logits[1], -1, h_bin.unsqueeze(-1)
                ).squeeze()

                w_logprobs = torch.log_softmax(size_logits[0], dim=-1)
                w_logprob = torch.gather(w_logprobs, -1, w_bin.unsqueeze(-1)).squeeze()

                h_logprobs = torch.log_softmax(size_logits[1], dim=-1)
                h_logprob = torch.gather(h_logprobs, -1, h_bin.unsqueeze(-1)).squeeze()

                w = torch.pow(2.0, (w_bin.float() / 1023.0) * 10.0 - 10.0)
                h = torch.pow(2.0, (h_bin.float() / 1023.0) * 10.0 - 10.0)

                next_emb = (
                    encode_size(
                        torch.tensor(
                            [w, h], device=model.device, dtype=size_logits.dtype
                        ),
                        model.region,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                out.append(
                    {
                        "x_min": x_center.item() - w.item() / 2,
                        "y_min": y_center.item() - h.item() / 2,
                        "x_max": x_center.item() + w.item() / 2,
                        "y_max": y_center.item() + h.item() / 2,
                        "x_logprob": x_logprob,
                        "y_logprob": y_logprob,
                        "w_logprob": w_logprob,
                        "h_logprob": h_logprob,
                    }
                )
            else:
                out.append(
                    {
                        "x": x_center.item(),
                        "y": y_center.item(),
                        "x_logprob": x_logprob,
                        "y_logprob": y_logprob,
                    }
                )

            mask[:, :, pos], pos_ids[0] = 1, pos
            logits, hidden = _decode_one_tok(model, next_emb, mask, pos_ids)
            pos += 1
            next_token = torch.argmax(logits, dim=-1)

    return out


def detect(
    model: MoondreamModel,
    image: Union[Image.Image, EncodedImage],
    object_str: str,
    settings: Optional[ObjectSamplingSettings] = None,
    temperature: float = 0.0,
):
    if model.config.tokenizer.templates["detect"] is None:
        raise NotImplementedError("Model does not support object detection.")

    image = model.encode_image(image, settings)
    model.load_encoded_image(image)

    prompt_tokens = torch.tensor(
        [
            model.config.tokenizer.templates["detect"]["prefix"]
            + model.tokenizer.encode(" " + object_str).ids
            + model.config.tokenizer.templates["detect"]["suffix"]
        ],
        device=model.device,
    )

    _, hidden, next_token, pos = _prefill_prompt(
        model, prompt_tokens, image.pos, temperature=0, top_p=0
    )
    hidden = hidden[:, -1:, :]

    max_objects = (
        settings.get("max_objects", DEFAULT_MAX_OBJECTS)
        if settings
        else DEFAULT_MAX_OBJECTS
    )
    objects = _generate_points(
        model,
        hidden,
        next_token,
        pos,
        include_size=True,
        max_objects=max_objects,
        temperature=temperature,
    )

    return {"objects": objects}


def encode_image_grad(
    model: MoondreamModel,
    image: Union[Image.Image, EncodedImage],
    settings: Optional[Dict] = None,
) -> EncodedImage:
    if isinstance(image, EncodedImage):
        return image
    elif not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image or EncodedImage")

    img_emb = _run_vision_encoder(model, image)
    bos_emb = text_encoder(
        torch.tensor([[model.config.tokenizer.bos_id]], device=model.device),
        model.text,
    )
    inputs_embeds = torch.cat([bos_emb, img_emb[None]], dim=1)
    mask = model.attn_mask[:, :, 0 : inputs_embeds.size(1), :]
    pos_ids = torch.arange(inputs_embeds.size(1), dtype=torch.long)
    _prefill(model, inputs_embeds, mask, pos_ids)

    return EncodedImage(
        pos=inputs_embeds.size(1),
        caches=[
            (
                b.kv_cache.k_cache[:, :, : inputs_embeds.size(1), :].clone(),
                b.kv_cache.v_cache[:, :, : inputs_embeds.size(1), :].clone(),
            )
            for b in model.text.blocks
        ],
    )


def _prefill_prompt_grad(
    model: MoondreamModel,
    prompt_tokens: torch.Tensor,
    pos: int,
    temperature: float,
    top_p: float,
    spatial_refs: Optional[SpatialRefs] = None,
    attn_mask: Optional[torch.Tensor] = None,
):
    prompt_emb = text_encoder(prompt_tokens, model.text)
    torch._dynamo.mark_dynamic(prompt_emb, 1)

    if attn_mask is None:
        attn_mask = model.attn_mask

    mask = attn_mask[:, :, pos : pos + prompt_emb.size(1), :]
    pos_ids = torch.arange(pos, pos + prompt_emb.size(1), dtype=torch.long)
    hidden_BC = _prefill(model, prompt_emb, mask, pos_ids)
    logits_BV = lm_head(hidden_BC, model.text)

    if temperature == 0:
        next_token = torch.argmax(logits_BV, dim=-1).unsqueeze(1)
    else:
        probs = torch.softmax(logits_BV / temperature, dim=-1)
        probs = _apply_top_p(probs, top_p)
        next_token = torch.multinomial(probs, num_samples=1)

    pos = pos + prompt_emb.size(1)
    return logits_BV, hidden_BC, next_token, pos


def _generate_points_grad(
    model: MoondreamModel,
    hidden: torch.Tensor,
    next_token: torch.Tensor,
    pos: int,
    include_size: bool = True,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    temperature: float = 0.0,
):
    out = []
    out_logits = []
    out_logprobs = []
    mask = torch.zeros(1, 1, 2048, device=model.device, dtype=torch.bool)
    mask[:, :, :pos] = 1

    while next_token.item() != model.config.tokenizer.eos_id and len(out) < max_objects:
        x_logits = decode_coordinate(hidden, model.region)
        if temperature > 0:
            x_probs = torch.softmax(x_logits.squeeze(0) / temperature, dim=-1)
            x_bin = torch.multinomial(x_probs, num_samples=1)

        else:
            x_bin = torch.argmax(x_logits, dim=-1)

        x_logprobs = torch.log_softmax(x_logits.squeeze(1), dim=-1)
        x_logprob = torch.gather(x_logprobs, -1, x_bin).squeeze()
        out_logprobs.append(x_logprob)
        x_logit = torch.gather(x_logits.squeeze(1), -1, x_bin).squeeze()
        x_center = x_bin.float() / x_logits.size(-1)
        next_emb = encode_coordinate(
            x_center.unsqueeze(-1).to(dtype=x_logits.dtype), model.region
        )

        mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = _decode_one_tok(model, next_emb, mask, pos_ids)
        pos += 1

        y_logits = decode_coordinate(hidden, model.region)
        if temperature > 0:
            y_probs = torch.softmax(y_logits.squeeze(0) / temperature, dim=-1)
            y_bin = torch.multinomial(y_probs, num_samples=1)
        else:
            y_bin = torch.argmax(y_logits, dim=-1)

        y_logprobs = torch.log_softmax(y_logits.squeeze(1), dim=-1)
        y_logprob = torch.gather(y_logprobs, -1, y_bin).squeeze()
        out_logprobs.append(y_logprob)
        y_logit = torch.gather(y_logits.squeeze(1), -1, y_bin).squeeze()
        y_center = y_bin.float() / y_logits.size(-1)
        next_emb = encode_coordinate(
            y_center.unsqueeze(-1).to(dtype=y_logits.dtype), model.region
        )

        if include_size:
            mask[:, :, pos] = 1
            pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
            _, hidden = _decode_one_tok(model, next_emb, mask, pos_ids)
            pos += 1
            size_logits = decode_size(hidden, model.region)

            if temperature > 0:
                w_probs = torch.softmax(size_logits[0] / temperature, dim=-1)
                w_bin = torch.multinomial(w_probs, num_samples=1)[0]
                h_probs = torch.softmax(size_logits[1] / temperature, dim=-1)
                h_bin = torch.multinomial(h_probs, num_samples=1)[0]
            else:
                w_bin = torch.argmax(size_logits[0], dim=-1)
                h_bin = torch.argmax(size_logits[1], dim=-1)

            w_logit = torch.gather(size_logits[0], -1, w_bin.unsqueeze(-1)).squeeze()
            h_logit = torch.gather(size_logits[1], -1, h_bin.unsqueeze(-1)).squeeze()

            w_logprobs = torch.log_softmax(size_logits[0], dim=-1)
            w_logprob = torch.gather(w_logprobs, -1, w_bin.unsqueeze(-1)).squeeze()
            out_logprobs.append(w_logprob)

            h_logprobs = torch.log_softmax(size_logits[1], dim=-1)
            h_logprob = torch.gather(h_logprobs, -1, h_bin.unsqueeze(-1)).squeeze()
            out_logprobs.append(h_logprob)

            w = torch.pow(2.0, (w_bin.float() / 1023.0) * 10.0 - 10.0)
            h = torch.pow(2.0, (h_bin.float() / 1023.0) * 10.0 - 10.0)

            next_emb = (
                encode_size(
                    torch.tensor(
                        [w, h], device=model.device, dtype=size_logits[0].dtype
                    ),
                    model.region,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            out.append(
                {
                    "x_min": x_center.item() - w.item() / 2,
                    "y_min": y_center.item() - h.item() / 2,
                    "x_max": x_center.item() + w.item() / 2,
                    "y_max": y_center.item() + h.item() / 2,
                    "x_logit": x_logit,
                    "y_logit": y_logit,
                    "w_logit": w_logit,
                    "h_logit": h_logit,
                }
            )
            out_logits.append(x_logit)
            out_logits.append(y_logit)
            out_logits.append(w_logit)
            out_logits.append(h_logit)
        else:
            out.append(
                {
                    "x": x_center.item(),
                    "y": y_center.item(),
                    "x_logit": x_logit,
                    "y_logit": y_logit,
                }
            )
            out_logits.append(x_logit)
            out_logits.append(y_logit)

        mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        logits, hidden = _decode_one_tok(model, next_emb, mask, pos_ids)
        pos += 1
        next_token = torch.argmax(logits, dim=-1)

    return out, out_logits, out_logprobs


def detect_grad(
    model: MoondreamModel,
    image: Union[Image.Image, EncodedImage],
    object_str: str,
    settings: Optional[ObjectSamplingSettings] = None,
    temperature: float = 0.0,
):
    model._setup_caches()
    if model.config.tokenizer.templates["detect"] is None:
        raise NotImplementedError("Model does not support object detection.")

    image = encode_image_grad(model, image, settings)
    model.load_encoded_image(image)

    prompt_tokens = torch.tensor(
        [
            model.config.tokenizer.templates["detect"]["prefix"]
            + model.tokenizer.encode(" " + object_str).ids
            + model.config.tokenizer.templates["detect"]["suffix"]
        ],
        device=model.device,
    )

    _, hidden, next_token, pos = _prefill_prompt_grad(
        model, prompt_tokens, image.pos, temperature=0, top_p=0
    )
    hidden = hidden[:, -1:, :]

    max_objects = (
        settings.get("max_objects", DEFAULT_MAX_OBJECTS)
        if settings
        else DEFAULT_MAX_OBJECTS
    )
    objects, out_logits, out_logprobs = _generate_points_grad(
        model,
        hidden,
        next_token,
        pos,
        include_size=True,
        max_objects=max_objects,
        temperature=temperature,
    )

    return {"objects": objects, "out_logits": out_logits, "out_logprobs": out_logprobs}

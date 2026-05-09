import torch
import torch.nn as nn


class KVCache(nn.Module):

    def __init__(self, n_heads, n_kv_heads, max_context, dim, device, dtype):
        super().__init__()
        cache_shape = (1, n_kv_heads, max_context, dim // n_heads)
        self.register_buffer(
            "k_cache", torch.zeros(*cache_shape, device=device, dtype=dtype)
        )
        self.register_buffer(
            "v_cache", torch.zeros(*cache_shape, device=device, dtype=dtype)
        )

    def update(self, pos_ids, k, v):
        kout = self.k_cache.clone()
        vout = self.v_cache.clone()

        kout[:, :, pos_ids, :] = k
        vout[:, :, pos_ids, :] = v

        return kout, vout


def setup_caches(model, config):
    c = config.text
    for b in model.text.blocks:
        b.kv_cache = KVCache(
            c.n_heads,
            c.n_kv_heads,
            c.max_context,
            c.dim,
            device=model.device,
            dtype=model.vision.pos_emb.dtype,
        )

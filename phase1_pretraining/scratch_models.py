"""From-scratch PyTorch implementations (no transformers import) of the two Phase 1 architectures:
   * ModernBERT-style masked language model  (the encoder used in Phases 4, 5, 6, 7)
   * GPT-2-style causal language model        (the decoder used for the Phase 1 submission)
   Parameter names match the Hugging Face layout, so existing checkpoints load with strict=True."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================== ModernBERT ===============================================================
class MBEmbeddings(nn.Module):
    def __init__(self, vocab, d, pad_id, eps):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.norm = nn.LayerNorm(d, eps=eps, bias=False)

    def forward(self, ids):
        return self.norm(self.tok_embeddings(ids))


class MBMLP(nn.Module):
    """Gated GELU feed-forward block: Wo( gelu(a) * g ) with [a, g] = Wi(x)."""
    def __init__(self, d, ff):
        super().__init__()
        self.Wi = nn.Linear(d, 2 * ff, bias=False)
        self.Wo = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        a, g = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(a) * g)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class MBAttention(nn.Module):
    """Bidirectional multi-head self-attention with rotary position embeddings; one fused q/k/v projection."""
    def __init__(self, d, heads):
        super().__init__()
        self.heads, self.hd = heads, d // heads
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q, k, v = self.Wqkv(x).view(B, T, 3, self.heads, self.hd).unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)               # (B, heads, T, head_dim)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)                    # scale = 1/sqrt(head_dim)
        return self.Wo(out.transpose(1, 2).reshape(B, T, -1))


class MBLayer(nn.Module):
    """Pre-norm block. The first layer has no attention norm (the embedding LayerNorm already normalises its input)."""
    def __init__(self, d, ff, heads, layer_idx, eps):
        super().__init__()
        self.attn_norm = nn.Identity() if layer_idx == 0 else nn.LayerNorm(d, eps=eps, bias=False)
        self.attn = MBAttention(d, heads)
        self.mlp_norm = nn.LayerNorm(d, eps=eps, bias=False)
        self.mlp = MBMLP(d, ff)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        return x + self.mlp(self.mlp_norm(x))


class MBPredictionHead(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.dense = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d, eps=eps, bias=False)

    def forward(self, x):
        return self.norm(F.gelu(self.dense(x)))


class ModernBertCore(nn.Module):
    def __init__(self, vocab, d, ff, layers, heads, local_window, global_every, pad_id, eps, theta_global, theta_local):
        super().__init__()
        self.hd, self.half_window, self.pad_id = d // heads, local_window // 2, pad_id
        self.layer_is_global = [i % global_every == 0 for i in range(layers)]
        self.embeddings = MBEmbeddings(vocab, d, pad_id, eps)
        self.layers = nn.ModuleList([MBLayer(d, ff, heads, i, eps) for i in range(layers)])
        self.final_norm = nn.LayerNorm(d, eps=eps, bias=False)
        for name, theta in (("global", theta_global), ("local", theta_local)):
            self.register_buffer(f"inv_freq_{name}", 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float) / self.hd)), persistent=False)

    def rope(self, T, name, device, dtype):
        freqs = torch.arange(T, device=device).float()[:, None] * getattr(self, f"inv_freq_{name}").to(device)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)           # (1, 1, T, head_dim)

    def forward(self, ids, attention_mask=None):
        B, T = ids.shape
        x = self.embeddings(ids)
        full = torch.ones(T, T, dtype=torch.bool, device=ids.device)
        pos = torch.arange(T, device=ids.device)
        local = (pos[:, None] - pos[None, :]).abs() <= self.half_window                   # sliding window, inclusive
        full, local = full[None, None].expand(B, 1, T, T), local[None, None].expand(B, 1, T, T)
        if attention_mask is not None:                                                    # key padding mask
            keep = attention_mask.bool()[:, None, None, :]
            full, local = full & keep, local & keep
        cs = {n: self.rope(T, n, ids.device, x.dtype) for n in ("global", "local")}
        for is_global, layer in zip(self.layer_is_global, self.layers):
            n = "global" if is_global else "local"
            x = layer(x, cs[n][0], cs[n][1], full if is_global else local)
        return self.final_norm(x)


class ModernBertMLM(nn.Module):
    """Masked language model: core encoder + prediction head + decoder tied to the input embeddings."""
    def __init__(self, vocab_size=32001, hidden_size=768, intermediate_size=1152, num_layers=12, num_heads=12, local_attention=128,
                 global_attn_every_n_layers=3, pad_id=0, norm_eps=1e-5, global_rope_theta=160000.0, local_rope_theta=10000.0, init_std=0.02):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_layers=num_layers, num_heads=num_heads,
                        local_attention=local_attention, global_attn_every_n_layers=global_attn_every_n_layers)
        self.model = ModernBertCore(vocab_size, hidden_size, intermediate_size, num_layers, num_heads, local_attention, global_attn_every_n_layers, pad_id, norm_eps,
                                    global_rope_theta, local_rope_theta)
        self.head = MBPredictionHead(hidden_size, norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.decoder.weight = self.model.embeddings.tok_embeddings.weight                 # tied
        self.init_weights(init_std)

    @torch.no_grad()
    def init_weights(self, std, cutoff=2.0):
        """Truncated-normal initialisation: input projections std, output projections std/sqrt(2*layers), embeddings std; norms start at 1."""
        out_std = std / math.sqrt(2.0 * self.cfg["num_layers"])
        def tn(w, s): nn.init.trunc_normal_(w, mean=0.0, std=s, a=-cutoff * s, b=cutoff * s)
        tn(self.model.embeddings.tok_embeddings.weight, std)
        if self.model.embeddings.tok_embeddings.padding_idx is not None:
            self.model.embeddings.tok_embeddings.weight[self.model.embeddings.tok_embeddings.padding_idx].zero_()
        for layer in self.model.layers:
            tn(layer.attn.Wqkv.weight, std); tn(layer.attn.Wo.weight, out_std); tn(layer.mlp.Wi.weight, std); tn(layer.mlp.Wo.weight, out_std)
        tn(self.head.dense.weight, out_std)
        nn.init.zeros_(self.decoder.bias)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm): nn.init.ones_(m.weight)

    def forward(self, input_ids, attention_mask=None, labels=None, ignore_index=-100):
        h = self.model(input_ids, attention_mask)
        if labels is not None:                                                            # sparse prediction: only masked positions are scored
            keep = labels.view(-1) != ignore_index
            h = h.reshape(-1, h.shape[-1])[keep]
            logits = self.decoder(self.head(h))
            return logits, F.cross_entropy(logits.float(), labels.view(-1)[keep])
        return self.decoder(self.head(h)), None


# =============================================================== GPT-2 ===============================================================
class Conv1D(nn.Module):
    """A linear layer whose weight is stored as (in, out), the layout used by GPT-2 checkpoints."""
    def __init__(self, nf, nx):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.empty(nx, nf)), nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).view(*x.shape[:-1], -1)


class GPT2Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.c_attn, self.c_proj = Conv1D(3 * d, d), Conv1D(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.c_attn(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(out.transpose(1, 2).reshape(B, T, D))


class GPT2MLP(nn.Module):
    def __init__(self, d, inner):
        super().__init__()
        self.c_fc, self.c_proj = Conv1D(inner, d), Conv1D(d, inner)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class GPT2Block(nn.Module):
    def __init__(self, d, heads, inner, eps):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(d, eps=eps), GPT2Attention(d, heads)
        self.ln_2, self.mlp = nn.LayerNorm(d, eps=eps), GPT2MLP(d, inner)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2Core(nn.Module):
    def __init__(self, vocab, positions, d, layers, heads, inner, eps):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(positions, d)
        self.h = nn.ModuleList([GPT2Block(d, heads, inner, eps) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d, eps=eps)

    def forward(self, ids):
        x = self.wte(ids) + self.wpe(torch.arange(ids.shape[1], device=ids.device))[None]
        for blk in self.h: x = blk(x)
        return self.ln_f(x)


class GPT2LM(nn.Module):
    def __init__(self, vocab_size=32000, n_positions=4096, n_embd=768, n_layer=12, n_head=12, n_inner=None, eps=1e-5, init_std=0.02):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
        self.transformer = GPT2Core(vocab_size, n_positions, n_embd, n_layer, n_head, n_inner or 4 * n_embd, eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight                                 # tied
        self.init_weights(init_std)

    @torch.no_grad()
    def init_weights(self, std):
        for m in self.modules():
            if isinstance(m, Conv1D): nn.init.normal_(m.weight, std=std)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=std)
            elif isinstance(m, nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        for name, p in self.named_parameters():                                           # residual-path scaling, as in GPT-2
            if name.endswith("c_proj.weight"): nn.init.normal_(p, std=std / math.sqrt(2 * self.cfg["n_layer"]))

    def forward(self, input_ids, labels=None):
        logits = self.lm_head(self.transformer(input_ids))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1)) if labels is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=300, temperature=0.8, top_k=50):
        """Sampling with temperature and top-k, as used for the Phase 1 submission."""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.cfg["n_positions"]:])[0][:, -1, :] / temperature
            v, ix = logits.topk(min(top_k, logits.shape[-1]))
            nxt = ix.gather(1, torch.multinomial(F.softmax(v, dim=-1), 1))
            ids = torch.cat([ids, nxt], dim=1)
        return ids

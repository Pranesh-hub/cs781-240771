import torch, sys, time
sys.path.insert(0, ".")
from scratch_models import ModernBertMLM, GPT2LM
from transformers import ModernBertConfig, ModernBertForMaskedLM, GPT2Config, GPT2LMHeadModel
torch.manual_seed(0)

# ---------------- ModernBERT: the exact Phase 1 encoder configuration
cfg = ModernBertConfig(vocab_size=32001, max_position_embeddings=512, hidden_size=768, intermediate_size=1152, num_hidden_layers=12, num_attention_heads=12,
                       local_attention=128, global_attn_every_n_layers=3, pad_token_id=0, attention_dropout=0.0, mlp_dropout=0.0, embedding_dropout=0.0,
                       classifier_pooling="mean", sparse_prediction=True, sparse_pred_ignore_index=-100, reference_compile=False)
hf = ModernBertForMaskedLM(cfg).eval()
mine = ModernBertMLM().eval()
sd = {k: v for k, v in hf.state_dict().items()}
res = mine.load_state_dict(sd, strict=True); print("ModernBERT strict load:", res)
print("parameters  HF:", sum(p.numel() for p in hf.parameters()), "| scratch:", sum(p.numel() for p in mine.parameters()))
ids = torch.randint(1, 32000, (2, 300))
am = torch.ones(2, 300, dtype=torch.long); am[1, 220:] = 0
with torch.no_grad():
    for name, mask in (("no padding", None), ("with padding", am)):
        a = hf(input_ids=ids, attention_mask=mask).logits
        b = mine(ids, attention_mask=mask)[0]
        valid = torch.ones(2, 300, dtype=torch.bool) if mask is None else mask.bool()
        d = (a - b)[valid].abs().max().item()
        print(f"  logits {name}: max |HF - scratch| = {d:.2e}  (logit scale {a[valid].abs().mean():.2f})")
    labels = torch.full((2, 300), -100); labels[:, ::7] = ids[:, ::7]
    lh = hf(input_ids=ids, labels=labels).loss.item(); lm = mine(ids, labels=labels)[1].item()
    print(f"  masked-LM loss: HF {lh:.6f} | scratch {lm:.6f}")
# training signal check: a few steps reduce the loss on a toy batch
mine.train(); opt = torch.optim.AdamW(mine.parameters(), lr=3e-4); l0 = None
for i in range(6):
    loss = mine(ids[:, :128], labels=torch.where(torch.arange(128)[None] % 5 == 0, ids[:, :128], torch.full((2, 128), -100)))[1]
    opt.zero_grad(); loss.backward(); opt.step(); l0 = l0 or loss.item()
print(f"  toy training: loss {l0:.3f} -> {loss.item():.3f}")

# ---------------- GPT-2: the Phase 1 submission decoder (12 x 768, vocab 32,000, 4,096 positions)
gcfg = GPT2Config(vocab_size=32000, n_positions=4096, n_ctx=4096, n_embd=768, n_layer=12, n_head=12, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
hg = GPT2LMHeadModel(gcfg).eval(); mg = GPT2LM().eval()
sd = {k: v for k, v in hg.state_dict().items() if not k.endswith(".attn.bias") and not k.endswith("masked_bias")}
print("\nGPT-2 strict load:", mg.load_state_dict(sd, strict=True))
print("parameters  HF:", sum(p.numel() for p in hg.parameters()), "| scratch:", sum(p.numel() for p in mg.parameters()), "(millions: %.1f)" % (sum(p.numel() for p in mg.parameters()) / 1e6))
ids = torch.randint(0, 32000, (2, 200))
with torch.no_grad():
    a = hg(ids).logits; b = mg(ids)[0]
print(f"  logits: max |HF - scratch| = {(a - b).abs().max().item():.2e}")
with torch.no_grad(): out = mg.generate(ids[:1, :10], max_new_tokens=8); print("  generate shape:", tuple(out.shape))

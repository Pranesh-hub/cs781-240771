# %% CELL 0: config and setup
import os, sys, re, json, math, time, random, glob, gc
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

SELFTEST = False          # the runner patches this
EPOCHS = 2
MAX_LEN, STRIDE_PRED, BATCH = 512, 256, 8
LR_BB, LR_HEAD = 5e-4, 1e-3
WORK = Path(os.environ.get("P2_WORK", "/kaggle/working/phase2_bilstm")); WORK.mkdir(parents=True, exist_ok=True)
ENT = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
BIO = ["O"] + [f"{p}-{l}" for l in ENT for p in "BI"]; B2I = {b: i for i, b in enumerate(BIO)}
IGN = -100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); AMP = device.type == "cuda"
random.seed(42); np.random.seed(42); torch.manual_seed(42)
T0 = time.time()
print("device:", device, "| SELFTEST:", SELFTEST, "| training cap (min):", CAP_MIN)   # CAP_MIN comes from the runner (cell 0)

# %% CELL 1: data, tokenizer, BiLSTM checkpoint
def find(name, roots=("/kaggle/input", "/kaggle/working")):
    for r in roots:
        h = sorted(glob.glob(f"{r}/**/{name}", recursive=True))
        if h: return h[0]
    return None

if SELFTEST:
    SD = WORK / "synth"; SD.mkdir(exist_ok=True)
    first = ["Ram", "Sita", "Arjun", "Mohan", "Priya", "Kiran", "Dev", "Anil"]; last = ["Kumar", "Sharma", "Verma", "Singh", "Rao", "Das"]
    def mk_doc(k):
        a, b, j = random.choice(first), random.choice(last), random.choice(last)
        t = f"IN THE SUPREME COURT OF INDIA\nCivil Appeal No. {k} of 2001\nShri {a} {b}, Advocate for the appellant.\nJustice {first[k % 8]} {j} delivered the judgment on 12.03.2001.\nMr. {b} {a}, Advocate, for the respondents.\n"
        g = []
        for m in re.finditer(r"SUPREME COURT OF INDIA", t): g.append([m.start(), m.end(), "COURT"])
        for m in re.finditer(r"\d\d\.\d\d\.\d{4}", t): g.append([m.start(), m.end(), "DATE"])
        for m in re.finditer(r"Justice (\w+ \w+)", t): g.append([m.start(1), m.end(1), "JUDGE"])
        m = re.search(r"Shri (\w+ \w+)", t); g.append([m.start(1), m.end(1), "A.COUNSEL"])
        m = re.search(r"Mr\. (\w+ \w+)", t); g.append([m.start(1), m.end(1), "R.COUNSEL"])
        return t, sorted(g)
    for fold, n in (("fold1", 40), ("fold2", 12), ("fold3", 8)):
        docs = [mk_doc(i + (0 if fold == "fold1" else 100)) for i in range(n)]
        json.dump({f"d{i}": d[0] for i, d in enumerate(docs)}, open(SD / f"{fold}-text.json", "w"))
        if fold != "fold3": json.dump({f"d{i}": d[1] for i, d in enumerate(docs)}, open(SD / f"{fold}-gold.json", "w"))
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tk = Tokenizer(models.BPE(unk_token="[UNK]")); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator([json.load(open(SD / "fold1-text.json"))[k] for k in json.load(open(SD / "fold1-text.json"))], trainers.BpeTrainer(vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tk.save(str(SD / "legal_bpe_32k_mask.json")); V = tk.get_vocab_size()
    sys.path.insert(0, os.environ.get("CS781_ROOT", str(SD))); import arch_loaders as AL
    mc = dict(family="bilstm", vocab_size=V, hidden_size=64, context_length=512, num_layers=3, emb=16, lstm_hidden=32)
    torch.save(dict(model_config=mc, model_state_dict=AL.build_full(mc).state_dict()), SD / "final_model.pt")
    DATA_DIR, TOK_PATH, CKPT_PATH = SD, str(SD / "legal_bpe_32k_mask.json"), str(SD / "final_model.pt")
else:
    need = ["fold1-text.json", "fold1-gold.json", "fold2-text.json", "fold2-gold.json", "fold3-text.json"]
    if not all(find(n) for n in need):
        from huggingface_hub import snapshot_download
        tok = os.environ.get("HF_TOKEN")
        if not tok:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("HF_TOKEN")
        snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset", allow_patterns=["Phase_2_LNER/*"], token=tok, local_dir=str(WORK / "hf"))
    DATA_DIR = Path(find("fold1-text.json", (str(WORK / "hf"), "/kaggle/input")) or find("fold1-text.json")).parent
    TOK_PATH = find("legal_bpe_32k_mask.json"); CKPT_PATH = next((p for p in glob.glob("/kaggle/input/**/final_model.pt", recursive=True) if "bilstm" in p.lower()), None)
    assert TOK_PATH and CKPT_PATH, "attach the dataset cs781-phase1-v3-bilstm-recurrent (final_model.pt + legal_bpe_32k_mask.json)"
    sys.path.insert(0, "/kaggle/working"); import arch_loaders as AL
def jl(n):   # fold1/fold2 live in .../Train, fold3 in the sibling .../Test, so search the parent folder too
    p = find(n, (str(DATA_DIR), str(Path(DATA_DIR).parent))); assert p, f"{n} not found under {DATA_DIR} or its parent"
    return json.load(open(p, encoding="utf-8"))
train_texts, train_gold, val_texts, val_gold, test_texts = jl("fold1-text.json"), jl("fold1-gold.json"), jl("fold2-text.json"), jl("fold2-gold.json"), jl("fold3-text.json")
print(f"fold1 {len(train_texts)} docs | fold2 {len(val_texts)} | fold3 {len(test_texts)} | tokenizer {TOK_PATH} | checkpoint {CKPT_PATH}")
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(TOK_PATH)

# %% CELL 2: BIO windows
def doc_tokens(text):
    e = TOK.encode(text, add_special_tokens=False); return e.ids, e.offsets
def bio_labels(offsets, gold):
    spans = sorted((int(x[0]), int(x[1]), x[2]) for x in gold); out = []; j = 0; last = None
    for a, b in offsets:
        while j < len(spans) and spans[j][1] <= a: j += 1
        if b > a and j < len(spans) and spans[j][0] < b and spans[j][1] > a:
            out.append(B2I[("I-" if last == j else "B-") + spans[j][2]]); last = j
        else: out.append(0); last = None if b > a else last
    return out
def make_windows(texts, golds):
    W = []
    for d, t in texts.items():
        ids, off = doc_tokens(t); lab = bio_labels(off, golds.get(d, []))
        for s in range(0, len(ids), MAX_LEN): W.append((ids[s:s + MAX_LEN], lab[s:s + MAX_LEN]))
    return W
TRW = make_windows(train_texts, train_gold); print("train windows:", len(TRW), "| tokens:", sum(len(w[0]) for w in TRW))
cnt = Counter(l for _, lab in TRW for l in lab); freq = torch.tensor([max(cnt.get(i, 0), 1) for i in range(len(BIO))], dtype=torch.float)
CLS_W = (freq[0] / freq).sqrt().clamp(1.0, 10.0); CLS_W[0] = 1.0

# %% CELL 3: BiLSTM tagger (partial fine-tuning: embeddings and the lowest LSTM layer frozen) and training under a time cap
bb, hid, causal, ctx, fam = AL.load_backbone(CKPT_PATH, device="cpu"); assert fam == "bilstm", fam
with torch.no_grad(): DIM = bb(input_ids=torch.zeros(1, 4, dtype=torch.long), attention_mask=torch.ones(1, 4, dtype=torch.long)).last_hidden_state.shape[-1]
class Tagger(nn.Module):
    def __init__(s, bb, dim, n): super().__init__(); s.bb = bb; s.drop = nn.Dropout(0.1); s.head = nn.Linear(dim, n)
    def forward(s, ids, mask): return s.head(s.drop(s.bb(input_ids=ids, attention_mask=mask).last_hidden_state))
model = Tagger(bb, DIM, len(BIO)).to(device)
for n_, p in model.bb.inner.named_parameters():
    if n_.startswith("emb.") or "_l0" in n_: p.requires_grad = False
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"BiLSTM tagger: feature dim {DIM} | trainable {trainable/1e6:.2f}M of {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
def collate(items):
    L = max(len(i) for i, _ in items); ids = torch.zeros(len(items), L, dtype=torch.long); m = torch.zeros_like(ids); y = torch.full_like(ids, IGN)
    for k, (i, l) in enumerate(items): ids[k, :len(i)] = torch.tensor(i); m[k, :len(i)] = 1; y[k, :len(l)] = torch.tensor(l)
    return ids.to(device), m.to(device), y.to(device)
opt = torch.optim.AdamW([{"params": [p for p in model.bb.parameters() if p.requires_grad], "lr": LR_BB}, {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda", enabled=AMP); lossf = nn.CrossEntropyLoss(weight=CLS_W.to(device), ignore_index=IGN)
order_len = sorted(range(len(TRW)), key=lambda i: len(TRW[i][0]))
batches = [order_len[i:i + BATCH] for i in range(0, len(order_len), BATCH)]; total = EPOCHS * len(batches); step = 0; t_train = time.time(); stop = False
for ep in range(EPOCHS):
    random.shuffle(batches); model.train(); run = 0.0; nb = 0
    for bi in batches:
        for g in opt.param_groups: g["lr"] = g.get("base", g["lr"]) if "base" in g else g["lr"]
        ids, m, y = collate([TRW[i] for i in bi])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP):
            lo = model(ids, m); loss = lossf(lo.float().reshape(-1, len(BIO)), y.reshape(-1))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        run += loss.item(); nb += 1; step += 1
        if step % 50 == 0: print(f"  ep {ep + 1} step {step}/{total} loss {run / nb:.4f} | {(time.time() - t_train) / 60:.1f} min")
        if (time.time() - t_train) / 60 >= CAP_MIN: stop = True; break
    print(f"epoch {ep + 1} done: loss {run / max(nb, 1):.4f}")
    if stop: print(f"STOPPED by the {CAP_MIN}-minute cap after {step} steps"); break
print(f"training took {(time.time() - t_train) / 60:.1f} min")

# %% CELL 4: prediction (sliding windows), span decoding, scoring, rule layer
@torch.inference_mode()
def doc_probs(text):
    ids, off = doc_tokens(text)
    if not ids: return off, None
    P = np.zeros((len(ids), len(BIO)), dtype=np.float32); C = np.zeros(len(ids), dtype=np.float32); model.eval()
    for s in range(0, len(ids), STRIDE_PRED):
        e = min(s + MAX_LEN, len(ids)); x = torch.tensor([ids[s:e]], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP): p = model(x, torch.ones_like(x)).float().softmax(-1)[0].cpu().numpy()
        P[s:e] += p; C[s:e] += 1
        if e == len(ids): break
    return off, P / np.maximum(C, 1)[:, None]
def decode(text, off, P):
    pred = P.argmax(-1); spans = []; cur = None
    def close():
        nonlocal cur
        if cur:
            s, e, lab, ps = cur;
            while s < e and text[s].isspace(): s += 1
            while e > s and text[e - 1].isspace(): e -= 1
            if s < e: spans.append((s, e, lab, float(np.mean(ps))))
        cur = None
    for (a, b), k, p in zip(off, pred, P):
        if b <= a: continue
        tag = BIO[int(k)]
        if tag == "O": close(); continue
        pre, lab = tag.split("-", 1)
        if pre == "B" or cur is None or cur[2] != lab: close(); cur = [a, b, lab, [p[k]]]
        else: cur[1] = b; cur[3].append(p[k])
    close(); return [tuple(x) for x in spans]
def predict(texts):
    out = {}
    for d, t in texts.items():
        off, P = doc_probs(t); out[d] = [] if P is None else decode(t, off, P)
    return out
def strict_macro(pred, gold):
    f = {}
    for lab in ENT:
        G = {(d, int(x[0]), int(x[1])) for d, e in gold.items() for x in e if x[2] == lab}
        if not G: continue
        Pp = {(d, s, e) for d, sp in pred.items() for (s, e, l, *_) in sp if l == lab}; tp = len(G & Pp)
        p = tp / len(Pp) if Pp else 0.0; r = tp / len(G); f[lab] = 2 * p * r / (p + r) if p + r else 0.0
    return float(np.mean(list(f.values()))) if f else 0.0, f
RULES = {"__name__": "p2_rules"}; exec(open(os.environ.get("P2_RULES_PATH", "/kaggle/working/p2_rules_core.py"), encoding="utf-8").read(), RULES)
def select_defaults(texts, gold):
    f1s = {pol: RULES["counsel_f1"](texts, gold, pol) for pol in ("R", "A", "drop")}
    RULES["chosen"] = max(f1s, key=lambda p: (f1s[p]["A.COUNSEL"] + f1s[p]["R.COUNSEL"], p == "R"))
    G = {(d, int(x[0]), int(x[1])) for d, e in gold.items() for x in e if x[2] == "JUDGE"}; Pn = {n: set() for n in RULES["NAMES"]}
    for d, t in texts.items():
        for n, sp in RULES["judge_patterns2"](t).items(): Pn[n] |= {(d, s, e) for s, e in sp}
    keep = [n for n in RULES["NAMES"] if len(G & Pn[n]) >= 5 and len(G & Pn[n]) / max(len(Pn[n]), 1) >= 0.4]
    print(f"rule layer chosen on fold 1 only: counsel default '{RULES['chosen']}' | JUDGE patterns kept: {keep}"); return keep
KEEP = select_defaults(train_texts, train_gold)
VAL_PRED = predict(val_texts); m0, f0 = strict_macro(VAL_PRED, val_gold)
VAL_RULES = RULES["merge_all"](val_texts, VAL_PRED, KEEP); m1, f1_ = strict_macro({d: [(s, e, l) for s, e, l in sp] for d, sp in VAL_RULES.items()}, val_gold)
print(f"FOLD 2 strict macro-F1 (12-label convention, labels present in gold): BiLSTM alone {m0:.4f} | BiLSTM + counsel/JUDGE rules {m1:.4f}")
print({k: round(v, 3) for k, v in f1_.items()})

# %% CELL 5: fold 3 predictions, validation, files
TEST_PRED = predict(test_texts); TEST_RULES = RULES["merge_all"](test_texts, TEST_PRED, KEEP)
def to_obj(P): return {str(d): [[int(s), int(e), l] for s, e, l, *_ in sp] for d, sp in P.items()}
def validate(obj, texts):
    bad = 0
    for d, sp in obj.items():
        t = texts[d]
        for s, e, l in sp:
            if not (0 <= s < e <= len(t)) or l not in ENT: bad += 1
    assert set(obj) == set(map(str, texts)), "document ids differ"; return bad
obj_rules, obj_model = to_obj({d: [(s, e, l) for s, e, l in sp] for d, sp in TEST_RULES.items()}), to_obj(TEST_PRED)
print("invalid spans: with rules", validate(obj_rules, {str(k): v for k, v in test_texts.items()}), "| model only", validate(obj_model, {str(k): v for k, v in test_texts.items()}))
json.dump(obj_rules, open(WORK / "prediction.json", "w", encoding="utf-8")); json.dump(obj_model, open(WORK / "prediction_model_only.json", "w", encoding="utf-8"))
json.dump(dict(fold2_bilstm_only=m0, fold2_bilstm_plus_rules=m1, per_label_with_rules=f1_, train_steps=step, minutes=round((time.time() - T0) / 60, 1)), open(WORK / "results.json", "w"), indent=1)
print("wrote", WORK / "prediction.json", "| total minutes", round((time.time() - T0) / 60, 1))

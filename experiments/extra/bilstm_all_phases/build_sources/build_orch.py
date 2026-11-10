"""Builds cs781_final/bilstm_all_phases/bilstm_all_phases.ipynb: ONE notebook that runs a short BiLSTM experiment for each of Phases 2-7.
Phases 3-7 reuse the cells of your existing, already-run notebooks (patched only to run the BiLSTM once under a time cap); Phase 2 is a new BiLSTM tagger + your counsel/JUDGE rule layer."""
import json, gzip, base64, re, sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
F = "C:/Users/PRANESH S/Downloads/cs781_final/"
SC = os.path.dirname(os.path.abspath(__file__)) + "/"
OUT_DIR = F + "bilstm_all_phases/"; os.makedirs(OUT_DIR, exist_ok=True)

def cells_of(path):
    nb = json.load(open(F + path, encoding="utf-8")); return ["".join(c["source"]) for c in nb["cells"]]
def sub(src, old, new, count=1):
    assert src.count(old) == count, f"patch target found {src.count(old)}x (expected {count}): {old[:80]!r}"
    return src.replace(old, new)

SHIM = ("import builtins\n"
        "def display(*a, **k):\n    for x in a: print(x)\n"
        "builtins.display = display\n"
        "CAP_MIN = {cap}   # minutes for the BiLSTM training in this phase (set by the runner)\n")
children = {}   # phase -> list of code strings

# ---------------- Phase 2: new BiLSTM tagger + rule layer
p2 = open(SC + "p2_child.py", encoding="utf-8").read()
p2 = sub(p2, '        for g in opt.param_groups: g["lr"] = g.get("base", g["lr"]) if "base" in g else g["lr"]\n', "")
children[2] = [SHIM] + [c.strip("\n") for c in re.split(r"^# %% .*$", p2, flags=re.M)[1:]]

# ---------------- Phase 3: phase3_rr_pipeline cells + BiLSTM registration
c3 = cells_of("Phase3_Rhetorical_Roles/notebook6d390d6f69 (12).ipynb")   # the version that actually ran on Kaggle (fixed parser); the older starter pipeline cannot read the real data
STUB3 = '''
HF_TOKEN = None      # cell 5 falls back to the HF_TOKEN environment variable / Kaggle secret
DURABLE_PUBLISH_ENABLED = False; PUBLISH_EVERY_N_EPOCHS = 999; PUBLISH_AT_EACH_EXPERIMENT_END = False; KAGGLE_DATASET_SLUG = ""     # no mid-run dataset uploads in this notebook
'''

OLD11 = """if not os.path.isfile(PHASE1_CHECKPOINT):
    raise FileNotFoundError(
        f"Required 3,000-step checkpoint not found: {PHASE1_CHECKPOINT}. "
        "Attach the Kaggle dataset 'CS781 Phase 1 Checkpoints'."
    )
ARCHS = {"gpt2_a": {"path": PHASE1_CHECKPOINT, "family": "gpt2"}}"""
NEW11 = """ARCHS = {"gpt2_a": {"path": PHASE1_CHECKPOINT, "family": "gpt2"}} if os.path.isfile(PHASE1_CHECKPOINT) else {}   # the earlier decoder is optional here; the BiLSTM is registered in the next cell"""
c3_11 = sub(c3[11], OLD11, NEW11)
c3_8 = c3[8]; assert "legal_bpe_32k.json" in c3_8; c3_8 = c3_8.replace("legal_bpe_32k.json", "legal_bpe_32k_mask.json")   # the BiLSTM dataset ships the mask tokenizer
A3 = '''
# ---- BiLSTM registration: the pipeline only knew the GPT-2 family; arch_loaders.py (Phase 1 loader) adds the BiLSTM backbone
sys.path.insert(0, "/kaggle/working"); import arch_loaders as AL
BL_CKPT = next(p for p in glob.glob("/kaggle/input/**/final_model.pt", recursive=True) if "bilstm" in p.lower())
ARCHS["bilstm"] = dict(path=BL_CKPT, family="bilstm")
_orig_build_backbone = build_backbone
def build_backbone(arch, bb_dropout=0.1, n_layers=None):
    if ARCHS[arch]["family"] == "bilstm":
        bb, hid, causal, ctx, fam = AL.load_backbone(ARCHS[arch]["path"])
        with torch.no_grad(): dim = bb(input_ids=torch.zeros(1, 4, dtype=torch.long), attention_mask=torch.ones(1, 4, dtype=torch.long)).last_hidden_state.shape[-1]
        return bb, dim, False, ctx, "bilstm"
    return _orig_build_backbone(arch, bb_dropout, n_layers)
print("registered architectures:", {k: v["path"] for k, v in ARCHS.items()})
'''
B3 = '''
PLAN = {"E7_bilstm_full": dict(method="full", ctx=True, arch="bilstm")}
run_experiment("E7_bilstm_full", epochs=3, patience=2, **PLAN["E7_bilstm_full"])      # full fine-tuning of the Phase 1 BiLSTM, both domains
BEST = "E7_bilstm_full"
'''
children[3] = [SHIM, STUB3, c3[5], c3[7], c3_8, c3[9], c3[10], c3_11, c3[12], A3, B3, c3[21], c3[22]]

# ---------------- Phase 4: v4_final cells, only P5 (BiLSTM partial fine-tuning)
c4 = cells_of("Phase4_CJPE/phase4_cjpe_v4_final.ipynb")
c4_8 = sub(c4[8], 'RUN = {"P3", "P4", "P5", "P7", "X1"}', 'RUN = {"P5"}')
c4_11 = sub(c4[11], 'epochs=2, lr=DEFAULT_LR["bilstm"], p_tail=0.5))', 'epochs=2, max_minutes=CAP_MIN, lr=2e-4, p_tail=0.5))')
c4_2 = sub(c4[2], "REQUIRE_HF, REQUIRE_KAGGLE = True, True", "REQUIRE_HF, REQUIRE_KAGGLE = True, False")   # the Kaggle key is only for the optional backup upload
children[4] = [SHIM, c4[1], c4_2, c4[3], c4[4], c4[5], c4[6], c4_8, c4_11]

# ---------------- Phase 5: part2 cells, only E4 (BiLSTM partial fine-tuning); the pilot is skipped (default LR)
c5 = cells_of("Phase5_LSI/phase5_lsi_part2_E4_E5_final.ipynb")
c5_12 = sub(c5[12], 'LR_RNN = safe("BiLSTM pilot", lambda: pilot_or_skip("bilstm", DEFAULT_LR["bilstm"], grid=(2e-4, 5e-4, 1e-3), minutes=6, L=512, method="partial", hp=dict(frac=0.5))) or DEFAULT_LR["bilstm"]', 'LR_RNN = DEFAULT_LR["bilstm"]')
c5_12 = sub(c5_12, "epochs=2, lr=LR_RNN, save_model=True))", "epochs=2, max_minutes=CAP_MIN, lr=LR_RNN, save_model=False))")
children[5] = [SHIM] + [c5[i] for i in range(1, 10)] + [c5[11], c5_12]

# ---------------- Phase 6: part4 cells, only S7 (BiLSTM sequence-to-sequence); no restore needed
c6 = cells_of("Phase6/phase6_part4_kd_bilstm_S6_S7.ipynb")
c6_1 = sub(c6[1], "REQUIRE_RESTORE = True", "REQUIRE_RESTORE = False")
S7 = '''
import traceback
if "safe" not in globals():
    def safe(label, fn):
        try: return fn()
        except Exception: print(f"{label} FAILED:"); traceback.print_exc(); return None
S7 = safe("S7", lambda: run_or_skip("S7_bilstm_partial", "bilstm", technique="partial", hp=dict(frac=0.5), epochs=1.0, max_minutes=CAP_MIN, exp_id="S7", val_cap=150, gen=dict(no_repeat=3)))
'''
children[6] = [SHIM, c6_1] + [c6[i] for i in range(2, 12)] + [S7]

# ---------------- Phase 7: bilstm retriever cells, only BL1 (partial fine-tuning), no continued pre-training, no dataset upload
c7 = cells_of("Phase7_PCR/phase7_bilstm_retriever.ipynb")
c7_1 = sub(c7[1], "UPLOAD_DATASET = True", "UPLOAD_DATASET = False"); c7_1 = sub(c7_1, "DAPT_ON = True", "DAPT_ON = False")
c7_14 = sub(c7[14], "(330, 15) if TIME_PROFILE", "(CAP_MIN * 2 + 40, 10) if TIME_PROFILE")
c7_14 = sub(c7_14, "CAPS = dict(DAPT=40, BL1=30, BL3=30, BL2=30, BL4=30) if TIME_PROFILE", "CAPS = dict(DAPT=40, BL1=CAP_MIN, BL3=30, BL2=30, BL4=30) if TIME_PROFILE")
c7_17 = sub(c7[17], "for key, name, tech, hp, init, note in PLAN:", "for key, name, tech, hp, init, note in PLAN[:1]:")
children[7] = [SHIM, c7_1] + [c7[i] for i in range(2, 14)] + [c7_14, c7[15], c7_17, c7[20]]   # cell 15 (zero-shot) also defines EVAL_MIN, which the retriever cell needs

# ---------------- every cell of every child must at least parse
for ph, cs in children.items():
    for i, c in enumerate(cs):
        try: compile(c.replace("{cap}", "10"), f"p{ph}_cell{i}", "exec")
        except SyntaxError as e: raise SystemExit(f"Phase {ph} cell {i}: {e}")
print({ph: len(cs) for ph, cs in children.items()})

# ---------------- the orchestrator notebook
blob = base64.b64encode(gzip.compress(json.dumps({str(k): v for k, v in children.items()}).encode("utf-8"), 9)).decode()
arch_src = open(F + "docs/arch_loaders.py", encoding="utf-8").read()
rules_src = open(SC + "p2_rules_core.py", encoding="utf-8").read()
runner = open(SC + "runner_cell.py", encoding="utf-8").read()
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(True)}
cells = [
 md("# CS781: one BiLSTM prediction for each of Phases 2-7, in one notebook (about 3-4 hours on a T4)\n\n"
    "The Phase 1 BiLSTM (560 steps) is fine-tuned **once per phase** under a time cap, and each phase writes a schema-checked `prediction.json` to `/kaggle/working/predictions/phase_N/`.\n"
    "Phases 3-7 reuse the cells of the notebooks you already ran on Kaggle (patched only to run the BiLSTM experiment once); Phase 2 is a new BiLSTM tagger followed by your counsel/JUDGE rule layer.\n\n"
    "**Attach:** the dataset `cs781-phase1-v3-bilstm-recurrent` (final_model.pt + legal_bpe_32k_mask.json). **Secrets:** `HF_TOKEN` (and `KAGGLE_USERNAME` / `KAGGLE_KEY` if a child's credentials check asks for them). **Settings:** Internet ON, GPU T4 x1.\n\n"
    "**Honest status:** the Phase 2 child was run on CPU end to end with a tiny model and synthetic data. Phases 3-7 are your own cells with small patches; they were checked for syntax and patch targets only, and **have not been run** with the BiLSTM. Phase 6 uses the existing S7 BiLSTM sequence-to-sequence model (not the encoder + GPT-2 decoder variant). Run order and times are estimates until the first real run."),
 code('''# ================= CONFIG =================
RUN_PHASES = [2, 3, 4, 7]                  # Phases 5 and 6 already produced BiLSTM predictions in the first run; add them back to rerun
CAP_MIN = {2: 25, 3: 0, 4: 25, 5: 25, 6: 30, 7: 20}   # BiLSTM training minutes per phase (Phase 3 has no minute cap: it trains 3 epochs on about 6-10k sentences)
CHILD_TIMEOUT_MIN = {2: 70, 3: 60, 4: 90, 5: 90, 6: 100, 7: 70}   # hard stop per phase, so one stuck phase cannot eat the session
SELFTEST = False
print("planned training minutes:", {p: CAP_MIN[p] for p in RUN_PHASES}, "| hard stops:", {p: CHILD_TIMEOUT_MIN[p] for p in RUN_PHASES})'''),
 code("# ================= SUPPORT FILES (written once) =================\nimport os, json, gzip, base64, textwrap\nROOT = os.environ.get('CS781_ROOT', '/kaggle/working'); os.makedirs(ROOT, exist_ok=True)\n"
      "ARCH_LOADERS_SRC = " + repr(arch_src) + "\nRULES_SRC = " + repr(rules_src) + "\nCHILDREN_B64 = " + repr(blob) + "\n"
      "open(ROOT + '/arch_loaders.py', 'w', encoding='utf-8').write(ARCH_LOADERS_SRC)\nopen(ROOT + '/p2_rules_core.py', 'w', encoding='utf-8').write(RULES_SRC)\n"
      "CHILDREN = json.loads(gzip.decompress(base64.b64decode(CHILDREN_B64)).decode('utf-8'))\nprint({k: len(v) for k, v in CHILDREN.items()}, 'cells per phase')"),
 code(runner),
]
import uuid
for c in cells: c["id"] = uuid.uuid4().hex[:8]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = OUT_DIR + "bilstm_all_phases.ipynb"; json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", path, f"({os.path.getsize(path)/1e3:.0f} KB)")

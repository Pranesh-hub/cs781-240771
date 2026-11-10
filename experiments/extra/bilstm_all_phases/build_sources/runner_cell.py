# ================= RUNNER: one phase = one subprocess (its own Python process, so variables of different phases never collide) =================
import os, sys, json, time, glob, shutil, subprocess, traceback, zipfile
ROOT = os.environ.get("CS781_ROOT", "/kaggle/working")
PRED_DIR = os.path.join(ROOT, "predictions"); os.makedirs(PRED_DIR, exist_ok=True)
NB_T0 = time.time()
RERUN = False                      # True = run a phase again even if its prediction already exists in predictions/phase_N/

def child_script(phase, cap):
    cells = list(CHILDREN[str(phase)]); cells[0] = cells[0].replace("{cap}", str(cap))
    if SELFTEST and phase == 2: cells[1] = cells[1].replace("SELFTEST = False          # the runner patches this", "SELFTEST = True")
    d = os.path.join(ROOT, "children"); os.makedirs(d, exist_ok=True)
    json.dump(cells, open(os.path.join(d, f"phase{phase}_cells.json"), "w", encoding="utf-8"))
    script = ("import json, sys, traceback\nG = {'__name__': '__main__'}\ncells = json.load(open(%r, encoding='utf-8'))\n"
              "for i, src in enumerate(cells):\n    print('\\n===== phase %d, cell %%d =====' %% i, flush=True)\n"
              "    try: exec(compile(src, 'phase%d_cell%%d' %% i, 'exec'), G)\n"
              "    except BaseException as e:\n        if isinstance(e, KeyboardInterrupt): raise\n        print('CELL FAILED:', type(e).__name__, e, flush=True); traceback.print_exc()\n") % (os.path.join(d, f"phase{phase}_cells.json"), phase, phase)
    p = os.path.join(d, f"phase{phase}_run.py"); open(p, "w", encoding="utf-8").write(script); return p

def find_output(phase):
    cands = {2: ["phase2_bilstm/prediction.json"], 3: ["phase3_rr/prediction.json"], 4: ["cjpe/submissions/P5_bilstm_partial*.json"], 5: ["lsi/submissions/E4_bilstm_partial*.json"],
             6: ["p6/submissions/S7_bilstm_partial*.json"], 7: ["pcr_bilstm/prediction_bilstm_only.json", "pcr_bilstm/prediction.json", "**/prediction_bilstm_only.json"]}[phase]
    for pat in cands:
        hits = [h for h in glob.glob(os.path.join(ROOT, pat), recursive=True) if os.path.isfile(h) and PRED_DIR not in h and "/children/" not in h.replace("\\", "/")]
        if hits: return sorted(hits, key=os.path.getmtime)[-1]
    return None

def validate(phase, path):
    o = json.load(open(path, encoding="utf-8"))
    if phase == 2:
        assert isinstance(o, dict) and o, "expected {doc_id: [[start, end, label], ...]}"
        for d, sp in o.items():
            for x in sp: assert len(x) == 3 and isinstance(x[0], int) and isinstance(x[1], int) and x[0] < x[1] and isinstance(x[2], str), f"bad span {x} in {d}"
        return f"{len(o)} docs, {sum(map(len, o.values()))} spans"
    if phase == 3:
        assert isinstance(o, dict) and set(o) >= {"CL", "IT"}, "expected keys CL and IT (CHECK this against the official Phase 3 schema)"
        return "keys " + ",".join(o) + " (schema is the pipeline's 'keyed' mode: verify against the spec)"
    if phase == 4:
        assert isinstance(o, list) and all(set(x) >= {"id", "pred"} and x["pred"] in (0, 1) for x in o), "expected [{id, pred in {0,1}}]"
        return f"{len(o)} cases, {sum(x['pred'] for x in o) / len(o):.1%} predicted 1"
    if phase == 5:
        assert isinstance(o, list) and all(set(x) >= {"id", "pred"} and isinstance(x["pred"], list) and len(set(x["pred"])) == len(x["pred"]) and all(isinstance(v, int) for v in x["pred"]) for x in o), "expected [{id, pred: [unique ints]}]"
        return f"{len(o)} cases, {sum(len(x['pred']) for x in o) / len(o):.2f} statutes/case"
    if phase == 6:
        assert isinstance(o, list) and all(set(x) >= {"id", "summary"} and isinstance(x["summary"], str) for x in o), "expected [{id, summary}]"
        return f"{len(o)} summaries, {sum(len(x['summary'].split()) for x in o) / max(len(o), 1):.0f} words on average"
    if phase == 7:
        assert isinstance(o, dict) and all(isinstance(v, list) and len(v) == 7 and len(set(v)) == 7 for v in o.values()), "expected {query_id: [7 unique candidate ids]}"
        return f"{len(o)} queries x 7"

def run_phase(phase):
    dst = os.path.join(PRED_DIR, f"phase_{phase}"); os.makedirs(dst, exist_ok=True)
    if os.path.exists(os.path.join(dst, "prediction.json")) and not RERUN:
        return dict(phase=phase, status="already done (kept)", minutes=0.0, check=validate(phase, os.path.join(dst, "prediction.json")))
    cap = CAP_MIN[phase]; script = child_script(phase, cap); log = os.path.join(dst, "log.txt"); t0 = time.time()
    env = dict(os.environ)
    if SELFTEST: env.update(P2_WORK=os.path.join(ROOT, "phase2_bilstm"), P2_RULES_PATH=os.path.join(ROOT, "p2_rules_core.py"))
    status = "ok"
    with open(log, "w", encoding="utf-8") as lf:
        pr = subprocess.Popen([sys.executable, "-u", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT)
        try:
            for line in pr.stdout:
                lf.write(line); print(line, end="")
                if time.time() - t0 > CHILD_TIMEOUT_MIN[phase] * 60: pr.kill(); status = f"STOPPED at the {CHILD_TIMEOUT_MIN[phase]}-minute hard limit"; break
        finally:
            if pr.poll() is None: pr.kill()
    out = find_output(phase); check = ""
    if out:
        try: check = validate(phase, out); shutil.copy2(out, os.path.join(dst, "prediction.json"))
        except Exception as e: status += f" | OUTPUT INVALID: {e}"; shutil.copy2(out, os.path.join(dst, "prediction_INVALID.json"))
    else: status += " | no prediction file was produced (read log.txt)"
    for extra in ("results.json", "prediction_model_only.json"):
        for h in glob.glob(os.path.join(ROOT, "**", extra), recursive=True):
            if "phase2_bilstm" in h.replace("\\", "/"): shutil.copy2(h, os.path.join(dst, extra))
    return dict(phase=phase, status=status, minutes=round((time.time() - t0) / 60, 1), check=check, source=out)

RESULTS = []
for ph in RUN_PHASES:
    print(f"\n######## PHASE {ph}  (notebook time so far {(time.time() - NB_T0) / 60:.0f} min) ########", flush=True)
    try: RESULTS.append(run_phase(ph))
    except Exception as e: traceback.print_exc(); RESULTS.append(dict(phase=ph, status=f"RUNNER ERROR: {e}", minutes=0.0, check=""))
    json.dump(RESULTS, open(os.path.join(PRED_DIR, "summary.json"), "w"), indent=1, default=str)

# ================= SUMMARY AND ZIP =================
print("\n================ SUMMARY ================")
for r in RESULTS: print(f"Phase {r['phase']}: {r['status']:<60} {r['minutes']:>6} min | {r['check']}")
print(f"total notebook time: {(time.time() - NB_T0) / 60:.0f} min")
zp = os.path.join(ROOT, "bilstm_predictions.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for h in glob.glob(os.path.join(PRED_DIR, "**", "*"), recursive=True):
        if os.path.isfile(h): z.write(h, os.path.relpath(h, PRED_DIR))
print("wrote", zp)

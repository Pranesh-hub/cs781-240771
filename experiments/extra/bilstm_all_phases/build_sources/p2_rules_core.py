import re, random

A_, R_ = 'A.COUNSEL', 'R.COUNSEL'

chosen = 'R'   # counsel default policy; select_defaults() re-chooses it on fold 1

ENT = ["APP","RESP","A.COUNSEL","R.COUNSEL","JUDGE","WIT","AUTH","COURT","STAT","PREC","DATE","CASENO"]

TYPES = ["O"] + ENT

WEAK = ["RESP", "A.COUNSEL", "R.COUNSEL", "WIT"]

HON = re.compile(r"\b(?:Mr|Ms|Mrs|Shri|Sri|Smt|Dr)\b\.?[ \t]*(?=[A-Z])")

HON_RE  = re.compile(r"\b(?:Mr|Ms|Mrs|Shri|Sri|Smt)\b\.?[ \t]*(?=[A-Z])")

NAME_TOK = re.compile(r"[A-Z][A-Za-z\.']*")

WS = re.compile(r"[ \t]+")

SIDE_A = re.compile(r"petitioner|appellant|applicant|plaintiff|complainant|prosecution", re.I)

SIDE_R = re.compile(r"respondent|defendant|accused", re.I)

def rule_spans(text):
    out = []
    for m in HON_RE.finditer(text):
        pos = m.end()
        if text[pos:pos + 7] == "Justice":          # judges' honorific, not counsel
            continue
        p, name_end = pos, None
        while True:
            t = NAME_TOK.match(text, p)
            if not t: break
            name_end = t.end()
            w = WS.match(text, name_end)
            if not w or not text[w.end():w.end() + 1].isupper(): break
            p = w.end()
        if name_end is None: continue
        e = name_end
        while e > pos and text[e - 1] in ".,;:'": e -= 1
        if e > pos: out.append((pos, e))
    return out

HON_BEFORE = re.compile(r"(?:Mr|Ms|Mrs|Shri|Sri|Smt)\.?[ \t]*$")

PARTY_A = r"(?:petitioners?|appellants?|applicants?|plaintiffs?|complainants?|prosecution|revisionists?)"

PARTY_R = r"(?:respondents?|defendants?|accused|non-\s*applicants?|opposite\s+part(?:y|ies))"

FOR = r"(?:for|on\s+behalf\s+of|appearing\s+for|representing)"

MODS = r"(?:(?:learned|first|second|third|\d+\w*|all|other)\s+)*"

CUE_A = re.compile(rf"\b{FOR}\s+(?:the\s+)?{MODS}{PARTY_A}", re.I)

CUE_R = re.compile(rf"\b{FOR}\s+(?:the\s+)?{MODS}{PARTY_R}", re.I)

HDR_A = re.compile(rf"{PARTY_A}s?[\s\.\-:]*(?:through|by)\s*[:\-]?\s*$", re.I)

HDR_R = re.compile(rf"{PARTY_R}s?[\s\.\-:]*(?:through|by)\s*[:\-]?\s*$", re.I)

HON_TAIL = re.compile(r"(?:Mr|Ms|Mrs|Shri|Sri|Smt)\.?[ \t]*$")

def _cues(text, lo, hi):
    out = [(m.start(), m.end(), A_) for m in CUE_A.finditer(text, lo, hi)]
    out += [(m.start(), m.end(), R_) for m in CUE_R.finditer(text, lo, hi)]
    return out

def assign_side(text, s, e, order, prev, W_L=300, W_R=200):
    for step in order:
        if step == "right":
            c = _cues(text, e, min(len(text), e + W_R))
            if c: return min(c)[2], "right"
        elif step == "left":
            c = _cues(text, max(0, s - W_L), s)
            c = [x for x in c if x[1] <= s]
            if c: return max(c, key=lambda x: x[1])[2], "left"
        elif step == "hdr":
            tail = HON_TAIL.sub("", text[max(0, s - 80):s])
            if HDR_A.search(tail): return A_, "hdr"
            if HDR_R.search(tail): return R_, "hdr"
        elif step == "inherit":
            if prev and s - prev[0] <= 200 and prev[2] != "default":
                return prev[1], "inherit"
    return R_, "default"

VARIANTS = {"V1 left-for": ["left"], "V2 right-for": ["right"], "V3 right>left": ["right", "left"],
            "V4 left>right": ["left", "right"], "V5 hdr>right>left>inh": ["hdr", "right", "left", "inherit"],
            "V6 right>hdr>left>inh": ["right", "hdr", "left", "inherit"]}

ORDER = ["hdr", "right", "left", "inherit"]

def counsel_spans(text, policy):
    out, prev = [], None
    for s, e in sorted(rule_spans(text)):
        lab, src = assign_side(text, s, e, ORDER, prev)
        prev = (e, lab, src)
        if src == "default":
            if policy == "drop": continue
            if policy == "A": lab = A_
        out.append((s, e, lab))
    return out

def counsel_f1(texts, gold, policy):
    G = {A_: set(), R_: set()}; P = {A_: set(), R_: set()}
    for d, ents in gold.items():
        for x in ents:
            if x[2] in G: G[x[2]].add((d, int(x[0]), int(x[1])))
    for d, text in texts.items():
        for s, e, lab in counsel_spans(text, policy): P[lab].add((d, s, e))
    f = {}
    for lab in (A_, R_):
        tp = len(G[lab] & P[lab]); p = tp / max(len(P[lab]), 1); r = tp / max(len(G[lab]), 1)
        f[lab] = 2 * p * r / (p + r) if p + r else 0.0
    return f

LOOK = ["APP", "RESP", "WIT", "JUDGE"]

TOK = r"(?:[A-Z]\.(?:\s?[A-Z]\.)*|[A-Z][a-z']+|[A-Z]{2,})"

TOK_RE = re.compile(TOK)

NAME = rf"{TOK}(?:[ \t]+{TOK}){{0,3}}"

STOP = {"JUDGMENT","JUDGEMENT","ORDER","BENCH","CORAM","BEFORE","PRESENT","HON'BLE","MR","MRS","MS",
        "JUSTICE","THE","AND","OF","DATE","NEW","DELHI","BY","DELIVERED"}

def back_name(text, end, max_toks=4):
    ws = max(0, end - 80)
    toks = [(m.start() + ws, m.end() + ws) for m in TOK_RE.finditer(text[ws:end])]
    run = []
    for a, b in reversed(toks):
        nxt = run[0][0] if run else end
        if text[b:nxt].strip(" \t") != "": break
        if text[a:b].upper().strip(".") in STOP: break
        run.insert(0, (a, b))
        if len(run) == max_toks: break
    return (run[0][0], run[-1][1]) if run else None

def trim(text, s, e):
    while s < e and (text[s].isspace() or text[s] in ".,;:"): s += 1
    while e > s and (text[e-1].isspace() or text[e-1] in ",;:"): e -= 1
    return (s, e) if e > s else None

A_RE = re.compile(rf"\bJustice[ \t]+({NAME})")

B_RE = re.compile(r",[ \t]*J{1,2}\b\.?")

C_RE = re.compile(r"\bJJ?\.?[ \t]*\(([^()\n]{3,45})\)")

D_RE = re.compile(r"(?:BENCH\s*:|\(J\)|\(J\.\))\s*([A-Z][A-Za-z']+,[ \t]*[A-Z](?:[ \t]*\.?[ \t]*[A-Z])*\.?)[ \t]*\(J\)")

CAP   = r"[A-ZÀ-Ö][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*"

INIT  = r"[A-Z]\.(?:[ \t]*[A-Z]\.)*"

WORD  = rf"(?:{INIT}|{CAP})"

NAME2 = rf"{WORD}(?:[ \t]+{WORD}){{0,3}}"

TOK_RE = re.compile(WORD)

AJ_RE = re.compile(rf"(?i:\bjustice)[ \t]+({NAME2})")

B_RE  = re.compile(r",[ \t]*J{1,2}\b\.?")

B2_RE = re.compile(r"[ \t]J{1,2}\.?(?=[ \t]*(?:in\b|speaking|writing|observed|held\b|was\b|said\b|while\b|who\b|delivered|dissent|agreed|also\b|noted|stated))")

C_RE  = re.compile(r"\bJJ?\.?[ \t]*\(([^()\n]{3,45})\)")

D_RE  = re.compile(r"(?:BENCH\s*:|\(J\)|\(J\.\))\s*([A-Z][A-Za-z']+,[ \t]*[A-Z](?:[ \t]*\.?[ \t]*[A-Z])*\.?)[ \t]*\(J\)")

E_RE  = re.compile(r"\bJJ?\.?[ \t]*\[[ \t]*([A-Za-zÀ-ÿ\.'’ \t\-]{3,45}?)[ \t]*\]")

F_RE  = re.compile(r"BENCH[ \t]*:[ \t]*([^\n]{3,120}?)(?=[ \t]*(?:JUDG(?:E)?MENT|CITATION|ACT\b|\n|$))")

NAMES = ["A Justice NAME", "B NAME, J.", "B2 NAME J (no comma)", "C J.(NAME)", "D BENCH: X, Y (J)", "E J [NAME]", "F BENCH: A & B"]

def judge_patterns2(text):
    out = {n: set() for n in NAMES}
    def add(n, s, e):
        sp = trim(text, s, e)
        if sp: out[n].add(sp)
    for m in AJ_RE.finditer(text): add(NAMES[0], m.start(1), m.end(1))
    for rx, n in [(B_RE, NAMES[1]), (B2_RE, NAMES[2])]:
        for m in rx.finditer(text):
            bn = back_name(text, m.start())
            if bn: add(n, *bn)
    for rx, n in [(C_RE, NAMES[3]), (E_RE, NAMES[5])]:
        for m in rx.finditer(text):
            inner = m.group(1).strip()
            if inner and inner[0].isupper() and inner.upper() not in {"J", "JJ"} and re.fullmatch(r"[A-Za-zÀ-ÿ\.'’, \-]+", inner):
                add(n, m.start(1), m.end(1))
    for m in D_RE.finditer(text): add(NAMES[4], m.start(1), m.end(1))
    for m in F_RE.finditer(text):
        seg, base, pos = m.group(1), m.start(1), 0
        for part in re.split(r"[ \t]*(?:&|\bAND\b)[ \t]*", seg):
            k = seg.find(part, pos)
            if k < 0: continue
            pos = k + len(part)
            part2 = re.sub(r"[ \t]*\((?:C?J|J\.?)\)\s*$", "", part)
            sp = trim(text, base + k, base + k + len(part2))
            if sp and re.fullmatch(NAME2, text[sp[0]:sp[1]]): out[NAMES[6]].add(sp)
    return out

def resolve(spans):
    out = []
    for s, e in sorted(set(spans), key=lambda x: (x[0], -(x[1] - x[0]))):
        if out and s < out[-1][1]: continue
        out.append((s, e))
    return out

def judge_spans_doc(text, keep):
    allsp = set()
    for n, sp in judge_patterns2(text).items():
        if n in keep: allsp |= sp
    return resolve(allsp)

def merge_all(texts, champ_filtered, keep):
    out = {}
    for d, text in texts.items():
        rules = list(counsel_spans(text, chosen)) + [(s, e, "JUDGE") for s, e in judge_spans_doc(text, keep)]
        kept = [k for k in champ_filtered[d] if k[2] not in (A_, R_, "JUDGE")]
        kept = [k for k in kept if not any(k[0] < re_ and k[1] > rs for rs, re_, _ in rules)]
        out[d] = sorted(set(kept + rules))
    return out
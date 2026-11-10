"""Extract the pure rule-layer code (counsel A/R side rule + JUDGE header rules + merge_all) from the champion+rules notebook.
Keeps only imports, function definitions and simple constant assignments; drops every top-level statement that evaluates, prints or needs a model."""
import ast, json, sys, builtins
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NB = "C:/Users/PRANESH S/Downloads/cs781_final/Phase2_Legal_NER/cs781_champion_plus_rules_run_cells.ipynb"
CELLS = [39, 41, 43, 44, 45, 46, 47, 48, 49, 50]
nb = json.load(open(NB, encoding="utf-8"))

def simple_value(node):
    """True if the assigned value only builds constants / regexes / literals (no calls except re.compile and simple builtins)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
            if name not in ("compile", "set", "frozenset", "tuple", "list", "dict", "sorted", "len", "str"): return False
        if isinstance(n, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            pass
    return True

kept, dropped = [], []
for ci in CELLS:
    src = "".join(nb["cells"][ci]["source"])
    tree = ast.parse(src)
    for node in tree.body:
        seg = ast.get_source_segment(src, node)
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            kept.append((ci, seg))
        elif isinstance(node, ast.Assign) and simple_value(node.value) and not any(
                isinstance(n, ast.Name) and n.id in ("train_texts", "train_gold", "val_texts", "val_gold", "champion_model", "model") for n in ast.walk(node.value)):
            kept.append((ci, seg))
        else:
            dropped.append((ci, (seg or "")[:70].replace("\n", " ")))

code = "\n\n".join(s for _, s in kept)
open(sys.argv[1], "w", encoding="utf-8").write(code)
print(f"kept {len(kept)} top-level blocks, dropped {len(dropped)}")
# which global names are used but never defined here?
defined = set(dir(builtins))
tree = ast.parse(code)
for n in ast.walk(tree):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)): defined.add(n.name)
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names: defined.add((a.asname or a.name).split(".")[0])
    elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): defined.add(n.id)
    elif isinstance(n, ast.arg): defined.add(n.arg)
    elif isinstance(n, ast.ExceptHandler) and n.name: defined.add(n.name)
used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
print("undefined globals:", sorted(used - defined))
print("defs:", [n.name for n in tree.body if isinstance(n, ast.FunctionDef)])

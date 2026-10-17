#!/usr/bin/env python3
"""Zero-LLM mapper: learned header dictionary + label rules, evaluated against the verified data.

  python3 qcl_rules.py eval      # leave-one-repo-out: the dictionary never sees the card it maps

Header dictionary: normalised column header -> metric family, learned from verified values that sit in that column.
Fallback: autoextract.header_family regexes. Roles from the row label: "this/ours/(this repo)/own file name" -> own;
another publisher's name -> cross; reference rows (bf16, baseline, teacher, reference) -> not ranked; else ladder."""
import json, re, sys, time, collections
import autoextract as A

CORE = {"kld", "top1", "ppl", "ppl_delta"}
def norm_h(h): return re.sub(r"\s+", " ", re.sub(r"\d+(\.\d+)?", "#", A.clean(h).lower())).strip()

def training(gt):
    """(repo, normalised header) -> family, from verified values found in table cells."""
    pairs = collections.defaultdict(collections.Counter)
    by_repo = collections.defaultdict(list)
    for g in gt: by_repo[g["repo"]].append(g)
    for rp, gs in by_repo.items():
        cands = A.extract(rp)
        for g in gs:
            if g["family"] not in CORE: continue
            for c in cands:
                if c["L"] in g["Ls"] and g["tok"] in c["toks"] and c["header"] not in ("(prose)", "(label)"):
                    pairs[norm_h(c["header"])][(rp, g["family"])] += 1
    return pairs

def dictionary(pairs, exclude_repo=None):
    d = {}
    for h, cnt in pairs.items():
        fam = collections.Counter()
        for (rp, f), n in cnt.items():
            if rp != exclude_repo: fam[f] += n
        if fam: d[h] = fam.most_common(1)[0][0]
    return d

OWNERS = None
NAMES = None   # repo -> distinctive quant names, as the fetcher's file listing provides (simulated from verified own variants)
def qtok(s):
    s = A.clean(s).lower()
    return {t for t in re.findall(r"(?:ud-)?(?:i?q\d[\w]*|\d+(?:\.\d+)?\s*-?bpw|nvfp4|mxfp4|fp8|w\d+a\d+|oq\d[\w.]*|\d-?bit|exl3|awq|gptq)", s)}
def build_names(R):
    global NAMES
    NAMES = collections.defaultdict(set)
    for r in R:
        for e in r.get("variants") or []: NAMES[r["repo"]] |= qtok(str(e.get("name") or "")) | qtok(r["repo"].split("/")[-1])
def role(repo, label):
    global OWNERS
    if OWNERS is None: OWNERS = {r.split("/")[0].lower() for r in A.IDX}
    l = A.clean(label).lower()
    if re.search(r"\b(bf16|fp16|fp32|baseline|reference|teacher|unquanti[sz]ed|original)\b", l) and not re.search(r"\bthis\b|\bours\b", l): return "ref"
    if re.search(r"\bthis (model|repo|checkpoint|quant|build|pack)\b|\(this|\bours\b|\bour\b", l): return "own"
    me = repo.split("/")[0].lower()
    if any(o != me and len(o) > 3 and o in l for o in OWNERS): return "cross"
    if NAMES is not None:
        lt = qtok(label)
        if lt & NAMES.get(repo, set()): return "own"
        if lt and any(lt & v for k, v in NAMES.items() if k.split("/")[0] != repo.split("/")[0]) and not lt & NAMES.get(repo, set()):
            if re.match(r"\s*ud-", l) or "unsloth" in l: return "cross"
    toks = [t for t in re.split(r"[^a-z0-9.]+", A.short(repo).lower() if hasattr(A, "short") else repo.split("/")[1].lower()) if len(t) > 2]
    if toks and sum(t in l for t in toks) >= max(1, len(toks) // 2): return "own"
    return "ladder"

def map_repo(repo, d):
    out = []
    for c in A.extract(repo):
        if c["header"] in ("(label)",): continue
        fam = d.get(norm_h(c["header"])) if c["header"] != "(prose)" else None
        if fam is None: fam = c["family"]
        if fam not in CORE: continue
        r = role(repo, c["label"]) if c["header"] != "(prose)" else "own"
        if r == "ref": continue
        for t in c["toks"][:1]: out.append((repo, fam, t, r, c))
    return out

def evaluate():
    t0 = time.time()
    R, gt = A.ground_truth()
    pairs = training(gt); build_names(R); t1 = time.time()
    repos = sorted({g["repo"] for g in gt if g["family"] in CORE})
    G = collections.Counter((g["repo"], g["family"], g["tok"]) for g in gt if g["family"] in CORE)
    kind = {(g["repo"], g["family"], g["tok"]): {"variants": "own", "ladder": "ladder", "cross": "cross"}[g["kind"]] for g in gt}
    P = collections.Counter(); roles = []; t2 = time.time()
    for rp in repos:
        d = dictionary(pairs, exclude_repo=rp)
        for (r, f, t, ro, c) in map_repo(rp, d):
            P[(r, f, t)] += 1
            if (r, f, t) in kind: roles.append((kind[(r, f, t)], ro))
    t3 = time.time()
    tp = sum((G & P).values()); extra = P - G
    print(f"repos {len(repos)}; ground-truth core values {sum(G.values())}; produced {sum(P.values())}; matched {tp}")
    print(f"recall {tp/sum(G.values()):.1%}  raw precision {tp/sum(P.values()):.1%}")
    agree = sum(1 for a, b in roles if a == b); print(f"role agreement on matched values: {agree}/{len(roles)} = {agree/max(1,len(roles)):.1%}")
    print(f"time: learn {t1-t0:.2f}s, map {len(repos)} repos {t3-t2:.2f}s ({(t3-t2)/len(repos)*1000:.1f} ms/repo)")
    return G, P

if __name__ == "__main__":
    if sys.argv[1:2] == ["eval"]: evaluate()

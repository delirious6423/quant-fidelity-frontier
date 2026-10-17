#!/usr/bin/env python3
"""Deterministic first-pass extractor + evaluation against the hand/agent-verified claims.

  python3 autoextract.py eval        # score against verified/part*.py (ground truth)
  python3 autoextract.py run REPO..  # print candidate claims for cards

Idea: most quality claims live in markdown tables. Code can copy every table cell verbatim with its line number,
bind it to a metric family from the column header and to a row label from the leading cells. An LLM is then only
needed to (a) map row labels to the publisher's file names / roles (own, ladder, cross, quoted), (b) read prose
claims, (c) state the method. Values are verbatim by construction, so the verbatim / column / sign gates pass by
construction for everything the code extracts."""
import json, glob, os, re, sys, collections

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("QCL_ROOT") or CODE          # data root: one folder per model (models/<slug>/); code stays here
IDX = json.load(open(f"{ROOT}/snapshots/index.json"))["repos"]

# column header -> metric family (order matters: most specific first)
HEADER_FAMILY = [
    ("ppl_delta", r"(ppl|perplex)\w*\s*(Δ|delta|change|diff|ratio|vs|increase|%)|Δ\s*(ppl|perplex)|ppl\s*ratio|ppl\(q\)/ppl"),
    ("top1", r"top[- ]?1|same top|argmax|token agree|agreement|top1|match rate|same[- ]token"),
    ("kld", r"\bkld?\b|kl[- ]?div|divergence|d_?kl|millinat|\bkl\b|mean kl|kl mean|\bkl\s*\("),
    ("ppl", r"\bppl\b|perplex"),
    ("size", r"\bsize\b|\bgb\b|\bgib\b|\bvram\b|memory|\bram\b|download|\bdisk\b|resident|weights|\bfile\b|footprint|bytes"),
    ("bpw", r"\bbpw\b|bits per|bits/w|\bbits\b"),
    ("speed", r"tok/s|t/s|tokens/s|tps|speed|throughput|latency|ms\b|prefill|decode"),
]
def header_family(h):
    h = clean(h).lower()
    for fam, rx in HEADER_FAMILY:
        if re.search(rx, h): return fam
    return "other"

NUM = re.compile(r"[−\-+]?\d{1,3}(?:[   ]\d{3})+(?:\.\d+)?|[−\-+]?\d[\d,]*(?:\.\d+)?|[−\-+]?\.\d+")
VOC_ANY = re.compile(r"\bkld?\b|kl\d|ce[- ]gap|teacher|kl[- ]?div|perplex|\bppl\b|top[- ]?1|agree|mmlu|gsm8k|humaneval|ifeval|gpqa|aime|accuracy|pass|score|bench|eval|error|err\b|identical|logit|correl|diff|accept|retain|quality|match|ratio|size|gib|gb\b|bpw", re.I)
def numtoks(s): return [re.sub(r"[,\s  ]", "", t).replace("−", "-").lstrip("+") for t in NUM.findall(s)]
def cells(line):
    s = line.strip()
    if not s.startswith("|"): return None
    s = re.sub(r"^\|", "", s); s = re.sub(r"(?<!\\)\|\s*$", "", s)
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]          # an escaped \| stays inside its cell
def clean(s):
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", s); s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"<[^>]+>", " ", s).replace("**", "").replace("`", "")
def is_sep(line): return re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line.strip()) is not None

def html_rows(lines):
    """<tr> rows written one per line (common in cards that embed HTML tables)."""
    hdr = None; rows = []
    for i, l in enumerate(lines):
        if "<tr" not in l: continue
        cs = [clean(c).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", l, re.S)]
        if not cs: continue
        if "<th" in l: 
            if rows: yield hdr, rows; rows = []
            hdr = cs
        elif hdr: rows.append((i + 1, cs))
    if hdr and rows: yield hdr, rows

def tables(lines):
    """Yield (header_cells, [(lineno, row_cells)]) for every markdown or HTML table. Line numbers are 1-based."""
    yield from html_rows(lines)
    i = 0
    while i < len(lines) - 1:
        if lines[i].strip().startswith("|") and is_sep(lines[i + 1]):
            hdr = cells(lines[i]); rows = []; j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append((j + 1, cells(lines[j]))); j += 1
            yield hdr, rows; i = j
        else: i += 1

def extract(repo):
    """Candidate claims: every numeric table cell with line, row label, column header, metric family, raw cell."""
    txt = open(f"{ROOT}/{IDX[repo]['file']}", encoding="utf-8", errors="replace").read()
    lines = txt.split("\n"); out = []
    for hdr, rows in tables(lines):
        cap = " ".join(lines[max(0, rows[0][0] - 6): rows[0][0] - 2]) if rows else ""
        if not (VOC_ANY.search(" ".join(hdr)) or VOC_ANY.search(cap) or any(VOC_ANY.search(" ".join(rc[:2])) for _, rc in rows)): continue
        fams = [header_family(h) for h in hdr]
        pivot = sum(f == "other" for f in fams[1:]) >= max(2, len(fams) - 2)   # variants as columns, metrics as rows
        for ln, rc in rows:
            first_num = next((k for k, c in enumerate(rc) if numtoks(c) and not re.search(r"[a-zA-Z]{3,}", re.sub(r"(GiB|GB|MB|bpw|%|nats?|x|bits?)", "", c))), len(rc))
            label = " ".join(rc[:max(1, first_num)])
            row_fam = header_family(label) if pivot else "other"
            for k, c in enumerate(rc):
                if k < first_num or not numtoks(c): continue
                if re.fullmatch(r"`?[0-9a-f]{7,64}`?|\d{4}-\d{2}-\d{2}.*", clean(c).strip()): continue      # hashes, commits, dates
                fam = fams[k] if k < len(fams) else "other"
                if (pivot or fam == "other") and row_fam != "other": fam = row_fam
                cc = clean(c)
                if fam in ("other", "speed", "bpw") and re.search(r"\d\s*(GiB|GB|MB|TB)\b", cc) and not re.search(r"/s\b", cc): fam = "size"
                if fam == "other" and re.search(r"\bvs\.?\s*(bf16|bfloat16|fp16|base|baseline|reference|q8)", clean(hdr[k] if k < len(hdr) else "").lower()) and "%" in cc:
                    fam = "ppl_delta" if any(f == "ppl" for f in fams) else "other"
                out.append({"L": ln, "label": label, "header": hdr[k] if k < len(hdr) else "", "family": fam, "raw": c, "toks": numtoks(c)})
        # bits per weight stated in the row label (e.g. "AD-4.27bpw-Q4_K_M")
    for hdr, rows in tables(lines):
        for ln, rc in rows:
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?bpw", clean(" ".join(rc[:2])), re.I):
                out.append({"L": ln, "label": clean(rc[0]), "header": "(label)", "family": "bpw", "raw": m.group(0), "toks": [m.group(1)]})
    # prose: "<metric word> ... <number>" within a short window, outside tables
    PROSE = [("kld", r"\b(?:mean\s+)?kld?\b|kl[- ]divergence"), ("top1", r"top[- ]?1(?:\s+agreement)?|same[- ]top"), ("ppl", r"\bppl\b|perplexity"),
             ("size", r"(?=\d[\d.,]*\s*(?:GiB|GB)\b)")]
    for i, l in enumerate(lines):
        if l.strip().startswith("|") or "<tr" in l: continue
        t = clean(l)
        for fam, rx in PROSE:
            for m in re.finditer(rx, t, re.I):
                w = t[m.start(): m.start() + 40] if fam != "size" else t[m.start(): m.start() + 16]
                nm = NUM.search(w[len(m.group(0)):] if fam != "size" else w)
                if nm: out.append({"L": i + 1, "label": "(prose)", "header": "(prose)", "family": fam, "raw": w.strip(), "toks": numtoks(nm.group(0))})
    return out

def digest(repo, ctx=1):
    """Compact evidence for the LLM: every table (header + all rows) and every prose line with a metric word and a number,
    plus +/- ctx lines of context and the method-ish lines. Line numbers kept so the LLM can cite them."""
    lines = open(f"{ROOT}/{IDX[repo]['file']}", encoding="utf-8", errors="replace").read().split("\n")
    keep = set()
    for i, l in enumerate(lines):
        if l.strip().startswith("|") or "<tr" in l:
            if i + 1 < len(lines) and is_sep(lines[i + 1]): keep.update(range(max(0, i - 3), i + 1))   # table header + caption; rows are in the candidate list
        
        elif l.strip().startswith("|") or "<tr" in l: pass
        elif NUM.search(l) and VOC_ANY.search(l): keep.update(range(max(0, i - ctx), min(len(lines), i + ctx + 1)))
        elif re.search(r"\b(reference|baseline|measured|evaluat|dataset|wikitext|context|ctx|chunks|tokens|methodology|harness|llama-perplexity|reported by|quoted|from their)\b", l, re.I): keep.add(i)
        elif re.match(r"\s*#", l): keep.add(i)
    return "\n".join(f"{i+1}: {lines[i]}" for i in sorted(keep))

# ---------- ground truth from the verified claims ----------
def gt_family(k):
    if k == "size": return "size"
    if k == "bpw": return "bpw"
    if re.match(r"^(ppl_delta|ppl_ratio)", k): return "ppl_delta"
    if re.match(r"^(top1|same_top1|top-1)", k): return "top1"
    if re.fullmatch(r"ppl(:.*)?", k): return "ppl"
    if (re.search(r"(^|:|_)(kld|kl)(_|:|$)|mean_kl|^kl", k)) and "99" not in k: return "kld"
    return "other"

def ground_truth():
    R = []
    for p in sorted(glob.glob(f"{ROOT}/verified/part*.py")):
        if p.endswith("part9_auto.py"): continue
        ns = {}; exec(open(p).read(), ns); R += ns["R"]
    gt = []
    for r in R:
        if r["repo"] not in IDX: continue
        for kind in ("variants", "ladder", "cross"):
            for e in r.get(kind) or []:
                Ls = [e["L"]] if isinstance(e.get("L"), int) else list(e.get("L") or [])
                Ls += list(e.get("L_extra") or [])
                if not Ls or e.get("src"): continue
                items = list((e.get("m") or {}).items())
                if e.get("size") is not None: items.append(("size", str(e["size"])))
                if e.get("bpw") is not None: items.append(("bpw", str(e["bpw"])))
                for k, v in items:
                    t = numtoks(str(v))
                    if t: gt.append({"repo": r["repo"], "Ls": Ls, "family": gt_family(k), "key": k, "tok": t[0], "kind": kind})
    return R, gt

def evaluate():
    R, gt = ground_truth()
    repos = sorted({g["repo"] for g in gt})
    cand = {rp: extract(rp) for rp in IDX}
    # recall: a GT value is recovered if a candidate on one of its lines carries the same leading number token
    #         (any family) / with the right family (family-correct)
    by = collections.Counter(); hit = collections.Counter(); fam_ok = collections.Counter(); in_table = collections.Counter()
    for g in gt:
        f = g["family"]; by[f] += 1
        cs = [c for c in cand[g["repo"]] if c["L"] in g["Ls"]]
        if cs: in_table[f] += 1
        m = [c for c in cs if g["tok"] in c["toks"]]
        if m: hit[f] += 1
        if any(c["family"] == f or (f == "other" and c["family"] in ("other", "speed")) for c in m): fam_ok[f] += 1
    tot = sum(by.values())
    print(f"ground-truth values: {tot} across {len(repos)} repos")
    print(f"{'family':10} {'n':>6} {'in a table':>11} {'copied':>8} {'copied+family':>14}")
    for f in ["kld", "top1", "ppl", "ppl_delta", "size", "bpw", "other"]:
        if by[f]: print(f"{f:10} {by[f]:6} {in_table[f]/by[f]:10.1%} {hit[f]/by[f]:8.1%} {fam_ok[f]/by[f]:13.1%}")
    print(f"{'ALL':10} {tot:6} {sum(in_table.values())/tot:10.1%} {sum(hit.values())/tot:8.1%} {sum(fam_ok.values())/tot:13.1%}")
    core = [f for f in ("kld", "top1", "ppl", "ppl_delta")]
    n = sum(by[f] for f in core); print(f"core quality metrics (KLD, top-1, PPL, PPL change): {n} values, {sum(fam_ok[f] for f in core)/n:.1%} copied with the right family")
    # precision of family labels among candidates on lines the verified data cites
    cited = collections.defaultdict(set)
    for g in gt: [cited[g["repo"]].add(L) for L in g["Ls"]]
    cand_core = [c for rp in repos for c in cand[rp] if c["family"] in core and c["L"] in cited[rp]]
    gtset = {(g["repo"], L, g["tok"], g["family"]) for g in gt for L in g["Ls"]}
    ok = sum(1 for rp in repos for c in cand[rp] if c["family"] in core and c["L"] in cited[rp] and any((rp, c["L"], t, c["family"]) in gtset for t in c["toks"]))
    print(f"precision of core-family labels on cited lines: {ok}/{len(cand_core)} = {ok/max(1,len(cand_core)):.1%}")
    # triage: which cards could skip the LLM entirely
    has_claims = {r["repo"] for r in R if any((e.get("m") or {}) for k in ("variants", "ladder", "cross") for e in (r.get(k) or []))}
    VOC = re.compile(r"\bkld?\b|kl[- ]?div|perplex|\bppl\b|top[- ]?1|agreement|mmlu|gsm8k|humaneval|ifeval|gpqa|aime|livecodebench|hellaswag|piqa|arc[-_ ]|accuracy|pass@|score|benchmark|eval|wikitext|\bnll\b|\bcer\b|\bwer\b|bleu|recall|fidelity|cosine|rmse|snr|error|err\b|rel[_ -]?err|pass(ed|es)?\b|cases|correct|exact|identical|logits?|correlation|\bdiff|acceptance|accept|retained|mass|tool[- ]call|mae\b|\bwin|\bloss|quality|degrad|match|ratio|\d+\s*/\s*\d+\s*(?:ok|pass|correct)", re.I)
    def needs_llm(rp):
        t = open(f"{ROOT}/{IDX[rp]['file']}", encoding="utf-8", errors="replace").read()
        return bool(re.search(r"!\[[^\]]*\]\([^)]+\.(png|jpe?g|webp|svg)", t, re.I)) or any(VOC.search(l) and NUM.search(l) for l in t.split("\n"))
    nl = {rp: needs_llm(rp) for rp in IDX}
    miss = sorted(rp for rp in has_claims if not nl[rp])
    skip = [rp for rp in IDX if not nl[rp]]
    print(f"triage: {len(skip)} of {len(IDX)} cards skip the LLM entirely; cards with claims wrongly skipped: {len(miss)} {miss[:5]}")
    json.dump({"skip": skip, "missed_claim_cards": miss}, open(f"{ROOT}/audit/autoextract_triage.json", "w"), indent=1)

STRICT = re.compile(r"\bkld?\b|kl[- ]?div|kl\d|perplex|\bppl\b|top[- ]?1|same[- ]top|token agree|argmax|mmlu|gsm8k|humaneval|ifeval|gpqa|aime|livecodebench|livebench|lcb\b|hellaswag|piqa|winogrande|arc[-_ ](c|e|challenge|easy)|mbpp|math-?500|bbh|truthfulqa|swe-?bench|tau2|terminal-?bench|pass@|\bnll\b|ce[- ]gap|teacher|wikitext|logit|rel[_ -]?err|cosine|snr\b|rmse|acceptance|tok(en)?s? (agree|match)|accuracy|\bacc\b|exact match|passed \d+|\d+\s*/\s*\d+\s*(ok|pass|correct|tasks|cases)", re.I)
NUMD = re.compile(r"\d+\.\d+|\d+\s*%|\d+\s*/\s*\d+")
def strict_need(t):
    """Wave 1: a metric word with a decimal/percent/count on the same line, or in a table's header or caption; or an image.
    On Flash-Next this skips no card with KLD, top-1 or PPL claims."""
    if re.search(r"!\[[^\]]*\]\([^)]+\.(png|jpe?g|webp|svg)", t, re.I): return True
    lines = t.split("\n")
    if any(STRICT.search(l) and NUMD.search(l) for l in lines): return True
    for hdr, rows in tables(lines):
        cap = " ".join(lines[max(0, rows[0][0] - 8): rows[0][0] - 2]) if rows else ""
        if (STRICT.search(" ".join(hdr)) or STRICT.search(cap)) and any(NUMD.search(" ".join(rc)) for _, rc in rows): return True
    return False

def triage():
    """Cards that need a mapper (any metric word near a number, or an image) vs cards that state no quality claim."""
    VOC = re.compile(r"\bkld?\b|kl[- ]?div|perplex|\bppl\b|top[- ]?1|agreement|mmlu|gsm8k|humaneval|ifeval|gpqa|aime|livecodebench|hellaswag|piqa|arc[-_ ]|accuracy|pass@|score|benchmark|eval|wikitext|\bnll\b|\bcer\b|\bwer\b|bleu|recall|fidelity|cosine|rmse|snr|error|err\b|rel[_ -]?err|pass(ed|es)?\b|cases|correct|exact|identical|logits?|correlation|\bdiff|acceptance|accept|retained|mass|tool[- ]call|mae\b|\bwin|\bloss|quality|degrad|match|ratio|\d+\s*/\s*\d+\s*(?:ok|pass|correct)", re.I)
    need, skip = [], []
    for rp, m in IDX.items():
        t = open(f"{ROOT}/{m['file']}", encoding="utf-8", errors="replace").read() if m.get("file") else ""
        img = bool(re.search(r"!\[[^\]]*\]\([^)]+\.(png|jpe?g|webp|svg)", t, re.I))
        hit = any(VOC.search(l) and NUM.search(l) for l in t.split("\n"))
        (need if (hit or img) else skip).append(rp)
    wave1 = sorted(rp for rp in need if strict_need(open(f"{ROOT}/{IDX[rp]['file']}", encoding="utf-8", errors="replace").read()))
    wave2 = sorted(set(need) - set(wave1))
    json.dump({"need": sorted(need), "wave1": wave1, "wave2": wave2, "skip": sorted(skip)}, open(f"{ROOT}/audit/triage.json", "w"), indent=1)
    print(json.dumps({"cards": len(IDX), "wave1_core_metrics": len(wave1), "wave2_other_claims": len(wave2), "no_claim_vocabulary": len(skip)}))

if __name__ == "__main__":
    if sys.argv[1:2] == ["triage"]: triage()
    elif sys.argv[1:2] == ["eval"]: evaluate()
    else:
        for rp in sys.argv[2:]:
            for c in extract(rp): print(c["L"], c["family"], "|", c["label"][:40], "|", c["header"][:25], "|", c["raw"])

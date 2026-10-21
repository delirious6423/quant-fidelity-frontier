#!/usr/bin/env python3
"""Gate 1c (cell-aware): when a cited line is a markdown table row, the value must sit in a column whose header
matches the metric it is filed under (KLD under a KL header, PPL under a PPL header, size under a size header...).
Catches swapped columns that a number-presence check cannot. Reviewed exceptions live in audit/column_exceptions.json.
Exit 1 on any unexplained mismatch."""
import json, glob, os, re, sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]   # sibling modules
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")          # data root: one folder per model (models/<slug>/); code stays here
IDX = json.load(open(f"{ROOT}/snapshots/index.json"))["repos"]
UNSLOTH = open(f"{ROOT}/snapshots/unsloth.ai__docs__qwen3.8-next.md").read().split("\n") if os.path.exists(f"{ROOT}/snapshots/unsloth.ai__docs__qwen3.8-next.md") else []
EXC = json.load(open(f"{ROOT}/audit/column_exceptions.json")) if os.path.exists(f"{ROOT}/audit/column_exceptions.json") else {}

FAMILY = [  # (metric-key test, header keywords that must appear)
    (lambda k: re.search(r"(^|:|_)(kld|kl)(_|:|$)|mean_kl|kld_|^kl", k) and "99" not in k, r"\bkl|kld|divergence|d_?kl|millinat"),
    (lambda k: k.startswith("top1") or k.startswith("top-1"), r"top|agree|same|match|argmax"),
    (lambda k: re.fullmatch(r"ppl(:.*)?", k) is not None, r"ppl|perplex"),
    (lambda k: k.startswith("ppl_delta") or k.startswith("ppl_ratio"), r"ppl|perplex|Δ|change|ratio|vs|base"),
    (lambda k: k == "size", r"bytes|backbone|trunk|gb|gib|size|vram|memory|\bram\b|download|disk|total|resident|weights|file|gpu|ssd"),
    (lambda k: k == "bpw", r"bpw|bits|bit"),
]

def cells(line):
    s = line.strip()
    if not s.startswith("|"): return None
    return [c.strip() for c in s.strip("|").split("|")]

def norm(s): return re.sub(r"\*\*|`|\\_", lambda m: "_" if m.group() == "\\_" else "", s)

def numtoks(s): return [re.sub(r"[,\s\u202f\u2009]", "", t) for t in re.findall(r"\d{1,3}(?:[ \u202f\u2009]\d{3})+(?:\.\d+)?|\d[\d,]*(?:\.\d+)?|\.\d+", s)]

def header_for(lines, i):
    """Nearest header row above table row i (row followed by a |---| separator)."""
    j = i - 1
    while j >= 1:
        if not lines[j].strip().startswith("|"): return None
        if re.fullmatch(r"\|?\s*:?-{2,}.*", lines[j].strip()) and lines[j - 1].strip().startswith("|"):
            return cells(lines[j - 1])
        j -= 1
    return None

def load():
    import current as CUR   # gates/current.py: entries whose card changed since review are withdrawn, not failed
    return CUR.load(ROOT)

# ---- row-label binding: a value taken from a table row must belong to the entry named for it ----
STOP = set("""the a an of and or vs in on at to for with by from this that our ours we us it its is are was build builds
model quant quants quantized file files repo pack checkpoint version gguf mlx exl3 qwen qwen3 qwen3.8 flash next
qwen3.8-flash-next flash-next bpw bit bits gb gib mb tb size new old current previous updated here row run""".split())
def _strip_md(s):
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", s)          # keep link text, drop URL
    s = re.sub(r"https?://\S+", " ", s)
    return s.replace("*", "").replace("`", "").replace("\\_", "_")
def _ltoks(s, stop):
    out = []
    for t in re.split(r"[^a-z0-9_.+]+", _strip_md(s).lower()):
        t = t.strip("._+")
        if len(t) < 2: continue                                  # single characters / digits ('4' of '4-bit') carry no identity
        if stop and t in STOP: continue
        out.append(t)
    return tuple(out)
def ltoks(s, stop=True): return list(_ltoks(s, stop))
def compact(s): return re.sub(r"[^a-z0-9]", "", _strip_md(s).lower())
def match_n(name_toks, label):
    lt = set(ltoks(label, stop=False)); lc = compact(label)
    return sum(1 for t in name_toks if t in lt or (len(compact(t)) >= 3 and compact(t) in lc))
def _lab(s): return re.sub(r"\s+", " ", _strip_md(s).lower()).strip()
def row_labels(e):
    rl = e.get("row_label")
    return [rl] if isinstance(rl, str) else list(rl or [])
def _shape(t): return re.sub(r"\d+(?:\.\d+)?", "#", t)
def _memoize():
    """The label helpers above are pure functions of their text. A table's header and row labels are scored against
    every entry of the document, so the same strings are normalized tens of thousands of times per model; caching them
    changes nothing but the run time (cached results are never mutated: ltoks returns a fresh list each call)."""
    from functools import lru_cache
    for name in ("_strip_md", "_ltoks", "compact", "_lab", "_shape"):
        globals()[name] = lru_cache(maxsize=None)(globals()[name])
_memoize()
def score(e, text):
    """How strongly a label/header text names entry e, as (matched, contradicted). An explicit row_label (the card's
    own wording for the entry; str or list) is matched as a substring and wins outright. Otherwise matched =
    distinctive name tokens found in the text, and contradicted = name tokens absent from the text where the text
    has a different token of the same shape (Q5_K_M vs Q4_K_M, 4bpw vs 2bpw; MCTiny vs 'Small' is no shape clash,
    so pivoted columns like perkel's need row_label)."""
    rls = row_labels(e)
    if rls and any(_lab(x) and _lab(x) in _lab(text) for x in rls): return (100, 0)
    t = ltoks(e.get("name") or e.get("variant") or "")
    lt = set(ltoks(text, stop=False)); lc = compact(text)
    lt |= {p for y in list(lt) for p in y.split(".") if len(p) >= 2}       # 'q4.gguf' also offers 'q4'
    m = c = 0
    for x in t:
        if x in lt or (len(compact(x)) >= 3 and compact(x) in lc): m += 1
        elif re.search(r"[a-z]", x) and re.search(r"\d", x) and any(_shape(y) == _shape(x) and y != x for y in lt): c += 1
    return (m, c)
def names(e, text):
    """Does this text clearly name entry e (used only to decide whether a table names its entries at all)?"""
    if row_labels(e) and score(e, text)[0] >= 100: return True
    t = ltoks(e.get("name") or e.get("variant") or "")
    m, c = score(e, text)
    return bool(t) and c == 0 and m >= max(1, len(t) - len(t) // 3)
def _tok_score(e, text):
    return score({k: v for k, v in e.items() if k != "row_label"}, text)
def owns(e, others, text):
    """e owns a label/header/cell text if its own row_label occurs in it, or if its name tokens match with no
    shape clash and no other entry matches more tokens (or carries this exact text as its row_label)."""
    if row_labels(e) and score(e, text)[0] >= 100: return True
    m, c = _tok_score(e, text)
    if m <= 0 or c > 0: return False
    if any(_lab(text) in [_lab(x) for x in row_labels(o)] for o in others): return False
    return m >= max([_tok_score(x, text)[0] for x in others if _tok_score(x, text)[1] == 0] + [0])
def names_nobody(entries_, text): return all(score(x, text)[0] <= 0 for x in entries_)
def _sub(needle, hay):
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))
_DOCN = {}
def DOCN(doc):
    k = id(doc)
    if k not in _DOCN or _DOCN[k][0] is not doc: _DOCN[k] = (doc, [norm(x) for x in doc])
    return _DOCN[k][1]
def table_span(doc_n, i):
    a = i
    while a > 0 and doc_n[a - 1].strip().startswith("|"): a -= 1
    b = i
    while b + 1 < len(doc_n) and doc_n[b + 1].strip().startswith("|"): b += 1
    return a, b
def table_rows(doc_n, a, b):
    rows = []
    for j in range(a, b + 1):
        if re.fullmatch(r"\|?\s*:?-{2,}.*", doc_n[j].strip()): continue
        if j + 1 <= b and re.fullmatch(r"\|?\s*:?-{2,}.*", doc_n[j + 1].strip()): continue   # header row
        c = cells(doc_n[j])
        if c: rows.append(c)
    return rows
def combined_corpus_cell(doc_n, i, header, cell, key, value):
    """Bind an explicitly documented KL/top-1 pair to its exact corpus column.

    Corpus-only headers do not normally establish a metric's meaning. A preceding
    paragraph can do so for the narrow ``mean KL, then top-1`` cell format, but
    each metric must still select its own component, never merely occur in it.
    """
    metric = re.fullmatch(r"(kl|kld|top1|top-1):(.+)", key)
    if not metric: return False
    corpus = lambda text: re.sub(r"\s+", "_", _strip_md(text).strip().lower())
    if corpus(metric[2]) != corpus(header): return False
    number = r"(?:\d+(?:\.\d+)?|\.\d+)"
    pair = re.fullmatch(rf"({number})\s*·\s*({number})\s*%", norm(cell).strip())
    if not pair: return False
    component = 0 if metric[1] in ("kl", "kld") else 1
    raw_value = norm(str(value)).strip()
    if not re.fullmatch(number + (r"\s*%?" if component else ""), raw_value): return False
    if numtoks(raw_value) != [pair[component + 1]]: return False
    a, _ = table_span(doc_n, i)
    j = a - 1
    while j >= 0 and not doc_n[j].strip(): j -= 1
    end = j + 1
    while j >= 0 and doc_n[j].strip() and not doc_n[j].lstrip().startswith(("#", "|")):
        j -= 1
    paragraph = re.sub(r"\s+", " ", _strip_md(" ".join(doc_n[j + 1:end]))).lower()
    return re.search(r"\beach cell shows (?:the )?mean kl(?: lower bound)?(?: in nats)?, then top-1 agreement\b", paragraph) is not None
def caption_above(doc_n, i):
    """The paragraph directly above the table that holds row i (blank lines skipped; stops at a heading or table)."""
    a, _ = table_span(doc_n, i)
    j = a - 1
    while j >= 0 and not doc_n[j].strip(): j -= 1
    end = j + 1
    while j >= 0 and doc_n[j].strip() and not doc_n[j].lstrip().startswith(("#", "|")): j -= 1
    return re.sub(r"\s+", " ", _strip_md(" ".join(doc_n[j + 1:end]))).strip()
def corpus_column_caption(doc_n, i, header, cell, key, value, fam):
    """A per-corpus value (kld:<corpus>, ppl:<corpus>, top1:<corpus>) in a table whose columns are corpora and whose
    metric is named only in the caption right above it ("Mean KL-divergence vs the bf16 model: | | wikitext-2 | ...").
    Accepted only when the column header is exactly the filed corpus, the cell holds just this one number, and the
    caption names this metric and no other metric family (so the caption cannot be about a different column)."""
    m = re.fullmatch(r"(kld|kl|top1|ppl):(.+)", key)
    if not m: return False
    corpus = lambda text: re.sub(r"[^a-z0-9]", "", _strip_md(text).lower())
    if not corpus(m[2]) or corpus(m[2]) != corpus(header): return False
    if numtoks(norm(cell)) != numtoks(norm(str(value))) or len(numtoks(norm(cell))) != 1: return False
    cap = caption_above(doc_n, i)
    if not cap or not re.search(fam, cap, re.I): return False
    others = [kw for test, kw in FAMILY[:3] if kw != fam]   # KL, top-1, perplexity
    return not any(re.search(kw, cap, re.I) for kw in others)

def label_of(row, first_val_col):
    return " | ".join(row[:max(1, min(first_val_col, 3))])
label_bad, label_checked, label_pivot, label_single = [], 0, 0, 0

bad, checked = [], 0
for r in load():
    meta = IDX.get(r["repo"])
    if not meta: continue
    src = open(f"{ROOT}/{meta['file']}", encoding="utf-8", errors="replace").read().split("\n")
    entries = [v for v in r.get("variants", [])] + list(r.get("cross") or []) + list(r.get("ladder") or [])
    for e in entries:
        if e.get("src") == "chart_image": continue
        doc = UNSLOTH if e.get("src") == "unsloth_docs" else src
        # -- row-label binding (every cited table row that holds one of the entry's values) --
        doc_n = DOCN(doc)
        ename = e.get("name") or e.get("variant") or ""
        same_doc = [x for x in entries if (x.get("src") == "unsloth_docs") == (e.get("src") == "unsloth_docs") and x.get("src") != "chart_image"]
        others = [x for x in same_doc if x is not e]
        vals_all = dict(e.get("m") or {})
        for k in ("size", "bpw"):
            if e.get(k) is not None: vals_all[k] = e[k]
        place = {}            # value key -> list of (line, ok, where)
        lines_seen = []
        for Li in [e.get("L")] + list(e.get("L_extra") or []):
            if not isinstance(Li, int) or not (0 < Li <= len(doc)): continue
            row = cells(doc_n[Li - 1])
            if not row or len(row) < 2: continue
            hdr = header_for(doc_n, Li - 1)
            if not hdr or len(hdr) != len(row): continue
            located = {}
            for k, v in vals_all.items():
                want = numtoks(str(v))
                if not want: continue
                cols = [ci for ci in range(1, len(row)) if _sub(want, numtoks(row[ci]))]
                if cols: located[k] = (v, cols)
            if not located: continue
            fv = min(c for _, cols in located.values() for c in cols)
            label = label_of(row, fv)
            ta, tb = table_span(doc_n, Li - 1)
            rows = table_rows(doc_n, ta, tb)
            texts = [rw[0] for rw in rows if rw] + hdr[1:]
            names_entries = any(names(x, t) for t in texts for x in same_doc)
            citers = sum(1 for x in same_doc if any(isinstance(j, int) and ta < j <= tb + 1 for j in [x.get("L")] + list(x.get("L_extra") or [])))
            if not names_entries and citers < 2:
                label_single += 1; continue                 # single-subject table naming no entry: bound by its caption, not checkable here
            label_checked += 1
            lines_seen.append(Li)
            row_ok = owns(e, others, label)
            pivot = (not row_ok) and any(score(e, h)[0] > 0 and score(e, h)[1] == 0 for h in hdr[1:])
            if pivot: label_pivot += 1
            for k, (v, cols) in located.items():
                if row_ok or any(owns(e, others, row[ci]) for ci in cols): ok, where = True, f"row '{label[:40]}'"   # or the value's own cell names it ('v1: 88.6%')
                elif pivot:
                    ok = any(owns(e, others, hdr[ci]) or names_nobody(same_doc, hdr[ci]) for ci in cols)
                    where = f"column(s) {[hdr[ci] for ci in cols]}"
                else: ok, where = False, f"row '{label[:60]}'"
                place.setdefault(k, []).append((Li, ok, where))
        for k, pl in place.items():
            if any(ok for _, ok, _ in pl): continue
            Ls_ = sorted({Li for Li, _, _ in pl})
            key = f"rowlabel|{r['repo']}|{ename}|L{Ls_[0]}"
            if key in EXC or f"rowlabel|{r['repo']}|{ename}|{k}" in EXC: continue
            label_bad.append(f"{key}: {k}={vals_all[k]!r} found only at {[(Li, w) for Li, _, w in pl]}, none of which names {ename!r}; set row_label= if the card labels it differently")
        L = e.get("L")
        if not isinstance(L, int) or not (0 < L <= len(doc)): continue
        row = cells(norm(doc[L - 1]))
        if not row: continue
        hdr = header_for([norm(x) for x in doc], L - 1)
        if not hdr or len(hdr) != len(row): continue
        vals = dict(e.get("m") or {})
        for k in ("size", "bpw"):
            if e.get(k) is not None: vals[k] = e[k]
        for k, v in vals.items():
            fam = next((kw for test, kw in FAMILY if test(k)), None)
            if not fam: continue
            want = numtoks(str(v))
            if not want: continue
            cols = [ci for ci, c in enumerate(row) if want[0] in numtoks(c)]
            if not cols: continue                       # value is not in this row (cited via L_extra); presence gate covers it
            checked += 1
            if any(combined_corpus_cell(doc_n, L - 1, hdr[ci], row[ci], k, v) for ci in cols): continue
            if any(corpus_column_caption(doc_n, L - 1, hdr[ci], row[ci], k, v, fam) for ci in cols): continue
            if any(re.search(fam, hdr[ci], re.I) or re.search(fam, row[ci], re.I) for ci in cols): continue
            if row and re.search(fam, row[0], re.I) and 0 not in cols: continue          # key/value table: label in first cell
            if re.search(fam, k.split(":", 1)[-1], re.I) and any(re.search(fam, h, re.I) for h in hdr) is False and \
               re.search(fam, " ".join(row[:1] + hdr[:1]), re.I): continue
            key = f"{r['repo']}|{e.get('name') or e.get('variant')}|{k}"
            if key in EXC: continue
            bad.append(f"{key} = {v!r} sits under header {[hdr[ci] for ci in cols]} (expected /{fam}/) at L{L}")
print(f"table-cell bindings checked: {checked}; unexplained mismatches: {len(bad)}")
for b in bad: print("  MISMATCH", b)
print(f"row-label bindings checked: {label_checked} (pivoted-column checks: {label_pivot}; single-subject tables naming no entry, not checkable: {label_single}); unexplained mismatches: {len(label_bad)}")
for b in label_bad: print("  ROWLABEL", b)
sys.exit(1 if bad or label_bad else 0)

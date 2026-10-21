#!/usr/bin/env python3
"""Gate 1 (mechanical): every published value we cite must appear verbatim on its cited line(s)
of the pinned snapshot. No model judgment involved. Exit 1 on any failure."""
import hashlib, json, glob, os, re, sys
from pathlib import Path
from decimal import Decimal

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]   # sibling modules
import current as CUR
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")          # data root: one folder per model (models/<slug>/); code stays here
IDX = json.load(open(f"{ROOT}/snapshots/index.json"))["repos"]
SIZES = json.load(open(f"{ROOT}/snapshots/hf_sizes_summary.json"))
BRANCH = json.load(open(f"{ROOT}/snapshots/hf_branch_sizes.json"))
UNSLOTH = []  # loaded only through a content pin when an active claim cites it

def load_parts():
    return CUR.load(ROOT)

def pinned_lines(meta):
    """Read the exact indexed source bytes, never merely trust its commit label."""
    path = (Path(ROOT) / meta["file"]).resolve()
    if not path.is_relative_to(Path(ROOT).resolve()):
        raise ValueError("source path is outside the model directory")
    raw = path.read_bytes()
    digest = meta.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        raise ValueError("source has no valid pinned SHA-256")
    if hashlib.sha256(raw).hexdigest() != digest.lower():
        raise ValueError("source SHA-256 differs from its pinned bytes")
    if "bytes" in meta and meta["bytes"] != len(raw):
        raise ValueError("source byte count differs from its pin")
    return raw.decode("utf-8").split("\n")

def norm(s):  # strip markdown emphasis / code / escapes / thin spaces, collapse whitespace
    s = s.replace("**", "").replace("`", "").replace("\\_", "_").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", s)

def numtoks(s):
    return [re.sub(r"[,\s\u202f\u2009]", "", t) for t in re.findall(r"\d{1,3}(?:[ \u202f\u2009]\d{3})+(?:\.\d+)?|\d[\d,]*(?:\.\d+)?|\.\d+", s)]

def fmt_num(x):
    if isinstance(x, float) and x.is_integer(): return [str(int(x)), f"{x:.1f}", f"{x:.2f}"]
    if isinstance(x, (int,)): return [str(x), f"{x}.0", f"{x}.00"]
    if isinstance(x, float):
        s = repr(x); outs = {s}
        for d in range(len(s.split(".")[1]) if "." in s else 0, 5): outs.add(f"{x:.{d}f}")
        return list(outs)
    return [str(x)]

def check_value(val, lines):
    """True if the published string (or number) appears in any of the given lines."""
    text = " || ".join(norm(l) for l in lines)
    if isinstance(val, str):
        v = norm(val)
        if v in text: return True
        # allow table cell variants: '86.95 ± 0.14 %' vs '86.95 ± 0.14%'
        if v.replace(" %", "%") in text.replace(" %", "%"): return True
        # every number token of the value must appear as a whole token on the line(s)
        toks = numtoks(v); have = set(numtoks(text))
        return bool(toks) and all(t in have for t in toks)
    if isinstance(val, (int, float)):
        have = set(numtoks(text))
        return any(c in have for c in fmt_num(val))
    return False


# ---- sign gate: a published value that starts with a sign must appear on the line with the SAME sign ----
MINUS = "−–‒—﹣－"
def unminus(s): return re.sub(f"[{MINUS}]", "-", s)
SIGNED = re.compile(r"(?<![\w.])([+-])\s?(\d[\d,]*(?:\.\d+)?|\.\d+)")
def signed_tok(val):
    """'+0.1270%' -> '+0.1270'; '−0.16 %' -> '-0.16'; None if the value carries no leading sign."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return f"-{abs(val)}" if val < 0 else None
    if not isinstance(val, str): return None
    m = re.match(r"^\s*([+-])\s?(\d[\d,]*(?:\.\d+)?|\.\d+)", unminus(norm(val)))
    return (m.group(1) + m.group(2).replace(",", "")) if m else None
def check_sign(val, lines):
    want = signed_tok(val)
    if want is None:
        # An unsigned value is positive, not an excuse to drop a source minus.
        # This check only rejects a matching negative magnitude; the presence
        # and column gates remain responsible for locating the value itself.
        raw = unminus(norm(str(val))).strip()
        pattern = r"\d{1,3}(?:[ \u202f\u2009]\d{3})+(?:\.\d+)?|\d[\d,]*(?:\.\d+)?|\.\d+"
        number = re.match(pattern, raw)
        if number is None: return True
        magnitude = Decimal(numtoks(number.group())[0])
        tokens = re.compile(r"(?<![\w.\d])([+-]?)\s*(" + pattern + r")")
        negative = False
        for line in lines:
            # A Markdown bullet or spaced em dash is punctuation, not a minus.
            text = re.sub(r"^\s*[-*+]\s+", "", norm(line))
            text = re.sub(r"[−﹣－]|[–‒—](?=\d|\.\d)", "-", text)
            for match in tokens.finditer(text):
                if Decimal(numtoks(match[2])[0]) != magnitude: continue
                if match[1] != "-": return True
                negative = True
        return not negative
    have = set()
    for l in lines:
        for m in SIGNED.finditer(unminus(norm(l))):
            have.add(m.group(1) + m.group(2).replace(",", ""))
    if want in have: return True
    if isinstance(val, (int, float)):   # numeric literal: compare as numbers
        return any(h.startswith("-") and abs(float(h) - float(val)) < 1e-12 for h in have)
    return False

# ---- unit gate: the size's unit must stand next to the size number on the cited line (or head its table column) ----
UNIT_RE = re.compile(r"(?<![A-Za-z])(GiB|GB|MiB|MB|TiB|TB|KiB|KB|bytes?|B)(?![A-Za-z])")
def unit_of(tok):
    t = tok.lower()
    return {"gib": "GiB", "gb": "GB", "mib": "MiB", "mb": "MB", "tib": "TiB", "tb": "TB", "kib": "KiB", "kb": "KB",
            "byte": "bytes", "bytes": "bytes", "b": "bytes"}[t]
UNIT_SCAN = re.compile(r"(?<![A-Za-z])(GiB|GB|MiB|MB|TiB|TB|KiB|KB|bytes?)(?![A-Za-z])", re.I)
NUM_RE = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:\.\d+)?|\d[\d,]*(?:\.\d+)?|\.\d+")
def _cells(line):
    s = line.strip()
    return [c.strip() for c in s.strip("|").split("|")] if s.startswith("|") else None
def _header(doc_lines, i):
    j = i - 1
    while j >= 1:
        if not doc_lines[j].strip().startswith("|"): return None
        if re.fullmatch(r"\|?\s*:?-{2,}.*", doc_lines[j].strip()) and doc_lines[j - 1].strip().startswith("|"):
            return _cells(norm(doc_lines[j - 1]))
        j -= 1
    return None
TO_BYTES = {"bytes": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}
def units_near_size(size, doc, Ls, unit=None, win=12):
    """Units printed next to an occurrence of the size number on the cited lines: a unit token within `win` chars
    (no other number in between, same table cell), else the unit in the column header for a bare table cell, else a
    conversion of the same quantity printed on the line in another unit (e.g. '97,059,902,320 (90.4 GiB)')."""
    cands = set(fmt_num(size)); found = []
    for Li in Ls:
        if not (isinstance(Li, int) and 0 < Li <= len(doc)): continue
        line = norm(doc[Li - 1]); row = _cells(line); hdr = _header(doc, Li - 1) if row else None
        nums = [(m, re.sub(r"[,\s\u202f\u2009]", "", m.group())) for m in NUM_RE.finditer(line)]
        for m, tok in nums:
            if tok not in cands: continue
            if m.start() > 0 and line[m.start() - 1] in ".0123456789": continue
            chain = re.match(r"(?:\s*(?:/|→|->|–|-|to)\s*~?\d[\d.,]*)+\s?(GiB|GB|MiB|MB|TiB|TB|bytes?)(?![A-Za-z])",
                             line[m.end(): m.end() + win + 6].split("|")[0], re.I)    # '118.1 / 162.8 GiB', '80.78 → 78.27 GB'
            if chain: found.append(unit_of(chain.group(1))); continue
            after = line[m.end(): m.end() + win].split("|")[0]; before = line[max(0, m.start() - win): m.start()].split("|")[-1]
            after = re.split(r"\d", after)[0]; before = re.split(r"\d", before)[-1]
            ua = UNIT_SCAN.search(after) or re.match(r"\s?(B)(?![A-Za-z])", after); ub = list(UNIT_SCAN.finditer(before))
            if ua: found.append(unit_of(ua.group(1))); continue
            if ub: found.append(unit_of(ub[-1].group(1))); continue
            if row and hdr and len(hdr) == len(row):          # bare number in a table cell: unit given by the column header
                pos = line[:m.start()].count("|") - (1 if line.lstrip().startswith("|") else 0)
                if 0 <= pos < len(hdr):
                    hu = UNIT_SCAN.findall(hdr[pos])
                    if hu: found.append(unit_of(hu[-1])); continue
            if unit in TO_BYTES:                               # a printed conversion of the same quantity corroborates the unit
                for m2, tok2 in nums:
                    if m2 is m: continue
                    u2 = UNIT_SCAN.match(re.split(r"\d", line[m2.end(): m2.end() + 6])[0].strip())
                    if not u2 or unit_of(u2.group(1)) == unit: continue
                    d = len(tok2.split(".")[1]) if "." in tok2 else 0
                    conv = float(size) * TO_BYTES[unit] / TO_BYTES[unit_of(u2.group(1))]
                    if abs(conv - float(tok2)) <= 0.5 * 10 ** -d + 1e-9:
                        found.append(unit + " (by conversion)"); break
    return found

# ---- quoted-context gate: cross/ladder rows whose context says the numbers are someone else's must say so ----
QUOTED_RE = re.compile(r"reported by|\bquoted\b|\bquotes?\b|per (?:their|the(?:ir)? publisher'?s?|the) card|as published|according to|"
                       r"from (?:their|the(?:ir)?|its) (?:own )?card|publisher[- ]reported|their (?:own )?numbers|"
                       r"their (?:own )?(?:published|reported) (?:numbers|figures|results)|reprint", re.I)

fails, passes, notes, image_claims = [], 0, [], 0
sign_checked = sign_fails = unit_checked = unit_fails = quoted_ctx = 0
R = load_parts()
external_doc = "snapshots/unsloth.ai__docs__qwen3.8-next.md"
external_claims = [entry for record in R for kind in ("variants", "cross", "ladder")
                   for entry in (record.get(kind) or []) if entry.get("src") == "unsloth_docs"]
external_claims += [record[key] for record in R for key in ("baseline", "baseline2")
                    if isinstance(record.get(key), dict) and record[key].get("src") == "unsloth_docs"]
if external_claims:
    try:
        registry = json.load(open(f"{ROOT}/audit/review_pins.json"))
        UNSLOTH = pinned_lines({**registry["sources"][external_doc], "file": external_doc})
    except (OSError, ValueError, KeyError, TypeError) as error:
        fails.append(f"external unsloth_docs: SOURCE {error}")
seen = set()
for r in R:
    name = r["repo"]; seen.add(name)
    meta = IDX.get(name)
    if not meta: fails.append(f"{name}: not in snapshot index"); continue
    if not r.get("auto") and not CUR.fresh(r, IDX):
        fails.append(f"{name}: reviewed source does not match the current snapshot")
    try:
        src_lines = pinned_lines(meta)
    except (OSError, ValueError, KeyError, TypeError) as error:
        fails.append(f"{name}: SOURCE {error}")
        continue
    def lines_for(e):
        doc = UNSLOTH if e.get("src") == "unsloth_docs" else src_lines
        Ls = [e.get("L")] + list(e.get("L_extra", []) or [])
        return [doc[i-1] for i in Ls if isinstance(i, int) and 0 < i <= len(doc)], [i for i in Ls if i is not None]
    groups = [("variant", v) for v in r.get("variants", [])] + [("cross", v) for v in r.get("cross", []) or []] \
           + [("ladder", v) for v in r.get("ladder", []) or []]
    for bk in ("baseline","baseline2"):
      if isinstance(r.get(bk), dict): groups.append((bk, {"name": bk, "m": {k: v for k, v in r[bk].items() if k not in ("L", "L_extra", "src")}, "L": r[bk].get("L"), "L_extra": r[bk].get("L_extra"), "src": r[bk].get("src")}))
    for kind, e in groups:
        vals = dict(e.get("m") or {})
        for k in ("size", "bpw", "mem"):
            if e.get(k) is not None: vals[k] = e[k]
        if not vals: continue
        if e.get("src") == "chart_image":
            image_claims += len([k for k in vals if k not in ("size",)])   # values checked by verify_images.py
            vals = {k: x for k, x in vals.items() if k == "size"} if e.get("basis") == "hf_branch_total" else {}
            if not vals: continue
        if e.get("size") is not None and not e.get("unit"):
            unit_fails += 1; fails.append(f"{name} [{kind}] {e.get('name') or e.get('variant')}: size {e['size']} has no unit")
        if kind in ("cross", "ladder") and e.get("src") != "chart_image":
            doc_q = UNSLOTH if e.get("src") == "unsloth_docs" else src_lines
            ctx = []
            for Li in [e.get("L")] + list(e.get("L_extra") or []):
                if isinstance(Li, int) and 0 < Li <= len(doc_q):
                    ctx += [(j + 1, doc_q[j]) for j in range(max(0, Li - 4), Li)]
                    if doc_q[Li - 1].strip().startswith("|"):          # table row: also the 3 lines above the table (its caption)
                        h = Li - 1
                        while h > 0 and doc_q[h - 1].strip().startswith("|"): h -= 1
                        ctx += [(j + 1, doc_q[j]) for j in range(max(0, h - 3), h)]
            hits = sorted({j for j, l in ctx if QUOTED_RE.search(norm(l))})
            if hits:
                quoted_ctx += 1
                if "measured" not in e or not isinstance(e["measured"], bool):
                    fails.append(f"{name} [{kind}] {e.get('name') or e.get('variant')}: QUOTED context at L{hits} ('{QUOTED_RE.search(norm(doc_q[hits[0]-1])).group()}') but no explicit measured=True/False")
        if e.get("measured") is not None and not isinstance(e.get("measured"), bool):
            fails.append(f"{name} [{kind}] {e.get('name') or e.get('variant')}: measured must be True or False")
        if kind in ("cross", "ladder") and any(re.search(r"quoted|not measured", str(f), re.I) for f in list(e.get("flags") or []) + [e.get("name"), e.get("variant"), e.get("note")] if f) and e.get("measured") is not False:
            fails.append(f"{name} [{kind}] {e.get('name') or e.get('variant')}: flag says quoted/not measured but measured is not False")
        if e.get("basis") == "hf_branch_total":
            branch = BRANCH[name][e["branch"]]
            unit = e.get("unit")
            ok = unit in TO_BYTES and abs(branch["bytes"] / TO_BYTES[unit] - e["size"]) < 0.02
            (passes := passes + 1) if ok else fails.append(f"{name} {e.get('name')}: branch size or UNIT mismatch")
            vals.pop("size", None)
            if not vals: continue
        lines, Ls = lines_for(e)
        if e.get("basis") == "hf_listing_total" and e.get("size") is not None:
            s = SIZES[name]; unit = e.get("unit")
            column = {"GB": 0, "GiB": 1}.get(unit)
            candidates = ([s[f"total_{unit}"]] + [group[column] for group in s["groups"].values()]) if column is not None else []
            ok = any(abs(value - e["size"]) < 0.02 for value in candidates)
            (passes := passes + 1) if ok else fails.append(f"{name} [{kind}] {e.get('name')}: size {e['size']} {unit} not in HF listing (value/UNIT mismatch)")
            vals.pop("size", None)
        if vals and not lines:
            fails.append(f"{name} [{kind}] {e.get('name')}: values but no valid line refs {Ls}"); continue
        for k, v in vals.items():
            if not check_value(v, lines):
                fails.append(f"{name} [{kind}] {e.get('name') or e.get('variant')}.{k} = {v!r} NOT FOUND on L{Ls}: {norm(' || '.join(lines))[:200]}")
            elif not check_sign(v, lines):
                sign_fails += 1
                fails.append(f"{name} [{kind}] {e.get('name') or e.get('variant')}.{k} = {v!r} SIGN: signed token {signed_tok(v)!r} not on L{Ls}: {norm(' || '.join(lines))[:200]}")
            else:
                passes += 1
                if signed_tok(v) is not None: sign_checked += 1
        if "size" in vals and isinstance(e.get("size"), (int, float)):
            doc = UNSLOTH if e.get("src") == "unsloth_docs" else src_lines
            unit_checked += 1
            u = e.get("unit")
            got = units_near_size(e["size"], doc, Ls, u)
            got = [g.replace(" (by conversion)", "") for g in got]
            ul = e.get("unit_L")        # explicit citation of the line where the card states the unit for a bare number (caption/heading)
            if ul is not None and not got:
                ok_ul = isinstance(ul, int) and 0 < ul <= len(doc) and u and \
                        {unit_of(t) for t in UNIT_SCAN.findall(norm(doc[ul - 1]))} == {u}
                if ok_ul: got = [u]
                else:
                    unit_fails += 1; fails.append(f"{name} [{kind}] {e.get('name') or e.get('variant')}: UNIT unit_L={ul} does not state {u} unambiguously for a bare size: {norm(doc[ul-1])[:120] if isinstance(ul,int) and 0<ul<=len(doc) else ''}")
                    continue
            if not u:
                pass                     # already reported above as 'has no unit'
            elif u not in got:
                unit_fails += 1
                fails.append(f"{name} [{kind}] {e.get('name') or e.get('variant')}: UNIT size {e['size']} {u} — line(s) L{Ls} give {sorted(set(got)) or 'no unit'} next to it: {norm(' || '.join(lines))[:160]}")
    for key in ("methodology",):
        m = r.get(key)
        if m and m.get("L"):
            doc = src_lines
            bad = [i for i in m["L"] if not (0 < i <= len(doc))]
            if bad: fails.append(f"{name}: methodology line refs out of range {bad}")

# A repository the collector installed (pipeline/qcl_collect.py) that no mapping run has read yet is pending, not missing:
# it has nothing to publish and is left off the site until the next live-updates cycle maps it. A repository that was
# mapped (maps/mapped.json) and still has no entry is a gap in the data and fails the gate as before.
try:
    MAPPED = set(json.load(open(f"{ROOT}/maps/mapped.json")))
except (OSError, ValueError):
    MAPPED = set()
pending = sorted((set(IDX) - seen) - MAPPED)
missing = sorted(set(IDX) - seen - set(pending))
print(f"repos covered: {len(seen)}/{len(IDX)}" + (f"  PENDING MAPPING: {len(pending)}" if pending else "") + (f"  MISSING: {missing}" if missing else ""))
print(f"values verified verbatim: {passes}")
print(f"image-sourced values (see verify_images.py): {image_claims}")
print(f"signed values checked for sign: {sign_checked} (sign failures: {sign_fails})")
print(f"sizes checked for unit next to the number: {unit_checked} (unit failures: {unit_fails})")
print(f"cross/ladder entries with quoted-context wording: {quoted_ctx}")
print(f"failures: {len(fails)}")
for f in fails: print("  FAIL", f)
sys.exit(1 if fails or missing else 0)

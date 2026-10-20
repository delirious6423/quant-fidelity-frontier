#!/usr/bin/env python3
"""Map-don't-type extraction.

  python3 qcl_map.py tasks OUT_DIR REPO...     # write one compact task file per card (candidates + digest)
  python3 qcl_map.py apply MAP.json...         # turn LLM mappings into repo(...) entries (values copied by code)
  python3 qcl_map.py score MAP.json...         # compare applied entries with the verified data (ground truth)

The LLM never types a number. It receives every numeric table cell as a numbered candidate (cell id, line, header,
row label, raw text) and answers with a small JSON mapping: which candidate is which metric of which entry, the
entry's role (own / ladder / cross / quoted), and the method. Code then copies the published strings verbatim.
"""
import hashlib, json, os, re, sys, collections
import autoextract as A
import mapping_provenance as MP

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]   # sibling modules
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")          # data root: one folder per model (models/<slug>/); code stays here

SCHEMA = """Answer with ONE JSON object, no prose:
{"repo": "<repo id>",
 "source_sha256": "<copy the Source SHA-256 from this task exactly>",
 "candidates_sha256": "<copy the Candidates SHA-256 from this task exactly>",
 "no_quality_claims": false,          // true if the card states no quality number about any quant (speed/size-only cards count as true)
 "no_claims_reason": "",              // when true: one short reason with line numbers, e.g. "speed table only [L40-52]"
 "weights_modified": false, // true for pruned, finetuned, abliterated or otherwise modified base weights
 "method": {"reference": "...", "dataset": "...", "ctx": "...", "tool": "...", "L": [..]},   // as stated; "not stated" if absent
 "entries": [
   {"role": "own|ladder|cross|quoted",  // own = this repo's file(s); ladder = this publisher's comparison builds;
                                        // cross = another publisher's file THIS card's author measured; quoted = a figure the card copies from elsewhere
    "name": "<file/variant as the card labels it>",
    "panel": "<exact test-panel section heading when this card contains multiple measurement panels; omit for one panel>",
    "methodology": {"reference": "...", "dataset": "...", "ctx": "...", "tool": "...", "L": []}, // optional entry-specific method; distinguish separate panels

    "of": "<owner/repo of the measured file, only for cross/quoted; write 'not named on card' if the card names no repo>",
    "metrics": {"<metric key>": "<candidate id>"},
       // keys: mean_kld | kld:<corpus> | top1_pct | top1:<corpus> | ppl | ppl:<corpus> | ppl_delta_pct | ppl_ratio | kld_99 | bench:<Name>
       // use the corpus form whenever a card reports a metric per corpus/domain (e.g. kld:code, top1:english, ppl:wikitext);
       // millinat KL goes to mean_kl_mnats / kl_mnats:<corpus>. Ids look like "c17".
    "size": "<candidate id or null>",   // the size printed in the SAME table/row as the metrics when there is one
    "unit": "GB|GiB|bytes|null", "basis": "disk|vram|resident|ram_in_use|text_weights|excl_engram|decoder_only|engram_only|routed_experts|null",
    "bpw": "<candidate id or null>",    // bits per weight if the card states it (table cell or a cNN taken from the row label)
    "flags": ["<short neutral note with [Lnn]>"]}
 ],
 "prose": [{"entry": "<name>", "panel": "<same panel as target entry, omit for one panel>", "key": "<metric key>", "L": <line>, "raw": "<exact substring of that line>"}]   // only for values NOT in the candidate list
}
Read the WHOLE prompt before answering; if a tool shows it truncated, read the rest with an offset.
Rules: never invent or round a value; use candidate ids for every table value; a value the card attributes to someone
else ("reported by", "from their card", "publisher-reported") is role "quoted"; results measured on a different model
are omitted and mentioned in a flag; if a table's metric is ambiguous, leave it out and add a flag.
Separate panels need separate entries with panel set to the exact source section heading and entry methodology.
Do not mix top-k KL with full-vocabulary KL, legacy panels with current panels, or domain-equal with position-weighted means.
Native MXFP4/FP8 checkpoints are not BF16 references. For split storage, distinguish decoder-only (basis decoder_only or excl_engram as stated) from total download.
Read the full source card as well when the digest omits measurement context."""

def task(repo):
    pin = source_pin(repo)
    cands = A.extract(repo)
    seen = {}
    for c in cands:
        k = (c["L"], c["raw"])
        if k not in seen: seen[k] = c
    cl = []
    for c in seen.values():                       # one id per number: "88.6 / 89.0 / 96.0" -> c7.0, c7.1, c7.2
        parts = [m.group(0) for m in A.NUM.finditer(A.clean(c["raw"]))]
        if len(parts) > 1 and not re.search(r"±|\+/-|\[|\(", c["raw"]):
            for j, t in enumerate(parts): cl.append(dict(c, raw=t, sub=j, cell=c["raw"]))
        else: cl.append(c)
    n = 0; last = None
    for c in cl:
        if "sub" in c:
            if last != (c["L"], c.get("cell")): n += 1; last = (c["L"], c.get("cell"))
            c["id"] = f"c{n}.{c['sub']}"
        else: n += 1; last = None; c["id"] = f"c{n}"
    text = "\n".join(f'{c["id"]} L{c["L"]} [{c["header"][:30]}] row="{c["label"][:50]}" ' + (f'value="{c["raw"]}" of cell="{c["cell"][:60]}"' if "sub" in c else f'cell="{c["raw"][:40]}"') for c in cl)
    digest = A.digest(repo)
    if source_pin(repo) != pin:
        raise ValueError(f"source changed while preparing mapping task: {repo}")
    candidates = {c["id"]: {"L": c["L"], "raw": c["raw"], "header": c["header"], "label": c["label"]} for c in cl}
    candidate_pin = MP.candidates_digest(candidates)
    return {"repo": repo, "source_sha256": pin, "candidates_sha256": candidate_pin, "candidates": candidates,
            "prompt": f"Card: {repo}\nSource SHA-256: {pin}\nCandidates SHA-256: {candidate_pin}\n\n## Numeric table cells (candidates)\n{text}\n\n## Card digest (line: text)\n{digest}\n\n## Output\n{SCHEMA}"}

def clean_raw(s): return re.sub(r"\*\*|`", "", s).strip()

def clean_label(label):
    code = []
    def keep(m):
        code.append(m.group(1))
        return f"QFFINLINECODE{len(code)-1}PLACEHOLDER"
    text = A.clean(re.sub(r"`([^`]+)`", keep, label)).strip()
    for i, value in enumerate(code): text = text.replace(f"QFFINLINECODE{i}PLACEHOLDER", value)
    return text

QUAL = re.compile(r"(?:(?:^|[\s(|:=,])(?:<|>|≤|≥)=?|~|≈|∼|±|\babout|\baround|\bapprox(?:imately|\.)?|\broughly|\bup to|\bunder|\bover|\bnearly|\balmost)\s*$", re.I)   # not "<td>" or "->"
def exact_in(raw, line):
    """True when raw occurs in line as a whole value: not inside a longer number, and not right after a qualifier
    (~, ≈, <, about, up to, ...), which the published string would otherwise silently drop."""
    for m in re.finditer(re.escape(raw), line):
        before, after = line[:m.start()], line[m.end():]
        if raw[:1].isdigit() and re.search(r"[\d.,]$", before): continue
        if raw[-1:].isdigit() and re.match(r"[\d]|[.,]\d", after): continue
        if raw[:1] not in "~≈<>≤≥" and QUAL.search(before): continue
        return True
    return False
APPROX = re.compile(r"(?:~|≈|∼|\babout|\baround|\bapprox|\broughly|\bup to|(?:^|[\s(])(?:<|>|≤|≥))\s*\d|\d\s*(?:[A-Za-z]{1,4}\s*)?(?:–|—|\bto\b)\s*~?\d|\d\s+-\s+\d", re.I)

# Images on a card can carry the results (charts) or details (tables) that the text lacks. A model cannot read them in a
# mapping run, so code flags the card instead of leaving the gap silent. Badges, logos and shields are ignored.
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)[^)]*\)|<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"'][^>]*>", re.I)
IMAGE_SKIP = re.compile(r"shields\.io|badge|badgen|logo|avatar|emoji|star-history|visitor|favicon|icon|buymeacoffee|ko-fi|\.svg(\?|$)", re.I)

def card_images(lines):
    """[(line number, alt text, url)] for the content images on a card (badges and logos left out)."""
    found = []
    for n, line in enumerate(lines, 1):
        for m in IMAGE_RE.finditer(line):
            alt, url = (m.group(1), m.group(2)) if m.group(2) else ("", m.group(3))
            if url and not IMAGE_SKIP.search(url) and not IMAGE_SKIP.search(alt or ""): found.append((n, (alt or "").strip(), url))
    return found

def image_flag(lines, has_claims):
    """One flag line for a card whose images may hold quant results or details, or None."""
    imgs = card_images(lines)
    if not imgs: return None
    where = ", ".join(f"L{n}" for n, _, _ in imgs[:6]) + (" …" if len(imgs) > 6 else "")
    names = "; ".join(sorted({a for _, a, _ in imgs if a})[:4])
    lead = "Card embeds" if has_claims else "No numbers in the card text, but it embeds"
    return (f"{lead} {len(imgs)} image{'s' if len(imgs) != 1 else ''} [{where}]{(' (' + names + ')') if names else ''} that may show quant results or details; "
            "image contents are not read by the mapper, so anything shown only in them is not on this site")

def apply(mapping, cands, *, source_sha256=None):
    """-> repo(...) dict in the verified/ schema. Raw strings come from the candidate cells, never from the LLM."""
    ent = {"variants": [], "ladder": [], "cross": []}
    for e in mapping.get("entries", []):
        m = {}; Ls = set(); labels = set()
        for k, cid in (e.get("metrics") or {}).items():
            c = cands.get(str(cid))
            if not c: continue
            m[k] = clean_raw(c["raw"]); Ls.add(c["L"])
            if c.get("label") and c["label"] != "(prose)": labels.add(clean_label(c["label"]))
        d = {"name" if e["role"] in ("own", "ladder") else "variant": e.get("name"), "m": m}
        sc = cands.get(str(e.get("size"))) if e.get("size") else None
        if sc:
            if sc.get("label") and sc["label"] != "(prose)": labels.add(clean_label(sc["label"]))
            try: line = A_lines(mapping["repo"])[sc["L"] - 1] if sc.get("L") else ""
            except (NameError, KeyError, IndexError, OSError): line = ""   # no card text (unit tests): the raw-string check below still applies
            core = clean_raw(sc["raw"])
            ranged = any(re.match(r"\s*(?:–|—|-|to)\s*~?\d", line[m.end():]) or re.match(r"\s*[A-Za-z]{1,4}\s*(?:–|—|to)\s*~?\d", line[m.end():]) or re.search(r"\d\s*(?:–|—|-|to)\s*$", line[:m.start()])
                         for m in re.finditer(re.escape(core), line)) if core in line else False
            if APPROX.search(sc["raw"]) or ranged or (core in line and not exact_in(core, line)):
                # an approximate size or a range is not a point: hold it rather than publish one end of it
                d["size_held"] = f"approximate or range on card: {clean_raw(sc['raw'])} [L{sc['L']}]"; sc = None
        if sc:
            t = A.numtoks(sc["raw"]); d["size"] = float(t[0]) if t else None; d["unit"] = e.get("unit"); d["basis"] = e.get("basis"); Ls.add(sc["L"])
            u = d["unit"]
            if u and not re.search(rf"\d\s*{u}\b", sc["raw"]) and u.lower() not in (sc.get("header") or "").lower():
                lines = A_lines(mapping["repo"])          # the unit is stated elsewhere: cite the nearest caption naming exactly one unit
                order = sorted(range(max(0, sc["L"] - 40), min(len(lines), sc["L"] + 25)), key=lambda j: abs(j - (sc["L"] - 1)))
                for j in order:
                    if j == sc["L"] - 1 or lines[j].strip().startswith("|"): continue
                    us = set(re.findall(r"\b(GiB|GB|MB|TB|bytes)\b", lines[j]))
                    if us == {u}: d["unit_L"] = j + 1; break
        bc = cands.get(str(e.get("bpw"))) if e.get("bpw") else None
        if bc:
            t = A.numtoks(bc["raw"]); d["bpw"] = float(t[0]) if t else None; Ls.add(bc["L"])
        Ls = sorted(Ls)
        if Ls: d["L"] = Ls[0]; d["L_extra"] = Ls[1:]
        if e.get("flags"): d["flags"] = e["flags"]
        if e.get("panel"): d["panel"] = e["panel"]
        if e.get("methodology"): d["methodology"] = e["methodology"]
        if isinstance(e.get("measured"), bool): d["measured"] = e["measured"]
        if labels: d["row_label"] = sorted(labels)          # the card's own row wording, read by code
        if e["role"] in ("cross", "quoted"):
            d["of"] = e.get("of") or "not named on card"
            if e["role"] == "quoted": d["measured"] = False
            ent["cross"].append(d)
        else: ent["variants" if e["role"] == "own" else "ladder"].append(d)
    for p in mapping.get("prose", []) or []:
        line = A_lines(mapping["repo"])[p["L"] - 1] if p.get("L") else ""
        if p.get("raw") and exact_in(p["raw"], line):      # prose must be a whole, unqualified value on its cited line
            tgt = next((x for k in ent for x in ent[k] if (x.get("name") or x.get("variant")) == p.get("entry") and x.get("panel") == p.get("panel")), None)
            if tgt is not None:
                tgt["m"][p["key"]] = p["raw"]
                if "L" not in tgt: tgt["L"] = p["L"]; tgt["L_extra"] = []
                elif p["L"] != tgt["L"] and p["L"] not in tgt.get("L_extra", []): tgt.setdefault("L_extra", []).append(p["L"])
    return {"repo": mapping["repo"], "methodology": mapping.get("method"), "weights_modified": mapping.get("weights_modified"), **ent,
            **({"source_sha256": source_sha256} if source_sha256 is not None else {}),
            "flags": ([f"No quality claims for the quant (card read in full): {mapping.get('no_claims_reason','')}"] if mapping.get("no_quality_claims") else [])}

_lc = {}
def source_pin(repo):
    """Pin the exact source consumed by an application; refresh its line cache."""
    meta = A.IDX[repo]
    path = os.path.realpath(os.path.join(ROOT, meta["file"]))
    if os.path.commonpath([os.path.realpath(ROOT), path]) != os.path.realpath(ROOT):
        raise ValueError(f"source outside model directory: {repo}")
    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != meta.get("sha256"):
        raise ValueError(f"source bytes changed before mapping: {repo}")
    _lc[repo] = raw.decode("utf-8").split("\n")
    return digest

def A_lines(repo):
    if repo not in _lc: source_pin(repo)
    return _lc[repo]

def score(applied):
    """Value-level comparison with the verified data: same repo, same leading number token, same metric family, same role."""
    R, gt = A.ground_truth()
    repos = {a["repo"] for a in applied}
    G = [g for g in gt if g["repo"] in repos and g["family"] != "other"]
    role = {"variants": "own", "ladder": "ladder", "cross": "cross"}
    P = []
    for a in applied:
        for kind in ("variants", "ladder", "cross"):
            for e in a[kind]:
                for k, v in e["m"].items():
                    t = A.numtoks(v)
                    if t: P.append((a["repo"], A.gt_family(k), t[0], role[kind] if not e.get("measured") is False else "quoted"))
                if e.get("size") is not None: P.append((a["repo"], "size", A.numtoks(str(e["size"]))[0], role[kind]))
                if e.get("bpw") is not None: P.append((a["repo"], "bpw", A.numtoks(str(e["bpw"]))[0], role[kind]))
    gtr = {"variants": "own", "ladder": "ladder", "cross": "cross"}
    Gs = collections.Counter((g["repo"], g["family"], g["tok"]) for g in G)
    Ps = collections.Counter((p[0], p[1], p[2]) for p in P if p[1] != "other")
    tp = sum((Gs & Ps).values())
    print(f"core+size values in ground truth: {sum(Gs.values())}; produced: {sum(Ps.values())}; matched: {tp}")
    print(f"recall {tp/max(1,sum(Gs.values())):.1%}  precision {tp/max(1,sum(Ps.values())):.1%}")
    # role agreement on matched values
    GR = {(g["repo"], g["family"], g["tok"]): gtr[g["kind"]] for g in G}
    rolem = [(GR[(p[0], p[1], p[2])], p[3]) for p in P if (p[0], p[1], p[2]) in GR]
    agree = sum(1 for a, b in rolem if a == b or (a == "cross" and b == "quoted"))
    print(f"role agreement on matched values: {agree}/{len(rolem)} = {agree/max(1,len(rolem)):.1%}")
    by = collections.defaultdict(lambda: [0, 0])
    for (rp, f, t), n in Gs.items(): by[rp][0] += n; by[rp][1] += min(n, Ps.get((rp, f, t), 0))
    for rp, (n, h) in sorted(by.items(), key=lambda x: x[1][1] / max(1, x[1][0])):
        print(f"   {h}/{n}  {rp}")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tasks":
        out = sys.argv[2]; os.makedirs(out, exist_ok=True)
        for rp in sys.argv[3:]:
            t = task(rp); fn = rp.replace("/", "__")
            json.dump({key: t[key] for key in ("repo", "source_sha256", "candidates_sha256", "candidates")}, open(f"{out}/{fn}.cands.json", "w"))
            open(f"{out}/{fn}.prompt.txt", "w").write(t["prompt"])
            print(fn, len(t["candidates"]), "candidates", len(t["prompt"]), "chars")
    elif cmd in ("apply", "score"):
        applied = []
        for p in sys.argv[2:]:
            m = json.load(open(p)); c = json.load(open(MP.candidate_path(p)))
            pin = MP.mapping_source(m, c, A.IDX, MP.recorded_pins(ROOT, os.path.basename(os.path.dirname(p))), path=p)
            if source_pin(m["repo"]) != pin: raise ValueError("mapped source changed before application")
            applied.append(apply(m, MP.candidate_values(c), source_sha256=pin))
        if cmd == "apply": print(json.dumps(applied, indent=1)[:4000])
        else: score(applied)

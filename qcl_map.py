#!/usr/bin/env python3
"""Map-don't-type extraction.

  python3 qcl_map.py tasks OUT_DIR REPO...     # write one compact task file per card (candidates + digest)
  python3 qcl_map.py apply MAP.json...         # turn LLM mappings into repo(...) entries (values copied by code)
  python3 qcl_map.py score MAP.json...         # compare applied entries with the verified data (ground truth)

The LLM never types a number. It receives every numeric table cell as a numbered candidate (cell id, line, header,
row label, raw text) and answers with a small JSON mapping: which candidate is which metric of which entry, the
entry's role (own / ladder / cross / quoted), and the method. Code then copies the published strings verbatim.
"""
import json, os, re, sys, collections
import autoextract as A

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("QCL_ROOT") or CODE          # data root: one folder per model (models/<slug>/); code stays here

SCHEMA = """Answer with ONE JSON object, no prose:
{"repo": "<repo id>",
 "no_quality_claims": false,          // true if the card states no quality number about any quant (speed/size-only cards count as true)
 "no_claims_reason": "",              // when true: one short reason with line numbers, e.g. "speed table only [L40-52]"
 "method": {"reference": "...", "dataset": "...", "ctx": "...", "tool": "...", "L": [..]},   // as stated; "not stated" if absent
 "entries": [
   {"role": "own|ladder|cross|quoted",  // own = this repo's file(s); ladder = this publisher's comparison builds;
                                        // cross = another publisher's file THIS card's author measured; quoted = a figure the card copies from elsewhere
    "name": "<file/variant as the card labels it>",
    "of": "<owner/repo of the measured file, only for cross/quoted; write 'not named on card' if the card names no repo>",
    "metrics": {"<metric key>": "<candidate id>"},
       // keys: mean_kld | kld:<corpus> | top1_pct | top1:<corpus> | ppl | ppl:<corpus> | ppl_delta_pct | ppl_ratio | kld_99 | bench:<Name>
       // use the corpus form whenever a card reports a metric per corpus/domain (e.g. kld:code, top1:english, ppl:wikitext);
       // millinat KL goes to mean_kl_mnats / kl_mnats:<corpus>. Ids look like "c17".
    "size": "<candidate id or null>",   // the size printed in the SAME table/row as the metrics when there is one
    "unit": "GB|GiB|bytes|null", "basis": "disk|vram|resident|ram_in_use|text_weights|null",
    "bpw": "<candidate id or null>",    // bits per weight if the card states it (table cell or a cNN taken from the row label)
    "flags": ["<short neutral note with [Lnn]>"]}
 ],
 "prose": [{"entry": "<name>", "key": "<metric key>", "L": <line>, "raw": "<exact substring of that line>"}]   // only for values NOT in the candidate list
}
Read the WHOLE prompt before answering; if a tool shows it truncated, read the rest with an offset.
Rules: never invent or round a value; use candidate ids for every table value; a value the card attributes to someone
else ("reported by", "from their card", "publisher-reported") is role "quoted"; results measured on a different model
are omitted and mentioned in a flag; if a table's metric is ambiguous, leave it out and add a flag."""

def task(repo):
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
    return {"repo": repo, "candidates": {c["id"]: {"L": c["L"], "raw": c["raw"], "header": c["header"], "label": c["label"]} for c in cl},
            "prompt": f"Card: {repo}\n\n## Numeric table cells (candidates)\n{text}\n\n## Card digest (line: text)\n{A.digest(repo)}\n\n## Output\n{SCHEMA}"}

def clean_raw(s): return re.sub(r"\*\*|`", "", s).strip()

def apply(mapping, cands):
    """-> repo(...) dict in the verified/ schema. Raw strings come from the candidate cells, never from the LLM."""
    ent = {"variants": [], "ladder": [], "cross": []}
    for e in mapping.get("entries", []):
        m = {}; Ls = set(); labels = set()
        for k, cid in (e.get("metrics") or {}).items():
            c = cands.get(str(cid))
            if not c: continue
            m[k] = clean_raw(c["raw"]); Ls.add(c["L"])
            if c.get("label") and c["label"] != "(prose)": labels.add(A.clean(c["label"]).strip())
        d = {"name" if e["role"] in ("own", "ladder") else "variant": e.get("name"), "m": m}
        sc = cands.get(str(e.get("size"))) if e.get("size") else None
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
        if labels: d["row_label"] = sorted(labels)          # the card's own row wording, read by code
        if e["role"] in ("cross", "quoted"):
            d["of"] = e.get("of") or "not named on card"
            if e["role"] == "quoted": d["measured"] = False
            ent["cross"].append(d)
        else: ent["variants" if e["role"] == "own" else "ladder"].append(d)
    for p in mapping.get("prose", []) or []:
        line = A_lines(mapping["repo"])[p["L"] - 1] if p.get("L") else ""
        if p.get("raw") and p["raw"] in line:      # prose must be an exact substring of its cited line
            tgt = next((x for k in ent for x in ent[k] if (x.get("name") or x.get("variant")) == p.get("entry")), None)
            if tgt is not None:
                tgt["m"][p["key"]] = p["raw"]
                if "L" not in tgt: tgt["L"] = p["L"]; tgt["L_extra"] = []
                elif p["L"] != tgt["L"] and p["L"] not in tgt.get("L_extra", []): tgt.setdefault("L_extra", []).append(p["L"])
    return {"repo": mapping["repo"], "methodology": mapping.get("method"), **ent,
            "flags": ([f"No quality claims for the quant (card read in full): {mapping.get('no_claims_reason','')}"] if mapping.get("no_quality_claims") else [])}

_lc = {}
def A_lines(repo):
    if repo not in _lc: _lc[repo] = open(f"{ROOT}/{A.IDX[repo]['file']}", encoding="utf-8", errors="replace").read().split("\n")
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
            json.dump(t["candidates"], open(f"{out}/{fn}.cands.json", "w"))
            open(f"{out}/{fn}.prompt.txt", "w").write(t["prompt"])
            print(fn, len(t["candidates"]), "candidates", len(t["prompt"]), "chars")
    elif cmd in ("apply", "score"):
        applied = []
        for p in sys.argv[2:]:
            m = json.load(open(p)); c = json.load(open(p.replace(".map.json", ".cands.json")))
            applied.append(apply(m, c))
        if cmd == "apply": print(json.dumps(applied, indent=1)[:4000])
        else: score(applied)

#!/usr/bin/env python3
"""Ingest a saved browser result (gzip+base64 payload between GZB64: and :END) into the local ledger.
Usage: ingest.py <saved-result-file>   (writes cache/catalogue.json or cache/cards/*.md and updates cache/manifest.json)"""
import json, sys, base64, gzip, hashlib, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
raw = json.load(open(sys.argv[1]))[0]["text"]
m = re.search(r"GZB64:([A-Za-z0-9+/=]+):END", raw)
if not m: sys.exit("no payload (maybe rate-limited): " + raw[:300])
d = json.loads(gzip.decompress(base64.b64decode(m.group(1))))
man = json.load(open(f"{ROOT}/cache/manifest.json"))
if "catalogue" in d:
    json.dump(d, open(f"{ROOT}/cache/catalogue.json", "w"), indent=0)
    man["entries"]["catalogue"] = {"url": "/api/models?filter=base_model:quantized:Qwen/Qwen3.8-Flash-Next", "path": "cache/catalogue.json",
                                   "fetched_at": d["captured_at"], "n": len(d["catalogue"])}
    print("catalogue:", len(d["catalogue"]), "repos")
if "cards" in d:
    os.makedirs(f"{ROOT}/cache/cards", exist_ok=True)
    for c in d["cards"]:
        p = f"cache/cards/{c['id'].replace('/', '__')}@{c['sha'][:12]}.md"
        b = c["readme"].encode()
        open(f"{ROOT}/{p}", "wb").write(b)
        man["entries"][f"{c['id']}@{c['sha']}/README.md"] = {"url": f"https://huggingface.co/{c['id']}/raw/{c['sha']}/README.md", "path": p,
                                                             "status": c["status"], "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "fetched_at": d["captured_at"]}
    print("cards ingested:", len(d["cards"]), "| stopped:", d.get("stopped"))
json.dump(man, open(f"{ROOT}/cache/manifest.json", "w"), indent=1)
have = [k.split("@")[0] for k in man["entries"] if k.endswith("/README.md")]
print("ledger README total:", len(have))

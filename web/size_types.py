"""Which kinds of size a model's claims use, and whether an n-gram (Engram) table is involved.

Counts come from the size basis the scanner tagged on each claimed size (build.py BASIS_LABEL vocabulary); nothing is
estimated. A size is resident / in-memory only when the card says so ('resident', 'RAM in use'); a plain download or file
size is not, and says so. An n-gram table is flagged when a size is tagged excl_engram / engram_only, or the model's size
note describes one."""
import re

GROUPS = [
    ("Resident / in-memory", {"resident", "ram_in_use"}),
    ("GPU weights", {"vram"}),
    ("Weights only (text, trunk, decoder or experts)", {"text_weights", "trunk_only", "decoder_only", "routed_experts"}),
    ("Download / disk", {"disk", "hf_listing_total", "hf_branch_total", "excl_engram"}),
    ("Engram table only", {"engram_only"}),
    ("Not stated", {None, ""}),
]

def summarize(rows, size_note=""):
    repos_by = {name: set() for name, _ in GROUPS}
    ngram = bool(re.search(r"n-gram|engram", size_note or "", re.I))
    for r in rows:
        if not r.get("size_gb"):
            continue
        b = r.get("basis")
        if b in ("excl_engram", "engram_only"):
            ngram = True
        for name, members in GROUPS:
            if b in members:
                repos_by[name].add(r["repo"])
    groups = [{"label": n, "repos": len(repos_by[n])} for n, _ in GROUPS if repos_by[n]]
    return {"groups": groups, "ngram": ngram, "resident": len(repos_by["Resident / in-memory"])}

def line(summary):
    parts = " · ".join(f"{g['label']}: {g['repos']:,}" for g in summary["groups"])
    return "Size types in this model's claims (repos): " + (parts or "none stated") + "."

NGRAM_LINE = ("This model carries an n-gram (Engram) table that files may keep on disk instead of loading fully. Sizes marked "
              "download (excl. Engram), GPU weights, RAM in use or resident leave it out; a plain download size may include it.")

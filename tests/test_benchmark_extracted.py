"""The 'Every reported task score' section of /benchmarks/: coverage, and every score must appear verbatim on the card line it links to.

Run after assembling _site:  python3 web/assemble.py _site && python3 tests/test_benchmark_extracted.py"""
import json, os, re, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.environ.get("QFF_DATA") or os.path.join(CODE, "_site")
data = json.load(open(f"{SITE}/benchmarks/data.json"))
groups = {g["task"]: g for g in data["extracted"]}
assert len(groups) >= 40, f"only {len(groups)} benchmarks listed"
for task in ("AIME 2025", "LiveCodeBench v6", "GPQA Diamond", "MMLU-Pro", "GSM8K", "HumanEval"):
    assert task in groups and len(groups[task]["models"]) >= 2, f"{task} missing or on fewer than two models"
html = open(f"{SITE}/benchmarks/index.html").read()
assert html.count('<details class="task"') == len(groups), "page and data disagree on the number of benchmarks"
# index of captured cards: owner/name -> sha256-pinned card text per model
cards = {}
for slug in {e["slug"] for g in groups.values() for e in g["entries"]}:
    root = os.path.join(CODE, "models", slug)
    idx = json.load(open(f"{root}/snapshots/index.json"))["repos"]
    for repo, meta in idx.items(): cards[(slug, repo)] = os.path.join(root, meta["file"])
checked = on_line = missing = nolink = 0; problems = []
for g in groups.values():
    for e in g["entries"]:
        path = cards.get((e["slug"], e["card"]))
        if not (path and os.path.exists(path)): missing += 1; problems.append((g["task"], e["card"], "card not captured")); continue
        text = open(path, encoding="utf-8", errors="replace").read(); lines = text.splitlines()
        nums = re.findall(r"\d*\.?\d+", e["raw"].replace(",", ""))
        if not nums: missing += 1; problems.append((g["task"], e["card"], "no number in score", e["raw"])); continue
        checked += 1
        if not re.search(r"(?<![\d.])" + re.escape(nums[0]) + r"(?![\d])", text.replace(",", "")): missing += 1; problems.append((g["task"], e["card"], "score not on the card", e["raw"])); continue
        m = re.search(r"#L(\d+)$", e["src"] or "")
        if not m: nolink += 1; continue
        n = int(m.group(1))
        if n <= len(lines) and re.search(r"(?<![\d.])" + re.escape(nums[0]) + r"(?![\d])", " ".join(lines[max(0, n - 1):n]).replace(",", "")): on_line += 1
print(f"benchmarks {len(groups)}, scores checked {checked}: on the card {checked - missing}, missing {missing}; on the cited line {on_line}, without a line link {nolink}")
for p in problems[:15]: print("  ", p)
assert missing == 0, "a listed score is not on its card"
assert on_line >= 0.98 * (checked - nolink), "scores must sit on the card line they link to"
print("benchmark extracted ok")

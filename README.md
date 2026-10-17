# Quant Claims Ledger

**Live site:** https://delirious6423.github.io/quant-claims/

Hugging Face has dozens of quantizations of every popular model, and each publisher reports quality its own way: KLD against BF16, top-1 agreement against somebody's Q8_0, wikitext perplexity, a handful of benchmarks, or nothing at all. There is no single place that lines those claims up. This repo does that, starting with **Qwen/Qwen3.8-Flash-Next**.

It does **not** re-measure anything. Every number is a publisher's own claim, copied from the model card at a pinned commit, with the line it came from, the reference model, the eval data and any problems with the method.

## What's in here

| Path | What it is |
|---|---|
| `snapshots/` | Verbatim `README.md` of every covered repo at a pinned commit, plus `index.json` (repo → sha, file, sha256) and HF file listings |
| `verified/part*.py` | The claims, one `repo(...)` entry per repository. Every value is the exact published string with its README line number |
| `verify.py` | Mechanical check: every cited value must appear verbatim on its cited line of the snapshot. Runs in CI |
| `diff_blind.py` | Compares against an independent blind re-extraction (`audit/blind.json`) |
| `build.py` | Builds `docs/index.html` (the site) and `docs/data.json` |
| `audit/` | Verification counts, the blind-extraction diff and the adversarial review |

## Rules for a claim

1. It must be in the publisher's own card or docs, at a commit we snapshot.
2. Copy the string exactly as printed (`0.0771 ± 0.0007`, `86.95 ± 0.14 %`), never rounded.
3. Cite the line. Record the reference model, eval data, context and tool as stated, or `not stated`.
4. Numbers a publisher reports for someone else's file go under `cross`, never into that file's own row.
5. Flags describe what the card says, with a line reference. No guesses about motives.

## Build locally

```bash
python3 verify.py      # must print failures: 0
python3 build.py       # writes site/ ; copy to docs/ for Pages
```

Licence: code MIT; the compiled dataset CC BY 4.0. The claims themselves belong to their publishers.

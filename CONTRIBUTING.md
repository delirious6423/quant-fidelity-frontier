# Adding or correcting a claim

Thanks for helping. A pull request is accepted when `python3 verify.py` passes and a reviewer can follow every number back to a source line.

## Correct an existing claim
1. Find the repo's entry in `verified/part*.py`.
2. If the publisher changed their card, refresh the snapshot: save the raw card from `https://huggingface.co/<repo>/raw/<new sha>/README.md` into `snapshots/`, update `snapshots/index.json` (sha, file, sha256, lastModified), and update the entry's line numbers.
3. Run `python3 verify.py`.

## Add a new quant repo
1. Snapshot its card at a fixed commit as above.
2. Add a `repo(...)` entry following the existing ones: exact published strings, `L=` line numbers, methodology as stated, sizes with unit (`GB` or `GiB`) and basis (`disk`, `vram`, `resident`, `ram_in_use`, `trunk_only`, `hf_listing_total`).
3. Measurements of other publishers' files go in `cross=[...]`; the publisher's own comparison builds in `ladder=[...]`.
4. Flags must cite a line and describe the card, not the publisher's intent.

## Are you the publisher?
If a claim about your repo is wrong or out of date, open an issue with the commit of your card that shows it. We fix factual errors quickly.

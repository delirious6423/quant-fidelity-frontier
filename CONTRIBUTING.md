# Adding or correcting a claim

Thanks for helping. A pull request is accepted when `build.py` passes its gates for the model you touched (`QCL_ROOT=models/<slug> python3 build.py`) and a reviewer can follow every number back to a source line. Never edit `verified/part8_machine.py` or `part9_auto.py` by hand; they are generated. The fastest correction is an issue with the repo and the card line.

## Correct an existing claim
1. Find the repo's entry in `models/<slug>/verified/part*.py`.
2. If the publisher changed their card, refresh the snapshot: save the raw card from `https://huggingface.co/<repo>/raw/<new sha>/README.md` into `models/<slug>/snapshots/`, update its `snapshots/index.json` (sha, file, sha256, lastModified), and update the entry's line numbers.
3. Run `python3 build.py` (it runs every gate and refuses to build on any failure).

## Add a new quant repo
1. Snapshot its card at a fixed commit as above.
2. Add a `repo(...)` entry following the existing ones: exact published strings, `L=` line numbers, methodology as stated, sizes with unit (`GB` or `GiB`) and basis (`disk`, `vram`, `resident`, `ram_in_use`, `trunk_only`, `hf_listing_total`).
3. Measurements of other publishers' files go in `cross=[...]`; the publisher's own comparison builds in `ladder=[...]`.
4. Flags must cite a line and describe the card, not the publisher's intent.

## Are you the publisher?
If a claim about your repo is wrong or out of date, open an issue with the commit of your card that shows it. We fix factual errors quickly.

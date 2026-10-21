# Discovery rollout

## Model discovery job (queue maintenance)
`pipeline/qcl_discover.py` finds new base models, counts their quant repos, admits the ones that pass the rules in
`discovery.json` (current family of a listed lab, at least `admission.min_quants` quant repos, released within
`admission.max_age_months`), refreshes the counts of queued models, re-ranks the queue with `pipeline/qcl_rank.py` and
writes `watchlist.json`, `audit/discovery.json` (every input) and `docs/QUEUE.md`. The rules, counts, release estimate,
ranking and output files are one set of functions (`found`, `finish`); only the source of the search results differs.

- **Automatic (default): the Hub API.** `python3 pipeline/qcl_discover.py --http [--base URL] [--commit]` makes the
  searches itself against `/api/models` (`author`, `filter`, `sort`, `limit`, `Link` pagination) with the same request
  helper as `pipeline/qcl_fetch.py` and `pipeline/runner_fetch.py`: `HF_TOKEN`, if set, is sent only to the Hub origin and
  never printed; a 429 is reported with its `Retry-After` and the run exits with code 2 before any file is written; 5xx
  replies are retried with bounded waits. Quant counts come from the complete `base_model:quantized:<model>` listing
  (paginated), so `searched_sorts` is 1 and the count is exact. The script never calls git unless `--commit` is given,
  which commits the three output files (never pushes) with the same message as the connector path. A fresh
  `discover.lock` (a connector-driven run in progress) makes it skip. `.github/workflows/discover.yml` runs it every
  Sunday at 02:11 UTC and on `workflow_dispatch` with `HF_TOKEN` from the repository secrets, then pushes with
  `git pull --rebase` and three tries (concurrency group `discover`, `contents: write`).
- **Fallback: the `hf_fs` connector.** `python3 pipeline/qcl_discover.py` prints the padded `hf_fs` op lists; the session
  makes the calls and passes the saved result files back with `--results P1 P2 ...` in op-list order, round by round
  (lab listings and trending models, then one `downloads` search per candidate, then four more sorts for counts that hit
  the connector's 80-per-search cap). This path takes `discover.lock`, commits and pushes as before; `--abort` ends early.
  `docs/runbooks/HOURLY.md` has the scheduled-task wording.
- Replays and tests: `--root DIR` (where `watchlist.json`, `discovery.json`, `audit/` and `docs/QUEUE.md` live),
  `--now ISO-UTC` (the run's clock) and `--no-git` (connector path only). `tests/test_discover_http.py` runs both paths on
  a local fake Hub and requires byte-identical outputs.

## Ready in this change
- Search-focused page titles and visible interpretation guidance.
- Three static explainers, linked from the landing page and model pages, included in the sitemap.
- `llms.txt` for clients that choose to use it; no claim of ranking benefit.
- Reproducible Hugging Face dataset export: after building and assembling, run `python3 pipeline/export_hf_dataset.py`. Review `dist/hf-dataset/` before publishing. Original claim objects and companion model files are preserved; source cards are excluded.
- Validation: `python3 tests/test_discovery.py` after assembly.

## Repository metadata
The current connector has no repository-settings mutation operation. An authenticated owner can run:

```bash
gh repo edit delirious6423/quant-fidelity-frontier --description 'LLM quantization comparison: source-linked publisher claims, KLD vs size, compatible measurement sets and read-only MCP tools.' --homepage 'https://delirious6423.github.io/quant-fidelity-frontier/' --add-topic llm --add-topic quantization --add-topic gguf --add-topic exl3 --add-topic huggingface --add-topic mcp --add-topic benchmarks
```

## Search ownership and crawler rules
Verify the URL-prefix property `https://delirious6423.github.io/quant-fidelity-frontier/` in [Google Search Console](https://search.google.com/search-console) and [Bing Webmaster Tools](https://www.bing.com/webmasters/). Supply the issued HTML verification file or meta token before adding it to the build; do not invent a token. Submit `https://delirious6423.github.io/quant-fidelity-frontier/sitemap.xml` and inspect the landing page and live model pages.

The assembled `robots.txt` is under a project path and cannot control hostname-level crawling. Effective rules must be served at `https://delirious6423.github.io/robots.txt` via the account-level Pages repository, or at a custom domain's root. Do not change policies for other projects or opt out of training on the owner's behalf. A robots file is not required merely to allow crawling. Check the actual hostname response before claiming a block or successful configuration.

Reference: https://developers.google.com/crawling/docs/robots-txt/robots-txt-spec

## External publication still pending
- Hugging Face: select an owner/dataset namespace, review the generated card and export, then publish with account access. Dataset card reference: https://huggingface.co/docs/hub/datasets-cards
- Remote MCP: the existing Space scaffold still needs a target account, explicit deployment approval, an endpoint smoke test, and product-specific connection tests. Do not claim directory availability or connector acceptance before verification.
- Community outreach: prepare a specific comparison with its methodology and pinned sources. Recompute coverage counts from the manifest. Send only with explicit authorization.
- Share images and a custom domain are separate follow-ups; avoid an overview image implying cross-method comparability.

## Measure outcomes
Use existing Umami referral data and campaign-tagged links to distinguish community, search and identifiable AI referrals. Referrer suppression means these are incomplete. Search Console and Bing account reports provide indexing evidence. Neither crawl permission nor `llms.txt` guarantees a citation. Establish a baseline in those accounts before reporting growth; keep MCP calls separate from website visits.

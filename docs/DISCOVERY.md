# Discovery rollout

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

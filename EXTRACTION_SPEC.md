# Claim extraction spec — Qwen3.8-Flash-Next quant comparison

Goal: capture exactly what each Hugging Face quant publisher CLAIMS about quality/size, with enough
methodology to tell whether two claims are comparable. Do not invent or estimate numbers.

## How to read a repo
1. WebFetch `https://huggingface.co/<repo>/raw/main/README.md` (raw card). If 404, fetch `https://huggingface.co/<repo>`.
2. If the card links to a benchmark page / blog / gist that holds the numbers, fetch that too (max 2 extra fetches per repo).
3. Optionally `https://huggingface.co/api/models/<repo>?blobs=true` for file sizes when the card gives none.
   Ask WebFetch prompts to return tables VERBATIM, every row.

The shell cannot reach huggingface.co — use WebFetch only.

## Output
Write one JSON file per repo to `/home/claude/quantcmp/claims/<owner>__<name>.json`:

```json
{
  "repo": "owner/name",
  "url": "https://huggingface.co/owner/name",
  "format": "GGUF | NVFP4 | FP8 | MXFP4 | AWQ | GPTQ | AutoRound | EXL3 | MLX | other",
  "runtime": "llama.cpp mainline | llama.cpp fork (name/PR) | ik_llama.cpp | vLLM | SGLang | MLX | exllamav3 | ...",
  "weights_modified": false,          // true for abliterated/uncensored/REAP-pruned/finetuned — not a faithful quant
  "method": "short description of quant method (imatrix? dataset? dynamic per-layer? QAT? GPTQ calib?)",
  "imatrix_or_calib_data": "string or null",
  "methodology": {                     // for the measured claims below; null fields if not stated
    "reference": "BF16 | FP8 | Q8_0 | official API | not stated",
    "eval_dataset": "e.g. wikitext-2 test, custom chat corpus, not stated",
    "ctx": null, "n_tokens": null, "tool": "llama-perplexity --kl-divergence | lm-eval | own script | not stated"
  },
  "variants": [
    {
      "name": "Q4_K_M",
      "size_gb": 111.3,               // as stated; note GB vs GiB in size_unit
      "size_unit": "GB | GiB | not stated",
      "bpw": null,
      "metrics": {                     // ONLY numbers the publisher states. Keys from this vocabulary:
        "mean_kld": null, "kld_p99": null, "kld_p99_9": null, "kld_max": null,
        "top1_agree_pct": null, "ppl": null, "ppl_ratio": null, "delta_p_rms": null,
        "bench": {}                    // e.g. {"MMLU-Pro": 81.2, "GPQA": 70.1, "LiveCodeBench": ...}
      },
      "qualitative": "publisher's own description e.g. 'Low quality', 'recommended'"
    }
  ],
  "headline_claims": ["short paraphrases of marketing/headline claims, e.g. 'matches BF16 on MMLU-Pro within 0.3'"],
  "baseline_numbers": {},              // any numbers given for the unquantized / reference model (bench or ppl)
  "red_flags": ["e.g. no methodology, numbers copied from another repo, reference not BF16, placeholder repo with no weights"],
  "evidence_urls": ["every URL you used"],
  "fetched_at": "2026-10-17"
}
```

Rules: numbers exactly as published. If a card has no quantitative claims at all, still write the file with
`variants` (names + sizes) and note "no quality claims" in red_flags. Keep `headline_claims` paraphrased (no long quotes).
Finish with a 5-line summary of what you found per repo.

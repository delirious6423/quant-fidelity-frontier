# Source-backed task benchmarks

Every live model page now has a benchmark section after Fidelity and its collapsed controls. This first reviewed coverage contains 113 score records, 30 source-specific benchmark choices and 7 evaluation sources across five models. It is deliberately not an exhaustive catalogue of every `bench:*` field.

| Model | Evaluation sources | Benchmark choices | Score records |
| --- | ---: | ---: | ---: |
| DeepSeek V4 Flash 0731 | 2 | 6 | 11 |
| DeepSeek V4.1 Flash | 1 | 7 | 14 |
| GLM 5.3 Flash | 1 | 4 | 5 |
| Qwen 3.8 27B | 2 | 10 | 68 |
| Qwen 3.8 Flash Next | 1 | 3 | 15 |

## Evidence contract

Each `models/<slug>/benchmarks.json` is a reviewed manifest transcribed from captured raw model cards. A source declares its exact card SHA-256, setup, score units and notes. Each score and each file size carries its own exact line and full quote. Sizes are full model-file/download figures as the source defines them; optional components and scope caveats are stated in the notes. GiB converts to decimal GB. A scalar fraction stays a fraction; an unspecified unit never acquires a percent sign.

The renderer validates the current captured card bytes, exact quotes, scalar numbers, score ranges and supported size units before publishing a source. If a later scan changes that card, the whole evaluation source is withdrawn with an explanatory message until reviewed. Other sources and ordinary claim ingestion continue. Tests permit this expected withdrawal while rejecting malformed evidence against unchanged card bytes.

An evaluation selector keeps source/protocol panels separate. The benchmark selector preserves exact benchmark variants, sample subsets and scoring rules. There are no cross-source rankings, averaged scores or benchmark frontiers. A single result is still displayed, explicitly without a pairwise comparison. Missing-size API references are not assigned artificial x-coordinates.

Only audited task-score tables enter this initial manifest. Generic `bench:*` fields include speed, stderr, reconstruction diagnostics, disagreement statistics, counts, composite cells, pending runs and measurements on modified weights. Several rows also link a size/fidelity line instead of the exact task-score line. Accepting that entire namespace would misrepresent these fields.

## Reviewed sources

- Qwen 27B: eaddario GGUF six-task table (eight builds); tooltd EvalPlus (five exactly linked artifacts, four metrics). Fractional pass@1 scores remain fractional. Unlinked or unmatched artifacts and old versions are excluded.
- Qwen Flash Next: ISTA GSQ-RCO AIME25, GPQA-Diamond and LiveCodeBench v6, four quants plus the BF16 reference. Download size includes the n-gram shard; the optional vision projector is excluded as stated by the card for text tests.
- GLM: mbehr90's four task metrics and source FP8 reference where reported. CPU offload is recorded; no throughput comparison is made.
- DeepSeek V4.1: Intel lm-eval source/quant pairs, seven metrics. The chart uses the card's exact 510.51 GB quant size, not the legacy ledger's lossy 475.0 GiB conversion (510.03 GB). The source's inconsistent native GB/GiB pair is disclosed; its explicit 510.29 GB value is retained.
- DeepSeek V4 0731: Intel's five metrics for native and AutoRound artifacts; separate MLX MMLU-Pro 600-question sample. The API reference has no file size and is excluded from the plot.

These chart manifests add reviewed views without editing existing verified ledgers or initial fidelity measurement sets.

## Adding or refreshing a source

1. Read the complete captured card, including model identity, modified-weight flags, table headers, evaluation setup, uncertainty and artifact links.
2. Match each score to the exact tested artifact and its full-size evidence. Do not borrow a current size for a previous checkpoint revision, attach an unrelated adapter size, or reuse component-only/Engram-excluded sizes as full models.
3. Keep unlike protocols in separate sources/panels. Preserve score scale and exact benchmark name. Add exact quotes, line numbers and card SHA-256.
4. Have another reader check metric columns, artifact matching, units and scope against the raw card.
5. Run `python tests/test_benchmarks.py`, rebuild every live model and run `python tests/test_benchmarks_ui.py`. Review mobile and desktop screenshots. The UI suite exercises every reviewed source/metric at 320, 390 and 1366px, plus no-JavaScript fallback.

The assembled site remains HTML/CSS/JS/JSON only and needs no API or server at runtime. `data.json` includes the source-separated benchmark payload, hashes and withheld-source reasons.

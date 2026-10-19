# Shared benchmark view

`/benchmarks/` is a standalone scatter chart of published task scores against full model-file size in decimal GB. It is linked immediately after Models in the homepage navigation, from the homepage's compact comparison button, and above the provider groups in every model page selector.

The assembler reads the reviewed `benchmarks.sources` already included in each live model's build output. A view requires at least two distinct model slugs with the exact same benchmark variant and declared unit category. Different scoring variants and unit categories are separate views. Unstated units remain unstated; scores are displayed as published. Every source report retains its own point, even when a build appears more than once. Model colors do not imply verified protocol compatibility, ranking, or independent reproduction by QFF.

Current reviewed coverage:

| View | Models | Reports |
| --- | --- | ---: |
| GPQA-Diamond | Qwen3.8-27B, Qwen3.8-Flash-Next | 13 |
| GSM8K strict (5-shot) | DeepSeek-V4-Flash-0731, DeepSeek-V4.1-Flash | 4 |

The default is GPQA-Diamond where available, with all eligible models, source reports and reference builds shown. Reference builds use diamonds; quantized builds use circles. Model filters, file-size limit, log size axis, zero score baseline and reference visibility are shareable through the page URL. Score and file-size citations, source setup, flags, capture dates and card hashes remain available in the detail panel, collapsed evaluation notes and `benchmarks/data.json`. Without JavaScript, the default chart and all source-linked rows remain available.

Changed source cards continue to be withdrawn by the existing reviewed-manifest validator. Shared views are derived afresh on each assembly: if fewer than two models remain eligible, the corresponding view disappears. The chart catalog is generated after `models.json` so its data file is never mistaken for an additional model by the manifest or MCP server. Existing model measurement sets and claim rows are unaffected.

Validation: `python tests/test_benchmark_view.py` checks exact task/unit matching, all source values and citations, missing sizes, repeated reports, controls, sharing, keyboard access, no-JavaScript fallback, navigation, and 320/390/768/1366px layouts. The regular build, homepage, model-page, claim, discovery and MCP checks also apply.

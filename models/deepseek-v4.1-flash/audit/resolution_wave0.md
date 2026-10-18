# Resolver wave 0: Krishando EXL3 2.86

Resolved only `Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw`, the sole repository assigned in `work/deepseek-v4.1-flash/resolve_0.json`. Read the entire pinned card, both A/B maps, and all candidates. Source bytes match the index SHA-256. The complete resolved map is in `maps/resolved/`. No A/B map, card, or verified data was edited.

## Reference and measurement scope

The reference is the native DeepSeek release at the card's stated source commit, computed layer by layer in BF16 and checked against DeepSeek's inference code. BF16 describes arithmetic after loading the native source weights; it does not establish an original BF16 checkpoint. The stored reference contains top-20 probabilities. Every distributional entry explicitly retains that restricted reference and must not be merged into a full-vocabulary KL series.

KLD measurements were obtained through vLLM with an EXL3 overlay, teacher-forced at 2,048-token windows. The separate TensorFold engine supplies speed/task qualification; the card expressly says KLD was not remeasured there. The private agent-session and public general-text holdouts remain distinct, with no combined mean computed. Their scored positions differ from raw holdout token totals; the scored counts are retained in entry methods.

## Resolution decisions

- Used A's richer current-panel and release-check coverage, with consistent corpus keys aligned to B (`agent_sessions`, `general_text`, and named decision subsets). Native full-model controls are `cross` entries for the official source with explicit reference names/flags; they are not this publisher's ladder quants. Actual published zero-KL controls are retained as controls, without synthesizing zero values.
- Current agent and general entries retain their separate KL, top-1, PPL and PPL-change claims. Median and percentile KL are named distributional statistics, not downstream benchmark scores. Top-20 overlap is labeled separately and is not promoted to top-1 agreement. No additional full-vocabulary mean is inferred.
- Decision-point own and Mia-AiLab results remain measured comparisons on the same teacher-forced positions, not quotations or whole-agent task success. The Mia repository is explicitly linked in the source provenance. Individual decision subsets, their aggregate, and the every-position row remain separate metric keys. No result is inferred for the linked 0xSero build because this card does not publish its numeric result.
- Full MMLU, the 100-question GSM8K test, thinking-off/on release subsets, and the single-prompt long-generation probe remain separate entries. The release subset is not presented as full MMLU. The long-generation working-game and completion-within-budget values remain counts with denominators in their labels; timings, output-token counts and stability/kernel pass counts are not promoted to model-fidelity metrics.
- Historical v3, v4, v4.1 and current v5 retain the exact development-panel heading and same stated harness, with separate in-domain/general KL. Historical entries are ladder builds; v5 is own. Expert rates are the source's component rates. No missing historical download sizes are estimated from bpw.
- Fixed B's proposed prose `size` metric: the explicit model-details size is now an actual size field. Its scope is repository download excluding the separately supplied official Engram shards. Every current own configuration refers to this same checkpoint size, while native and historical sizes remain unstated. No Engram-inclusive total is invented. The expert bitrate is distinct from the download-size scope.

## Supplemental candidate provenance

All additions below were extracted programmatically from exact pinned source substrings. The MMLU candidate uses its real first-cell identity and result-column header; the original candidate concatenated test identity and setup across columns, which did not match the gate's pipe-separated source-row identity. No result bytes changed.

```json
[
  {
    "candidate": "c9001",
    "L": 20,
    "raw": "2.86 bpw",
    "reason": "Expert-average bitrate is explicitly stated in introductory prose.",
    "sha256": "9a7f1a9b900fef06db7d18c3571029ef8c2c642d3adfaeb93d64b1d36477d2aa"
  },
  {
    "candidate": "c9002",
    "L": 232,
    "raw": "214 GB",
    "reason": "Explicit repository download size; Engram separately supplied as stated immediately above.",
    "sha256": "9a7f1a9b900fef06db7d18c3571029ef8c2c642d3adfaeb93d64b1d36477d2aa"
  },
  {
    "candidate": "c9003",
    "L": 119,
    "raw": "88.0% (12,359)",
    "header": "Result",
    "label": "MMLU",
    "reason": "Exact first-cell identity avoids concatenating test identity and setup across a pipe boundary; same original result bytes.",
    "sha256": "9a7f1a9b900fef06db7d18c3571029ef8c2c642d3adfaeb93d64b1d36477d2aa"
  }
]
```

## Gate verification

Applied the resolved map into an isolated scratch data root and ran the actual presence/unit/sign and table-family/row-label gates. Initial results: 87 values verified against source, nine size units checked, two signed values checked, and zero numerical/sign/unit/table-family failures. Table bindings checked: 13; row-label bindings checked: 44. The MMLU source-identity supplement resolves its row-binding issue.

Two remaining initial row-label failures are the call-or-answer row containing the literal inline code `</think>`: the old `qcl_map` label cleanup removed that code as HTML while the gate correctly retained the source content. Root is applying a narrow label-cleaning fix that preserves inline code; no data or gate exception is needed. Final gate outcome will be appended after that fix lands.

Final rerun after the source-preserving inline-code label cleanup: both gates pass. All 87 values, nine size units, two signs, 13 table-family bindings and 44 row-label bindings pass with zero unexplained failures. No numeric/source changes or gate exceptions were used.

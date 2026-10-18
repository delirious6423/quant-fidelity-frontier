# Independent cold review: additional six mappings

Reviewed by the core resolver, who did not author these six mappings. Read `resolution_other.md`, all six complete pinned cards, and each corresponding resolved map. No mapping, candidate, verified-data, or source-card edits were made.

**Result: no blocking defect found.** The reviewed set contains 23 entries and 319 metric fields after applying the maps.

## Checks performed

- Independently recomputed each pinned card's SHA-256 and matched it to `snapshots/index.json`.
- Resolved every metric, size, and bpw candidate reference. Checked metric raw strings against their actual cited source lines after removing Markdown emphasis/code wrappers.
- Checked all 151 prose mappings for exact source-line containment, the intended target entry, and panel identity.
- For both dealignai cards, independently checked the intended base/CRACK table column of every aggregate, HarmBench category, MMLU subject, and subset result, including the prose-restored duplicate cells. No value was found in the wrong intended column.
- Read all benchmark descriptions, precision statements, file-size captions, and provenance statements to assess claims that line-presence gates cannot establish.

## Findings by repository

| Repository | Result |
|---|---|
| dealignai UNCENSORED FP8 | Weight abliteration is correctly distinguished from quantization loss. The base cross entries identify the native release, and reference flags explain that these are pre-abliteration baselines. Native FP8 with FP4 experts is preserved; there is no BF16 claim. ASR is labeled compliance, effort methods remain separate, and all MMLU subject columns were checked. |
| dealignai UNCENSORED EXL3 2.9 | Baseline exact HF repository is correctly left unspecified. Base HarmBench sample sizes differ from the modified full samples and are explicitly described. Own footprint is correctly scoped to the pack excluding separately supplied Engram. MMLU/HarmBench panels and effort settings remain separate. Abliteration is classified as modified weights. |
| NVIDIA NVFP4 | The native MXFP4 source is a measured cross reference, not a second own quant. Current Engram-inclusive approximate disk size is taken from the current conversion, not the intermediate checkpoint or per-rank host-memory figure. All twelve HTML accuracy values match the correct source/own row and benchmark column. Terminal-Bench uses its separate agentic/eight-trajectory methodology. No synthetic zero KL follows from the expert lossless-conversion statement. |
| pfeifferj GSQ-RCO | Reported file size includes Q8_0 Engram but excludes separate draft/projector files; flags state that boundary. Bpw is backbone-only. MMLU-Pro uses the stated subset, zero-shot/no-reasoning protocol and context. Absolute PPL is not supplied with an invented reference/corpus. IFEval remains a checker-pass count including the truncated response, not a full-suite completion percentage. GSM8K remains the small correct-answer probe. |
| YoungAi | Next-token agreement is correctly treated as code agreement, not downstream task accuracy. Base and sidecar results remain separate. The ambiguous original-full-precision reference wording is explicitly attributed to the publisher without asserting BF16. Unknown size boundary is retained, and the base size is not conflated with full download. Repository-wide modified-weight classification is conservative because of the sidecar configuration; flags preserve that the base vector quantization itself used no training data. Moonshine speech metrics were excluded as another model. |
| OpensourceWTF streaming repack | HumanEval raw fraction/count, greedy one-sample protocol, offline final fenced-code scoring and full-package disk size match the source. Expert-only bitrate is not called package bitrate. The qualitative BF16 sentence is not promoted to a numeric BF16 comparison. The card explicitly says repacking adds no fine-tuning; modified-weight classification is consistent with that definition. |

## Nonblocking scope limits preserved

The claims remain publisher claims from pinned cards, not independently reproduced benchmark measurements. YoungAi does not establish its base size boundary or exact reference checkpoint precision. Pfeifferj does not name the PPL corpus or its own PPL context. HarmBench measures compliance, and the two abliterated models belong outside isolated quantization-fidelity comparisons. Those limits are already represented in the resolved methods/flags; none requires changing a published value.

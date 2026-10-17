# Resolution wave 2

Reviewed the six assigned repositories against both independent maps, complete pinned cards, and numeric candidates. Resolved maps replace whole-repository interpretations; A/B originals were not edited. Reported values use existing candidate IDs, programmatically captured exact source substrings, or source-checked supplemental candidates.

| Repository | Source-backed resolution |
|---|---|
| pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit-engram6 | Preserve all per-layer divergence and cosine recipe rows with scoped metric names; the card never identifies those divergence columns as next-token KL. The dedicated Engram-6 row is own; other recipes are publisher ladder entries. Preserve the contradictory Engram precision summary as a flag. The only PPL belongs to the separately measured Engram-4 sibling, with its complete published interval; no own PPL is inferred. Sources: L15–26, L52–81. |
| coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw | Original source is measured cross, previous upload is ladder, current weights are own. Separate text agreement from correct-next-token accuracy and NLL; retain published KL without asserting vocabulary scope. Image KL/agreement uses the full image-target slice; image PPL uses reference-answer targets. Generated-answer task checks remain previous-upload only. Retain the current image-suite range and older-runtime comparison as published. Use total shard storage for the current measured artifact and retain active/Engram/draft payloads as separately labeled subsets. Sources: L37–77, L107–115. |
| coolbho3k/DeepSeek-V4.1-Flash-GRAFT-3bpw | Resolve naming/panel disagreement using the source title. Keep expert-error ratio, excess NLL in nats, reported KL, and top-choice agreement distinct. Reference is the native MXFP4 checkpoint; corpus, context, runtime and KL vocabulary/aggregation scope remain unstated. EXL3 is the publisher comparison build. No inferred size. Sources: L14–37. |
| pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit | Restore all source recipe-ladder divergence/cosine claims with scoped keys rather than treating unspecified layer divergence as KL or dropping it. No recipe row is unambiguously identified as this requantized Engram build. Its independently stated full-model PPL and complete interval remain an own claim with the stated disk size and separate methodology. Sources: L21–26, L52–81. |
| dongnhdev/DeepSeek-V4.1-Flash-Q2-EngramQ4-GGUF | NLL changes stay NLL changes, not PPL/KL. Reference is already-Q2 antirez GGUF with FP8 Engram; official API continuations are the scoring targets. Separate general and long-context methods. Retain general-set first-token match counts and greedy LCP for both measured files; long-set Engram controls are ladder entries. Preserve the author's small-sample/noise caveat. Joined GGUF and source-file byte sizes stay total disk sizes. Sources: L13–20, L26–35, L66–79. |
| amd/DeepSeek-V4.1-Flash-Quark-PTPC-FP8 | Use the exact Accuracy heading and explicit eight-shot flexible-extract metric. Both model columns publish the same source value, so the deduplicated candidate is valid for own and measured cross. Retain separately reported score recovery without calling it perfect task accuracy. Reference is native mixed MXFP4/MXFP8. Sources: L31–35, L75–84, L99–115. |

All assigned maps retain `weights_modified: false`: these are quantization/format variants rather than finetuned, abliterated or expert-pruned main-model weights. Omitted draft layers and source precision conflicts are recorded in flags rather than silently resolved.

## Validation

- Candidate references and exact-source prose substrings checked for every assigned resolver map.
- Applied the six maps into a throwaway data root containing only assigned source-index entries and symlinked pinned source cards; no original verified files were written.
- Actual `gates/verify.py`: passed, six of six repositories covered, 92 values verified, nine size/unit checks, four signed-value checks, zero failures.
- Actual `gates/verify_columns.py`: passed, zero unexplained column or row-label mismatches.
- Applied result contains 36 entries and 83 metric claims, plus the separately verified size claims.
- Supplemental candidates were appended only to the two owned PipeNetwork candidate files (exact PPL plus published interval) and coolbho3k EXL3 candidate file (exact total shard-storage byte count). Every supplemental raw string is checked against its pinned source line before saving.

## Resolver pin inputs

The parent agent writes the manifest. These are the source and exact resolver-map byte hashes after validation:

| Repository | Source SHA-256 | Mapping SHA-256 |
|---|---|---|
| pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit-engram6 | `7e0dd5e12d27a0e9fcb8a06d6487065dfadad8b6ff69f72995db94fb3d83d6c3` | `9b29e00f7127346c84991cfc7557d01f618726648dbcc3c8c624f6aff5451257` |
| coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw | `8e0cbb100deba4b9e638a13b2fa076b1e6037c2bb4bf556033d42ccd4bc78308` | `b0449782c1747e944ba2fad015fbce2388f229daa452c7c6ff8db4d8d8de6d72` |
| coolbho3k/DeepSeek-V4.1-Flash-GRAFT-3bpw | `365370018286aa8c31ee06f623269a659dd7af405f7c3daaf858bb01bad1481c` | `7e2a97a2302e2146b1978063e6963183d5ff6917f4bd412db82252bc8bdab52f` |
| pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit | `b44fe62c448a724138325f9f2b03b545f80212383a357ebade9e9aaf59a5cd21` | `3c01e56d55979fb04593e242c79a3f2f82308b76ec4cc94bb290819bdecd0cac` |
| dongnhdev/DeepSeek-V4.1-Flash-Q2-EngramQ4-GGUF | `29d247feb082c05f7d10875d177519e94f7a76860ba55f3b033fbc37a5ee1f09` | `08ad5d5ea8deba517b2163e39550ddb775410a3a70067ff6cecb153b8bada109` |
| amd/DeepSeek-V4.1-Flash-Quark-PTPC-FP8 | `d3725411672376490ad5b15cf56a54f8c3be8f4a136c716d178047c05a8e5942` | `4eb0b27712fd00e9b1e833b7d4a95239185c5fe37c39ec43bf7f50c55715d1ae` |

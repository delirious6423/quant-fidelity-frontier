---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model:
- LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- fr
tags:
- exl3
- mlx
- lfm2.5
- moe
- quantized
- apple-silicon
---

# LFM2.5-24B-A2B — EXL3 4 bpw, v2

An evaluated EXL3 conversion of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), source revision `1769516ac588c4c558089c94f27d96b0dc67b183`. **K4 experts, K5 dense MLPs and K6 attention/convolution**, MCG/W16. The four weight shards total **12,614,197,321 bytes (12.61 GB), or 4.209716 effective bits per source parameter**, including plain tensors and shard overhead. This is 1.37% larger than v1, and 15.21% smaller than the user's measured Unsloth UD-Q4_K_XL file.

V2 improves the measured average perplexity and KL against v1 on all five protocols below. Results vary by window, and comparison with the user's GGUF is mixed. The [frozen v1](https://huggingface.co/0xzknw/LFM2.5-24B-A2B-EXL3-4bpw/tree/v1) remains available.

## Runtime and download

Tested with **MLXL3 engine 1.4.4** on Apple M5/24 GiB, source `8386fbfc557f1e740e46a5a6ab250102b047a665`, executable SHA256 `355b52620cfcf3e8a6d3b053efc9c28eda34963d8f132406e86aa733acdc2cd8`. The native router supports all 64 experts. Use the supplied tokenizer and chat template; this is a reasoning model whose thinking tokens consume the output budget. Ordinary Transformers loading is not validated.

Download into a **new directory**, so old v1 shard names cannot remain alongside the four v2 shards:

```bash
hf download 0xzknw/LFM2.5-24B-A2B-EXL3-4bpw --revision v2 --local-dir LFM2.5-24B-A2B-EXL3-4bpw-v2
```

[MLXL3](https://github.com/0xZKnw/mlxl3) provides the supported native runtime. Full CUDA inference was additionally checked on two 1,024-token windows with the original export, using the same quantized weight tensors; removing its redundant tied head preserves native logits bit for bit. This is not a general cross-backend parity claim.

## Conversion recipe and timing

Official ExLlamaV3 **1.5.4**, source `16a49792a3c93d8432d72e6c4bce800841566577`, CUDA12.8/Torch2.10/Python3.12 wheel SHA256 `32b253a84612fae5bfd3787a3fc8a81450b976c59b3d6d29c2994fe5b451c2bc`. Five critical converter source files were SHA-checked before conversion.

```text
-b 4 -hb 16 -cb mcg --out_scales auto --hq
-cr 250 -cc 2048 -ss 4096 -cpi 120
```

512,000 calibration tokens from the bundled mixed corpus, quantized states propagated layer by layer, LDLQ, GSS scale search, output sensitivity scaling and the upstream post-LDLQ scale refit. Embeddings/head remain unquantized at 16 bits. This checkpoint uses the official CUDA converter, not the experimental routed Metal capture in MLXL3 PR37.

A single H10080GB with 8 CPU/96 GiB host RAM completed the function in **53 min29.23 s**, including source download, preparation, calibration, conversion, export and hashes. V1 took3h58min39 on M5 excluding source download; hardware and recipes differ, so these are not a kernel A/B speedup.

The redundant tied head was removed losslessly after conversion: all262,144,000 head values equal the embedding after the runtime's F16 cast,782 other tensors in the rewritten shard have identical bytes, and16,384,000 real native F16 logits match bit for bit before/after. All7,402 expected projections are present; every stored floating value passed finiteness checks. The manifest records final shard hashes.

## Quality: same engine, tokens and BF16 teachers

Lower PPL/KL is better. V1 was rerun on the exact engine used for v2 and reproduced its previous scores. KL direction is **BF16 reference → quantized model**.

| Evaluation protocol | Scored labels | PPL v1 | PPL v2 | KL v1 | KL v2 |
|---|---:|---:|---:|---:|---:|
| WikiText test,8×1024,no BOS,all positions |8192|29.47950|**26.86163**|0.323047|**0.187928**|
| French technical text,2×512,no BOS |1024|64.68621|**61.00814**|0.226630|**0.139829**|
| Python source,2×512,no BOS |1024|11.33344|**10.96518**|0.215710|**0.103885**|
| WikiText test,8×256,no BOS |2048|54.45852|**49.93228**|0.427937|**0.200784**|
| WikiText test,BOS/8×1024,positions512–1022 |4088|18.36556|**16.93209**|0.157411|**0.083458**|

The primary protocol improves PPL by **8.88%** and KL by **41.83%** against v1. Its BF16 teacher PPL is28.15363; the paired BOS teacher PPL is17.55602. PPL below BF16 on a small sample does not establish superior general quality. The three WikiText protocols overlap, and the small French/Python samples are not general capability benchmarks.

One primary window regresses in KL: v1≈0.2734 to v2≈0.5265. The other seven average0.13956; **0.187928 remains the official all-window mean**. The same difficult window gives KL0.5198 in ExLlamaV3 CUDA. A nonquantized source control in that backend gives BF16-teacher→F16-source KL0.05988 and F16-source→v2 KL0.45281, pointing to quantization/recipe sensitivity on that window. KL is not additive and these values must not be subtracted as a quantization-only loss.

Six real native bridge checks passed: French explanation,17×23=391, exact JSON, Python addition function, and two-turn project-code recall. All generated code was inspected rather than executed. No matched inference speed benchmark or broad task-accuracy suite is claimed.

## Comparison with the user's Unsloth GGUF

The actual file is **UD-Q4_K_XL**, not Q4_K_M,14.8769 GB/4.96484 effective bpw;33.59% of parameters are Q5_K. It was evaluated through official llama.cpp b11620/21a19cd3b. Full **unclamped F32 logits**, identical imposed tokens, reset contexts and the same BF16 teachers were used for KL.

| Matched protocol | GGUF PPL | V2 PPL | GGUF KL | V2 KL |
|---|---:|---:|---:|---:|
| No BOS,8192 labels |28.199009|**26.861628**|**0.134553**|0.187928|
| BOS,4088 labels |18.346692|**16.932095**|0.085507|**0.083458**|

V2 improves PPL in both cases but loses KL in the no-BOS case. The BOS KL advantage is only2.40%; no universal EXL3-over-GGUF claim follows. Runtime arithmetic differs. The GGUF's complete original weight provenance has not been proved, although tokenizer/config and all38 router matrices match the source checks.

`v2-quality.json` records the protocols and scores. MLXL3's detailed reports retain per-window results, failed attempts, source identities and verification limits. Software validation includes570 passing local Python tests/one missing-model skip and five limited CrossHair conditions; this is not a proof of GPU arithmetic or model quality.

## License and attribution

The weights retain the **LFM Open License v1.0** in [LICENSE](LICENSE). These are modified derivatives: source BF16 projections are converted to the EXL3 recipe above, and redundant tied-head storage is omitted. Tokenizer and chat template are retained from the source. `v2-manifest.json` identifies every distributed file except itself and preserves the previous v1 revision.

Thanks to [Liquid AI](https://huggingface.co/LiquidAI) for LFM, [turboderp/ExLlamaV3](https://github.com/turboderp-org/exllamav3) for EXL3 and the CUDA converter, and [PonyExl3](https://github.com/beamivalice/PonyExl3) for the MLX foundation. [MLXL3](https://github.com/0xZKnw/mlxl3) provides conversion orchestration, native inference and the measured validation; its custom Metal quantization kernels produced v1, while this v2 was quantized by upstream CUDA.

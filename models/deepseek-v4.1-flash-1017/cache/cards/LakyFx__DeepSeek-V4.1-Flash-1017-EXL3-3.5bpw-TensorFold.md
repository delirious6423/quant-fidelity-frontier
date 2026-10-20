---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: tensorfold
tags:
- deepseek
- exl3
- tensorfold
- quantization
- dgx-spark
- tp3
- 3.5bpw
---

# DeepSeek-V4.1-Flash-1017 · EXL3 3.5 bpw · TensorFold

![DeepSeek V4.1 Flash 1017: EXL3 3.51 bpw main experts, TensorFold TP3 on three DGX Sparks. Weights live and recipe v1.2.0 ready.](assets/deepseek-1017-release.png)

> **READY - complete TP3 release, GPU serving verified on 3 x DGX Spark (18 October 2026).** Published artifacts match the active 1017 deployment through the checks below. [Download layout and required cache filename](RELEASE-LAYOUT.md) | [Serving qualification and limits](verification/SERVING-QUALIFICATION.md) | [SHA-256 manifest](RELEASE-MANIFEST.json).

This is an independent quantization of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), tested with the existing TensorFold native CUDA TF351 configuration on **3 × DGX Spark (TP3)**.

**About the Hub's model-size badge:** its automatic `0.5B params / BF16` figure counts only the auxiliary `vision.safetensors` file. The EXL3 main layers and TP3 rank caches use a different layout, so that badge does **not** describe this model's size or overall precision. The [original checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) is a 763B-parameter model.

The **3.51 bpw** designation describes the main routed experts. It is **not** an average precision for the entire model. The release also includes a K6 head, K4 DSpark draft projections, and the new revision's FP8 Engram tables. Other projections use their individually specified precisions. All output values must come from the pinned 1017 revision; older model weights are not reused.

The original checkpoint is [`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170). The original model and weights are MIT licensed. Please credit DeepSeek for the base model and TensorFold for the runtime.

## Provenance, acknowledgements, and historical quality reference

The earlier TensorFold TF351 setup combined main routed-expert weights from [bot-lab-21's DeepSeek-V4.1-Flash EXL3 3.5bpw Pollard checkpoint](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) with other components, including the DSpark draft, from [Mia-AiLab's DeepSeek-V4.1-Flash EXL3 release](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw). The expert precision map and quantization approach for this 1017 release draw on that prior work. All weights here are generated from the pinned DeepSeek 1017 source; no older model weights are reused.

Thank you to **bot-lab-21** for openly publishing the 3.51 bpw expert weights, layer allocation, calibration tools, and measurements; **WestWaters** for the [Pollard Weights](https://github.com/WestWaters/pollard-weights) method; **Mia-AiLab** for the earlier EXL3 export; **turboderp** for [ExLlamaV3 and EXL3](https://github.com/turboderp-org/exllamav3); and **tonyd2wild and Kai** for the [DGX Spark serving groundwork](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark). DeepSeek created the base model, and TensorFold supplies this release's TP3 runtime.

**Special thanks to the [TensorFold project](https://tensorfold.dev/) and its creator, [Ash Hart (@ashxhart)](https://x.com/ashxhart).** Ash, your persistence in making demanding models run on real hardware inspired us to build these weights for our three DGX Sparks. Thank you for continuing to push TensorFold forward and for sharing the work openly. We hope these weights become a useful contribution to the community around your project.

**Historical measurements, attributed to bot-lab-21's older V4.1-Flash checkpoint:** on 39 in-domain calibration sequences of 2048 tokens, its published reference NLL was 1.3209 and its 3.51 bpw expert build was 1.3177 (paired difference -0.0032 +/- 0.0055). Those sequences were also used for expert calibration, so this is not an independent held-out comparison. The same model card reports perplexity 4.043 on six held-out texts (4,210 tokens) and HumanEval / HumanEval+ greedy pass@1 of 95.1% / 92.1%, without matching unquantized baselines for those tests. These results document the predecessor; **they are not quality measurements of the new 1017 weights**. Results for this release will be published separately after its own verification.

## Release status and verification

All 40 main `exl3_L*.safetensors` layers, model configuration, tokenizer, vision wrapper, three TP3 rank caches and three rank-specific sparse Engram archives are complete on the Hub and checked against component SHA-256 receipts. The [release manifest](RELEASE-MANIFEST.json) lists file sizes and digests. The [artifact layout](RELEASE-LAYOUT.md) explains the files for each Spark, sparse extraction, and the **required regular `rankNof3-*.bin` hardlink** for the tested native reader.

**The running three-Spark deployment passed** basic MAX-reasoning arithmetic, automatic tool calls and tool-result roundtrip, vision, fresh-revision prefix-cache reuse, DSpark drafting, and 12 simultaneous short requests (12/12 correct). All three ranks were active with zero restarts at the publication check. [Machine-readable qualification](verification/PRODUCTION-QUALIFICATION.json) | [Check scope and limitations](verification/SERVING-QUALIFICATION.md).

Production-to-Hub verification reused the completed full rank-cache disk-readback hashes and Engram owned-range checks, with fresh process/path binding, file-state checks, Engram SOURCE/header checks and SHA-256 checks of the model wrapper. It did not rescan all large weight payloads while production was serving. The runtime **`chat_template.jinja`** sidecar is included at the root and in `models/TF351-1017/`; its 56 comparisons against the frozen 1017 encoder passed. The original tokenizer metadata remains unchanged.

Engram data is published only as **TP3 rank-specific GNU sparse archives**. Each archive is about 67.6 GB, or about 203 GB combined. Preserve sparse holes when extracting or copying; logical extracted extents are much larger. The [Engram row-ownership audit](verification/ENGRAM-ROW-AUDIT.json) checks the three disjoint rank ranges against the pinned source headers. Original full Engram shards belong to the [DeepSeek source checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), not this TP3 runtime layout.

**Tested settings:** TP3, context setting 401,408, 8M shared KV pool, 2 GiB prompt-state cache, S12/R32, MAX by default, DSpark enabled. These short operational tests are not a new throughput, capacity or comparative quality benchmark. Original CPU-stage receipts keep their historical pending-GPU fields; the newer serving qualification supersedes that stage's release status.

The **[DS41F 1.2.0 / DeepSeek 1017 recipe](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/v1.2.0/docs/DS41F-1.2.0.md)** is now published, with the `--variant 1.2.0` launcher profile and these prebuilt HF weights. [GitHub release v1.2.0](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/releases/tag/v1.2.0). It retains earlier variants and historical tests; no comparative benchmark was rerun for 1017. Follow the recipe and [artifact layout](RELEASE-LAYOUT.md); older recipe benchmarks and 10M pool qualification do not describe these new weights.

Project: [LakyFx DeepSeek Flash EXL3 TP3 recipe](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark).

---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
- nvidia/DeepSeek-V4.1-Flash-1017-NVFP4
library_name: vllm
pipeline_tag: text-generation
tags:
- deepseek
- deepseek_v41
- ds41-flash
- nvfp4
- modelopt
- abliterated
- uncensored
---

# WARMIND-1017-NVFP4

**Experimental.** WARMIND-1017-NVFP4 is NVIDIA's [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) with the refusal-reduction edits from [WARMIND-1017](https://huggingface.co/jcartu/WARMIND-1017) applied. The weights are NVIDIA's NVFP4 release, unchanged except for the 132 edited tensors.

## At a glance

| | |
|---|---|
| Base | `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` @ `480550d349ede47a048eb65a276c5c01a06256e2` (itself a quantization of `deepseek-ai/DeepSeek-V4.1-Flash-1017`) |
| Edit | The same 132 dense tensors as WARMIND-1017: `attn.wo_b` and `ffn.shared_experts.w2` weights and block scales, layers 4–36 |
| Everything else | Byte-identical to NVIDIA's files: NVFP4 routed experts and their calibration, both MXFP4 Engram tables, MTP/DSpark draft layers, config, tokenizer |
| Size | 400 GB, 48 shards |
| Runtime | vLLM 0.32 **plus two upstream patches** for SM120. Recipe and exact launch in [`runtime/`](runtime/README.md) |
| Tested on | 4x RTX PRO 6000 Blackwell (96 GB each), TP4, text only |
| License | MIT, unchanged from DeepSeek / NVIDIA ([LICENSE](LICENSE)) |

## How it was made

The 132 edited tensors in WARMIND-1017 are stored in the same FP8 E4M3 / UE8M0 format as in NVIDIA's NVFP4 release; NVIDIA's quantization left them untouched. Their unedited bytes in NVIDIA's checkpoint match the original DeepSeek bytes exactly. So the edit transfers as a byte-for-byte copy, with no requantization:

1. Verify every NVIDIA source file against the pinned Hub revision (LFS sha256 for all shards).
2. Check that each of the 132 target ranges in NVIDIA's shards equals the recorded pristine bytes from the WARMIND-1017 derivation.
3. Copy the 132 edited payloads into those ranges.
4. Compare every byte outside them against NVIDIA's source.

[`DERIVATIVE.json`](DERIVATIVE.json) records each edited range (tensor, shard, offset, length, before and after sha256). [`SHA256SUMS`](SHA256SUMS) covers every file in the repository.

NVIDIA's activation calibration for the routed experts was kept as-is and not recalibrated. The edits only touch dense tensors outside the expert path.

One upstream quirk: NVIDIA's `model.safetensors.index.json` still reports the pre-quantization `metadata.total_size` (510.3 GB), while the stored tensor payload is 429.0 GB. That file is left exactly as NVIDIA shipped it.

## What we measured

All numbers come from our own runs on one 4x RTX PRO 6000 box, with stock NVFP4 and this model served by the **same runtime image and arguments**. This is a small set of checks, not a benchmark suite.

**Refusals.** On a fixed set of 100 requests that the stock model tends to refuse (temperature 0, thinking off), stock NVFP4 refused **92/100**; WARMIND-1017-NVFP4 refused **0/100** and gave substantive answers. A keyword screen flagged refusals; the raw answers were kept and spot-checked.

**Long-context correctness.** Sixteen synthetic ledger tasks (16k–256k-token prompts: 8 original, 8 held-out), three seeds each, thinking on, reasoning effort 75, 128k output allowance:

| Model | Correct of 48 |
|---|---:|
| Stock NVFP4 | 40 |
| **WARMIND-1017-NVFP4** | **43** (4 wrong, 1 hit the output limit) |

Paired by prompt and seed: 35 both correct, 5 stock only, 8 WARMIND only; exact McNemar p = 0.58. **No measurable difference**: the small advantage is within noise, and we don't claim the edit improves reasoning. For reference, our CSF-format build of WARMIND-1017 scored 46/48 on the same set, but on a different runtime, so that is not a like-for-like comparison.

**Function.** 34/34 API checks passed: arithmetic, strict JSON schema output, and 16 tool-call round trips across four concurrent conversations.

**Decode speed** (llm-inference-bench 0.7.5, 30 s cells, temperature 1, thinking on, two repetitions averaged; aggregate tok/s):

| Context | C1 | C4 | C8 | C32 |
|---|---:|---:|---:|---:|
| 0 | 214 | 525 | 819 | 1,814 |
| 64k | 210 | 524 | 813 | 1,720 |

KV cache at 0.95 GPU memory utilization: about **3.75M tokens**. Without speculative decoding the same runtime managed about 127 tok/s at C1; [`runtime/README.md`](runtime/README.md) lists what helped and what didn't.

## What we did not test

- **Vision**: we served with `--language-model-only`.
- **Long agentic coding runs and multi-hour soaks.** We ran those on other builds of WARMIND-1017, not this one.
- **Other GPUs, runtimes or vLLM versions.** The stock vLLM 0.32 image does not load this checkpoint on SM120 without the two patches; behaviour on NVIDIA's own GB300 recipe is untested.
- **Broad quality or safety benchmarks.** NVIDIA's published scores in [`README.upstream.md`](README.upstream.md) apply to their unedited release, not to this derivative.

As with any abliterated model, it answers requests the stock model would refuse. You're responsible for how you use it.

## Lineage and credits

- **DeepSeek-AI**: [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT License.
- **NVIDIA**: the [NVFP4 / ModelOpt quantization](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) this derivative is built on.
- **grimjim**: [norm-preserving biprojected abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration).
- **distributedcognition**: the [DeepSeek-V4.1-Flash abliteration recipe and refusal directions](https://huggingface.co/distributedcognition/DeepSeek-V4.1-Flash-abliterated).
- **jcartu**: the WARMIND-1017 derivation, the byte-exact transfer onto NVIDIA's release, the SM120 runtime recipe, and the measurements above.
- vLLM PRs [#61530](https://github.com/vllm-project/vllm/pull/61530) and [#60762](https://github.com/vllm-project/vllm/pull/60762) by their respective authors make this runnable on SM120.

## License

MIT, as published by DeepSeek and carried unchanged by NVIDIA's release ([LICENSE](LICENSE)). The edits add no terms of their own.

---
library_name: mlx
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- apple-silicon
- deepseek-v41
- quantized
- 2-bit
- reap
- experimental
- rapid-mlx
---

## Run it on your Mac

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
rapid-mlx serve deepseek-v41-flash-reap-2bit
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

Use it from OpenCode / Claude Code / Codex: `rapid-mlx agents <name> --setup` · [Agent setup](https://rapidmlx.com/docs/agents)

[Download the Desktop app](https://rapidmlx.com/download) · 256 GB unified memory required for this experimental 212.9 GB target and its DSpark sidecar.

# DeepSeek V4.1 Flash · REAP · MLX 2-bit


> **Experimental research artifact — not yet a product-qualified Rapid-MLX
> model.** It requires a 256 GiB Apple silicon Mac and the experimental loader
> that ships in [Rapid-MLX 0.14.1](https://github.com/raullenchai/Rapid-MLX/releases/tag/v0.14.1).
> Standard model loaders are not supported. Vision and MTP are omitted.

This is a text-only, native packed MLX artifact produced from
[`Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP`](https://huggingface.co/Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP),
itself derived from
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
The conversion preserves the source affine 2-bit/group-64 values while stacking
routed experts for native batched execution. Conservative routing-saliency
pruning keeps 336 of 384 routed experts in every layer.

## Qualification result

Hardware: Apple M3 Ultra, 60 GPU cores, 256 GiB unified memory, MLX 0.32.2.

| Measurement | Result |
| --- | ---: |
| Tensor payload | 212,930,051,680 bytes (~199 GiB) |
| Strict resident load | 239.44 seconds |
| Peak MLX memory | 213.51 GB |
| Conservative decode | 7.31 tok/s |
| Best short-decode ceiling | 7.92 tok/s |
| Routed experts retained | 336 / 384 per layer |
| Mean retained routing mass | 99.9949% |
| Worst-layer retained routing mass | 99.8564% |

Decode throughput uses greedy generation and `(N-1)/elapsed`, excluding
prefill. The same greedy token chain was reproduced at evaluation intervals 4,
20, and 40. Correctly framed smoke output was `The capital of France is Paris.`
followed by EOS.

The artifact misses Rapid-MLX's 12 tok/s product floor by 34.0% and is therefore
published for reproducibility and further optimization only. Uploading it does
not mean that Rapid-MLX catalog, server, or Desktop support is available.

## Initial evaluation results

These are the initial qualification checks completed before publication. They
are intentionally reported as narrow pass/fail evidence, not as a general model
quality score.

| Check | Result | Evidence |
| --- | --- | --- |
| One-sentence factual chat | Pass | Prompt: `What is the capital of France? Answer in one short sentence.` Output: `The capital of France is Paris.` then EOS. |
| Greedy stability | Pass | Evaluation intervals 4, 20, and 40 produced the identical token sequence. |
| Prefill/decode numerical parity | Pass | Tiny-config prefill, decode, and chunked-prefill relative differences were at most 1.4e-6. |
| Next-token parity | Pass | 100% argmax agreement across the tiny-config comparison battery. |
| Quantized tensor fidelity | Pass | Quantize/dequantize cases were bit-exact; strict float and quantized layouts round-tripped. |
| Full artifact load | Pass | All 41 shards strict-loaded with no missing or unexpected text-model parameters. |
| Vision, tool use, long context | Not evaluated | These capabilities are not claimed by this artifact. |

The factual chat case used the required model framing:

```text
<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? Answer in one short sentence.<｜Assistant｜></think>
```

Raw prompts without these role markers produced invalid diagnostics and are not
counted as evaluation results.

## Calibration and validation

- REAP calibration used 2,048 real causal tokens split into two disjoint halves.
- Split-half expert keep-set overlap was 92.1354%.
- Tiny-config parity checks passed with relative differences no greater than
  1.4e-6 and 100% argmax agreement.
- Quantize/dequantize cases were bit-exact; strict float/quantized layouts and
  streaming output matched.
- The 41-shard artifact strict-loaded without missing or unexpected text-model
  parameters.

These checks do not establish broad quality, tool-use, long-context, or safety
parity with the original model. Two-bit quantization and expert pruning may
reduce quality. Long context has not been qualified.

## Experimental usage

Clone the exact qualification implementation and download the model through the
standard Hugging Face cache:

```bash
git clone --branch v0.14.1 https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX

hf download rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX
python scripts/qualify_deepseek_v41_native.py \
  ~/.cache/huggingface/hub/models--rapid-mlx--DeepSeek-V4.1-Flash-REAP-2bit-MLX/snapshots/<revision> \
  --tokens 16 --eval-intervals 4
```

Allow substantial memory headroom for macOS and context growth. This artifact
is not suitable for machines below 256 GiB unified memory.

## Background

The runtime is a native MLX implementation of the V4.1 architecture — attention,
MoE, engram, hyper-connections, indexer, sparse attention and a chunked-prefill
streaming generator — written for this qualification and shipped in Rapid-MLX
0.14.1 under `scripts/deepseek_v41_native/`.

Why the artifact is published even though it misses the product floor, and what
the numbers mean in practice, is written up here:
[DeepSeek V4.1 on a Mac, and Why Your LLM API Router Is Reading Your Prompts](https://rapidmlx.com/blog/deepseek-v41-on-a-mac-and-the-router-problem).

For models that *are* product-qualified and run on an ordinary Mac, see
[rapidmlx.com](https://rapidmlx.com).

## License and attribution

The MIT license from the original release is included in `LICENSE`. Model
design and training belong to DeepSeek and its contributors. The source 2-bit
conversion was published by Vontra; this independent REAP repack was produced
by the Rapid-MLX project. See `NOTICE` for the pinned source revision.

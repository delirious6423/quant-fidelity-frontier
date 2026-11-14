---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - qwen
  - jblaze
---

# Qwen3.8-27B-jBlaze (Q4_K_M GGUF)

A weight-edited Qwen3.8-27B by Apollo Raines, edited with jBlaze and quantized to a standard Q4_K_M GGUF. It loads in unmodified llama.cpp (tested on build 10630, commit d222767c7). No custom fork, no runtime changes, nothing to build. Ollama and LM Studio are untested.

## What it does

Answers finish sooner on long generations. The edit reduces long generations: 28 pct fewer output tokens than the vanilla model on the eval below, with the gains concentrated in long generations. Turning on llama.cpp speculative decoding with the model's built-in MTP head (one stock flag, see below) shortens the slow tail further.

## Measured results

Sealed 85-item eval (math, knowledge, science, structured output), two seeds pooled, 170 generations per arm, identical prompts and settings, run back to back on stock llama.cpp build 10630 (commit d222767c7), one RTX 3090. Times are wall time to the completed answer.

| Model | Mean | p90 | Raw decode | Accuracy |
| --- | --- | --- | --- | --- |
| Vanilla Qwen3.8-27B Q4_K_M | 10.249 s | 14.891 s | 38.12 tok/s | 96.47 |
| Vanilla Qwen3.8-27B Q4_K_M + MTP flag | 9.058 s | 12.583 s | 48.41 tok/s | 95.88 |
| Qwen3.8-27B-jBlaze | 7.408 s | 12.817 s | 38.48 tok/s | 95.88 |
| Qwen3.8-27B-jBlaze + MTP flag | 7.505 s | 9.651 s | 47.93 tok/s | 95.29 |

In plain terms: with no flags, mean time-to-answer is 1.38x faster than vanilla (28 pct less time). With the MTP flag, p90 (the slow one-in-ten answer) is 1.54x faster than vanilla without the flag (35 pct less time).

Most of the MTP gain comes from the MTP flag itself, which vanilla also gets. Measured against vanilla with the same MTP flag, the edit's own share is a mean ratio of 0.829x and a p90 ratio of 0.767x; the median item is unchanged (paired median ratio 1.005).

For reference, the public Thinklite-27B Q4_K_M, measured on the same instrument, scored mean 5.985 s, p90 11.878 s, accuracy 96.47. Without flags this model is slower than Thinklite on both mean and p90; with the MTP flag its p90 (9.651 s) is lower than Thinklite's (11.878 s), while its mean (7.505 s) is still higher.

## Run it

VRAM: fits on a single 24 GB GPU; every measurement above ran on one RTX 3090. The weights take about 16 GB; the rest is KV cache, which grows with context length (the eval ran generations up to 8192 tokens inside that budget).

```
llama-cli -m Qwen3.8-27B-jBlaze-Q4_K_M.gguf -p "your prompt"
```

For the best tail latency (uses the model's built-in MTP draft head):

```
llama-cli -m Qwen3.8-27B-jBlaze-Q4_K_M.gguf --spec-type draft-mtp -p "your prompt"
```

## Honest limitations

- This model was built against a 2x speed challenge. It does not reach 2x. The typical (median) prompt finishes in about the same time as vanilla; the gains are concentrated in long generations.
- Accuracy on the sealed eval is 95.88 vs 96.47 for vanilla, a 0.59 point difference. On a separate 80-item general control set, accuracy was higher than the vanilla baseline (81.25 vs 78.75), with one additional non-delivery (4 vs 3).
- Distributional shift was measured and bounded before release: KL divergence against a Q8_0 reference is 0.022477, inside the pre-registered ceiling of 0.027788; perplexity delta 0.049245 against a ceiling of 0.057469.
- With the MTP flag on, outputs are not guaranteed byte-identical to runs without it. That is a property of the speculative decoding path in our testing, not of the edit.

## Files

- `Qwen3.8-27B-jBlaze-Q4_K_M.gguf` - the model. SHA-256 begins `80ada412b555ae9d`; the full checksum is in the `SHA256SUMS` file in this repository.

## License

Apache-2.0, the same terms as the base model, Qwen/Qwen3.8-27B. This is a derivative work.

Edited with jBlaze. Benchmarks measured 2026-11-13.

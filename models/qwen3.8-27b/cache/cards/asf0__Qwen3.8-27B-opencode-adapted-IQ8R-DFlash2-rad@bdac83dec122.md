---
license: apache-2.0
base_model:
- harness-hub/Qwen3.8-27B-opencode-adapted
- Qwen/Qwen3.8-27B
- z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
- radiance
- rocm
- amd
- rdna4
- int8
- w8a8
- iq8r
- dflash2
- speculative-decoding
- opencode
- agent
---

# Qwen3.8-27B OpenCode adapter, IQ8R + DFlash2: radiance container

The [harness-hub/Qwen3.8-27B-opencode-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted) LoRA adapter merged into [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and converted to a single `.rad` container for the
[radiance](https://codeberg.org/StillDeadcode/radiance) inference engine, with the [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter merged in for speculative decoding. The adapter
was trained for the OpenCode agent with thinking off, but it also works with thinking on: in my
Terminal-Bench 2.1 runs it had the best score of seven configurations in both modes, 51 of 87 with
thinking off and 59 at `xhigh` (stock: 43 and 56). Neither gain over stock is significant; its clearest
benefit is less output (a median of 55% less than stock with thinking off and 31% less at xhigh).

| | |
|---|---|
| File | `qwen3.8-27b-opencode-iq8r-dflash.rad` |
| Source | adapter merged onto the base in bf16 (256 LoRA weights, largest relative change 0.0026), then quantised |
| Format | IQ8R by round to nearest (recipe `q38-27b-iq8r-rtn.recipe`, in this repository): every trunk linear int8 with a u8 scale per row and 256 of K under a bf16 scale a row, each 128 of K rotated by the Hadamard transform; int8 activations (W8A8); the lm_head int8. No calibration data. |
| Speculator | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) block-diffusion drafter, merged in, 4-bit codebook |
| Size | 28.76 GiB |
| Engine | [radiance](https://codeberg.org/StillDeadcode/radiance) 2.2.0 (AMD RDNA4, ROCm) |

## Results

### Terminal-Bench 2.1 (agentic, OpenCode)

| Configuration | Solved (of 87) | Easy (4) | Medium (53) | Hard (30) | Output tokens | Tokens processed |
|---|---|---|---|---|---|---|
| Stock Qwen3.8-27B, thinking off | 43 (49%) | 3 | 32 | 8 | 2.90M | 402M |
| **OpenCode adapter, thinking off** | **51 (59%)** | 3 | 41 | 7 | 1.77M | 181M |
| ThinkingCap + OpenCode adapter, thinking off | 47 (54%) | 3 | 34 | 10 | 1.93M | 219M |
| Stock Qwen3.8-27B, xhigh | 56 (64%) | 2 | 40 | 14 | 4.44M | 167M |
| ThinkingCap, xhigh | 56 (64%) | 2 | 40 | 14 | 4.19M | 165M |
| **OpenCode adapter, xhigh** | **59 (68%)** | 3 | 40 | 16 | 3.83M | 168M |
| ThinkingCap + OpenCode adapter, xhigh | 58 (67%) | 4 | 41 | 13 | 3.44M | 147M |

Terminal-Bench 2.1 through the Harbor harness with the OpenCode agent (2.2.7), web search off, a
1-hour agent budget per task, one run per configuration (24–26 November 2026; stock and the adapter
with thinking off were run twice, and the table shows the second runs). All seven use the same
IQ8R-RTN + DFlash2 containers on radiance 2.2.0, two AMD Radeon AI PRO R9700 cards (tensor parallel
2), 8 tasks at once, a 131,072-token context, temperature 0.8 and top-p 1.0. The two QEMU tasks failed
in every run on a missing Debian package and are left out (87 tasks scored); tasks lost to
infrastructure were rerun once. Token counts are OpenCode's own per-step reports over all 87 tasks.
"Stock" is the RTN container of the base model, made with the same recipe.

What the comparison shows, from paired tests on the same tasks (exact McNemar):

- **Thinking is the big effect:** stock goes from 43 to 56 tasks (p = 0.03), the adapter from 51 to
  59 (p = 0.13), ThinkingCap + adapter from 47 to 58 (p = 0.04).
- **The OpenCode adapter, thinking off:** 51 against stock's 43 (p = 0.20), in the direction of the
  adapter card's reported gain; it needs about half of stock's agent steps and writes a median of 55%
  less output on tasks both solved.
- **With thinking on, the four configurations tie:** 56 to 59, every pair p ≥ 0.58. ThinkingCap alone
  scores exactly like stock (56, the same at every difficulty).
- **The fine-tunes save output at xhigh:** on tasks solved by both, a median of 12% less than stock
  for ThinkingCap, 31% less for the adapter and 44% less for ThinkingCap + adapter.

Repeat runs with thinking off barely moved the totals (stock 40 then 43, the adapter 49 then 51) but
disagreed on 31 and 22 of the 87 tasks, so with one run per configuration, differences of a few
tasks can be chance.

### General benchmarks (QualityBench quick)

| Task | Stock (GPTQ), off | Stock (RTN), off | **OpenCode adapter, off** |
|---|---|---|---|
| GSM8K (132) | 97.7 | 97.7 | 97.7 |
| IFEval (55) | 87.3 | 90.9 | 85.5 |
| MMLU-Pro (100) | 82.0 | 77.0 | 82.0 |
| HumanEval (17) | 94.1 | 94.1 | 94.1 |
| MBPP (38) | 81.6 | 81.6 | 81.6 |
| BFCL (40) | 90.0 | 90.0 | 85.0 |
| LiveCodeBench v6 (18) | 50.0 | 61.1 | 55.6 |

QualityBench quick run (a fixed tenth of
each task's items, so intervals are about three times wider than a full run's), same server, caps of
32,768 output tokens with thinking off and 98,304 with it on. Accuracy in %, items per task in
brackets. The stock column here is the GPTQ-fitted base container
(`qwen3.8-27b-iq8r-dflash.rad`); the RTN stock run with thinking on did not complete. No task
differs from stock by more than its 95% interval allows (QualityBench's T5 check passes 7/7).

## Serve

```yaml
# radiance.yaml
server:
  host: 0.0.0.0
  port: 8000
models:
  opencode:
    model: qwen3.8-27b-opencode-iq8r-dflash.rad
    tp: 2
    max-model-len: 131072
    max-num-seqs: 8
    kv-cache-dtype: i8
    reasoning-effort: xhigh
```

```sh
radiance --config radiance.yaml
```

For the best score run it at `xhigh`; set `none` (or send `reasoning_effort: none`) for fast runs with thinking off, the setting the adapter was trained in. `reasoning-effort` is the server's default for requests that set none; a request can override it
(`reasoning_effort`, or `chat_template_kwargs.reasoning_effort`). The server speaks the OpenAI API
(`/v1/chat/completions`, `/v1/completions`) with tool calls. Tested with radiance 2.2.0 on 2× Radeon
AI PRO R9700 at 131K context with eight sequences, the drafter on and the int8 KV cache.

## How this file was made

```sh
# Qwen3.8-27B-opencode-adapted/ = a local download of the adapter repository
rad-export Qwen3.8-27B-opencode-adapted/ --base Qwen/Qwen3.8-27B \
    --recipe q38-27b-iq8r-rtn --draft z-lab/Qwen3.8-27B-DFlash2 \
    -o qwen3.8-27b-opencode-iq8r-dflash.rad
```

`rad-export` (radiance's tools) merges the LoRA weights into a copy of the bf16 base and runs
`rad-convert` with the recipe above. IQ8R rotates the weights, so the adapter cannot be applied at run
time and is merged before quantisation.

## License

Apache-2.0, as the base model and the adapter (see `LICENSE`). Credit: the OpenCode adapter by
harness-hub, Qwen3.8-27B by the Qwen Team, the DFlash2 drafter by z-lab.

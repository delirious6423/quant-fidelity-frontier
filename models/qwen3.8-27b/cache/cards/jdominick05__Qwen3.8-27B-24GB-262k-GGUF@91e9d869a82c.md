---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - imatrix
  - mixed-precision
  - qwen3.8
  - rocm
  - 24gb
  - long-context
  - mtp
  - speculative-decoding
---

# Qwen3.8-27B GGUF: the full 262,144-token context on one 24 GB GPU

Two mixed-precision quantizations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), both measured to run
the model's **full native 262,144-token context** entirely in VRAM on one 24 GB card (RX 7900 XTX, llama.cpp/ROCm,
Windows), with a 4-bit KV cache. Pick by workload:

| File | Size | Use it for | Why |
|---|---|---|---|
| **`Qwen3.8-27B-R3-mix.gguf`** | 17.0 GiB | **Agents** (Hermes, Qwen Code, anything that reads tool output every turn) and accuracy | Most accurate (closest to Q8_0 of everything measured). Fast at reading new context: ~890 tok/s for appended tool output in real agent use. No MTP layer |
| **`Qwen3.8-27B-R5-mix-MTP.gguf`** | 16.4 GiB | **Chat and long answers** at any context length | Keeps the multi-token-prediction head: MTP drafting generates 33-64 tok/s (+22% to +80%). Appending to an existing context is ~5x slower with MTP on, so it is a poor fit for tool-heavy agent loops |

Both are text-only. Unsloth's `mmproj-F16.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
is built from the same base model and should work, but was not tested. Earlier files (R1-mix, R2-mix) were retired;
their measurements stay in the tables below and in `data/`.

## Quality: KL divergence against Q8_0

Reference: Q8_0 of the same BF16 weights; `llama-perplexity --kl-divergence` on 65,536 tokens (16 x 4,096) of technical
text (decompiler output, C, notes) interleaved with coding-agent transcripts. Lower KLD = closer to the reference.

| Model | Size (MiB) | Mean KLD | Median KLD | Same top token | PPL vs Q8_0 | 99.9th pct KLD |
|---|---|---|---|---|---|---|
| **R3-mix (this repo)** | 17,390 | **0.0142 ± 0.0011** | **0.00074** | **96.10%** | **+0.1%** | 1.91 |
| **R5-mix-MTP (this repo)**¹ | 16,830 | 0.0165 ± 0.0014 | 0.00097 | 95.90% | +0.3% | **1.89** |
| R2-mix (retired) | 14,677 | 0.0199 ± 0.0012 | 0.00131 | 95.08% | +0.5% | 2.46 |
| R1-mix (retired) | 16,167 | 0.0191 ± 0.0015 | 0.00122 | 94.96% | +0.8% | 2.47 |
| Unsloth `UD-IQ4_XS` | 13,593 | 0.0273 ± 0.0017 | 0.00161 | 94.74% | +1.0% | 4.22 |
| IST-DASLab `GSQ-RCO IQ3_S` | 11,226 | 0.0645 ± 0.0028 | - | 92.99% | +5.2% | 8.79 |

¹ Measured on the same weights without the MTP layer (the MTP head does not change the main model's outputs).

Mean KLD is dominated by rare spike tokens and shifts by ~0.002 between texts; compare on the median, top-token and
99.9% columns. The 99.9% tail matters most for agents: single bad tokens are what break tool calls. Single runs.
Ranking was done on paired chunk-by-chunk top-1 agreement (`scripts/paired.py`).

## Speed, memory and long context (RX 7900 XTX, 262,144-token context, KV q4_0)

| | R3-mix | R5-mix-MTP (MTP on) |
|---|---|---|
| Dedicated VRAM | 23.1-23.3 GB | 23.5 GB (needs `-ub 512`; `-ub 2048` spills ~3.4 GB) |
| Recall: 3 planted lines at 10/50/90% of 248,139 tokens | 3/3, 524 tok/s average fill | 3/3 on the same weights without MTP (526 tok/s) |
| Generation | 34.5 tok/s at ~28k (~34 in a 128k agent workload) | **64 / 56 / 33 tok/s** at ~1k / ~28k / ~128k (draft acceptance 72-79%) |
| Fresh prompt (average over the fill) | 673-961 tok/s at ~26-28k; 524 tok/s over 248k | 920 / 613 tok/s at ~28k / ~128k |
| **Appended tokens in real agent use** (Hermes call logs)² | **~890 tok/s** | **~186 tok/s** |

² R3 column: 275 calls on R3-mix and the closely related R3-M (median context 77k); R5 column: 13 calls (median 47k).

The last row decides the use case: an agent adds tool output every turn, and with MTP drafting enabled those appends
were ~5x slower, so a 7-8k-token tool result took ~47 s instead of ~10-15 s. For chat, where turns are short and
answers long, MTP's faster generation wins. Details in `data/speed.md`.

## Recommended llama-server commands (24 GB GPU, full context)

R3-mix (agents):
```
llama-server -m Qwen3.8-27B-R3-mix.gguf -dev ROCm0 -ngl 99 -c 262144 -np 1 -fa on -b 2048 -ub 2048 \
  -ctk q4_0 -ctv q4_0 --cache-ram 0 -ctxcp 8 -cms 2048 --reasoning-effort medium \
  --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0
```
R5-mix-MTP (chat): same, but `-m Qwen3.8-27B-R5-mix-MTP.gguf -ub 512` plus
`--spec-type draft-mtp --spec-draft-n-max 2 -ctkd q4_0 -ctvd q4_0`.

Notes from running this 24/7 on Windows/ROCm:
- **Batch:** `-ub 2048` is best without MTP. A 19-configuration sweep (batch sizes, hipBLASLt, forced MMQ, CUDA graphs,
  pinned memory, SDMA, a newer build) stayed within ~1% (`data/flag_sweep.md`).
- **ROCm vs Vulkan** (b11583, same card): Vulkan generates ~9% faster but processes prompts ~41% slower at 32k depth.
  At long context prompt processing dominates, so ROCm.
- **System memory:** Windows commits system memory roughly equal to the server's VRAM (~22-23 GB at 262k). Context
  checkpoints (~60-215 MB each) and `--cache-ram` come on top. On a 32 GB machine use a page file of 32 GB or more and
  `--cache-ram 0 -ctxcp 8` for a single agent; checkpoints are what make hybrid-model cache reuse work, so do not drop
  them to 0-1.
- **Idle VRAM eviction:** Windows moves an idle server's VRAM into system RAM (0 GB on the GPU, ~20 GB working set)
  and pages it back on the next request. The server is fine; do not use VRAM readings as a health signal.
- **Desktop compositor:** `dwm` can accumulate ~3 GB of VRAM over time; disabling and re-enabling the GPU in Device
  Manager released it here (any running server must be restarted afterwards).
- **`-lm none` with a partial GPU split** crashes this build at startup ("ROCm error: unspecified launch failure");
  full offload is fine.
- **Reasoning:** the chat template defaults to `reasoning_effort = xhigh` when the request sets none; set
  `--reasoning-effort medium` (or `chat_template_kwargs`) explicitly.
- If the CPU has integrated graphics, ROCm lists it as a second device; pin the GPU with `-dev ROCm0`.

## Recipes

Base type IQ4_XS with per-tensor overrides (`llama-quantize --tensor-type-file`, files in `recipes/`); imatrix and BF16
source from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). R3 raises the tensor groups
with the largest measured quality gain per GiB until 262k fits; R5 was chosen by a budget solver (`scripts/budget_solver.py`)
from paired per-tensor measurements and leaves room for the MTP head. Feed-forward weights below 4 bits (IQ3) cost far
more accuracy than they saved and were not faster on ROCm; a workspace-specific imatrix raised top-1 by ~0.2 points
but worsened the 99.9% tail, so both files use Unsloth's.

| Tensors | R3-mix | R5-mix-MTP |
|---|---|---|
| `ffn_up` | Q5_K | Q5_K |
| `ffn_gate` | IQ4_XS | IQ4_XS |
| `ffn_down` | Q5_K | Q5_K |
| recurrent `attn_qkv` | Q5_K | IQ4_XS |
| recurrent `attn_gate` | Q5_K | IQ4_XS |
| recurrent `ssm_out` | Q5_K | IQ4_XS |
| full attention `attn_q` | Q5_K | Q5_K |
| full attention `attn_output` | Q6_K | Q6_K |
| full attention `attn_k`, `attn_v` | Q8_0 | Q8_0 |
| `ssm_alpha`, `ssm_beta` | Q8_0 | Q8_0 |
| `output` | Q5_K | Q5_K |
| `token_embd` | Q8_0 | Q8_0 |
| MTP layer | removed | kept (same per-tensor types) |

```
llama-quantize --imatrix imatrix_unsloth.gguf --tensor-type-file recipes/R5.txt \
  --output-tensor-type q5_k --token-embedding-type q8_0 Qwen3.8-27B-BF16-00001-of-00002.gguf R5-mix-MTP.gguf IQ4_XS
```
R3: `recipes/R3.txt`, add `--prune-layers 64`, then `gguf-set-metadata out.gguf qwen35.nextn_predict_layers 0`
(otherwise llama.cpp still expects the removed MTP layer and runs 63 layers).

## Data and scripts

`data/` holds every KLD measurement (including all recipe steps and the comparison quants), the flag sweep and the
speed/memory results; `scripts/` the measurement tools, the budget solver, and a bit-exact PyTorch implementation of
llama.cpp's block formats as groundwork for GGUF-native QAT. See the READMEs in each folder.

## Related quants

- [jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF](https://huggingface.co/jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF): RTX 3090, 200K
  context with an 8-bit K cache and MTP; ties retired R2-mix on KLD.
- [Wa1k3r/Qwen3.8-27b-CODER-4q_xs-24GB-262k-Optimalcardfit](https://huggingface.co/Wa1k3r/Qwen3.8-27b-CODER-4q_xs-24GB-262k-Optimalcardfit):
  262k + MTP on a 7900 XTX with Vulkan; abliterated variant, no quality measurements published.
- [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF): learned rounding +
  budgeted per-tensor types at 2.5-3.5 bpw (8.4-11.8 GB); much smaller, further from Q8_0 (table above).
- [JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB):
  EXL3 for NVIDIA (vLLM), 215k with vision and MTP on an RTX 3090.

## License and credits

Apache 2.0, as the base model; see `LICENSE`. Base model by the Qwen team; BF16 GGUF conversion and imatrix by
[Unsloth](https://huggingface.co/unsloth). Recipes, measurements and scripts by jdominick05.

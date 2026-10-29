---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- gptq
- 4-bit
- agent
---

# sady4850/Qwen3.8-27B-4bit-GPTQ-agentmix

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) in exactly the layout of
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (MLX affine 4-bit, group size 64), quantized by GPTQ like
[`sady4850/Qwen3.8-27B-4bit-GPTQ`](https://huggingface.co/sady4850/Qwen3.8-27B-4bit-GPTQ), but with a third of the calibration taken from the model's own coding-agent
traces. Same tensors, shapes, dtypes, size (15 GB) and speed as the template; only the numbers inside the 4-bit tensors differ.

Against the published GPTQ checkpoint on held-out agent text: -32% KLD on reasoning, -26% on
tool calls, +5.5 points of turns with the same tool call as bf16. On general text it is about the same
(wiki slightly worse, chat equal), see below.

- 497 Linears requantized (all 64 layers incl. `lm_head`). The embedding and the vision tower are the template's.
- GPTQ fitted to MLX's affine format (group scale and bias from the error-updated weights, rounded to bf16 before coding),
  Hessians from the bf16 model, one per shared input, damp 0.01, block 128.
- Calibration: 40 chunks of UltraChat train conversations, 40 of calibration_v5 text and 40 of agent traces (120 x 2048 tokens, the published checkpoint's budget with a third swapped for agent traces). Agent chunks are 2048-token windows of Qwen3.8-27B bf16 Pi traces (low and medium reasoning) on the
  calibration half of the DeepSWE tasks, rendered as the model sees them (system prompt, tools, reasoning, tool calls, tool output).

## Use

```bash
pip install -U mlx-vlm
python -m mlx_vlm.generate --model sady4850/Qwen3.8-27B-4bit-GPTQ-agentmix --max-tokens 100 --prompt "Hello"
```

In [Splash](https://github.com/paperniuk/splash) use this repo id in place of `mlx-community/Qwen3.8-27B-4bit`.

## Agent text vs bf16

Agent test set: Qwen3.8-27B's own bf16 coding-agent traces from
[`kaitchup/DeepSWE1.1-trajectories-Qwen3.8-27B`](https://huggingface.co/datasets/kaitchup/DeepSWE1.1-trajectories-Qwen3.8-27B)
(Pi harness, xhigh reasoning), 49 tasks that are not in the calibration half, rendered with the Pi 0.84.2 system prompt
and tools and the Qwen3.8 chat template, up to 24K tokens per task: 1.13M tokens, 509 assistant turns with a tool call.
Teacher-forced in torch (A100) with every checkpoint dequantized into the same bf16 model. KLD = KL(bf16 || quant) of the
next-token distribution on the reasoning tokens (252K rows) and the tool-call tokens (32K rows). Same call = the
checkpoint's top-1 token equals bf16's on every tool-call token of the turn.

| model | reasoning KLD | p99 | tool-call KLD | p99 | same call |
|---|---|---|---|---|---|
| `mlx-community/Qwen3.8-27B-4bit` (RTN) | 0.0387 | 0.315 | 0.0091 | 0.183 | 47.3% |
| `sady4850/Qwen3.8-27B-4bit-GPTQ` (GPTQ) | 0.0260 | 0.216 | 0.0060 | 0.122 | 54.4% |
| this repo | **0.0176** | 0.147 | **0.0045** | 0.084 | **59.9%** |

Against `sady4850/Qwen3.8-27B-4bit-GPTQ`, paired bootstrap over the 49 tasks (95% CI): reasoning KLD -32% [-34%, -31%], tool-call KLD
-26% [-30%, -21%], same call +5.5 [+1.4, +9.6] points. For scale, greedy bf16 reproduces the recorded call in
30.3% of turns (the traces were sampled at temperature 1.0).

## General text vs bf16

Same texts and protocol as the [`sady4850/Qwen3.8-27B-4bit-GPTQ`](https://huggingface.co/sady4850/Qwen3.8-27B-4bit-GPTQ) card: wiki = wikitext-2 test, chat = UltraChat test,
30 chunks x 2048 tokens per text, scored on the second half of each chunk.

**torch** (A100; the GPTQ and RTN rows were rerun in the same session, within 1% of the published card):

| text | model | KLD mean | KLD median | KLD p99 | flips | PPL |
|---|---|---|---|---|---|---|
| wiki | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 0.0554 | 0.0228 | 0.433 | 9.53% | 6.484 |
| wiki | `sady4850/Qwen3.8-27B-4bit-GPTQ` (GPTQ) | 0.0367 | 0.0122 | 0.313 | 7.41% | 6.473 |
| wiki | this repo | 0.0391 | 0.0133 | 0.330 | 7.54% | 6.383 |
| chat | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 0.1817 | 0.0302 | 3.393 | 12.93% | 8.126 |
| chat | `sady4850/Qwen3.8-27B-4bit-GPTQ` (GPTQ) | 0.0986 | 0.0123 | 1.767 | 8.83% | 8.742 |
| chat | this repo | 0.1034 | 0.0151 | 1.779 | 9.37% | 8.251 |

**Splash** (`teacher-logits`, int8 KV, M1 Pro, same rows and bf16 top-512 reference as the GPTQ card):

| text | model | KLD mean | KLD median | KLD p99 | flips | PPL |
|---|---|---|---|---|---|---|
| wiki | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 0.0562 | 0.0221 | 0.469 | 9.58% | 6.479 |
| wiki | `sady4850/Qwen3.8-27B-4bit-GPTQ` (GPTQ) | 0.0399 | 0.0119 | 0.326 | 7.49% | 6.458 |
| wiki | this repo | 0.0419 | 0.0131 | 0.359 | 7.74% | 6.371 |
| chat | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 0.1906 | 0.0327 | 2.724 | 13.13% | 8.141 |
| chat | `sady4850/Qwen3.8-27B-4bit-GPTQ` (GPTQ) | 0.1002 | 0.0128 | 1.576 | 9.05% | 8.706 |
| chat | this repo | 0.1068 | 0.0156 | 1.453 | 9.54% | 8.195 |

On wiki this checkpoint is about 5% further from bf16 than the published GPTQ (seen in torch on two calibration draws and in
Splash, where the 30-window CI includes 0); on chat the two are equal within noise. Both stay far from RTN.

## Calibration variants

All five checkpoints of the study, against `sady4850/Qwen3.8-27B-4bit-GPTQ` (agent columns with 95% CI over tasks; wiki/chat = torch KLD mean):

| variant | reasoning KLD | tool-call KLD | same call (pts) | wiki KLD | chat KLD |
|---|---|---|---|---|---|
| `-agentmix` | -32% [-34%, -31%] | -26% [-30%, -21%] | +5.5 [+1.4, +9.6] | +7% | +5% |
| `-agentmix-b` | -31% [-33%, -29%] | -29% [-33%, -25%] | +5.5 [+1.2, +9.8] | +5% | -1% |
| `-agentmix20` | -24% [-26%, -22%] | -21% [-26%, -15%] | +1.8 [-2.2, +5.6] | +8% | +1% |
| `-agent` | -41% [-44%, -39%] | -41% [-45%, -35%] | +7.9 [+4.0, +11.7] | +54% | +54% |
| `-agentw` | -44% [-46%, -41%] | -42% [-47%, -36%] | +7.7 [+3.1, +12.1] | +48% | +66% |

- Drawing other agent chunks (`-agentmix-b`) moves the agent numbers by about 2-5%, so differences of that size between
  variants are calibration noise, not an effect.
- Agent traces alone (`-agent`, `-agentw`) are closest on agent text but lose the general-text gain (wiki at RTN level).
- A smaller agent share (`-agentmix20`) keeps less of the agent gain and does not remove the wiki cost.

## Limitations

The agent test set comes from the same harness (Pi) and task source (DeepSWE) as the agent calibration, on disjoint tasks;
other harnesses were not measured. Agent numbers are teacher-forced (no free-generation replay yet). Vision was not
evaluated, task accuracy was not measured, windows are at most 24K tokens.

Raw numbers in `eval/`.

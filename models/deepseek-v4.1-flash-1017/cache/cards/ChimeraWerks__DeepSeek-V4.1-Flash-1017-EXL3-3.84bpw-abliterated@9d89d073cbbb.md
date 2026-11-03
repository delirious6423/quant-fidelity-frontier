---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
- deepseek
- deepseek-v4
- exl3
- exllamav3
- abliterated
- uncensored
- mixture-of-experts
- dgx-spark
- tensorfold
- tensor-parallelism
---

<p align="center">
  <img src="assets/hero.jpg" width="100%" alt="A glowing orange-red mesh whale rising over dark reflective water among soft bokeh lights. DeepSeek V4.1 Flash, abliterated, 3x DGX Spark, 3.84 bpw EXL3: 3,833 tok/s prefill, 126 tok/s code, 1M context, 90.5% GPQA Diamond" />
</p>

# DeepSeek-V4.1-Flash-1017 · EXL3 3.84 bpw · abliterated

An uncensored **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** in EXL3,
built to run across three NVIDIA DGX Spark systems with three-way tensor parallelism. It is the closest pack to the
official model of every one we scored, abliterated or not: mean KL **0.0784**, 92.0% top-1 agreement.

The serving recipe, engine and every benchmark behind these numbers are on
**[GitHub](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated)**.

## At a glance

| | v1.1, measured 2–3 November 2026 |
| --- | --- |
| Prefill | **3,560–3,833 tok/s** from 4K to 128K tokens on 3× DGX Spark; a 128K-token prompt in 35 s |
| Decode | 126 tok/s single-stream code, 84 prose; up to 263 tok/s aggregate at 16 streams |
| Fidelity | Mean KL **0.0784** and 92.0% top-1 agreement with official DeepSeek-V4.1-Flash-1017 logits |
| Quality | **97.5%** AIME 2025; **90.5%** GPQA-Diamond (1,434 of 1,584); 88.7% IFEval; 88.0% LiveCodeBench v6; 88.0% MMLU-Pro (500-item subset) |
| Refusals | 0 of 100 HarmBench prompts refused (98 answered, 2 partial); 0 of 250 XSTest safe prompts fully refused |
| Context | 1,048,576 tokens configured; largest measured prompt 131,090 tokens |
| Quantization | EXL3, 3.8361 bpw over quantized linear layers (output head excluded, 6-bit) |
| Edit | [mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated)'s rank-2 refusal-removal edit, applied in FP32 and re-encoded at 6 bits |
| Size | About 274 GB: 35 safetensors shards |

## Speed

Measured 2 November 2026 on engine `14803a3` with [sparkDash](https://github.com/MiaAI-Lab/sparkDash).

<p align="center"><img src="assets/chart-prefill.png" width="100%" alt="Two line charts against prompt length from 4K to 128K tokens. Prefill tok/s: 3,560, 3,708, 3,811, 3,833, 3,800 and 3,721. Time to first token: 1.3, 2.4, 4.5, 8.7, 17.4 and 35.4 seconds" /></p>

| Context | Prompt tokens | Time to first token | Prefill tok/s |
| ---: | ---: | ---: | ---: |
| 4K | 4,115 | 1.32 s | 3,560 |
| 8K | 8,215 | 2.38 s | 3,708 |
| 16K | 16,404 | 4.47 s | 3,811 |
| 32K | 32,790 | 8.72 s | **3,833** |
| 64K | 65,556 | 17.42 s | 3,800 |
| 128K | 131,090 | 35.40 s | 3,721 |

Cold prompts, zero cached tokens, one request at a time; tok/s comes from time to first token with the collector's
165.82 ms request overhead subtracted.

<p align="center"><img src="assets/chart-decode.png" width="100%" alt="Two line charts of combined decode tok/s against requests at once. Greedy, prose and code: C1 84 and 126, C2 112 and 169, C4 166 and 192, C8 234 and 219, C12 249 and 250, C16 263 and 254. Thinking, prose and code: C1 69 and 86, C2 95 and 122, C4 135 and 147, C8 174 and 176, C12 197 and 193, C16 217 and 211" /></p>

400 output tokens per request. Greedy is temperature 0 with thinking off; thinking is temperature 0.7 with thinking on.
Every point and per-stream timing:
[benchmarks](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/benchmarks/README.md).

**What each request gets.** Mean decode tok/s per request, timed at the engine on short, already-cached prompts. Mean time to first token per load was 27–59 ms; single requests ranged from 10 to 110 ms.

| Requests at once | Prose, greedy | Code, greedy | Prose, thinking | Code, thinking |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 84.3 | 125.7 | 68.6 | 86.4 |
| 2 | 57.9 | 85.0 | 48.4 | 61.7 |
| 4 | 42.0 | 49.0 | 34.5 | 38.5 |
| 8 | 30.2 | 29.9 | 22.6 | 24.3 |
| 12 | 21.5 | 22.1 | 17.0 | 17.8 |
| 16 | 16.9 | 17.1 | 14.0 | 14.7 |

## Fidelity

<p align="center"><img src="assets/chart-kl.png" width="100%" alt="Lollipop chart of mean KL to the official model, with a line at the unedited 3.5 bpw quant: this release 0.0784; LakyFx 3.5 bpw unedited 0.0872, with mrfakename 0.0937, with heretic 0.1002; Keys overlay on our 3.84 bpw 0.1111, on LakyFx 3.5 bpw 0.1225, Mia's Keys K5 on LakyFx 3.5 bpw 0.1234, Keys overlay on LibertAIDAI 3.5 bpw 0.1268" /></p>

KL is mean per-token KL(official ‖ this model) against the published official-model logits
([brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits)):
65 cuts of 2,048 tokens, 133,120 positions, balanced across agentic, chat, code, SWE, math, science, legal,
multilingual, long-context, safety and pretraining text.

- Lower KL than the unedited 3.5 bpw quant on 59 of 65 cuts (paired mean 0.0088 lower, 95% CI 0.0035 to 0.0131).
- The same mrfakename edit scores 0.0937 on a 3.5 bpw base and 0.0784 on this 3.84 bpw mix.
- p99 KL 1.463. Every arm:
  [fidelity.csv](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/benchmarks/v1/fidelity.csv).

## Quality and refusals

<p align="center"><img src="assets/chart-quality.png" width="100%" alt="Quality and refusals. GPQA-Diamond 90.5% against DeepSeek's 90.9% at max effort, AIME 2025 97.5%, IFEval 88.7% against the official API's 89.8%, LiveCodeBench v6 88.0%, MMLU-Pro 88.0%. 0 of 100 harmful prompts refused, 97 of 100 complied per HarmBench's classifier, 0 of 250 XSTest safe prompts fully refused with 61 partial" /></p>

| Suite | Setting | Score |
| --- | --- | ---: |
| GPQA-Diamond ([EvalScope](https://github.com/modelscope/evalscope)) | 198 questions × 8 repeats, thinking on (effort 50), temperature 1, top_k 20, 32K cap | **90.5%** (1,434 of 1,584) |
| AIME 2025 | 30 problems × 4 repeats, thinking on (effort 50), temperature 1, 32K cap, rule-checked answers | **97.5%** (117 of 120) |
| IFEval | 541 prompts, greedy, thinking off; prompt-level strict (official API, same settings: 89.8%) | **88.7%** (480 of 541) |
| LiveCodeBench v6 | 175 problems, one pass, thinking on (effort 50), temperature 1, 32K cap | **88.0%** (154 of 175) |
| MMLU-Pro | 500-item stratified subset, thinking on (effort 50), temperature 1, 32K cap | **88.0%** (440 of 500) |

DeepSeek reports 90.9% GPQA-Diamond for DeepSeek-V4.1-Flash at maximum effort (100); these runs use effort 50. Quality runs used full-vocabulary sampling (top_k 0) except GPQA-Diamond (top_k 20, the serving default). GPQA ran mostly on engine `7d5b4d8` (greedy output identical) and AIME 2025 on `17800d4`; the rest on `14803a3`.

| Refusal test, greedy | Answered | Partial | Refused |
| --- | ---: | ---: | ---: |
| HarmBench harmful, chat (100) | 98 | 2 | 0 |
| HarmBench harmful, thinking (50) | 18 | 31 | 1 |
| XSTest safe, chat (50) | 49 | 1 | 0 |
| XSTest safe, thinking (25) | 18 | 6 | 1 |

Chat runs use thinking off and 768 output tokens; 50 of the 100 harmful chat replies reached that cap. Thinking runs use 3,072 tokens; 35 of the 50 harmful thinking
replies hit that cap, which is where most partials come from. HarmBench's own classifier
(`cais/HarmBench-Llama-2-13b-cls`) rates 97 of the 100 chat answers as complied. On all 250 XSTest safe prompts, a local
Qwen2.5-7B-Instruct judge with the published rubric found 0 full refusals and 61 partial ones (24.4%); 55 of those are
nonsense-premise prompts, where XSTest's rubric counts rejecting the premise as compliance. Data:
[quality.csv](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/benchmarks/v1.1/quality.csv).

## Run it on three DGX Sparks

The tested setup is exactly three DGX Spark GB10 systems (128 GB each), every pair cabled directly over RoCE. Each
rank file is about 93 GB, so the model needs all three. Budget about 440 GB of free disk per Spark (weights, rank file, Engram rows
and kernel kit) and another 256 GiB on the head for the spill tier. Build the engine first, then follow the
[GitHub recipe](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated#quick-start):

```bash
cp cluster.env.example cluster.env && chmod 600 cluster.env   # HF_REPO and HF_REVISION already pin this repo
./scripts/prep-sparks.sh status   # firmware, 64K kernel, memory saver; apply, reboot, repeat
./scripts/preflight.sh     # check all three Sparks; changes nothing
./scripts/prepare.sh       # download this repo on each Spark, build and verify its rank file
./scripts/start.sh         # start the three ranks and the OpenAI-compatible API
./scripts/smoke-test.sh    # one API request
```

`prepare.sh` downloads this repository to every Spark, builds that Spark's rank file from `tp3/rank<N>.keys` and
checks it against the SHA-256 values in the
[release manifest](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/release/manifest.json).
Hardware, network and memory prerequisites:
[DGX Spark prerequisites](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated#dgx-spark-prerequisites).

The folder is standard EXL3, but it has only been tested with the TensorFold `dsv41` engine on this three-node setup.

Prefer to let an agent do it? The GitHub README has a
[copy-paste setup prompt](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated#set-up-with-your-coding-agent).

### Call it

The API speaks OpenAI Chat Completions and Responses, with streaming, tool calls and per-request thinking control. It
listens on the head's loopback; call it there or through an SSH tunnel to port 18002.

```bash
set -a; . ~/ds1017/api/credentials.env; set +a   # on the head: loads SPARK_API_KEY without printing it
curl -s http://127.0.0.1:18002/v1/chat/completions \
  -H "Authorization: Bearer $SPARK_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"high"}'
```

| Setting | Default | Notes |
| --- | --- | --- |
| Sampling | temperature 1, top_p 1, top_k 20 | Request values win |
| Thinking | on, effort `high` | `"thinking": false` or `"reasoning_effort": "none"` turns it off |
| Effort levels | `minimal` 25, `low` 50, `medium` 62, `high` 75, `xhigh` 88, `max` 100 | Or any integer 1–100 |
| Context | 1,048,576 tokens per request | 8M-token KV pool shared by 16 concurrent streams |

The thinking evals above ran at effort 50 (`low`); IFEval and the chat refusal tests ran with thinking off. Full reference: [serving API](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/serving/README.md).

## Files

| Path | Contents |
| --- | --- |
| `model-000NN-of-00035.safetensors`, `model.safetensors.index.json` | Standard EXL3 weights and index |
| `config.json`, `quantization_config.json` | Model and per-tensor EXL3 settings |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `encoding/` | Tokenizer, chat template and DeepSeek's message encoder |
| `tp3/rank0.keys` … `rank2.keys` | Tensor lists for each of the three TP3 ranks |
| `runtime/` | Kernel kit and RoPE tables that `prepare.sh` stages; Engram rows come from DeepSeek's repository |
| `LICENSE` | DeepSeek's MIT license |

## How it was built

The pack starts from easthambuilds' 3.75 bpw EXL3 base and replaces selected tensors with LibertAIDAI, bertholomus and
LakyFx quantizations. Attention, shared experts and selected gate/up components use higher precision; the
least-used quarter of experts in layers 10–37 uses 3-bit components.

mrfakename's rank-2 refusal-removal edit changes attention `wo_b` in layers 0–31 and shared-expert `w2` in layers
10–39. It was applied to the official FP8 weights in FP32, then the affected tensors were re-encoded at 6 bits with
EXL3 mul1 K6 and an identity Hessian. The edit is baked in; there is no runtime switch.

| Source | Revision | Contribution |
| --- | --- | --- |
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `1bae0e1` | Original model |
| [easthambuilds](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw) | `3c39fdf9` | Base EXL3 3.75 bpw pack |
| [LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) | `04a492fc` | 6-bit attention and shared experts, 4-bit late-layer gate/up, selected 3-bit experts |
| [bertholomus](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw) | `f3657818` | 4-bit gate/up for the most-used half of experts in eight middle layers |
| [LakyFx](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold) | `8621801c` | Selected 3-bit experts and TP3 export lineage |
| [mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) | `517efab9` | Refusal-removal edit |

Tensor ranges and the bitrate method:
[weight provenance](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/release/weight-provenance.md).

## Intended use and risks

This model has had its refusal behavior largely removed; the refusal tables above show what remains. It will answer requests the official model declines, including
harmful ones, and it can be wrong, biased or unsafe. Use it for research into quantization fidelity, serving and
refusal behavior, and for applications where you add your own safeguards. You are responsible for how you deploy it
and for complying with the laws that apply to you. Keep the API behind access controls, keep credentials separate and
have people review consequential outputs; see the
[security guidance](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/SECURITY.md).

Tested scope is text on exactly three DGX Sparks. The base model's image input has not been tested with these weights.

## License and credits

MIT, following [DeepSeek's base-model license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE).
Every source above publishes under MIT. Provided as is, without warranty of any kind.

Thanks to [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017);
[easthambuilds](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw),
[LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw),
[bertholomus](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw) and
[LakyFx](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold) for the EXL3 tensors;
**[mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated)** for the edit;
[turboderp](https://github.com/turboderp-org/exllamav3) for EXL3; [Ash Hart](https://github.com/ashhart/TensorFold) and
[BertholomusAI](https://github.com/bertholomus/TensorFold) for TensorFold; [Mia's AI Lab](https://github.com/MiaAI-Lab/sparkDash)
for sparkDash; and [brandonmusic](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits) for the
official-model logits. Full list:
[CREDITS.md](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/CREDITS.md).

Built by [ChimeraWerks](https://github.com/ChimeraWerks): weight composition, TP3 engine work, serving and every measurement of this model on this card.

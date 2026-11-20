---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.8-27B
- harness-hub/Qwen3.8-27B-pi-adapted
base_model_relation: quantized
language:
- en
- ko
tags:
- gguf
- llama.cpp
- quantized
- conversational
- coding
- coding-agent
- agent
- agents
- tool-calling
- pi
- harbor
- terminal-bench
- swe-bench
- local-llm
- on-device
- pocket
- vidraft
- 27b
- q4_k_m
- q6_k
---

<p align="center">
  <a href="https://huggingface.co/FINAL-Bench/POCKET-Darwin-180B-GGUF"><img src="https://img.shields.io/badge/POCKET--Darwin--180B-Flagship%20reasoning-8b5cf6?style=for-the-badge" alt="POCKET-Darwin-180B: flagship reasoning model of the POCKET family"></a>
  <a href="https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF"><img src="https://img.shields.io/badge/POCKET--35B-On--device%20MoE-2563eb?style=for-the-badge" alt="POCKET-35B: on-device mixture-of-experts model"></a>
  <a href="https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF"><img src="https://img.shields.io/badge/POCKET--26B-Mobile%20%C2%B7%20multilingual-059669?style=for-the-badge" alt="POCKET-26B: mobile and multilingual model"></a>
</p>

![+47% coding ability at Q4. Terminal-Bench 2.1 tasks solved: original Qwen3.8-27B 39.6%, the pi model at full precision 43.8%, POCKET-27B-Pi Q4_K_M 58.3%, on the same 48 tasks, same GPU and same one-hour limit](https://huggingface.co/datasets/VIDraft/press-assets/resolve/main/press/compare3_pocket27b_pi_2026-11-20.png)

# POCKET-27B-Pi-GGUF

**The coding-specialist POCKET model.** A 27B coding agent that reads the repository, runs the commands, checks the tests and finishes the job, packed into a single GGUF file that runs on one workstation GPU. Built for the [pi](https://github.com/earendil-works/pi) coding agent.

> [!TIP]
> **+47% coding ability over the original Qwen3.8-27B, at Q4.** Coding ability on Terminal-Bench 2.1, same GPU, same 48 tasks, same one-hour limit: the original Qwen3.8-27B solved **39.6%**, the pi model at full precision (53.8 GB) solved **43.8%**, and the 16.5 GB POCKET-27B-Pi solved **58.3%**. That is 47% more tasks than the original and 33% more than the full-precision pi model.

| | Original model | Pi model | **Our model** |
|---|---|---|---|
| | Qwen3.8-27B, original weights | Qwen3.8-27B-pi-adapted, full precision | **POCKET-27B-Pi Q4_K_M** |
| File size | 22.4 GB | 53.8 GB | **16.5 GB** |
| Coding tasks solved | 39.6% | 43.8% | **58.3%** |

- **33% more tasks solved than the full-precision original, in a file 3.3 times smaller** (28 against 21 on the same 48 tasks). The small build runs faster on the same GPU, so it finishes more jobs inside the time limit.
- **About 1.5 times as many tasks solved as POCKET-27B** (28 against 19 on the same 48 tasks).
- **Fewer than half the turns per task** (median 15 against 36.5), so every job costs less time and fewer tokens.
- **Finishes inside the budget:** none of its 50 head-to-head runs reached the 100-turn cap. POCKET-27B reached it 14 times.
- **The small build keeps the full score:** Q4_K_M and Q6_K solved 39 and 39 on 62 head-to-head tasks.
- **One file, no adapter at serving time:** the coding adapter is merged into the weights, so any GGUF runtime loads it as a regular model.
- **Your code stays on your machine.** No API, no upload, no per-token bill.

**Pick your setup**

[![Results](https://img.shields.io/badge/results-Terminal--Bench%202.1-555?style=flat)](#results)
[![llama.cpp](https://img.shields.io/badge/run-llama.cpp-1f1f1f?style=flat)](#quick-start)
[![pi](https://img.shields.io/badge/agent-pi-f59e0b?style=flat)](#use-with-pi)
[![Llama.app](https://img.shields.io/badge/macOS-Llama.app-7f7f7f?style=flat)](#run-it-with-llamaapp-macos)
[![Files](https://img.shields.io/badge/download-GGUF-ff9d0b?style=flat)](#files)
[![Build](https://img.shields.io/badge/how%20it%20is-built-4b44c4?style=flat)](#how-it-is-built)

## Files

| File | Size | Use it for | SHA-256 |
|---|---|---|---|
| `POCKET-27B-Pi-Q4_K_M.gguf` | 16.5 GB | **Recommended.** A single 24 GB GPU, the fastest agent loop | `2428bb45ff065f8f98f8529d383d6bdbc9b79558b307d34ed5830c317a28831a` |
| `POCKET-27B-Pi-Q6_K.gguf` | 22.1 GB | Highest fidelity build, 32 GB GPUs and up, 57 tokens per second on an RTX PRO 6000 | `5d51e92ec352f071780bb83624bb206a34aad84bd3e2cd5197abd00f0b093cad` |

Both builds are reproducible: rebuilding from the same inputs gives the same SHA-256.

<a id="results"></a>

## Results

### POCKET-27B-Pi on Terminal-Bench 2.1 (measured on these GGUF files)

One attempt per task, thinking off, a 100-step budget and one hour per task, under the pi harness, sampling at temperature 0.8 with top-p 1.0. Every build ran under identical settings and is compared head to head on the tasks both builds completed.

| Build | File size | Tasks solved, same 48 tasks | Score |
|---|---|---|---|
| **Our model: POCKET-27B-Pi Q4_K_M** | **16.5 GB** | **28** | **58.3%** |
| **Our model: POCKET-27B-Pi Q6_K** | **22.1 GB** | **27** | **56.3%** |
| Pi model: Qwen3.8-27B-pi-adapted, full precision | 53.8 GB | 21 | 43.8% |
| Original model: Qwen3.8-27B, original weights (POCKET-27B) | 22.4 GB | 19 | 39.6% |

All four builds ran on the same GPU with eight parallel agent sessions and one hour per task, and are compared on the 48 tasks every build completed. Head to head, pair by pair:

| Comparison | Tasks | Result |
|---|---|---|
| POCKET-27B-Pi Q4_K_M against the full-precision original | 50 | **30** against 23 solved |
| POCKET-27B-Pi Q6_K against the full-precision original | 50 | **29** against 23 solved |
| POCKET-27B-Pi Q6_K against POCKET-27B | 50 | **29** against 19 solved |
| POCKET-27B-Pi Q4_K_M against POCKET-27B | 49 | **29** against 19 solved |
| POCKET-27B-Pi Q4_K_M against Q6_K | 62 | 39 against 39 solved |

| Work per task, Q6_K, same 50 tasks | POCKET-27B-Pi | POCKET-27B |
|---|---|---|
| Median agent turns | **15** | 36.5 |
| Runs that reached the 100-turn cap | **0** | 14 |

POCKET-27B-Pi solved 14 tasks that POCKET-27B did not. The quantized builds were evaluated themselves, on real agent tasks, so the numbers above describe the files you download.

### The coding adapter inside (reported by its authors, full precision)

POCKET-27B-Pi carries the [pi coding adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) merged into its weights. Its authors report, under the pi harness with one attempt per task:

| Benchmark | Qwen3.8-27B | Qwen3.8-27B, `xhigh` thinking | **With the pi adapter, thinking off** |
|---|---|---|---|
| SWE-bench Pro (hard 51) | 39.2% | 74.5% | **64.7%** |
| Terminal-Bench 2.1 | 41.0% | 54.2% | **56.6%** |

- **SWE-bench Pro (hard 51):** 39.2% to 64.7%, 15 tasks newly solved.
- **Terminal-Bench 2.1:** 39.5% to 55.8% on 86 paired tasks, 16 tasks newly solved (p = 0.0013).
- **Level with maximum-effort thinking on Terminal-Bench 2.1, without thinking,** while generating about 4.5 times fewer tokens per task.
- **Less work per solved task:** 8 turns against 18 on Terminal-Bench 2.1 and 23 against 59 on SWE-bench Pro, on the tasks both solve.

| Median per task, against the base model with thinking | With the pi adapter | Base model with thinking |
|---|---|---|
| Generated tokens, SWE-bench Pro | **9.4k** | 48.8k |
| Generated tokens, Terminal-Bench 2.1 | **3.7k** | 16.5k |
| Model calls, SWE-bench Pro | **21** | 61 |
| Model calls, Terminal-Bench 2.1 | **9** | 19 |

Our own baseline agrees with theirs: POCKET-27B, which carries the unchanged base weights, solved 38.0% in our run, next to the 39.5% they report for the base model.

<a id="quick-start"></a>

## Quick start

Serve it with [llama.cpp](https://github.com/ggml-org/llama.cpp):

```bash
llama-server -hf FINAL-Bench/POCKET-27B-Pi-GGUF:Q4_K_M \
  -ngl 999 -c 65536 --port 8080 --alias pocket-27b-pi \
  --temp 0.8 --top-p 1.0 --top-k 0 --min-p 0 \
  --chat-template-kwargs '{"enable_thinking":false}'
```

This is the configuration of our evaluation. Thinking off is what the model is tuned for: it reaches the answer through short notes between tool calls instead of a long reasoning block. Temperature 0.8 with top-p 1.0 is the sampling it was trained with.

Running several agents at once? Add `-np 8 -c 524288` to give eight sessions a 65,536-token window each.

<a id="use-with-pi"></a>

## Use with pi

Install pi (Node.js 22.19 or newer):

```sh
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
```

Add the server to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [{ "id": "pocket-27b-pi" }]
    }
  }
}
```

`"api": "openai-completions"` matters: it is the API the model was trained and evaluated with.

```sh
pi --provider local --model pocket-27b-pi --print "Find out why the tests in tests/test_parser.py fail and fix the bug."
```

| Setting | Value |
|---|---|
| API | OpenAI chat completions |
| Thinking | off |
| Context window | 65,536 |
| Max output tokens | 32,768 |
| Sampling | temperature 0.8, top-p 1.0, top-k 0, min-p 0 |

Any OpenAI-compatible coding agent can use the same endpoint.

## Run it with Llama.app (macOS)

1. Install [Llama.app](https://github.com/ggml-org/Llama-macOS) and launch it once. On first launch it creates `~/.config/llama/models.user.ini`.
2. Register the model. It is a single file, so there is no adapter line:

   ```sh
   cat >> ~/.config/llama/models.user.ini <<EOF

   [FINAL-Bench/POCKET-27B-Pi]
   hf-repo = FINAL-Bench/POCKET-27B-Pi-GGUF:Q4_K_M
   temp = 0.8
   ctx-size = 32768
   EOF
   ```

3. Restart Llama.app and choose **FINAL-Bench/POCKET-27B-Pi** from the model list. The app downloads the 16.5 GB file on first use.

## How it is built

POCKET-27B-Pi combines two Apache-2.0 releases and the POCKET packaging:

1. **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**, the base model.
2. **[harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted)**, a LoRA adapter trained with reinforcement learning to work as the model behind the pi coding agent.
3. **POCKET packaging.** The adapter is merged into the base weights (`llama-export-lora`), giving a standalone model with no LoRA at serving time, then quantized with llama.cpp to Q4_K_M and Q6_K. The merge was checked against the adapter-on-base reference before quantization, and each quantized build was then measured on real agent tasks.

How the adapter was trained, as published by its authors:

| | |
|---|---|
| Method | [AsyncGRPO](https://huggingface.co/docs/trl/main/en/async_grpo_trainer) reinforcement learning (TRL), LoRA r = 32 |
| Environment | [Harbor](https://github.com/laude-institute/harbor) tasks served by [OpenEnv](https://github.com/huggingface/OpenEnv), rollouts in sandboxes, tool calls screened by [safeagents](https://github.com/huggingface/safeagents) |
| Harness | pi |
| Tasks | 750 scale800 tasks interleaved with 300 TermGrade tasks |
| Reward | the task verifier's correctness score, plus a tool-efficiency bonus |
| Rollouts | 8 per task, temperature 0.8, thinking off |

The model learned from verifier rewards: it was paid for tasks that actually pass their tests, and for getting there with fewer tool calls.

## The POCKET family

| Model | What it is for |
|---|---|
| **POCKET-27B-Pi** (this model) | Coding agents on one GPU |
| [POCKET-Darwin-180B](https://huggingface.co/FINAL-Bench/POCKET-Darwin-180B-GGUF) | Flagship math and science reasoning on a desktop AI machine |
| [POCKET-35B](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF) | On-device mixture-of-experts model for laptops and phones |
| [POCKET-26B](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) | Mobile and multilingual chat |
| [POCKET-27B](https://huggingface.co/FINAL-Bench/POCKET-27B-GGUF) | General chat and vision at 27B |

## 한국어 요약

**POCKET-27B-Pi 는 코딩 전문 POCKET 모델입니다.** 저장소를 읽고, 명령을 실행하고, 시험을 돌려 가며 일을 끝내는 27B 코딩 에이전트를 GGUF 파일 하나에 담았습니다.

- **Q4 인데도 오리지날 Qwen3.8-27B 대비 코딩 능력 +47% 입니다.** 같은 그래픽카드, 같은 Terminal-Bench 2.1 과제 48개, 같은 1시간 제한에서 오리지날 Qwen3.8-27B 는 39.6%, 원 정밀도 pi 모델(53.8GB)은 43.8%, 16.5GB POCKET-27B-Pi 는 **58.3%** 를 풀었습니다.
- 과제 하나를 끝내는 데 쓰는 단계가 절반 이하입니다(중앙값 15단계 대 36.5단계). 100단계 한도까지 간 실행이 한 건도 없습니다.
- **16.5GB(Q4_K_M)** 가 22.1GB(Q6_K)와 같은 수의 과제를 풀었습니다(62과제에서 39 대 39). 24GB 그래픽카드 한 장이면 됩니다.
- 긴 「생각」 단계 없이 바로 일합니다. 안에 든 코딩 어댑터는 생각을 끈 채로 최대 강도 생각 모드와 같은 수준의 Terminal-Bench 2.1 점수를 내며, 생성하는 글자 수는 약 4.5분의 1 입니다(어댑터 제작자 발표).
- 어댑터를 가중치에 합쳐 파일 하나로 실행합니다. 코드가 밖으로 나가지 않고 API 요금도 없습니다.

## License

Apache-2.0. Built on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) and [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) (Apache-2.0).

Made by [VIDRAFT](https://vidraft.net).

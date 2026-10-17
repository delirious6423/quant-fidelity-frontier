---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - qwen3.8
  - trellis-quantization
  - ik_llama.cpp
  - long-context
  - 16gb-vram
  - eora
---

# Qwen3.8-27B-KTopt-GGUF — 200k context on a single 16 GB GPU

A 10.3 GB GGUF of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) built from trellis ("KT") quantization
types and a per-tensor recipe chosen by measured sensitivity, plus an optional 32 MB EoRA correction adapter.

- **Quality:** on GPQA Diamond, IFBench and LiveCodeBench v6 it is within 1 pp of Q8_0 on average (not significant)
  at 2.8× smaller size. Against Unsloth UD-IQ3_XXS of similar size it is +0.8 pp with a large token budget (not
  significant) and +2.8 pp with a 64k budget (significant), because it needs about 20–30% fewer reasoning tokens.
- **Context:** 160k to the full native 262k tokens on a 16 GB card, with speculative decoding at about 35–39 tokens/s.
  On a 12 GB card, 64k–96k tokens.
- **Engine:** this file needs a llama.cpp fork (see *Requirements*). It does **not** load in mainline llama.cpp,
  LM Studio or Ollama.

> [!IMPORTANT]
> **Requirements.** The weights use the `iq2_kt` / `iq3_kt` / `iq4_kss` / `iq4_ks` types from
> [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). The long-context profiles also use the VBR KV cache
> ("TurboQuant" tiers t2/t3/t4) and the DFlash2 drafter support of
> [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp). We ship an engine that has both:
> **[`lrozewicz/kt-llama-cpp`](https://github.com/lrozewicz/kt-llama-cpp)**, which is buun-llama-cpp merged with
> mainline llama.cpp plus a CUDA port of the KT types. The model is text-only (no vision tower).

## Files

| file | size | what it is |
|---|---|---|
| `Qwen3.8-27B-KTopt.gguf` | 10.31 GB (≈3.06 bits/weight) | the quantized model |
| `Qwen3.8-27B-KTopt-eora-output-r64.gguf` | 32.4 MB | optional EoRA adapter for `output.weight` (LoRA GGUF, load with `--lora`) |
| drafter, from [analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF](https://huggingface.co/analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF) | 705 MB | DFlash2 speculative drafter, used by the long-context profiles |

SHA-256:
```
f23aed2c97db0b5696311d21a218a9a63e1fe8aadd9c2077a49518c19703cade  Qwen3.8-27B-KTopt.gguf
3c285bbb843cae51bee4d2e405306309eed8b98b89a637ca5611a8c15d9b3575  Qwen3.8-27B-KTopt-eora-output-r64.gguf
```

## Benchmarks

All 629 questions of three benchmarks from the Qwen model card, one sample per question, thinking mode with the
card's sampling (temperature 1.0, top_p 0.95, top_k 20). Every configuration ran on the same engine build with the
same settings: plain q8_0 KV cache, no drafter. The reference is `unsloth/Qwen3.8-27B-GGUF` Q8_0.

We first ran with a 64k output-token limit. Every answer cut off at 64k was then continued from its saved reasoning,
first to 128k and then to 200k tokens. Continuing a cut-off answer is statistically the same as running with the
larger limit from the start, because the text before the old limit does not change. Three answers stuck in a
degenerate repetition loop (one KTopt, two UD-IQ3_XXS) were not continued and count as wrong, like any answer still
cut off at 200k.

Results at a 200k output limit:

| benchmark | Q8_0 (29.1 GB) | **KTopt + EoRA r64 (10.3 GB)** | Unsloth UD-IQ3_XXS (10.9 GB) | Qwen card (BF16) |
|---|---|---|---|---|
| GPQA Diamond, accuracy | 92.4 ± 3.7 | **89.4 ± 4.3** | 87.9 ± 4.5 | 89.2 |
| IFBench, prompt-level loose / strict | 81.7 / 79.0 | **85.3 / 81.7** | 83.0 / 80.3 | 79.5 |
| LiveCodeBench v6, pass@1 | 88.5 ± 5.5 | **85.5 ± 6.0** | 87.0 ± 5.8 | 90.3 |
| average of the three | 87.5 | **86.7** | 86.0 | |
| mean output tokens: GPQA / IFBench / LCB | 15.7k / 8.5k / 29.1k | **14.9k / 7.7k / 29.7k** | 20.7k / 10.3k / 36.3k | |

± is the half-width of a 95% confidence interval. Paired differences on the same questions, with a bootstrap 95% CI
(10,000 resamples, stratified by benchmark for the average):

| pair, 200k limit | GPQA | IFBench | LCB | average of the three |
|---|---|---|---|---|
| KTopt − UD-IQ3_XXS | +1.5 [−1.5, +4.5] | +2.3 [−1.7, +6.3] | −1.5 [−6.9, +3.8] | +0.8 [−1.6, +3.1] |
| KTopt − Q8_0 | −3.0 [−6.1, 0.0] | +3.7 [0.0, +7.7] | −3.1 [−9.2, +3.1] | −0.8 [−3.3, +1.8] |
| UD-IQ3_XXS − Q8_0 | −4.5 [−8.1, −1.0] | +1.3 [−2.7, +5.7] | −1.5 [−6.1, +3.1] | −1.6 [−4.0, +0.8] |

How the output limit changes the picture (average of the three benchmarks):

| output limit | Q8_0 | KTopt + r64 | UD-IQ3_XXS | KTopt − UD-IQ3_XXS |
|---|---|---|---|---|
| 64k | 82.2 | 82.1 | 79.3 | **+2.8 [+0.2, +5.4]** |
| 128k | 87.3 | 86.5 | 85.7 | +0.8 [−1.6, +3.1] |
| 200k | 87.5 | 86.7 | 86.0 | +0.8 [−1.6, +3.1] |

How to read this:
- With a large output budget the three configurations are within about 1.5 pp on average, and none of the average
  differences is significant. KTopt is 0.8 pp below Q8_0 and 0.8 pp above UD-IQ3_XXS.
- With a 64k budget KTopt is 2.8 pp ahead of UD-IQ3_XXS, and that gap is significant. UD-IQ3_XXS reasons 20–35%
  longer on the same questions, so it runs out of budget more often. KTopt's answer lengths match Q8_0. On a 16 GB
  card this means faster answers and less context used for the same result.
- Single-benchmark differences of 3–4 pp are within the noise of one sample per question. KTopt's IFBench lead over
  Q8_0 is most likely noise; we do not claim the quant beats the reference.
- With the large budget, the Q8_0 reference matches the Qwen card on all three benchmarks within the confidence
  intervals. At 64k its LiveCodeBench score was 74.8, because 22 of 131 answers were cut off. The card does not state
  its token budget or LCB date window. Our window is 2025-02-01 … 2025-04-06, 131 problems.
- The benchmark run uses a plain q8_0 cache. It measures the weights and the adapter. Long-context behaviour with the
  compressed VBR cache is measured separately below.

### KL divergence against BF16

Mean KLD of the next-token distribution against BF16, context 4096. The code corpus is 14 GitHub repositories created
after the model's release (16 chunks). The prose corpus is recent arXiv abstracts plus Polish Wikipedia (8 chunks).
Lower is better.

| model | file size | KLD code | KLD prose |
|---|---|---|---|
| Unsloth UD-IQ4_XS | 14.25 GB | 0.0138 | 0.0187 |
| Unsloth UD-Q3_K_XL | 13.15 GB | 0.0228 | 0.0287 |
| Unsloth UD-IQ3_S | 12.04 GB | 0.0330 | 0.0507 |
| **KTopt + EoRA r64** | 10.31 GB + 32 MB | **0.0419** | **0.0644** |
| **KTopt** | 10.31 GB | 0.0429 | 0.0662 |
| Unsloth UD-IQ3_XXS | 10.93 GB | 0.0487 | 0.0781 |
| ISTA-DASLab GSQ-RCO IQ3_XXS | 10.09 GB | 0.0602 | 0.0783 |
| Unsloth UD-Q2_K_XL | 9.83 GB | 0.0741 | 0.1061 |

### Long-context checks (16 GB profiles, VBR cache + DFlash2)

| check | result |
|---|---|
| HumanEval, all 164 tasks asked after a 188k-token filler (t3 cache) | 93.3% without adapter, **94.5% with EoRA r64**. UD-IQ3_XXS on the same engine (t2): 91.5% |
| Polish needle-in-a-haystack, 5 depths (10–90%) at 185k (t3) | 5/5, with and without adapter |
| same at 250k (t2, full native window) | 5/5 |
| dense sweep 90k–190k in 8k steps, 5 needles each, predecessor recipe KTsmall + t3 | 65/65, no empty answers |

The last row is relevant to the open mainline report [llama.cpp#27756](https://github.com/ggml-org/llama.cpp/issues/27756)
(silent EOS on prompts above ~130k). We did not observe it with this engine.

## Running it

**What you need**

- Linux, or Windows with WSL2.
- An NVIDIA GPU with at least 12 GB of memory; 16 GB for 160k context or more.
- NVIDIA driver 570 or newer.
- About 12 GB of free disk space.

The model uses quantization types that mainline llama.cpp, LM Studio and Ollama cannot load. Run it with the
kt-llama.cpp engine: as a ready Docker image (option A) or built from source (option B).

### Option A: Docker (recommended)

**1. Install Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).**
Check that containers can see your GPU; the command should print a table with your card:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

**2. Start the server and follow its log:**

```bash
docker run -d --name kt-llama --gpus all -p 8080:8080 -v kt-models:/models ghcr.io/lrozewicz/kt-llama-cpp:cuda && docker logs -f kt-llama
```

Ctrl+C stops following the log, not the server; `docker logs -f kt-llama` shows it again.

To get the newest version of the image, for example after an update or if you ran it before, remove the old
container and start it with `--pull always`; without it Docker reuses the image already on your machine. The
downloaded model files stay in the volume:

```bash
docker rm -f kt-llama
docker run -d --pull always --name kt-llama --gpus all -p 8080:8080 -v kt-models:/models ghcr.io/lrozewicz/kt-llama-cpp:cuda && docker logs -f kt-llama
```

**3. Wait until it is ready.** The first start downloads about 11 GB (the model and the DFlash2 drafter) into the
`kt-models` Docker volume; later starts reuse the files. During the download the log shows a line every 10 seconds,
such as `[kt] Qwen3.8-27B-KTopt.gguf: 5.83 of 10.30 GB (56%), 10.6 MB/s, about 7 min left`. The line
`[kt] free GPU memory ... -> profile 160k` shows the context size that was chosen. Until the server is up, every
request to port 8080 gets HTTP 503 with the reason: the download progress, then `Loading model` for about 30
seconds. It is ready when this command prints `{"status":"ok"}`:

```bash
curl http://127.0.0.1:8080/health
```

**4. Stop it and start it again later:**

```bash
docker stop kt-llama
docker start kt-llama
```

The context size is chosen from the GPU memory that is free when the container starts:

| your GPU | chosen automatically | to choose it yourself, add to step 2 |
|---|---|---|
| 16 GB, light desktop (under about 0.8 GB of GPU memory in use) | 200k | `-e PROFILE=200k` |
| 16 GB, browser and IDE open | 160k | `-e PROFILE=160k` |
| 12 GB | 96k, 64k or 32k, without the drafter | `-e PROFILE=96k` |
| 16 GB, full native window, light desktop | never chosen automatically | `-e PROFILE=262k` |

To change it later, remove the container with `docker rm -f kt-llama` and run step 2 again with the `-e` option; the
downloaded files stay in the volume. If port 8080 is taken, use `-p 8081:8080` and port 8081 in the URLs below.
All settings are in [the Docker guide](https://github.com/lrozewicz/kt-llama-cpp/blob/main/docs/kt/docker.md).

### Option B: build from source

**1. Build the engine.** You need the CUDA toolkit (12.8 or newer for RTX 50xx and RTX PRO Blackwell), CMake and a C++
compiler. Set `CMAKE_CUDA_ARCHITECTURES` to 86 for RTX 30xx, 89 for RTX 40xx or 120 for RTX 50xx and RTX PRO Blackwell.

```bash
git clone https://github.com/lrozewicz/kt-llama-cpp
cd kt-llama-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"
cmake --build build --config Release -j --target llama-server
```

**2. Download the model and the drafter.** The `hf` command comes with `pip install -U huggingface_hub`.

```bash
hf download wiklif/Qwen3.8-27B-KTopt-GGUF Qwen3.8-27B-KTopt.gguf --local-dir models
hf download analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF Qwen3.8-27B-DFlash2-Q2_K.gguf --local-dir models
```

**3. Start the server.** On a 16 GB GPU, 160k context (works with a browser and an IDE open):

```bash
VBR_VRAM_HEADROOM_MIB=256 build/bin/llama-server \
  -m models/Qwen3.8-27B-KTopt.gguf -c 163840 -ctk vbr --vbr-budget t3 \
  -md models/Qwen3.8-27B-DFlash2-Q2_K.gguf -cd 0 -ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 2 \
  -ctxcp 2 -np 1 -ngl 99 -ub 256 -b 1024 -fa on \
  --jinja --reasoning-format deepseek --alias qwen3.8-27b-ktopt \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

With a light desktop you can use `-c 200000` for 200k, or `-c 262144 --vbr-budget t2` for the full 262k.

On a 12 GB GPU, 96k context without the drafter:

```bash
VBR_VRAM_HEADROOM_MIB=256 build/bin/llama-server \
  -m models/Qwen3.8-27B-KTopt.gguf -c 98304 -ctk vbr --vbr-budget t2 \
  -ctxcp 2 -np 1 -ngl 99 -ub 256 -b 1024 -fa on \
  --jinja --reasoning-format deepseek --alias qwen3.8-27b-ktopt \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

**4. Check that it is ready.** Loading takes about 30 seconds; then `curl http://127.0.0.1:8080/health` prints
`{"status":"ok"}`. Flags, other profiles and troubleshooting: [Running and tuning](https://github.com/lrozewicz/kt-llama-cpp/blob/main/docs/kt/running.md).

### Test the API from Python

The server speaks the OpenAI Chat Completions API at `http://127.0.0.1:8080/v1`. The model id is `qwen3.8-27b-ktopt`, and
the sampling settings recommended by Qwen are already set on the server.

**A simple request**, with `pip install requests`:

```python
import requests

BASE_URL = "http://127.0.0.1:8080/v1"

resp = requests.post(
    f"{BASE_URL}/chat/completions",
    json={
        "model": "qwen3.8-27b-ktopt",
        "messages": [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}],
        "max_tokens": 4096,
        "chat_template_kwargs": {"reasoning_effort": "low"},  # low | medium | xhigh
    },
    timeout=600,
)
resp.raise_for_status()
message = resp.json()["choices"][0]["message"]
print("--- reasoning ---")
print(message.get("reasoning_content", ""))
print("--- answer ---")
print(message["content"])
```

**Streaming with the official OpenAI client**, with `pip install openai`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="qwen3.8-27b-ktopt",
    messages=[{"role": "user", "content": "Explain in three sentences what a KV cache is."}],
    max_tokens=4096,
    stream=True,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)

answering = False
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    thinking = getattr(delta, "reasoning_content", None)
    if thinking:
        print(thinking, end="", flush=True)      # the model's reasoning comes first
    if delta.content:
        if not answering:
            print("\n--- answer ---")
            answering = True
        print(delta.content, end="", flush=True)
print()
```

- `reasoning_effort` sets how long the model thinks: `low` (fastest), `medium`, or `xhigh` (the default, used in the
  benchmarks). Any other value returns HTTP 500.
- The thinking arrives in `reasoning_content`, separately from the answer in `content`.
- Long answers can take minutes, so keep the timeout generous.

### Measured memory and speed

Measured on an RTX 4070 Ti Super 16 GB (Linux, the desktop using about 0.6 GB). Prefill and decode were measured on
one long documentation prompt. Decode includes DFlash2 speculation, so it depends on the content.

| cache | context | adapter | prompt tested | prefill | decode | peak GPU memory | quality checks |
|---|---|---|---|---|---|---|---|
| t3 | 200k | yes | 144k / 188k | 683 t/s | 36.1 t/s | 15.56 / 15.83 GB | HumanEval@188k 94.5%, needle 5/5 |
| t3 | 200k | no | 144k / 188k | 692 t/s | 38.6 t/s | 15.28 / 15.51 GB | HumanEval@188k 93.3%, needle 5/5 |
| t3 | 160k | no | 160k | 660 t/s | 31.7 t/s | 15.38 GB ³ | measured in the Docker image |
| t4 | 160k | no | 160k | 805 t/s ¹ | 39.2 t/s ¹ | 15.64 GB ¹ | needle 5/5 |
| t2 | 262k | no | 250k | 593 t/s | 34.9 t/s | 15.49 GB | needle @250k 5/5 |
| t2, no drafter | 64k | yes | 58k | 1149 t/s | 38.5 t/s | 10.76 GB ² | needle 5/5 at 28k and 84k (32k and 96k runs) |
| t2, no drafter | 96k | yes | 86k | 1017 t/s | 36.2 t/s | 11.18 GB ² | needle @84k 5/5 |

¹ Measured on the predecessor recipe KTsmall, whose file is 34 MB smaller. Expect the same within noise.
² Memory of the server process alone, without the desktop, measured on the 16 GB card. A real 12 GB card with less memory bandwidth will decode more slowly.
³ With the desktop using about 1.1 GB of GPU memory.

- **The EoRA adapter.** Enable it with `-e LORA=1` in Docker, or download `Qwen3.8-27B-KTopt-eora-output-r64.gguf` and add
  `--lora models/Qwen3.8-27B-KTopt-eora-output-r64.gguf` when building from source. It adds about 0.3 GB of GPU memory and
  costs about 6% decode speed. It fits 200k with t3 only with roughly 60 MB to spare, when the desktop uses no more than
  about 0.6 GB. For 160k with t4 and 262k with t2, lower `-c` by about 16k when you add it. That figure is an estimate from about 21 MB per 1k tokens of context; we did
  not measure those two combinations.
- **12 GB cards.** 64k leaves about 1.2 GB for the desktop and driver. 96k needs other processes to use no more than about 0.9 GB. Skip the drafter here: it adds about 1.5 GB and was slower on our prompt (38.8 vs 42.1 t/s at 32k).
- **If you run out of memory**, lower `-c`. Browsers and IDEs with GPU acceleration each take a few hundred MB.
- **Use static tiers only** (`--vbr-budget t2|t3|t4`). The fork's dynamic mode (`--vbr-floor`) and mixed K/V tiers
  broke long-context quality in our tests, with HumanEval@188k falling to 0–26%.
- **`-ub 256`** is intentional. Larger micro-batches were slower at prefill on this model and cost ~0.65 GB more.
- **Short contexts.** At 32k or less, the plain cache without VBR (`-ctk q8_0 -ctv q8_0`) is also an option.
- **Turning thinking off.** Send `"chat_template_kwargs": {"enable_thinking": false}` instead of a `reasoning_effort`.

## How it was made

- **Types.** Quantized from BF16 with `llama-quantize` from ik_llama.cpp. The base type is `IQ3_KT`, with an
  importance matrix from a mixed calibration text: GitHub documentation, English chat prose, Polish Wikipedia and
  arXiv, 8192-token chunks. Per-tensor overrides:

  | tensors | type |
  |---|---|
  | FFN, layers 0–23 | `iq2_kt` |
  | FFN, layers 24–55 | `iq3_kt` |
  | FFN, layers 56–63 | `iq4_kss` |
  | attention and Gated DeltaNet projections | `iq3_kt` |
  | `attn_output` | `iq3_kt` |
  | `ssm_out` | `iq4_kss` |
  | `attn_k`, `attn_v` | `q5_K` |
  | `ssm_alpha`, `ssm_beta` | `q8_0` |
  | `output` | `iq4_ks` |
  | `token_embd` | `q2_K` |

- **Recipe search.** 23 single-group ablations around a baseline recipe, each scored by KLD on code and prose. The
  findings: precision of the late FFN layers and of `output` matters most; early attention/GDN and `ssm_out` look cheap
  on average KLD but blow up on hard LaTeX-heavy prose; effects are not additive. KTopt keeps only the changes that
  were neutral on the hard text.
- **EoRA adapter.** A fine-tuning-free rank-64 correction of the `output.weight` quantization error: a weighted SVD in the
  eigenspace of input activations collected from the BF16 model on 131k calibration tokens
  ([EoRA, Liu et al. 2024](https://arxiv.org/abs/2410.21271)). It captures about 50% of the weighted error energy of
  that tensor. Distilled adapters did not beat it on held-out text.

## Limitations

- Needs the engine above. CUDA only for practical speed.
- Text only. No vision.
- Memory margins on 16 GB are tight. The numbers assume a light desktop.
- Benchmarks use one sample per question. Differences of 3–4 pp on a single benchmark are within noise.
  Answers cut off at 64k were continued to 200k from their saved reasoning rather than regenerated.
- Speeds were measured on one GPU model with one prompt type.

## Credits

- [Qwen Team](https://huggingface.co/Qwen) for Qwen3.8-27B.
- [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for the trellis quantization types and the
  quantizer.
- [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) for the VBR KV cache and DFlash2 support.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
- [Unsloth](https://huggingface.co/unsloth) for the reference GGUFs.
- [analogalok](https://huggingface.co/analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF) for the DFlash2 drafter GGUF.

Licensed Apache-2.0, like the base model.

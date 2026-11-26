---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- ternary
- 2-bit
- gguf
- llama-cpp
- cuda
- metal
- on-device
- hybrid-attention
- agentic
- tool-calling
- speculative-decoding
- mtp
- prismml
- bonsai
base_model:
- Qwen/Qwen3.8-27B
---

<p align="center">
  <img src="./assets/bonsai-logo.svg" width="280" alt="Bonsai">
</p>

<p align="center">
  <a href="https://prismml.com"><b>Prism ML Website</b></a> &nbsp;|&nbsp;
  <a href="https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-5-27b-whitepaper.pdf"><b>Whitepaper</b></a> &nbsp;|&nbsp;
  <a href="https://github.com/PrismML-Eng/Bonsai-demo"><b>Demo &amp; Examples</b></a> &nbsp;|&nbsp;
  <a href="https://discord.gg/prismml"><b>Discord</b></a>
</p>

# Ternary Bonsai 2.5 27B — GGUF

A 27B ternary model built for agents, for llama.cpp (CUDA, Metal, CPU)

> **\~9×** smaller than FP16 | **FP16-level** agentic tool calling | **90%** of FP16 agentic coding | the **strongest agentic model under 10 GB**

Ternary Bonsai 2.5 27B is the agentic release of the Ternary Bonsai 27B line. It keeps the Qwen3.8-27B hybrid-attention architecture and the ternary weight format of Bonsai 2, so it is still **1.76 bits per weight** and **5.93 GB**, against 53.8 GB for the FP16 model. This release targets long-horizon, tool-driven work.

## Highlights

- **FP16-level agentic tool calling at one ninth of the size**: across five tool-calling benchmarks (τ²-Bench, τ³-Bench, BFCL v3, BFCL v4, and AppWorld), Bonsai 2.5 averages **77.2**, against **77.7** for FP16 — essentially indistinguishable from full precision
- **90% of FP16 on agentic coding, up from 63% for Bonsai 2**: on Terminal-Bench 2.1, SWE-bench Verified, and DeepSWE, DeepSWE rises from 0 to **27.0**, SWE-bench Verified from 60.8 to **75.6**, and Terminal-Bench 2.1 from 52.8 to **60.1**
- **The strongest agentic model under 10 GB**: the smallest of the low-bit GGUF builds of Qwen3.8-27B we compare against, and it leads every build under 10 GB on both tool calling and agentic coding. Against GSQ-RCO IQ3_XXS, which is 1.7× larger, it is ahead on tool calling (77.2 vs. 75.3) and close on coding (54.2 vs. 55.5)
- **General capability stays close to FP16**: averaged over five categories (16 benchmarks), Bonsai 2.5 scores **82.1** against 84.3 for FP16
- **Multi-token prediction (MTP) variants**: the MTP variants include a draft head that speeds up decode by **1.5–1.8×** on NVIDIA GPUs
- **2.3–3.0× faster decode than on Bonsai 2's release day** on six of the seven NVIDIA GPUs measured, with MTP and the kernels now in testing — reaching about **300 tokens/s** on H200 and B200
- **Runs on stock llama.cpp**: the `Q2_0` files run on upstream llama.cpp; `PTQ1_0` is under upstream review

## Resources

- **[Whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-5-27b-whitepaper.pdf)** — benchmarks, evaluation setup, and inference-speed measurements for this release
- **[Bonsai 2 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)** — the ternary format, storage accounting, and rotated weight basis, all unchanged in Bonsai 2.5
- **[Demo & examples](https://github.com/PrismML-Eng/Bonsai-demo)** — **the source of truth for running these models**: tested setup for every backend, pinned binaries, serving, MTP speculative decoding, benchmarking and integration, kept current as the runtimes move
- **Low-bit kernels**: [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (CUDA + Metal) · [MLX fork](https://github.com/PrismML-Eng/mlx) (Apple Silicon) · [mlx-swift fork](https://github.com/PrismML-Eng/mlx-swift) (iOS/macOS)
- **[Discord](https://discord.gg/prismml)** — join the community for support, discussion, and updates

## Model Overview

Changes from Bonsai 2 are marked with †.

| Item                   | Specification                                                                                    |
| :--------------------- | :----------------------------------------------------------------------------------------------- |
| Base model             | Qwen3.8-27B, a 27B hybrid-attention causal language model                                        |
| Parameters             | 24.35B language (64 blocks) + 0.47B vision tower (27 blocks) + 2.54B embedding/lm-head = 27.36B total |
| Architecture           | Hybrid attention (\~75% linear / \~25% full attention), SwiGLU MLP, RoPE, RMSNorm                   |
| Context length         | 262K tokens                                                                                      |
| Weight format          | Ternary g128 (FP16 group-wise scaling) in a blockwise Hadamard basis; 1.76 bits per weight, 5.93 GB (PTQ1_0) |
| Focus †                | Agentic and tool-use tasks                                                                       |
| Reasoning efforts      | `xhigh` (default, most accurate) or `medium`                                                     |
| Backends               | llama.cpp (CUDA, Metal, CPU, SYCL) and MLX (Python, Swift); GGUF packs PTQ1_0 and PQ2_0, plus Q2_0 for stock llama.cpp |
| Speculative decoding † | The MTP variants include a multi-token prediction (MTP) draft head                               |
| License                | Apache 2.0                                                                                       |

## Weight Representation: Ternary g128

The released weights are exactly ternary and use the Bonsai 2 format, unchanged. Each weight takes a value from {−1, 0, +1}, with one shared FP16 scale factor for every group of 128 weights; a ternary value carries log₂3 ≈ 1.585 bits of information.

The weights are stored in a **rotated basis**: each matrix is transformed blockwise by an orthogonal Hadamard rotation before the ternary assignment, and the runtime applies the matching transform to activations. The rotation is folded into the stored weights offline, so it costs no extra bits and no extra weight traffic.

The ternary format, storage accounting, and rotated weight basis are described in full in the [Bonsai 2 27B report](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf).

### Packings

| File       | Runtime                                   | Notes                                                                 |
| :--------- | :---------------------------------------- | :-------------------------------------------------------------------- |
| **PTQ1_0** | PrismML llama.cpp fork (upstream review)  | Dense trits; 1.76 bits per weight, 5.93 GB. Faster decode on every NVIDIA GPU measured; also runs on Apple Metal in the fork |
| **PQ2_0**  | PrismML llama.cpp fork                    | Each trit in a 2-bit slot; the format we benchmarked on Apple Metal, and the source for the MLX export |
| **Q2_0**   | Stock llama.cpp                           | llama.cpp's standard 2-bit type                                       |

All three are the same ternary model. The three base GGUF files do not include a draft head; the MTP variants include one for speculative decoding. The vision tower ships separately as an optional `mmproj` file, loaded only for image input, so text-only serving never pays for it.

## Best Practices

### Generation Parameters

We recommend using the following sets of sampling parameters for generation:
> - Thinking mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
> - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`

llama.cpp applies `min_p=0.05` by default, so pass `--min-p 0` explicitly (as in the commands below). The benchmark results below were measured with the sampling recommended by the Qwen3.8 model card (temperature 1.0, top-p 0.95, top-k 20, no presence or repetition penalty).

**The model uses `xhigh` reasoning effort by default (most accurate); use `medium` for shorter responses and a balance of speed and accuracy. `low` reasoning effort is not supported and when selected the model will behave close to `xhigh`.**

### System Prompt

You can use a simple system prompt such as:

```
You are a helpful assistant
```

### Choosing a Packing

**PTQ1_0** is the pick on NVIDIA GPUs: it is the smallest file, and the current kernels make it the faster decode on every NVIDIA GPU measured. Prefill is mixed: PQ2_0 is ahead on H200, H100, B200, and L40S, and PTQ1_0 is slightly ahead on the rest. **PQ2_0** is the format we benchmarked on Apple Metal, and all 27B Apple numbers in this card are for PQ2_0; PTQ1_0 also runs on Metal in the PrismML fork. **Q2_0** is for stock llama.cpp. See the speed tables below.

## Quickstart

> **For the most up-to-date instructions, see [PrismML-Eng/Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo).**
> It has tested setups for every backend, pinned binary releases, and run scripts for serving, tool
> calling, and MTP speculative decoding, and it is updated as the kernels evolve. If anything below
> differs from it, follow Bonsai-demo.

### Which llama.cpp runs which file

`PTQ1_0` and `PQ2_0` need the ternary hybrid-attention kernels in the
[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) fork. Stock llama.cpp rejects them
as unknown types. Use a binary from the fork.

`Q2_0` runs on stock llama.cpp from release
[b11556](https://github.com/ggml-org/llama.cpp/releases/tag/b11556) onward, which added the Hadamard
activation runtime these weights need. **Older stock builds load `Q2_0` without any warning and
produce garbage**, so check your build first (`llama-cli --version`). Apps that bundle their own
llama.cpp need an engine built from b11556 or newer.

```bash
# prebuilt, pick the archive for your platform
# https://github.com/PrismML-Eng/llama.cpp/releases/latest
tar -xzf llama-<tag>-bin-<platform>.tar.gz -C bin --strip-components=1

# or build it
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j    # drop -DGGML_CUDA=ON on macOS, Metal is default
```

The examples below use `PTQ1_0`, the faster decode on NVIDIA GPUs. On Apple Silicon, `PQ2_0` is the
format we benchmarked on Metal (`PTQ1_0` also runs there in the fork): swap it in for the download and the `-m` path, with the same flags.

```bash
hf download prism-ml/Ternary-Bonsai-2.5-27B-gguf Ternary-Bonsai-2.5-27B-PTQ1_0.gguf --local-dir .
```

```bash
./bin/llama-cli -m Ternary-Bonsai-2.5-27B-PTQ1_0.gguf \
    -ngl 99 -fa on -c 32768 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
    -p "Explain quantum computing in simple terms." -n 16384
```

With stock llama.cpp (b11556 or newer), use the `Q2_0` file and the same flags:

```bash
hf download prism-ml/Ternary-Bonsai-2.5-27B-gguf Ternary-Bonsai-2.5-27B-Q2_0.gguf --local-dir .
llama-cli -m Ternary-Bonsai-2.5-27B-Q2_0.gguf \
    -ngl 99 -fa on -c 32768 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
    -p "Explain quantum computing in simple terms." -n 16384
```

The binary is `./bin/llama-cli` from an extracted release archive, or `./build/bin/llama-cli` if you
built the fork yourself. `-ngl 99` offloads every layer, `0` is CPU-only; `-c` sets the context, up
to 262144. `-n 16384` leaves room for the thinking trace: the template defaults to `xhigh` reasoning
effort, so a small cap such as 256 tokens ends generation mid-thought, before any answer.
`--min-p 0` overrides llama.cpp's default of 0.05.

This is a reasoning model and it thinks by default. For the server, tool calling, reasoning budgets,
image input with the `mmproj` file, and **speculative decoding with the MTP variants**, follow
[Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo), which ships run scripts that pick the
right flags for your hardware.

For one request at a time on a GPU with limited VRAM, add `-np 1` to your
`llama-server` command. See [Known issues](#high-vram-use-with-automatic-server-parallelism).

## Known issues

### High VRAM use with automatic server parallelism

If you use the Prism llama.cpp server for one request at a time on a GPU with limited VRAM,
add `-np 1` to your existing `llama-server` command. This limits the server to one
request slot. In automatic mode, the server can start with four slots that share a
unified KV cache.

A community report on an RTX 3080 10 GB measured about 9.5 GB of total GPU memory
use before adding `-np 1` and about 7.8 GB afterward, including desktop usage.
Memory use depends on your build, context size, and other settings.
[See the Hugging Face discussion](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF/discussions/1).

## Benchmarks

All models are evaluated the same way, in **thinking mode at reasoning effort `xhigh`**, against conventional low-bit GGUF builds of the same base model (GSQ-RCO and Unsloth Dynamic). Sizes are the published GGUF files; see the [whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-5-27b-whitepaper.pdf) for the full evaluation setup.

### Agentic Tool Calling

| Benchmark        | Qwen3.8 FP16 | GSQ-RCO IQ3_XXS | GSQ-RCO IQ2_S | GSQ-RCO IQ2_XS | Unsloth UD-IQ2_XS | Bonsai 2 27B | **Bonsai 2.5 27B** |
| :--------------- | -----------: | --------------: | ------------: | -------------: | ----------------: | -----------: | -----------------: |
| Size (GB)        | 53.8         | 10.1            | 9.3           | 8.4            | 7.6               | 5.93         | **5.93**           |
| Bits/weight      | 16.0         | 3.00            | 2.75          | 2.50           | 2.27              | 1.76         | **1.76**           |
| τ²-Bench         | 80.6         | 77.7            | 78.8          | 79.5           | 82.0              | 80.2         | **82.0**           |
| τ³-Bench         | 72.8         | 70.7            | 69.6          | 67.5           | 67.2              | 69.1         | **73.9**           |
| BFCL v3 (native) | 71.2         | 71.6            | 71.8          | 70.2           | 71.5              | 70.3         | **72.6**           |
| BFCL v4          | 70.8         | 70.6            | 67.7          | 64.2           | 66.2              | 64.4         | **67.2**           |
| AppWorld         | 93.2         | 86.1            | 82.8          | 46.5           | 89.6              | 51.2         | **90.5**           |
| **Average (5)**  | **77.7**     | 75.3            | 74.1          | 65.6           | 75.3              | 67.0         | **77.2**           |

On average, Bonsai 2.5 is level with FP16 on tool calling (77.2 vs. 77.7) and ahead of every conventional build. It scores above FP16 on τ²-Bench and τ³-Bench, where the agent serves a simulated customer over many turns of conversation and tool calls, and on BFCL v3 with native tool calls. BFCL v4 is the exception: there it trails FP16 by 3.6 points and IQ2_S by 0.5.

AppWorld asks the agent to write code against the APIs of nine everyday apps. Bonsai 2.5 reaches 90.5, up from 51.2 for Bonsai 2 and ahead of every conventional build, including IQ3_XXS at 86.1. Most of that gap comes from finishing whole scenarios: Bonsai 2.5 completes every task in a scenario 89.3% of the time, against 80.4% for IQ3_XXS and 75.0% for IQ2_S.

Low-bit builds can also fail in ways their other scores do not predict. GSQ-RCO IQ2_XS, 1.4× the size of Bonsai 2.5, scores 79.5 on τ²-Bench but only 46.5 on AppWorld. In about one task in five, it writes a long script that guesses every API call and marks the task complete before seeing any output.

### Agentic Coding

| Benchmark          | Qwen3.8 FP16 | GSQ-RCO IQ3_XXS | GSQ-RCO IQ2_S | GSQ-RCO IQ2_XS | Unsloth UD-IQ2_XS | Bonsai 2 27B | **Bonsai 2.5 27B** |
| :----------------- | -----------: | --------------: | ------------: | -------------: | ----------------: | -----------: | -----------------: |
| Size (GB)          | 53.8         | 10.1            | 9.3           | 8.4            | 7.6               | 5.93         | **5.93**           |
| Bits/weight        | 16.0         | 3.00            | 2.75          | 2.50           | 2.27              | 1.76         | **1.76**           |
| Terminal-Bench 2.1 | 70.8         | 66.3            | 59.6          | 53.4           | 53.9              | 52.8         | **60.1**           |
| SWE-bench Verified | 80.6         | 73.2            | 75.4          | 74.4           | 71.8              | 60.8         | **75.6**           |
| DeepSWE            | 30.1         | 27.0            | 22.6          | 17.7           | 17.7              | 0.0          | **27.0**           |
| **Average (3)**    | **60.5**     | 55.5            | 52.5          | 48.5           | 47.8              | 37.9         | **54.2**           |

Compression has usually hurt agentic coding the most, because one bad action can derail a long episode. Bonsai 2 kept 63% of the FP16 score across the three coding benchmarks and did not solve a single DeepSWE task. Bonsai 2.5 keeps 90%: DeepSWE goes from 0 to 27.0 (90% of FP16's 30.1), SWE-bench Verified from 60.8 to 75.6 (94%), and Terminal-Bench 2.1 from 52.8 to 60.1.

Against the conventional builds, Bonsai 2.5 beats all three IQ2 builds on Terminal-Bench 2.1 and DeepSWE, and it scores highest on SWE-bench Verified, with IQ2_S 0.2 points behind. Only IQ3_XXS, which is 1.7× larger, has a higher coding average, on the strength of its Terminal-Bench 2.1 score.

### General Capability

The agentic gains should not come at the cost of everything else, so we also measure general capability: category averages over 16 benchmarks, in thinking mode.

| Category              | Benchmarks                                                       | Qwen3.8 FP16 | GSQ-RCO IQ3_XXS | GSQ-RCO IQ2_S | GSQ-RCO IQ2_XS | Unsloth UD-IQ2_XS | Bonsai 2 27B | **Bonsai 2.5 27B** |
| :-------------------- | :--------------------------------------------------------------- | -----------: | --------------: | ------------: | -------------: | ----------------: | -----------: | -----------------: |
| Size                  |                                                                  | 53.8 GB      | 10.1 GB         | 9.3 GB        | 8.4 GB         | 7.6 GB            | 5.93 GB      | **5.93 GB**        |
| Knowledge & reasoning | MMLU-Redux, GPQA Diamond, AA-LCR                                 | 86.67        | 85.57           | 83.63         | 82.40          | 80.97             | 83.80        | **84.27**          |
| Math                  | AIME26, HMMT26                                                   | 93.15        | 88.15           | 86.00         | 84.80          | 80.75             | 89.00        | **90.10**          |
| Coding                | LiveCodeBench v6, BigCodeBench                                   | 74.55        | 68.20           | 67.80         | 63.80          | 64.70             | 70.50        | **71.65**          |
| Instruction following | IFBench, IFEval                                                  | 81.15        | 81.95           | 81.40         | 79.25          | 78.20             | 82.55        | **80.90**          |
| Vision                | CharXiv, A-OKVQA, OmniDocBench v1.6, RealWorldQA, HallusionBench, MMMU-Pro, MathVision | 86.07        | 84.73           | 84.07         | 83.16          | 82.64             | 81.89        | **83.66**          |
| **Average (5 categories)** |                                                             | **84.32**    | 81.72           | 80.58         | 78.68          | 77.45             | 81.55        | **82.12**          |

Averaged over the five categories, Bonsai 2.5 scores 82.1 against 84.3 for FP16, the highest of the compressed builds, ahead of IQ3_XXS (81.7) and Bonsai 2 (81.6). Its largest gap to FP16 is 3.1 points, on math; every other compressed build falls at least four points behind FP16 in some category. Bonsai 2.5 leads every conventional build on math and coding, IQ3_XXS included. Instruction following and vision are the closest categories: there IQ3_XXS and IQ2_S are up to 1.1 points ahead, and Bonsai 2 leads on instruction following by 1.7.

<details>
<summary>Expand full per-benchmark results (general capability, thinking mode)</summary>

| Category              | Benchmark         | Qwen3.8 FP16 | GSQ-RCO IQ3_XXS | GSQ-RCO IQ2_S | GSQ-RCO IQ2_XS | Unsloth UD-IQ2_XS | Bonsai 2 27B | **Bonsai 2.5 27B** |
| :-------------------- | :---------------- | -----------: | --------------: | ------------: | -------------: | ----------------: | -----------: | -----------------: |
| Knowledge & reasoning | MMLU-Redux        | 91.5         | 90.7            | 89.2          | 88.1           | 88.6              | 88.6         | 89.1               |
|                       | GPQA Diamond      | 90.5         | 88.0            | 84.7          | 83.1           | 80.3              | 85.8         | 87.7               |
|                       | AA-LCR            | 78.0         | 78.0            | 77.0          | 76.0           | 74.0              | 77.0         | 76.0               |
| Math                  | AIME26            | 94.6         | 93.3            | 91.7          | 90.8           | 89.2              | 95.8         | 94.2               |
|                       | HMMT26            | 91.7         | 83.0            | 80.3          | 78.8           | 72.3              | 82.2         | 86.0               |
| Coding                | LiveCodeBench v6  | 87.6         | 78.6            | 77.1          | 73.7           | 74.2              | 83.8         | 83.0               |
|                       | BigCodeBench      | 61.5         | 57.8            | 58.5          | 53.9           | 55.2              | 57.2         | 60.3               |
| Instruction following | IFBench           | 71.0         | 71.3            | 71.7          | 68.3           | 65.3              | 74.0         | 71.0               |
|                       | IFEval            | 91.3         | 92.6            | 91.1          | 90.2           | 91.1              | 91.1         | 90.8               |
| Vision                | CharXiv           | 82.0         | 82.0            | 81.7          | 80.7           | 81.4              | 80.0         | 81.6               |
|                       | A-OKVQA           | 89.3         | 88.7            | 89.0          | 87.4           | 86.7              | 86.8         | 88.6               |
|                       | OmniDocBench v1.6 | 92.5         | 92.2            | 91.5          | 90.9           | 90.8              | 89.1         | 90.1               |
|                       | RealWorldQA       | 83.4         | 82.5            | 81.8          | 80.9           | 79.0              | 80.1         | 81.7               |
|                       | HallusionBench    | 84.8         | 83.1            | 82.9          | 83.1           | 81.9              | 81.1         | 81.5               |
|                       | MMMU-Pro          | 81.7         | 79.3            | 77.6          | 76.8           | 77.4              | 75.0         | 77.6               |
|                       | MathVision        | 88.8         | 85.3            | 84.0          | 82.3           | 81.3              | 81.1         | 84.5               |

</details>

## Inference Speed

### Plain Decoding

Plain decoding (no MTP or speculative decoding) for the PQ2_0 and PTQ1_0 files on the current PrismML llama.cpp runtime, in tokens/s, with the speedup over the September 2026 Bonsai 2 release build on the same files in parentheses. Decode is `llama-bench` `tg128` and prefill is `pp2048`, batch 1, one GPU, flash attention on, five repetitions in two order-swapped runs (run-to-run spread under 1%). Perplexity is the same on both builds, so the speedups do not come from changed output.

| GPU           | PQ2_0 decode | PTQ1_0 decode   | PQ2_0 prefill | PTQ1_0 prefill  |
| :------------ | -----------: | --------------: | ------------: | --------------: |
| NVIDIA H200   | 184 (1.56×)  | **189 (2.12×)** | 3441 (1.19×)  | 2962 (2.35×)    |
| NVIDIA H100   | 177 (1.50×)  | **186 (2.05×)** | 3401 (1.18×)  | 2987 (2.39×)    |
| NVIDIA B200   | 149 (1.29×)  | **192 (2.11×)** | 3376 (1.02×)  | 3228 (2.23×)    |
| NVIDIA A100   | 86 (1.16×)   | **107 (1.96×)** | 1387 (1.03×)  | 1405 (1.98×)    |
| RTX PRO 6000  | 137 (1.10×)  | **158 (1.32×)** | 4223 (1.02×)  | 4283 (2.11×)    |
| RTX 5090      | 143 (1.07×)  | **163 (1.28×)** | 4163 (1.00×)  | 4198 (2.22×)    |
| RTX 4090      | 98 (1.09×)   | **112 (1.17×)** | 3444 (1.06×)  | 3452 (2.06×)    |
| NVIDIA L40S   | 82 (1.09×)   | **95 (1.14×)**  | 3057 (1.07×)  | 3018 (1.87×)    |
| Apple M5 Pro\* | 30.2 (1.07×) | –              | 416 (1.05×)   | –               |

\* Metal, PQ2_0 only; prefill column is `pp512`.

### Speculative Decoding with the MTP Head


Backend presets for the `-mtp.gguf` variants are available for [CUDA](presets/llama.cpp/mtp-cuda.json), [Metal](presets/llama.cpp/mtp-metal.json), [SYCL](presets/llama.cpp/mtp-sycl.json), and [Hexagon](presets/llama.cpp/mtp-hexagon.json). Each lists the matching file, verified build settings, and benchmark caveats. Use the released settings with released builds. Testing settings require the corresponding unreleased build. The Galaxy S26 Ultra NPU preset requires the unreleased Hexagon fix.

To use a preset, download the named `-mtp.gguf` and select the JSON profile for your device and build. Apply the profile's environment variables and run its `llama-server` command. The JSON documents the settings. Pass the arguments on the command line.

For example, to print the released CUDA datacenter command from the preset:

```bash
hf download prism-ml/Ternary-Bonsai-2.5-27B-GGUF presets/llama.cpp/mtp-cuda.json --local-dir .
jq -r '.profiles[] | select(.id == "cuda-datacenter-and-blackwell") | .released_build.command' presets/llama.cpp/mtp-cuda.json
```

Install that profile's specified build before running the command. For other backends, inspect the selected JSON for build requirements and environment variables.

The MTP variants include a **multi-token prediction (MTP) draft head**: one extra layer, built by PrismML, that guesses the next few tokens. With speculative decoding, the full model checks all the guesses in one pass and keeps only the tokens it would have produced itself, so output matches decoding without drafting up to the rounding noise of checking several tokens at once: on an H200 with the PTQ1_0 file at the recommended setting, mean KL divergence between drafting on and off is 5.2 × 10⁻⁵, no higher than in a control that accepts every guess (not yet checked on Metal, SYCL or the NPU).

The head speeds up decode by **1.5–1.8× on NVIDIA GPUs** and **1.1–2.1× on the other devices**. Because it makes its own pass over the prompt, it slows prefill by 9–23% on NVIDIA GPUs and 2–8% elsewhere; with the drafting kernels in testing, prefill with MTP beats plain prefill on every NVIDIA GPU, and the M4 Pro's decode gain rises to 1.27×.

| Device                  | Setting | Decode off | Decode on   | Decode +kernels | Prefill (2K) off | Prefill on   | Prefill +kernels |
| :---------------------- | :------ | ---------: | ----------: | --------------: | ---------------: | -----------: | ---------------: |
| NVIDIA H200             | 3, 0.1  | 194        | 296 (1.53×) | 306 (1.58×)     | 2475             | 2252 (0.91×) | 3362 (1.36×)     |
| NVIDIA H100             | 3, 0.1  | 184        | 282 (1.53×) | 291 (1.58×)     | 2746             | 2151 (0.78×) | 3196 (1.16×)     |
| NVIDIA B200             | 3, 0.1  | 191        | 296 (1.55×) | 309 (1.61×)     | 3076             | 2620 (0.85×) | 3493 (1.14×)     |
| NVIDIA A100             | 3, 0.1  | 105        | 171 (1.63×) | 175 (1.66×)     | 1342             | 1201 (0.90×) | 1435 (1.07×)     |
| RTX PRO 6000            | 3, 0.1  | 158        | 277 (1.76×) | 283 (1.80×)     | 3930             | 3358 (0.85×) | 4445 (1.13×)     |
| RTX 5090                | 3, 0.1  | 174        | 305 (1.75×) | 312 (1.79×)     | 3854             | 2998 (0.78×) | 4173 (1.08×)     |
| RTX 4090                | 3, 0    | 111        | 204 (1.83×) | 204 (1.83×)     | 2808             | 2166 (0.77×) | 3014 (1.07×)     |
| Apple M4 Pro\*          | 1, 0.6  | 22.9       | 25.3 (1.10×) | 29.2 (1.27×)   | 126              | 122 (0.98×)  | 126 (1.00×)      |
| Core Ultra laptop†      | 8, 0.6  | 11.1       | 23.1 (2.08×) | 24.5 (2.21×)   | 236              | 218 (0.92×)  | n/a              |
| Galaxy S26 Ultra‡       | 1, 0    | 6.34       | 8.79 (1.39×) | n/a            | 137              | 132 (0.97×)  | n/a              |

Tokens/s. "+kernels": MTP on plus the kernels in testing that apply to the device's file. In parentheses, speedup over MTP off on the same build (for "+kernels", MTP and kernels combined). Setting: tokens drafted per pass, minimum draft probability. Unmarked rows are PTQ1_0.
\* Metal, PQ2_0; an M4 Pro, not the M5 Pro above; 18 of the 36 prompts.
† Integrated GPU, SYCL.
‡ Q2_0, whole model on the Hexagon NPU, with a fix not yet released; 10 of the prompts.
n/a: no kernel in testing applies.

Drafting pays off only when the head guesses right, which depends on the text, so this table cannot use `llama-bench`'s synthetic tokens. It instead times `llama-server`, as a user would run it, on 36 real prompts (chat, agents and tool calls, documents, structured output, code, reasoning and long documents) on separate machines; prefill is the server's rate on a 2,048-token prompt. Each ratio compares two runs on one machine and build, so read this table on its own, not against the plain-decoding table above.

### Since the Bonsai 2 Release

How much faster a user is today than on Bonsai 2's release day (September 17, 2026), in tokens/s. For a strong baseline, each Bonsai 2 cell is the fastest setup a user could run in its release week on the release build: PTQ1_0 or PQ2_0 without drafting, or PTQ1_0 with the community head published three days later ("+MTP"). Bonsai 2.5 is the "+kernels" column of the MTP table; in parentheses, speedup over Bonsai 2.

| Device              | Bonsai 2 setup | Bonsai 2 decode | Bonsai 2 prefill (2K) | Bonsai 2.5 decode | Bonsai 2.5 prefill (2K) |
| :------------------ | :------------- | --------------: | --------------------: | ----------------: | ----------------------: |
| NVIDIA H200         | PQ2_0          | 114             | 2618                  | **306 (2.69×)**   | 3362 (1.28×)            |
| NVIDIA H100         | PQ2_0          | 112             | 2648                  | **291 (2.60×)**   | 3196 (1.21×)            |
| NVIDIA B200         | PQ2_0          | 104             | 2778                  | **309 (2.98×)**   | 3493 (1.26×)            |
| NVIDIA A100         | PQ2_0          | 73              | 1281                  | **175 (2.39×)**   | 1435 (1.12×)            |
| RTX PRO 6000        | PQ2_0          | 120             | 3226                  | **283 (2.36×)**   | 4445 (1.38×)            |
| RTX 5090            | PQ2_0          | 135             | 3645                  | **312 (2.31×)**   | 4173 (1.14×)            |
| RTX 4090            | PTQ1_0+MTP     | 106             | 1446                  | **204 (1.92×)**§  | 3014 (2.08×)            |
| Apple M4 Pro\*      | PQ2_0          | 20.3            | 125                   | **29.2 (1.44×)**  | 126 (1.01×)             |
| Core Ultra laptop†  | PTQ1_0+MTP¶    | 11.5            | 85                    | **22.9 (2.00×)**  | 218 (2.56×)             |
| Galaxy S26 Ultra‡   | not available at release | –     | –                     | 8.79              | 132                     |

Same prompts and timers on both sides; PQ2_0 rows ran on different hosts. Blackwell GPUs run the release source built with CUDA 12.8 (the September binaries had no Blackwell code). L40S not measured.
\* Metal; a community PQ2_0 head needed a separate runtime without Metal support.
† Integrated GPU, SYCL.
‡ Hexagon NPU.
§ Conservative: different hosts, and a faster RTX 4090 reached 219 tokens/s on the same build.
¶ Nothing ran Bonsai 2 27B here on September 17: first SYCL release (October 9, 2026), community head at two drafts (prefill at one); both sides on the same 18 prompts, hence 22.9 here vs. 24.5 over all 36 in the MTP table.

Decode with MTP is now **2.3–3.0× faster on six of the seven NVIDIA GPUs**, 1.9× on the RTX 4090, 1.44× on the M4 Pro and 2.0× on the Core Ultra laptop. Drafting has turned around: on release day it lost to the fastest plain setup everywhere except the RTX 4090 (on the M4 Pro it ran at 0.38× of plain PTQ1_0 decoding), and today it pays on every device. On the H100 the gain splits about evenly: 1.65× from the newer runtime, model and file without drafting, and a further 1.58× from MTP with PrismML's kernels. Prefill gains are smaller, 1.1–1.4×, because PQ2_0 already read prompts quickly in September; they reach 2.1× on the RTX 4090 and 2.6× on the laptop, whose fastest earlier setups drafted, and are flat on the M4 Pro.

### Upstream llama.cpp

`Q2_0` files of this model run on stock llama.cpp: runtime support for the Bonsai 27B family was merged as [ggml-org/llama.cpp#29600](https://github.com/ggml-org/llama.cpp/pull/29600), and the blockwise Hadamard (FWHT) kernels it needs are upstream for CPU (#27779), CUDA (#29096, #29100), Metal (#29094, #29095, #29602), Vulkan (#29772), SYCL (#29243, #29605) and MUSA (#30167). On an H200, stock llama.cpp matches the PrismML runtime on the Q2_0 file (mean KL divergence 3.6 × 10⁻⁴, same top token at 99.3% of positions, identical greedy text) and decodes at 109 tokens/s against 150 on the PrismML runtime. `PTQ1_0` is under upstream review (#29672 for the format and CPU, #29676 for CUDA). `PQ2_0` and the kernels above currently ship in the PrismML fork.

### MLX

A text-only MLX build of this model, exported from the PQ2_0 file ([Ternary-Bonsai-2.5-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit); affine 2-bit, group 128, with a bundled loader that applies the Hadamard transforms) reproduces llama.cpp closely: mean KL divergence 4 × 10⁻⁶ to 3 × 10⁻⁵ per chunk and the same top token at all 1,020 positions checked. On an Apple M5 Pro it prefills at 348 tokens/s (`pp512`) and decodes at 21.7 tokens/s, against 416 and 30.2 for llama.cpp on the same machine.

## Use Cases

- **Local coding and terminal agents**: SWE-bench Verified at 75.6 and Terminal-Bench 2.1 at 60.1 from a 5.93 GB file, with the 262K context available for full-repository work
- **Tool-calling and customer-facing agents**: FP16-level multi-turn tool use (τ²-Bench 82.0, τ³-Bench 73.9) and app-API automation (AppWorld 90.5) without a datacenter-sized model
- **Fast single-GPU agent serving**: with the MTP head and the kernels in testing, about 300 tokens/s decode on H200 and B200 and over 280 tokens/s on an RTX PRO 6000 or RTX 5090
- **On-device and offline agents**: runs on laptops (Apple Silicon, Core Ultra), keeping prompts and data on the device; a phone path (Galaxy S26 Ultra, whole model on the Hexagon NPU) has been measured with a fix not yet released

## Citation

If you use Ternary Bonsai 2.5 27B, please cite:

```bibtex
@techreport{bonsai2_5_27b,
    title   = {Ternary Bonsai 2.5 27B: A 27B Ternary Model Built for Agents},
    author  = {PrismML},
    year    = {2026},
    month   = {November},
    url     = {https://prismml.com}
}
```

## License

Released under [Apache 2.0](LICENSE). See [NOTICE.txt](NOTICE.txt) for upstream attribution and the provenance of the draft head included in the MTP variants.

## Contact

For questions, feedback, or collaboration inquiries: **contact@prismml.com**

---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
quantized_by: turboderp
tags:
- exl3
- exllamav3
- quantized
- mixture-of-experts
---

# Qwen3.8-Flash-Next · EXL3 3.18 bpw (H6, NG5)

> **A repackaging with benchmarks, not a new quantization.** Every model file in this repository was
> quantized by [turboderp](https://huggingface.co/turboderp), the author of exllamav3, and is
> byte-identical to the file of the same name in
> [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3). What
> this repository adds is the pairing of two of his files in one download and the benchmarks below, run
> by one person on one home desktop (AMD Ryzen 9 5900X, 64 GB of RAM, an RTX 3090 and an RTX 3070 Ti),
> not on a lab cluster. It is not an official release of turboderp or of the Qwen team.

An EXL3 build of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (a 125B
mixture-of-experts model with about 6B active parameters per token, a 51B n-gram embedding table, a vision
encoder and a multi-token-prediction head), made of:

- the weights of turboderp's **self-calibrated 3.18 bpw** build, branch
  [`3.18bpw_h6_ng8`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.18bpw_h6_ng8)
  (commit `2bfc9da`);
- the **5-bit n-gram table** of his 3.05 bpw build, branch
  [`3.05bpw_h5_ng5`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5)
  (commit `69e3343`), in place of the 8-bit table the 3.18 bpw branch ships.

His card notes that `ngram_embedding.safetensors` is swappable between his builds. With the 5-bit table the
download is 19.2 GB smaller than the 3.18 bpw branch. All figures in [Benchmarks](#benchmarks) were
measured on exactly this pairing.

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Quantized by | [turboderp](https://huggingface.co/turboderp), with exllamav3 1.6.3 |
| Format | EXL3, for [exllamav3](https://github.com/turboderp-org/exllamav3) |
| Bitrate | 3.18 bpw (exllamav3's accounting), head 6 bits, n-gram table 5 bits |
| Size | 54.49 GB of weights + 32.64 GB n-gram table = 87.3 GB with the configuration files |
| Context | 262,144 tokens |
| Vision, MTP | both in the files: vision tower at 6 bits, MTP head at 3 bits |

## Files

| File | Size | From branch |
|---|---|---|
| `model-00001-of-00007.safetensors` … `model-00007-of-00007.safetensors` | 54.49 GB | `3.18bpw_h6_ng8` |
| `ngram_embedding.safetensors` | 32.64 GB | `3.05bpw_h5_ng5` |
| `config.json`, `quantization_config.json`, `model.safetensors.index.json`, tokenizer, chat template, preprocessor files, `LICENSE` | 0.15 GB | `3.18bpw_h6_ng8` |

Nothing was converted, edited or re-saved: the SHA-256 of every weight shard and of the table equals the
one Hugging Face lists for turboderp's file. The evaluation prompt files of his branches
(`qbench_prompts.*`) are not copied here.

## Quantization method

The method is turboderp's; this section describes it from the exllamav3 documentation and from what the
files themselves record. For anything beyond that, his repository and the
[exllamav3 documentation](https://github.com/turboderp-org/exllamav3) are the authority.

**EXL3.** EXL3 is exllamav3's format, a streamlined variant of
[QTIP](https://arxiv.org/abs/2406.11235): each weight matrix is rotated with sign flips and block Hadamard
transforms so that its entries look alike, then encoded by a trellis coder at a fixed number of bits per
weight, with the rounding error fed forward column by column (LDLQ) against the Hessian of the inputs the
matrix sees on calibration data. The model is converted one layer at a time, each layer calibrated on the
output of the already quantized layers before it.

**Self-calibration.** turboderp's card lists this build among his "self-calibrated" ones. In exllamav3's
[self-calibration pipeline](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md),
the calibration data is not a text corpus but a trace sampled from the model itself: its own answers,
reasoning included, to a seeded set of conversations across subjects, task formats and languages, under
its own chat template. The Hessians that steer the rounding then describe the inputs the deployed model
actually processes. The same pipeline can also measure how much each tensor's quantization error costs at
the model's output and derive a bitrate per tensor from it. The files record a calibration set of 217
rows of 2,048 tokens and the `mul1` trellis codebook.

**Bit allocation**, as stored in `quantization_config.json`:

| Part | Bits per weight |
|---|---|
| Routed experts (gate, up and down projections), layers 4–43 | 3 |
| Routed experts, layers 0–3 and 44–47 | 3.5 |
| Attention and Gated DeltaNet projections, sparse-attention indexer, shared experts | 7 |
| Output head | 6 |
| Vision tower | 6 |
| MTP head | 3 |
| Token embedding | 16 (unquantized) |
| N-gram embedding table (this repository) | 5 |

Norms, routers, gates, the gated-residual tensors and a few small projections are stored unquantized.
The n-gram table, the vision tower and the MTP head are quantized without calibration, as exllamav3
does for them.

## Benchmarks

Four benchmarks were run on this build. The base-model column is what the Qwen team publishes on the
[Qwen3.8-Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for the unquantized model; the
unquantized model does not fit the desktop named above and was not run here. Qwen's card does not state
the settings of these two rows, so the comparison is between two different harnesses.

| Benchmark | Problems | This build | Base model (Qwen's card) |
|---|---|---|---|
| GPQA-Diamond | 198 | **91.41 %** (181 correct) | 91.7 % |
| LiveCodeBench v6, official grader | 175 | **86.29 %** (151 passed) | 91.9 % |
| LiveCodeBench v6, programs run as processes | 175 | **90.86 %** (159 passed) | |
| AIME 2025 | 30 | **96.67 %** (29 correct) | not published |
| Zero-shot average (five tasks, below) | | **76.90 %** | not published |

Each figure is a single run, sampled at temperature 1.0. One question is 0.5 point on GPQA-Diamond,
0.6 point on LiveCodeBench and 3.3 points on AIME 2025, and a second run of the same build would move
by a few questions.

**The two LiveCodeBench rows.** The official grader executes each program inside its own Python process, with
standard input replaced by an in-memory text stream. A program that reads the binary stream
(`sys.stdin.buffer`) then stops with a runtime error or does not get the test's input, whatever it
computes. Of the 24 programs the official grader failed, 8, all of them reading `sys.stdin.buffer`, pass
every test when run as separate processes with the tests on their real standard input (30 seconds per
test, outputs compared exactly); the other 16 stay failed. The official figure is the one comparable
with other published scores; the second row is what the generated programs actually do. No failure was a
time limit.

### Zero-shot tasks

Multiple-choice tasks scored by log-likelihood, no generation and no thinking, lm-eval task defaults:

| Task | Questions | Accuracy | Length-normalised accuracy |
|---|---|---|---|
| ARC-Easy | 2,376 | 83.29 % | 77.90 % |
| ARC-Challenge | 1,172 | 61.43 % | 62.88 % |
| HellaSwag | 10,042 | 68.86 % | 87.31 % |
| WinoGrande | 1,267 | 73.56 % | not defined |
| PIQA | 1,838 | 81.88 % | 82.86 % |
| **Mean** | | **73.81 %** | **76.90 %** |

The second mean takes the length-normalised accuracy where lm-eval defines it and the plain accuracy for
WinoGrande.

### Settings

- **Generation benchmarks** (GPQA-Diamond, LiveCodeBench, AIME 2025): thinking on at the chat template's
  default effort, the sampling Qwen recommends for thinking mode (temperature 1.0, top_p 0.95, top_k 20,
  no penalties), at most 65,536 new tokens, one answer per problem. No answer reached the token limit;
  the answers total 6.8 million generated tokens.
- **GPQA-Diamond**: the `gpqa_diamond` set, the four choices shuffled with a fixed seed, the
  multiple-choice prompt of OpenAI's simple-evals ("Answer: $LETTER" on the last line), last letter
  given taken as the answer.
- **LiveCodeBench v6**: the 175 problems added by release v6 of `code_generation_lite`, the benchmark's
  own generic chat prompt, code extraction and grader (6 seconds per test), pass@1.
- **AIME 2025**: the 30 problems of `math-ai/aime25`, the model asked to reason step by step and put its
  final answer in `\boxed{}`, compared as an integer.
- **Zero-shot**: [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.13, the model's
  log-probabilities read from exllamav3 directly (same engine build and expert split), without a chat
  template.
- **Engine**: exllamav3 at commit `627c8710` (twelve commits after 1.6.7) behind
  [TabbyAPI](https://github.com/theroyallab/tabbyAPI), with the configuration shown in [Usage](#usage): both
  GPUs, 300 of the 512 experts of each layer computed on the CPU, the n-gram table streamed from disk, a
  262,144-token cache quantized to 4 bits, MTP drafting on, two requests at a time. The limit on how many
  experts the engine moves between CPU and GPU per step was raised locally, which changes speed, not what
  is computed. Image input was not tested.

## Usage

The files load in exllamav3 and in servers built on it, such as TabbyAPI; the build was served here with
exllamav3 1.6.7 and later. The whole model does not need to fit in VRAM: exllamav3 can compute part of
the experts of each layer on the CPU and reads the n-gram table from disk. The model section of the
TabbyAPI configuration used for the benchmarks, on a 24 GB and an 8 GB GPU with 64 GB of RAM:

```yaml
model:
  model_name: Qwen3.8-Flash-Next-3.18bpw-h6-ng5-exl3
  max_seq_len: 262144
  cache_size: 262144
  cache_mode: "4,4"
  gpu_split: [22.5, 7]
  cpu_moe_split_experts: 300
  cpu_moe_threads: 11
  cpu_moe_pinned_arena: true
  reasoning: true
  reasoning_start_token: "<think>"
  reasoning_end_token: "</think>"
draft_model:
  draft_mode: mtp
  dynamic_draft: true
```

`cpu_moe_split_experts` is the number of experts per layer kept on the CPU: lower it on GPUs with more
room. Qwen recommends temperature 1.0, top_p 0.95 and top_k 20 in thinking mode, and temperature 0.7,
top_p 0.8, top_k 20 with presence_penalty 1.5 without thinking. Qwen3.8-Flash-Next thinks by default:
its answers open with a `<think>` block.

## Credits

- **[turboderp](https://huggingface.co/turboderp)** for the quantization itself: every model file here
  is his, as are [exllamav3](https://github.com/turboderp-org/exllamav3), the EXL3 format and the
  self-calibration pipeline. His repository offers this model from 1.66 to 6.05 bpw.
- **[Qwen team](https://huggingface.co/Qwen)** for Qwen3.8-Flash-Next, and for the base-model figures
  quoted above.
- **[TabbyAPI](https://github.com/theroyallab/tabbyAPI)**, the server used for the generation benchmarks.
- The authors of **[QTIP](https://github.com/Cornell-RelaxML/qtip)**, the method EXL3 builds on.
- The authors of **GPQA**, **LiveCodeBench**, the **AIME 2025** set, **ARC**, **HellaSwag**,
  **WinoGrande**, **PIQA** and **lm-eval**.

## Citation

```bibtex
@techreport{qwen2026design,
    title       = {On the Design of {Qwen3.8-Next} Architecture: Evaluation, Efficiency, and Training Stability},
    author      = {{Qwen Team}},
    institution = {Alibaba Group},
    month       = {August},
    year        = {2026}
}
@misc{qwen3.8flashnext,
    title  = {{Qwen3.8-Flash-Next}: A New Architecture, Towards Ultimate Cost-Efficiency},
    author = {{Qwen Team}},
    month  = {August},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.8-flash-next}
}
@inproceedings{tseng2024qtip,
  title     = {{QTIP}: Quantization with Trellises and Incoherence Processing},
  author    = {Tseng, Albert and Sun, Qingyao and Hou, David and De Sa, Christopher},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

## License

[Qwen Community License 1.0](LICENSE), as the base model and turboderp's release.

## Support

The quantization is turboderp's work. If the benchmarks here are useful to you, you can support my
work on [Ko-fi](https://ko-fi.com/papykahan).

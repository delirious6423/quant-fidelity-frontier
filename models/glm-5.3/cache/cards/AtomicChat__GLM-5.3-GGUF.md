---
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
base_model:
- zai-org/GLM-5.3
base_model_relation: quantized
quantized_by: AtomicChat
language:
- en
- zh
pipeline_tag: text-generation
library_name: gguf
tags:
- atomic-chat
- glm
- glm-5
- zai-org
- moe
- gguf
- imatrix
- quantized
- llama.cpp
---

# How to Run GLM-5.3 Locally
<p style="margin-top: 0; margin-bottom: 0;">
  <em>Built from Z.ai's original weights with our own importance matrix. The <a href="https://huggingface.co/datasets/AtomicChat/calib-corpora">calibration corpora</a> behind our builds are public.</em>
</p>
<div style="display: flex; gap: 8px; align-items: center; margin-top: 10px; margin-bottom: 10px;">
  <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_glm_5_3&utm_content=btn_atomic"><img src="https://huggingface.co/AtomicChat/GLM-5.3-GGUF/resolve/main/btn_atomic.png" width="162" alt="Atomic Chat"></a>
  <a href="https://discord.gg/8wGSsvmg4V"><img src="https://huggingface.co/AtomicChat/GLM-5.3-GGUF/resolve/main/btn_discord.png" width="119" alt="Discord"></a>
  <a href="https://github.com/AtomicBot-ai/Atomic-Chat"><img src="https://huggingface.co/AtomicChat/GLM-5.3-GGUF/resolve/main/btn_github.png" width="115" alt="GitHub"></a>
</div>
<ul style="margin: 0 0 12px 0;">
  <li>GLM-5.3 keeps the GLM-5.2 base and takes every gain from post-training. Per Z.ai it is the most capable open-weights model for coding, with open-source SOTA on Terminal Bench 3.0 and Agents' Last Exam.</li>
  <li>You can now run GLM-5.3 in <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_glm_5_3&utm_content=bullet_app">Atomic Chat</a> with toggles for Low, High and Max thinking.</li>
  <li>The quants are still uploading and need a llama.cpp build with GLM-5.3 support; Atomic Chat runs it as support ships.</li>
</ul>
<hr style="margin: 0 0 16px 0;">

<img src="https://huggingface.co/AtomicChat/GLM-5.3-GGUF/resolve/main/hero.png" alt="Z.ai" style="width:150px; max-width:100%; height:auto;"/>

## Highlights

- **753B parameters**, read from the published weight index. Mixture-of-Experts: 256 routed experts with 8 active per token plus 1 shared, 78 layers, the first 3 dense.
- **Same base as GLM-5.2.** Z.ai state every gain comes from post-training, and the two checkpoints carry an identical parameter count.
- **Coding**: per Z.ai a 50% improvement over GLM-5.2 on their in-house Z.ai Code Bench, and open-source SOTA on Terminal Bench 3.0 and Agents' Last Exam.
- **Emergent cyber capability**: Z.ai report state of the art on CyberGym for vulnerability discovery, with gains largest further up the exploitation chain, more than doubling GLM-5.2 on exploitation benchmarks.
- **Thinking budget** controlled by `reasoning_effort` with three levels, `low`, `high` and `max`. It defaults to `max`.
- **Sparse attention** with a learned indexer (`glm_moe_dsa`, top-2048 index), and one multi-token-prediction block in the checkpoint.
- **Bilingual**, English and Chinese.
- **Ships in FP8** upstream, so our GGUF base is converted from the FP8 checkpoint rather than from BF16.
- **Full imatrix quantization** with our public [calibration corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora).

> [!NOTE]
> These GGUFs are **self-quantized from the original weights**, not a repack. The importance matrix keeps low-bit quants closer to the full-precision model.

> [!IMPORTANT]
> This is a 753B model. Even at four bits a full set of weights runs to several
> hundred gigabytes, and all of it has to fit in fast memory. In practice that
> means a large-RAM server or a serious multi-GPU rig, not a laptop and not most
> workstations.

> [!IMPORTANT]
> Always pass `--jinja` so the **GLM-5.3 chat template** is applied. Z.ai note that
> `clear_thinking` defaults to `false` in that template, so pass `clear_thinking=true`
> for chat scenarios.

## Model Overview

| Property | Value |
|---|---|
| Base model | `zai-org/GLM-5.3` |
| Parameters | 753B total, MoE with 8 routed experts plus 1 shared active per token |
| Layers | 78, the first 3 dense, 256 routed experts in the rest |
| Architecture | `GlmMoeDsaForCausalLM`, sparse attention with a learned indexer |
| Upstream precision | FP8 (`e4m3`, 128x128 block scales) |
| Vocabulary | 154,880 |
| Context length | 1,048,576 positions in the config; Z.ai evaluate up to 1M with context management |
| Languages | English, Chinese |
| Thinking | `reasoning_effort`: `low`, `high`, `max`. Defaults to `max` |
| This repo | GGUF quants (imatrix). The importance matrix we built is published here too |

<img src="https://huggingface.co/AtomicChat/GLM-5.3-GGUF/resolve/main/benchmark.png" alt="GLM-5.3 benchmark scores" style="width:100%; max-width:900px;"/>

Scores are Z.ai's published results for the base `zai-org/GLM-5.3`. Selected numbers from
their table: Terminal Bench 2.1 88.2, Terminal Bench 3.0 28.3, DeepSWE 1.1 66.9,
CyberGym 84.5, AutomationBench 48.2, HLE w/ tools 62.5, GDPval-AA v2 1769.

## Choosing a quant

| Quant | Size | Notes |
|---|---|---|
| `AD-IQ2_M` | — | Smallest usable. Aggressive low-bit for the tightest boxes. |
| `AD-IQ3_M` | — | Beats Q3 at similar size thanks to imatrix. Best low-memory pick. |
| **`AD-Q4_K_M`** | — | **Recommended default. Best balance of size, speed and quality.** |
| `AD-Q5_K_M` | — | A step up when the memory is there. |
| `AD-Q6_K` | — | Near lossless. |
| `AD-Q8_0` | — | Effectively lossless, reference quality. |

> [!TIP]
> Sizes fill in once the quants finish uploading. Pick the largest file that fits your
> memory with room for context.

`AD-` marks an Atomic Dynamic layout: bits are assigned per tensor role rather than left
to a preset, with the router and the shared expert held high and the routed experts
carrying the compression.

## Get started

> [!NOTE]
> GLM-5.3 uses the `glm_moe_dsa` architecture. The quants in this repo are still
> uploading, and running them needs a `llama.cpp` build that has landed GLM-5.3 support.
> Until then, [Atomic Chat](https://atomic.chat) is the easiest way to run it as support ships.

Run GLM-5.3 locally with:

- **[Atomic Chat](https://atomic.chat):** the easiest path. Open the app, search `AtomicChat/GLM-5.3-GGUF`, pick a quant, hit **Use this model**.
- **llama.cpp:** `llama-server -hf AtomicChat/GLM-5.3-GGUF:AD-Q4_K_M --jinja -c 8192`
- **Ollama:** `ollama run hf.co/AtomicChat/GLM-5.3-GGUF:AD-Q4_K_M`
- **LM Studio / Jan:** search the repo id, download any quant.

For the original FP8 weights rather than GGUF, Z.ai list SGLang, vLLM, TokenSpeed,
Transformers and KTransformers on the [base model card](https://huggingface.co/zai-org/GLM-5.3).

## Best practices

| Parameter | Value |
|---|---|
| temperature | 1.0 |
| top_p | 0.95 |
| reasoning_effort | `max` for benchmark reproduction, `low` or `high` to spend fewer tokens |
| clear_thinking | `true` for chat |

From Z.ai's evaluation settings (HLE w/ tools). Per-benchmark settings vary; see the base
model card for details.

## Run in llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --target llama-cli llama-server
```

```bash
./llama.cpp/build/bin/llama-server \
    -hf AtomicChat/GLM-5.3-GGUF:AD-Q4_K_M \
    --jinja -ngl 99 -c 8192 -fa on
```

## How these were made

1. Download `zai-org/GLM-5.3` (original FP8 weights).
2. Convert to GGUF with a [llama.cpp](https://github.com/ggml-org/llama.cpp) build that supports the GLM-5.3 architecture (`glm_moe_dsa`, sparse attention with a learned indexer).
3. Build an importance matrix over our public [calibration corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora).
4. Quantize the ladder with `--imatrix`, assigning bits per tensor role: router and shared expert high, routed experts carrying the compression.

## License

Released by Z.ai (zai-org) under the [GLM-5.3 license](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE). Quantized by Atomic Chat.

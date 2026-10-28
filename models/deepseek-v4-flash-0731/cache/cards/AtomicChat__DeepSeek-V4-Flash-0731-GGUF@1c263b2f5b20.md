---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: AtomicChat
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- llama.cpp
- deepseek
- deepseek-v4
- deepseek_v4
- moe
- mixture-of-experts
- imatrix
- quantized
- mxfp4
- qat
- 1-bit
- 2-bit
- 3-bit
- conversational
- atomic-chat
language:
- en
- zh
metrics:
- perplexity
- kl_divergence
---

# DeepSeek-V4-Flash-0731-GGUF

<div style="display: flex; gap: 8px; align-items: center; margin-top: 10px; margin-bottom: 10px;">
  <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_deepseek_v4_flash_0731&utm_content=btn_atomic"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_atomic.png" width="162" alt="Atomic Chat"></a>
  <a href="https://discord.gg/8wGSsvmg4V"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_discord.png" width="119" alt="Discord"></a>
  <a href="https://github.com/AtomicBot-ai/Atomic-Chat"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_github.png" width="115" alt="GitHub"></a>
  <a href="https://x.com/atomic_chat_hq"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_x.png" width="108" alt="X"></a>
</div>
<ul style="margin: 0 0 12px 0;">
  <li>Built by the team behind <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_deepseek_v4_flash_0731&utm_content=bullet_app">Atomic Chat</a>, the app that runs open-weight models locally and offline.</li>
</ul>
<hr style="margin: 0 0 16px 0;">

> [!NOTE]
> **Changes**
> - **28 Oct 2026:** the embedded chat template was updated to DeepSeek's current one in
>   10 of the 14 rungs: `AD-IQ3_S`, `AD-IQ3_XS`, `AD-IQ3_XXS`, `AD-IQ2_M`, `AD-IQ2_S_XL`, `AD-IQ2_S`, `AD-IQ2_XS`, `AD-IQ2_XXS`, `AD-IQ1_M_XL`, `AD-IQ1_M`.
>   For these, re-download the first file of the rung (`*-00001-of-*`); nothing else
>   changed. Reasoning from earlier turns now stays in the prompt across tool calls, as in
>   DeepSeek's reference encoder, and `reasoning_effort` reaches the model.

> [!WARNING]
> `AD-BF16`, `AD-MXFP4`, `AD-IQ3_M_XL`, `AD-IQ3_M` still embed the old template, which drops earlier reasoning in tool
> loops and never passes `reasoning_effort` to the model. Until they are repacked,
> pass the corrected template explicitly:
>
> ```bash
> curl -sLO https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF/resolve/main/chat_template.jinja
>
> llama-server -m <your-quant>/...-00001-of-*.gguf --jinja --chat-template-file ./chat_template.jinja
> ```
>
> `chat_template.jinja` at the repo root is the template already embedded in the other rungs.

DeepSeek-V4-Flash is a 284B parameter mixture of experts model (13B active per token): 43 layers, 256 routed experts with 6 active, one shared expert, 1M context, vocabulary 129,280. It is **quantization-aware-trained** — the official checkpoint already stores its routed experts in MXFP4 and everything else in FP8 or BF16.

## What that means in practice

The experts are 96% of the model and they are already 4.25 bits. There is nothing to gain above that: repacking MXFP4 into a wider format adds bytes and zero information. So `AD-BF16` is not a "high precision" quant, it is the original weights in a GGUF container, verified byte for byte.

Everything below it trades expert bits for size, and because the model was trained at 4 bits, it has no spare precision to give. Errors below 3 bits grow faster than they would on a BF16-trained model. The table shows exactly how fast.

## Quality vs size

![01_atomicchat_ladder_kld(1)](https://cdn-uploads.huggingface.co/production/uploads/6a54dea4f19f5386700504da/6AwA6uDut29CYNxq_jsdy.png)

| Quant | Size | Expert bits | PPL | Mean KLD | Top-1 match | Δp RMS |
|---|---:|---:|---:|---:|---:|---:|
| `AD-BF16` | 162.1 GB | 4.25 | 4.5289 | 0 | 100.000% | 0.000% |
| `AD-MXFP4` | 154.5 GB | 4.25 | 4.5446 | 0.1564 | 87.369% | 12.686% |
| `AD-IQ3_M_XL` | 143.6 GB | 3.94 | 4.5490 | 0.1675 | 86.864% | 13.004% |
| `AD-IQ3_M` | 135.8 GB | 3.71 | 4.5695 | 0.1798 | 86.317% | 13.615% |
| `AD-IQ3_S` | 130.8 GB | 3.56 | 4.6016 | 0.1891 | 85.945% | 13.988% |
| `AD-IQ3_XS` | 118.2 GB | 3.20 | 4.6657 | 0.2065 | 85.384% | 14.461% |
| `AD-IQ3_XXS` | 108.1 GB | 2.91 | 4.8491 | 0.2495 | 83.761% | 15.947% |
| **`AD-IQ2_M`** | **104.0 GB** | **2.79** | **4.8822** | **0.2567** | **83.560%** | **16.184%** |
| `AD-IQ2_S_XL` | 96.8 GB | 2.58 | 5.1406 | 0.3187 | 81.461% | 18.500% |
| `AD-IQ2_S` | 93.4 GB | 2.48 | 5.2152 | 0.3343 | 81.031% | 18.996% |
| `AD-IQ2_XS` | 85.1 GB | 2.25 | 5.4917 | 0.3947 | 79.240% | 20.817% |
| `AD-IQ2_XXS` | 78.5 GB | 2.06 | 5.7878 | 0.4544 | 77.459% | 22.378% |
| `AD-IQ1_M_XL` | 72.8 GB | 1.89 | 6.1786 | 0.5351 | 75.162% | 24.695% |
| `AD-IQ1_M` | 70.2 GB | 1.81 | 6.3813 | 0.5641 | 74.547% | 25.253% |

**PPL** — perplexity of the quant on wikitext-2. **Mean KLD** — how far its whole output distribution drifted from the lossless reference, in nats. **Top-1 match** — how often it picks the same next token as the reference; read this one if you read only one. **Δp RMS** — how much its confidence moved, in percentage points.

Names follow the real average bits per expert weight, not a size class. `_XL` means a rung sits above its tag. `AD-IQ2_M` at 2.79 bits carries the same expert budget that other publishers label `IQ3_XXS`.

## Against other publishers

Every number below was measured by us, on one harness, against one reference. Absolute values are not comparable to numbers published elsewhere — different corpora and context lengths shift them by tens of percent. Relative values within this table are.

![02_all_publishers_kld(1)](https://cdn-uploads.huggingface.co/production/uploads/6a54dea4f19f5386700504da/6Iiew-NBfobPIgVGOaR8H.png)

| Size | AtomicChat | KLD | unsloth | KLD |
|---:|---|---:|---|---:|
| 155 GB | `AD-MXFP4` | 0.1564 | `UD-Q4_K_XL` | **0.1557** |
| 136 GB | `AD-IQ3_M` | 0.1798 | `UD-IQ4_XS` | **0.1779** |
| 128–131 GB | `AD-IQ3_S` | **0.1891** | `UD-Q3_K_XL` | 0.1981 |
| 116–118 GB | `AD-IQ3_XS` | **0.2065** | `UD-IQ3_S` | 0.2565 |
| 104 GB | `AD-IQ2_M` | **0.2567** | `UD-IQ3_XXS` | 0.2610 |
| 96.8 GB | `AD-IQ2_S_XL` | **0.3187** | `UD-Q2_K_XL` | 0.3216 |
| 91 GB | — | — | `UD-IQ2_M` | 0.3700 |
| 82.5 GB | — | — | `UD-IQ1_S` | 0.4863 |
| 78.5 GB | `AD-IQ2_XXS` | 0.4544 | — | — |
| 70.2 GB | `AD-IQ1_M` | 0.5641 | — | — |

Two pairs land on the same byte count — 96.8 GB and 104 GB — and we are ahead in both. The largest margin is at 118 GB, where `AD-IQ3_XS` is 19% closer to the reference than `UD-IQ3_S` for two extra gigabytes. Above 135 GB the difference is inside measurement error. Below 82 GB no other publisher ships a rung.

## Choosing a quant

Pick the largest file that fits your memory with room for context.

| Your RAM | Pick | Why |
|---|---|---|
| 192 GB+ | `AD-BF16` | Bit-exact original |
| 160 GB | `AD-MXFP4` | Experts untouched, 7 GB lighter |
| 144 GB | `AD-IQ3_M_XL` | Barely distinguishable from lossless |
| 128 GB | `AD-IQ2_M` | Best quality that leaves room for context |
| 112 GB | `AD-IQ2_S_XL` | Sharp value, still above 81% top-1 |
| 96 GB | `AD-IQ2_XS` | Usable, noticeable drift |
| 80 GB | `AD-IQ1_M_XL` | Last rung that stays coherent |

## Requirements

DeepSeek V4 is supported by upstream llama.cpp — no PR checkout and no patching.
Build from `master`:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j --target llama-cli llama-server
```

For AMD, swap `-DGGML_CUDA=ON` for `-DGGML_HIP=ON -DGPU_TARGETS=<your gfx>`.

```bash
./build/bin/llama-cli -hf AtomicChat/DeepSeek-V4-Flash-0731-GGUF:AD-IQ2_M \
  --jinja -ngl 99 -c 8192 -fa off
```

> [!NOTE]
> Thinking is on by default. `reasoning_effort` takes `high` or `max`, per request or
> with `--reasoning-effort` on the server; no value or `low` adds no effort prompt, and
> other names are ignored, so a client that maps every level to `high` never reaches
> `max`. Requests with `tools` keep the reasoning of every earlier turn, as DeepSeek's
> encoder does. In chats without tools DeepSeek drops the reasoning before the last user
> message; llama-server keeps it by default (`--reasoning-preserve`), and
> `--no-reasoning-preserve` restores DeepSeek's behaviour.

> [!IMPORTANT]
> Use `-fa off`. With flash attention enabled, CUDA emits repeating `<`
> characters once the prompt exceeds a single pass
> ([#26509](https://github.com/ggml-org/llama.cpp/issues/26509)). Quantized
> K-cache is also still broken
> ([#25382](https://github.com/ggml-org/llama.cpp/issues/25382)) - leave the
> cache at its default type.

> [!NOTE]
> This model declares a 1,048,576-token context. `llama-cli` and `llama-server`
> honour that default and will try to allocate a KV cache for the full million
> tokens, which fails on any single GPU. Always pass an explicit `-c`.

> [!NOTE]
> Earlier revisions of this card warned that the hyper-connection operations had
> no GPU kernels and that offload therefore gained little. That is fixed
> upstream: [#25585](https://github.com/ggml-org/llama.cpp/pull/25585) fused the
> hyper-connection ops and [#25545](https://github.com/ggml-org/llama.cpp/pull/25545)
> added a CUDA lightning indexer. On a build without them, sinkhorn
> normalisation and the hyper-connections decompose into thousands of
> elementwise kernels per token. Measured on one MI300X with `AD-IQ3_M`, the
> same file went from 16.34 t/s to 37.27 t/s across that change — 2.28x, with no
> change to the model. Build from current `master`.


## How these were made

1. Download the original safetensors.
2. Convert with `convert_hf_to_gguf.py --no-lazy --outtype bf16`. The `--no-lazy` flag is not optional — lazy evaluation corrupts `token_embd.weight` with NaN values on this architecture.
3. Patch `conversion/deepseek.py` so FP8-sourced tensors resolve to BF16 instead of the hardcoded Q8_0. Without this the base is not lossless: Q8_0 on FP8 tensors costs 0.219 KLD, because a linear int8 grid cannot represent a logarithmic float grid at the same bit count, and QAT trained the model to tolerate one and not the other.
4. Build the importance matrix over our own calibration corpus.
5. Quantize each rung with explicit per-tensor overrides.

The base was verified against unsloth's independently produced lossless quant: top-1 agreement 100.000%, RMS Δp 0.000%, mean KLD within numerical noise.

### Per-tensor layout

Constant across every rung:

| Tensors | Type |
|---|---|
| norms, `attn_sinks`, `exp_probs_b.bias`, `*_ape`, all `hc_*` | F32 |
| `indexer.proj` | F32 |
| `ffn_gate_inp` (router) | BF16 |
| `attn_output_a/b`, `attn_q_b`, `attn_kv`, `compressor_gate/kv`, `indexer.attn_q_b` | Q8_0 |
| `attn_q_a`, `ffn_*_shexp`, `token_embd`, `output` | Q6_K |
| `ffn_gate_tid2eid` (hash routing table) | I32, untouched |

The experts vary. `ffn_down_exps` stays native MXFP4 on layers 41 and 42 in every rung; `ffn_gate_exps` and `ffn_up_exps` are split into the 18 highest-energy layers (20, 26–38, 40–42) and the remaining 25. The split comes from measured activation energy per layer via `llama-imatrix --show-statistics`, not from a hand-tuned map — energy rises monotonically with depth, from 45k at layer 0 to 661k at layer 42.

### Calibration corpus

1,868,626 tokens, 229 chunks of 8,192. Vocabulary coverage 87.08% of 129,280 tokens seen at least once, p99 document length 12,251 tokens. Composition: graphics and 3D code 31.8%, long context 14.4%, agentic tool-call traces 13.8%, general code 12.8%, general text 9.0%, vocabulary sweep 8.3%, reasoning 5.8%, structured output 4.3%.

The vocabulary sweep exists because the first three layers route by a hash table keyed on token id rather than a learned gate — their expert coverage depends on vocabulary breadth, not corpus size. The 8,192 context exists because at short contexts the model's compressors and lightning indexer barely engage.

`imatrix.dat` is published at the repo root.

## Reproducing the measurements

Reference is `AD-BF16`, corpus is the wikitext-2 test split from `Salesforce/wikitext` (parquet, concatenated), context 5,632, 51 chunks, llama.cpp build `f180ae8b2`.

```bash
python3 -c "
from huggingface_hub import hf_hub_download
import pandas as pd
p = hf_hub_download('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet',repo_type='dataset')
open('wiki.txt','w').write(''.join(pd.read_parquet(p)['text']))"

./build/bin/llama-perplexity -m AD-BF16/DeepSeek-V4-Flash-0731-AD-BF16-00001-of-00004.gguf \
  -f wiki.txt --kl-divergence-base ref.kld -ngl 99 -c 5632 -b 5632

./build/bin/llama-perplexity -m AD-IQ2_M/DeepSeek-V4-Flash-0731-AD-IQ2_M-00001-of-00004.gguf \
  -f wiki.txt --kl-divergence-base ref.kld --kl-divergence -ngl 99 -c 5632 -b 5632
```

Context 5,632 was chosen because perplexity of the lossless reference lands at 4.5289 there, matching the figure published for the official weights, and because at 512 tokens this architecture's indexer selects 512 of 512 and its compressors do nothing — the model is measured in a regime it was not built for.

## Why hardware is part of the harness

llama.cpp has two code paths for MXFP4 weights. One unpacks them to BF16 and uses a
standard tensor-core matmul; the other feeds the packed 4-bit data straight into
block-scaled tensor-core instructions. The second is gated in `ggml/src/ggml-cuda/common.cuh`:

    #define GGML_CUDA_CC_BLACKWELL 1200
    #if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
    #    define BLACKWELL_MMA_AVAILABLE
    #endif

Compute capability is encoded as `major*100 + minor*10`, so the threshold of 1200 means
compute capability 12.0 — consumer Blackwell only. H100 and H200 are 9.0, B200 is 10.0,
B300 is 10.3. All of them fall below the threshold and take the dequantize path, despite
B200 and B300 having FP4 tensor cores in hardware.

Since the routed experts are 96% of this model and they are MXFP4, that branch changes
the output distribution. Same file, same corpus, same context, same build commit:

| Reference model | ctx 512 | ctx 5632 |
|---|---:|---:|
| RTX 5090 (cc 12.0, native FP4) | 5.4312 | 4.5381 |
| H100 (cc 9.0, dequantize) | 5.1554 | 4.3406 |

That is a 4–5% gap from the GPU alone. **Perplexity and KL-divergence numbers for this
model are not portable between GPU generations**, which is why every number in this card
was measured by us on one machine — 8× RTX 5090, llama.cpp build `f180ae8b2`,
`-DCMAKE_CUDA_ARCHITECTURES=120`, `-ngl 99 -c 5632 -b 5632`.

Numbers published elsewhere were produced on different hardware; unsloth state 4× B200 for
their table. Do not read across tables — compare within one.

## Throughput on AMD MI300X

All rungs above run on a single MI300X (192 GB HBM3, gfx942) with no tensor
parallelism and no CPU offload. Measured with `llama-bench`, ROCm 7.2.4,
llama.cpp with the fused DeepSeek V4 kernels, `-ngl 99 -fa 0 -p 512 -n 128 -r 3`.

| Quant | File size | pp512 t/s | tg128 t/s | Peak VRAM |
|---|---:|---:|---:|---:|
| `AD-IQ2_M` | 96.85 GiB | 615.60 ± 0.32 | 37.16 ± 0.70 | 98.8 GiB |
| `AD-IQ3_XXS` | 100.70 GiB | 612.88 ± 1.01 | 36.92 ± 0.65 | 102.6 GiB |
| `AD-IQ3_S` | 121.77 GiB | 652.11 ± 0.93 | 37.31 ± 0.73 | — |
| `AD-IQ3_M` | 126.46 GiB | 647.62 ± 2.85 | 37.19 ± 0.79 | — |
| `AD-IQ3_M_XL` | 133.77 GiB | 665.47 ± 1.59 | 37.75 ± 0.76 | — |

**Generation speed does not depend on quant size.** Across a 1.4x spread in
file size, token generation stays inside 36.92–37.75 t/s — a spread smaller
than the run-to-run error. Prompt processing does vary, and there the larger
IQ3 rungs are faster, because prefill runs real GEMMs and the dequantisation
kernel differs per type.

The reason generation is flat is that this model is not memory-bound here. At
37 t/s with roughly 3.7 GB of active weights read per token, the effective
bandwidth is about 137 GB/s against the 5.3 TB/s the card can deliver — 2.6% of
it. What limits generation is kernel launch overhead across 43 layers, not
memory traffic.

The practical consequence: **pick the largest rung that fits your memory.**
On this hardware the bigger quant is free in generation speed and better in
quality and prefill.

Peak VRAM was sampled during the run and includes weights plus working buffers
at the benchmark's context. Budget headroom on top for your own `-c`.


## License

MIT, inherited from the original model. Copyright DeepSeek. Quantized and published by [Atomic Chat](https://huggingface.co/AtomicChat).
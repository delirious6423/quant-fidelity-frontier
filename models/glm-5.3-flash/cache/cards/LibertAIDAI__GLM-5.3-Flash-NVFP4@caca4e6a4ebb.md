---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: LibertAIDAI
tags:
  - nvfp4
  - blackwell
  - sglang
  - vllm
  - glm
  - glm-5
  - glm5_next
  - moe
  - multimodal
  - modelopt
language:
  - en
  - zh
pipeline_tag: image-text-to-text
---

<div align="center">

# GLM-5.3-Flash · NVFP4

### 320B total · 18B active · natively multimodal · 1M context

**598.5 GiB → 181 GiB** &nbsp;·&nbsp; **−70%** &nbsp;·&nbsp; round-trip cosine **0.99665**

[![Base](https://img.shields.io/badge/base-zai--org%2FGLM--5.3--Flash-1f6feb?style=for-the-badge)](https://huggingface.co/zai-org/GLM-5.3-Flash)
[![Format](https://img.shields.io/badge/format-NVFP4--A16-6f42c1?style=for-the-badge)](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
[![License](https://img.shields.io/badge/license-MIT-2da44e?style=for-the-badge)](https://opensource.org/license/mit)

*Weight-only NVFP4 quantization of Z.ai's GLM-5.3-Flash — the first GLM to combine **sparse and linear attention**.*

**Quantized by [LibertAI](https://libertai.io)** · not affiliated with Z.ai / Zhipu

</div>

---

## ✨ What this is

A **weight-only NVFP4 (NVFP4-A16)** checkpoint. The **routed-expert FFN** tensors — 97% of the model's parameters — are quantized to **NVFP4** (E2M1, with FP8-E4M3 per-16-block scales and an FP32 per-tensor global scale). Everything outlier-sensitive stays in **BF16**:

> both attention flavours (all 34 KDA linear-attention layers *and* the 11 DeepSeek-sparse layers, including the sparse indexer) · the **entire vision tower** · shared experts · MoE routers · dense-MLP layers · the MTP head · the Manifold-Constrained Hyper-Connection (mHC) tensors · token embeddings · `lm_head` · all norms

Activations stay BF16 — there is no activation quantization.

Because the experts dominate the footprint, quantizing **only** them buys the full 70% while protecting quality where it matters. Leaving the vision tower untouched keeps multimodal behaviour **bit-identical** to the source.

Produced with [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 0.45.0 through a memory-frugal shard-streaming pass — CPU-only, never more than one shard resident, no calibration data (weight-only NVFP4 derives its scales from the weights themselves).

---

## 🚦 Engine support — read this first

`glm5_next` is a brand-new architecture (released 2026-08-26) and is **not yet in vLLM `main`** ([vllm#53906](https://github.com/vllm-project/vllm/pull/53906), [sglang#36507](https://github.com/sgl-project/sglang/pull/36507)). Support ships in **per-model images**.

| Hardware | Engine | Status |
|---|---|---|
| H100 / B200 / GB200 | vLLM | ✅ vendor-verified |
| **GB10 / DGX Spark (sm_121)** | **SGLang** | ✅ **verified by us** — recipe below |
| **GB10 / DGX Spark (sm_121)** | **vLLM** | ✅ **verified by us** — needs the [recipe](https://github.com/Libertai/glm53-flash-vllm-gb10) |

```
vllm/vllm-openai:glm53-flash-x86_64-cu130     # x86_64, CUDA 13
vllm/vllm-openai:glm53-flash-arm64-cu130      # arm64 (GH200/GB200/GB10)
lmsysorg/sglang:glm-5.3-flash-arm64           # SGLang, arm64
```

> ✅ **vLLM on sm_121 (GB10) now works**, via [Libertai/glm53-flash-vllm-gb10](https://github.com/Libertai/glm53-flash-vllm-gb10). Two independent faults had to be fixed and either one alone leaves the model degenerate, which is why it long looked like a single unexplainable bug. First, no vLLM MLA backend accepted this model's NoPE dimensions on sm_121, since the sparse decode path asserts `pe_dim == 64` and GLM-5.3-Flash has `qk_rope_head_dim = 0`; that is fixed by a hand-written sparse-MLA CUDA kernel. Second, and not specific to GB10, see the vLLM note below. **The checkpoint was never at fault.**

**llama.cpp has no `glm5_next` support**, so there is no GGUF.

---

## 🚀 Usage

> ⚠️ **Point your engine at a local directory, not the repo id.** vLLM's
> `glm5next.py` opens `os.path.join(model_path, "processor_config.json")` directly
> instead of resolving it through the hub cache, so a repo-id launch fails with a
> missing-file error even though the file is right there
> ([discussion #2](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/discussions/2)).
> Run `hf download LibertAIDAI/GLM-5.3-Flash-NVFP4 --local-dir ./glm53` first and pass
> `./glm53`.


### SGLang — verified on 2× GB10 (sm_121), TP=2

```bash
python3 -m sglang.launch_server \
  --model-path LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  --trust-remote-code --tp-size 2 \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype bfloat16 \
  --disable-shared-experts-fusion \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --mem-fraction-static 0.84 \
  --context-length 65536 --max-running-requests 2
```

Every flag above is load-bearing on sm_121:

| Flag | Why it is required |
|---|---|
| `--disable-shared-experts-fusion` | The shared expert is **BF16** in this checkpoint (only *routed* experts are quantized). Shared-expert fusion would pack it into the NVFP4 buffer and the load fails on a shape mismatch. |
| `--dsa-*-backend tilelang` | The only DSA backend with a **NoPE** (`tail_dim == 0`) kernel. `flashinfer_sparse_mla` hardcodes a rope-bearing 448+64 page layout; all four `flashmla_*` are excluded by `index_kpool=4`. |
| `--kv-cache-dtype bfloat16` | TileLang on CUDA is **bf16-KV-only**. Merely *omitting* an fp8 flag is not enough — the DSA default re-selects fp8. |
| `--moe-runner-backend flashinfer_cutlass` | The NVFP4 MoE runner auto-selects a datacenter-Blackwell backend on `(12, 1)`. `marlin` also works but its repack costs ~19 GiB/rank. |
| `--reasoning-parser glm45` *(SGLang only — vLLM needs `deepseek_r1`)* | **Without it `reasoning_parser` is `None`** and the entire thinking trace is emitted inside `content`, ending in a bare `</think>` with no opening tag, while `reasoning_content` stays `null`. See below. |
| `--tool-call-parser glm47` | Enables tool calling. ⚠️ **Not `glm`** — see the warning below. Note the two parser flags legitimately take *different* spellings: `glm45` for reasoning, `glm47` for tools. |

### 🧠 Thinking / reasoning output

This model **starts inside its reasoning block without emitting an opening `<think>`** — the chat template opens it — and closes with `</think>`. With no reasoning parser configured you therefore get:

```jsonc
// ✗ no --reasoning-parser
"content": "The user is asking ... so the answer is 391.</think>391",
"reasoning_content": null
```

With `--reasoning-parser glm45` it splits correctly:

```jsonc
// ✓
"content": "391",
"reasoning_content": "17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391",
"usage": { "reasoning_tokens": 43 }
```

`deepseek-r1` also splits this shape (it is `force_reasoning=True` on `</think>`). **On SGLang, prefer `glm45`**: it additionally excludes `<tool_call>`, `</tool_call>`, `<eop>` and `<|user|>` from the reasoning span, which matters once tool calling is enabled. **On vLLM the choice inverts — you must use `deepseek_r1`**, because vLLM's `glm45` is an alias for the GLM-4.7-MoE parser engine and discards the reply entirely against this model's prompt-side `<think>`. See the vLLM section below.

### 🛠️ Tool calling — use `glm47`, and beware the silent failure

⚠️ **`--tool-call-parser glm` fails silently on this model.** It is the GLM-4.5
format; GLM-5.x emits a different one. The symptom is *not* an error — the request
succeeds and you get:

```jsonc
// ✗ --tool-call-parser glm
"finish_reason": "stop",
"tool_calls": null,
"content": ""          // <- the parser consumed the tool call and dropped it
```

An **empty** `content` alongside `tool_calls: null` is the tell: the parser matched
enough to swallow the output but not enough to emit a call. Use **`glm47`**, which
is also what the vendor vLLM recipe specifies for GLM-5.x.

### 📐 Sizing: the KDA state cache is the real constraint, not KV

GLM-5.3-Flash has **34 KDA linear-attention layers**, each needing a **per-request recurrent state**. That cost scales with *concurrency*, and on a memory-tight box it — not the KV cache — is what limits you:

```
RuntimeError: Hybrid (mamba/linear-attention) state cache is too small to serve any requests.
max_mamba_cache_size=1, mamba_ratio=5, resulting max_num_reqs=0.
```

So **context length and concurrency compete directly**. Measured on 2× GB10 (TP=2) at `--mem-fraction-static 0.84`:

| `--max-running-requests` | `--context-length` | result |
|---|---|---|
| 8 | 131072 | ❌ mamba cache → `max_num_reqs=0` |
| **2** | **65536** | ✅ KV **212,864 tok** (2.52 GB), mamba cache 31 slots (2.19 GB) |

If you hit that error, **lower `--max-running-requests` before lowering context** — the state cache is per-request, so concurrency is usually the cheaper thing to give up.

`--enable-radix-cache` (prefix reuse) and CUDA graphs both work; SGLang reports `cuda graph: True` in its decode lines and initializes a `UnifiedTreeCore` radix cache with a `MAMBA` component.

<details>
<summary><b>⚠️ GB10 also needs a small TileLang shared-memory patch</b></summary>

<br>

SGLang's stock TileLang tile requests **169,984 B** of dynamic shared memory. Consumer/workstation Blackwell allows far less — GB10 measures:

```
shared_memory_per_block_optin = 101,376 B
```

so the kernel fails to launch. Working tile on GB10: `block_I=32, num_stages=1, threads=128`. Measured coupling worth knowing: **`threads` must drop to 128 before `block_I` can drop to 32**, or the `m_i`/`alpha` fragment layouts become unsatisfiable; `block_I=16` hits an MMA assert.

Retuned kernel verified against a float32 gathered-attention reference: **rel err 2.4e-3** (bf16 rounding noise) — correct, not merely non-crashing.

Datacenter parts (H100/B200/GB200) have the shared-memory headroom and need no patch.

**If you add a speculative drafter, you need this tile even if plain decode worked.**
Reported by [@randomllama](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/discussions/6)
running the incoai DFlash2 drafter on 2× GB10 (SGLang TP=2): the DSA **verify** shape
(`num_tokens_per_req=8`) requests the same 169,984 B and blows the same 101,376 B
limit, while plain decode fits at stock tiles — so a drafter-less boot never surfaces
it, and the same `block_I=32, num_stages=1, threads=128` tile is the fix.

</details>

<details>
<summary><b>⚠️ A drafter also breaks the mem-fraction envelope below</b></summary>

<br>

The `--mem-fraction-static 0.84` figures in the table above are **drafter-less**. With a
drafter at `D=8` the hybrid state pool needs `per_req * (1 + D)` (~633 MB) on top of the
~`ratio=5` per-request amplification, and you get `max_mamba_cache_size <= 0`. Working
configuration reported in [discussion #6](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/discussions/6):

```bash
  --mem-fraction-static 0.88 \
  --max-total-tokens 131072 \
  --mamba-full-memory-ratio 2
```

measured at **27.6 tok/s** code decode, 1.88× over the same stack without speculation.

</details>

### ⚠️ vLLM + NVFP4 MoE: check your output before trusting it

This checkpoint is **weight-only** NVFP4: activations are not statically quantized.

> ### ⚠️ Updated 2026-08-30 (second time today) — the earlier fix was wrong
>
> **If you pulled this repo before 2026-08-30 15:00 UTC, re-download
> `model-input-scales.safetensors`.** The history, because the failure mode changed
> twice:
>
> 1. **Originally** this checkpoint carried no `input_scale` at all. vLLM's
>    `ModelOptNvFp4FusedMoE` then folded uninitialised memory into the dequant
>    alphas, zeroing every expert — the `locklocklock` degeneration.
> 2. **Earlier today** we shipped `input_scale = 1.0` placeholders and argued a
>    constant was "a legitimate fix, not a fudge", because the scale cancels
>    analytically and 1.0 sits inside the safe band. **That argument was wrong.**
>    It checked the band against the *tensor* amax, but the underflow bound is
>    **per 16-element block**: `input_scale <= amax_block / 0.1`. At 1.0, every
>    activation block whose amax is under 0.1 has its fp8 block scale flushed to
>    zero. Which blocks those are depends on the input, so the model degrades
>    *intermittently*, and worse as context grows — reported independently on
>    GB300 ([discussion #7](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/discussions/7))
>    and reproduced on our own sm_120 deployment.
> 3. **Now** `model-input-scales.safetensors` carries **real per-projection
>    calibrated values** (36,288 of them, taken from RedHat's calibration of the
>    same base model; the 864 MTP-layer entries get the median, as RedHat runs that
>    layer at FP8 with dynamic activations). Range `5.58e-04 … 3.72e-02`, median
>    `1.58e-03` — the 1.0 placeholder was **632x the median**. The underflow
>    threshold drops from `amax_block >= 0.1` to `>= 1.6e-04`.
>
> **If you use the [sparse-MLA plugin](https://github.com/Libertai/vllm-sparse-mla-blackwell)'s
> `VLLM_GLM53_MOE_INPUT_SCALE`, do not leave it at 1.0** — a single constant cannot
> satisfy both bounds across all projections (clipping needs up to `3.7e-02`,
> underflow wants small), so prefer the checkpoint's per-tensor values; if you must
> use a constant, `1.6e-03` is the measured median.
>
> A cleaner fix is on the **[`compressed-tensors` branch](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/tree/compressed-tensors)**
> (`--revision compressed-tensors`): same 4-bit expert tensors relaid out in the
> `compressed-tensors` format, MTP layer at FP8, and **NVFP4-A16 — activations are
> never quantized, so there is no scale to get wrong and no underflow at any
> context length**. ⚠️ Verified numerically only, and it **failed to load** on the
> `glm53-flash-x86_64-cu130` per-model image in our own test (crash loop, traceback
> not yet captured), so treat it as experimental. **SGLang users should stay on
> `main`** — the SGLang path is known to work with the ModelOpt layout here.

vLLM's `ModelOptNvFp4FusedMoE` registers the activation scale as an
uninitialized parameter and expects the checkpoint to fill it. With a checkpoint that
carries no `input_scale` it stays at zero, so the dequantization alpha for every expert becomes
`weight_scale_2 * 0`, the whole mixture of experts is multiplied by zero, and the
model emits one token repeatedly. There is no error and no warning. vLLM's NVFP4
*linear* methods refuse such a checkpoint explicitly; the MoE method does not.

If you see degenerate output, either:

```bash
  --moe-backend marlin          # dequantizes weights, never reads an activation scale
```

or install the plugin from the [recipe repo](https://github.com/Libertai/glm53-flash-vllm-gb10),
which supplies the missing scale and lets you keep `flashinfer_cutlass` (marlin
repacks at roughly 19 GiB per rank, which does not fit at TP=2 on 2× GB10).

We observed this on sm_121. The trigger is the checkpoint rather than the
architecture, so we expect it to reproduce on datacenter Blackwell as well; we
have not yet confirmed that.

---

### vLLM — H100 / B200 / GB200

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  vllm/vllm-openai:glm53-flash-x86_64-cu130 \
  --model LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  --tensor-parallel-size 4 \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser deepseek_r1
```

> ⚠️ **On vLLM use `deepseek_r1`, NOT `glm45`.** The `glm45` recommendation above is
> **SGLang-specific**. On vLLM that name resolves to `Glm47MoeParserReasoningAdapter`,
> whose state machine expects an opening `<think>` in the *output* — but this model's
> template emits `<think>` as the last **prompt** token, so the parser never opens a
> valid span and **silently discards the whole reply**: `completion_tokens: 400` with
> `content` *and* `reasoning_content` both empty, on `/v1/chat/completions` and
> `/v1/messages` alike. It presents as a broken or hanging model rather than a parser
> fault. `deepseek_r1` terminates on the bare `</think>` and splits correctly.
> Note also that vLLM returns the trace under **`reasoning`**, not `reasoning_content`.

The **MTP layer is kept in BF16**, so speculative decoding works as in the vendor recipe:

```bash
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

> 💡 Engine init on a 320B MoE is slow. Set `VLLM_ENGINE_READY_TIMEOUT_S=3600` or the server is killed mid-warmup.

---

## 📦 What's quantized

| Tensor group | Precision | Count |
|---|---|---|
| Routed-expert FFN (`language_model.layers.*.mlp.experts.*.{gate,up,down}_proj`) | **NVFP4** (g16, weight-only) | **37,152** |
| KDA linear attention (34 layers) · DeepSeek sparse attention + indexer (11 layers) | BF16 | — |
| Vision tower (`model.visual.*`) · shared experts · routers · dense/MTP MLP · mHC · embeddings · `lm_head` · norms | BF16 | 1,618 |

**38,770 tensors total.** Every non-expert Linear is listed in `config.json`'s `ignore`.

> 📝 That `ignore` list names modules as they appear **in the checkpoint** (unfused, e.g. `*.self_attn.q_proj`). Engines fuse at load time and each picks its own name for the fused module, so the list also carries the **fused** spellings (`qkv_proj`, `fused_qkvbfg_a_proj`, `fused_fg_b_proj`, `qkv_conv1d`, `fused_qkv_a_proj_with_mqa`). Without them SGLang treats BF16 attention as quantized and the load asserts.

## 💾 Size

| | BF16 source | This checkpoint |
|---|---|---|
| Routed experts (311.65B params) | 623 GB | **175 GB** (NVFP4) |
| Everything else (9.67B params) | 19 GB | 19 GB (BF16) |
| **Total** | **598.5 GiB** | **≈181 GiB** |

That is the difference between *"needs a GB200 tray"* and *"fits on two GB10 desktops."*

---

## 🔬 Provenance & verification

- **Base:** `zai-org/GLM-5.3-Flash` (BF16, 120 shards, 598.5 GiB)
- **Method:** shard-streaming ModelOpt `NVFP4QTensor` weight-only pass, CPU-only. Expert FFN → NVFP4; everything else copied through in BF16.
- **Partition check:** every Linear weight is provably quantized **XOR** ignored — no tensor both, none uncovered.
- **Round-trip:** per-expert cosine **≈ 0.99665**, relative error **≈ 0.0925** vs the BF16 source.
- **End-to-end:** generates correct, coherent text on 2× GB10 under SGLang (TP=2).

> ⬜ **No throughput numbers are published.** We have not benchmarked this checkpoint on any hardware, and we would rather publish nothing than publish a number we did not measure.

---

## 🧠 Why NVFP4?

On NVIDIA Blackwell GPUs (RTX 50-series, B100/B200, GB10/DGX Spark) NVFP4 weights run on **native FP4 tensor-core kernels**. For an MoE, whose decode is dominated by memory bandwidth, 4-bit expert weights cut both VRAM *and* the bytes moved per token. This gives format parity with the NVFP4 path used by vLLM, SGLang and TensorRT-LLM.

## 🌐 About LibertAI

[**LibertAI**](https://libertai.io) is a decentralized AI platform — private inference, an OpenAI-compatible API, and a chat UI, all running on community GPUs over [Aleph Cloud](https://aleph.cloud) instead of a single company's servers. No accounts required to chat, no logs sent home, and the same models you'd self-host are available behind a sovereign endpoint.

Want to put this model to work as an autonomous agent without running your own infrastructure? [**LiberClaw**](https://liberclaw.ai) hosts Hermes-style agents on Aleph Cloud with LibertAI inference. Free tier: 2 agents, no credit card, 5 minutes to deploy. Open source.

---

<div align="center">

**MIT licensed**, as is the base model · Quantized by [LibertAI](https://libertai.io)

</div>

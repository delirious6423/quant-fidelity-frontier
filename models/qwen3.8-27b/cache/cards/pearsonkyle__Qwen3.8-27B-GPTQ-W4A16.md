---
library_name: transformers
base_model:
- Qwen/Qwen3.8-27B
tags:
- vllm
- compressed-tensors
- gptq
- w4a16
- int4
- quantization
- quantized
- 4-bit
- llm-compressor
- qwen
- qwen3
- 27b
- tool-use
- function-calling
- agentic
- long-context
- 32k-calibration
- mtp
- speculative-decoding
- multimodal
- vision
- image-text-to-text
license: apache-2.0
language:
- en
pipeline_tag: image-text-to-text
---

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #c7d2fe; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.05); overflow: hidden; background: #ffffff; margin-bottom: 30px;">
  <div style="background: linear-gradient(135deg, #4f46e5 0%, #1e1b4b 100%); padding: 24px; color: white;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
      <h1 style="margin: 0; font-size: 26px; font-weight: 800; display: flex; align-items: center; gap: 12px; color: white; border: none;">⚡ Qwen/Qwen3.8-27B · W4A16 · vLLM</h1>
      <span style="background: #f59e0b; color: #1c1917; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;">GPTQ @ 32K calibration</span>
    </div>
  </div>
  <div style="display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 24px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
    <span style="background: #e0e7ff; color: #3730a3; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #c7d2fe;">📦 18.1 GiB · 2.8× smaller</span>
    <span style="background: #d1fae5; color: #065f46; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #a7f3d0;">compressed-tensors · int4 g128 sym</span>
    <span style="background: #fef08a; color: #713f12; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fde047;">🧬 calibrated at <b>ctx 32,768</b></span>
    <span style="background: #ede9fe; color: #5b21b6; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #ddd6fe;">⚡ MTP bundled · <b>1.71× decode</b></span>
    <span style="background: #f3e8ff; color: #6b21a8; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #e9d5ff;">👁️ Text + Image · vision 3/3</span>
    <span style="background: #fef3c7; color: #92400e; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fde68a;">🏅 tool-calling = bf16, turn for turn</span>
    <span style="background: #fce7f3; color: #9d174d; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fbcfe8;">🏗️ vLLM 0.27.1</span>
  </div>
  <div style="padding: 24px; display: flex; flex-direction: column; gap: 20px;">
    <div style="background: #eef2ff; border-left: 5px solid #4f46e5; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #3730a3; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>⚡</span> What this is</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">A <b>compressed-tensors W4A16</b> (GPTQ, int4 group-128 symmetric) quantization of <a href="https://huggingface.co/Qwen/Qwen3.8-27B"><b>Qwen/Qwen3.8-27B</b></a> that <b>vLLM serves directly</b> — no conversion, no custom runtime, quantization auto-detected from <code>config.json</code>. Calibrated on <b>4.26M tokens of real agentic-coding sessions</b> packed at 32K context. Sibling of the <a href="https://huggingface.co/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF"><b>GGUF ladder</b></a>, same corpus: <b>this one for vLLM</b>, the GGUF for llama.cpp / Ollama / LM Studio.</p>
    </div>
    <div style="background: #faf5ff; border-left: 5px solid #7c3aed; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #5b21b6; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>⚡</span> Speculative decoding + vision, both bundled</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">The trained <b>MTP draft head</b> ships <b>inside the checkpoint</b> — one <code>--speculative-config</code> flag, no second file, <b>1.71× decode at 71.2% acceptance</b>. The <b>vision tower</b> ships too, kept at bf16: shown a synthetic test image it named the colour, shape <b>and</b> position of all three shapes correctly, unprompted detail included ("inverted triangle, base horizontal at the top"). Both are additive — drop the flags and you are back to the identical text model.</p>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 10px;">
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #3730a3; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">📐 Read the size honestly</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;"><b>5.2 bits per weight overall</b>, not ~4.1. 90.5% of parameters are int4, but <code>embed_tokens</code> and <code>lm_head</code> (1.27 B each, over a 248k vocab) stay <b>bf16</b> — 4.74 GiB, 26% of the download. Quantizing an output head that wide is the classic rare-token failure mode. The quantized trunk itself is <b>4.05 bpw</b>.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #3730a3; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🛠️ Standard format</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Plain <code>compressed-tensors</code> — <b>vLLM serves it directly</b>, no conversion or forks.</span></div>
    </div>
  </div>
</div>

## 🚀 Quick start

Tuned for a 24 GB card; [what each flag does](#running-it-on-a-24-gb-gpu-rtx-3090) is below.

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER \
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=201326592 \
vllm serve pearsonkyle/Qwen3.8-27B-GPTQ-W4A16 \
    --max-model-len 106496 \
    --kv-cache-dtype fp8_e4m3 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 2048 \
    --language-model-only \
    --enable-prefix-caching \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3
```

> **`--tool-call-parser qwen3_xml` is required for tool calling.** Qwen3.8 emits XML tool
> calls (`<tool_call><function=NAME>`), not JSON. Without it `tool_calls` is always empty
> — which looks like a broken quant but is a serving flag.

---

## Quality

Measured on vLLM against a **bf16 reference served identically** — the only valid control.
(That same reference scores 0.563 here vs 0.494 on llama.cpp, so these do not compare
across stacks.)

| | bf16 *(ref)* | W4A16 |
|:---|---:|---:|
| Tool-selection accuracy · 174 held-out turns | 0.563 | **0.563** |
| Parameter accuracy | 0.345 | **0.350** |
| Schema-valid rate | 0.948 | **0.948** |
| Malformed tool calls | 0 | **0** |
| Median KLD vs bf16 · 6 distributions | — | **0.0021 – 0.0205** |
| Long-context retrieval · 29.7k tokens | — | **3/3** codes exact |
| Vision · colour + shape + position | — | **3/3** |
| SWE-rebench `dask__dask-11393` | — | **resolved** · 5 steps · 0 malformed |

Tool-calling is bf16-identical, turn for turn. KLD is lowest (0.0021) on agentic text —
exactly where it was calibrated — and highest (0.0205) on general English.

**Read these honestly.** SWE-rebench is one instance, not a pass rate. Agentic deltas are
noise-bound: vLLM is nondeterministic at `temperature=0`, and replaying the ladder moved
levels by up to 6 turns (3.4pp) — read the endpoints, not the third decimal. Video,
multi-image and long-context vision are untested. The vision tower and MTP head are bf16,
not quantized (+1.65 GiB), which is why they work.

<details>
<summary><b>How it was made</b> — quantization settings, what stayed bf16, calibration corpus</summary>

**GPTQ W4A16** via [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.13.0
→ compressed-tensors 0.18.0: int4, group 128, symmetric, static act-order. Sequential
pipeline over all 64 layers (48 linear-attention + 16 full-attention), 496 modules quantized.

**Kept bf16 on purpose:** `lm_head`, `embed_tokens`, vision tower, MTP head. The
248,320-token vocab with untied embeddings makes the two vocab tensors **4.74 GiB, 26% of
the download** — which is why a "4-bit" 27B lands at 5.2 bpw overall (trunk alone: 4.05)
and is bigger than the 14.5 GiB IQ4_XS GGUF. A quantized head over a 248k vocab is the
classic rare-token failure mode; the needle test is the check that this worked.

**Calibration:** 4,255,761 tokens / 3,436 windows at ctx 32,768 — 63% real agentic sessions
(CLI logs + SWE trajectories), 6,786 tool calls across 76 schemas, plus reasoning turns,
broad-instruct and red-team refusals. GPTQ drew 128 × 32,768-token sequences by
deterministic whole-corpus stride.

Built with [Quant-Tuner](https://github.com/pearsonkyle/Quant-Tuner); logs mined with
[LogMiner](https://github.com/pearsonkyle/LogMiner).

</details>

---

## Running it

The Quick start command is tuned for a **24 GB card (RTX 3090)**: **106,496 tokens of context at ~75 tok/s**
(45 without MTP), using 23.6 of 24.0 GiB. The download is 18.1 GiB and still fits because only
16 of the 64 layers keep a KV cache — the other 48 are linear attention.

| Flag | Why it matters |
|:---|:---|
| `--tool-call-parser qwen3_xml` | **Required for tool calls.** Qwen3.8 emits XML, not JSON. Without it `tool_calls` is always empty — which looks like a broken quant but is a serving flag. |
| `--reasoning-parser qwen3` | **Required for reasoning.** Without it the closing `</think>` is dropped as a special token and reasoning arrives glued onto the answer. With it, reasoning goes to `message.reasoning` — that name, not `reasoning_content`. |
| `--kv-cache-dtype fp8_e4m3` | Buys the context: halves the cache to 32 KiB/token, worth ~+55K tokens. Needs FlashInfer — on a 3090 the other attention backends refuse 8-bit KV. |
| `--speculative-config … qwen3_5_mtp` | The trained MTP draft head ships **inside** the checkpoint — no second file. **~1.7× decode** (75 vs 45 tok/s). Use `num_speculative_tokens: 3` on a 3090 (79 vs 74 tok/s); `2` was the optimum on Blackwell (128 tok/s, 1.71×, 71.2% acceptance). Re-tune per card. |
| `--language-model-only` | Skips the vision tower, worth ~25K tokens of context. Drop it if you need image input. |
| `--max-num-seqs 2` | Each concurrent request costs ~148 MiB of recurrent state on top of KV cache. Raising it to 8 costs ~30K tokens of context. |
| `--gpu-memory-utilization 0.95` | Don't raise it. 0.98 measures 131K of context and then dies on the first request. |

**Reasoning needs a 4096 budget.** Reasoning and the answer share `max_tokens`, and
reasoning is spent first — there is no separate thinking budget. A hard problem spent 2,714
tokens thinking; at 1–2K it returns an *empty* answer with `finish_reason: length`, which
looks like a broken model but is only the budget running out. Set effort per request with
`{"chat_template_kwargs": {"reasoning_effort": "low"}}` (`xhigh`/`high`/`medium`/`low`), or
`enable_thinking: false` — which measured best for tool calling (0.563 vs 0.437 at `xhigh`).
Tool calls cost only ~22 reasoning tokens either way.

<details>
<summary><b>Setup notes</b> — chat template, cold-start caching, FlashInfer</summary>

**Chat template.** Pass `--chat-template chat_template_safe_v2.jinja` (bundled). The stock
template raises on the OpenAI-standard `reasoning_effort: "high"`, so a normal client gets
HTTP 400. The safe template fixes that plus three rendering bugs and is byte-identical on
382/382 real holdout prefixes, so adopting it cannot change quality.

**The first start after changing any flag reports a smaller KV cache.** vLLM compiles the
model on a cold cache and still holds that memory while measuring what is free, so you get
~2.2 GiB instead of ~4.3 GiB — often too little to start. The second launch is correct.
`saved AOT compiled` in the log means cold (restart it); `Directly load AOT` means the
numbers are real.

**FlashInfer compiles kernels on first use**, and a stock `pip install vllm` cannot: there
is no `CUDA_HOME` (the PyTorch wheels ship a CUDA toolkit but do not put it on `PATH`), the
bundled compiler and CUDA headers can be different versions, and the linker wants `lib64`,
`libcudart.so` and `stubs/libcuda.so`, which the wheels do not create. Install `ninja`, pin
`nvidia-cuda-nvcc`/`crt`/`nvvm` to match your PyTorch CUDA version, and symlink those three
paths.

**Unrelated but expensive:** `pkill -f "vllm serve"` kills your own shell, because the
shell's command line contains the pattern too. Use `pkill -f "[v]llm serve"`.

</details>

---

## License

Apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

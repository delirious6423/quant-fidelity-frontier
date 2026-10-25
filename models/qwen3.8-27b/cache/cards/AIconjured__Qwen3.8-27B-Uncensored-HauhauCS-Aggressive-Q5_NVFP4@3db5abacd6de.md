---
license: apache-2.0
tags:
  - uncensored
  - qwen3.8
  - gguf
  - multimodal
  - vision
  - mtp
  - speculative-decoding
  - fastmtp
  - nvfp4
  - blackwell
  - imatrix
  - quantized
language:
  - en
  - zh
  - multilingual
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_NVFP4

> An **imatrix-optimized NVFP4** requantization of the HauhauCS *Aggressive* uncensored Qwen3.8-27B — **14.74 GiB / 4.63 BPW**, down from the 18.83 GiB / 5.92 BPW Q5_K_P source, with the native MTP/NextN head, vision, and the full Aggressive uncensoring profile preserved.

Built for **Blackwell (sm_120)** GPUs, where NVFP4 is natively accelerated, but the file is a standard GGUF and loads on any current llama.cpp build.

---

## Credits

This release builds on the work of two upstream projects:

- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — the original 27B dense multimodal model by the Qwen team, released under Apache 2.0.
- **Uncensoring / tuning:** [HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF) — the *Aggressive* uncensored variant and its K_P quant lineup, which is the exact source this file was requantized from.

The uncensoring and tuning are **baked into the weights** of the source model. Requantization changes only the numerical precision of those weights, not their values — so the *Aggressive* profile (direct answers, no refusal behavior, minimal preamble) and every text, reasoning, agentic, image, and video capability of the base model are retained.

---

## What this is

This is a **single-text-GGUF requantization** of the HauhauCS `Q5_K_P` file. The goal was to push the bulk of the model into **NVFP4** (4-bit floating point) as far as possible, using an **importance matrix (imatrix)** to keep the 4-bit float quantization high-quality, while deliberately keeping the numerically-sensitive tensors in higher precision.

The result is a model that is **~22% smaller** than the Q5_K_P source while staying on the same Blackwell-native 4-bit path, and which now imatrix-optimizes the **LM head** (`output.weight`) — something the original source did not do.

| | Source (Q5_K_P) | This release (Q5_NVFP4) |
|---|---:|---:|
| Size | 18.83 GiB (20,218,177,664 B) | **14.74 GiB (15,828,474,240 B)** |
| BPW | 5.92 | **4.63** |
| Bulk tensor type | Q5_K / Q6_K / Q4_K | **NVFP4** (imatrix) |
| `output.weight` (LM head) | Q6_K, no imatrix | **NVFP4, imatrix-optimized** |
| MTP / NextN head | preserved | preserved |
| Vision | separate BF16 projector | separate BF16 projector |
| Aggressive uncensoring | yes | yes (unchanged) |

---

## Specs

- Dense 27B causal language model with a vision encoder
- 64 language-model layers (+ 1 MTP/NextN layer)
- Hidden size 5,120; FFN size 17,408
- 248,320-token padded vocabulary
- 48 Gated DeltaNet (SSM) layers and 16 gated-attention layers
- Native embedded MTP/NextN head preserved (`nextn_predict_layers = 1`)
- 262,144-token native context (extensible per framework configuration)
- Native text, image, and video understanding
- Architecture `qwen35`; embedding length 5,120; 24 heads / 4 KV heads

---

## How it was made

### 1. Source

The starting point is the HauhauCS `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf` (18.83 GiB, 5.92 BPW, 866 tensors). Its tensor distribution was:

| Type | Count |
|---|---:|
| f32 | 360 |
| q8_0 | 18 |
| q4_K | 48 |
| q5_K | 275 |
| q6_K | 165 |
| **Total** | **866** |

### 2. Importance matrix (imatrix)

An imatrix was generated with `llama-imatrix` on a ~21.5k-word, 144,617-byte general-knowledge calibration corpus (encyclopedic prose spanning quantum computing, neural networks, computing and cooking history, music theory, the U.S. Constitution, evolution, economics, machine learning, and more). It was computed CPU-only (`-ngl 0`, 24 threads, `n_ctx=512`, 56 chunks), yielding a calibration perplexity of **4.7824 ± 0.0866**.

The imatrix was then **regenerated with `--process-output`** so that it also captures the **LM head** (`output.weight`). The resulting imatrix contains **994 entries** (992 layer tensors + `output.weight.in_sum2` / `output.weight.counts`); **497** are consumed during quantization.

> **Why this matters:** the imatrix records per-row activation statistics, which the quantizer uses to choose NVFP4 scale factors that minimize error on the values that actually matter. Without it, `output.weight` would be converted to NVFP4 with only plain per-block scaling. With it, the LM head — which produces the next-token logits — is quantized with the least distortion, directly improving output quality.
>
> **One unavoidable limit:** `token_embd.weight` (the input embedding) is *never* tracked by the imatrix collector, because its input is one-hot token IDs and there is no meaningful per-row activation to measure. It is therefore NVFP4 with plain per-block scaling — the best available for that tensor.

### 3. Tensor-type recipe

The following recipe was applied (first match wins, top to bottom), driving the bulk to NVFP4 while protecting the sensitive tensors:

```
blk\.64\.attn_(q|k|v|output)\.weight=f16
blk\.\d+\.attn_k\.weight=q4_k
blk\.\d+\.attn_output\.weight=q8_0
blk\.\d+\.ssm_beta\.weight=q8_0
blk\.\d+\.ssm_alpha\.weight=q8_0
blk\.\d+\.nextn\.eh_proj\.weight=q8_0
.*norm.*=f32
blk\.\d+\.ssm_a=f32
blk\.\d+\.ssm_dt\.bias=f32
blk\.\d+\.ssm_conv1d\.weight=f32
.*=nvfp4
```

| Rule | Target | Rationale |
|---|---|---|
| `.*norm.*=f32` | all LayerNorm / RMSNorm weights | Normalization layers are extremely quantization-sensitive; keep exact. |
| `blk.N.ssm_a`, `ssm_dt.bias`, `ssm_conv1d.weight` = `f32` | Gated DeltaNet (SSM) state parameters | The SSM state dynamics are numerically delicate; F32 avoids drift/instability. |
| `blk.64.attn_{q,k,v,output}` = `f16` | final (MTP) attention block | Keep the last attention layer high-precision for the NextN head. |
| `blk.N.attn_k` = `q4_k` | attention-key projections | Keys tolerate 4-bit well; small, low-impact tensors. |
| `blk.N.attn_output`, `ssm_alpha`, `ssm_beta` = `q8_0` | attention output + SSM gating | 8-bit keeps these accurate without much size cost. |
| `blk.64.nextn.eh_proj` = `q8_0` | MTP/NextN hidden-to-embedding projection | Protect the speculative head's projection quality. |
| `.*=nvfp4` | everything else (the bulk) | attn qkv/q/v/gate, ffn gate/up/down, ssm_out, `token_embd.weight`, `output.weight` → NVFP4 (imatrix where available). |

The quantization was run with `--allow-requantize` (required because `output.weight` is Q6_K in the source) and `--imatrix` pointing at the regenerated imatrix.

### 4. Resulting tensor distribution

| Type | Count | Covers |
|---|---:|---|
| `nvfp4` | **373** | bulk: attn qkv/q/v/gate, ffn gate/up/down, ssm_out, `token_embd.weight`, `output.weight` |
| `f32` | **360** | all norms + SSM state parameters (unchanged from source) |
| `q8_0` | **113** | attn_output, ssm_alpha, ssm_beta, `nextn.eh_proj` |
| `q4_K` | **16** | attention-key projections |
| `f16` | **4** | final-layer attention block (q/k/v/output) |
| **Total** | **866** | |

Vision is **not** in this file — it ships as the separate BF16 projector (see below).

---

## Quality impact vs. the source

- **Same model, same behavior.** Requantization is lossy compression of the *identical* weights. The Aggressive uncensoring, the chat template, the tokenizer, and all capabilities are unchanged.
- **Smaller, Blackwell-native.** Moving the bulk from Q5_K/Q6_K to NVFP4 cuts size by ~4.1 GiB (~22%) and puts the dominant matmuls on the native FP4 tensor-core path on sm_120 hardware.
- **Imatrix-optimized LM head.** `output.weight` is now NVFP4 *with* imatrix scaling (the source had it as Q6_K with no imatrix), reducing distortion in the next-token logits.
- **Sensitivity-aware.** Norms and the Gated DeltaNet state stay F32, and the MTP head's projection stays Q8_0, so the parts of the model most prone to quantization drift are protected.
- **MTP preserved.** The embedded NextN head is intact, so standard embedded-MTP speculative decoding (`--spec-type draft-mtp`) works, and the HauhauCS FastMTP sidecar (below) can be paired with it.

The trade-off: NVFP4 is a 4-bit float format, so the bulk carries more quantization error than the source's Q5_K/Q6_K. The imatrix and the F32/Q8_0 protection are what keep that error low where it matters. For maximum-fidelity serving, the source Q5_K_P (or Q6_K_P / Q8_K_P) remains the reference; this release targets the best size/speed/quality point on Blackwell.

---

## Files

| File | Contents | Size | SHA-256 |
|---|---|---:|---|
| `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_NVFP4.gguf` | text model (NVFP4, imatrix, MTP embedded) | 14.74 GiB | `1ed0752e92a4e5445a4cd1b9d1d35114fb611fb51c5b70f0db878fb551a9adaa` |
| `mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf` | vision projector (BF16) | 888 MiB | `5681b690bcb8eb10cd28d62d078cb4e01521a3ea4880a3fc7d54de72de2dd142` |
| `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf` | HauhauCS FastMTP draft sidecar | 903 MiB | `115e618e1f73cb50817ed5856f0551c6bf9c3d94df96f440eaca78dc63b8968b` |

The projector and the FastMTP sidecar are byte-for-byte identical to the HauhauCS release and work with this text quant. Download the projector only if you need image or video input; the FastMTP sidecar is only needed for the HauhauCS acceleration path (below).

> **Hugging Face's Hardware Compatibility widget may not recognize the NVFP4 quant.** If files appear to be missing, click **View variants** or open **Files and versions**.

---

## Usage

### Ollama

This model is published on Ollama as:

```
aiconjured/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_NVFP4
```

Pull and run it directly:

```bash
ollama run aiconjured/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_NVFP4 "Your prompt here"
```

The Ollama model bundles the text model **and** the BF16 vision projector, so image input works out of the box. It ships with `num_ctx 32768`, `num_predict 8192`, `repeat_penalty 1`, `temperature 0.7`, `top_p 0.9`.

To build it from the raw GGUF files instead, use a Modelfile with the two `FROM` lines (text model + projector) and the same parameters:

```
FROM ./Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_NVFP4.gguf
FROM ./mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf
PARAMETER num_ctx 32768
PARAMETER num_predict 8192
PARAMETER repeat_penalty 1
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

```bash
ollama create my-qwen3.8-q5-nvfp4 -f Modelfile
```

### llama.cpp (text + embedded MTP)

The text GGUF is standard and loads in any current Qwen3.8-capable llama.cpp build. For vision, pass the projector with `--mmproj`:

```bash
./build/bin/llama-server \
  --model Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_NVFP4.gguf \
  --mmproj mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf \
  --n-gpu-layers all \
  --ctx-size 32768 \
  --jinja
```

To enable **embedded MTP** speculative decoding (no sidecar needed), add `--spec-type draft-mtp`:

```bash
./build/bin/llama-server \
  --model Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_NVFP4.gguf \
  --mmproj mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf \
  --spec-type draft-mtp \
  --n-gpu-layers all \
  --ctx-size 32768 \
  --jinja
```

### HauhauCS FastMTP (maximum acceleration)

For the full HauhauCS FastMTP speedup, pair the text model with the `FastMTP-32K` draft sidecar and the HauhauCS runtime patch. Build the patched llama.cpp once, then serve:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout 4df29be4f4c3673f428170fda944a5b19f743bb8

curl -L -o HauhauCS-FastMTP-llama.cpp.patch \
  https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/HauhauCS-FastMTP-llama.cpp.patch
git apply HauhauCS-FastMTP-llama.cpp.patch

cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
```

```bash
MODEL=Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_NVFP4.gguf
DRAFT=Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf
DEPTH=3

CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  --model "$MODEL" \
  --mmproj mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf \
  --spec-draft-model "$DRAFT" \
  --spec-draft-ngl all \
  --spec-type draft-mtp \
  --spec-draft-n-max "$DEPTH" \
  --spec-draft-p-min 0 \
  --ctx-size 32768 \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --n-gpu-layers all \
  --split-mode none \
  --flash-attn on \
  --no-mmap \
  --temp 1.0 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0 \
  --presence-penalty 0 \
  --repeat-penalty 1.0 \
  --jinja \
  --reasoning on \
  --reasoning-effort xhigh \
  --reasoning-preserve \
  --reasoning-format deepseek \
  --host 127.0.0.1 \
  --port 8080
```

If draft loading reports `expected 5120, 248320, got 5120, 32768`, the sidecar is correct but the executable is unpatched — launch the freshly built `./build/bin/llama-server` from this checkout.

The HauhauCS FastMTP benchmark ladder (measured on the Q8_K_P reference) applies equally to this quant: **up to 3.02× document and 1.93× reasoning throughput versus MTP disabled**, and up to 35.2% / 21.1% faster than standard embedded MTP. The unchanged full target verifies every drafted token, so FastMTP accelerates generation without changing the model's answers.

---

## Recommended settings

From the [official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B):

**Thinking mode (default):**

- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=0.0`
- `repetition_penalty=1.0`
- `reasoning_effort=xhigh` for the deepest reasoning

**Instruct / non-thinking mode:**

- `temperature=0.7`
- `top_p=0.80`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`
- `enable_thinking=false`

**Important:**

- Use `--jinja` for the embedded chat template.
- Use the BF16 projector for Vision.
- The model's native maximum context is `262144`.
- Context length and KV precision have a large VRAM cost. Reduce context before reducing model quality if your workload does not need maximum native context.
- Keep default F16 K/V on the lower tiers unless memory pressure requires otherwise.

### Turning thinking off

Qwen3.8 uses thinking mode by default. Disable it for shorter, faster direct responses:

```bash
--chat-template-kwargs '{"enable_thinking":false}'
```

Or per request through the OpenAI-compatible API:

```json
{
  "model": "qwen3.8-27b-aggressive-q5-nvfp4",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

For multi-turn agents, preserve prior reasoning context with `{"preserve_thinking": true}`.

---

## Compatibility

- **llama.cpp:** recommended; use a current Qwen3.8/MTP-capable build
- **LM Studio, Jan, KoboldCpp, and other GGUF frontends:** base compatibility depends on their bundled llama.cpp version
- **Ollama:** works out of the box (text + vision); see the Ollama name above
- **Embedded MTP:** optional and stock-compatible in current llama.cpp
- **HauhauCS FastMTP:** optional; requires the sidecar and [HauhauCS-FastMTP-llama.cpp.patch](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/HauhauCS-FastMTP-llama.cpp.patch)
- **Vision:** requires the separate BF16 projector
- **NVFP4:** best on Blackwell (sm_120) GPUs; loads and runs on any current GGUF runtime

---

## Authenticity

The text model's exact file SHA-256 is `1ed0752e92a4e5445a4cd1b9d1d35114fb611fb51c5b70f0db878fb551a9adaa`. The FastMTP sidecar's exact file SHA-256 is `115e618e1f73cb50817ed5856f0551c6bf9c3d94df96f440eaca78dc63b8968b` (identical to the HauhauCS release; its canonical tensor fingerprint is `49e248e799f169b6ccc6a8127b9300a95f06cf3d96a8353266f5d457e81d1c87`).

This is an independent requantization of the HauhauCS source; it is not part of the signed HauhauCS release manifest. Verify the SHA-256 of the downloaded text file against the value above after renaming or mirroring.

---

## Reproducing this requant

To reproduce the exact file from the HauhauCS `Q5_K_P` source:

```bash
# 1. Generate the imatrix (CPU-only, with the LM head)
llama-imatrix \
  -m Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf \
  -f calib.txt \
  -o imatrix.gguf \
  --process-output \
  -ngl 0 -t 24 -c 512 --batch-size 512

# 2. Requantize with the tensor-type recipe
llama-quantize \
  --allow-requantize \
  --imatrix imatrix.gguf \
  --tensor-type-file recipe.txt \
  Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf \
  Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_NVFP4.gguf \
  Q4_K_M 16
```

where `recipe.txt` is the tensor-type recipe in section 3 and `calib.txt` is the ~144 KB general-knowledge calibration corpus.

---

## Other models

- [HauhauCS models on Hugging Face](https://huggingface.co/HauhauCS/models)
- [Qwen3.8-27B on Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B)

---

Qwen3.8-27B is released by Qwen under the Apache 2.0 license. This requantized NVFP4 variant retains that license.

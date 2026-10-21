---
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - nvfp4
  - modelopt
  - abliterated
  - uncensored
  - moe
  - mamba
  - speculative-decoding
---

# Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash-GGUF

GGUF build of [`pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash`](https://huggingface.co/pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash),
for use with **llama.cpp**.

The 4-bit weights are **repacked bit-exact** from the NVFP4 checkpoint into `GGML_TYPE_NVFP4` —
they are *not* dequantized and re-quantized, so there is no double-quantization penalty.
On Blackwell GPUs llama.cpp runs these through **native FP4 tensor cores**.

**19.5 GB**, plus a 772 MB DFlash draft model for speculative decoding.
Runs on 2×16 GB consumer GPUs (tested on 2× RTX 5070 Ti).

> ⚠️ **Text-only.** No vision tower, no MTP head (the abliteration was done on a
> language-model-only export). Converted with `--no-mtp`.

> ⚠️ **Uncensored.** Safety refusal behaviour has been deliberately removed. You are responsible
> for how you use it.

---

## 1. How the abliteration was done

Classic **refusal-direction ablation** (orthogonalization), single direction:

| Step | Detail |
|---|---|
| Base | `ornith-ai/Ornith-1.5-35B-A3B` (BF16) |
| Probe layer | **24** — `int(num_layers × 0.6)`, 40 layers total |
| Samples | 64 harmful + 64 harmless prompts (`random.seed(0)`), last-token hidden state |
| Direction | `d = normalize(mean(harmful) − mean(harmless))` |
| Ablation | For every `.o_proj` and `.down_proj`: `W ← W − outer(d, dᵀW)` |

Tooling derived from
[`remove-refusals-with-transformers`](https://github.com/Sumandora/remove-refusals-with-transformers).
BF16 weights: [`pottokao/Ornith-1.5-35B-A3B-abliterated`](https://huggingface.co/pottokao/Ornith-1.5-35B-A3B-abliterated).

## 2. How the quantization was done

NVIDIA **TensorRT Model Optimizer 0.45.0**, per-layer recipe matched exactly to the official
[`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)
(verified tensor-by-tensor: `weight_scale_2` 30841, `input_scale` 130, 291 quantized layers, 0 diff
in the language model). Calibration: 64 × 512 tokens from `abisee/cnn_dailymail`.

| Module | HF checkpoint | → GGUF |
|---|---|---|
| `mlp.experts` (256/layer), `mlp.shared_expert`, `lm_head` | NVFP4 W4A16, group 16 | **`GGML_TYPE_NVFP4` ×241, bit-exact** |
| `linear_attn.{out,in_qkv,in_z}`, `self_attn.{q,k,v,o}` | FP8 W8A8 | **`Q8_0` ×130** (GGML has no FP8 type) |
| embeddings | BF16 | `BF16` ×61 |

Conversion (latest `llama.cpp`, which has a ModelOpt-aware branch):

```bash
python3 convert_hf_to_gguf.py /path/to/NVFP4-model \
  --outfile Ornith-1.5-35B-A3B-abliterated-NVFP4.gguf --fp8-as-q8 --no-mtp

# DFlash draft (needs the target model for its tokenizer + dflash_config)
python3 convert_hf_to_gguf.py /path/to/NVFP4-model/dflash_draft \
  --target-model-dir /path/to/NVFP4-model --outfile dflash-draft-Ornith15.gguf
```

`--no-mtp` is **required** for this checkpoint: the config still declares
`mtp_num_hidden_layers: 1` but the MTP weights were stripped during abliteration, so without it the
converter writes `block_count: 41` and loading fails with `blk.40.attn_norm.weight not found`.

---

## 3. Running it

Build with CUDA for your arch (`120` = Blackwell / RTX 50-series). **This matters**: the native FP4
path is gated on `blackwell_mma_available()`, which checks the *compiled* arch.

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j --target llama-server
```

```bash
./build/bin/llama-server \
  -m Ornith-1.5-35B-A3B-abliterated-NVFP4.gguf \
  -md dflash-draft-Ornith15.gguf --spec-draft-n-max 8 \
  -ngl 99 -ngld 99 --split-mode layer -c 8192 -fa on \
  --host 0.0.0.0 --port 8080
```

llama.cpp auto-detects the draft type from the GGUF metadata:

```
common_specu: auto-detected speculative type 'draft-dflash' from the draft model metadata
  - n_max=8, n_min=0, block_size=16, mask_token_id=248077, sample_from_anchor=true
```

### Multi-GPU: we settled on `layer` (pipeline)

| `--split-mode` | What we observed on CUDA |
|---|---|
| `layer` (pipeline) | fastest in our tests — what the numbers below use |
| `tensor` (TP, EXPERIMENTAL) | loads and generates fine, but slower in every category we measured |
| `row` | not supported — the CUDA backend has no split-buffer implementation, so it refuses to load |

**Caveat: this is one configuration, not a verdict on tensor-parallel.** We are not familiar with
llama.cpp's TP path and only tested 2× RTX 5070 Ti over PCIe (no NVLink) at `-c 8192`. TP works —
it is not broken — it just did not win here, plausibly because single-stream decode is
bandwidth-bound and the per-layer all-reduce costs more than the parallelism gains on this
interconnect. Building with `-DGGML_CUDA_NCCL=ON` recovered ~5 % over the internal AllReduce but
still did not beat `layer`. We also hit
`llama_params_fit is not implemented for SPLIT_MODE_TENSOR`, so memory has to be sized by hand.

If there is a better way to configure TP here, a newer build that changes this, or something we
simply missed — **corrections and suggestions are very welcome.**

---

## 4. Benchmarks

2× RTX 5070 Ti (16 GB, 250 W), DFlash K=8, `-c 8192`, `-fa on`.

### 4.1 Spec-Bench suite (8 prompts/category, concurrency 1)

| Category | `layer` tok/s | `tensor` (TP) tok/s | `layer` TTFT | `tensor` TTFT |
|---|---|---|---|---|
| math_reasoning | **238.5** | 198.5 | **133 ms** | 183 ms |
| code (held-out) | **179.0** | 152.4 | **220 ms** | 270 ms |
| summarization | **170.4** | 153.0 | **270 ms** | 327 ms |
| rag | **159.1** | 140.2 | **354 ms** | 414 ms |

`rag` and `summarization` carry 3.1–3.4 K-character prompts, so TTFT is a meaningful share of the
work — and in our runs TP was 17–21 % slower there too, i.e. we could not find a prefill-heavy case
where it came out ahead. Again: one configuration, and we may well be holding it wrong.

### 4.2 DFlash acceptance (K=8)

Across the full Spec-Bench run above (40 recorded generations):

| metric | value |
|---|---|
| overall acceptance | 34.1 % (7428 accepted / 21788 drafted) |
| mean accepted length | **3.88** (range 2.38 – 6.07) |

The spread matters more than the average. On ad-hoc single prompts we measured anywhere from
**5.60** (step-by-step arithmetic) down to **3.07** (free-form Chinese prose) — structured output
drafts very well, free-form prose drafts poorly. **Always report the prompt mix alongside an
acceptance number**; the same model and settings can look 2× better or worse depending on what you
feed it.

### 4.3 Compared with the vLLM (NVFP4) build

Same benchmark suite, same 8 prompts per category, concurrency 1:

| Category | vLLM + NVFP4 + DFlash K=8 | llama.cpp GGUF + DFlash 8 (`layer`) |
|---|---|---|
| math_reasoning | **429.6** | 238.5 |
| code | **295.6** | 179.0 |
| rag | **319.1** | 159.1 |
| summarization | **317.2** | 170.4 |

**vLLM is 1.6–2.0× faster.** This is worth stating plainly, because llama.cpp is not being held back
by either of the two things you might suspect:

- **Quantization format**: llama.cpp runs these weights through **native FP4 tensor cores**, while
  vLLM falls back to Marlin for W4A16 on sm120. If anything llama.cpp has the advantage here.
- **Draft quality**: on the *same* benchmark suite, mean accepted length is **3.88 (llama.cpp) vs
  3.69 (vLLM)** — essentially equivalent, with llama.cpp marginally ahead.

So the gap is in per-step execution of the MoE + hybrid-Mamba forward itself, not in the
quantization format or the speculative decoding.

Pick this build if you want the llama.cpp runtime/ecosystem; pick the
[vLLM one](https://huggingface.co/pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash) for raw speed.
Note the GGUF is 19.5 GB — the same size as the source — so it still does **not** fit on a single 16 GB card.

---

## 5. Quality sanity check

An AIME 2026 run (**29/30**) was done on the source NVFP4 checkpoint under vLLM; since this GGUF is a
bit-exact repack of the same 4-bit weights, it is not re-reported here. See
[the NVFP4 repo](https://huggingface.co/pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash#5-aime-2026--sanity-check-only)
for the numbers and the caveats — in short, it is a check that abliteration + quantization did not
cause catastrophic degradation, **not** a capability claim, and there is no external baseline to
compare against.

---

## 6. Provenance

```
ornith-ai/Ornith-1.5-35B-A3B  (BF16, MIT)
  └── refusal-direction ablation (layer 24, o_proj + down_proj)
        └── pottokao/Ornith-1.5-35B-A3B-abliterated              (BF16, 65 GB)
              └── modelopt 0.45.0 NVFP4, recipe matched to official
                    └── pottokao/…-NVFP4-DFlash                  (20 GB, vLLM)
                          └── convert_hf_to_gguf.py (bit-exact NVFP4 repack)
                                └── this repo                    (19.5 GB, llama.cpp)
```

DFlash draft: [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash),
converted to GGUF unmodified; original weights and license belong to z-lab.

---
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: transformers
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
pipeline_tag: text-generation
tags:
  - lfm2
  - lfm2_moe
  - moe
  - w8a8
  - w8a16
  - int8
  - compressed-tensors
  - llm-compressor
  - vllm
  - ampere
  - quantized
  - 8-bit
---

# LFM2.5-24B-A2B — Selective W8A8/W8A16 INT8 (`W8A8-EXPERTS`, Path 4)

**Main (`W8A8-EXPERTS`, 2026-10-18):** mean teacher-forced full-vocab KLD vs BF16 **0.1527** — lower (better) than the official [`LiquidAI/LFM2.5-24B-A2B-FP8`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-FP8) on the same harness (**0.1818**, floor-free paired t = -36.0) — with **82.91% top-1 agreement** vs BF16 (FP8: **81.18%**), **28.88 PPL** (BF16: 29.32, FP8: 29.29), and **+8.2% to +8.6% faster decode throughput** than official FP8 on Ampere SM86 (`200.4 tok/s` vs `184.5 tok/s` at 5.5K context; `176.9 tok/s` at 50.6K context).

All **7,296 routed MoE expert Linears** (**95.7% of model parameters**, 22.95B) run **W8A8 INT8×INT8** on Ampere tensor cores — `compressed-tensors` method `CompressedTensorsW8A8Int8MoEMethod`, executed by vLLM's **Triton Int8 MoE kernel** (`TritonExperts`). The **40 full-attention projections** (0.4% of parameters) and **6 early dense MLP projections in layers 0–1** (0.6% of parameters) run **per-channel W8A16 INT8** (`CompressedTensorsWNA16`, executed by the **AllSpark** linear kernel) with BF16 activations, eliminating **80.1% of the INT8 activation-quantization KLD penalty** while giving up **0%** of the W8A8 decode speedup.

Companion checkpoints from the same compression study:
- **`W8A16-CHANNEL` (Path 3, weight-only isolation probe):** all 7,342 Linears at per-channel INT8 weights + BF16 activations. Scores **0.1369 KLD / 84.05% top-1 / 29.04 PPL** and runs ~1% faster decode than official FP8 on Ampere.
- **`W8A8` (Path 1, uniform all-Linear W8A8):** all 7,342 Linears at W8A8 INT8. Scores **0.2161 KLD / 79.72% top-1 / 29.21 PPL** at +6.9% decode over FP8.

No calibration-fitted weight distortion (no GPTQ — pure data-free symmetric per-channel RTN across all weights so no calibration-domain bias is baked into the checkpoint).

---

## License

**LFM Open License v1.0** (`lfm1.0`) — this checkpoint is an unofficial quantized derivative of [`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B). Model licenses ship in the model repos on Hugging Face (canonical copy [here](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)). Not affiliated with or endorsed by Liquid AI.

The one restriction worth knowing before you download: **commercial use is licensed only under \$10M annual revenue** (LFM License §5) — that threshold travels with these weights to every downloader.

---

## Results (same harness for every row)

### 1. Statistical parity — teacher-forced full-vocab KLD vs BF16

BF16 reference ([`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)) captured offline via `transformers`; all candidate checkpoints served through **vLLM `v0.29.0`** (`TP=2` on 2× RTX 3090 Ti / RTX 3090, Ampere SM86) so dynamic INT8 activation rounding and vLLM kernel numerics are measured end-to-end. Evaluated across **240 × 512-token WikiText-2 test windows = 122,640 token positions**, top-K 512 (`vocab_size=128000`). "Floor-free" is the paired intersection estimator (`kld-paired-floorfree.py`, adapted from [`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8)) that removes the top-K tail-truncation floor.

| Checkpoint | Precision split (7,342 quantized Linears) | Mean KLD ↓ | Median KLD ↓ | Paired Δ vs FP8 (floor-free) | Top-1 agree ↑ | PPL ↓ (BF16 29.32) | `q` floor hits |
|---|---|---:|---:|---:|---:|---:|---:|
| **Path 3: `W8A16-CHANNEL`** | 7,342 W8A16 (per-channel INT8 weight-only) | **0.1369** | **0.0652** | −0.0364 (t = -57.9) | **84.05%** | **29.04** | 15.63% |
| **Path 4: `W8A8-EXPERTS` (this)** | **7,296 experts W8A8 + 46 attn/dense W8A16** | **0.1527** | **0.0749** | **−0.0236 (t = -36.0)** | **82.91%** | **28.88** | 16.58% |
| [`LiquidAI/LFM2.5-24B-A2B-FP8`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-FP8) | Official FP8 (weight-only W8A16 on Ampere) | 0.1818 | 0.0954 | baseline | 81.18% | 29.29 | 18.27% |
| **Path 1: `W8A8`** | 7,342 W8A8 (experts + attn + dense 0–1) | 0.2161 | 0.1180 | +0.0265 (t = +36.8) | 79.72% | 29.21 | 19.83% |
| Path 2: `SQ-W8A8` (pre-fix, retired) | SmoothQuant + uniform W8A8 (unscaled router bug) | 0.7042 | — | — | 65.10% | 36.72 | — |

- **Paired Path 4 (`W8A8-EXPERTS`) vs official FP8:** Δ = -0.0236 ± 0.00066 (t = -36.0, −16.7% floor-free KLD), Path 4 closer to BF16 at **60.99%** of token positions.
- **Paired Path 4 (`W8A8-EXPERTS`) vs Path 1 (`W8A8`):** Δ = -0.0509 ± 0.00071 (t = -71.9, −30.3% floor-free KLD), Path 4 closer to BF16 at **68.46%** of token positions.

### 2. Speed & long-context agentic extraction (2× RTX 3090 Ti / 3090, TP2, vLLM `v0.29.0`)

Measured from a separate client host over LAN (`extract_test.py`, sequential single-stream, `temperature=0`, `--enable-thinking`) across four real arXiv HTML papers spanning **5.5K to 50.6K prompt tokens**:

| Checkpoint | `2607.15866v1` (5,525 tok) | `2606.16140v1` (20,888 tok) | `2603.28052v1` (24,105 tok) | `2610.22111v1` (50,572 tok) | vs FP8 decode |
|---|---|---:|---:|---:|---:|
| `W8A16-CHANNEL` | 185.9 tok/s / 757 ms | 176.5 tok/s / 2,820 ms | 174.1 tok/s / 3,163 ms | 164.6 tok/s / 7,090 ms | ~+0.7% |
| **`W8A8-EXPERTS`** (shipped) | **200.4 tok/s / 661 ms** | **189.9 tok/s / 2,919 ms** | **187.8 tok/s / 2,805 ms** | **176.9 tok/s / 6,322 ms** | **+8.2 to +8.6%** |
| LiquidAI FP8 (vendor) | 184.5 tok/s / 778 ms | 175.0 tok/s / 2,923 ms | 173.4 tok/s / 3,281 ms | 163.5 tok/s / 7,335 ms | baseline |
| `W8A8` | 197.2 tok/s / 653 ms | 187.4 tok/s / 2,884 ms | 185.1 tok/s / 2,752 ms | 174.5 tok/s / 6,234 ms | ~+6.9% |

*(Note: Path 3 vs Path 4 is a back-to-back comparison showing the pure +7.5% to +7.9% decode and −10.8% 50.6K-prefill gain from moving the 7,296 experts onto W8A8 INT8 tensor cores.)*

---

## Where the INT8 activation error lives on LFM2.5-24B-A2B (why this recipe)

`LFM2.5-24B-A2B` has a highly asymmetric parameter vs token-flow distribution across its 40 layers:

| Component class | Layers | Linear modules | Parameters | Share of model | Tokens routed through module |Scheme in `W8A8-EXPERTS` |
|---|---:|---:|---:|---:|---|---|
| **Routed MoE experts (`w1`, `w3`, `w2`)** | 2–39 (38) | **7,296** | **22.95B** | **95.7%** | Sparse (top-4 of 64 experts) | **W8A8** (`int-quantized`, dynamic token INT8) |
| **Dense MLPs (`w1`, `w3`, `w2`, inter 11,776)** | 0–1 (2) | **6** | 0.145B | 0.6% | **100% of tokens** (shapes initial residual stream) | **W8A16** (`pack-quantized`, per-channel INT8) |
| **Full-attention (`q`, `k`, `v`, `out_proj`)** | 2, 6, …, 38 (10) | **40** | 0.105B | 0.4% | **100% of tokens** | **W8A16** (`pack-quantized`, per-channel INT8) |
| **Gated short-conv (`conv`, `in_proj`, `out_proj`)** | other 30 | 60 (+30 Conv1d) | 0.504B | 2.1% | 100% of tokens | **BF16** (kept unquantized per Liquid AI recipe) |
| **MoE routers (`feed_forward.gate`) + `expert_bias`** | 2–39 (38) | 38 | 0.005B | <0.1% | 100% of tokens | **BF16 / FP32** (kept unquantized) |
| **Tied `embed_tokens` + RMSNorms** | all | — | 0.262B | 1.1% | 100% of tokens | **BF16** (kept unquantized) |

### The 4-path engineering progression

1. **Path 1 (`W8A8` uniform RTN):** Quantizing all 7,342 eligible Linears to W8A8 gave ~+7% faster decode than official FP8 and slightly better WikiText PPL (29.21 vs 29.29), but trailed FP8 by +0.034 KLD (`0.2161` vs `0.1818`) and −1.46 pt top-1 (`79.72%` vs `81.18%`).
2. **Path 2 (SmoothQuant + E12 investigation):** Following the dense-model component-precision study in [`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8), we first tested **E12** (keeping `down_proj` / `w2` at W8A16 while putting `w1/w3` at W8A8). On an MoE model, **vLLM `0.29.0` rejects within-expert mixed schemes at startup** (`CompressedTensorsMoEMethod.get_moe_method` enforces a single uniform scheme across `w1`, `w3`, and `w2` inside `RoutedExperts`). Furthermore, a uniform SmoothQuant build revealed a critical MoE fold pitfall: folding `ffn_norm` into expert `w1/w3` without *also* scaling the router's (`feed_forward.gate`) input columns feeds `x / s_a` into the router and scrambles top-4 expert selection on 62% of tokens (fixed and pinned in `tests/test_fold_plan.py`). Even with exact router folding, re-rounding folded router weights to BF16 introduces near-tie routing flips that limit SmoothQuant's net MoE block gain.
3. **Path 3 (`W8A16-CHANNEL` weight-only isolation probe):** Because FP8 on Ampere runs weight-only (BF16 activations, no INT8 compute path), we built a weight-only probe with **Path 1's exact per-channel INT8 weights** (verified bit-identical across sampled tensors; not the stock `compressed-tensors` `W8A16` preset, which defaults to `group_size=128` and would flatter weight fidelity). Path 3 scored **0.1369 KLD / 84.05% top-1 / 29.04 PPL**, proving two things:
   - **Symmetric per-channel INT8 weights beat FP8 e4m3 weights** by **−0.045 KLD** and **+2.87 pt top-1** (8-bit integer grid vs e4m3's 3 mantissa bits).
   - **100% of Path 1's gap to FP8 was dynamic INT8 activation quantization** (+0.0792 KLD / −4.33 pt top-1).
4. **Path 4 (`W8A8-EXPERTS` selective component split):** Instead of splitting *inside* an expert (which vLLM's `FusedMoE` forbids), we split **between dense/attention modules and routed experts**. Keeping just the **46 dense/attention Linears** (layers 0–1 dense MLPs + 10 attention blocks = 1.0% of parameters) at W8A16 while moving all **7,296 MoE expert Linears** (95.7% of parameters) to W8A8 eliminated **80.1% of the activation KLD penalty** (`0.2161 → 0.1527`, recovering `0.0634` of `0.0792` nats) and retained the **entire W8A8 tensor-core speedup**.

---

## Recipe & vLLM 0.29.0 Compatibility Fixes

Built with `scripts/quantize_lfm2_rtn.py --scheme W8A8-EXPERTS` (`llm-compressor 0.13`, `compressed-tensors 0.18.0`, `transformers 5.15.1`):

1. **Two `compressed-tensors` `config_groups` (`format: "mixed-precision"`):**
   - `group_0` (`format: "pack-quantized"`, W8A16): targets `self_attn.{q,k,v,out}_proj` and dense `layers.{0,1}.feed_forward.w{1,2,3}` (46 Linears). Symmetric per-channel INT8 weights (`strategy="channel"`, `group_size=-1`), `input_activations=None`.
   - `group_1` (`format: "int-quantized"`, W8A8): targets `feed_forward.experts.*.{w1,w2,w3}` (7,296 Linears). Symmetric per-channel INT8 weights, dynamic per-token symmetric INT8 input activations (`strategy="token"`, `dynamic=True`).
2. **Keep-outs (`IGNORE_RTN`, mirroring `LiquidAI/LFM2.5-24B-A2B-FP8`):**
   - `re:.*\.gate$` (all 38 MoE routers stay BF16; note LFM2 names the MoE block `feed_forward.gate`, not `mlp.gate`)
   - `re:.*\.(?:conv|short_conv)\..*` (all 30 gated short-convolution blocks, including `in_proj` and `out_proj`, stay BF16)
   - `re:.*embed_tokens$`, `re:.*lm_head`, all RMSNorms, `q_layernorm`/`k_layernorm`, and FP32 `expert_bias`.
3. **Three required post-processing steps for LFM2.5 MoE + vLLM `0.29.0` (automated in `quantize_lfm2_rtn.py`):**
   - **Concrete class load (`Lfm2MoeForCausalLM`):** `AutoModelForCausalLM` demotes `config.json` and causes vLLM to reject the checkpoint.
   - **Expert tensor name restoration (`restore_expert_names`):** `llm-compressor`'s `LinearExperts2D` linearizes unfused 2D experts (`2D -> 3D -> 2D`) and saves them as `gate_proj / up_proj / down_proj`. Both `transformers` and vLLM's `Lfm2MoeSparseMoeBlock` (`ckpt_names=("w1", "w2", "w3")`) expect `.w1 / .w3 / .w2`; all 14,592 expert weight and scale tensors are renamed in-place in the safetensors headers and index.
   - **Channel-wise `group_size: -1` normalization (`set_channel_group_size`):** `compressed-tensors` writes `group_size: null` for channel-wise weights, which crashes vLLM `0.29.0` when `WNA16` MoE/Marlin checks `group_size <= 0`. Rewriting `group_size: -1` in `config.json` fixes this cleanly.

---

## Serving (vLLM ≥ 0.29.0, Ampere SM86 or newer)

Kernel routing on Ampere (`SM86`), as selected by vLLM `v0.29.0` at load time. Note that the `compressed-tensors` **method class** that vLLM picks is not the same thing as the **GEMM kernel** that actually executes:

| Modules | `compressed-tensors` method | Executing kernel |
|---|---|---|
| Routed MoE experts — 7,296 Linears, W8A8 | `CompressedTensorsW8A8Int8MoEMethod` | **Triton Int8 MoE** (`TritonExperts`) |
| Attention `qkv_proj`/`out_proj` + layers 0–1 dense `w13`/`w2` — 46 Linears, W8A16 | `CompressedTensorsWNA16` | **AllSpark** (`AllSparkLinearKernel`) |
| Attention backend (all layers) | — | **FlashAttention 2** |

```bash
vllm serve Don-oz/LFM2.5-24B-A2B-W8A8-EXPERTS \
  --served-model-name LFM2.5-24B-A2B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.94 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 6 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --enable-auto-tool-choice \
  --tool-call-parser lfm2 \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

Prefer compose? The exact serve from our benchmark is [`docker-compose.lfm-w8a8-experts.yml`](https://github.com/ozdok/quant-lab/blob/c9f5269c01f12ad8f7e8b4a9a46d0f16ee8a659d/lfm-w8a8/eval/composes/docker-compose.lfm-w8a8-experts.yml) (pinned to `c9f5269`; set `MODEL_WEIGHTS_DIR` / `MODEL_CACHE_DIR`, see `.env.example`).

**On `--reasoning-parser qwen3` and `enable_thinking`:** this model *always* thinks — its chat template unconditionally opens `<|im_start|>assistant\n`, and it has no `enable_thinking` variable to toggle. The thinking trace is split into `message.reasoning` purely by the server-side reasoning parser, which reads `chat_template_kwargs.enable_thinking` and **defaults it to `True`** (`vllm/parser/qwen3.py`). So `--default-chat-template-kwargs '{"enable_thinking": true}'` is inert here (it restates the default) and is kept only so serve flags stay identical across arms in our benchmark; passing `false` would be the observable direction (it disables the split, leaving the trace in `content` and reporting `reasoning_tokens: 0`). Keep `--reasoning-parser qwen3` — that flag is load-bearing.

**VRAM:** ~24 GB on disk (2 shards). On 2× 24 GB GPUs (RTX 3090 / 3090 Ti / 4090) at `TP=2`, weights take ~12 GB/rank, leaving comfortable room for long-context KV cache up to 131K tokens.

---

## Reproduction

To rebuild this checkpoint from BF16 source, see [`ozdok/quant-lab`](https://github.com/ozdok/quant-lab/tree/c9f5269c01f12ad8f7e8b4a9a46d0f16ee8a659d) (pinned to `c9f5269`) — full pipeline, tests, and benchmark harness live there. Terminology used in this card — the four Paths, the smoothing folds, alpha, and the measurement and floor conventions — is defined in [`docs/GLOSSARY.md`](https://github.com/ozdok/quant-lab/blob/c9f5269c01f12ad8f7e8b4a9a46d0f16ee8a659d/lfm-w8a8/docs/GLOSSARY.md), pinned to the same commit.

---

## References & Credits

- **[`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8)** — primary inspiration for our compression methodology ("Where the INT8 activation error lives"), the `compressed-tensors` mixed-precision `config_groups` structure (`int-quantized` W8A8 + `pack-quantized` W8A16), the E12 (`down_proj` at W8A16) experiment, and the teacher-forced full-vocab KLD + floor-free paired evaluation harness (`kld-fullvocab.py` and `kld-paired-floorfree.py`).
- **[`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)** & **[`LiquidAI/LFM2.5-24B-A2B-FP8`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-FP8)** — Liquid AI for the base BF16 model, the official FP8 reference checkpoint, and the published keep-out recipe (`conv.*`, `feed_forward.gate`, `embed_tokens`, `lm_head`).
- **vLLM & `llm-compressor` / `compressed-tensors` teams** — for the MoE INT8 tensor-core and WNA16 kernels and quantization lifecycle tooling.

---
license: mit
license_name: mit
license_link: LICENSE
library_name: transformers
pipeline_tag: image-text-to-text
base_model: zai-org/GLM-5.3-Flash
tags:
- quantized
- mxfp8
- compressed-tensors
- auto-round
- vllm
- glm5_next
- moe
- multimodal
---

# GLM-5.3-Flash-MXFP8-CT-AutoRound

MXFP8 (OCP Microscaling 8-bit: E4M3 values + one E8M0 scale per 32 elements) **W8A8** quantized
checkpoint of [**GLM-5.3-Flash**](https://huggingface.co/zai-org/GLM-5.3-Flash) — a 321 B-parameter,
~18 B-active natively multimodal MoE model — produced with
[Intel AutoRound](https://github.com/intel/auto-round) in **model-free RTN** mode (no calibration
dataset, no model load), exported as **compressed-tensors** (`format: mxfp8-quantized`).

**Quantization scope in one sentence: all MoE weight matrices are MXFP8; everything else is BF16.**
On the four-task evaluation protocol below the checkpoint is indistinguishable from the BF16
reference: **AVG 0.8411 vs 0.8399 (+0.12 pp), every per-task delta positive and ≤ 0.35 σ.**

## 1. Model summary

| | Base model | This checkpoint |
|---|---|---|
| Base | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (official block-FP8) / `zai-org/GLM-5.3-Flash-BF16` (quantization input) | `intel-ai/GLM-5.3-Flash-MXFP8-CT-AutoRound` |
| Architecture | `Glm5NextForConditionalGeneration` (`model_type: glm5_next`): hybrid 34× KDA linear attention + 11× sparse-MLA (DSA) layers (+1 MLA in the MTP layer), 288-expert MoE, mHC hyper-connections, 1M context | identical |
| Parameters | 321.323 B total | 321.323 B (weights re-encoded) |
| Quantization | block-FP8 (E4M3, 128×128 blocks) | **W8A8 MXFP8**, `group_size=32`, symmetric, dynamic activations |
| Format | safetensors (native FP8) | safetensors, `compressed-tensors` / `mxfp8-quantized` |
| Size on disk | 328.33 GB (official FP8) / 642.65 GB (BF16 source) | **339.39 GB / 316.08 GiB**, 120 shards |
| Compression vs BF16 | 1.96× (official FP8) | **1.89×** |
| Effective bits/param | 16.0 (BF16 source) | 8.45 on disk (8.25 over the quantized span) |
| License | MIT (Z.AI Co., Ltd) | same (see License) |

Validated serving configuration: **TP=4 on 4× NVIDIA B300 (sm103)** with a nightly vLLM build (see
Usage).

## 2. Quantization scope — what is and is not quantized

Of the **37 862** 2-D Linear layers in the model, **37 287 (97.4 % of Linear parameters) are MXFP8**
and **575 stay BF16**. The full machine-readable scope is `quantization_config.json` (`ignore`,
679 entries).

**Quantized to MXFP8 (W8A8):**

| Modules | Layer count | Params |
|---|---:|---:|
| `mlp.experts.{gate,up,down}_proj` (routed experts) | 43 layers × 288 experts × 3 = 37 152 | 311.7 B |
| `mlp.shared_experts.{gate,up,down}_proj` | 43 × 3 = 129 | 1.08 B |
| dense `mlp.{gate,up,down}_proj`, layers 1–2 | 6 | 0.30 B |
| **Total MXFP8** | **37 287** | **313.04 B** |

**Not quantized (BF16 / FP32, as shipped):**

| Modules | Count | Reason |
|---|---:|---|
| KDA linear-attention `self_attn.*` (34 layers) | 408 | kept raw in the official checkpoint too; vLLM builds the KDA path without quantization support |
| sparse-MLA `self_attn.{q_a,q_b,kv_a_proj_with_mqa,o_proj}` (12 layers) | 48 | **intentional — see §3** |
| sparse-MLA `self_attn.kv_b_proj` | 12 | BF16 in the official FP8 checkpoint as well |
| DSA `indexer.*` | 36 | selects which KV pages are read; 0.09 B params, nothing to gain |
| MoE router `mlp.gate` | 43 | FP32 sigmoid routing; quantizing it perturbs expert selection directly |
| `visual.*` (vision tower) | 126 | not exercised by the text serving path; loads BF16 either way |
| `embed_tokens` / `lm_head` / MTP `eh_proj` | 3 | 0.63 B each; plain `nn.Linear` in vLLM |
| layer-0 dense MLP | 3 | see §3.2 |
| hyper-connections, RMSNorm, `A_log`/`dt_bias`, MTP structural pieces | — | non-Linear parameters (BF16/F32) |

## 3. Why the scope differs from the official FP8 checkpoint

The official `zai-org/GLM-5.3-Flash` ships its **sparse-MLA attention projections** (`q_a_proj`,
`q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` — 48 layers) and the **layer-0 dense MLP** (3 layers) as
block-FP8; this checkpoint keeps those **51 layers in BF16**. That is the entire scope difference —
routed experts, shared experts, routers, indexer, KDA, vision and embeddings match the official
scope exactly.

### 3.1 The decisive reason: vLLM cannot serve MXFP8 sparse-MLA attention

vLLM's `glm5_next` loader **dequantizes the sparse-MLA projections back to BF16 at load time**
(`_try_load_fp8_attn_proj`), and its scale-geometry assumption is hard-coded to block-FP8
(`[⌈N/128⌉, K/128]`). An MXFP8 `[N, K/32]` scale tensor fails that assumption and the engine dies
during weight loading:

```
RuntimeError: The size of tensor a (512) must match the size of tensor b (16384)
```

This was measured on a full-official-scope sibling checkpoint (all 37 338 layers MXFP8): it crashes
in vLLM at ~5 % of shard loading. Even if the loader were fixed, those modules end up **BF16 on the
device anyway** — so quantizing them saves neither memory nor compute. Leaving the 48 MLA
projections in BF16 is what makes this checkpoint loadable, and it is a deliberate property of the
shipped artifact.

### 3.2 How the artifact came out this way (produced with stock AutoRound)

The quantization run was launched with the full official scope in mind, but AutoRound registers a
predefined, non-removable ignore list for `model_type=glm5_next` containing a bare **`self_attn`**
substring and **`layers.0.mlp`**. Those two entries silently overrode the requested scope
(37 287 layers instead of the audited 37 338, with no warning); the run log line
`Using predefined ignore_layers from config: indexer, layers.0.mlp, self_attn, weights_proj`
records it. Given §3.1, the resulting scope turned out to be exactly the one vLLM can serve, so it
is the scope we validate and ship. The cost is +1.3 GB of BF16 weights versus the full-scope plan.

### 3.3 Side-by-side with the official FP8 checkpoint

| | Official `zai-org/GLM-5.3-Flash` (FP8) | This checkpoint (MXFP8) |
|---|---|---|
| Quant format | FP8 E4M3, 128×128 blocks, F32 scales | MXFP8: E4M3 + E8M0 scale per 32 elements |
| Export format | native FP8 safetensors | compressed-tensors (`mxfp8-quantized`) |
| Quantized Linear layers | 37 338 | 37 287 |
| sparse-MLA attn projections | FP8 (dequantized to BF16 at load) | BF16 |
| dense MLP layers 0–2 | FP8 | MXFP8 for layers 1–2; layer 0 BF16 (§3.2) |
| `kv_b_proj` / KDA / indexer / router / vision / embed | BF16 | BF16 (identical) |
| Scale overhead | 0.077 GB | 9.78 GB (MX per-32-group tax) |
| Size on disk | 328.33 GB (1.96× vs BF16) | 339.39 GB (1.89× vs BF16) |
| vLLM serving | official path | validated (TP=4, B300) — see Usage |

Honest framing: **8-bit MX is not a size play on this model** — the per-32 E8M0 scales cost ~3 % of
weight bytes versus 0.024 % for 128×128 block-FP8, so this checkpoint is +11 GB over the official
FP8 one. Its value is the compressed-tensors/MXFP8 toolchain (AutoRound reproducibility, MX-format
compatibility), not compression. If size is the goal, use the 4-bit sibling recipe instead:
[`INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`](https://huggingface.co/INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound)
(≈183 GB, 3.5×, reported AVG 0.8366 on the same protocol).

## 4. Evaluation results

Harness `lm-eval` 0.4.13 with the vLLM backend, **TP=4 on 4× B300**, `seed=42`, `batch_size=32`,
text-only (`language_model_only=true`), thinking off. Scores for **this** checkpoint, measured
2026-09-21; the BF16 row is the reference table supplied for the base checkpoint on the same four
tasks.

| | GSM8K (strict) | MMLU | PIQA | HellaSwag | **AVG** |
|---|---|---|---|---|---|
| BF16 reference | 0.9735 | 0.8666 | 0.8292 | 0.6903 | **0.8399** |
| **MXFP8 (this checkpoint)** | **0.9742** | **0.8668** | **0.8313** | **0.6919** | **0.8411** |
| Δ | +0.07 pp | +0.02 pp | +0.21 pp | +0.16 pp | **+0.12 pp** |

All four deltas are positive and far inside one standard error — i.e. **no measurable loss on this
protocol** (not "lossless"). What this does **not** cover: long-context (evals ran at
`max_model_len=8192` on a 1M-native model), agentic/tool use, coding, reasoning-on decoding, and
throughput. The multimodal path is unvalidated (vision tower is untouched BF16 and would load, but
every number here is text-only).

## 5. Usage (vLLM)

```bash
vllm serve intel-ai/GLM-5.3-Flash-MXFP8-CT-AutoRound \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --language-model-only \
  --reasoning-parser glm45
```

Hard constraints for `glm5_next` (violating one fails at start-up):

- **Requires a nightly/recent vLLM** — `glm5_next` support landed via
  [vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906) (first tag
  v0.29.1rc0); vLLM 0.29.0 on PyPI cannot load this architecture at all. The DSA indexer also
  hard-requires **DeepGEMM** (not in the vLLM wheel; see vLLM's `tools/install_deepgemm.sh`).
- `--block-size` must be a multiple of `index_kpool × 32` = 128.
- TP must divide both head counts (64 attention heads, 64 KDA heads); pipeline parallelism is
  gated off for this architecture.
- `dtype=bfloat16` is mandatory (hyper-connection scratch is BF16-hardcoded).
- KV cache: `auto` / `fp8` / `fp8_e4m3` are fine; **`fp8_ds_mla` / `nvfp4_ds_mla` are not** (those
  packing layouts assert `pe_dim == 64` and GLM-5.3 is NoPE).
- Cap `--max-model-len` explicitly: the indexer decode workspace scales with
  `max_num_seqs × max_model_len`.
- No expert parallelism (the CUTLASS MX expert kernels require `ep_size == 1`); no speculative
  decoding on quantized checkpoints (MTP load-time false positive).

## 6. Reproduce

### 6.1 Quantization

```bash
auto-round \
  --model_name zai-org/GLM-5.3-Flash-BF16 \
  --scheme MXFP8 \
  --ignore_layers lm_head,embed_tokens,visual,hc_,indexer,mlp.gate.,eh_proj,enorm,hnorm,shared_head,norm,A_log,dt_bias,self_attn.b_proj,f_a_proj,f_b_proj,g_a_proj,g_b_proj,conv1d,self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.kv_b_proj \
  --format llm_compressor \
  --output_dir ./GLM-5.3-Flash-MXFP8-CT-AutoRound \
  --model_free
```

Notes for an exact scope match: AutoRound's predefined `glm5_next` ignore list (`indexer,
layers.0.mlp, self_attn, weights_proj`) is unioned in regardless of `--ignore_layers`, which is why
the sparse-MLA projections and the layer-0 MLP stay BF16 (§3.2). `mlp.gate.` keeps its trailing dot
(a bare `mlp.gate` would substring-match the dense `mlp.gate_proj`), and `self_attn.b_proj` must be
spelled in full (a bare `b_proj` would also match `q_b_proj`/`kv_b_proj`). The measured run took
~25 min, peaked at 5.5 GB host RAM (model-free mode streams the 642 GB source), and produced 120
shards.

### 6.2 Evaluation

The scores in §4 were produced with `lm-eval` 0.4.13 on the vLLM backend (TP=4, 4× B300,
`seed=42`), one engine per command. The `--model_args` must be a **single JSON object** (this
lm_eval build `json.loads` the first token and rejects comma-separated `k=v` strings containing
nested dicts):

```bash
MODEL_ARGS='{"pretrained": "intel-ai/GLM-5.3-Flash-MXFP8-CT-AutoRound",
  "tensor_parallel_size": 4, "max_model_len": 8192, "max_num_seqs": 256, "block_size": 128,
  "gpu_memory_utilization": 0.85, "dtype": "bfloat16", "trust_remote_code": true,
  "add_bos_token": true, "enable_prefix_caching": false, "max_gen_toks": 2048,
  "enable_thinking": false, "language_model_only": true, "reasoning_parser": "glm45"}'

# gsm8k: 5-shot generative, chat template + few-shot as multi-turn (~1 h)
lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks gsm8k \
  --batch_size 32 --seed 42 --apply_chat_template --fewshot_as_multiturn

# piqa / mmlu / hellaswag: 0-shot log-likelihood, no chat template (~35 min)
lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks piqa,mmlu,hellaswag \
  --batch_size 32 --seed 42
```

All engine constraints from §5 apply (nightly vLLM, DeepGEMM, `block_size=128`, bf16, no EP/spec
decode). The scores were measured **thinking-off**; Z.AI's published numbers for the base model are
thinking-on, so do not mix the two protocols.


## 7. Known limitations

- Four academic benchmarks only; no long-horizon/agentic/code/thinking-on evaluation, no throughput
  numbers. Absence of regression here is not evidence of parity on those workloads.
- Multimodal path unvalidated (`language_model_only=true` everywhere above).
- Slightly larger than the official FP8 checkpoint (+11 GB); for maximum compression use the MXFP4
  sibling linked in §3.3.
- Requires a nightly vLLM + DeepGEMM (see Usage); PyPI `vllm==0.29.0` cannot load this
  architecture.

## 8. License and attribution

Base model [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) is **MIT**
(Z.AI Co., Ltd); the `LICENSE` file in this repository applies to this derivative unchanged.

Quantization performed with [Intel AutoRound](https://github.com/intel/auto-round) (Apache-2.0) in
model-free RTN mode — **no calibration corpus was used**, so no dataset attribution applies. Serving
via [vLLM](https://github.com/vllm-project/vllm) with DeepGEMM / FlashInfer kernels; evaluation via
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). MXFP8 follows the
[OCP Microscaling Formats (MX) specification](https://opencompute.org): one E8M0 shared scale per
32-element block.

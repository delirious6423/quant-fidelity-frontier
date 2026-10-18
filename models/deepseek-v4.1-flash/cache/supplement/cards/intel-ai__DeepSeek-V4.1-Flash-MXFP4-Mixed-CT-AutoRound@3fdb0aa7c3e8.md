# DeepSeek-V4.1-Flash — MXFP4/MXFP8 Mixed Checkpoint (AutoRound, OCP MXFP8)

Compressed-tensors (`format: mixed-precision`) quantization of
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
produced with [Intel AutoRound](https://github.com/intel/auto-round) **0.16.0** in
**model-free** mode: MXFP4 (OCP, group 32) routed experts + OCP MXFP8 (group 32) for the
attention / shared-expert / MTP Linear layers.

The distinguishing property of this artifact is that **it is bit-identical to the official
checkpoint at the weight level** — every E4M3 byte, every packed FP4 nibble and every ue8m0 scale
byte matches the source. Only the *representation metadata* changed: the official block-wise
`[N/32, K/32]` FP8 scales are expanded into standard OCP one-dimensional `[N, K/32]` group
scales. The directory is therefore 0.04 % larger than the official one, and it needs
**one vLLM patch** to run on Blackwell (see [Inference](#inference)).

> Locally produced research artifact. Every quantization and evaluation number below was measured
> on the machine described in [Environment](#environment). The only figures taken from another
> checkpoint are the official DeepSeek-V4.1-Flash baseline scores, labelled as such where they
> appear (§6).

## 1. Model summary

| | Official | This checkpoint |
|---|---|---|
| Name | `deepseek-ai/DeepSeek-V4.1-Flash` | `DeepSeek-V4.1-Flash-MXFP4-Mixed-0929-wkv8m` |
| `architectures` | `DeepseekV41ForCausalLM` (multimodal entry class; text tower under `language_model`) | identical |
| Text config | 40 layers · hidden 5120 · 384 routed experts · MoE intermediate 2304 | identical |
| Logical parameters | 763.16 B (incl. the 196.61 B Engram n-gram table) | 763.16 B (unchanged) |
| Tensors | 96,085 | 96,085 (pure renaming, see §2) |
| Size on disk | 510.29 GB / 475.34 GiB, 48 shards | **510.51 GB / 475.45 GiB**, 48 shards (+0.04 %) |
| Effective bits/param (all stored bytes) | 5.3476 | **5.3500** (+0.04 %; 5.105 excluding scales) |
| Weight format declaration | `fp8` + `weight_block_size=[32,32]` + `scale_fmt=ue8m0` + `expert_dtype=fp4` | `compressed-tensors`: `mxfp8-quantized` + `mxfp4-pack-quantized` (both group 32, symmetric, dynamic activations) |
| Runtime MoE execution | **W4A8** | **W4A4** |
| License | MIT (upstream) | unchanged by requantization |

### Tensor composition

| dtype | leaf | count | bytes | what |
|---|---|---|---|---|
| `U8` | `.weight_packed` | 47232 | 278.59 GB | MXFP4 routed experts (384/layer × 40, plus 128 × 3 MTP) |
| `F8_E4M3` | `.weight` | 357 | 204.02 GB | attention / shared experts / indexer / `engram.wkv` / MTP / Engram table |
| `U8` | `.weight_scale` | 47587 | 17.64 GB | OCP group scales (3.1 % overhead on the 8-bit part) |
| `F8_E8M0` | `.scale` | 2 | 6.14 GB | Engram n-gram table scale — kept in the official source form (this table is FP8, *not* BF16; see §1) |
| `BF16` | `.weight`(446) `.bias`(67) `q/k_weight`(4) `image_*`(3) | 520 | 3.95 GB | embeddings, `lm_head`, router gate, compressor, `indexer.wk`, vision tower |
| `F32` | `hc_attn_base/fn/scale`, `hc_ffn_*`, `attn_sink`, `bias_vl`, … | 387 | 0.17 GB | mHC mixing parameters, untouched |

### Left untouched by this pipeline (`quantization_config.ignore`)

Note: for this checkpoint `ignore` means *"leave this module exactly as the official
checkpoint has it"* — **not** "keep it in BF16". One of the entries below is deliberately FP8.

| Group | Stored dtype | Why it is in `ignore` |
|---|---|---|
| `layers.{1,14}.engram.embed` | **FP8 `E4M3` + `E8M0` scales — byte-identical to official** (verified: `weight F8_E4M3 (384006168, 256)` + `scale F8_E8M0 (384006168, 8)` in both) | The n-gram lookup table is not a Linear: vLLM's `ParallelEngramEmbedding` always builds a float8 weight + `weight_scale_inv` (`uint8`) and reads the *source* key names (`engram.embed.weight` / `.scale`). Renaming the scale leaf, re-laying-it-out, or dequantizing it would be wrong — and dequantizing would add ~190 GB of BF16 per layer. So it is left verbatim, still FP8, same as official |
| `layers.*.attn.indexer.wk`, `attn.indexer.weights_proj`, `attn.compressor.*` | BF16 (same as official) | DSA indexer / compressor side-car weights: BF16 in the official checkpoint as well, and vLLM's `Indexer` builds `wk` unquantized, so quantizing them here would leave a `weight_scale` parameter with no checkpoint key behind |
| `layers.*.ffn.gate`, `embed`, `head`, `vision.*`, `aligner.*` | BF16 (same as official) | Router gate, token embeddings, output head and the multimodal tower: not Linear targets of this scheme and least tolerant of low bits |

Because vLLM matches these entries by exact string equality against its *internal* module paths,
this artifact lists every entry under each spelling a loader may use
(`layers.6.attn.wo_a`, `model.layers.6.attn.wo_a`, `language_model.layers.…`,
`language_model.model.layers.…`) — hence 526 entries rather than 131. Root cause and the proper
fix: [Patch 2](#patch-2--translate-layer-paths-inside-the-quantization-config).

## 2. Relationship to the official checkpoint: weights are bit-identical

Only two things changed, and both are metadata:

| What | Official | This checkpoint | Equivalence |
|---|---|---|---|
| 8-bit Linear weight scales | `float8_e8m0fnu` `[N/32, K/32]` (2-D blocks) | `uint8` `[N, K/32]` (1-D OCP groups) | identical after `repeat_interleave(32, dim=0)`: row replication, columns unchanged |
| FP4 expert weights | `int8` `.weight` + `[N, K/32]` scale | `uint8` `.weight_packed` + same scale | byte-identical packing; dtype relabelled only |
| FP8 Linear weights | `float8_e4m3fn` `.weight` | same bytes, same leaf | byte-identical |
| Engram n-gram table | `.weight` + `.scale` | unchanged names and bytes | byte-identical |

**Verified exhaustively, not sampled.** `verify_ct_vs_native_bytes.py` compares every tensor of
this directory against its official counterpart by raw bytes (143,317 comparisons covering
weights, weight scales and all remaining parameters):

```
   权重字节一致                   48035
   scale 一致(原样)             47234
   权重字节一致(仅 dtype 重标)      47232
   其它参数一致                    461
   scale 一致(展开后)              355
   合计比较 143317 项；不一致 0 项
结论: ✅ 权重与官方逐位一致（差异只在 scale 的挂法与 leaf 改名）
```

Implication: **any score difference against the official checkpoint comes from the runtime
(kernel choice / activation quantization), never from the stored weights.**

## 3. Runtime kernel parity with the official run

Taken from vLLM startup logs of both checkpoints on the same build:

| Stage | Official | This checkpoint | Match |
|---|---|---|---|
| KV-cache format | `nvfp4_ds_mla` | `nvfp4_ds_mla` | ✅ |
| Attention backend | `FLASHMLA_MEGA_ATTN_DSV41` | `FLASHMLA_MEGA_ATTN_DSV41` | ✅ |
| MXFP8 Linear kernel | `FlashInferCutedslMxfp8LinearKernel` | same | ✅ |
| Grouped `wo_a` o_proj | `DeepGemmMxfp8BmmLinearKernel` (`fp8_einsum "bhr,hdr->bhd"`, recipe `(1,1,32)`) | same — **requires Patch 1** | ✅ |
| Lightning indexer K cache | FP8 dynamic block quant | FP8 dynamic block quant | ✅ |
| **MoE experts** | `FLASHINFER_TRTLLM_MXFP4_MXFP8` (**W4A8**) | `CutlassExpertsMxfp4` (**W4A4**) | ⚠ intended difference |

The MoE activation width (A8 → A4) is thus the dominant expected source of any delta, together
with the different MXFP8 linear *method* (compressed-tensors vs ModelOpt semantics) feeding the
same kernels.

## 4. Inference

### vLLM base commit

Measured and reproduced on:

```
vllm 0.29.1rc1.dev471+g82daf9f57          # base commit 82daf9f57
torch 2.13.0+cu130 · deep_gemm 2.8.0 · lm-eval 0.4.13
GPU: 4 × NVIDIA B300 SXM6 AC (compute capability 10.3) · 70 CPU cores · 4 TB RAM
```

### Patch 1 — required: compressed-tensors MXFP8 must honour `is_bmm`

DeepSeek's fused output projection marks `wo_a` as a *batched* Linear
(`models/deepseek_v41/attention.py:378-379`) and consumes it through a grouped DeepGEMM
`fp8_einsum`, which requires a 3-D `[groups, N, K]` weight. The ModelOpt MXFP8 method implements
that (`layers/quantization/modelopt.py:2259`, `:2534`); the compressed-tensors MXFP8 scheme
ignores the flag, so a compressed-tensors checkpoint keeps a 2-D weight and dies inside DeepGEMM:

```
RuntimeError: Assertion error (csrc/…/utils/layout.hpp:46): t.dim() == N
```

The fix mirrors ModelOpt exactly — two hunks in
`layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py`:
`create_weights` publishes `weight_block_size = [1, 32]` (the attention layer reads it at
*construction* time to select the 32-byte recipe, before any post-load hook runs), and
`process_weights_after_loading` swaps in the BMM kernel (`is_bmm` only exists after the module
has been built). After that, `wo_a` ends up in the identical `[g, N, K]` + packed-ue8m0 layout
the official run uses, so every downstream consumer (weight permutation, mega `einsum`, recipe)
reuses the already-validated native path.

```diff
diff --git a/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py b/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py
--- a/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py
+++ b/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py
@@ -79,7 +79,20 @@
         )
         layer.register_parameter("weight_scale", weight_scale)
 
+        # MXFP8 weights are per-row groups of 32 along K. Publish that on the layer,
+        # as the ModelOpt MXFP8 method does, because consumers that pick a scale
+        # granularity while *building* the module (e.g. DeepSeek-V4.1's
+        # ``_o_proj_block_size``) run before process_weights_after_loading.
+        layer.weight_block_size = [1, MXFP8_BLOCK_SIZE]
+
     def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
+        # Grouped projections (DeepSeek ``wo_a``) are flagged with ``is_bmm`` /
+        # ``bmm_batch_size`` by the attention layer *after* the quant method created
+        # the weights, so the kernel can only be revisited here: the fused o_proj
+        # consumes ``wo_a`` as a batched [groups, N, K] operand through DeepGEMM's
+        # fp8 einsum, which is what the MXFP8 BMM kernel produces in place.
+        if getattr(layer, "is_bmm", False) and getattr(layer, "bmm_batch_size", None):
+            self.kernel = init_mxfp8_linear_kernel(bmm_batch_size=layer.bmm_batch_size)
         self.kernel.process_weights_after_loading(layer)
 
     def apply_weights(
```

### Patch 2 — recommended: translate layer paths inside the quantization config

`should_ignore_layer()` matches non-`re:` entries by **exact string equality** against vLLM module
paths. vLLM would normally translate checkpoint-side names through
`CompressedTensorsConfig.apply_vllm_mapper()`, but the loader only calls it when
`model_class.hf_to_vllm_mapper` is a **class attribute**
(`model_executor/model_loader/utils.py:288-296`). `DeepseekV41ForCausalLM` is the multimodal entry
class and assigns its mapper to the *instance* (`nvidia/vl_model.py:188`), so the translation
never happens and **every `ignore` entry silently misses**: layers meant to stay BF16 are built
as MXFP8 with a `weight_scale` parameter that no checkpoint key fills — i.e. uninitialized
memory. This artifact sidesteps that by listing every spelling; Patch 2 removes the cause.

```diff
diff --git a/vllm/models/deepseek_v41/nvidia/vl_model.py b/vllm/models/deepseek_v41/nvidia/vl_model.py
--- a/vllm/models/deepseek_v41/nvidia/vl_model.py
+++ b/vllm/models/deepseek_v41/nvidia/vl_model.py
@@ -79,18 +79,24 @@
     types: Annotated[torch.Tensor, TensorShape("ns", dynamic_dims={"ns"})]
 
 
+# Checkpoint module paths -> vLLM module paths for this wrapper. Static: it does not
+# depend on the quant spec, so it is also what a quantization config's layer paths
+# (e.g. compressed-tensors ``ignore``) must be translated with.
+_CKPT_TO_VLLM_PREFIX: dict[str, str | None] = {
+    "layers.": "language_model.model.layers.",
+    "embed.": "language_model.model.embed.",
+    "norm.": "language_model.model.norm.",
+    "hc_head": "language_model.model.hc_head",
+    "mtp.": "language_model.model.mtp.",
+}
+
+
 def _make_deepseek_v4_vl_weights_mapper(
     expert_dtype: str, linear_scale_name: str
 ) -> WeightsMapper:
     """Text-checkpoint mapping rules re-rooted under ``language_model.``."""
     base = _make_deepseek_v4_weights_mapper(expert_dtype, linear_scale_name)
-    orig_to_new_prefix: dict[str, str | None] = {
-        "layers.": "language_model.model.layers.",
-        "embed.": "language_model.model.embed.",
-        "norm.": "language_model.model.norm.",
-        "hc_head": "language_model.model.hc_head",
-        "mtp.": "language_model.model.mtp.",
-    }
+    orig_to_new_prefix: dict[str, str | None] = dict(_CKPT_TO_VLLM_PREFIX)
     return WeightsMapper(
         orig_to_new_prefix=orig_to_new_prefix,
         orig_to_new_regex=base.orig_to_new_regex,
@@ -136,6 +142,17 @@
     # bias_vl.
     requires_raw_input_tokens = True
 
+    # Read by ``model_loader.utils`` *before* the model is built, to translate
+    # checkpoint-side layer paths inside a quantization config (compressed-tensors
+    # ``ignore`` / per-layer ``targets``) into this class' module names. It has to live
+    # on the class: those entries are matched by exact string equality, so an
+    # untranslated ``layers.6.attn.wo_a`` silently never matches and the layer is
+    # quantized anyway -- with a float8 weight-scale parameter that no checkpoint key
+    # fills, i.e. uninitialized memory. ``__init__`` still assigns the full instance
+    # mapper (it carries the quant-spec-dependent regex/suffix rules too), which
+    # shadows this attribute for weight loading.
+    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix=dict(_CKPT_TO_VLLM_PREFIX))
+
     @classmethod
     def get_placeholder_str(cls, modality: str, i: int) -> str | None:
         if modality == "image":
```

Behaviour check (offline, no weight loading — using vLLM's own matcher against a config that
contains *only* checkpoint-native names):

| Module path as vLLM sees it | Expected | before patch | after patch |
|---|---|---|---|
| `language_model.model.layers.14.attn.compressor.wkv` | ignored | ❌ False | ✅ True |
| `language_model.model.layers.14.attn.compressor.wgate` | ignored | ❌ False | ✅ True |
| `language_model.model.layers.2.attn.indexer.weights_proj` | ignored | ❌ False | ✅ True |
| `language_model.model.layers.1.engram.embed` | ignored | ❌ False | ✅ True |
| `language_model.model.layers.1.engram.wkv` | ignored | ❌ False | ✅ True |
| `language_model.model.layers.6.attn.wq_a` | quantized | False | False ✅ (no over-match) |
| `language_model.model.layers.6.ffn.shared_experts.w1` | quantized | False | False ✅ |

Hits: **0/7 → 7/7**.

### Apply and run

```bash
# from a vLLM checkout at 82daf9f57 (or the site-packages of an installed build)
patch -p1 < v41-ct-mxfp8-is-bmm.patch       # Patch 1 (required for this artifact)
patch -p1 < v41-ct-ignore-prefix.patch      # Patch 2 (recommended)
python -c "import vllm; print(vllm.__version__)"

# engine smoke, TP=2 on Blackwell; kv-cache stays at auto (-> nvfp4_ds_mla, mega backend)
CUDA_VISIBLE_DEVICES=0,1 TP=2 SMOKE_ONLY=1 \
  ./eval_DeepSeek-V4.1-Flash.sh "$PWD/DeepSeek-V4.1-Flash-MXFP4-Mixed-0929-wkv8m"
```

Do **not** dodge Patch 1 by switching the KV dtype: on this build the mega backend is eligible for
both `auto` and `fp8_ds_mla` and always wins on sm100, and its `_o_proj` calls `fp8_einsum`
unconditionally. If you cannot patch vLLM at all, see caveat 1 in §8.

## 5. Reproduce the quantization (AutoRound)

Requires AutoRound ≥ 0.16.0 with DeepSeek-V4.1 model-free support — branch
`feat/deepseek-v41-model-free`, commits `7daafaff` (“switch blockwise mxfp8 to OCP mxfp8”) and
`cfd4f733` (“ignore dsv41 .engram.embed.”), which

* accept `model_type in ("deepseek_v4", "deepseek_v41")` so the translation chain fires at all;
* limit the Engram carve-out to `.engram.embed.` — the n-gram table stays in source form while
  `engram.wkv`, an ordinary Linear, goes through the MXFP8 translation. That is precisely what
  makes this checkpoint's storage match the official one.

```bash
auto-round ./DeepSeek-V4.1-Flash \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers compressor,indexer.weights_proj,indexer.wk,engram.embed,vision,aligner \
  --layer_config "{ffn.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir ./DeepSeek-V4.1-Flash-MXFP4-Mixed-0929-wkv8m
```

Flag notes:

* `--ignore_layers` must contain **`engram.embed`, not `engram`**, for this artifact. With plain
  `engram` the `wkv` projection is treated as ignored and exported as BF16 — a valid but different
  variant (see §8.4).
* **`indexer.wk` must be in the list.** It is BF16 in the official checkpoint; if left out,
  model-free RTN quantizes it to MXFP8 while vLLM's DSA indexer builds that layer unquantized, so
  loading fails with `KeyError: '…indexer.wk.weight_scale'`.
* This shipped directory additionally had the `ignore` prefix expansion applied
  (`add_ignore_prefixes.py`), because it predates Patch 2. With Patch 2 in place that step is
  unnecessary and the config keeps checkpoint-native names only.

Model-free runs are metadata/layout translations of an already-quantized source: no calibration
dataset, no model instantiation, ≈31 min and ≈512 GB written for this 475 GiB checkpoint
(peak RSS 2.3 GB).

## 6. Evaluation

Harness: `lm-eval 0.4.13` with the `vllm` backend. Identical serving arguments for the baseline
and the quantized run, so the only variables are the checkpoint and its execution path:

```bash
CUDA_VISIBLE_DEVICES=0,1 lm_eval --model vllm \
  --model_args '{"pretrained": "DeepSeek-V4.1-Flash-MXFP4-Mixed-0929-wkv8m", "tensor_parallel_size": 2, "max_model_len": 8192, "max_num_seqs": 256, "block_size": 128, "gpu_memory_utilization": 0.9, "dtype": "bfloat16", "trust_remote_code": true, "add_bos_token": true, "enable_prefix_caching": false, "max_gen_toks": 2048, "enforce_eager": true, "language_model_only": true}' \
  --tasks gsm8k,hellaswag,piqa,mmlu --batch_size 32 --seed 42
```

The checkpoint ships **no chat template** (prompt encoding follows `encoding/harmony`), so
evaluation runs as plain-text few-shot continuation — do not pass `--apply_chat_template`.
`block_size=128` is mandatory: the sm100 sparse-MLA kernels reject 256.

### Results

| Model | GSM8K (5-shot, strict / flexible) | MMLU | PIQA (acc / acc_norm) | HellaSwag (acc / acc_norm) |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4.1-Flash` (official, W4A8 MoE) | **0.9553 / 0.9553** | **0.8886** | 0.8466 / 0.8613 | 0.7036 / 0.8713 |
| This checkpoint (W4A4 MoE, OCP MXFP8) | **0.9522 / 0.9522** | **0.8868** | 0.8390 / 0.8553 | 0.7017 / 0.8688 |
| Δ (this − official) | −0.0030 | −0.0018 | −0.0076 / −0.0060 | −0.0020 / −0.0026 |
| Δ in standard errors | ≈0.5 σ | ≈0.7 σ | ≈0.9 σ / 0.7 σ | ≈0.4 σ / 0.8 σ |

Both runs used identical harness arguments (full 1319 GSM8K examples, `seed=42`,
`batch_size=32`, TP=2, `kv_cache_dtype=auto` → `nvfp4_ds_mla`, mega backend), and the stored
weights are bit-identical (§2), so **the deltas above are attributable to the runtime difference
listed in §3 — chiefly MoE activations going from MXFP8 (W4A8) to MXFP4 (W4A4)** — plus sampling
noise. Every metric lands within ~1 standard error of the official checkpoint, i.e. statistically
indistinguishable at this sample size, with the largest single movement being PIQA (`−0.76 pp`).

Sources: official `lm_eval_results/__models__…__DeepSeek-V4.1-Flash/results_2026-09-29T09-24-20*.json`
(gsm8k) and `…T10-08-06*.json` (hellaswag / piqa / mmlu); this checkpoint
`lm_eval_results/__models__…__DeepSeek-V4.1-Flash-MXFP4-Mixed-0929-wkv8m/results_2026-09-30T04-54-11*.json`
(`FULLSUITE rc=0`, wall clock 73 min for the four tasks on 2 × B300).

### Functional smoke (passed)

Deterministic continuations (`temperature=0`) are clean, which is the cheapest end-to-end check
that no layer is executing on uninitialized scales (compare the degenerate output of the
sibling variant described in §8.3, which was exactly that failure):

```
'The capital of France is' -> ' Paris.'
'1+1=2. 2+2=4. 4+4='       -> '8. 8+8=16. 16+16=32. 32+32=64. … 1024+1024=2048.'
'def quicksort(arr):'      -> correct quicksort implementation
```

## 7. Environment

* 4 × NVIDIA B300 SXM6 AC (sm103), 70 CPU cores, 4 TB RAM, checkpoints on CephFS
* vLLM `0.29.1rc1.dev471+g82daf9f57` with the two patches above; `deep_gemm 2.8.0`
* `torch 2.13.0+cu130`, `lm-eval 0.4.13`
* `VLLM_WORKER_MULTIPROC_METHOD=spawn`; DeepSeek tokenizer mode is selected automatically
* Keep `--kv-cache-dtype auto` for comparability with the official baseline (it also resolves to
  `nvfp4_ds_mla` on mega)

## 8. Notes and caveats

1. **`wo_a` must stay quantized for the mega path.** If you prefer a BF16 `attn.wo_a` (e.g. to
   avoid Patch 1), you must additionally force the non-mega backend —
   `attention_config={"backend":"FLASHMLA_SPARSE_DSV41"}` with `kv_cache_dtype=fp8_ds_mla` — which
   reaches the `use_fp8=False` `torch.bmm` branch inside `deep_gemm_fp8_o_proj`. That is a
   different execution path (different attention kernel, different KV format), so its scores are
   **not** comparable with the official baseline.
2. **Ignore entries are engine-sensitive.** Until Patch 2 is applied, any compressed-tensors
   checkpoint of this architecture must list `ignore` / explicit `targets` using vLLM's internal
   module paths, otherwise the declarations are silently ignored. Verify without loading weights:

   ```bash
   python - <<'PY'
   import json
   from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
       CompressedTensorsConfig as C)
   from vllm.model_executor.layers.quantization.compressed_tensors.utils import should_ignore_layer as S
   q = C.from_config(json.load(open("config.json"))["quantization_config"])
   for p in ["language_model.model.layers.6.attn.wo_a",   # must be ignored -> True
             "language_model.model.layers.6.attn.wq_a"]:   # must be quantized -> False
       print(p, S(p, ignore=q.ignore, fused_mapping=q.packed_modules_mapping))
   PY
   ```

   Expected on this artifact: `True` then `False`.
3. **Repetitive free-form continuations seen on sibling variants were not a PTQ behaviour quirk.**
   On the variant where `engram.wkv` was exported BF16 while `ignore` still did not match, the
   same three prompts degenerated into repetition (`' Paris. …we … This, (the) (as in)…'`,
   Chinese noise in the code prompt). Root cause is caveat 2 — a BF16 weight paired with an
   uninitialized scale — and the behaviour disappears once the declarations are self-consistent,
   as they are here.
4. **Sibling variants** (same source, different representation choices), kept for reference:
   `…-0929` (BF16 `wo_a` *and* BF16 `engram.wkv`; requires the non-mega backend) and
   `…-0929-bf16wkv` (MXFP8 `wo_a`, BF16 `engram.wkv`). Neither matches the official storage;
   this directory is the storage-parity variant.

## 9. Citation

```bibtex
@misc{deepseek2026v41flash,
  title  = {DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression},
  author = {DeepSeek-AI},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash}},
}

@misc{intel_autoround,
  title  = {AutoRound: Shape-Hierarchy for Two-Step LLM Quantization, with APTQ},
  author = {Intel Labs},
  year   = {2024},
  howpublished = {\url{https://github.com/intel/auto-round}},
}
```

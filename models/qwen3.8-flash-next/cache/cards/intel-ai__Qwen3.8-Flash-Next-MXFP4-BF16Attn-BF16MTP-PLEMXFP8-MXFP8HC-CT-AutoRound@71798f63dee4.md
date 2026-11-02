---
base_model:
- Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- auto-round
- mxfp4
- mxfp8
- compressed-tensors
- vllm
- qwen4_exp
- mtp-bf16
---

## Model Details

This model is a mixed-precision (MXFP4 / MXFP8) build of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) produced with
[Intel auto-round](https://github.com/intel/auto-round) RTN (`--model_free`), following the
recipe of
[intel-ai/Qwen3.8-Flash-Next-MXFP4-BF16Attn-PLEMXFP8-CT-AutoRound](https://huggingface.co/intel-ai/Qwen3.8-Flash-Next-MXFP4-BF16Attn-PLEMXFP8-CT-AutoRound)
(**MTP kept in BF16**) plus **MXFP8 quantization of the hyper-connection GEMMs**.
Please follow the license of the original model.

- **MXFP4 (W4A4, group 32, E8M0 scales)** — routed expert `gate_proj` / `up_proj` /
  `down_proj` of all 48 decoder layers (73 728 tensors). This is 67 % of the parameters and
  the reason the checkpoint is ~2.7x smaller than BF16.
- **MXFP8 (W8A8, group 32)** — the PLE n-gram embedding table (128 shards,
  `float8_e4m3fn` + UE8M0 per-32-column scales, 28 % of the parameters) and the **three
  hyper-connection GEMMs** (`input_mix_weight_down`, `input_mix_weight_up`,
  `block_inject_weight` — 294 tensors across the 48 decoder layers, the final mixer and the
  MTP block).
- **Kept in BF16** — the MTP draft layer, including its 512 routed experts (1 536 tensors);
  attention (`self_attn`, `linear_attn`); the QSA sparse indexer; the shared expert and its
  gate; the MoE router (`mlp.gate`); token embeddings; `lm_head`; the vision tower;
  `mtp.fc_*`; and all norms.
- **KV cache** — not quantized (`kv_cache_scheme: null`).

Hyper-connection runtime note: vLLM's `GatedResidual` builds these three projections with
`quant_config=None` and merges down+inject into one `MergedColumnParallelLinear` unless the
patch below is applied. The checkpoint stores the real FP8 codes and UE8M0 scales, so with
the patch they execute as MXFP8 GEMMs; without it they cannot be used and the projections are
skipped as unquantized BF16.

## How to Use

### Environment

```bash
# produced and validated with this stack:
#   vllm   0.32.1rc1.dev253+g30aded62d   torch 2.14.0+cu130
#   auto-round 0.16.1   compressed-tensors 0.17.0   lm-eval 0.4.13
uv pip install "vllm==0.32.1rc1.dev253"   # or a newer build carrying qwen4_exp
```

The MXFP8 n-gram table (PLE) needs a runtime with
`Qwen4ExpPLEMxfp8EmbeddingMethod` (`vllm/models/qwen4_exp/common/ngram_embedding.py`);
`0.32.1rc1.dev253` ships it.

Apply the hyper-connection quantization patch (required for this checkpoint, and for
`--speculative-config mtp` as well):

```bash
python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import subprocess

root = Path(find_spec("vllm").origin).parent.parent   # <venv>/lib/python3.*/site-packages
patch = "patches/vllm_qwen4exp_hc_mxfp8_linear.patch"
cmd = ["patch", "--batch", "--fuzz=0", "-p1", "-d", str(root), "-i", patch]
if subprocess.run(cmd + ["--reverse", "--dry-run"], capture_output=True).returncode == 0:
    print("already applied")
else:
    subprocess.run(cmd + ["--forward", "--dry-run"], check=True)
    subprocess.run(cmd + ["--forward"], check=True)
PY
```

### vLLM Usage

```bash
vllm serve /path/to/quantized_hc_fp8_mtp_bf16 \
    --tensor-parallel-size 1 \
    --language-model-only \
    --dtype bfloat16 \
    --reasoning-parser qwen3

# speculative decoding with the BF16 MTP draft layer
vllm serve /path/to/quantized_hc_fp8_mtp_bf16 \
    --tensor-parallel-size 1 \
    --language-model-only \
    --dtype bfloat16 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Deployment requirements / known constraints:

- **`--tensor-parallel-size 1` is required** as shipped. At TP>1 the MTP MXFP8 MoE hits a
  vLLM scale-swizzle defect (`RuntimeError: shape '[2560, 10]' is invalid for input of size
  30720`); this build ships the MTP experts as BF16, but TP=1 is still the validated topology.
  Memory is comfortable: the PLE table lives in pinned host memory, so only ~70 GiB of
  weights go to the GPU.
- On hosts where NVLink-SHARP multicast cannot be bound, export `NCCL_NVLS_ENABLE=0`.
- FlashInfer JIT needs a CUDA toolkit matching torch cu130; point `CUDA_HOME` at the CUDA 13
  toolkit shipped inside the venv if the system nvcc is older.
- The HC MXFP8 GEMMs go through FlashInfer **CuTeDSL**, which JIT-compiles per shape on first
  use (`CuTeDSL JIT compilation during inference` — a first-token latency spike). Warm up
  representative shapes, or set `VLLM_DISABLED_KERNELS=FlashInferCutedslMxfp8LinearKernel` to
  fall back to Cutlass.

## Evaluation

`lm_eval` + vLLM, TP=1, `enable_thinking=false`, `language_model_only=true`, seed 42.
PIQA / HellaSwag / MMLU use loglikelihood; GSM8K uses the chat template with 5-shot.

| Benchmark | This model (MTP = BF16) | BF16 base |
|---|---:|---:|
| PIQA (acc) | 82.26  | 81.94 |
| HellaSwag (acc) | 68.28  | 69.27 |
| MMLU (acc) | 86.63 | 86.52 |
| GSM8K (strict-match) | 96.66  | 96.74 |
| **Average** | **83.46** | **83.62** |

Note: the MTP layer is a separate draft model (`Qwen4ExpMTP`) that is only instantiated when
speculative decoding is enabled, so it does not participate in these numbers. These scores
therefore measure the main path with BF16 attention, MXFP4 experts, MXFP8 PLE and MXFP8
hyper-connection.

## Generate the Model

```bash
# 1) quantize (inherits the MXFP8 hyper-connection build)
OUT=<out>/quantized HC_SCHEME=mxfp8 bash scripts/qwen38fn_quant_hcfp8.sh
python scripts/qwen38fn_hc_to_true_fp8.py \
    <out>/quantized <out>/quantized/hc_mxfp8_true_fp8_sidecar.safetensors <out>/quantized_hc_fp8

# 2) restore the MTP expert projections to BF16 (this model)
python scripts/qwen38fn_mtp_to_bf16.py \
    <out>/quantized_hc_fp8 /path/to/Qwen3.8-Flash-Next <out>/quantized_hc_fp8_mtp_bf16
```

The compressed-tensors config: `group_0` = `mxfp4-pack-quantized` (73 728 explicit expert
targets + `RoutedExperts`), `group_1` = `mxfp8-quantized` (`Linear`, `RoutedExperts`, the PLE
n-gram module and the 294 hyper-connection modules), `ignore` = 2 185 entries (attention,
indexer, shared expert, router, embeddings, lm_head, vision tower, `mtp.fc`, norms, PLE
projections and the 1 536 MTP expert modules).

### Known vLLM issue: the MTP draft ignores compressed-tensors `ignore`

`mtp.*` is a separate draft model (`Qwen4ExpMTP`) that vLLM builds with shifted module
names: `mtp.layers.0.<x>` becomes `mtp.layers.<num_hidden_layers>.<x>` (i.e. `mtp.layers.48`
for this model). `_make_draft_vllm_config()` in
`vllm/models/qwen4_exp/nvidia/mtp.py` remaps `ignored_layers` / `exclude_modules` /
`quantized_layers` for that shift, but **not** `CompressedTensorsConfig.ignore` — the list that
`get_scheme_dict()` -> `should_ignore_layer()` actually consults. Consequence: every ignore
entry written against the checkpoint names (`mtp.layers.0....`) silently misses, so modules the
checkpoint stores in BF16 (MTP experts, `shared_expert`, attention projections) are allocated
as MXFP8 and then receive BF16 weights:

```
Attempted to load weight layers.0.mlp.shared_expert.down_proj.weight with dtype torch.bfloat16
    into parameter with dtype torch.float8_e4m3fn
Attempted to load weight o_proj.weight with dtype torch.bfloat16 into parameter with dtype
    torch.float8_e4m3fn
```

The **checkpoint's `config.json` is correct** (its `ignore` list carries the real checkpoint
names); the defect is purely draft-side in vLLM, and it affects any compressed-tensors
checkpoint with an MTP layer, including the reference recipe. Impact is limited to speculative
decoding: the target model verifies every draft token, so outputs stay correct, but the
draft's quality — and therefore the acceptance rate — degrades.

The patch shipped with this model (`patches/vllm_qwen4exp_hc_mxfp8_linear.patch`) fixes both
issues; its last hunk remaps `draft_quant_config.ignore` with the same helper. With the patch
the draft load is clean (0 dtype warnings, `SPEC SMOKE OK`, correct generations).

## Ethical Considerations and Limitations

The model can produce factually incorrect output, and should not be relied on to provide
factually accurate information. Because of the limitations of the pretrained model and the
finetuning datasets, it is possible that this model could generate lewd, biased or otherwise
offensive outputs. Therefore, before deploying any applications of this model, developers
should perform safety testing.

## Caveats and Recommendations

- MXFP4 weights with **dynamic MXFP4 activations** (W4A4) is the largest single source of
  quantization error in this build.
- The PLE n-gram table dominates deployment memory (51 B parameters, stored as MXFP8 and
  looked up from pinned host memory); the checkpoint cannot shrink much below ~123 GiB while
  it stays at 8 bits.
- MTP is kept in BF16 by design. It only affects speculative decoding and is not part of the
  benchmark table above. The draft path needs the **same vLLM patch** as the hyper-connection
  GEMMs, because of one additional vLLM defect (see below).
- Hyper-connection GEMMs are 0.37 % of the parameters: they change accuracy by less than
  run-to-run noise (±0.25 pt) and cost a CuTeDSL JIT warmup on vLLM.

Users (both direct and downstream) should be made aware that the risks of this model involve
hallucinations, biased or offensive content, and factual errors.

## Cite

```bibtex
@article{cheng2023optimize,
  title={Optimize weight rounding via signed gradient descent for the quantization of large language models},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```

## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the
actions of third parties using this model. Please consult an attorney if you need legal
advice about your use of this model, and follow the license of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

Signed-off-by: {member name} <{member email}>

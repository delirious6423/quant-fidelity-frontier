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
---

## Model Details

This model is a mixed-precision (MXFP4 / MXFP8) build of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with
[Intel auto-round](https://github.com/intel/auto-round) RTN (`--model_free`), following the
recipe of
[intel-ai/Qwen3.8-Flash-Next-MXFP4-BF16Attn-MXFP8MTP-PLEMXFP8-CT-AutoRound](https://huggingface.co/intel-ai/Qwen3.8-Flash-Next-MXFP4-BF16Attn-MXFP8MTP-PLEMXFP8-CT-AutoRound),
plus **MXFP8 quantization of the hyper-connection GEMMs**.
Please follow the license of the original model.

- **MXFP4 (W4A4, group 32, E8M0 scales)** — routed expert `gate_proj` / `up_proj` /
  `down_proj` of all 48 decoder layers (73 728 tensors). This is 67 % of the parameters and
  the reason the checkpoint is ~2.8x smaller than BF16.
- **MXFP8 (W8A8, group 32)** — MTP expert projections (1 536 tensors); the PLE n-gram
  embedding table (128 shards, `float8_e4m3fn` + UE8M0 per-32-column scales, 28 % of the
  parameters); and the **three hyper-connection GEMMs** (`input_mix_weight_down`,
  `input_mix_weight_up`, `block_inject_weight` — 294 tensors across the 48 decoder layers,
  the final mixer and the MTP block).
- **Kept in BF16** — attention (`self_attn`, `linear_attn`), the QSA sparse indexer, the
  shared expert and its gate, the MoE router (`mlp.gate`), token embeddings, `lm_head`, the
  vision tower, `mtp.fc_*`, and all norms / bias-free scalars.
- **KV cache** — not quantized (`kv_cache_scheme: null`).

Hyper-connection storage note: vLLM's `GatedResidual` builds the three projections with
`quant_config=None` and merges down+inject into a single `MergedColumnParallelLinear`, so a
stock runtime can only execute them as BF16 GEMMs. The checkpoint therefore stores the
*round-tripped* (quantize → dequantize) BF16 values for those 294 tensors, which reproduces
the MXFP8 quantization error exactly. The exact FP8 codes and UE8M0 scales are kept in
`hc_mxfp8_true_fp8_sidecar.safetensors`, and a **true-FP8 variant** of the checkpoint (with
the patch below) runs them as real MXFP8 GEMMs.

## How to Use

### Environment

```bash
# produced and validated with this stack (any build carrying Qwen4Exp + the MXFP8 PLE
# embedding method works):
#   vllm   0.32.1rc1.dev253+g30aded62d
#   torch  2.14.0+cu130
#   auto-round 0.16.1   compressed-tensors 0.17.0   lm-eval 0.4.13
uv pip install "vllm==0.32.1rc1.dev253"   # or a newer dev/nightly build with qwen4_exp
```

The MXFP8 n-gram table (PLE) needs a runtime that ships
`Qwen4ExpPLEMxfp8EmbeddingMethod` (`vllm/models/qwen4_exp/common/ngram_embedding.py`);
`0.32.1rc1.dev253` has it natively.

For the **true-FP8 hyper-connection variant** (`quantized_hc_fp8/`), apply the
hyper-connection quantization patch to the installed vLLM:

```bash
python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import subprocess

root = Path(find_spec("vllm").origin).parent.parent  # <venv>/lib/python3.*/site-packages
patch = "patches/vllm_qwen4exp_hc_mxfp8_linear.patch"   # shipped with this model / repo
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
# stock runtime — hyper-connection runs as BF16 GEMM
vllm serve /path/to/quantized \
    --tensor-parallel-size 1 \
    --language-model-only \
    --dtype bfloat16 \
    --reasoning-parser qwen3

# true-FP8 hyper-connection variant (with the patch above) — HC runs as MXFP8 GEMM
vllm serve /path/to/quantized_hc_fp8 \
    --tensor-parallel-size 1 \
    --language-model-only \
    --dtype bfloat16 \
    --reasoning-parser qwen3
```

Deployment requirements / known constraints:

- **`--tensor-parallel-size 1` is required** as shipped. At TP>1 the MTP MXFP8 MoE hits a
  vLLM scale-swizzle defect: `moe_intermediate_size / (TP * 32)` columns are padded to a
  multiple of 4 and then reshaped back to the unpadded shape
  (`RuntimeError: shape '[2560, 10]' is invalid for input of size 30720`). TP=1 is also
  comfortable memory-wise: the PLE table lives in pinned host memory
  (`weight_device=cpu, pinned=True`), so only ~70 GiB of weights go to the GPU.
- On hosts where NVLink-SHARP multicast cannot be bound, export `NCCL_NVLS_ENABLE=0`.
- FlashInfer JIT needs a CUDA toolkit matching torch cu130; point `CUDA_HOME` at the CUDA 13
  toolkit shipped inside the venv (`site-packages/nvidia/cu13`) if the system nvcc is older.
- With the patch, the HC MXFP8 GEMMs go through FlashInfer **CuTeDSL**, which JIT-compiles per
  shape at first use (`CuTeDSL JIT compilation during inference` — a first-token latency
  spike). Warm up representative shapes, or set
  `VLLM_DISABLED_KERNELS=FlashInferCutedslMxfp8LinearKernel` to fall back to Cutlass.

## Evaluation

Evaluated with `lm_eval` + vLLM, TP=1, `enable_thinking=false`, `language_model_only=true`,
seed 42. PIQA / HellaSwag / MMLU use loglikelihood; GSM8K uses the chat template with 5-shot.

| Benchmark | This model (HC = MXFP8) | Reference MXFP4 (HC = BF16) | BF16 base |
|---|---:|---:|---:|
| PIQA (acc) | 82.05 | 81.83 | 81.94 |
| HellaSwag (acc) | 68.32 | 68.32 | 69.27 |
| MMLU (acc) | 86.84 | 86.56 | 86.52 |
| GSM8K (strict-match) | 96.82 | 96.36 | 96.74 |
| **Average** | **83.51** | **83.27** | **83.62** |

With the patch and the true-FP8 hyper-connection checkpoint the numbers are unchanged within
noise (82.48 / 68.27 / 86.75 / 96.74, average **83.56**), and the GPU footprint drops by
0.63 GiB.

## Generate the Model

```bash
auto-round \
    --model_name Qwen/Qwen3.8-Flash-Next \
    --model_free \
    --scheme MXFP8 \
    --ignore_layers "visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,indexer,shared_expert,self_attn,linear_attn,mtp.fc" \
    --layer_config '{"mtp.layers.0.mlp.experts":{"bits":8,"data_type":"mx_fp"},"mlp.experts":{"bits":4,"data_type":"mx_fp"}}' \
    --format llm_compressor \
    --output_dir Qwen3.8-Flash-Next-MXFP4-BF16Attn-MXFP8MTP-PLEMXFP8-CT-AutoRound

```

The compressed-tensors config it produces: `group_0` = `mxfp4-pack-quantized`
(73 728 explicit expert targets + `RoutedExperts`), `group_1` = `mxfp8-quantized`
(`Linear`, `RoutedExperts`, the PLE n-gram module — plus the 294 hyper-connection modules in
the true-FP8 variant), `ignore` = 943 entries (attention, indexer, shared expert, router,
embeddings, lm_head, vision tower, `mtp.fc`, norms, PLE projections).

## Ethical Considerations and Limitations

The model can produce factually incorrect output, and should not be relied on to provide
factually accurate information. Because of the limitations of the pretrained model and the
finetuning datasets, it is possible that this model could generate lewd, biased or otherwise
offensive outputs. Therefore, before deploying any applications of this model, developers
should perform safety testing.

## Caveats and Recommendations

- MXFP4 weights with **dynamic MXFP4 activations** (W4A4) is the largest single source of
  quantization error in this build; the routed experts are what make the compression, and
  they are also what to inspect first if quality regresses on a domain benchmark.
- The PLE n-gram table dominates deployment memory (51 B parameters). It is stored as MXFP8
  and looked up from pinned host memory; do not expect the checkpoint to shrink much below
  ~130 GB while it remains at 8 bits.
- Hyper-connection GEMMs are 0.37 % of the parameters — quantizing them changes accuracy by
  less than run-to-run noise (±0.25 pt on the four benchmarks above), and on vLLM it costs a
  CuTeDSL JIT warmup and a slightly different kernel path. Treat it as a format-consistency
  choice rather than an accuracy lever.

Users (both direct and downstream) should be made aware that the risks of this model
involve hallucinations, biased or offensive content, and factual errors.

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

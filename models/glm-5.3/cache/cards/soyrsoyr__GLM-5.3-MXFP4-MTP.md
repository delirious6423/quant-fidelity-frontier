---
base_model: zai-org/GLM-5.3
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
license: other
license_name: glm-5.3
license_link: LICENSE
language:
  - en
  - zh
tags:
  - llm-compressor
  - compressed-tensors
  - mxfp4
  - mtp
---

# GLM-5.3 MXFP4 with MTP

MXFP4 quantization of the full [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) checkpoint, including its multi-token prediction (MTP) layer. This is the full `glm_moe_dsa` model, not GLM-5.3 Flash.

The checkpoint was produced with [llm-compressor PR #3239](https://github.com/vllm-project/llm-compressor/pull/3239), on top of [#3225](https://github.com/vllm-project/llm-compressor/pull/3225), using the existing MTP loading, quantization, and export path.

- Source revision: `aca966e4e02791568aa6a4ced368624b3d897f42`.
- llm-compressor commit: `7ebb01341195269a2236e24fe14336a74dcda17a`.
- Format: compressed-tensors `mxfp4-pack-quantized`, with group size 32 for weights and dynamic MXFP4 input activations.
- The LM head, MTP `eh_proj`, MoE routers, and indexer `wk` / `weights_proj` projections remain dense.
- Weight files occupy approximately 376 GiB. The MTP weights are in `model_mtp.safetensors`, indexed under `model.layers.78`.

## Validation

Full quantization and export passed on four NVIDIA B200 GPUs in approximately 54 minutes. The exported checkpoint contains 118,607 tensors, including 1,568 MTP tensors. All 777 quantized MTP matrices passed numerical reconstruction checks, with a maximum normalized MSE of 0.0142504545. Dense exclusions matched the source exactly; scale and checkpoint-index checks passed. See [validation.json](validation.json).

The exported checkpoint also passed a vLLM 0.28.0 decoding smoke test on four B200s: **474 accepted draft tokens out of 544 drafted tokens (87.13%)**, using one MTP draft token per step. The test used eight prompts, greedy generation, and a 128-token output cap per prompt. Every output reached that cap while producing reasoning text, even though the test passed `enable_thinking=False` to the chat template. These results establish nonzero MTP acceptance; they are not a quality evaluation or a speedup benchmark. See [mtp-acceptance.json](mtp-acceptance.json).

## Quantization recipe and code

The exact scripts used for the run are included:

- [ohio-glm53-download.py](reproduction/ohio-glm53-download.py): downloads the pinned source checkpoint.
- [ohio-glm53-runtime.py](reproduction/ohio-glm53-runtime.py): loads the FP8 source, quantizes the backbone and MTP layer, saves the checkpoint, and verifies the result.
- [recipe.yaml](recipe.yaml): the serialized llm-compressor recipe.
- [ohio-glm53-acceptance.py](reproduction/ohio-glm53-acceptance.py): the inference smoke test used for the acceptance numbers above.

The quantization call in that harness was:

```python
import torch
from transformers import AutoModelForCausalLM, FineGrainedFP8Config
from compressed_tensors.offload import set_onload_device
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# `source` is the local snapshot at the pinned source revision.
# Run under the included harness: it initializes distributed execution and
# installs the padded-block FP8 loader workaround described below.
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=FineGrainedFP8Config(dequantize=True),
        dtype=torch.bfloat16,
        device_map="auto_offload",
        max_memory={"cpu": "300GiB"},
        offload_folder=str(RUN / "offload"),
        local_files_only=True,
    )
set_onload_device(model, "cuda")
if not model.generation_config.do_sample:
    model.generation_config.top_p = 1.0
model.generation_config.validate(strict=True)

recipe = QuantizationModifier(
    scheme="MXFP4",
    targets=["Linear", r"re:^mtp\.layers\."],
    ignore=[
        "lm_head",
        r"re:.*\.eh_proj$",
        r"re:.*\.mlp\.gate$",
        r"re:.*\.indexer\.(?:wk|weights_proj)$",
    ],
)
oneshot(model=model, recipe=recipe, output_dir=str(destination))
```

This recipe does not require a calibration dataset. It uses the MXFP4 weight observer and dynamic activation quantization. The MTP configuration, native FP8 scale restoration, and integration with the existing export path come from the pinned llm-compressor commit.

### Environment and run commands

The quantization venv used Python 3.12, PyTorch 2.13.0 with CUDA 13.0, Transformers 5.17.0, and compressed-tensors 0.19.0. The full installed package inventory is in [environment-quantization.json](reproduction/environment-quantization.json). The run used four B200s, a 300 GiB CPU offload limit, disk offload, and a host with about 1.9 TiB RAM. The harness's download step requires at least 4 TiB of free disk headroom; the completed run occupied about 2.8 TiB before cleanup.

Use a dedicated venv containing those dependencies, then install the tested llm-compressor checkout without changing them:

```bash
# From a local copy of this repository, with the quantization venv activated:
cd reproduction
git clone https://github.com/soyr-redhat/llm-compressor.git
git -C llm-compressor checkout 7ebb01341195269a2236e24fe14336a74dcda17a
python -m pip install --no-deps -e ./llm-compressor

export HF_HUB_CACHE="$PWD/hf-cache"
export HF_XET_CACHE="$PWD/xet-cache"
export TMPDIR="$PWD/tmp"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"
python ohio-glm53-download.py
export HF_HUB_OFFLINE=1
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  ohio-glm53-runtime.py full
```

The Ohio run reserved four GPUs through `chg` before executing the `torch.distributed.run` command. Ensure four GPUs are assigned to your process on a shared host. The output is `reproduction/GLM-5.3-MXFP4-MTP`.

The source checkpoint uses FP8 block quantization. This run required three explicit harness adjustments, all present in the included runtime script:

1. Transformers 5.17's backbone FP8 dequantizer mishandles padded blocks, including a 576-row projection with five 128-row scale blocks. The harness replaces that operation with compressed-tensors' existing `FP8BlockDequantizer`, preserving the source scales. The replacement was checked against a direct numerical reference.
2. Distributed initialization uses a 60-minute collective timeout while rank 0 loads and offloads the full model.
3. The source generation configuration contains inactive `top_p=0.95` while sampling is disabled. The harness sets `top_p=1.0` to preserve greedy behavior and satisfy Transformers' strict save validation.

This is therefore not a stock-Transformers end-to-end quantization result. The snippet above depends on the included harness for those adjustments. The llm-compressor checkout was installed with `--no-deps`; its development metadata requests a newer compressed-tensors prerelease, but this run explicitly used 0.19.0.

## Serve with vLLM

The acceptance test used a separate activated venv with vLLM 0.28.0, PyTorch 2.13.0, Transformers 5.16.1, and compressed-tensors 0.17.0. See [environment-serving.json](reproduction/environment-serving.json). Activate your serving venv before launching so its `ninja` executable is available; CUDA 13.0's `nvcc` must also be on `PATH` for kernel compilation.

This server command uses the same engine settings as the successful Python `LLM` smoke test:

```bash
source /path/to/serving-venv/bin/activate
export PATH="${CUDA_HOME:-/usr/local/cuda-13.0}/bin:$PATH"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

vllm serve soyrsoyr/GLM-5.3-MXFP4-MTP \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --seed 42 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

The four-GPU smoke test loaded approximately 96.13 GiB of model weights per GPU, before KV cache and other runtime allocations. Two 180 GiB B200s are insufficient for this fully GPU-resident checkpoint. The server command is an equivalent launch configuration; the recorded acceptance test used vLLM's Python `LLM` API, not the HTTP server.

## License

The weights retain the original [GLM-5.3 license](LICENSE), including its copyright notice and conditions. This is an experimental quantized derivative, not an official Z.AI release.

---
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
  - ocp-mx
  - quantization
  - amd-quark
  - moe
  - deepseek
---

# DeepSeek-V4.1-Flash-MXFP4

## Model Overview

- **Model Architecture:** DeepseekV41ForCausalLM
  - **Input:** Text, Image
  - **Output:** Text
- **Supported Hardware Microarchitecture:** AMD MI355 / MI350 (gfx950)
- **ROCm:** 7.2.0
- **PyTorch:** 2.12.0
- **Transformers:** 5.17.0
- **Operating System(s):** Linux
- **Inference Engine:** [vLLM](https://docs.vllm.ai/en/latest/)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html) (v0.13.0)
  - **Quantized layers:** `experts`, `shared_experts`, and `self_attn` in language model. The vision part is not quantized.
  - **experts and shared_experts:** OCP MXFP4 for both weights and activations
  - **self_attn:** OCP MXFP8 for both weights and activations

## Model Quantization

Quantized from [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
with [AMD Quark](https://quark.docs.amd.com/latest/index.html). The MoE expert
projections (`experts`, `shared_experts`) are quantized to OCP MXFP4 and the
language-model attention (`self_attn`) to OCP MXFP8 — weights and activations in
both cases. The vision tower and the remaining modules (router gate, embeddings,
output head, engram and MTP blocks) are kept in their original precision.

### Quantization script

```python
from quark.torch import LLMTemplate, ModelQuantizer

MODEL_DIR = "deepseek-ai/DeepSeek-V4.1-Flash"
OUTPUT_DIR = "amd/DeepSeek-V4.1-Flash-MXFP4"

template = LLMTemplate(
    model_type="deepseek_v41",
    kv_layers_name=["*wkv"],
    q_layer_name=["*wq_a", "*wq_b"],
    exclude_layers_name=[
        "*attn*",
        "embed",
        "*head*",
        "*ffn.gate*",
        "hc_*",
        "*engram*",
        "*vision*",
        "*aligner*",
        "*main_proj*",
    ],
)

LLMTemplate.register_template(template)

quant_config = template.get_config(scheme="mxfp4")

quantizer = ModelQuantizer(quant_config)
quantizer.direct_quantize_checkpoint(
    pretrained_model_path=MODEL_DIR,
    save_path=OUTPUT_DIR,
    keep_excluded_layers_as_original_model_state=True,
)
```

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend based on the Docker image `vllm/vllm-openai-rocm:deepseekv41-flash-0909`.

## Evaluation

The model was evaluated with vLLM on GSM8K, GPQA Diamond, AIME26, MMMU-Pro, AA-LCR and IFBench.

### Accuracy

| Benchmark | deepseek-ai/DeepSeek-V4.1-Flash | amd/DeepSeek-V4.1-Flash-MXFP4 (this model) | Recovery |
|---|---:|---:|---:|
| GSM8K (8-shot, flexible-extract) | 92.87 | 92.34 | 99.4% |
| GPQA Diamond (0-shot, thinking) | 89.39 | 90.40 | 101.1% |
| AIME26 | 97.50 | 97.71 | 100.2% |
| MMMU-Pro, standard (10 options) | 76.47 | 76.36 | 99.9% |
| MMMU-Pro, vision | 72.95 | 72.02 | 98.7% |
| AA-LCR | 78.25 | 77.88 | 99.5% |
| IFBench | 78.93 | 78.13 | 99.0% |

GSM8K is measured with lm-eval and GPQA Diamond with sgl-eval. For each benchmark
the base and quantized models are run with the same framework, settings, and host.

GPQA Diamond is scored over all 198 questions using stochastic sampling
(`temperature=1.0, top_p=0.95`), so scores vary from run to run; the 95% confidence
interval on 198 questions is ±2.33%. On this draw the quantized model answered
179/198 correctly against 177/198 for the base model. That gap is well inside the
confidence interval, so the two should be read as equivalent rather than as the
quantized model outperforming the base.

### Reproduction

#### GSM8K and GPQA Diamond

The evaluation runs against a vLLM server, using the Docker image
`vllm/vllm-openai-rocm:deepseekv41-flash-0909`.

##### Launching server
```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_MOE=0
vllm serve amd/DeepSeek-V4.1-Flash-Quark-MXFP4 --tensor-parallel-size 4 \
  --trust-remote-code --tokenizer-mode deepseek_v41 \
  --max-model-len 73728 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --enforce-eager
```

The context length must accommodate GPQA Diamond's 65536-token generation budget
in thinking mode; GSM8K alone runs comfortably at `--max-model-len 8192`.

##### Evaluating GSM8K in a new terminal
```bash
lm_eval --model local-completions \
    --model_args model=amd/DeepSeek-V4.1-Flash-Quark-MXFP4,base_url=http://localhost:8000/v1/completions,tokenized_requests=False,num_concurrent=8,tokenizer=amd/DeepSeek-V4.1-Flash-Quark-MXFP4 \
    --tasks gsm8k --batch_size auto --num_fewshot 8 --seed 42 \
    --gen_kwargs "temperature=0,max_gen_toks=4096"
```

##### Evaluating GPQA Diamond
```bash
pip install sgl-eval
sgl-eval run gpqa \
    --model amd/DeepSeek-V4.1-Flash-Quark-MXFP4 \
    --api-key EMPTY \
    --base-url http://localhost:8000/v1 \
    --num-threads 8 \
    --n-repeats 1 \
    --max-tokens 65536 \
    --temperature 1.0 \
    --top-p 0.95 \
    --thinking
```

#### AIME26, MMMU-Pro, AA-LCR and IFBench

##### Settings

- Thinking mode on (the DeepSeek-V4.1 default). `reasoning_effort` not set, so the chat template's default (`high` = 75) applies.
- Sampling: `temperature` 1.0, `top_p` 0.95, `max_tokens` 98,304.
- One TP=2 server per model; the base model uses `deepseek-ai/DeepSeek-V4.1-Flash` with the same flags.
- Repeats are separate samples (no per-request seed); mean@k is the mean accuracy over all k samples.

##### Environment

| Component | Version |
|---|---|
| GPUs | AMD Instinct MI355X × 2 per model (TP=2) |
| Container | `vllm/vllm-openai-rocm:nightly-rocm100-5d35ce3934a7804b298559d2d99eec33792e1eb9` (vLLM `0.32.1rc1.dev602+g5d35ce393`) |
| AITER | rebuilt from main `59fae2078` (`amd-aiter 0.1.28.dev152`) with `flydsl 0.4.1`; v0.1.28 contains the same fixes |
| Evaluator | EvalScope `1.14.1` |
| Checkpoint | this repository at commit `0d56db2133ce922656be8025eac1e8450243aed5` |

##### Container and AITER

```bash
VGID=$(getent group video | cut -d: -f3); RGID=$(getent group render | cut -d: -f3)
docker run -d --name dsv41 \
  --entrypoint /bin/bash --network=host --ipc=host --shm-size 64G \
  --device=/dev/kfd --device=/dev/dri --group-add $VGID --group-add $RGID \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MHA=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e SAFETENSORS_FAST_GPU=1 -v /data/models:/data/models \
  vllm/vllm-openai-rocm:nightly-rocm100-5d35ce3934a7804b298559d2d99eec33792e1eb9 \
  -c 'sleep infinity'

# Inside the container: AITER with the native MXFP4 MoE fixes
git clone --recursive https://github.com/ROCm/aiter.git /opt/aiter
cd /opt/aiter && git checkout 59fae2078 && git submodule update --init --recursive   # or v0.1.28
pip install flydsl==0.4.1
AITER_USE_SYSTEM_TRITON=1 python3 -m pip install -e . --no-build-isolation
```

The image has no `render` group name, so numeric group IDs are passed to `--group-add`.

##### Launching server

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve amd/DeepSeek-V4.1-Flash-Quark-MXFP4 \
  --tensor-parallel-size 2 --trust-remote-code \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --max-model-len 262144 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --port 8000
```

- `--max-model-len 262144` is needed for AA-LCR (inputs average ~95K tokens) and was used for IFBench; AIME26 and MMMU-Pro were run with 131072.
- `--max-num-batched-tokens 8192` is required: larger prefill chunks hit an untuned MoE kernel path.

##### Evaluation client

```bash
python3 -m venv ~/es-venv
~/es-venv/bin/pip install 'evalscope==1.14.1' langdetect
~/es-venv/bin/pip install -r ~/es-venv/lib/python3*/site-packages/evalscope/benchmarks/ifbench/requirements.txt
~/es-venv/bin/python -c "import nltk; [nltk.download(p) for p in ('punkt_tab','stopwords','averaged_perceptron_tagger_eng')]"
```

##### Commands

Common arguments:

```bash
MODEL=amd/DeepSeek-V4.1-Flash-Quark-MXFP4      # must match the served model name
URL=http://127.0.0.1:8000/v1/chat/completions
GEN='{"temperature":1.0,"top_p":0.95,"max_tokens":98304,"timeout":43200}'
```

**AIME26** (mean@16; 16 runs of this command in separate work dirs):

```bash
evalscope eval --model $MODEL --api-url $URL --datasets aime26 \
  --eval-batch-size 32 --generation-config "$GEN" --work-dir runs/aime26_run1
```

Do not name a work dir after the dataset (e.g. `aime26`): it shadows the dataset and the run fails with `DataFilesNotFoundError`.

**MMMU-Pro** (standard and vision):

```bash
evalscope eval --model $MODEL --api-url $URL --datasets mmmu_pro \
  --dataset-args '{"mmmu_pro":{"extra_params":{"dataset_format":"standard (10 options)"}}}' \
  --eval-batch-size 32 --generation-config "$GEN" --work-dir runs/mmmu_pro_standard
# vision: "dataset_format":"vision"
```

**AA-LCR** (mean@8, LLM judge):

```bash
evalscope eval --model $MODEL --api-url $URL --datasets aa_lcr --dataset-hub huggingface \
  --dataset-args '{"aa_lcr":{"hf_dataset_revision":"bdae010bbce259820c0e34c1d7cce210d966fb75"}}' \
  --eval-batch-size 32 --repeats 8 --generation-config "$GEN" \
  --judge "$(cat judge.json)" --work-dir runs/aa_lcr_r8
```

Judge: `gpt-5.6-luna` through the OpenAI Responses API, `reasoning_effort` medium, `max_tokens` 4096. `judge.json` (fill in your own endpoint and key):

```json
{"strategy": "auto",
 "models": [{"model_id": "gpt-5.6-luna", "api_url": "<endpoint>/responses", "api_key": "<key>",
             "eval_type": "openai_responses_api",
             "generation_config": {"reasoning_effort": "medium", "max_tokens": 4096, "timeout": 600}}]}
```

- EvalScope masks only fields named `api_key`; credentials in other fields (e.g. `extra_headers`) are written to the run log and config in plain text.
- Start the first run alone until it has downloaded the AA-LCR documents. Two runs starting on an empty cache race and both fail.

**IFBench** (mean@5):

```bash
evalscope eval --model $MODEL --api-url $URL --datasets ifbench --dataset-hub huggingface \
  --dataset-args '{"ifbench":{"primary_metric":"prompt_level_loose"}}' \
  --eval-batch-size 32 --repeats 5 --generation-config "$GEN" --work-dir runs/ifbench_r5
```

## License

This model is a quantized derivative of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
and is distributed under the same license as the source model: the
[MIT License](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE).
A copy of the upstream LICENSE is included in this repository.

Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
AMD has modified the model weights of the MoE expert layers by quantizing them to
MXFP4 with AMD Quark; the modifications are provided under the same MIT License and
are not subject to any separate or different license.

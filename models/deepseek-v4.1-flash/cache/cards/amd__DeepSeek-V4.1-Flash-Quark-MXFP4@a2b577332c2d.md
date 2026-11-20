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

This model is served with the [vLLM](https://docs.vllm.ai/en/latest/) backend using the
**Docker image `vllm/vllm-openai-rocm:v0.33.0` with AITER rebuilt at v0.1.28 and FlyDSL 0.4.1**,
TP=2, and `VLLM_ROCM_USE_AITER_MOE=1`, which runs the native MXFP4 MoE kernel (`AITER_MXFP4_MXFP4`).
This is the only supported setup.

- **The AITER rebuild is required.** The image ships AITER 0.1.26, whose MXFP4 MoE fails during CUDA-graph capture with `Memory access fault`. See [Container and AITER](#container-and-aiter) and [Launching server](#launching-server).
- **Older images do not work.** The `vllm/vllm-openai-rocm:deepseekv41-flash-0909` image (vLLM 0.28.1rc1), listed in earlier versions of this card, cannot load this checkpoint: at TP=4 the shared-expert FP8 block check fails, and at TP=2 weight loading fails.

## Evaluation

The model was evaluated with vLLM on GSM8K, GPQA Diamond, AIME26, MMMU-Pro, AA-LCR and IFBench,
and on the agentic coding benchmarks DeepSWE v1.1 and Terminal-Bench 4.0.

### Accuracy

| Benchmark | deepseek-ai/DeepSeek-V4.1-Flash | amd/DeepSeek-V4.1-Flash-Quark-MXFP4 (this model) | Recovery |
|---|---:|---:|---:|
| GSM8K (8-shot, flexible-extract) | 92.49 | 92.04 | 99.5% |
| GPQA Diamond (0-shot, thinking, mean@8) | 90.15 | 88.64 | 98.3% |
| AIME26 (mean@16) | 97.50 | 97.71 | 100.2% |
| MMMU-Pro, standard (10 options, mean@1) | 76.47 | 76.36 | 99.9% |
| MMMU-Pro, vision (mean@1) | 72.95 | 72.02 | 98.7% |
| AA-LCR (LLM judge, mean@8) | 78.25 | 77.88 | 99.5% |
| IFBench (prompt-level loose, mean@5) | 78.93 | 78.13 | 99.0% |

GSM8K is measured with lm-eval, GPQA Diamond with sgl-eval, and AIME26, MMMU-Pro,
AA-LCR and IFBench with EvalScope 1.14.1. For each benchmark the base and quantized
models are run with the same framework, settings, and host.

**Image:** all instructions on this card use `vllm/vllm-openai-rocm:v0.33.0` + AITER v0.1.28 + FlyDSL 0.4.1.
- **Measured on this image:** GSM8K and GPQA Diamond.
- **Measured earlier on a vLLM nightly** (vLLM 0.32.1rc1, 2026-10-27) with the same AITER MoE fixes: AIME26, MMMU-Pro, AA-LCR, IFBench and the agentic benchmarks below.
- **Re-verified on this image** with AA-LCR: 78.88 for this model, within run-to-run variation of the 77.88 above.

**GPQA Diamond** uses stochastic sampling (`temperature=1.0, top_p=0.95`), so each of the 8 runs over the 198 questions varies:

| | Per-run range | Mean@8 |
|---|---|---|
| Base | 87.88–91.41 | 90.15 |
| This model | 86.87–89.90 | 88.64 |

Paired by question, the difference is −1.52 ± 0.85 points (sign-flip p = 0.09), which is not significant.

### Agentic coding

| Benchmark | deepseek-ai/DeepSeek-V4.1-Flash | amd/DeepSeek-V4.1-Flash-Quark-MXFP4 (this model) | Recovery |
|---|---:|---:|---:|
| DeepSWE v1.1 (resolved, 113 tasks, mean@3) | 71.4 | 72.0 | 100.8% |
| Terminal-Bench 4.0 (pass rate, 63 tasks, mean@3) | 24.9 | 22.2 | 89.4% |

Both use mini-swe-agent 2.4.6 at DeepSeek's evaluation settings: `reasoning_effort=max`, `temperature=1.0`, `top_p=0.95`, 1M-token context, at most 500 agent steps.

**DeepSWE v1.1:**
- **Harness:** DeepSeek's `evaluation/README.md` setup, i.e. Pier `0c802fc` with `dsh-minimal.patch` and deep-swe `0b9fabb`.
- **Per-run solved counts:** baseline 78 / 88 / 76, this model 79 / 82 / 83.
- **Comparison:** paired by task over the 3 runs, the difference is +0.6 ± 2.7 points (sign-flip p = 0.91), which is not significant.

**Terminal-Bench 4.0:**
- **Harness:** v4.0.0 on Harbor 0.25.
- **Tasks:** the 3 tasks that need NVIDIA H100 GPUs are excluded.
- **Comparison:** paired by task, the difference is −2.6 ± 4.0 points (sign-flip p = 0.60), which is not significant.
- **Not directly comparable to DeepSeek's 31.2.** DeepSeek used the DeepSeek Harness on all 66 tasks.
- **One task fails for both models:** `vllm-deepseek-streaming` cannot be solved with vLLM's `deepseek_v41` tokenizer mode, which rejects message text containing the image token.

Grader timeouts count as unsolved for both models. Single runs vary by several points on both benchmarks, so only the means over 3 runs should be compared.

**Serving for agentic use:** the command under [Launching server](#launching-server), with three changes:
- `--enable-auto-tool-choice --tool-call-parser deepseek_v41` added (tool calling);
- `--max-model-len 1048576`;
- `--gpu-memory-utilization 0.85`.

### Reproduction

All benchmarks use the environment, container and server below; the agentic benchmarks add the
serving flags listed under [Agentic coding](#agentic-coding). The base model uses
`deepseek-ai/DeepSeek-V4.1-Flash` with the same flags, on its own TP=2 server.

#### Environment

| Component | Version |
|---|---|
| GPUs | AMD Instinct MI355X × 2 per model (TP=2) |
| Container | `vllm/vllm-openai-rocm:v0.33.0` (vLLM `0.33.0`, PyTorch `2.13.0+rocm10.0.0`) |
| AITER | rebuilt at `v0.1.28` (`f1965cd`) with `flydsl 0.4.1`; the image ships AITER `0.1.26` and FlyDSL `0.4.0` |
| Evaluators | lm-eval `0.4.13` (GSM8K), sgl-eval `0.1.3` (GPQA Diamond), EvalScope `1.14.1` (AIME26, MMMU-Pro, AA-LCR, IFBench) |
| Checkpoint | this repository; the weights are unchanged since the initial upload |

#### Container and AITER

```bash
VGID=$(getent group video | cut -d: -f3); RGID=$(getent group render | cut -d: -f3)
docker run -d --name dsv41 \
  --entrypoint /bin/bash --network=host --ipc=host --shm-size 64G \
  --device=/dev/kfd --device=/dev/dri --group-add $VGID --group-add $RGID \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MHA=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e SAFETENSORS_FAST_GPU=1 -v /data/models:/data/models \
  vllm/vllm-openai-rocm:v0.33.0 \
  -c 'sleep infinity'

# Inside the container: AITER v0.1.28 with the native MXFP4 MoE fixes
git clone --recursive --branch v0.1.28 https://github.com/ROCm/aiter.git /opt/aiter
cd /opt/aiter
pip install flydsl==0.4.1
AITER_USE_SYSTEM_TRITON=1 python3 -m pip install -e . --no-build-isolation
python3 -c "import aiter, importlib.metadata as m; print(aiter.__file__, m.version('amd-aiter'))"
# expect: /opt/aiter/aiter/__init__.py 0.1.28
```

- The image has no `render` group name, so numeric group IDs are passed to `--group-add`.
- Rebuild AITER before starting any server in the container. A server started on the image's AITER 0.1.26 compiles kernels into `/usr/local/lib/python3.12/dist-packages/aiter/jit`; that directory survives the upgrade and hides the new install (`ImportError: cannot import name 'ActivationType'`). If that happens, move the directory away.

#### Launching server

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve amd/DeepSeek-V4.1-Flash-Quark-MXFP4 \
  --tensor-parallel-size 2 --trust-remote-code \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --max-model-len 262144 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --port 8000
```

- `--max-num-batched-tokens 8192` is required: larger prefill chunks hit an untuned MoE kernel path.
- The server log should show `Using 'AITER_MXFP4_MXFP4' Mxfp4 MoE backend`.
- `--max-model-len`:
  - 262144 was used for GSM8K, GPQA Diamond, AA-LCR and IFBench; AA-LCR needs it, since its inputs average ~95K tokens.
  - AIME26 and MMMU-Pro were run with 131072.

#### GSM8K

8-shot, greedy, all 1,319 questions:

```bash
python3 -m venv ~/lmeval-venv
~/lmeval-venv/bin/pip install 'lm-eval[api]==0.4.13' transformers   # the tokenizer needs transformers
~/lmeval-venv/bin/lm_eval --model local-completions \
  --model_args model=amd/DeepSeek-V4.1-Flash-Quark-MXFP4,base_url=http://localhost:8000/v1/completions,tokenized_requests=False,num_concurrent=8,tokenizer=amd/DeepSeek-V4.1-Flash-Quark-MXFP4,tokenizer_backend=huggingface,trust_remote_code=True \
  --tasks gsm8k --batch_size auto --num_fewshot 8 --seed 42 \
  --gen_kwargs "temperature=0,max_gen_toks=4096"
```

#### GPQA Diamond

Mean@8, thinking mode, default `reasoning_effort`:

```bash
python3 -m venv ~/sgleval-venv
~/sgleval-venv/bin/pip install 'sgl-eval==0.1.3'
~/sgleval-venv/bin/sgl-eval run gpqa \
  --model amd/DeepSeek-V4.1-Flash-Quark-MXFP4 --api-key EMPTY \
  --base-url http://localhost:8000/v1 \
  --num-threads 64 --n-repeats 8 \
  --max-tokens 65536 --temperature 1.0 --top-p 0.95 --thinking
```

#### AIME26, MMMU-Pro, AA-LCR and IFBench

##### Settings

- Thinking mode is on (the DeepSeek-V4.1 default). `reasoning_effort` is not set, so the chat template's default (`high` = 75) applies.
- Sampling: `temperature` 1.0, `top_p` 0.95, `max_tokens` 98,304.
- Repeats are separate samples with no per-request seed; mean@k is the mean accuracy over all k samples.

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

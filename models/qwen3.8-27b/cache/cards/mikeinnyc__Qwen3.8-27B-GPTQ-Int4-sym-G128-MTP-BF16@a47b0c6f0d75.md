---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags: [qwen, qwen3.8, gptq, int4, g128, vllm, xpu, intel, intel-arc, intel-arc-pro, mtp, speculative-decoding, long-context]
---

# Qwen3.8-27B GPTQ INT4 symmetric G128 + unquantized MTP

Fresh GPTQ INT4 quantization of **Qwen/Qwen3.8-27B**, produced directly from the original BF16 checkpoint.

This is the G128 checkpoint selected from controlled G128 vs G32 testing on a single **Intel Arc Pro B70 32 GB** using vLLM XPU.

**No quantization step is required when downloading this repository.**

Full reproduction procedure, Dockerfiles, patches, benchmarks, calibration methodology, and deployment guide:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm

## Checkpoint

* Base model: `Qwen/Qwen3.8-27B`
* Quantization: GPTQ INT4
* Group size: 128
* Symmetric: `true`
* `desc_act`: `false`
* LM head quantization: disabled
* MTP tensors: excluded from GPTQ quantization
* MTP draft path used in testing: unquantized
* Quantizer: GPTQModel 7.3.2
* Tested serving stack: vLLM XPU
* Tested GPU: Intel Arc Pro B70 32 GB

This is a quantized derivative, not a fine-tune. No additional training was performed.

## Why G128

G128 and G32 were produced from the same BF16 source using the same frozen calibration token stream.

| Checkpoint | Approx. size | Tested context |
| ---------- | -----------: | -------------: |
| G128       |     18.22 GB |        161,000 |
| G32        |     19.54 GB |        128,000 |

Under the tested `gpu_memory_utilization=0.90` configuration with FP8 KV cache, G128 successfully served at a configured 161,000-token context on the Intel Arc Pro B70.

G32 required more model memory and did not provide enough remaining KV-cache capacity for the same 161K configuration. It also did not show enough source-fidelity improvement in the controlled tests to justify the additional memory use.

For this experiment, **G128 is the preferred checkpoint**.

## Quantization calibration

Frozen calibration:

```text
128 samples
1024 tokens per sample
131,072 calibration tokens total
```

Calibration file:

```text
c4-fixed-128x1024.json
```

SHA256:

```text
ddfc570e23458c048951501231c2ff75fa175440b120045bbeb1790bea5d2599
```

The calibration data was used only for GPTQ quantization calibration. It was not used to train or fine-tune the model.

Quantization procedure:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm/blob/main/docs/QUANTIZATION.md

## Tested Intel Arc Pro B70 serving profile

```text
quantization = gptq
dtype = float16
max_model_len = 161000
gpu_memory_utilization = 0.90
kv_cache_dtype = fp8
max_num_seqs = 1
max_num_batched_tokens = 8192
prefix caching = disabled
speculative decoding = MTP
num_speculative_tokens = 4
```

Environment used during testing:

```text
B70_MTP_BF16_DRAFT=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_TARGET_DEVICE=xpu
ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
ZE_AFFINITY_MASK=0
PYTORCH_ALLOC_CONF=expandable_segments:True
```

The tested B70 MTP4 configuration also uses the vLLM XPU patches documented in the GitHub repository.

Downloading this checkpoint removes the need to perform GPTQ quantization yourself, but the tested B70 MTP4 serving path still requires the documented vLLM XPU environment and patches.

Deployment guide:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm/blob/main/docs/B70-QUALITY-DEPLOYMENT.md

## Download

This is **not a single GGUF file**.

The checkpoint is stored as five `.safetensors` model shards plus tokenizer and configuration files. Download the **entire repository**, not an individual shard.

Using the Hugging Face CLI:

```bash
hf download mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --local-dir ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

The complete download is approximately **19.6 GB** on Hugging Face.

After download, the directory should contain files including:

```text
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
model.safetensors.index.json
config.json
quantize_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

## Run with vLLM + Open WebUI

This checkpoint is intended to be **served by vLLM** and then accessed through an OpenAI-compatible client such as Open WebUI.

It is not loaded directly into Open WebUI.

The basic flow is:

```text
Hugging Face checkpoint
        |
        v
      vLLM
        |
        v
OpenAI-compatible API
        |
        v
    Open WebUI
```

### 1. Download the model

```bash
hf download mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --local-dir ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

### 2. Serve it with vLLM XPU

For Intel Arc Pro B70, use the tested Docker/vLLM XPU configuration documented here:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm/blob/main/docs/B70-QUALITY-DEPLOYMENT.md

The reference configuration exposes vLLM on host port:

```text
11444
```

with the OpenAI-compatible API available at:

```text
http://127.0.0.1:11444/v1
```

The reference served-model name is:

```text
qwen38
```

Verify that vLLM is running:

```bash
curl http://127.0.0.1:11444/v1/models
```

You should receive a JSON response containing the available model.

### 3. Connect Open WebUI

If you already use Open WebUI with Ollama or another backend, you can add vLLM as an additional OpenAI-compatible connection.

In Open WebUI, add an OpenAI-compatible API connection.

#### Open WebUI running directly on the host

Use:

```text
API URL:
http://127.0.0.1:11444/v1
```

#### Open WebUI running in Docker and vLLM exposed on the host

Inside an Open WebUI container, `localhost` refers to the Open WebUI container itself.

Use a host-reachable address such as:

```text
http://host.docker.internal:11444/v1
```

On Linux Docker hosts, `host.docker.internal` may need to be explicitly mapped to the host gateway.

For example, the Open WebUI container can be started with:

```bash
--add-host=host.docker.internal:host-gateway
```

#### Open WebUI and vLLM on the same Docker network

If both containers share a Docker network, Open WebUI can connect directly to the vLLM container using its container name and internal port.

For the reference container:

```text
http://qwen38-quality-mtp4:8000/v1
```

This avoids routing the connection back through the host port.

### 4. Select the model

After the vLLM connection is saved, the model served as:

```text
qwen38
```

should appear in Open WebUI's model selector.

Select it and use Open WebUI normally.

### Important difference from GGUF / llama.cpp

Users coming from `llama.cpp` or Ollama may be accustomed to downloading a single `.gguf` file.

This model is different.

It uses a Hugging Face GPTQ checkpoint consisting of multiple `.safetensors` shards:

```text
5 model shards
+ model index
+ tokenizer
+ GPTQ configuration
+ model configuration
```

Do **not** download only one `.safetensors` shard.

Download the entire repository and point vLLM at the resulting model directory.

### B70 users

For the exact Intel Arc Pro B70 Docker image, MTP4 patches, XPU environment variables, FP8 KV configuration, and 161K-context launch command, use:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm/blob/main/docs/B70-QUALITY-DEPLOYMENT.md

That is the tested configuration used for the published B70 results.

## Quality validation

A 30-prompt source-fidelity suite tested conflicting documents, missing information, timeline ambiguity, document hierarchy, source provenance, and unsupported explanations.

Final deterministic profile:

```text
temperature = 0.0
top_p = 1.0
top_k = -1
max_tokens = 4096
```

Two back-to-back deterministic G128/MTP4 runs completed:

```text
30/30 requests completed
0 errors
0 truncations
```

Aggregate generation throughput on this thinking/source-fidelity workload was approximately:

```text
57.75 tok/s
```

This is **not directly comparable** to the separately published **84.65 tok/s** short-context decode benchmark because the workloads are different.

Quality methodology and published result files:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm/tree/main/benchmarks/quality

## Known limitation

One conflicting-document test repeatedly produced a disputed interpretation involving quantities of 240 and 238.

The model treated the receiving log's 238-unit figure as definitively received rather than leaving the conflicting quantities unresolved.

This behavior was observed across G128, G32, multiple prompt variants, sampled inference, and deterministic inference. There is therefore no evidence that it is simply caused by GPTQ group size.

The benchmark itself also contains semantic ambiguity: an invoice quantity and a receiving-log quantity can represent different facts. Future testing should explicitly distinguish fields such as `units shipped` and `units received`.

This limitation is documented intentionally rather than presenting the checkpoint as perfect source fidelity.

## Recommended high-reliability RAG flow

For high-stakes document QA:

```text
documents
  -> extract facts
  -> normalize fields
  -> detect conflicts
  -> classify evidence
  -> construct grounded context
  -> LLM generation
```

Do not rely exclusively on prompt engineering to detect every evidence conflict.

## Performance reference

Separate short-context MTP4 benchmark:

```text
Median decode: 84.65 tok/s
Mean decode:   84.49 tok/s
Prompt:        512 tokens
Generation:    128 tokens
```

Full methodology and evidence:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm

## Credits

### Qwen

Original model:

`Qwen/Qwen3.8-27B`

https://huggingface.co/Qwen/Qwen3.8-27B

### GPTQModel

GPTQModel 7.3.2 was used for the fresh GPTQ INT4 quantization.

### vLLM / Intel XPU

vLLM XPU was used for serving, long-context validation, and MTP speculative decoding testing.

### SergiioB

Special thanks to **SergiioB** and the Intel Arc Pro B70 inference cookbook for the B70 vLLM/XPU/MTP groundwork:

https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook

### Reproduction and quantization

Fresh quantization, controlled G128/G32 testing, calibration freezing, B70 validation, quality benchmarking, and documentation:

**Mike Caldera**

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm

## License

This checkpoint is a quantized derivative of Qwen3.8-27B.

See the upstream model repository and license for the original model:

https://huggingface.co/Qwen/Qwen3.8-27B

## Disclaimer

Performance and maximum context capacity depend on the complete hardware and software environment.

The 161K configuration is a measured result from the tested Intel Arc Pro B70 system and is not a guarantee for every installation.

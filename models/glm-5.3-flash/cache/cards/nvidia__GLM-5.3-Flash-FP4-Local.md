---
pipeline_tag: image-text-to-text
base_model:
- zai-org/GLM-5.3-Flash
license: other
license_name: nvidia-open-model-agreement
library_name: Model Optimizer
tags:
- nvidia
- ModelOpt
- GLM-5
- quantized
- FP4
- fp4
---

# Model Overview

**IMPORTANT**: the NVFP4 scale factors in this checkpoint are in a new lossless compression format, but upstream vLLM support for this format has not yet landed. **You must use one of the containers provided below.**

## Description:
The NVIDIA GLM-5.3-Flash-FP4-Local model is the quantized version of ZAI's GLM-5.3-Flash model, which is an auto-regressive language model that uses an optimized transformer architecture. GLM-5.3-Flash is a natively multimodal Mixture-of-Experts (MoE) model for reasoning, coding, and agentic tasks; it uses a hybrid sparse and linear attention architecture with Manifold-Constrained Hyper-Connections (mHC) to support a long context. For more information, please check [here](https://huggingface.co/zai-org/GLM-5.3-Flash). The NVIDIA GLM-5.3-Flash-FP4-Local model is quantized with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) using [quantization-aware distillation (QAD)](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/).

This checkpoint is best suited for local AI hardware and uses approximately **166 GiB of vRAM**. It has been validated with vLLM on the following hardware: <br>
* **2 x NVIDIA RTX PRO 6000 Blackwell with 1.8M tokens of FP8 KV Cache** <br>
* **2 x NVIDIA DGX Spark 128 GB with 2M tokens of FP8 KV Cache** <br>

This model is ready for commercial or non-commercial use.  <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA [(GLM-5.3-Flash) Model Card](https://huggingface.co/zai-org/GLM-5.3-Flash) from ZAI.

### License/Terms of Use:
**GOVERNING DOWNLOAD TERMS:** Use of this model is governed by [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

**ADDITIONAL INFORMATION** : The GLM-5.3-Flash base model is governed by the [MIT License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) . **GLM-5.3-Flash**).

### Deployment Geography:
Global <br>

### Use Case: <br>
Developers looking to take off-the-shelf, pre-quantized models for deployment in AI Agent systems, chatbots, RAG systems, and other AI-powered applications. <br>

### Release Date:  <br>
Huggingface 10/05/2026 via https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local <br>

## References
NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer

## Model Architecture:
**Architecture Type:** Transformers  <br>
**Network Architecture:** Hybrid sparse and linear attention with Manifold-Constrained Hyper-Connections (`Glm5NextForConditionalGeneration`) <br>
**Number of Model Parameters:** 320B in total and 18B activated <br>
**This model was developed based on [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** <br>

## Input:
**Input Type(s):** Text, Image, Video <br>
**Input Format(s):** String, Red, Green, Blue (RGB), Video (MP4/WebM) <br>
**Input Parameters:** One-Dimensional (1D), Two-Dimensional (2D), Three-Dimensional (3D) <br>
**Other Properties Related to Input:** Context length up to 1M <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** One-Dimensional (1D): Sequences <br>
**Other Properties Related to Output:** None <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Supported Runtime Engine(s):** <br>
* **vLLM** <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell <br>

**Preferred Operating System(s):** <br>
* Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
The model version is FP4 QAD 1.0 and is quantized with nvidia-modelopt **v0.47.0** <br>

## Training and Evaluation Datasets:

## Training Dataset:
**Link**: Subset of [Nemotron-Post-Training-Dataset-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) and subset of[Nemotron-Pretraining-Dataset](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) <br> only prompts from the datasets were used for data synthesis.
**Data Modality:** Text <br>
**Text Training Data Size:** 500 Million Tokens, repeated for 3 epochs <br>
**Data Collection Method by dataset:** Hybrid: Automated, Manually-Collected, Synthetic <br>
**Labeling Method by dataset:** Hybrid: Automated, Manually-Labeled, Synthetic <br>
**Properties:** The GLM-5.3-Flash-FP4-Local model was trained exclusively on a post-training corpus includes synthetic data for reasoning, code, science, tool use, instruction following, structured outputs, and multilingual tasks. Dataset disclosures are published in the `nvidia/nemotron-post-training-v2` and subset of `nvidia/nemotron-post-training-v3` and subset of `nvidia/nemotron-pre-training-datasets` Hugging Face collections. <br>

## Evaluation

**Datasets:** GPQA Diamond, ToolEvalBench <br>
**Data Collection Method by dataset:** Hybrid: Automated, Manually-Collected <br>
**Labeling Method by dataset:** Hybrid: Manually-Labeled, Automated <br>
**Properties:** GPQA Diamond contains 198 graduate-level multiple-choice questions written by domain experts in biology, physics, and chemistry. ToolEvalBench is a tool-calling quality benchmark that runs deterministic multi-turn agent scenarios against mock tools, scoring tool selection, parameter precision, multi-step chains, error recovery, and structured output. <br>

The accuracy benchmark results are presented in the table below:

| Precision | GPQA Diamond | ToolEvalBench |
|:---|:---:|:---:|
| BF16 | 92.1% | 91.9% |
| **FP4-Local** | **91.3%** | **92.4%** |

> Baseline: [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

## Quantization-Aware Distillation
The BF16 teacher generates chat responses and supplies probability and hidden-state targets for the student.

During training, the student's routed-expert weights are quantized to NVFP4 and reconstructed for each forward pass. Gradients update the underlying BF16 weights so their low-precision representations better match the teacher's next-token probabilities and final hidden states.

The student uses its own expert routing during training. Routers, shared experts, dense MLPs and normalization weights are also adapted, allowing the unquantized components to compensate alongside the NVFP4 experts.

### What's quantized

| Component | Stored precision | Distillation |
|:---|:---|:---|
| Decoder routed experts | QAD NVFP4; E2M1 values, E4M3 block scales, FP32 global/input scales | Trained |
| Decoder attention | MXFP8 projection selection from the pinned source | Trained |
| MTP routed experts | NVFP4, BF16 activations; block scales stored directly | Trained |
| Vision attention linears | MXFP8 E4M3 values; UE8M0 scales per 32 weights; 48 matrices | Trained |
| Vision MLP and merger linears | NVFP4 E2M1 values; E4M3 scales per 16 weights and FP32 global scale; 76 matrices; BF16 activations | Updated during distillation |
| Other vision tensors | BF16; 223 tensors, including normalization, biases, position embeddings, and convolutions | Frozen |
| Normalization parameters | 277 BF16 tensors; original bytes preserved | Unchanged; outside text distillation |

## Inference:
**Acceleration Engine:** **vLLM** <br>
**Test Hardware:** **NVIDIA DGX Spark and NVIDIA RTX PRO 6000 Blackwell** <br>

## Usage

This checkpoint requires community vLLM containers (provided in cooperation with the "Local Inference Lab" community organization) until all features are available in upstream vLLM.

### 2 x NVIDIA DGX Spark

vLLM container: [`eugr/spark-vllm-b12x:latest`](https://hub.docker.com/r/eugr/spark-vllm-b12x/tags?name=latest)

One step launcher:

```sh
git clone https://github.com/eugr/spark-vllm-docker
cd spark-vllm-docker
./run-recipe.sh recipes/glm-5.3-flash.yaml --setup
```

Manual launch:

Start the container on both nodes. Set `HF_TOKEN` in the host environment, and set the network values below to match your setup.

```sh
# Example two-Spark values (adjust to your network):
#   HEAD_IP="10.100.32.1"        # head node IP on the ETH subnet
#   ETH_IF="enp1s0f0np0"        # ConnectX baseboard ethernet (socket/control)
#   IB_IF="rocep1s0f0,roceP2p1s0f0"   # RoCE HCA names (data plane) - use both "twins"!
export HEAD_IP="<head-connectx-ip>"
export ETH_IF="<connectx-interface>"
export IB_IF="<roce-interface-1>,<roce-interface-2>"
export MASTER_PORT=29501
export IMAGE="eugr/spark-vllm-b12x:latest"
export HF_CACHE_DIR="$HOME/.cache/huggingface"

mkdir -p "$HF_CACHE_DIR" "$HOME/.cache/vllm" "$HOME/.cache/flashinfer" "$HOME/.triton" "$HOME/.cache/b12x"

docker run --rm -it \
  --privileged \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size=16g \
  --ulimit nofile=1048576:1048576 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MN_IF_NAME="$ETH_IF" \
  -e UCX_NET_DEVICES="$ETH_IF" \
  -e NCCL_SOCKET_IFNAME="$ETH_IF" \
  -e NCCL_IB_HCA="$IB_IF" \
  -e NCCL_IB_DISABLE=0 \
  -e OMPI_MCA_btl_tcp_if_include="$ETH_IF" \
  -e GLOO_SOCKET_IFNAME="$ETH_IF" \
  -e TP_SOCKET_IFNAME="$ETH_IF" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e HEAD_IP="$HEAD_IP" \
  -e MASTER_PORT="$MASTER_PORT" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/.triton:/root/.triton" \
  -v $HOME/.cache/b12x:/root/.cache/b12x \
  --entrypoint /bin/bash \
  "$IMAGE"
```

Both nodes share the same environment:

```sh
export CUTE_DSL_ARCH="sm_121a"
export SAFETENSORS_FAST_GPU="1"
export VLLM_ENABLE_ROCE_ALLREDUCE="1"
export VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_CTAS=8
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_SSM_CONV_STATE_LAYOUT="DS"
export VLLM_USE_AOT_COMPILE="1"
export VLLM_USE_MEGA_AOT_ARTIFACT="1"
export VLLM_USE_V2_MODEL_RUNNER="1"
export VLLM_ENABLE_PCIE_ALLREDUCE="0"
export B12X_POLICY_MODE="auto"
export VLLM_B12X_MOE_FP4_FORCE_A16=1
export B12X_W4A16_A4_PREFILL=1
```

Launch the head node:

```sh
vllm serve nvidia/GLM-5.3-Flash-FP4-Local \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --attention-backend B12X \
  --block-size 256 \
  --moe-backend b12x \
  --linear-backend b12x \
  --no-enable-flashinfer-autotune \
  --load-format b12x \
  --max-model-len 500000 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"moe_backend":"humming","attention_backend":"B12X"}' \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.87 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "$HEAD_IP" \
  --master-port "$MASTER_PORT"
```

Launch the worker node:

```sh
vllm serve nvidia/GLM-5.3-Flash-FP4-Local \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --attention-backend B12X \
  --block-size 256 \
  --moe-backend b12x \
  --linear-backend b12x \
  --no-enable-flashinfer-autotune \
  --load-format b12x \
  --max-model-len 500000 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"moe_backend":"humming","attention_backend":"B12X"}' \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.87 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr "$HEAD_IP" \
  --master-port "$MASTER_PORT" \
  --headless
```

> Use the same CX7 port on every node; assign both port subinterfaces on separate subnets; cache the image and checkpoint on every node; start workers before the head. Docker runs privileged.

### 2 x NVIDIA RTX PRO 6000 Blackwell

vLLM container: [`ghcr.io/local-inference-lab/vllm:release-10142026`](https://github.com/local-inference-lab/blackwell-llm-docker/pkgs/container/vllm)

Simplified run command:

```sh
  docker run -d --name glm53-tp2 --init --gpus '"device=0,1"' \
    --restart on-failure --network host --ipc host --shm-size 32g --stop-timeout 30 \
    --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
    -v model-cache:/root/.cache/huggingface -v glm53-runtime:/cache \
    -e PRESET=glm53-tp2 -e PORT=8000 \
    ghcr.io/local-inference-lab/vllm:release-10152026
```

Two step verbose commands:

Start the container. Set `HF_TOKEN` in the host environment first.

```sh
docker run --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  ghcr.io/local-inference-lab/vllm:release-10152026
```

Then launch vLLM inside the container:

```sh
export B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1
export B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1
export B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224
export B12X_DYNAMIC_SPLIT_FAST_PREPARE=1
export B12X_DYNAMIC_SPLIT_LOW_SMEM=1
export B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1
export B12X_DYNAMIC_WORK_SOURCE=persistent_grid
export B12X_MHC_PDL=1
export B12X_PCIE_ONESHOT_BLOCK_LIMIT=4
export B12X_PCIE_ONESHOT_PDL=1
export B12X_PCIE_ONESHOT_THREADS=512
export B12X_W4A16_A4_PREFILL=1
export B12X_W4A16_FP32_TOPK_WEIGHTS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUTE_DSL_ARCH=sm_120a
export GLOO_SOCKET_IFNAME=lo
export INSTANTTENSOR_BACKEND=BUFFERED
export NCCL_BUFFSIZE=1048576
export NCCL_IB_DISABLE=1
export NCCL_MAX_NCHANNELS=2
export NCCL_MIN_NCHANNELS=2
export NCCL_NET_PLUGIN=none
export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO=LL,LL128,Simple
export NCCL_SOCKET_IFNAME=lo
export NCCL_TUNER_PLUGIN=none
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:12
export SAFETENSORS_FAST_GPU=1
export TORCH_CUDA_ARCH_LIST=12.0a
export VLLM_B12X_MLA_CKV_GATHER=1
export VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=65536
export VLLM_B12X_MOE_FP4_FORCE_A16=1
export VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1
export VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export VLLM_GLM53_DFLASH_ATTN=1
export VLLM_GLM53_EMBED_HOST=1
export VLLM_GLM53_KDA_GATE_SIDE_STREAM=1
export VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE=prefill
export VLLM_GLM53_L2_PREFETCH=1
export VLLM_GLM53_L2_PREFETCH_PERSIST_MB=0
export VLLM_GLM53_MTP_DRAFT_HEAD=bf16
export VLLM_GLM53_ONLINE_DENSE_MXFP8=0
export VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto
export VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048
export VLLM_LM_HEAD_A16=1
export VLLM_MTP_NVFP4_LM_HEAD=0
export VLLM_MXFP8_LM_HEAD=0
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PCIE_DMA_MIN_BYTES=6MB
export VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=off
export VLLM_PLUGINS=''
export VLLM_SHARE_PYNCCL_COMMS=1
export VLLM_USE_FASTOKENS=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve nvidia/GLM-5.3-Flash-FP4-Local \
  --host 0.0.0.0 \
  --port 8000 \
  --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
  --attention-backend B12X \
  --block-size 256 \
  --chat-template /opt/lil/runtime/templates/glm53-flash.jinja \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --cp-kv-cache-interleave-size 4 \
  --cudagraph-capture-sizes 1 2 4 8 12 16 20 24 28 32 \
  --dcp-kv-cache-interleave-size 4 \
  --decode-context-parallel-size 2 \
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":false}' \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.985 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 7650410496 \
  --linear-backend b12x \
  --load-format instanttensor \
  --mamba-cache-mode align \
  --max-cudagraph-capture-size 32 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --max-parallel-prefills 1 \
  --moe-backend b12x \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --prefill-compute-share 0.4 \
  --prefill-schedule-interval 1 \
  --quantization modelopt_mixed \
  --reasoning-parser glm45 \
  --recurrent-checkpoint-policy request_boundaries \
  --speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","attention_backend":"B12X","num_speculative_tokens":3}' \
  --tensor-parallel-size 2 \
  --tool-call-parser glm47
```

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards below.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

SUBCARDS:
## Explainability


| Field                                    | Response                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|:----------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| **Intended Task/Domain:**                | Text generation, reasoning, summarization, and question answering.                                                                                                                                                                                                                                                                                                                                                                                       |
| **Model Type:**                          | Text, Image, and Video-to-text transformer                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Intended Users:**                      | This model is intended for developers, researchers, and customers building/utilizing LLMs, while balancing accuracy and efficiency.                                                                                                                                                                                                                                                                                                                     |
| **Output:**                              | Text String(s)                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **Describe how the model works:**        | Generates text by predicting the next word or token based on the context provided in the input sequence using multiple self-attention layers.                                                                                                                                                                                                                                                                                                            |
| **Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:**        | Not Applicable    |
| **Technical Limitations & Mitigation:**  | The model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. Therefore, before deploying any applications of this model, developers should perform safety testing and tuning tailored to their specific applications of the model.                                       |
| **Verified to have met prescribed NVIDIA quality standards:** | Yes                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Performance Metrics:**                 | Accuracy, Throughput, and user-side throughput                                                                                                                                                                                                                                                                                                                                                                                                           |
| **Potential Known Risk**                 | The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.                                                                                                                                                                                                       |
| **Licensing:**                           | **GOVERNING DOWNLOAD TERMS:** Use of this model is governed by [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

**ADDITIONAL INFORMATION** : The GLM-5.3-Flash base model is governed by the [MIT License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) . **GLM-5.3-Flash**).                                                                                                                           |


## Bias

| Field                                                                                                           | Response |
|:---------------------------------------------------------------------------------------------------------------:|:--------:|
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/protected-classes) in model design and testing:       | None     |
| Measures taken to mitigate against unwanted bias                                                                | None     |


## Safety & Security

| Field                              | Response                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|:-----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Model Application Field(s):         | Chat, Instruction Following, Chatbot Development, Code Generation, Reasoning                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Describe the life critical impact (if present) | Not Applicable                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Use Case Restrictions:              | Abide by the **GOVERNING DOWNLOAD TERMS:** Use of this model is governed by [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

**ADDITIONAL INFORMATION** : The GLM-5.3-Flash base model is governed by the [MIT License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) . **GLM-5.3-Flash**).                                                                                                                                                                                                                   |
| Model and Dataset Restrictions:     | The Principle of least privilege (PoLP) is applied limiting access for dataset generation. Restrictions enforce dataset access during training, and dataset license constraints adhered to. Model checkpoints are made available on Hugging Face, and may become available on cloud providers' model catalog.                                                                                                                                                                                                                                   |


## Privacy


| Field                                                                          | Response                                                                           |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| **Generatable or reverse-engineerable personal data**                          | No 
| **Personal data used to create this model**                                    | None Known                                                                         |      |
| **Was consent obtained for any personal data used**                            | Not Applicable                                                                     |
| **How often is the dataset reviewed**                                          | Before Release                                                                     |
| **Was data from user interactions with the AI model used to train the model**  | No                                                                                 |
| **Is there provenance for all datasets used in training**                      | Yes                                                                                |
| **Does data labeling comply with privacy laws**                                | Yes                                                                                |
| **Is the data compliant with data-subject requests for correction or removal** | No, not possible with externally-sourced data.                                     |
| **Applicable NVIDIA Privacy Policy**                                           | [NVIDIA Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) |                 |
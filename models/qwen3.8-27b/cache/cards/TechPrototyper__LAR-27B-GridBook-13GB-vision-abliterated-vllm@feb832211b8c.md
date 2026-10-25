---
license: apache-2.0
base_model:
- amazon/LAR-27B
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: vllm
pipeline_tag: image-text-to-text
tags:
- lar
- qwen3
- gridbook
- codebook-quantization
- nvfp4
- fp8
- vllm
- abliterated
- vision
- mtp
- long-context
inference: false
---

## Warning: this does not run in vanilla vLLM. It needs Robert Tand's `gridbook==0.8.8` plugin and `nvcc` (see How to run).

# LAR-27B — GridBook 13 GB, vision-enabled, abliterated in format

## Credits first: Robert Tand

**Made with PrismaQuant by Robert Tand - https://github.com/RobTand/prismaquant**

Everything that makes this a 13 GB model is Robert Tand's work, not ours. His
**[PrismaQuant](https://github.com/RobTand/prismaquant)** toolkit is the reason we
could quantize a 27B model ourselves with a real end-to-end-KL-driven allocator
instead of a uniform bit width. His **[GridBook](https://github.com/RobTand/gridbook)**
format and vLLM plugin turn that allocation into product-vector codebooks on a hardware
grid, with every Linear in its own format and codeword width, and serve it with native
CUDA/CUTLASS kernels on consumer Blackwell. The 5080 allocation used here (the formats
assigned to each body Linear, ~3.6 bpp) is his, carried over unchanged from
[`rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm`](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-gridbook-13GB-5080-vllm).
He publishes his methods, his measured numbers (including the unflattering ones) and
his tooling openly, and he answers questions. A 27B agentic model with a vision tower
and a 262k context on a single 32 GB card exists because of that work. Thank you, Rob.

If you use this checkpoint, please cite PrismaQuant:

```bibtex
@software{prismaquant2026,
  title  = {PrismaQuant: Mixed-Precision LLM Quantization Selected on Real End-to-End KL},
  author = {Tand, Robert},
  year   = {2026},
  url    = {https://github.com/RobTand/prismaquant},
}
```

Further credits:

- **Base model: [`amazon/LAR-27B`](https://huggingface.co/amazon/LAR-27B)** (Apache-2.0), the
  Lean4AgenticRL model by Ruida Wang, Gerald Friedland, Tong Zhang and Chris Kong
  ([arXiv 2610.22448](https://arxiv.org/abs/2610.22448)). This is an independent community
  derivative. It is **not affiliated with, reviewed or endorsed by Amazon** or the LAR authors.

  ```bibtex
  @misc{wang2026lean4agenticrl,
        title={Lean4AgenticRL: Advancing Agentic RL with Verifiable Multi-level Process Rewards},
        author={Ruida Wang and Gerald Friedland and Tong Zhang and Chris Kong},
        year={2026},
        eprint={2610.22448},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2610.22448},
  }
  ```
- **Architecture, vision tower and MTP head: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)** (Apache-2.0).
- **Abliteration direction: [`trohrbaugh/Qwen3.8-27B-heretic-ara`](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara)**
  (Apache-2.0), made with **[Heretic](https://github.com/p-e-w/heretic)** by p-e-w (ARA mode).

## What this is

`amazon/LAR-27B`, quantized by us with PrismaQuant into the GridBook `nvfp4_cb` format using
Rob's 5080 allocation, with the Qwen3.8 vision tower and MTP head attached, and with refusal
behaviour reduced by an **in-format** edit (no requantization after the edit).

Changes relative to `amazon/LAR-27B` (Apache-2.0 §4b):

| | `amazon/LAR-27B` | this checkpoint |
|---|---|---|
| Body Linears | BF16 | GridBook `NVFP4_CB` / `FP8_CB`, Rob's 5080 allocation, ~3.6 bpp |
| `model.embed_tokens` | BF16 | BF16 (Rob's release: NVFP4) |
| `lm_head` | BF16 | FP8 E4M3 + per-row scale |
| Vision tower | none (text only) | Qwen3.8 tower, 333 tensors BF16 passthrough, `visual.safetensors` |
| MTP head | none | Qwen3.8 MTP, 15 tensors BF16, `mtp.safetensors` |
| Refusal behaviour | unchanged | reduced in format: abliteration delta transferred, 60 tensors re-encoded |
| Activation scales / imatrix | n/a | our own probe on LAR (Rob's imatrix was not available for LAR) |

The abliteration is a **delta transfer**: for the `o_proj` / `out_proj` / `down_proj` matrices
of layers 26–55 we take `W_heretic-ara − W_Qwen3.8-27B` and add it to LAR's BF16 weights, then
re-encode only those targets inside the existing GridBook checkpoint with the production
encoder against frozen scale planes. Targets on the coarse grids (`NVFP4_CB_K12/14/16`,
`FP8_CB_K32/40`) are left bit-identical. All other tensors are bit-identical to the
non-abliterated GridBook build (arm L1 below).

Calibration data is **not included**. The probe corpus is PrismaQuant's `diverse-v1` recipe
(256 rows × 4096 tokens, seed 20260509, 40 % prose / 20 % code / 20 % math / 20 % multilingual)
built from: `HuggingFaceFW/fineweb-edu` (sample-10BT), `codeparrot/codeparrot-clean` (substitute
for the gated `bigcode/starcoderdata`), `EleutherAI/proof-pile-2` (algebraic-stack),
`open-web-math/open-web-math`, `Muennighoff/flores200`, `hlillemark/flores200_8_val_test` and
`HuggingFaceFW/fineweb-2`. Exact revisions are in `stage-manifest.json`.

Architecture: `Qwen3_5ForConditionalGeneration`, 64 layers (16 attention, 48 SSM),
`quant_method=gridbook`, `format=nvfp4_cb`, `layout_version=2`, codebooks in `cb_codebooks.pqcb`,
224 `ignore` entries. Files: `model.safetensors` (13.94 GB), `visual.safetensors` (0.92 GB),
`mtp.safetensors` (0.85 GB). Per-file sha256 in `STAGE-MANIFEST.md`; local build paths in the
manifests are redacted.

## Measurements

Our own harness, not public benchmarks. Arms: **REF** = `amazon/LAR-27B` BF16 (text only);
**L1** = LAR GridBook + vision + MTP, not abliterated; **L2** = L1 + in-format abliteration =
**this checkpoint**; **Q0** = our previous production model
[`TechPrototyper/Qwen3.8-27B-GridBook-13GB-vision-abliterated-vllm`](https://huggingface.co/TechPrototyper/Qwen3.8-27B-GridBook-13GB-vision-abliterated-vllm).
Measured on a DGX Spark (GB10, sm_121): vLLM 7d0b4e57 + FlashInfer 0.7.1 + gridbook 0.8.8,
`--gpu-memory-utilization 0.45` (REF 0.72), max-len 40960, max-num-seqs 8, no speculative decoding
unless noted, thinking disabled.

| Metric | REF | L1 | **L2 (this)** | Q0 | Method |
|---|---|---|---|---|---|
| PPL de / en / code | 6.800 / 3.556 / 1.507 | 7.236 / 4.823 / 1.821 | **7.278 / 4.869 / 1.833** | 7.250 / 4.896 / 1.840 | teacher-forced logprobs, frozen 65,437-token corpus |
| KL vs REF (mean) | – | 0.218 | **0.223** | n/a (other base) | top-20 approximation; L2 vs L1: KL 0.049, PPL +0.74 % |
| GSM8K | 0.976 (123/200 \*) | 0.965 (193/200) | **0.964 (189/196)** | 0.968 (180/186) | paired n=200, greedy; McNemar L1–L2 p=1.0, L2–Q0 p=1.0 |
| Refusals hard / soft / none (n=100) | 32/0/0 of 32 (stopped) | 100 / 0 / 0 | **39 / 51 / 10** | 33 / 59 / 8 | Heretic eval prompts via API, strict marker classifier |
| Refusals, first-token gate (BF16, before encoding) | base 34/100 | – | target 0/100, KL 0.033 | – | Heretic evaluator on the BF16 delta target |
| Tool calls (fixed 5-scenario suite) | 4/5 (old suite \*\*) | 5/5 | **5/5** | 5/5 | 5 serial repeats, greedy |
| Vision (synthetic text images) | no tower | 2/3 | **2/3** | 1/3 | 3-image OCR gate |
| Needle 32k / determinism | 3/3 / 5/5 | 3/3 / 5/5 | **3/3 / 5/5** | 3/3 / 5/5 | depths 0.1/0.5/0.9; temp 0 × 5 |
| Prefill 2k / 8k tok/s (Spark) | – | 1111 / 1164 | **1107 / 1159** | 1110 / 1162 | median, salted prompts |
| Decode c=1 / c=4 aggregate tok/s (Spark) | – | 15.0 / 40.2 | **15.1 / 28.5** | 15.1 / 40.2 | 768 tokens, T 0.6 |
| MTP n=2 acceptance / c=1 tok/s (Spark) | – | 0.787 / 20.3 | **0.792 / 20.7** | – | vLLM spec-decode metrics over tool + speed + GSM8K-50 runs; GSM8K-50 under MTP 50/50 |
| KV tokens (Spark, util 0.45) | 502,627 (util 0.72) | 486,660 | ~486k | 482,494 | vLLM startup log |

\* The BF16 REF GSM8K run was cut by the time budget after 123 of 200 items. \*\* The old tool
suite rejected a valid `workspace/list` call in one scenario; the fixed suite accepts it.

**RTX 5090 (32 GB, sm_120), production configuration below:** KV cache **765,668 tokens** at
262,144 max-len with NVFP4 KV and `--gpu-memory-utilization 0.97`; single-stream decode
**~61 tok/s** including prefill; our RTX deployment gate passed before the switch to production.

How to read this honestly:

- **Abliteration is partial.** This checkpoint refuses 39 of 100 adversarial prompts outright,
  slightly *more* than our previous Qwen production model (33). The edit is far stronger in BF16
  (0/100 first-token refusals) than after re-encoding on a ~3.6 bpp grid. It is not an
  uncensored model.
- The ~20 % PPL gap to REF is the cost of GridBook at ~3.6 bpp; the Qwen-based Q0 carries the
  same gap on this corpus. The abliteration itself adds +0.74 % PPL over L1.
- The c=4 decode drop of L2 vs L1 (28.5 vs 40.2) was measured on the Spark only and is not
  explained yet; single-stream speed is unchanged.
- The refusal counter is a keyword classifier; "soft" means hedged or partial answers, not
  compliance. We did not measure coding agent benchmarks (SWE-bench, Terminal-Bench) on this
  quantized build; LAR's published scores are for the BF16 model.

## How to run

The same images serve this model and the previous
[`TechPrototyper/Qwen3.8-27B-GridBook-13GB-vision-abliterated-vllm`](https://huggingface.co/TechPrototyper/Qwen3.8-27B-GridBook-13GB-vision-abliterated-vllm)
(swap the repo name and `--served-model-name`; everything else is identical).

Images (vLLM nightly 7d0b4e57 + gridbook 0.8.8), public on GHCR:

| Tag | Platform | Digest |
|---|---|---|
| `ghcr.io/techprototyper/vllm-gridbook:7d0b4e57-gb088` | multi-arch index | `sha256:355cd9460a32d10598f30ec8f898c19cb075d3a4df28c867310e4ffec7fea489` |
| `ghcr.io/techprototyper/vllm-gridbook:7d0b4e57-gb088-sm120-amd64` | linux/amd64, RTX 50xx (sm_120); + FlashInfer 0.7.1 + vLLM PR #46558 (NVFP4 KV on sm_120) | `sha256:8a12b20302fd12a029f01b5abc36889e15e6fc75ddcc17a4fcb1a4865e06829e` |
| `ghcr.io/techprototyper/vllm-gridbook:7d0b4e57-gb088-sm121-arm64` | linux/arm64, DGX Spark (sm_121); stock FlashInfer; gridbook main/v2/bf16-grouped kernels prebuilt for sm_121, CUDA 13.0 headers added | `sha256:ba142c7d1b408fa1b3db0e9ec22ce314f1a93f15ba56a38e09143ffdc028c725` |

What is verified where:

| | RTX 5090 (amd64 image) | DGX Spark (arm64 image) |
|---|---|---|
| Settings below | **verified**, our production since 2026-10-25 | settings (util 0.45, max-len 40960, 8 seqs, NVFP4 KV) **verified with a different local image**; the published arm64 image is **not verified end-to-end** (build-time kernel compile + CPU import test only, no GPU serve test) |
| Kubernetes manifest / Helm chart | schema-validated (`kubeconform`, `helm lint`), structure mirrors our production Deployment; **not applied as-is** | same, untested |

### 1. Docker on DGX Spark / GB10 (arm64, sm_121)

```bash
docker pull ghcr.io/techprototyper/vllm-gridbook:7d0b4e57-gb088-sm121-arm64
pip install -U huggingface_hub
hf download TechPrototyper/LAR-27B-GridBook-13GB-vision-abliterated-vllm \
  --local-dir ~/models/LAR-27B-GridBook-13GB-vision-abliterated-vllm --exclude "deploy/*"

docker run -d --name lar-gridbook --gpus all --ipc=host --shm-size 8g -p 8000:8000 \
  -v ~/models:/models:ro -v gridbook-ext:/opt/gridbook/ext-cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint python3 \
  ghcr.io/techprototyper/vllm-gridbook:7d0b4e57-gb088-sm121-arm64 \
  -m vllm.entrypoints.openai.api_server \
  --model /models/LAR-27B-GridBook-13GB-vision-abliterated-vllm \
  --served-model-name lar-27b-gridbook \
  --trust-remote-code --reasoning-parser qwen3 --host 0.0.0.0 --port 8000 \
  --generation-config vllm \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}' \
  --kv-cache-dtype fp8 \
  --max-model-len 40960 --max-num-seqs 8 \
  --gpu-memory-utilization 0.45 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image": 2, "video": 0}' \
  --enable-prefix-caching --enable-chunked-prefill
```

- 128 GB is unified memory shared with the OS: `0.45` left room for other services in our
  measurements (~486k KV tokens with our NVFP4-KV image); raise it if the box serves only this
  model. The stock FlashInfer in this image is why the example uses `fp8` KV (not verified).
- No `CUDA_HOME`/header workaround is needed with this image (headers are baked in). The fused FP8
  kernel is compiled once on first load (needs >8 GiB free host RAM, a few minutes) and kept in the
  `gridbook-ext` volume.
- Optional MTP: add `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'`
  (acceptance ~0.79, c=1 15 → ~21 tok/s on the Spark in our measurements).

### 2. Docker on RTX 5090 / sm_120 (amd64)

```bash
docker pull ghcr.io/techprototyper/vllm-gridbook:7d0b4e57-gb088-sm120-amd64
hf download TechPrototyper/LAR-27B-GridBook-13GB-vision-abliterated-vllm \
  --local-dir ~/models/LAR-27B-GridBook-13GB-vision-abliterated-vllm --exclude "deploy/*"
# CUDA 13.0 library headers for gridbook's kernel JIT (this image lacks cusparse/cublas/cusolver headers):
mkdir -p ~/cuda-include-13.0 && docker run --rm -v ~/cuda-include-13.0:/out nvidia/cuda:13.0.2-devel-ubuntu24.04 \
  sh -c 'cd /usr/local/cuda/include && cp -a cusparse*.h cublas*.h cusolver*.h cufft*.h curand*.h /out/'

docker run -d --name lar-gridbook --gpus all --ipc=host --shm-size 8g -p 8000:8000 \
  -v ~/models:/models:ro -v ~/cuda-include-13.0:/cuda-include:ro -v gridbook-ext:/ext \
  -e CPATH=/cuda-include -e PRISMAQUANT_CB_EXT_DIR=/ext -e TORCH_CUDA_ARCH_LIST=12.0 \
  -e VLLM_NVFP4_KV_SPLIT_KV=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint python3 \
  ghcr.io/techprototyper/vllm-gridbook:7d0b4e57-gb088-sm120-amd64 \
  -m vllm.entrypoints.openai.api_server \
  --model /models/LAR-27B-GridBook-13GB-vision-abliterated-vllm \
  --served-model-name lar-27b-gridbook \
  --trust-remote-code --reasoning-parser qwen3 --host 0.0.0.0 --port 8000 \
  --generation-config vllm \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}' \
  --kv-cache-dtype nvfp4 \
  --attention-config '{"use_trtllm_attention": false}' --attention-backend FLASHINFER \
  --max-model-len 262144 --max-num-seqs 8 --block-size 32 \
  --gpu-memory-utilization 0.97 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image": 2, "video": 0}' \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4]}' \
  --async-scheduling
```

This is our production configuration: **~765k KV tokens** at 262,144 context, ~61 tok/s single
stream. The header step is the workaround our production uses (headers via `CPATH`); pip
`nvidia-*` headers (CUDA 13.4) do **not** compile against this image's `nvcc` 13.0. The first start
compiles gridbook's kernels (several minutes); the `gridbook-ext` volume keeps them.
Optional MTP: `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'` — it takes
KV budget (draft weights/activations come out of the pool), so production runs without it.
Serve without speculative decoding if you need prompt logprobs.

### Test

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "lar-27b-gridbook",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": false}}' | python3 -m json.tool
```

### 3. Kubernetes

Copy-paste manifest: [`deploy/k8s/lar-gridbook.yaml`](deploy/k8s/lar-gridbook.yaml) (also below) —
PVC, an initContainer that downloads from HF only if the model is missing (temp dir + atomic `mv`),
an initContainer that stages the CUDA 13.0 headers (amd64 only), the Deployment (`Recreate`,
`nvidia.com/gpu: 1`, `/dev/shm` as memory `emptyDir`, readiness on `/health`, a 40-minute
startupProbe for load + first kernel compile) and a ClusterIP Service. Defaults are the RTX settings;
lines marked `ARM64:` give the Spark values.

```bash
kubectl apply -f https://huggingface.co/TechPrototyper/LAR-27B-GridBook-13GB-vision-abliterated-vllm/resolve/main/deploy/k8s/lar-gridbook.yaml
kubectl port-forward svc/vllm-gridbook 8000:8000
```

Convenience path: Helm chart in [`deploy/helm/vllm-gridbook/`](deploy/helm/vllm-gridbook/)
(`helm lint` clean, `helm template` validated for both arches), values `arch`, `model.repo`,
`image.tag`, `gpuMemoryUtilization`, `maxModelLen`, `kvCacheDtype`, `mtp`:

```bash
hf download TechPrototyper/LAR-27B-GridBook-13GB-vision-abliterated-vllm --include "deploy/*" --local-dir lar
helm install lar lar/deploy/helm/vllm-gridbook --set arch=amd64            # RTX
helm install lar lar/deploy/helm/vllm-gridbook --set arch=arm64 --set mtp=true   # DGX Spark
```

<details><summary>deploy/k8s/lar-gridbook.yaml</summary>

```yaml
# LAR-27B GridBook (vision, abliterated) on vLLM + gridbook 0.8.8 - single-GPU Kubernetes deployment.
# Made with PrismaQuant by Robert Tand - https://github.com/RobTand/prismaquant
# Defaults = RTX 5090 (amd64, sm_120) production settings. For DGX Spark / GB10 (arm64) see the comments marked ARM64.
# Swap MODEL_REPO to TechPrototyper/Qwen3.8-27B-GridBook-13GB-vision-abliterated-vllm for the previous Qwen build (same image).
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-gridbook-cache
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 60Gi            # 15.7 GB weights + kernel cache + headroom
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-gridbook
  labels: {app: vllm-gridbook}
spec:
  replicas: 1
  strategy: {type: Recreate}   # one GPU, never two pods at once
  selector:
    matchLabels: {app: vllm-gridbook}
  template:
    metadata:
      labels: {app: vllm-gridbook}
    spec:
      # nodeSelector: {kubernetes.io/arch: amd64}   # ARM64: arm64
      initContainers:
      # 1) Download the model only if it is not on the PVC yet; download into a temp dir, then atomic mv.
      - name: fetch-model
        image: python:3.12-slim
        env:
        - {name: MODEL_REPO, value: "TechPrototyper/LAR-27B-GridBook-13GB-vision-abliterated-vllm"}
        - {name: MODEL_NAME, value: "LAR-27B-GridBook-13GB-vision-abliterated-vllm"}
        - {name: HF_HUB_DISABLE_PROGRESS_BARS, value: "1"}
        # - name: HF_TOKEN            # not needed for public repos
        #   valueFrom: {secretKeyRef: {name: hf-token, key: token}}
        command: ["sh", "-c"]
        args:
        - |
          set -e
          D=/cache/models/$MODEL_NAME
          if [ -f "$D/model.safetensors" ] && [ -f "$D/.complete" ]; then echo "model present"; exit 0; fi
          pip install -q "huggingface_hub>=0.34"
          T=/cache/models/.tmp-$MODEL_NAME; rm -rf "$T"; mkdir -p "$T"
          hf download "$MODEL_REPO" --local-dir "$T" --exclude "deploy/*"
          touch "$T/.complete"; rm -rf "$D"; mv "$T" "$D"; echo "model ready"
        volumeMounts: [{name: cache, mountPath: /cache}]
      # 2) amd64 image only: the vLLM nightly lacks cusparse/cublas/cusolver headers that gridbook's JIT needs.
      #    Copy the CUDA 13.0 headers (matching nvcc 13.0.88) to the PVC once. ARM64: delete this initContainer
      #    and the CPATH env below (the arm64 image ships the headers).
      - name: cuda-headers
        image: nvidia/cuda:13.0.2-devel-ubuntu24.04
        command: ["sh", "-c"]
        args:
        - |
          set -e
          H=/cache/cuda-include-13.0
          [ -f "$H/cusparse.h" ] && { echo "headers present"; exit 0; }
          mkdir -p "$H.tmp"; cd /usr/local/cuda/include
          cp -a cusparse*.h cublas*.h cusolver*.h cufft*.h curand*.h "$H.tmp"/
          mv "$H.tmp" "$H"
        volumeMounts: [{name: cache, mountPath: /cache}]
      containers:
      - name: vllm
        image: ghcr.io/techprototyper/vllm-gridbook:7d0b4e57-gb088-sm120-amd64   # ARM64: ...:7d0b4e57-gb088-sm121-arm64
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - --model=/cache/models/LAR-27B-GridBook-13GB-vision-abliterated-vllm
        - --served-model-name=lar-27b-gridbook
        - --trust-remote-code
        - --reasoning-parser=qwen3
        - --host=0.0.0.0
        - --port=8000
        - --generation-config=vllm
        - '--override-generation-config={"temperature":1.0,"top_p":0.95,"top_k":20}'
        - --kv-cache-dtype=nvfp4                  # ARM64: fp8
        - '--attention-config={"use_trtllm_attention": false}'
        - --attention-backend=FLASHINFER          # ARM64: remove (let vLLM choose)
        - --max-model-len=262144                  # ARM64: 40960 (measured)
        - --max-num-seqs=8
        - --block-size=32
        - --gpu-memory-utilization=0.97           # ARM64: 0.45 (unified memory, shared with the OS)
        - --enable-auto-tool-choice
        - --tool-call-parser=qwen3_coder
        - '--limit-mm-per-prompt={"image": 2, "video": 0}'
        - --enable-prefix-caching
        - --enable-chunked-prefill
        - --max-num-batched-tokens=4096
        - '--compilation-config={"cudagraph_capture_sizes": [1, 2, 4]}'
        - --async-scheduling
        # Optional MTP (costs KV budget):
        # - '--speculative-config={"method":"qwen3_5_mtp","num_speculative_tokens":2}'
        env:
        - {name: PRISMAQUANT_CB_EXT_DIR, value: /cache/gridbook-ext}   # persistent kernel cache
        - {name: TORCH_EXTENSIONS_DIR, value: /cache/torch_extensions}
        - {name: FLASHINFER_WORKSPACE_BASE, value: /cache/flashinfer}
        - {name: XDG_CACHE_HOME, value: /cache/xdg}
        - {name: HF_HOME, value: /cache/hf}
        - {name: CPATH, value: /cache/cuda-include-13.0}               # amd64 only (see initContainer 2)
        - {name: TORCH_CUDA_ARCH_LIST, value: "12.0"}                  # ARM64: 12.1a
        - {name: VLLM_NVFP4_KV_SPLIT_KV, value: "1"}                   # amd64 nvfp4 KV path
        - {name: VLLM_ALLOW_LONG_MAX_MODEL_LEN, value: "1"}
        - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
        ports: [{name: http, containerPort: 8000}]
        resources:
          requests: {nvidia.com/gpu: "1", memory: 8Gi, cpu: "3"}
          limits: {nvidia.com/gpu: "1", memory: 75Gi}
        startupProbe:              # weight load + first-use kernel compile can take 15-30 min on a cold cache
          httpGet: {path: /health, port: http}
          periodSeconds: 15
          failureThreshold: 160
        readinessProbe:
          httpGet: {path: /health, port: http}
          periodSeconds: 10
          failureThreshold: 3
        volumeMounts:
        - {name: cache, mountPath: /cache}
        - {name: dshm, mountPath: /dev/shm}
      volumes:
      - name: cache
        persistentVolumeClaim: {claimName: vllm-gridbook-cache}
      - name: dshm
        emptyDir: {medium: Memory, sizeLimit: 8Gi}
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-gridbook
spec:
  type: ClusterIP
  selector: {app: vllm-gridbook}
  ports: [{name: http, port: 8000, targetPort: http}]
```
</details>

### Without the images

`pip install gridbook==0.8.8` (wheel sha256
`a982e8842d0ce183eaad8978941a375dc984fa0697be7c4519dd741efd1153a3`) into a vLLM build that knows
`Qwen3_5ForConditionalGeneration`, with `nvcc` on `PATH` and the CUDA library headers matching it.

## Build pipeline

Stages (all CPU-or-single-GPU, run sequentially):

1. **Probe.** PrismaQuant activation probe of `amazon/LAR-27B` BF16 on our `diverse-v1` corpus.
2. **imatrix.** Column weights harvested from that probe (`build_imatrix_from_act_cache`), our own,
   since Rob's imatrix for LAR does not exist.
3. **Allocation reuse.** Layer config taken from Rob's 5080 `quant_config` (formats per Linear),
   with `embed_tokens` kept BF16; no new allocator run.
4. **Export.** `prismaquant.export_nvfp4_cb_streaming` with `--scale-coding two_tier`,
   `--codebook-source lattice`, `--activation-scale-policy mse_grid_calibrated.v1`,
   `--allow-unstamped-research`; then a serve-and-generate validation with gridbook 0.8.8.
5. **Transplant.** Qwen3.8 vision tower (333 tensors) and MTP head (15 tensors) copied in as separate
   shards, bit-identical to our previous production checkpoint; `quant_config` ignores and the
   `lm_head` target adjusted as in production.
6. **Abliteration.** Delta transfer of the Heretic-ARA direction (see above), gated in BF16
   (first-token refusals, KL) before encoding; then in-format re-encoding with frozen scales,
   alpha 1.0, coarse grids excluded, byte-range write-back.
7. **Measure.** The arms REF / L1 / L2 / Q0 above on the Spark, then an RTX gate before production.

Reproducibility notes:

- PrismaQuant at commit **`1ccdf58`** (frozen `uv.lock`), serving pin **gridbook 0.8.8**.
- `--allow-unstamped-research` is required: the transplanted allocation carries no
  `cb_render_identity` cost stamp from the allocator run. The formats are Rob's; the stamp is not.
- `--scale-coding two_tier` is required: gridbook 0.8.8 only loads FP4-CB in layout v2, and the
  repository's `v1` default is refused at load.
- imatrix and static W4A4 activation scales come from our own probe, so this is not byte-identical to
  anything Rob would produce for LAR.

## Intended use

For research and for agentic/coding use where the operator handles content policy at the
application layer. Refusal behaviour is reduced, not removed; do not rely on model-level refusals
as a safety mechanism. Not a Transformers checkpoint: `AutoModelForCausalLM` will not load it.

## License

Apache-2.0, see `LICENSE`. Inherited from `amazon/LAR-27B`, `Qwen/Qwen3.8-27B` and
`trohrbaugh/Qwen3.8-27B-heretic-ara` (all Apache-2.0). The changes are listed under
"What this is".

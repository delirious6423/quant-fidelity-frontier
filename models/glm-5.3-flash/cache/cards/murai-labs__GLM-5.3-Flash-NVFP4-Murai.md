---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- nvfp4
- quantization
- vllm
- dgx-spark
- modelopt
---

# GLM-5.3-Flash-NVFP4-Murai

**Weight-only NVFP4 (W4A16) quantization of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) for 2× NVIDIA DGX Spark (GB10, sm_121), with the deep-clip recipe fix ("Murai-P3") and a serving-time patch for a vLLM ModelOpt repack bug that corrupts output on this checkpoint lineage.**

- **This repo carries the quantized weights** (181.3 GiB, 120 safetensors) plus the release artifacts: the serving patch (`patches/`), the regression probe (`probes/`), the launcher (`launch/`), and the documentation of the recipe and the corruption fix (`docs/`).
- **Companion GitHub repo:** [Murai-Labs/GLM-5.3-Flash-NVFP4-Murai](https://github.com/Murai-Labs/GLM-5.3-Flash-NVFP4-Murai) — same artifacts, issue tracker, and development history.

Every number below traces to project logs (`SESSION_MEMORY.md` §13–§30, `cluster/redhat_marlin_report.md`, `quant/phase2b_report.md` in the originating workspace). Where a number is someone else's published measurement, it is labeled as such.

---

## What this is

| | |
|---|---|
| Base model | zai-org/GLM-5.3-Flash (~320B MoE, BF16 master) |
| Quantization | NVFP4 **weight-only (W4A16)**, group-16, E2M1 weights + FP8-E4M3 block scales + FP32 global scale |
| Producer | NVIDIA ModelOpt 0.45.0 (`NVFP4QTensor.quantize`), CPU-only shard streaming, **data-free** (no calibration set) |
| Scope | 37,152 routed-expert FFN tensors quantized; 1,618 tensors passthrough BF16 (attention, shared experts, routers, embeddings, lm_head, norms, MTP head, first dense layers) — partition replicates LibertAI's published card |
| Recipe delta vs stock | per-block MSE scale search, multiplier grid restricted to ≥ 1.0 (`mse_ge1`) — eliminates deep clipping (see `docs/RECIPE.md`) |
| Serving stack | vLLM image `radixark/vllm-glm53-flash:sm121-v8`, TP=2, marlin MoE backend, fp8_e4m3 KV cache |
| Required patch | `patches/modelopt_gscale_fixed.py` bind-mounted over the image's `modelopt.py` (see **Known issue** — without it this checkpoint emits U+FFFD corruption) |

## Headline results

### Weight-space error (measured, all 37,152 quantized tensors, round-trip vs BF16 master)

| checkpoint | mean rel² | mean rel_l2 | mean cos | notes |
|---|---|---|---|---|
| Murai-P1 (stock LibertAI replication) | 0.849% | 0.0921 | 0.99576 | exact recipe replica; matches LibertAI's published rel err |
| Murai-P2 (MSE scale search) | 0.634% | 0.0796 | 0.996829 | −25.4% rel² vs P1; 37,152/37,152 tensors improved |
| **Murai-P3 (mse_ge1, this repo)** | **0.655%** | 0.0809 | 0.996727 | keeps ~all of P2's gain with **zero deep clipping** (245 scale clamps, 0 zeroed blocks, all clamps in MTP layer 45) |

### Behavioral gates head-to-head (same probes, budgets, grader; temp 1.0, n=3 unless noted)

| probe | LibertAI NVFP4 | Murai-P2 | Murai-P3 | EXL3 TR3-4bpw |
|---|---|---|---|---|
| lc-01-32k (long-code) | 6/6 | 0/4 | 2/3 | 2/3 |
| lc-01-100k (long-code) | 6/6 | 0/4 | 2/3 | 1/3 |
| tc-10 (tool-call) | pass | 0/2 | 1/3 | 2/3 |
| ce-01 (code-exec) | starved | 2/2 | 1/2 † | 0/2 |
| dj-01 (diorama @64k) | 0/5 | **only working scene ever rendered** | 0/3 (1 unrenderable) | 0/3 (zero-content ×3) |

† **Grader revision (§30):** under the hardened grader (zero-content-at-cap check), Murai-P3's ce-01 cell revises from 1/2 to **0/2** — the historical pass was a zero-content artifact. The verdict is unchanged (EXL3 was also 0/2).

Reading: P3 recovers the P2 regressions on lc-01 (0/4 → 2/3 + 2/3) and partially tc-10 (0/2 → 1/3), supporting the pre-registered deep-clipping hypothesis. EXL3's published KLD advantage (0.0246 vs NVFP4 0.0605, malaiwah's panel — third-party numbers, and see caveats in `docs/RECIPE.md`) does **not** survive into gate behavior on this battery.

### Serving-time corruption fix (the reason the patch ships)

| measurement | unpatched vLLM | with gscale patch |
|---|---|---|
| U+FFFD events (Korean tool-call + code/JSON probe, 6 passes, temp 0) | **8 / 6 passes** | **0 / 6 passes** |
| up-proj rel_l2 (local numeric proof, layer-10 expert 0, gate/up scale ratio 1.23) | 0.2549 | 0.0929 (= correct dequant) |
| gates spot-check after fix (lc-01/tc-10/ce-01) | — | behavior-neutral to slightly better (within temp-1.0 noise) |

Root cause, numeric proof, and affected-lineage analysis: `docs/UPSTREAM_vllm_54150.md`.

---

## Quickstart (2× DGX Spark)

Requirements: both nodes reachable over the ConnectX-7 fabric, the image `radixark/vllm-glm53-flash:sm121-v8` present on both, and the checkpoint at the same path on both nodes.

1. **Download the weights** on both nodes (181.3 GiB each — transfer node1→node2 over the fabric rather than downloading twice):

   ```bash
   hf download murai-labs/GLM-5.3-Flash-NVFP4-Murai \
     --local-dir ~/models/glm-53-flash-nvfp4-murai-p3
   ```

2. **Stage the patch** on both nodes:

   ```bash
   mkdir -p ~/patches
   cp patches/modelopt_gscale_fixed.py ~/patches/
   ```

3. **Adapt the launcher**: copy `launch/launch_glm53_murai_p3_gscalefix.sh` to `~/` on both nodes and edit the marked variables (host IPs, interface names, model path) for your cluster. The shipped values are our cluster's and will not match yours.

4. **Launch worker first, then head:**

   ```bash
   # on node 2 (worker)
   ~/launch_glm53_murai_p3_gscalefix.sh 1
   # ~25 s later, on node 1 (head)
   ~/launch_glm53_murai_p3_gscalefix.sh 0
   ```

   Boot takes ~19–20 min. API on `http://<head-ip>:8000/v1`, served id `glm-5.3-flash-murai-p3`.

5. **Verify the patch is live** — the boot log must contain, per MoE layer:

   ```
   MURAI-EXPERIMENT gscale compensation: scale shape (288, 2048, 256), w3/w1 ratio mean ~1.0 max ~2.0, fp8 clamps ~20-30k
   ```

   (Observed on our boot: ratio mean 1.0197 / max 2.0000 / 28,558 clamps on one layer; clamps ≈ 0.03% of up-half scales.) The stock warning `w1_weight_scale_2 must match w3_weight_scale_2` still fires once — it is upstream of the compensation hunk; expected.

6. **Run the regression probe** (should print `CLEAN (0 events)`):

   ```bash
   python3 probes/probe_ufffd.py <head-ip> 8000 glm-5.3-flash-murai-p3
   ```

### The other bind-mounts

The launcher also bind-mounts three runtime fixes from the community (tonyd2wild's SM121 top-k >24K fix; vcruz305's K-pool tail OOB fix ×2 files). Those files are **not** in this repo — get them from their respective upstreams (see `launch/` comments). Serving this checkpoint long-context without them is not recommended.

## Hardware & serving requirements

- 2× NVIDIA DGX Spark (GB10, sm_121a, 128 GB unified memory each), NVLink-C2C / ConnectX-7 fabric, TP=2. The checkpoint is 181.3 GiB — it does not fit on one node.
- Image lineage: `radixark/vllm-glm53-flash:sm121-v8` (vLLM `0.1.dev20051+g487ecf187`, FlashInfer `0.6.18.dev20260819`). The engine runs `quantization=modelopt_fp4`.
- Load-bearing serving flags (all in the launcher): `--tensor-parallel-size 2`, `--gpu-memory-utilization 0.85`, `--max-model-len 262144`, `--block-size 2304`, `--moe-backend marlin`, `--kv-cache-dtype fp8_e4m3`, `--enforce-eager`, MTP speculative decode `num_speculative_tokens=4`, NCCL over RoCE (`NCCL_IB_GID_INDEX=3`).
- `--block-size 2304` + GMU 0.85 are load-bearing on this stack (community-reported; we adopted and kept them).
- KV pool with this config: 672,606 tokens (fp8_e4m3, measured from boot log).

## KNOWN ISSUE: vLLM ModelOpt single-gscale repack bug (affects this checkpoint)

**If you serve this checkpoint on unpatched vLLM ModelOpt NVFP4 marlin MoE path, you will get corrupted output** — U+FFFD replacement characters and tool-call desync, worst in multibyte scripts, invisible in ASCII. This is a vLLM bug, not a checkpoint defect; it hits every ModelOpt-packed NVFP4 checkpoint with separate gate/up projections where the two global scales differ (69.1% of our 12,384 expert gate/up pairs; mean ratio 1.145, max 10.0) — including the LibertAI checkpoint.

- Mechanism: `modelopt.py` takes `w13_weight_scale_2[:, 0]` (gate's global scale) as the single scale for the fused [gate;up] marlin GEMM; the up half is dequantized with the wrong scale.
- Fix: `patches/modelopt_gscale_fixed.py` — a 20-line hunk that rescales the up half's FP8 block scales by `ws2_up/ws2_gate` before repack. Bind-mount over `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py` in the image. No image rebuild.
- Upstream: tracked as [vllm-project/vllm#54150](https://github.com/vllm-project/vllm/issues/54150); our root-cause draft is in `docs/UPSTREAM_vllm_54150.md`.
- A compressed-tensors checkpoint (e.g. RedHatAI's, which shares gate/up scales by construction) does not trigger the bug — verified clean (0/6) on our stack by us.

## Limitations (read before adopting)

- **ce-01 (code-exec): 0/2** on P3 under the hardened grader — reasoning starvation at the token cap, zero content. P2 passed 2/2 here; treat ce-01-class workloads as at-risk.
- **dj-01 (long generative, 64k budget): 0/3** on P3. Murai-P2 remains the only checkpoint that ever produced a working scene on this probe; on P3 runs either starve or complete-but-unrenderable. If your workload is very long single-shot generation, P3 has not demonstrated it.
- **Sample sizes are small** (n=2–3 per probe, temp 1.0). Single-cell differences (e.g. lc-01-100k 2/3 vs 1/3) are within sampling noise; the table above is a decision aid, not a benchmark suite.
- **Gates ran on a partially unhardened stack historically.** All P3 gate numbers predate the SM121 top-k and K-pool runtime fixes; past FAILs cannot be retro-attributed (old logs deleted). The gscalefix spot-check (lc-01/tc-10/ce-01) was run on the hardened stack and was behavior-neutral.
- **W4A16 without activation scales:** this checkpoint carries no `input_scale`s. It is validated **only on the marlin MoE backend** (which does not read activation scales for W4A16). On the default flashinfer NVFP4 path, vLLM reads placeholder 1.0 input scales and the model degenerates (LibertAI documented the same for their checkpoint). Do not drop `--moe-backend marlin`.
- **Probe depth:** post-fix battery covered 46 passes / ~90k tokens (Korean/English 10-pass originals, 8 additional script families, two 8k-token long generations): 0 U+FFFD, 0 surrogates, 0 control chars, 0 repetition loops. Prior to the fix the same stack emitted 8 events / 6 passes.
- We have **no KLD measurement of our own**; all KLD numbers cited are third-party (malaiwah's panel).

## License & attribution

- **Weights:** derived from `zai-org/GLM-5.3-Flash`, which is released under the **MIT License** (verified from the base model's LICENSE file). See `NOTICE.md`. This repo intentionally ships no `LICENSE` file for the weights; the base model's MIT terms govern.
- **Quantization tooling:** NVIDIA Model Optimizer 0.45.0. Recipe design replicates LibertAI's published card (`LibertAI/GLM-5.3-Flash-NVFP4`) as the P1 baseline.
- **`patches/modelopt_gscale_fixed.py`** is a derivative of vLLM's `modelopt.py` (Apache-2.0, SPDX header retained) with one added hunk marked `MURAI-EXPERIMENT`.
- Runtime fixes referenced by the launcher are by tonyd2wild and vcruz305 (see `launch/` comments).

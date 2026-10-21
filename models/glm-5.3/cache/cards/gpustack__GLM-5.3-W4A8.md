---
base_model: zai-org/GLM-5.3
base_model_relation: quantized
language:
- en
- zh
library_name: vllm
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- glm
- glm_moe_dsa
- mixture-of-experts
- w4a8
- int4
- fp8
- quantized
- reasoning
- vllm
- sglang
- hopper
---

# GLM-5.3-W4A8

Hopper-native W4A8 quantization of [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3). Routed experts are INT4 (group size 128); activations and non-expert layers stay FP8. On 8×H20 the model runs on CUTLASS W4A8 grouped GEMM — no dequantization to BF16. **The same checkpoint loads in both vLLM and SGLang**; you do not need two copies.

Two things this checkpoint is for:

- **Half the weights, roughly twice the KV cache.** The checkpoint shrinks from 704 GB to 372 GB. On 8×H20 (~1128 GB), KV capacity goes from 608k tokens to 1.26M tokens (+107%).
- **No measurable accuracy drop.** GPQA-Diamond and teacher-forced NLL both sit inside the sampling noise of the official FP8 release. MTP speculative decoding works out of the box.

> [!NOTE]
> This model still has the full ~753B parameters of GLM-5.3. If the sidebar shows ~370B, that is a counting artifact: each int32 slot stores eight 4-bit weights, and the metadata counts storage slots, not parameters.

**Fully verified on 8×H20-3e in both engines**: vLLM nightly (`0.26.1rc1.dev229+g124154a88`) and SGLang dev (`0.0.0.dev1+gbb5e61986`). For SGLang use `lmsysorg/sglang:dev`; v0.5.16 was not verified. A native-load fix is in [sgl-project/sglang#37207](https://github.com/sgl-project/sglang/pull/37207).

Storage format is compressed-tensors: routed experts `pack-quantized` INT4, non-experts `float-quantized` FP8 128×128 block. This is **not** SGLang's native `w4afp8` packing. vLLM uses `CompressedTensorsW4A8Fp8MoEMethod`; SGLang uses `CompressedTensorsW4AFP8MoE`. Both sit on the SM90-only CUTLASS W4A8 kernel. vLLM loads as-is; SGLang needs an extra `PYTHONPATH` (see Deploy / SGLang).

Also published on ModelScope: [skyai/GLM-5.3-W4A8](https://www.modelscope.cn/models/skyai/GLM-5.3-W4A8).

## How this differs from other 4-bit releases

| | This repo (`gpustack/GLM-5.3-W4A8`) | Official FP8 | PhalaCloud/GLM-5.3-W4AFP8 |
|---|---|---|---|
| Size | **372.3 GiB** | 703.7 GiB | 372.3 GiB |
| Expert weights | INT4 group-128 (RTN + per-group MSE clip search) | FP8 block 128×128 | INT4 group-128 (AWQ calibration) |
| Quantized from | Official FP8 (dequant then requant) | Official release | BF16 parent |
| Non-expert layers | FP8 block (byte-identical) | FP8 block | FP8 block |
| Activations | FP8 per-token dynamic | FP8 dynamic | FP8 dynamic |
| Format | compressed-tensors (experts pack-quantized, non-experts float-quantized) | fp8 | w4afp8 (SGLang-native) |
| Engines | **vLLM / SGLang** | vLLM / SGLang | SGLang |

Requantizing from FP8 adds a second quantization step, but measured NLL is only +0.001 nats/token. In return you get a checkpoint vLLM can serve today, without waiting on an upstream `w4afp8` PR, and SGLang can use the same files.

## Accuracy

Measured with vLLM on 8×H20 (TP8, `--kv-cache-dtype fp8_ds_mla`, temperature 1.0, top_p 0.95, thinking on by default). Official FP8 and this checkpoint used **identical** prompts, option shuffles, and sampling. The PhalaCloud column is from that model's card (SGLang, different protocol — not a direct subtraction).

| Benchmark | GLM-5.3 (FP8, local) | GLM-5.3-W4A8 (local) | PhalaCloud W4AFP8 (reported) |
|---|---|---|---|
| GPQA-Diamond pass@1 (198 × 4 samples) | 89.65% | **90.28%** (+0.63pp) | 91.92% (182/198) |
| GPQA-Diamond majority@4 | 90.91% | 90.91% | — |
| Teacher-forced NLL (64×2048-token held-out text) | 0.8929 nats | 0.8940 nats (+0.001) | +0.282 nats vs BF16 |
| Perplexity | 2.442 | 2.445 | — |
| AA-LCR | — | — | 73.0 |
| BFCL (45-item live subset) | — | — | 82.2 |
| NIAH @ ~930k tokens | KV only 608k, does not fit | In principle measurable (KV 1.26M) | 3/3 |
| deepswe1.1 71/113 | — | — | 62.8 |   it provide by vllm bencswong2 ,many thanks 

Zero GPQA parse failures on both local runs. W4A8 pass@1 is slightly above FP8; the gap is about five samples inside 792 draws — sampling noise. majority@4 matches exactly. Window-to-window NLL SE is ~0.074; +0.001 is far below that.

PhalaCloud's GPQA is ~1.6pp higher, mostly from AWQ calibration, quantizing from BF16 (no second quant step), and a different engine / truncation-retry protocol.

Full tables: [EVAL_DETAILS.md](EVAL_DETAILS.md). Chinese card: [README.zh.md](README.zh.md).

## Performance (8×H20-3e, single node)

MoE backend: vLLM auto selects CUTLASS W4A8. Attention: `--kv-cache-dtype fp8_ds_mla` → FLASHMLA_SPARSE. W4A8 **requires** `--enable-expert-parallel`; FP8 is faster without EP.

Offline batching (no rate limit, `max_model_len=16384`). Prefill rows are input throughput; decode rows are output tokens / wall time including the first prefill.

| Setup | Weight VRAM / GPU | KV capacity | prefill 8 conc. × 8K in | decode 128 conc. × 128 in → 256 out | decode 256 conc. × 128 in → 256 out |
|---|---|---|---|---|---|
| FP8 TP8 | ~88 GiB | 608,128 | **3675** tok/s | 1686 tok/s | 2094 tok/s |
| W4A8 TP8 + EP | 52.6 GiB | **1,257,280** | 3441 tok/s | 1617 tok/s | 1779 tok/s |
| **W4A8 2×(TP4+EP)** | 93 GiB | 638,336 | **5993** tok/s | **2256** tok/s | **3249** tok/s |

A single TP8 replica is not faster than FP8 (prefill 0.94×, decode 0.85–0.96×): H20 has spare bandwidth, and CUTLASS dequant eats the bandwidth win. The real gains are:

1. **2.07× KV cache** — longer context and higher concurrency on the same 8 GPUs.
2. **TP4 dual replica becomes possible.** 372 GiB / 4 = 93 GiB per GPU, which fits in 141 GiB; FP8's 704 GiB / 4 = 176 GiB does not. Dual-replica aggregate vs FP8 TP8: prefill **1.63×**, decode 128 conc. **1.34×**, 256 conc. **1.55×**.

## MTP / speculative decoding

vLLM loads the layer-78 draft head via `glm_moe_dsa → deepseek_mtp → DeepSeekMTPModel`. Test: 512-token input / 256-token output, `num_speculative_tokens=1`.

| Checkpoint | conc. 1 | conc. 4 | conc. 16 | conc. 64 |
|---|---|---|---|---|
| FP8 no MTP → MTP | 86 → 149 tok/s (**1.72×**) | 265 → 350 (1.32×) | 551 → 695 (1.26×) | 1105 → 1306 (1.18×) |
| W4A8 no MTP → MTP | 66 → 104 tok/s (**1.56×**) | 196 → 282 (1.44×) | 480 → 572 (1.19×) | 963 → 1190 (1.24×) |

Quantization does not break MTP. PhalaCloud reports EAGLE steps=3 acceptance length ~2.93, which is not comparable to `num_speculative_tokens=1`.

## Deploy

**Hardware: Hopper GPUs only (H20 / H100 / H200, compute capability exactly SM90).** Ada (SM89) and Blackwell (SM100 / SM120) are not supported — see Limitations. Pick one engine:

- **vLLM nightly** (0.26.1rc1 or newer recommended; the v0.26.0 release may not have full GLM-5.3 DSA + SM90 W4A8). The vLLM examples below work as written.
- **SGLang dev** (`lmsysorg/sglang:dev`, verified at `0.0.0.dev1+gbb5e61986`). Needs `PYTHONPATH`; see the end of this section. Release v0.5.16 was not verified on this repo.

All performance numbers below are from vLLM.

### Throughput: two replicas × TP4 (recommended)

```bash
M=gpustack/GLM-5.3-W4A8
for i in 0 1; do
  [ $i -eq 0 ] && D='"device=0,1,2,3"' PORT=8000 || D='"device=4,5,6,7"' PORT=8001
  eval docker run -d --name glm53-r$i --gpus "$D" --ipc=host --shm-size=32g \
    -p $PORT:8000 \
    vllm/vllm-openai:nightly \
    --model $M --served-model-name GLM-5.3 \
    --tool-call-parser=glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser=glm45 \
    --tensor-parallel-size 4 --enable-expert-parallel \
    --kv-cache-dtype fp8_ds_mla \
    --gpu-memory-utilization 0.92 --max-model-len 65536 --trust-remote-code
done
```

### Long context: single instance TP8

```bash
docker run -d --name glm53 --gpus all --ipc=host --shm-size=32g \
  -p 8000:8000 \
  vllm/vllm-openai:nightly \
  --model gpustack/GLM-5.3-W4A8 \
  --tool-call-parser=glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser=glm45 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8_ds_mla \
  --gpu-memory-utilization 0.90 --max-model-len 262144 --trust-remote-code
```

To enable MTP speculative decoding, add:

```bash
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

The server exposes a standard OpenAI-compatible API. Sampling defaults live in `generation_config.json`: temperature 1.0, top_p 0.95. Thinking mode is on by default via the chat template (`<think>`).

**Do not** set `--moe-backend marlin` or `triton`: on SM90, W4A8 has only the CUTLASS path, and auto will pick it. **Do** pass `--enable-expert-parallel`; otherwise each GPU holds 256 experts × intermediate 256, and CUTLASS grouped GEMM slows down a lot.

### SGLang

The same weights run on SGLang if you put the repo directory on `PYTHONPATH`. **Use `lmsysorg/sglang:dev`**; release v0.5.16 was not verified here:

```bash
python -m sglang.launch_server \
    --model-path gpustack/GLM-5.3-W4A8 \
    --tp-size=8 \
    --trust-remote-code \
    --mem-fraction-static=0.85 \
    --reasoning-parser=glm45 \
    --tool-call-parser=glm47 \
    --context-length=300000 \
    --kv-cache-dtype=fp8_e4m3 \
    --cuda-graph-max-bs=32 \
    --speculative-algorithm=EAGLE \
    --speculative-num-steps=5 \
    --speculative-eagle-topk=1 \
    --speculative-num-draft-tokens=6 \
    --enable-nsa-prefill-context-parallel \
    --nsa-prefill-cp-mode=round-robin-split \
    --attn-cp-size=8
```

`PYTHONPATH` makes every worker load this repo's `sitecustomize.py`. On checkpoint load it renames non-expert FP8 scales from `weight_scale` to `weight_scale_inv`: SGLang's `Fp8LinearMethod` and DSA indexer fusion hard-code the latter, vLLM wants the former. Both names **cannot live on disk at once** — either engine `KeyError`s on a tensor with no matching parameter — so the rename has to happen at read time. The weight bytes themselves are unchanged.

you can add follow to set default thinking type to low

```bash
--default-chat-template-kwargs='{"reasoning_effort":"low"}' 
```

vLLM does not need and will not load that file. On `sglang:dev`, TP8 used 48.2 GiB/GPU with `max_total_num_tokens=803264`.

To drop the shim, SGLang upstream needs three hard-coded lookups fixed (tracked in [sgl-project/sglang#37207](https://github.com/sgl-project/sglang/pull/37207)):

| Location | Issue |
|---|---|
| `compressed_tensors_w4a8_fp8_moe.py` | MoE reads `num_bits` from `target_scheme_map["Linear"]` and requires INT4 |
| `compressed_tensors.py` `weight_block_size` | Reads `block_structure` from the **same** key and requires FP8 block |
| `deepseek_weight_loader.py` | DSA indexer fusion hard-codes `.weight_scale_inv` |

The first two contradict each other, and the compressed-tensors schema forbids `block_structure` together with group strategy, so no legal config satisfies both.

### Sampling defaults

Same as the base model: temperature 1.0, top_p 0.95. Without top-p truncation, thinking mode can occasionally loop.

## Quantization

- **Routed experts** (~734B parameters, 96% of the model): dequantize FP8 blocks to float32, then symmetric INT4, group size 128. Each group searches 11 shrink ratios (1.00 → 0.70) for the MSE-best scale, then packs as compressed-tensors `pack-quantized` uint4b8 / int32. Measured relative L2 error ~0.103, better than naive RTN (0.124) and MXFP4 group-32 (0.113).
- **Attention / shared experts / dense MLP / DSA indexer**: official FP8 128×128 block, **byte-identical**. Only the scale tensor name changes from official `weight_scale_inv` to compressed-tensors `weight_scale`. SGLang still expects the former, so `sitecustomize.py` renames it back at read time.
- **Embedding, lm_head, router, norms**: stay BF16.
- Activations are FP8 per-token dynamic at runtime, same as the official FP8 path.

Hopper has no FP4 tensor cores, so MXFP4 would be Marlin W4A16 (half the peak FLOPS). This checkpoint uses INT4-W4A8 instead: 4-bit storage, dequant to FP8 inside the kernel, then wgmma at 296 TFLOPS peak.

## Limitations

- **Hopper (SM90) only** — H100 / H200 / H20. This is not a conservative estimate: vLLM picks the W4A8 CUTLASS path with `_check_scheme_supported(90, match_exact=True)`, i.e. capability **exactly 9.0**, not “9.0 or above”. Ada (SM89, L40S / L4 / 4090), **Blackwell (SM100, B200 / GB200)**, and RTX 5090 / Pro 6000 (SM120) are **unsupported** — even the newer ones. On Blackwell, use NVFP4 (native FP4 tensor cores).
- On non-SM90 cards you **will not get a clean error**. vLLM's `_is_dynamic_token_w4a8_int` only compares `num_bits` and ignores activation type, so this config can mis-hit `CompressedTensorsW4A8Int8MoEMethod`, which expects unpacked `torch.int8` weights; this repo is `pack-quantized` int32 and fails at load with a shape/dtype error. SGLang is looser: its MoE path does not check capability and will call an SM90-only CUTLASS kernel at runtime.
- Format is compressed-tensors, not SGLang `--quantization w4afp8`. vLLM loads directly; SGLang loads with `PYTHONPATH` pointing at this repo's `sitecustomize.py`, and you must use `lmsysorg/sglang:dev` (release v0.5.16 not verified). See Deploy / SGLang.
- Accuracy and speed numbers are from vLLM. SGLang was only smoke-tested for load + generation; the kernels differ, so do not copy the numbers across engines.
- Requantized from official FP8, so in theory slightly worse than BF16 + AWQ. Local GPQA / NLL put the gap inside noise.
- TP split: `moe_intermediate_size` after TP must be divisible by 256. TP8: 2048/8=256, OK. TP16: 128, not OK.
- Inherits the capabilities and limits of base GLM-5.3.

## License

Same as [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (see `LICENSE`). Please cite the original GLM-5.3 work.

---

# Base model card (official)

> Excerpted from the official base-model repo. Full text: [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3).

GLM-5.3 uses the same base as GLM-5.2 — every gain comes from post-training. Versus GLM-5.2 it is much stronger at complex coding and long-horizon tasks:

- **Stronger coding**: strongest open-weights coding model; +50% vs GLM-5.2 on the in-house Z.ai Code Bench; open-source SOTA on Terminal Bench 3.0 and Agents' Last Exam.
- **Emergent security-research capability**: post-training scaled those skills faster than expected.

Official reference numbers (model card): HLE 62.5 (with tools), Deep-SWE 66.9, Terminal-Bench 2.1 88.2, Terminal-Bench 3.0 28.3.

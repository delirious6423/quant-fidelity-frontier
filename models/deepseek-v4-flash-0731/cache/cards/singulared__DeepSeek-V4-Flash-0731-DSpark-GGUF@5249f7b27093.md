---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
library_name: llama.cpp
tags:
  - gguf
  - deepseek-v4
  - dspark
  - speculative-decoding
  - draft-model
---

# DeepSeek-V4-Flash-0731 DSpark drafters (GGUF, for mainline llama.cpp)

A DSpark speculative-decoding drafter extracted from the official
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) weights,
in the `dflash` architecture that mainline llama.cpp uses.

## Which file to use

| file | size | routed experts | decode @12K [*] |
| --- | ---: | --- | ---: |
| `DeepSeek-V4-Flash-0731-DSpark-Q2K-Q4K.gguf` | **7.83 GiB** | Q2_K gate/up, **Q4_K down** | **16.5 t/s** |
| `DeepSeek-V4-Flash-0731-DSpark-BF16.gguf` | 10.15 GiB | MXFP4 (converter default) | 15.2 t/s |

**Use the Q2K-Q4K file.** It is 2.3 GiB smaller *and* ~8% faster. Decode on this hardware is
bandwidth-bound, so a smaller drafter streams fewer bytes per draft step; the accuracy lost by
dropping the experts to 2-3 bpw costs less than the bandwidth saved. `ffn_down` is kept at Q4_K
because it is the error-sensitive projection - a Q3_K variant measured 15.9 t/s, and an
IQ2_XXS/Q4_K variant 15.8 t/s.

[*] 11,929-token prompt, `--spec-draft-n-max 2`, same target and method as the table below.
Confirmed over a 768-token generation: 16.0 t/s vs 14.9 t/s at `n-max 3`.

**It pairs with any 0731 target GGUF.** The drafter is a separate model, so a target quantized
before MTP support existed - which is most of them, including files with no `mtp.*` tensors at
all - still gets speculation with no rebuild.

## Target model used for testing

All measurements below use [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf),
specifically `DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf`
(90.9 GiB). It is a good reference target for two reasons:

* Its importance matrix was computed on the 0731 weights themselves - 202,100 chunks of
  chat-template-rendered prompts, covering the routed expert tensors.
* It **embeds a chat template**. The 0731 release ships no Jinja template at all (only the
  `encoding_dsv4` Python scripts), so quants that rely on a converter's bundled template may
  behave differently for tool calls and reasoning.

That repo also has a plain `IQ2XXS-...-0731` variant (80.8 GiB) and a larger
`Q4KExperts-F16HC-F16Compressor-F16Indexer-...-0731` (153 GiB) if you have the memory.

Note this target contains **no `mtp.*` tensors** - it was built before MTP export existed. That
is precisely the case this drafter is for: speculation without rebuilding the target.

## Requirements

Mainline llama.cpp with commit `596a579` ("DeepseekV4 MTP + DSpark", #25784, merged 2026-08-02)
or newer. Nothing else is needed to *use* this file; the extra PR below was only needed to
*build* it.

## Usage

```bash
llama-server \
  -m DeepSeek-V4-Flash-...-fixed-0731.gguf \   # e.g. from antirez/deepseek-v4-gguf
  -md DeepSeek-V4-Flash-0731-DSpark-Q2K-Q4K.gguf \
  --spec-type draft-dspark --spec-draft-n-max 2 \
  -ngl 99 -ngld 99 -fa on
```

## Draft depth

`--spec-draft-n-max 2` is the optimum here; the llama.cpp default of 3 is worse. Measured on the
Q2_K/Q3_K drafter, 11,929-token prompt:

| `n-max` | 2 | 3 | 4 | 6 |
| --- | ---: | ---: | ---: | ---: |
| decode t/s | **15.9** | 14.1 | 13.8 | 13.4 |

Deeper drafting loses: DeepSeek-V4's MTP head does not stay accurate past ~2 tokens, so the extra
drafts are rejected and their verification cost is wasted.

## How the Q2K-Q4K file was built

Mainline's converter repacks DeepSeek-V4's routed experts to **MXFP4** and `llama-quantize`
refuses MXFP4 as input, so there is no supported path to a smaller drafter. Producing this file
needed a converter patch adding `--dspark-expert-type`, which dequantises the packed E2M1 experts
to f32 (or Q8_0) instead of repacking them:

```bash
python convert_hf_to_gguf.py DeepSeek-V4-Flash-0731 --dspark \
  --target-model-dir DeepSeek-V4-Flash-0731 \
  --dspark-expert-type q8_0 --outfile dspark-q8-intermediate.gguf   # 19.7 GiB

llama-quantize --allow-requantize \
  --tensor-type "ffn_gate_exps=q2_k" --tensor-type "ffn_up_exps=q2_k" \
  --tensor-type "ffn_down_exps=q4_k" \
  dspark-q8-intermediate.gguf DeepSeek-V4-Flash-0731-DSpark-Q2K-Q4K.gguf Q8_0
```

The patch is **not upstream**; you do not need it to *use* this file, only to rebuild it.

## Measured results

AMD Radeon 8060S (gfx1151, Strix Halo), 128 GB unified memory, ROCm 7.2.4, HIP backend,
mainline llama.cpp `0b14b87`. Target as described above. Temperature 0, f16 KV cache.
Swap disabled, one discarded warmup request per server, distinct filler text per depth so no
prompt-cache prefix is shared between measurements.

### Speedup vs context depth (`--spec-draft-n-max 2`)

Decode, tokens/s. Every cell measured on the same machine and target, swap disabled, one
discarded warmup request per server, and distinct filler text per depth so no prompt-cache
prefix is shared between measurements. `-c 49152` for the first three rows, `-c 131072` for the
last two.

| prompt tokens | mainline | mainline + drafter | gain | hipCUB [1] | hipCUB + drafter |
| --- | --- | --- | --- | --- | --- |
| cached (4) | 13.61 | 19.47 | +43% | 13.50 | 19.31 |
| 11,929 | 9.84 | 14.18 | +44% | 12.14 | 16.30 |
| 24,530 | 9.38 | 13.65 | +45% | 11.34 | 15.04 |
| 39,628 | 9.48 | 13.35 | +41% | 10.67 | 13.12 |
| 84,829 | 7.45 | 10.95 | **+47%** | 9.05 | 10.95 |

**The drafter gives +41-47% at every depth tested, and the gain does not fade as context grows**
- at 85K it is the largest measured. Draft acceptance was 0.67-0.76 across the range.

Note the cached row: 13.61 vs 13.50 without the drafter, 19.47 vs 19.31 with it. With a prompt
that short the hipCUB code path is never reached, so those columns are a control - they should
be identical, and they are.

### About the hipCUB columns

They are a separate, unofficial AMD patch [1], included because the interaction turns out to be
depth-dependent:

* **Without the drafter it helps everywhere**: +23% at 12K, +21% at 24K, +13% at 40K, +22% at 85K.
* **With the drafter it helps only up to ~32K**: +15% at 12K and +10% at 24K, but -2% at 40K and
  0% at 85K (10.95 either way).

The likely reason is that speculation cuts the number of target forward passes, so the indexer's
CPU fallback stops being the limiting factor at depth and memory bandwidth over the 90 GiB of
weights dominates instead. If you run this drafter, the patch is worth having for mid-length
contexts and is neutral beyond that; if you run the target without a drafter, it helps throughout.

---

[1] `argsort`/`top_k` above 1024 columns have no GPU path on HIP, so they fall back to the CPU
backend - for DeepSeek-Sparse-Attention models that puts V4's per-token lightning-indexer `top_k`
on the CPU, and the cost grows with context depth (visible as ~416% CPU on mainline at depth
versus ~100% with the patch, the latter being the backend's normal busy-wait). Enabling hipCUB
(rocPRIM's CUB-compatible API) restores the GPU path. Branch:
<https://github.com/singulared/llama.cpp/tree/hipcub-port> - not upstream, feature-detected so
builds without `hipcub-dev` are unchanged, and `-DGGML_HIP_NO_HIPCUB` opts out.

> **Note on `-ffast-math` (2026-08-09).** Enabling hipCUB is safe on mainline, which compiles
> ggml-hip with `-funsafe-math-optimizations` and deliberately *not* `-ffast-math` (see the comment
> in `ggml/src/ggml-hip/CMakeLists.txt`). If you build a fork that adds `-ffast-math` — ROCmFPX
> does — hipCUB's radix sort is miscompiled on gfx1151: `DeviceRadixSort`/`DeviceSegmentedSort`
> return out-of-range values for most `ncols > 1024`, which corrupts the indexer's token selection
> or faults in `GET_ROWS`. Reproduced standalone with no ggml involved: identical hipCUB call,
> `-O2` correct, `-O2 -ffast-math` gives 3072 bad indices at `ncols=1536`; `-O3` alone is fine.
> In that case compile `argsort.cu` and `top-k.cu` with `-fno-fast-math`. Verified on this branch
> rebased onto master `69bf643`: all sizes correct.

### Choosing `--spec-draft-n-max` (short prompt, `-c 32768`)

| n-max | decode | draft acceptance |
| --- | --- | --- |
| none (baseline) | 14.03 t/s | - |
| 1 | 17.73 t/s | 0.801 |
| **2** | **20.43 t/s** | 0.759 |
| 3 | 16.04 t/s | 0.591 |
| 4 | 16.10 t/s | 0.540 |
| 6 | 13.75 t/s | 0.439 |

**n-max 2 is the knee.** Acceptance falls monotonically as drafting goes deeper, and past 2 the
extra verification costs more than the accepted tokens save; by n-max 6 the gain is gone.

KV cache: f16 measured fastest here (20.61 vs 17.06 q8_0 vs 17.76 q4_0 t/s on one prompt), and
V4's MLA already keeps the cache small, so there is little to save by quantizing it. Those three
figures are from a single prompt at one context length - treat them as provisional.

Memory, measured at `-c 131072` on a 128 GB machine: the target alone occupies ~92 GiB of GTT
(~21 GiB free), and target + drafter ~104 GiB (~11 GiB free). It fits, but with little room -
run with swap disabled if you are benchmarking, otherwise memory pressure degrades throughput
silently rather than failing.

## Contents

81 tensors, 3 blocks, 10.15 GiB. The converter fixes the layout - MXFP4 experts, Q8_0 attention
and shared experts, BF16/F32 router, norms, sinks and hyper-connections - so it is not
re-quantizable (`llama-quantize` rejects mxfp4 input). Precision outside the experts is
deliberately high: drafter precision drives acceptance.

## How it was built

```bash
python convert_hf_to_gguf.py <fp8-checkpoint-dir> \
  --dspark --target-model-dir <same-dir> --outtype bf16 \
  --outfile DeepSeek-V4-Flash-0731-DSpark-BF16.gguf
```

`--target-model-dir` is mandatory: the drafter needs the target's tokenizer.

`--dspark` was added by PR [#26452](https://github.com/ggml-org/llama.cpp/pull/26452), merged as
`3581ba0`, so current master has it. It is needed because the 0731 checkpoint ships **three**
`mtp.*` blocks (matching `dspark_target_layer_ids: [40, 41, 42]`) while `config.json` declares
`num_nextn_predict_layers: 1` - so plain `--mtp` fails with
`ValueError: Unexpected DeepSeek-V4 MTP layer 1`.

## Credits

DeepSeek for the weights (MIT), `am17an` for the llama.cpp DSpark implementation and the
`--dspark` export path, [`antirez`](https://huggingface.co/antirez/deepseek-v4-gguf) for the target quant and its
imatrix, used for every measurement here.

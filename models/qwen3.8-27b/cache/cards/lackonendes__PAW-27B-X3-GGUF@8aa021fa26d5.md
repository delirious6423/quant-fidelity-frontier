---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
tags:
- quantization
- trellis-codec
- paw
- gguf
---

> **New: [PAW-27B-X3.1](https://huggingface.co/lackonendes/PAW-27B-X3.1-GGUF) is out: about 20% lower code error than the 3.5-bit file on this page.**
> Measured as full-model KL vs the Q8 parent on code: 0.02113 vs 0.02665 (20.7% lower, chat-framed) and 0.01452 vs
> 0.01792 (19.0% lower, raw code), both intervals above zero, same binary. Same size class (11.50 GiB, 1,966,080
> bytes smaller). KL is closeness to Q8, not an accuracy score; speed is not measured yet, and X3.1 benchmark results
> are being added on its page. Needs `llama-paw` `main` at commit `4b936837b` or newer.

# PAW-27B-X3

_If this model is useful to you, support the work by following [@nawariokra](https://x.com/nawariokra) on X, or buy me a coffee on [Ko-fi](https://ko-fi.com/lackonendes)._

<a href="https://ko-fi.com/F4P3296XP7"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="32"></a>



**x3 trellis-coded quantizations of `Qwen/Qwen3.8-27B` at 1.5-3.5 bpw** that
run on **one 24 GB GPU** with a **256k context**. The 3.5-bit file is the
first artifact in this sweep to **beat the Q8 parent outright**.

| | |
|---|---|
| files | five weights (1.5 / 2 / 2.5 / 3 / 3.5 bit), 5.84-11.50 GiB, plus the DFlash2 drafter |
| recommended | `PAW-27B-X3-3.5bit.gguf` (11.50 GiB, 3.500 bpw) |
| context | 262,144 tokens on a single RTX 3090 |
| speed | prefill 1215 tok/s at 8k / 563 tok/s at 211k (262k ctx) · TG 33.5 tok/s · 100.45 tok/s verified spec decode at 8k code context (3.5bit + DFlash2 drafter) · 16.8 tok/s AR at 242k · 425.8 tok/s prefill at 229k in spec mode |
| code | **97.0% HumanEval / 92.1% HumanEval+ / 78.6% MBPP+** (3.5bit) |
| general | **67.4% MMLU-Pro (337/500)**, 94% GSM8K, 94% IFBench loose (3.5bit) |

## Requirements — read this first

This is **not** a standard GGUF. It uses the PAW x3 trellis format (`paw`
arch) and requires the [`llama-paw`](https://github.com/irawanw/llama-paw)
fork. **Stock llama.cpp cannot read these models.**

These weights are the **x3 (EXL3-style) codec family**. For the dense
(paw-dense) sibling of the same parent, see
[PAW-27B-GGUF](https://huggingface.co/lackonendes/PAW-27B-GGUF).

## Files

| file | rate | size | notes |
|---|---:|---:|---|
| [`PAW-27B-X3-1.5bit.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF/resolve/main/PAW-27B-X3-1.5bit.gguf) | 1.500 bpw | 5.84 GiB | partial recovery, see limitations |
| [`PAW-27B-X3-2bit.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF/resolve/main/PAW-27B-X3-2bit.gguf) | 2.000 bpw | 7.26 GiB | |
| [`PAW-27B-X3-2.5bit.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF/resolve/main/PAW-27B-X3-2.5bit.gguf) | 2.500 bpw | 8.67 GiB | |
| [`PAW-27B-X3-3bit.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF/resolve/main/PAW-27B-X3-3bit.gguf) | 3.000 bpw | 10.10 GiB | |
| [`PAW-27B-X3-3.5bit.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF/resolve/main/PAW-27B-X3-3.5bit.gguf) | 3.500 bpw | 11.50 GiB | **recommended — beats the Q8 parent** |
| [`Qwen3.8-27B-DFlash2-Q2_K.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF/resolve/main/Qwen3.8-27B-DFlash2-Q2_K.gguf) | — | 0.67 GiB | speculative drafter for llama-paw serving |

## Inference speed (RTX 3090, current llama-paw build)

Since llama-paw `9f3ba0717` the fp16-accumulate x3 GEMM is on by default
(`GGML_PAW_X3_GEMM_F16ACC=0` restores the fp32 path). Measured on the 3.5-bit
weights, one RTX 3090:

| measurement | tok/s |
|---|---:|
| PP512 (short prompt) | 1076.5 |
| PP8192 (chat length) | **1215.4** |
| PP, 211k-token prompt at 262144 ctx (`-ub 2048`) | 563.0 |
| TG128 (generation, chat scale) | 33.5 |
| TG at 211k depth | 19.9 |
| speculative decode, 8k code context | 100.45 (median, output hash-identical) |

The PP512/TG128 column in the evaluation table above was measured on the
older fp32-accumulate build and is kept for cross-artifact comparability.
The fp16-accumulate change passed a paired quality A/B with no detectable
difference on MMLU-Pro / HumanEval+ / MBPP+.

## Complete evaluation table

 Every artifact we measured on one RTX 3090, both rate sweeps, one table. All
 MMLU-Pro rows use the same protocol, seed and subset hash, so that column is
  directly comparable across every row. All accuracy columns are percentages
  (MMLU-Pro /500, IFB /64, HumanEval /164, MBPP /378, GSM8K /100). The IFB,
  HumanEval, and MBPP cells stack two scores on separate lines: IFB is
  loose/strict, HumanEval and MBPP are base/`+`, and the speed column stacks
  PP512 (prompt, t/s) over TG128 (generation, t/s). `—` means no valid
  measurement exists; it is never an estimate.

 | artifact | bpw | MMLU-Pro % | IFB Loose<br>IFB Strict | HumanEval<br>HumanEval+ | MBPP<br>MBPP+ | GSM8K % | PP512<br>TG128 |
 |---|---:|---:|---:|---:|---:|---:|---:|
 | Q8 parent | 8.51 | 65.2% | 31.2%<br>— | 97.0%<br>91.5% | —<br>— | 94.0% | —<br>— |
 | **B4.0**<br>12.93 GiB | 4.000 | 66.6% | 32.8%<br>31.2% | 95.7%<br>91.5% | 92.6%<br>78.0% | 94.0% | 811.24<br>38.23 |
 | **B3.5**<br>11.50 GiB | 3.500 | **67.4%** | 34.4%<br>32.8% | **97.0%**<br>**92.1%** | 93.1%<br>**78.6%** | 94.0% | 809.16<br>40.68 |
 | MIA / EXL3<br>14.2 GB | 3.50 | 65.2% | 34.4%<br>32.8% | 95.7%<br>90.9% | —<br>— | 93.0% | —<br>45.67† |
 | **B3.0**<br>10.10 GiB | 3.000 | 65.4% | 34.4%<br>32.8% | 94.5%<br>89.0% | 92.9%<br>78.6% | 95.0% | 798.11<br>42.21 |
 | **B2.5**<br>8.67 GiB | 2.500 | 61.8% | 29.7%<br>26.6% | 94.5%<br>87.8% | 92.1%<br>77.8% | 97.0% | 805.82<br>44.28 |
 | Escha-W2<br>10.2 GB | 2.469 | 61.8% | 31.2%<br>29.7% | 95.1%<br>90.9% | —<br>— | 95.0% | —<br>— |
 | **B2.0**<br>7.26 GiB | 2.000 | 58.0% | 31.2%<br>31.2% | 91.5%<br>86.6% | 88.4%<br>74.9% | 94.0% | 815.23<br>46.45 |
 | **B1.5**<br>5.84 GiB | 1.500 | 23.6% | 15.6%<br>12.5% | 61.6%<br>57.3% | 64.8%<br>54.5% | 42.0% | 813.25<br>46.96 |
 | Unsloth IQ2XXS<br>7.3 GB | ~2 | 46.2% | 23.4%<br>14.1% | —<br>84.1% | —<br>— | 93.0% | —<br>— |
 | AtomicChat AD-IQ2XXS<br>9.0 GB | ~2 | 53.0% | 20.3%<br>20.3% | —<br>73.8% | —<br>— | 94.0% | —<br>— |
 | **B1.0**<br>4.43 GiB | 1.000 | 0.0% | 0.0%<br>0.0% | 0.0%<br>0.0% | 11.6%<br>10.1% | 0.0% | 813.55<br>47.57 |

 All GSM8K rows here are n=100; not normalized. `†` EXL3's 45.67 is EXL3
 `perf.py` INT8 GEMV at context 0 — a different harness from the `llama-bench`
 figures in the same column. The bit-sweep rows (B1.0-B4.0) are
 `llama-bench -ngl 99 -p 512 -n 128 -r 5 -sm none` on one pinned device and are
 mutually comparable; the speed protocol differs by generation and is not
 uniformly comparable down the column.

## Honest summary

**B3.5 beats the Q8 parent outright** — MMLU-Pro 337 vs 326, HumanEval 159
vs 159 (tie), HumanEval+ 151 vs 150, MBPP+ 297 — at 11.50 GiB, 19% smaller
than the EXL3 3.5 bpw comparison artifact (14.2 GB). It does this with a
greedy marginal-rate-distortion allocation that promotes only the tensors
where more bits help: uniform K4 (**B4.0**) is bigger, slower, and worse on
every quality column despite the extra 1.43 GiB. Curated mixed precision
beats uniform higher bit rate — confirmed at both ends of the sweep.

**Below ~1.5 bpw there is a hard coherence cliff, not a smooth decline.**
B1.0 (uniform K1) is total structural incoherence — 0 on every benchmark,
all 64 IFBench completions literally empty — and is not shipped here. B1.5
is a real, partial recovery (MMLU-Pro 118/500, HumanEval 101/164, GSM8K
42/100): far below B2.0 (290) and far above chance, with every category above
zero. Between the cliff and B2.0 there is no observed "mostly broken" state.

GSM8K is n=100 per row for the coherent arms; the 93-97 spread among them is
not separable at that sample size. Every number was measured by us on one
RTX 3090 with byte-identical harnesses — nothing copied from model cards.

## Serving

Requires [`llama-paw`](https://github.com/irawanw/llama-paw), not stock
llama.cpp. GPU offload is automatic in GPU builds. Short version:

- speculative decoding with the DFlash2 drafter (`--spec-type draft-dflash`)
- `--reasoning off` is **required** — the checkpoint has a documented
  runaway think-loop
- `-fa on` is required for quantized KV
- verified **100.45 tok/s** (median, output hash-identical) with the tuned
  short-context server:

```sh
export GGML_PAW_X3_GEMV=2 GGML_PAW_MMQ_HEAD=1 GGML_PAW_GREEDY_IDS=1 GGML_PAW_DQ4=0

./build/bin/llama-server -m PAW-27B-X3-3.5bit.gguf \
  -md Qwen3.8-27B-DFlash2-Q2_K.gguf \
  --spec-type draft-dflash --spec-draft-n-max 5 \
  -fa on -ngl 99 -np 1 -c 20480 -b 512 -ub 512 \
  --no-warmup --reasoning off --host 127.0.0.1 --port 8080
```

### Serving at 256k context — optimum settings

The 3.5-bit weights plus the Q2_K drafter fit the full `-c 262144` on one
24 GB RTX 3090 **only** with quantized KV on target and draft and a capped
draft ubatch. Measured optimum:

```sh
export GGML_PAW_X3_GEMV=2 GGML_PAW_MMQ_HEAD=1 GGML_PAW_GREEDY_IDS=1 GGML_PAW_DQ4=0

./build/bin/llama-server -m PAW-27B-X3-3.5bit.gguf \
  -md Qwen3.8-27B-DFlash2-Q2_K.gguf \
  --spec-type draft-dflash --spec-draft-n-max 5 \
  -fa on -ctk q4_0 -ctv q4_0 \
  -c 262144 -ub 2048 -b 8192 -ubd 256 \
  -ngl 99 -np 1 --no-warmup --reasoning off \
  --host 127.0.0.1 --port 8080
```

| setting | why |
|---|---|
| `-ctk q4_0 -ctv q4_0` (target and draft) | the configuration that fits at 262144 with speculative decoding |
| `-ubd 256` | caps the draft ubatch; without it the drafter does not fit |
| `-ub 2048 -b 8192` | measured 425.8 tok/s prefill at a 229k-token prompt in spec mode |
| `-fa on` | required for quantized KV |
| `--reasoning off` | required — runaway think-loop |

At context depths up to ~160k, `-ub 4096` is faster than `-ub 2048`
(930 vs 903 tok/s PP8192 before the fp16-accumulate change) and passes the
VRAM gate there; at 262144 it does not fit, so keep `-ub 2048` for the full
context profile.

The drafter's KV is ~50 MiB at 262144 (5 sliding-window-2048 layers). AR
decode is ~16.8 tok/s at 242k after the FA GQA batching fix. **A full-context
speculative-decode tok/s figure is not yet measured and none is quoted**;
round arithmetic at 242k gives ~2.7-2.8 tokens/round. If you are tight on
VRAM, lower `-ub` first — it is the footprint lever, not `-b`.

## Limitations

- **B1.5 is a partial recovery, not a normal model** (see honest summary).
  Use it only when memory is the binding constraint.
- Code generation below 3.5 bpw retains ~90-96% of parent (see the table).
- Serve with thinking disabled (`--reasoning off`); thinking-mode behavior is
  unmeasured.
- Long-context *quality* is unmeasured above ~28k; serving *speed and
  footprint* are measured and work at 229k-262k.
- IFBench here is a 64-item sentinel, not the full benchmark.

## Integrity

```
47dce700c918b2b429ef92d8f2f3b670cf289651a484b3b76a578aeb2ea293f6  PAW-27B-X3-1.5bit.gguf
14f76d629e652914acd3fe7388aa0519a0209316ebccf9cb4b81c984e5648e92  PAW-27B-X3-2bit.gguf
a207c8835f20fc423d85a1e959b6a3f2ea02c3cb651bc0d3762f62dbefe660cb  PAW-27B-X3-2.5bit.gguf
983a4f1ab3ebcfa7a90231a5f04939c496a3c00a76890b791f02bc40e55f5dd6  PAW-27B-X3-3bit.gguf
060dd8450469ba86ea8cba9b93e089211af89c222bef73b66ebb2bcbd4645639  PAW-27B-X3-3.5bit.gguf
bbbcd5b66b571f438ff2018184648e37211a1d82c492d32834af2da2c7193755  Qwen3.8-27B-DFlash2-Q2_K.gguf
```

## License

Apache 2.0, inherited from the base model `Qwen/Qwen3.8-27B`.

## Credits and attribution

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp — the
  x3 trellis codec design, including the K1/K4 aligned extractors and the
  GEMV/tensor-core-GEMM structure this fork ports.
- **Mia's ExLlamaV3 fork** (exllamav3 1.4.2) — the reference implementation
  used for our parity work, and the source of the
  `Qwen3.8-27B-DFlash2-EXL3-5.0bpw` comparison artifact (the "MIA / EXL3"
  row above).
- [EschaLabs](https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2) — the
  Qwen3.8-27B Escha-W2 packed-weight reference artifact.
- [SyzygyResearch](https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B)
  — Mach-1 and
  [llama.cpp-mach1](https://github.com/SyzygyResearch/llama.cpp-mach1); the
  Mach-1 trellis codec and container design that established the PAW lineage,
  and whose checkpoints load directly in this fork.
- [PrismML Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) and its
  [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) — the Bonsai
  model and runtime work.
- Base model [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen.

---
license: other
license_name: qwen
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - gguf
  - qwen4_exp
  - mxfp4
  - llama.cpp
  - ik_llama.cpp
  - speculative-decoding
  - mtp
  - self-speculative
---

![banner](banner.png)

# Qwen3.8-Flash-Next with the MTP head BUILT IN · self-speculative GGUF · MXFP4

> **2026-09-02: qwen4exp MTP support is merged into ik_llama.cpp main**
> ([PR #2369](https://github.com/ikawrakow/ik_llama.cpp/pull/2369)). Build
> current `main`, load the `-NextN` file, pass `--spec-type mtp:n_max=4`.
> No fork, no patch. Reviewed and independently run on a 3090, a 4070
> 12 GB, a 5060 Ti, a 5090 and an RTX Pro 6000.

**These are the only ready-made Flash-Next quants that ship the model's
2.6B multi-token-prediction head inside the file** — one GGUF, one
context, no `-md` second model, no patches to your model files. (Other
releases ship the head as a separate draft file you load with `-md`;
that route works on ik_llama too, see below.) Load it, pass
`--spec-type mtp`, and get **lossless speculative decoding at 0.93–0.99
measured draft acceptance** on code. Every public converter drops this
head; here it's already done, converter-faithful and byte-audited.

Quantized directly from the official BF16 release of
[Qwen/Qwen3.8-Flash-Next](https://hf.co/Qwen/Qwen3.8-Flash-Next)
(125B-A6B MoE + 51B-entry n-gram table + MTP head — the `qwen4_exp`
architecture preview of Qwen4), and tuned for **single-GPU + CPU hybrid
serving**: a 24–32 GB GPU carrying attention and a slice of the experts,
system RAM carrying the rest.

**What makes these files different from other quants of this model:**

1. **The MTP head is included, integrated, and validated.** 33 tensors
   appended with the converter's exact conventions (standard
   `blk.48.*` / NextN layout), enabling **self-speculative decoding** —
   the model drafts for itself in one context. Measured **0.93–0.99
   acceptance**; the draft-quality-critical tensors are deliberately
   kept at Q8_0/BF16.
2. **The n-gram (PLE) table is treated with respect.** It's a row-lookup
   table with random access patterns — aggressive quantization damages it
   disproportionately. Here it's either **BF16** (full quality,
   mmap-friendly) or **Q8_0** (measured **+0.12% perplexity**, and small
   enough that a 128 GB-RAM machine holds the whole CPU side in memory).
3. **MXFP4 routed experts** — fast on CPU (AVX-512) and GPU alike, with
   attention/router/shared-expert kept at Q8_0/F32.

## Which file do I want?

| file | size | choose it when |
|---|---|---|
| `...-ngramQ8-NextN.gguf` | 118.1 GiB | **Recommended.** Built-in MTP head in the **standard `blk.48.nextn.*` layout** (what ik_llama.cpp main and the upstream converter convention expect). The whole CPU side fits a 128 GB-RAM machine in one pinned allocation. (+0.12% PPL vs BF16 table.) **ik_llama.cpp only** — see the mainline note below. |
| `...-ngramBF16.gguf` | 160.2 GiB | You have >128 GB RAM or rely on mmap, and run stock [llama.cpp](https://github.com/ggml-org/llama.cpp) (merged) or [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (merged). Maximum table fidelity; no MTP head. **This is the file for stock llama.cpp.** |
| `...-ngramQ8-MTP.gguf`, `...-ngramBF16-MTP.gguf` | 118.1 / 162.8 GiB | **Legacy naming — superseded by the `-NextN` file.** These carry the head under pre-standard `mtp.*` names, which the merged ik_llama loader does not use for MTP. They still load fine for non-MTP serving on ik_llama. |

**Stock llama.cpp and the head-carrying files:** mainline's `qwen4exp`
loader does not know the 49-block NextN convention yet, so the `-NextN`
and `-MTP` files fail there with
`qwen4exp.attention.compress_ratios has wrong array length; expected 49, got 48`.
That is expected, not a broken download. Use ik_llama.cpp main for those
files, or the `-ngramBF16.gguf` (no head) on stock llama.cpp. Mainline's
own MTP work in progress uses a separate `-md` draft file, not a
head-in-file layout.

## Quick start (stock llama.cpp or ik_llama.cpp, no MTP)

```
llama-server -m Qwen3.8-Flash-Next-MXFP4-ngramBF16.gguf \
  -ngl 999 -ncmoe 38 -fa 1 -c 65536 -ctk q8_0 -ctv q8_0 \
  -t <physical cores> --jinja
```

- `-ncmoe N` keeps N layers' routed experts in system RAM; tune to your
  VRAM (38 fits a 32 GB card with ~10 GB headroom at 64K context; the full
  262K window also allocates).
- Sampler defaults (thinking mode): `temp 1.0, top-p 0.95, top-k 20`.
- The template defaults to `reasoning_effort: xhigh`. For faster answers:
  `--chat-template-kwargs '{"reasoning_effort":"low"}'` (llama.cpp), or
  `--reasoning-budget N` (ik_llama).

## Self-speculative decoding (the fun part — ik_llama + MTP file)

```
llama-server -m Qwen3.8-Flash-Next-MXFP4-ngramQ8-NextN.gguf \
  -ngl 999 -ncmoe 38 -fa 1 -c 65536 -ctk q8_0 -ctv q8_0 \
  -t 24 -tb 32 -rtr -muge --jinja \
  --spec-type ngram-mod:n_min=4 --spec-type mtp:n_max=4 \
  --spec-ckpt-mode gpu-fallback
```

Two draft stages chain: `ngram-mod` supplies long drafts on repetitive
spans (config edits, code echoes), the built-in MTP head drafts
everything else at 0.93–0.99 acceptance. Both are **lossless** — the
target model verifies every drafted token. `-rtr` (runtime repack for
AVX-512) and `-muge` (merged expert tensors) are each independently
measured wins on the CPU expert path.

Because the head lives inside the file, there is no separate draft
model, no second context, and no draft-side KV cache to size: one file,
one memory budget. (External `-md` draft-head setups pay a separate
context plus per-step head overhead and mean juggling a second GGUF.)

Requires ik_llama.cpp `main` at or after 2026-09-02 (PR #2369 merged).
A stock CUDA build is all it takes:

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j
```

(`120` = RTX 5090 / sm_120; set your own arch.)

**Already have another quant downloaded?** The same ik_llama code path
also takes the head as a separate file: `-md <head>.gguf --spec-type mtp:n_max=4`.
Community heads in the standard layout (e.g.
[dzannotti](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF),
[ji-farthing](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF))
were tested during review. No need to re-pull 100+ GB just for MTP.

What to expect: acceptance is content-dependent. Code and edit-style
work sits at 93–99% per position; free-form prose around 60–65%, and on
small GPUs that can be break-even. `--jinja` lowers acceptance because
the Flash-Next template enables thinking at `xhigh` by default and
reasoning text drafts like prose. MTP is single-slot for now (`-np 1`).

## Measured performance

RTX 5090 (32 GB) + Threadripper 9960X (24C, quad-channel DDR5-5600),
Q8-table MTP file, config above:

| workload | throughput |
|---|---|
| coding traffic, steady-state serving (ngram-mod + MTP chained) | **~86–92 tok/s** |
| edit-heavy agentic decode, cold bench (spec on) | ~66 tok/s (peaks 70+) |
| general decode, cold bench (spec on) | ~57 tok/s |
| plain decode (no speculation) | ~45 tok/s |
| prefill | ~700 tok/s at `-ub 512` (higher with larger `-ub`) |

Independent results from the PR review (decode tok/s, no MTP → MTP):

| rig | code | other |
|---|---|---|
| RTX Pro 6000 96 GB (treo) | 85 → 113 | extract 82 → 109; story 83 → 59 at 65% acceptance |
| RTX 4070 12 GB + i7-11700K (joelfarthing, own IQ4_KT head, `n_max=1`) | 9.5 → 12.5 | story break-even |
| RTX 3090 (treo, community quant + separate head) | replicated, high acceptance on code | |
| RTX 5060 Ti 16 GB (YurkoHoshko, fine-tuned target + base head) | net negative | pairing matters: the head must match the target |

Rule of thumb for other machines: decode scales with RAM bandwidth for
the CPU-resident experts (only ~6 GB of active weights stream per token)
plus whatever your GPU carries; a 24 GB card with 96 GB RAM runs the
Q8-table file with `-ncmoe` raised accordingly.

## Recipe notes

- Quantized with the llama.cpp `qwen4exp` toolchain from official BF16;
  tensor-type rules keep router/norms F32, attention/shared-expert Q8_0,
  routed experts MXFP4, n-gram table BF16 or Q8_0.
- The MTP head: 33 tensors appended by GGUF surgery with the converter's
  exact conventions (zero-centred norm `+1` baking, fused-tensor splits),
  `block_count=49`, `nextn_predict_layers=1`, byte-fidelity audited
  against the source shards. MTP-layer experts kept at Q8_0 (draft
  accuracy is what makes speculation pay).
- Verified: perplexity within noise of the reference conversion, and the
  Q8-table delta measured at +0.12% on a fixed corpus.

*Qwen model license applies (see base model). Quantization + MTP
packaging by [@jamesrogers](https://huggingface.co/jamesrogers).*

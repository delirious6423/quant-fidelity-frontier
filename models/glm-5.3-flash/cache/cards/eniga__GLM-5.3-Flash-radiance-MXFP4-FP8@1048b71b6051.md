---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- radiance
- rad
- glm5-next
- mxfp4
- fp8
- moe
- rocm
---

# GLM-5.3-Flash: radiance conversion (MXFP4 experts, FP8 trunk)

This is a [radiance](https://codeberg.org/StillDeadcode/radiance) `.rad` conversion of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B parameters in total, 18B active,
288 routed experts, top-8). The experts are MXFP4 and the trunk's linear layers are block FP8. On two 32 GB cards
with 128 GB of RAM it serves **200k tokens of context** with an FP8 KV cache.

> **It needs a patched radiance.** Radiance 1.17.0's `glm5next` plugin can't serve the FP8 KDA projections in this
> file, and it refuses `--kv-cache-dtype fp8`. The three patches in [`radiance-patches/`](radiance-patches) add both,
> on top of radiance 1.17.0. See [Building radiance](#building-radiance). Upstream main gained its own fp8 GLM cache
> after 1.17.0, so on main or a later release only the two KDA patches are needed. The BF16-trunk sibling,
> [eniga/GLM-5.3-Flash-radiance-MXFP4](https://huggingface.co/eniga/GLM-5.3-Flash-radiance-MXFP4), runs on stock
> radiance with a bf16 cache and at most 64k of context on the same hardware.

| | |
|---|---|
| File | `glm-5.3-flash-mxfp4-fp8.rad`: 164.83 GiB of weights, 154,856-token vocabulary |
| SHA-256 | `0b744aca5bb49bf51292b6e043857bc6250e7b22c78351f415a261afa6d0a66d` (177,001,291,776 bytes) |
| Source | [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) |
| Converted with | `rad-convert` from radiance 1.17.0 with the patches below |
| Routed and shared experts | MXFP4: FP4 E2M1 codes, one E8M0 scale per 32 values, round to nearest, no calibration. Mean relative error 0.123 over 24,854 weights; the worst is 0.144 (`blk.44.ffn_down_exps.184`). |
| Trunk linears | Block FP8: E4M3 codes, one f32 scale per 128×128 block. These are the MLA `q_a`, `kv_a`, `q_b` and `o` projections, the KDA `qkv`, forget, gate and output projections, and the dense FFN of layers 0–2, including the MTP layer's. Mean relative error is 0.0016 to 0.026, depending on the projection. |
| Kept as released | `kv_b` (cast to BF16), KDA `beta`, the DSA indexer, the router, index gates, mHC, norms, embeddings, output head, MTP `eh_proj` and the vision tower |

The recipe ([`glm53-flash-mxfp4-fp8.recipe`](glm53-flash-mxfp4-fp8.recipe)):

```
*attn_kv_b.weight        cast  dtype=bf16
*exp*                    rtn   codes=fp4_e2m1 group=32 scale=e8m0 rule=mx
*attn_qkv.weight         rtn   codes=fp8_e4m3 block=128x128 scale=f32
*attn_forget_a.weight    rtn   codes=fp8_e4m3 block=128x128 scale=f32
*attn_forget_b.weight    rtn   codes=fp8_e4m3 block=128x128 scale=f32
*attn_gate_a.weight      rtn   codes=fp8_e4m3 block=128x128 scale=f32
*attn_gate_b.weight      rtn   codes=fp8_e4m3 block=128x128 scale=f32
*attn_output.weight      rtn   codes=fp8_e4m3 block=128x128 scale=f32
*attn_q_a.weight         rtn   codes=fp8_e4m3 block=128x128 scale=f32
*attn_kv_a.weight        rtn   codes=fp8_e4m3 block=128x128 scale=f32
*attn_q_b.weight         rtn   codes=fp8_e4m3 block=128x128 scale=f32
*ffn_gate_up.weight      rtn   codes=fp8_e4m3 block=128x128 scale=f32
*ffn_down.weight         rtn   codes=fp8_e4m3 block=128x128 scale=f32
```

```
rad-convert GLM-5.3-Flash-BF16/ -o glm-5.3-flash-mxfp4-fp8.rad --recipe glm53-flash-mxfp4-fp8.recipe
```

The conversion took 58 minutes on a Core Ultra 7 270K.

## Building radiance

The patches apply to upstream commit `19d4e8f`, the radiance 1.17.0 release. They change only the `glm5next`
plugin and its attention kernels:

| Patch | What it does |
|---|---|
| `0001` | The KDA projections are served in the encoding the file stores them in (BF16, block FP8 or MXFP4). In 1.17.0 the MLA, dense and expert layers already were; KDA was BF16 only. |
| `0002` | Each KDA projection's GEMM path is chosen from its own stored encoding. Adds tests for FP8 and MXFP4 at TP1 and TP2. |
| `0003` | `--kv-cache-dtype fp8`: the MLA latent cache is stored as E4M3, which halves its size. The attention kernels widen each key to BF16 as they load it; E4M3 to BF16 is exact, so the attention arithmetic is unchanged. A BF16 cache builds the same graph as before. |

```
git clone https://codeberg.org/StillDeadcode/radiance && cd radiance
git checkout -b glm5next-fp8-mx 19d4e8f    # radiance 1.17.0
git am /path/to/radiance-patches/*.patch
# then build the Docker image as radiance's own docs describe
```

Testing on the build below:
- All 156 host tests pass.
- On gfx1201, `r4d_selftest` passes every new `mla_attn_*_fp8kv` case (60, against the reference, maximum error
  about 1e-4). The existing BF16 attention cases still pass.
- The fp8-cache decode kernel is slightly faster than the BF16 one (50.6 µs against 56.2 µs at 2,052 tokens),
  because it reads half the bytes.

Upstream main (after 1.17.0) already has an fp8 latent cache built the same way (commits `17f0eae` and `d858aa7`),
so `0003` is needed only on 1.17.0. The KDA patches (`0001`, `0002`) are not upstream yet. Once they are, a stock
release will serve this file.

## Requirements

- **Patched radiance** (above).
- **Memory.** At `--tp 2`, each rank owns 144 whole experts (expert parallelism), and the trunk is split across
  the ranks. For TP4 and up, see [More cards](#more-cards-tp4-and-tp8-not-tested). Radiance places each expert in VRAM, the pinned host pool, or the file on disk. The `gate_up` experts
  are stored as two planes and joined at load, so the disk tier can't read them in place. In practice every expert
  needs **VRAM or pinned host memory**: about 82 GiB a rank at `--tp 2`. `--host-pool-mib` is **per rank**.
- **The KV cache is replicated on every rank.** The MLA latent is not split by head, so each card holds a full
  copy: 1.39 GiB a card at 200k tokens with `fp8`, or about 2.4 GiB with `bf16`. Adding cards does not shrink it.

## Running it: two 32 GB cards, 128 GB RAM, 200k context

This is the command used for the results below, on two Radeon AI PRO R9700 (gfx1201) cards with 128 GB of DDR5:

```
radiance --model glm-5.3-flash-mxfp4-fp8.rad --tp 2 --p2p off \
  --override-chat-template glm53-flash-tools.jinja \
  --max-num-seqs 1 --max-model-len 204800 --kv-cache-dtype fp8 \
  --placement expert_tiered --host-pool-mib 56832 --host-headroom-mib 1536 \
  --weights-disk-tier --gpu-headroom-mib 104 \
  --mm-max-patches 0 --prefix-cache-mib 0 --checkpoint-slots 2 \
  --served-model-name GLM-5.3-Flash --host :: --port 8000
```

Each card holds 28.2 GiB of weights (the FP8 trunk and about a third of its experts) and 1.39 GiB of KV cache.
Each rank keeps 54.1 GiB of experts in the pinned host pool, and nothing is read from disk. While no request holds
the idle KV blocks, radiance lends them to the experts.

| Flag | Why |
|---|---|
| `--kv-cache-dtype fp8` | Halves the latent cache. With `bf16`, 200k starts, but on stock 1.17.0 a 127k-token request failed at about 75k: the experts that had borrowed the cache's VRAM had no room to move back to. |
| `--host-pool-mib 56832` | 54.1 GiB is used a rank, plus the KV blocks lent to the experts. A larger pool leaves too little RAM for the checkpoint slots. |
| `--max-num-seqs 1` | One sequence at 200k. Each extra sequence costs KV cache and about 140 MiB of KDA state a card. |
| `--mm-max-patches 0` | Text only: keeps the vision tower off the cards. |
| `--prefix-cache-mib 0 --checkpoint-slots 2` | Cuts the pinned host memory taken by the prefix cache and the linear-state snapshots. |
| `--override-chat-template glm53-flash-tools.jinja` | The release template uses Jinja's `unique` filter twice, and radiance's bundled Jinja engine doesn't have it, so tool support was detected as off. [`glm53-flash-tools.jinja`](glm53-flash-tools.jinja) replaces those filters with a loop and is otherwise unchanged. |

If radiance stops with "out of memory", the lines above its startup report name the shortfall and the flag to
change. `--p2p` defaults to `auto`. P2P is disabled on the R9700s above, so this setup passes `--p2p off`.

## Results: TP2, two R9700 cards, 128 GB RAM

Hardware: an Intel Core Ultra 7 270K Plus, 4×32 GB DDR5-5400 (STREAM copy 68 GB/s), and two Radeon AI PRO R9700
cards capped at 210 W, each on a PCIe 5.0 x8 link. The model's MTP head drafts 3 tokens a step.

**Quality.** GSM8K, 150 questions, greedy, thinking off:

| Build | KV cache | Score |
|---|---|--:|
| BF16 trunk ([MXFP4 sibling](https://huggingface.co/eniga/GLM-5.3-Flash-radiance-MXFP4)) | bf16 | 98.0% |
| This file | bf16 | 98.0% |
| This file | fp8, 200k | 98.0% |

With a bf16 cache, this file gives the same final answers as the BF16-trunk build on all 150 questions. Switching
to the fp8 cache changes one final answer, and the score doesn't move.

**Long context.** One 188,644-token prompt (Python source) hid three passphrases, at 5%, 50% and 95% of its
length. The model returned all three: 3/3. Time to first token was 614 s, and decode at that depth was 12.2 tok/s.

**Speed** (`ctxtest`: non-thinking, 400 tokens out):

| Prompt | Prefill | Decode |
|--:|--:|--:|
| 0.5k | — | 14.3 to 16.4 tok/s |
| 18.5k | 303 tok/s | 14.1 tok/s |
| 188.6k | 307 tok/s (614 s) | 12.2 tok/s |

**BetterBench 0.6.0** (corpus v1.0, 20 passes per category, temperature 0.7, cold prefix cache), with the command
above at 200k context:

| Single stream | This file, fp8 KV, 200k | [BF16-trunk sibling](https://huggingface.co/eniga/GLM-5.3-Flash-radiance-MXFP4), bf16 KV, 16k |
|---|--:|--:|
| Decode, weighted across categories | **14.7 tok/s** (13.2 chat to 16.2 json) | 13.6 tok/s |
| Time to first token, p50 | 2.3 s | 2.3 s |
| Gap between stream updates, p99 | 285 ms | 277 ms |
| Tokens per update (MTP) | 2.3 to 3.2 | 2.3 to 3.2 |
| Aggregate decode, 48 requests at concurrency 1 | 13.6 tok/s | 12.1 tok/s |

| Prompt processing | Prompt tokens | TTFT | Prefill |
|--:|--:|--:|--:|
| 2k | 1,514 | 5.6 s | 269 tok/s |
| 8k | 5,892 | 19.2 s | 307 tok/s |
| 16k | 11,799 | 38.0 s | 310 tok/s |
| 32k | 23,548 | 75.1 s | 314 tok/s |
| 64k | 47,014 | 146.1 s | 322 tok/s |

The sibling's numbers come from radiance 1.14.0 at 16k context with two sequences, because that build can't reach
200k. Prefill here is about 11% lower than the sibling's at 8k–16k (307–310 against 344–350 tok/s): the larger cache
and smaller host pool leave less room for experts.

At the same 64k context and bf16 cache, the FP8 trunk decodes about 11% faster than the BF16 trunk (13.9 against
12.5 tok/s), and its prefill is about 28% faster at 47.5k tokens (373 against 292 tok/s). Raising the context to
200k costs about 20% of prefill (303 against 373 tok/s at 18.5k). The larger cache and the smaller host pool leave
less room for experts.

Decode is paced by PCIe. About two thirds of the experts sit in host RAM and are read over each card's x8 link
when a token routes to them. Changing the MTP depth, making the draft window adaptive, turning MTP off, or pinning
the hottest experts all moved decode by less than 1 tok/s. `--deterministic` was slower, at about 10 tok/s.
`--num-speculative-tokens 0` makes prefill about 20% faster and frees about 2 GiB of RAM a rank, for about 1 tok/s
less decode. GLM-5.3-Flash thinks at length, so long answers take minutes at these speeds.

## More cards: TP4 and TP8 (not tested)

The `glm5next` plugin accepts `--tp` 1, 2, 4 or 8. Every split width in this file is a multiple of the 128×128 FP8
block at all four sizes. Radiance's own tests build BF16 GLM graphs at TP1, 2, 4 and 8. This file and the patches
were run only at TP2: the patches' tests cover the FP8 and MXFP4 trunk at TP1 and TP2, and the fp8 cache at TP2.
The estimates below are not measurements.

**How the experts are laid out changes by version.** On 1.17.0, and so with these patches, each rank owns 288/N
whole experts at every TP size. Upstream main (commits `8ef50a7` and `f713f44`) splits each expert by width across
the ranks at TP4 and up when every routed expert is MXFP4, as this file's are, because that is the shape its MXFP4
kernels are tuned for. Upstream measured whole experts at four ranks on 4 × R9700 costing two thirds of prefill and
37% of four-stream throughput. Main keeps whole experts at TP2, or when `--weights-disk-tier` is set. **For TP4 or
TP8, use upstream main with patches `0001` and `0002`, and drop `--weights-disk-tier`.** The memory per rank below is
about the same with either layout.

| | TP2, 32 GB cards | TP4, 32 GB cards | TP8, 32 GB cards |
|---|--:|--:|--:|
| Experts a rank | 144 whole | 72 whole, or ¼ of all 288 on main | 36 whole, or ⅛ of all 288 on main |
| Weights a rank (approx.) | 82 GiB | 41–42 GiB | 21 GiB |
| In VRAM a card | 28 GiB | ~28 GiB | all of it |
| Pinned host pool a rank | 54 GiB | ~13–14 GiB | ~0 |
| Pinned host RAM in total | ~110 GiB | ~55 GiB | a few GiB |
| fp8 KV at 200k, each card | 1.39 GiB | 1.39 GiB | 1.39 GiB |

- **TP4.** About two thirds of each rank's experts fit in VRAM. Radiance keeps the hottest experts there, and on
  this model the hottest third take about 73% of decode routings, so far fewer bytes cross PCIe per token.
  System RAM is no longer the limit: there is room for several sequences, the prefix cache and more checkpoint
  slots. A smaller `--host-pool-mib` is enough. Start near 16384 and read the startup report.
- **TP8.** Every expert fits in VRAM with room to spare. Decode stops being link-bound and is limited by the GPUs
  and the collectives instead. The spare VRAM goes to KV, so several 200k sequences fit.
- **Collectives.** Every layer exchanges partial results between ranks. With `--p2p off` they go through pinned
  host memory, and that costs more at 4 and 8 ranks. If your platform supports P2P, measure both.
- **Load balance.** With whole experts, each token's 8 experts are spread unevenly over the ranks, and the busiest
  rank sets the step time. Main's width split at TP4 and up avoids this, because every rank works on every chosen
  expert.
- **PCIe.** Four or eight cards usually need a workstation or server platform. Give every card a full x16 link: at
  x8, or behind a switch, the host-RAM expert reads at TP2 and TP4 get slower.

## License

MIT, the same as the base model. Converted and patched by eniga; all credit for the model goes to Z.ai.

---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: radiance
tags:
  - radiance
  - exl3
  - moe
  - rocm
  - gfx1201
---

# Qwen3.8-Flash-Next: EXL3 experts for radiance (`Dyluhn/qwen3.8-flash-next-exl3`)

A [radiance](https://huggingface.co/StillDeadcode) container (`.rad`) of **Qwen3.8-Flash-Next** with its routed
experts in **EXL3**. Each routed expert of the trunk and the MTP head comes from turboderp's EXL3 conversion: 4.05
bpw, `turboderp/Qwen3.8-Flash-Next-exl3`, branch `4.05bpw_h6_ng6`, commit
`55a732e0c4c3d4614bc42b68493bb930d9b02c0a`. **Every other weight is byte-identical to radiance's published container**
`StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`: the attention and delta-net trunk, the shared experts, the int8
lm_head, the n-gram table and the protected bf16 experts. Only the routed-expert rule of the recipe changed.

It needs the **radiance-exl3 plugins** ([`Dyluhn/radiance-exl3`](https://github.com/Dyluhn/radiance-exl3)), which add the EXL3 expert kernels and the
`("qwen4exp", "exl3")` architecture to radiance 1.6.0 (plugin release v1.2.0; v1.1.0 is for radiance 1.5.0, v1.0.1 for 1.4.0). Nothing in radiance itself is modified.

| file | size | sha256 |
|---|---|---|
| `qwen3.8-flash-next-exl3.rad` | 120,531,828,736 B (112.3 GiB) | `187ae8186cc1f2ba8bce96565592af7fc6f6ec91c9ece6b4d352eaf0432ee26f` |

## Quality

Measured with radiance's KL mode against the bf16 model on **ref64**. That is 64 documents, 52,260 tokens and
52,132 scored positions, stratified over code, niche topics, prose, math, multilingual text, agent traces and one
12K-token long document. The reference is the bf16 model's full-vocabulary log-probabilities at every position.
Both containers were scored with the same flags, in the same session, on the same reference (each run twice; rows
byte-identical across runs).

| | baseline container (fp8 / iq4 experts) | this container (EXL3 experts) |
|---|---|---|
| mean KL vs bf16 | 0.0537 | **0.0467** (−12.9%; 95% CI of the difference [−0.0102, −0.0041]) |
| top-1 agreement | 93.6% | **94.0%** |
| perplexity (bf16 4.0775) | 4.0983 (×1.0051) | 4.0507 (×0.9934) |

EXL3's mean KL is below the baseline's on every source slice (code, long, math, multilingual, niche, prose with
95% intervals below zero; agent −0.022 with an interval that includes zero). Top-1 is at or above the baseline's
on every slice.
It was re-scored on the shipped kernels at tp2: mean 0.0467, top-1 94.06%.

On radiance 1.5.0 with plugin v1.1.0, and on radiance 1.6.0 with plugin v1.2.0, the output at tp2 is identical to
radiance 1.4.0's (the same deterministic ident hashes), so these numbers hold there.

## Speed

Radeon AI PRO R9700 ×2 (gfx1201, 32 GB; one card behind a PCIe Gen5 x16 path, one behind Gen4 x4), radiance 1.6.0
(`stilldeadcode/radiance:1.6.0`) with plugin release v1.2.0, `--tp 2 --tp-wire wht6 --max-model-len 200000
--placement expert_tiered --kv-cache-dtype fp8 --max-num-batched-tokens 2048 --gpu-headroom-mib 3072`. The baseline is
at its `--host-pool-mib 12288`, this container at 11793, which gives the same VRAM split. Same session, interleaved.
All numbers come from this one machine. On it the x4 card holds the baseline back more than this container at high
concurrency, so the C=4 and C=8 figures favour this container compared with a machine whose cards all sit on fast
links. Nothing in the plugin is tuned to these links.

**Decode, MTP off** (ms a step, lower is better; radiance's cstep2, one clean pass each; it sends the same prompt to
every sequence, so at temperature 0 they route through the same experts -- with a distinct prompt per sequence this
container read 10.65 / 11.89 / 13.91 ms against the baseline's 11.20 / 12.12 / 15.62 at C=1/2/4):

| C | baseline | this container |
|---|---|---|
| 1 | 10.79 | 10.86 |
| 2 | 10.91 | 10.90 |
| 4 | 12.04 | 11.85 |
| 8 | 14.07 | 14.24* |

\* the harness flagged that level: a card read under 95% busy inside its window, so it may read slow.

**BetterBench 0.6.0** (a corpus of distinct prompts over eight categories, temperature 0.7, cold prefix cache;
`--num-speculative-tokens 3`; one run each):

| | baseline | this container |
|---|---|---|
| combined decode (weighted over categories) | 130.2 tok/s | 139.5 tok/s |
| TTFT p50 | 264 ms | 199 ms |
| total tok/s at C=1 / 2 / 4 | 108 / 134 / 141 | 116 / 150 / 171 |
| total tok/s at C=8 / 16 | 126 / 122 | 170 / 165 |
| prefill, 2K .. 64K prompt | 1,033 .. 1,365 tok/s | 1,084 .. 1,442 tok/s |
| MTP draft acceptance | 52–57% | 52–56% |

Draft acceptance is the same; the gain is per-step speed. Single-prompt benchmarks moved either model's acceptance by
10–20 points with the prompt's text, so they are not a fair MTP comparison.

**Prefill and time to first token** (radiance's default prefill staging):

| | baseline | this container |
|---|---|---|
| prefill, 8K-token prompt | 1,546 tok/s | 1,663–1,673 tok/s |
| prefill, 24K-token prompt | 1,600 tok/s | 1,541–1,709 tok/s |
| TTFT p50, 1 request | 164-170 ms | 112-114 ms |
| TTFT p50, 8 concurrent | 1,205–1,209 ms | 1,133–1,609 ms |

TTFT for 8 concurrent requests depends on how a burst lands against the free sequence slots and moves by a few
hundred ms between runs of the same build. On a card behind a slow PCIe link (measured: PCIe 4.0 x4), serve with
`-e EXL3_PREFILL_STAGE=auto` (see the plugin README): on this machine it roughly doubled tp2 prefill.

## Running it

1. Build or download the plugin home for radiance 1.6.0 (release v1.2.0) ([`Dyluhn/radiance-exl3`](https://github.com/Dyluhn/radiance-exl3), README "Building the plugin home").
2. Serve with the plugin home ahead of radiance's own:

```sh
docker run --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 \
    --init -p 8000:8000 \
    --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
    -v /path/to/plugin-home:/plugins:ro -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance \
    -v /path/to/models:/models:ro stilldeadcode/radiance:1.6.0 \
    --model /models/qwen3.8-flash-next-exl3.rad \
    --tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered \
    --host-pool-mib 11793 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 --num-speculative-tokens 3
```

The server listens on port 8000. It needs two 32 GB cards and about 65 GiB of free RAM for pinned memory. Keep
`--max-num-seqs` small: the default of 256 does not fit a 32 GB card. Every flag the baseline container is served
with applies. The device kernels are built for gfx1201 (RDNA4) only.
[`Dyluhn/radiance-exl3`](https://github.com/Dyluhn/radiance-exl3)'s `docs/TROUBLESHOOTING.md` covers memory sizing (`--host-pool-mib`), the plugin path and every
refusal message.

## Changes

- Plugin v1.2.0 (radiance 1.6.0): the fused expert kernel's hang is fixed at its cause, so it runs at 4 to 16 rows
  again (a C=1 MTP step 0.26 ms shorter); `--tp 4` splits the experts in four classes by number mod 4, as the baseline
  does (every rank holds the same expert bytes); prefill staging follows radiance's default. Same output at tp2.
- Plugin v1.1.0 (radiance 1.5.0) and v1.0.1 (radiance 1.4.0) fix a hang at 4 concurrent requests with MTP on in
  plugin v1.0.0. Same output.

## Known limitations

- **One machine.** Every speed figure above comes from one box with one card on a PCIe 4.0 x4 path; machines whose
  cards all sit on fast links have not been measured by us with this release.
- **Room to improve, measured:** after the routed experts of a step of more than 16 rows the next memory-bound kernel
  runs a few microseconds slower than after the baseline's (about 0.35-0.55 ms a step at 8-16 concurrent requests);
  at C=1 with MTP a step is longer than the baseline's.
- **tp1** is served. On the x16 card prefill measured a few percent under the baseline (radiance 1.4.0/1.5.0, single
  passes) with TTFT ahead; on the x4 card prefill and decode are ahead.
- **radiance KV-loan recall (seen on 1.4.0):** a long request can fail with `expert slab could not give back the loan` when
  the host pool is full. Give the pool room (`--host-pool-mib 43000` at tp1 served every request where 42010
  failed).
- **tp3 / tp4** are covered by host tests and kernel suites only, without engine runs. **tp8 is refused** (an
  expert is five 128-column blocks), as the baseline refuses it.

## Credits

- **Qwen** for Qwen3.8-Flash-Next (Qwen Community License 1.0, `LICENSE`).
- **turboderp** for EXL3 and the 4.05 bpw expert conversion (`turboderp/Qwen3.8-Flash-Next-exl3`).
- **The radiance author (StillDeadcode)** for radiance, its container format, the converter, and the baseline
  container every non-expert weight here is copied from.

## License

The model weights are a derivative of Qwen3.8-Flash-Next and are distributed under the **Qwen Community License
1.0**. Its full text is in `LICENSE`, and its copyright and permission notice must accompany every copy:

> Qwen Community License 1.0 — Copyright (c) 2026 Qwen. See `LICENSE` for the conditions, including the
> display-name requirement above 100,000,000 monthly active users or US$ 20,000,000 monthly revenue and the
> separate licence required for Model-as-a-Service and AI Work Assistant businesses.

The radiance-exl3 plugin code ([`Dyluhn/radiance-exl3`](https://github.com/Dyluhn/radiance-exl3)) is licensed separately: Apache-2.0.

---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
quantized_by: janghaneul
pipeline_tag: text-generation
library_name: gguf
tags:
- atomic-chat
- qwen
- qwen3.8
- flash-next
- moe
- gguf
- imatrix
- llama.cpp
- n-gram-ple
---

# AtomicChat/Qwen3.8-Flash-Next-GGUF AD-5.00bpw-Q5_K_M derived builds (gate/up Q4_K · IQ2_S-IQ3_S-Q4_K)

Two hand-assembled, measurement-driven quantizations of **Qwen3.8-Flash-Next** built for
local MoE inference (Strata / llama.cpp), from the same `AD-5.00bpw-Q5_K_M` recipe. Both
change only the routed experts — everything non-routed (attention, GDN `ssm_out`, shared
experts, embeddings, and the n-gram PLE table) stays Q8_0, and down experts stay IQ4_NL.
Pick by how much RAM you have: the point of these builds is *where the bits go* in
gate/up, the one group that decides quality on this model.

## The builds

| Build | gate/up | Total | Resident (PLE streamed) | bpw | bpw excl. PLE |
|---|---|---:|---:|---:|---:|
| `Q4_K`  | Q4_K ×96 | 119.05 GiB | **68.37 GiB** | 5.7788 | 4.6715 |
| `IQ2_S-IQ3_S-Q4_K` | Q4_K ×8 + IQ3_S ×80 + IQ2_S ×8 | 109.22 GiB | **58.56 GiB** | 5.30 | 4.051 |

`Q4_K` is the quality build: every gate/up expert at Q4_K. `IQ2_S-IQ3_S-Q4_K` is the leaner build: it keeps Q4_K only on the four highest-tolerance
layers (40, 44, 45, 47), drops the four lowest-tolerance layers (32, 34, 36, 39) to IQ2_S,
and fills everything else with IQ3_S. The tier placement follows where three independent
vendor allocations (AtomicChat AD, ISTA GSQ, agention Gyro) agree on the extremes. It
is the direct successor of the earlier `IQ3_S-Q4_K` build (416 bpw, delisted): same four
Q4_K anchor layers, 36 of its gate/up layers byte-identical, 12 layers downgraded for
−2.36 GiB.

## Composition (byte-verified against the shipped files)

Both builds:

| Group | Type | Count | Size |
|---|---|---:|---:|
| `ffn_down_exps` | IQ4_NL | 48 | 21.09 GiB |
| attention (`attn_qkv`, `attn_gate`), `ssm_out`, shared expert, `token_embd`, `output` | Q8_0 | ~690 | ~5.1 GiB |
| `per_layer_token_embd` (n-gram PLE table) | Q8_0 | 1 | 50.66 GiB |
| router / norm / indexer | F32/F16 | 389 | 0.24 GiB |

gate/up only:

| Build | `ffn_gate_exps` / `ffn_up_exps` | Layers |
|---|---|---|
| `Q4_K` | Q4_K ×96 | all |
| `IQ2_S-IQ3_S-Q4_K` | Q4_K ×8 · IQ3_S ×80 · IQ2_S ×8 | 40, 44, 45, 47 · all others · 32, 34, 36, 39 |

## Design rationale

1. **Gate/up is where quality lives.** This build starts from AtomicChat's
   `AD-5.00bpw-Q5_K_M` recipe and replaces only the gate/up expert tensors with Q4_K
   (byte-level diff vs AD-5.00: gate/up `IQ2_S×72 + IQ3_S×24 → Q4_K×96`, everything
   else identical). On our internal task benchmarks this was the single change that
   eliminated all remaining errors, at +15.3 GiB.
2. **Down experts are stuck at block-32 — and that's fine.** `moe_intermediate_size=640`
   is not divisible by 256, so k-quant/i-quant block-256 types silently fall back for
   `ffn_down_exps`. IQ4_NL (4.5 bpw) is the practical floor/quality sweet spot there;
   we deliberately do *not* spend bits on down.
3. **The PLE/n-gram table stays Q8_0.** It is a random-access `GET_ROWS` table with no
   imatrix statistics; quantizing it below ~5 bpw measurably hurts retrieval-style
   behaviour. Q8_0 costs 50.66 GiB of *disk*, not RAM, when the engine streams it.
4. **Non-expert tensors are cheap insurance.** Attention/GDN/embedding at Q8_0 is ~5 GiB
   total; every build of this model that trimmed them (attention → Q6_K) regressed in
   our tests (correlation, not proven causation).

## Shards

Each build is a 3-shard split GGUF. The third shard of every build is the same n-gram
table, shipped under each build's own name so that each set is complete on its own
(same bytes, sha256 `290aab43…`):

| `Q4_K` | `IQ2_S-IQ3_S-Q4_K` | Size |
|---|---|---:|
| `…467bpw-q5_k_m-00001-of-00003.gguf` | — | 59.98 GB |
| `…467bpw-q5_k_m-00002-of-00003.gguf` | — | 13.45 GB |
| `…467bpw-q5_k_m-00003-of-00003.gguf` | — | 54.40 GB |
| — | `…405bpw-q4_k_m-00001-of-00003.gguf` | 50.74 GB |
| — | `…405bpw-q4_k_m-00002-of-00003.gguf` | 12.15 GB |
| — | `…405bpw-q4_k_m-00003-of-00003.gguf` | 54.40 GB |

## Provenance

- BF16 source: `unsloth/Qwen3.8-Flash-Next-GGUF` (BF16 tier, pinned revision `d9620119`)
- imatrix: AtomicChat's public imatrix for Qwen3.8-Flash-Next
- Quantizer: llama.cpp, imatrix dot-quantization
- Base recipe: `AtomicChat/Qwen3.8-Flash-Next-GGUF` `AD-5.00bpw-Q5_K_M`
- Construction: per-tensor Range download + byte splice; no re-quantization of
  unchanged tensors

## Running

Strata engine (262K context, PLE streamed) — same command for either build, swap the shard
names:

```bash
# quality build
strata --serve \
  --native Qwen3.8-Flash-Next-AD-467bpw-q5_k_m-00001-of-00003.gguf \
  --ple-gguf Qwen3.8-Flash-Next-AD-467bpw-q5_k_m-00003-of-00003.gguf \
  --ple-io direct --resident-experts \
  --max-context 262144 --kv int8

# 58.6 GiB build (leanest)
strata --serve \
  --native Qwen3.8-Flash-Next-AD-405bpw-q4_k_m-00001-of-00003.gguf \
  --ple-gguf Qwen3.8-Flash-Next-AD-405bpw-q4_k_m-00003-of-00003.gguf \
  --ple-io direct --resident-experts \
  --max-context 262144 --kv int8
```

Standard llama.cpp works too (load shard 1 of either set). With ~94 GiB system RAM, run
the PLE table from SSD and keep context ≤262K with int8 KV. Without the PLE table on
disk, `IQ2_S-IQ3_S-Q4_K` is the build to try first.

The vision adapter `mmproj-Qwen3.8-Flash-Next-F16.gguf` is included (byte-identical to
the one in the upstream
[`AtomicChat/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)
repo).

**The n-gram PLE shard is swappable.** Shard 3 (`-00003-of-00003`) contains only the
shared `per_layer_token_embd` table and is byte-identical to the `-M64-00003-of-00003`
shard shipped in every M64 build of `AtomicChat/Qwen3.8-Flash-Next-GGUF`. If you already
own one of those quants, point `--ple-gguf` at its existing n-gram shard instead of
downloading this one again; the smaller n-gram shards from the 3.84/4.27 bpw builds
(35.8 GiB) load too, trading a little retrieval quality for disk.

**Quality caveat:** composition and byte-level equality claims are verified against the
shipped files for all three builds. `IQ2_S-IQ3_S-Q4_K` passed an internal agentic
creative-coding benchmark (Three.js voxel scene, 60 fps, zero shader errors) served with
the same engine settings as the quality build; no standard benchmark suite was run on any
of the three.

## Demos

One-shot creative-coding outputs from these builds are published in a demo gallery:
**[Atomic Creative Demos](https://huggingface.co/spaces/janghaneul/atomic-creative-demos)**
— a pelican-on-a-bicycle SVG animation (`4.67 bpw`), three independently written voxel
pagoda Three.js scenes (`4.67 bpw` high- and medium-effort runs and `4.05 bpw`,
identical prompt under anti-copy rules), and a voxel harbour scene (`4.67 bpw`, high effort)
written from a single sentence of prompt with no rules and no verification pass. Every
page is a one-shot result: the model's unedited single-file output from one agentic run.
Card titles follow the scene each run was asked for; where a run titled its own page
differently, the caption says so.
Build labels carry the PLE-excluded bpw the shard filenames use (4.67 and 4.05),
not the whole-file totals in the table above.

<table>
<tr>
<td><a href="https://janghaneul-atomic-creative-demos.static.hf.space/pelican/index.html"><img src="https://huggingface.co/janghaneul/Qwen3.8-Flash-Next-AD-gateup-GGUF/resolve/main/imgs/pelican.png" width="320" alt="Pelican on a Bicycle — SVG animation by the 4.67 bpw build"></a><br><sub>Pelican on a Bicycle · <code>4.67 bpw</code> — click to open</sub></td>
<td><a href="https://janghaneul-atomic-creative-demos.static.hf.space/pagoda-q4k/index.html"><img src="https://huggingface.co/janghaneul/Qwen3.8-Flash-Next-AD-gateup-GGUF/resolve/main/imgs/pagoda-q4k.png" width="320" alt="Voxel pagoda floating island — Three.js scene by the 4.67 bpw build"></a><br><sub>Floating Pagoda · <code>4.67 bpw</code> high-effort run — click to open live</sub></td>
</tr>
<tr>
<td><a href="https://janghaneul-atomic-creative-demos.static.hf.space/pagoda-iq2s-iq3s-q4k/index.html"><img src="https://huggingface.co/janghaneul/Qwen3.8-Flash-Next-AD-gateup-GGUF/resolve/main/imgs/pagoda-iq2s.png" width="320" alt="Voxel pagoda floating island — Three.js scene by the 4.05 bpw build"></a><br><sub>Floating Pagoda · <code>4.05 bpw</code> — click to open live</sub></td>
<td><a href="https://janghaneul-atomic-creative-demos.static.hf.space/harbour-q4k/index.html"><img src="https://huggingface.co/janghaneul/Qwen3.8-Flash-Next-AD-gateup-GGUF/resolve/main/imgs/harbour-q4k.png" width="320" alt="Voxel sailing ship calling at a harbour — Three.js scene by the 4.67 bpw build"></a><br><sub>Harbour Call · <code>4.67 bpw</code> high-effort run, one-sentence prompt — click to open live</sub></td>
</tr>
</table>

<em>Harbour Call</em> came from the shortest possible ask — one sentence, no anti-copy rules
and no headless verification loop — 97,548 voxels in 5 instanced draws, 84 draw calls,
1.16 M triangles, all geometry procedural; an
<a href="https://janghaneul-atomic-creative-demos.static.hf.space/harbour-q4k/harbour-offline.html">offline build</a> (three.js inlined) sits next
to it in the gallery. The gallery also holds the `4.67 bpw` medium-effort pagoda run, which is
not pictured here.

## 한국어 요약

- `AD-5.00bpw-Q5_K_M` 원본에서 전문가 gate/up만 다시 채운 두 가지 빌드입니다. down 48개는 IQ4_NL, attention·shared expert·임베딩·n-gram PLE 표는 Q8_0 그대로입니다.
- **`Q4_K-Q4_K`**: gate/up 96개 전부 Q4_K. 전체 119.05 GiB · 5.78bpw, PLE 표를 SSD에서 스트리밍하면 실제 RAM 약 **68.4 GiB**. 품질 우선 빌드.
- **`IQ2_S-IQ3_S-Q4_K`**: gate/up 중 Q4_K는 내성 최상위 4개 레이어(40, 44, 45, 47)만, IQ2_S는 최하위 4개(32, 34, 36, 39), 나머지는 IQ3_S. 전체 109.22 GiB · 5.30bpw, 실제 RAM 약 **58.6 GiB**(PLE 제외 4.05bpw). 이전 416bpw 빌드와 12개 레이어만 다르고(전부 강등, −2.36 GiB) 나머지 36개 레이어는 바이트 동일. 내부 에이전틱 벤치 통과.
- 두 빌드의 n-gram 샤드(00003)는 바이트 단위로 같은 파일이고, AtomicChat M64 빌드들의 n-gram 샤드와도 동일합니다. 이미 다른 빌드의 n-gram 샤드가 있으면 그대로 쓰면 되고, 더 작은 n-gram 샤드로 교체해도 동작합니다(검색 계열 품질 소폭 손실).
- 이 빌드들 자체에서는 추론·품질 측정을 하지 않았습니다. 텐서 구성과 바이트 검증만 완료된 상태입니다.
- 데모 갤러리: [Atomic Creative Demos](https://huggingface.co/spaces/janghaneul/atomic-creative-demos) — **원샷 결과물** 모음. `4.67 bpw`가 만든 자전거 펠리컨 SVG 애니메이션과, 세 가지 실행(`4.67 bpw` high·medium, `4.05 bpw` high)이 같은 프롬프트로 독립적으로 작성한 복셀 오층탑 three.js 씬 비교. 전부 손대지 않은 단일 파일 원샷 결과물입니다.
  같은 갤러리에 `4.67 bpw` high 실행이 만든 복셀 항구 씬 **Harbour Call**(97,548 복셀 · 5회 인스턴스 드로 · 1.16M 삼각형)도 있습니다. 이건 규칙도 검증 루프도 없는 한 문장 프롬프트 원샷 결과물이고, three.js를 내장한 오프라인 빌드가 함께 있습니다. 카드 제목은 프롬프트가 요구한 장면 이름으로 통일했고, 페이지 스스로 단 제목은 캡션에 따로 적었습니다.
- 라이선스는 원본과 동일한 Qwen Community License 1.0을 상속합니다.

## Contents

- `Q4_K`: 3 GGUF shards (single logical model)
- `IQ2_S-IQ3_S-Q4_K`: 3 GGUF shards (single logical model)
- `mmproj-Qwen3.8-Flash-Next-F16.gguf` (vision adapter)
- `validation.json` / `manifest.json` — per-tensor types, sizes, checksums

## License

Inherits the upstream **Qwen Community License 1.0**; the imatrix and base recipe derive
from AtomicChat's quantization work and the BF16 tier from Unsloth — see their repos.
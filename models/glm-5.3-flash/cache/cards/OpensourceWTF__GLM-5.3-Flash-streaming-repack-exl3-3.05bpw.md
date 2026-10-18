---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: mlx-serve
pipeline_tag: text-generation
language:
- en
- zh
tags:
- mlx
- mlx-serve
- mlx-stream
- apple-silicon
- mixture-of-experts
- expert-streaming
- exl3
- glm
---

# GLM-5.3-Flash, EXL3 K=3 fp8res mlx-stream pack

A serving pack of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total parameters, 18B
active, arch `glm5_next`) for [mlx-serve](https://github.com/ddalcu/mlx-serve) with the
[mlx-stream](https://github.com/davidtai/mlx-stream) plugin on Apple silicon. The routed experts (the bulk of the model)
stay on disk and stream from the SSD per request. The attention, dense, shared-expert and router weights stay resident.
On one M5 Max (128 GB) this serves the model at about 1,090 tok/s prefill and 33-37 tok/s decode for a 16K-token
prompt, and holds the full 1,048,576-token context.

What is in the pack:

- **Routed experts: EXL3, K=3 (3 bits per weight), mul1 codebook.** A record-by-record repack of
  [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), branch `3.05bpw` (revision
  `332ab457b709b7ba30dd9a448be5de03b80a7ac9`), into fixed 9,474,048-byte records (`experts.bin`). The trellis codes and
  scales are turboderp's bytes, unchanged (each record byte-verified against the source tensors on read-back). The
  quantization is exllamav3's calibrated EXL3; we did not re-quantize the experts.
- **Residents: FP8 e4m3, 64x128 blocks (`fp8_block128`), plus BF16.** The trunk's KDA, DSA, indexer, shared-expert and
  dense projections are FP8 (round to nearest, one scale per tile); the embeddings, `lm_head`, norms, routers and the
  mHC tensors stay BF16. Source: the BF16 tensors of [vcruz305/GLM-5.3-Flash-NVFP4](https://huggingface.co/vcruz305/GLM-5.3-Flash-NVFP4)
  (revision `b7d3cbf6`, the release's non-expert weights). The DSA, indexer, shared and dense groups were FP8 in the
  release and re-quantize almost exactly; the KDA projections are the lossy group (relative RMS error about 0.027).
  `residents-fp8-manifest.json` has the recipe and the per-group error.
- **The MTP draft bank: NVFP4.** The release's multi-token-prediction layer's 288 experts, quantized to NVFP4
  (`mtp-experts-nvfp4.bin`). The draft lane only proposes tokens; the trunk verifies every token with exact acceptance.
- **The usage profile `expert-usage-v1.json`.** Per-layer expert route counts from 65 requests (HumanEval problems 0-63
  and one 16K prompt). At load the plugin fills the prompt rows from it (`resident_seed: usage`, the default when the
  file is present). It cuts the first request's TTFT by 62 % (see [Warm layers](#warm-layers)).

License: MIT, inherited from the release (zai-org/GLM-5.3-Flash is MIT; turboderp/GLM-5.3-Flash-exl3 and
vcruz305/GLM-5.3-Flash-NVFP4 are MIT). The model is Z.ai's.

`hf_quant_config.json` is the NVFP4 source pack's file, kept unchanged. Because of it the host prints
`4-bit nvfp4 quant` in its model line at load, and the plugin's install line says `experts NVFP4 g16 streamed`. Both are
cosmetic: the experts are EXL3 K=3, and the plugin's experts line says so (`glm5_next experts: exl3 k3 mul1 routes`).

## Files (130.8 GB)

| file | bytes | sha256 | what |
|---|---:|---|---|
| `experts.bin` | 114,598,084,608 | `345ac3f68ab3...9580e4` | the expert bank: 42 MoE layers (model layers 3-44) x 288 routed experts = 12,096 records of 9,474,048 bytes (gate, up, down: EXL3 K=3 codes + input/output sign vectors) |
| `expert-manifest-v2.json` | 4,628,687 | `ac7ffe3d78d9...edc24e` | the bank's manifest (format `mtplx-expert-manifest-v2`): geometry, per-layer offsets, segments |
| `expert-manifest.json` | 22,556,421 | `ea23e6a7392d...59f072` | the runtime manifest (v1): per-record segments and tensor names |
| `model-0000{1..5}-of-00005.safetensors` | 12,043,235,276 | in `manifest.sha256` | the residents: FP8 projections + BF16 tensors; also the MTP layer's non-expert tensors and the vision tower (not loaded) |
| `model.safetensors.index.json` | 197,774 | `6bcf39862348...422944` | the residents' index |
| `residents-fp8-manifest.json` | 520,635 | `f2a421ecc683...dedcb4` | the FP8 residents' recipe, per-tensor sha256 and per-group error |
| `mtp-experts-nvfp4.bin` | 4,078,043,136 | `74cb500dc6c5...d7c973` | the MTP draft bank: 288 experts, NVFP4 |
| `mtp-expert-manifest-nvfp4.json` | 118,615 | `a0bb0bc4758d...2d2ab5` | the draft bank's manifest |
| `expert-usage-v1.json` | 53,207 | `d3bcc805a3e9...869720` | the usage profile that seeds the prompt rows at load |
| `config.json` | 211,018 | `da91aacd79a3...6254d8` | the release config plus the `expert_streaming` block the plugin reads (`quant: exl3-k3-mul1`) and `residents_format: fp8_block128` |
| `hf_quant_config.json` | 7,371 | `342b0dcc8028...1e16b4` | from the NVFP4 source pack (cosmetic, see above) |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`, `processor_config.json`, `LICENSE` | | | the release's |

Full sha256 of every file: [`manifest.sha256`](manifest.sha256).

`experts.bin` must be a real file or a hard link on the serving machine (the stream reads it with uncached I/O).

## Serving

Requirements: Apple silicon with 128 GB unified memory, the pack on the internal SSD, mlx-serve with the GLM host
changes (ddalcu/mlx-serve#823 at `9a48a3c3` or later; the measured hosts were its commits `d01c0532` and `a8f6c6b8`) built with the mlx-stream plugin on its
`main` (`2595423` or later), and the wired limit raised:

```sh
sudo sysctl iogpu.wired_limit_mb=114688
```

The command the measurements used (the harness's host, port and timeout flags dropped):

```sh
MLX_SERVE_GPU_CEILING_MB=114688 mlx-serve serve --model /abs/path/to/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw \
  --ctx-size 32768 --no-vision --no-warmup-eager --prefix-cache-entries 0 --metrics --log-file off \
  --kv-quant off --wired-margin-gib 2
```

`--ctx-size` is the context the memory bill covers (prompt plus generation). The settings go in the pack's
`config.json` under `expert_streaming.settings`. The pack ships without a settings block. The measured headline
settings:

```json
"expert_streaming": { "settings": {
  "expert_prefill_rows": 199, "expert_rows": 225,
  "mtp_lane": true, "draft_depth": 3, "draft_gate": 0.5, "draft_head": "q8" } }
```

- Rows are resident expert rows per layer (prompt / decode), out of 288. More rows give fewer SSD reads and faster
  decode, but cost memory. Without a rows pin the plugin fills the rows from its memory bill at load and logs the fill
  (`glm5_next memory: ... prompt_rows= decode_rows=`).
- The pin must fit the machine's memory in use at load. With the q8 head, 190/217 fits up to about 12.0 GB of other
  physical use, 197/223 up to 9.4 GB, 199/225 up to 8.6 GB (on the exact head: 12.7, 10.9 at 195/221, 9.3 at 199/225).
  A typical desktop session is above 8.6 GB; then use 190/217. A pin over the fill loads with the line
  `proceeding (operator pin)` and can run the machine out of memory.
- `draft_depth` 3 and `draft_gate` 0.5 must be set by name: the code defaults are depth 2 and gate 0.75.
- `"draft_head": "q8"` is the headline (rounding-class) head. Leave it out for the exact head (BF16 draft head): then
  every token id equals the serial path's at equal rows.
- Requests use the MTP lane by default (`"enable_mtp": true`).

The server speaks the OpenAI-compatible API (`/v1/completions`, `/v1/chat/completions`). The plugin's docs
([glm5-next.md](https://github.com/davidtai/mlx-stream/blob/main/docs/glm5-next.md),
[benchmarks/glm53-flash](https://github.com/davidtai/mlx-stream/blob/main/docs/benchmarks/glm53-flash/README.md)) have
every setting, the memory bill, the lever record and the limitations.

## Measurements

One M5 Max, 128 GB, one internal SSD, fans at maximum, nothing else loaded, each cell a fresh server and one request.
The standard cell: a 16,381-token coding prompt (chat template, thinking on), `max_tokens` 1,024, KV bf16, the MTP lane
on (depth 3, gate 0.5), card sampling (temperature 1.0, top_p 0.95, seed 1). Prefill and decode are the server's own
per-request line.

| draft head | rows | prefill tok/s | decode tok/s | TTFT s | build |
|---|---|---:|---:|---:|---|
| q8 | 199 / 229 | 1,086.5 | **36.70** | 15.08 | g37, operator pin 2.97 GB over the fill, box peak 118.40 GB |
| q8 | 199 / 225 | 1,086.0 | 36.30 | | g26-dsa3 |
| q8 | 190 / 217 | 1,094.2 | 35.00 | 14.97 | g37 |
| q8 | 190 / 217 | 1,093.0 | 34.95 | | g26-dsa3 (2 arms) |
| bf16 (exact) | 199 / 225 | 1,077.1 | 34.50 | | g25-dsa3 |
| bf16 (exact) | 190 / 217 | 1,093.5 | 33.15 | | g25-dsa |

Builds (ReleaseFast, the real mlx-serve binary): g25-dsa3 = host `a8f6c6b8` + plugin `0457394`, server sha256 `6481fa1b...`;
g25-dsa = plugin `88d2aad`, `5bbd8512...`; g26-dsa3 = plugin `cf30601`, `de1ab1f6...`; g37 = host `d01c0532` + plugin
`14a95ba` (branch `glm5next/draft-chain`: two draft-lane changes, `fuse_draft_step` and `draft_chain: enqueued`, not yet
on plugin `main`), `4d096aa0...`, `resident_seed: none`. The 199-row cells are single cells run first in their window
(n=1), because the 199-row pins need a clean memory floor; the 190/217 cells are the B arms of ABBA pairs or single cells.

### Context sweep 1K-1M

One cold request per prompt length on the exact head (build g28-clean, host `a8f6c6b8` + plugin `6ac6dbf`, server sha256
`e20b4548...`), card sampling, thinking on, `max_tokens` 16,384, rows pinned per rung at the memory bill's fill for a
12.6 GB machine baseline. Every point is one request.

| prompt tokens | rows | prefill tok/s | decode tok/s | TTFT s | completion tokens | finish |
|---:|---|---:|---:|---:|---:|---|
| 1,023 | 191 / 217 | 122.0 | 33.0 | 8.39 | 11,410 | stop |
| 2,047 | 191 / 217 | 238.4 | 33.8 | 8.59 | 3,324 | stop |
| 4,095 | 191 / 217 | 462.2 | 33.3 | 8.86 | 10,278 | stop |
| 8,191 | 191 / 217 | 889.4 | 32.1 | 9.21 | 12,446 | stop |
| 16,381 | 190 / 217 | 1,091.5 | 32.8 | 15.01 | 6,263 | stop |
| 32,766 | 189 / 214 | 1,047.8 | 31.4 | 31.27 | 10,099 | stop |
| 65,534 | 188 / 213 | 996.2 | 28.0 | 65.78 | 16,384 | length |
| 131,071 | 185 / 210 | 930.6 | 27.0 | 140.85 | 11,373 | stop |
| 262,143 | 178 / 203 | 839.5 | 15.2 | 312.26 | 8,314 | stop |
| 524,287 | 161 / 190 | 704.7 | 9.8 | 743.99 | 2,397 | stop |
| 1,032,191 | 128 / 165 | 500.4 | 5.4 | 2,062.73 | 5,732 | stop |

![Full 1K-1M sweep, exact head: decode, prefill, TTFT, MLX allocator peak, process footprint peak and completion tokens by context](img/sweep/summary.png)

![Decode tokens/s by context, exact head](img/sweep/decode_tok_s.png)

![Prefill tokens/s by context, exact head](img/sweep/prefill_tok_s.png)

The other charts: [TTFT](img/sweep/ttft_s.png), [MLX allocator peak](img/sweep/peak_memory_gb.png),
[process footprint peak](img/sweep/peak_footprint_gb.png), [completion tokens](img/sweep/completion_tokens.png).
The MLX allocator peak stayed at 95.5-96.3 GiB and the process footprint peak at 96.2-97.1 GiB on every rung.

The same sweep with the q8 draft head (build g29-clean, plugin `1f3a3c0`, server sha256 `89297e60...`; rows one to two
under the exact head's, because the q8 head adds 0.68 GB to the bill):

| prompt tokens | rows | prefill tok/s | decode tok/s | TTFT s | completion tokens | finish |
|---:|---|---:|---:|---:|---:|---|
| 1,023 | 189 / 216 | 122.0 | 34.4 | 8.39 | 12,369 | stop |
| 2,047 | 189 / 216 | 238.2 | 36.6 | 8.59 | 5,164 | stop |
| 4,095 | 189 / 216 | 462.2 | 33.6 | 8.86 | 7,788 | stop |
| 8,191 | 189 / 215 | 888.6 | 32.1 | 9.22 | 12,796 | stop |
| 16,381 | 189 / 215 | 1,088.7 | 34.4 | 15.05 | 5,548 | stop |
| 32,766 | 188 / 213 | 1,045.8 | 31.9 | 31.33 | 6,297 | stop |
| 65,534 | 186 / 211 | 992.2 | 29.5 | 66.05 | 16,384 | length |
| 131,071 | 183 / 208 | 925.7 | 27.1 | 141.59 | 9,611 | stop |
| 262,143 | 176 / 201 | 835.7 | 14.6 | 313.68 | 10,075 | stop |
| 524,287 | 159 / 189 | 698.5 | 9.9 | 750.59 | 2,978 | stop |
| 1,032,191 | 126 / 164 | 494.4 | 5.1 | 2,087.76 | 7,054 | stop |

![Full 1K-1M sweep, q8 draft head](img/sweep-q8/summary.png)

![Decode tokens/s by context, q8 draft head](img/sweep-q8/decode_tok_s.png)

![Prefill tokens/s by context, q8 draft head](img/sweep-q8/prefill_tok_s.png)

The other charts: [TTFT](img/sweep-q8/ttft_s.png), [MLX allocator peak](img/sweep-q8/peak_memory_gb.png),
[process footprint peak](img/sweep-q8/peak_footprint_gb.png), [completion tokens](img/sweep-q8/completion_tokens.png).

### HumanEval

164 problems, one sample each, pass@1, rows 190/217, MTP lane on (depth 3, gate 0.5), temperature 1.0, top_p 0.95,
seed 1, thinking on, `max_tokens` 16,384, through `/v1/chat/completions`, scored with evalplus on the last code block
(build g29-clean).

| draft head | pass@1 | truncated | errors | decode median tok/s | failed |
|---|---:|---:|---:|---:|---|
| bf16 (exact) | 0.9817 (161 / 164) | 2 | 0 | 38.35 | 47, 116, 132 |
| q8 | 0.9817 (161 / 164) | 2 | 0 | 39.9 | 116, 132, 139 |

The truncated completions hit the 16,384-token cap.

### Warm layers

A short request's prompt phase is expert reads, not compute, so the state of the expert rows at the start of the
prompt sets the TTFT. Three states:

- **Cold, empty rows:** the first request after a load with `resident_seed: none`. The prompt reads every expert it
  routes to from the SSD.
- **Cold, seeded rows:** the first request after a load with `resident_seed: usage` (the default with this pack). At
  load the plugin fills the prompt rows from `expert-usage-v1.json` (190 of 190 rows per layer, 75.6 GB read in 5.6 s;
  the load takes about 6 s longer).
- **Warm, kept rows:** a later request with `prompt_residents: keep` (the default). The residents that decode left in
  the prompt rows stay for the next prompt.

A kept or seeded slot holds its own expert record's bytes: the state changes which experts the stream reads, not the
arithmetic. All outputs were identical across the arms.

![GLM-5.3-Flash TTFT in the three states](img/warm-layers/warm_ttft_three_states_glm.png)

12 sequential chat requests per arm (HumanEval prompts, thinking off, `max_tokens` 256), arms A B B A, rows 190/217, q8
head:

| pair | A | B | result |
|---|---|---|---|
| keep vs forget (build g30-keep, plugin `2e032cb`) | forget: warm median 6.59 s | keep: warm median 2.44 s | keep cuts the warm TTFT by 63 % |
| seed vs none (build g31-seed) | none: cold 6.95 s | seeded: cold 2.62 s | the seed cuts the cold TTFT by 62 % |

![TTFT per request, forget vs keep](img/warm-layers/warm_ttft_per_request_glm.png)

![TTFT per request, seed none vs seeded](img/warm-layers/warm_ttft_seed_glm.png)

![Warm TTFT on the three mlx-stream models](img/warm-layers/warm_ttft_summary.png)

## Limits

- Decode is 33-37 tok/s at 16K, not yet 40. The cost is the SSD reads of the expert misses and the decode arithmetic
  of the EXL3 kernels.
- The pack is lossy against the BF16 release: EXL3 3.05 bpw experts and FP8 residents. The q8 draft head is
  rounding-class (its ids diverge from the exact head's; it passed our three-fixture grader battery). The bf16 draft
  head is exact against the serial path.
- The 199-row pins need a quiet machine (about 8.6-9.3 GB of other memory in use at load). 199/229 ran 2.97 GB over the
  bill's fill as an operator pin and peaked at 118.40 GB on the machine; we do not recommend it for a desktop session.
- Text only. The vision tower is in the shards but the plugin does not load it (`--no-vision`).
- Decode falls with context (15.2 tok/s at 256K, 5.4 at 1M) because the rows shrink to give memory to the KV.
- Every measurement is one machine; sweep points and the 199-row cells are n=1.

## Provenance

- Experts: turboderp/GLM-5.3-Flash-exl3 at `332ab457b709b7ba30dd9a448be5de03b80a7ac9` (branch `3.05bpw`), 12,096
  records repacked (42 layers x 288), each byte-verified on read-back against the source tensors; codebook mul1 checked
  on every projection. `experts.bin` sha256 `345ac3f68ab33b5011c1e5bb992b8b51c1c871a22633d421ef1caa02e29580e4`.
- Residents and the MTP layer: vcruz305/GLM-5.3-Flash-NVFP4 at `b7d3cbf6a387259a187654cd1b7779918cb5ba40` (BF16
  non-expert tensors, base zai-org/GLM-5.3-Flash-BF16), FP8 by `scripts/quantize_glm5next_trunk_fp8.py`, the MTP bank by
  `scripts/quantize_glm5next_mtp_nvfp4.py` (the script hashes are in the manifests).
- Tokenizer, chat template and generation config: the release's (zai-org/GLM-5.3-Flash).
- Measurements: the plugin's
  [docs/benchmarks/glm53-flash](https://github.com/davidtai/mlx-stream/blob/main/docs/benchmarks/glm53-flash/README.md)
  and [docs/benchmarks/warm-layers](https://github.com/davidtai/mlx-stream/blob/main/docs/benchmarks/warm-layers/README.md)
  (plugin `main` `2595423`) hold every cell, build and chart source.

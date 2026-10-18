---
license: mit
license_link: LICENSE
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- zh
tags:
- cnq
- exl3
- mul1
- nvfp4
- 3-bit
- quantized
- moe
- glm5_next
- windows
- rtx-5090
- crow-nest
---

# GLM-5.3-Flash-MUL1K3

> [!Note]
> This repository holds a quantization of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) in the CNQ container format of the [crow-nest](https://github.com/nibor1896/crow-nest) engine: routed experts at 3.05 bpw in exllamav3's EXL3 format (MUL1 codebook, K=3), the dense part in NVFP4. It is not the upstream checkpoint and it is not an EXL3 model directory: it does not load in Transformers, vLLM, SGLang, llama.cpp or ExLlamaV3, only in crow-nest. (Card of 2026-10-11.)
>
> Only files made by crow-nest are in this repository. The tokenizer, chat template and original weights stay at the upstream repository and are linked below, not copied. For what the model can do, read the [upstream card](https://huggingface.co/zai-org/GLM-5.3-Flash); every number on this page is about this quant and names its date and conditions.

## About this quant

| item | value |
|---|---|
| base model | `zai-org/GLM-5.3-Flash`, revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (upstream `lastModified` 2026-09-07T12:13:47Z, still `main` on 2026-10-11); the FP8 originals were the conversion input |
| model | `glm5_next` (`Glm5NextForConditionalGeneration`), a sparse MoE with 320B total and 18B active parameters per the upstream card (read 2026-10-11); 45 decoder layers, the first 3 dense, layers 3 to 44 with 288 routed experts (8 per token) and a shared expert, plus one MTP block as layer 45 (config in the container index, read 2026-10-11) |
| routed experts | exllamav3 EXL3, MUL1 codebook, K=3, 3.05 bpw, quantized from the FP8 originals by exllamav3's `quantize_exl3` with Hessians from 131,072 calibration tokens (see Quantization); 2026-10-11 |
| dense part | crow-nest recipe `cnq4.5-glm5-next`: NVFP4 (4.5 bpw, `--scales mse`) on the attention projections, the dense FFN of layers 0 to 2 and the shared experts; BF16 and F32 keeps elsewhere (see What differs); 2026-10-11 |
| container | one file, `GLM-5.3-Flash-MUL1K3.cnq`, 124,591,634,567 B (116.0 GiB), 38,398 tensors, index v2; converted 2026-10-09 |
| modality | text only: the container has no vision section (index read 2026-10-11) |
| engine | crow-nest `https://github.com/nibor1896/crow-nest`, the `serve` binary from v0.12.0 or later: own HTTP server with an OpenAI-style `/v1/chat/completions`, Rust + CUDA |
| platform | Windows 11, CUDA, one NVIDIA Blackwell GPU (`sm_120`); every number on this card comes from one RTX 5090 (32 GB) on Windows 11; 2026-10-11 |
| intended use | local inference on one 32 GB GPU with the experts tiered over VRAM, pinned host RAM and NVMe, as the engine behind the [Crow](https://github.com/nibor1896/Crow) agent; no hosted endpoint is offered; 2026-10-11 |
| licence | MIT, the upstream licence, shipped verbatim as `LICENSE`; crow-nest itself is Apache-2.0 |

## Files

### Files in this repository

| file | size | `sha256` | purpose |
|---|---|---|---|
| `GLM-5.3-Flash-MUL1K3.cnq` | 124,591,634,567 B | `9ce11456c68f80926e1fe8289888a3571309a1f8558bc73d1578f0db2b30213e` | the container (dense NVFP4 part + 12,384 MUL1 expert records); 2026-10-11 |
| `GLM-5.3-Flash-MUL1K3.cnq.sidecar.jsonl` | 9,738,525 B | `3acecc72fa383d2e0070a65c78b753e67cf2f561c3184ba71ad85144eb93ed31` | verification sidecar written by the converter in the same run: one line per tensor (38,398), 98 `code_summary` lines and one `section_summary` line; 2026-10-11 |
| `GLM-5.3-Flash-MTP-overlay.cnq` | 164,771,937 B | `191d5e182f91aae053e544eb91080990fcf956f103eaecc398e8beb326da962a` | optional: the 25 non-expert tensors of the MTP block (layer 45) for speculative decoding with `CROW_GLM_MTP`; not used by the operating set below; 2026-10-11 |
| `GLM-5.3-Flash-MTP-overlay.cnq.sidecar.jsonl` | 13,458 B | `a9a06bc3e203ac0bab19778762930ab5f8f5bfdfa060aa356b09bb7de5290986` | the overlay's sidecar |
| `glm53-cal4-decode-counts.json` | 54,279 B | `5584c8a03d665798d2b98b573ad09b05e5acf04cf80d0aff071308620c0fb65a` | expert warm-start scores for `CROW_GLM_ARENA_WARM` (which experts start in VRAM and pinned RAM); the same file as `runs/glm53-flash/arena-warm/` in the crow-nest repository |
| `SHA256SUMS` | 498 B | n/a, it carries the sums | the five sums above |
| `LICENSE` | 1,070 B | `30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1` | MIT License of `zai-org/GLM-5.3-Flash`, byte-identical to the upstream file at the revision above (compared 2026-10-11) |
| `NOTICE` | see the file | n/a | attribution: model, expert format, quantization method, engine |

- Sums computed 2026-10-11 from the files the numbers on this card were measured with; the container sum equals the one recorded at conversion on 2026-10-09 (crow-nest CHANGELOG, #182).
- The sidecar belongs to this container: its 38,398 tensor lines match the container index in name, dtype and section, 38,398 of 38,398 (checked 2026-10-11), and both files carry the same write time.
- Verify in the download directory with `sha256sum -c SHA256SUMS` (5 of 5 OK; the container line reads 116 GiB). `grep -v MUL1K3.cnq$ SHA256SUMS | sha256sum -c -` checks the small files alone. (2026-10-11)

### Files from other repositories, needed at run time

| file | from | size | `sha256` | why |
|---|---|---|---|---|
| `tokenizer.json` | `zai-org/GLM-5.3-Flash` @ `eb9eb208` | 20,217,442 B | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` | the tokenizer `serve` loads (equals the Hub's LFS record, checked 2026-10-11) |
| `tokenizer_config.json` | `zai-org/GLM-5.3-Flash` @ `eb9eb208` | 761 B | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` | tokenizer settings; it has no `chat_template` field |
| `chat_template.jinja` | `zai-org/GLM-5.3-Flash` @ `eb9eb208` | 10,950 B | `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5` | the chat template `serve` renders with (read beside `tokenizer_config.json`) |

- `config.json` and `generation_config.json` are NOT needed: the container's index v2 carries both verbatim in its `model` block, and crow-nest reads them from there.
- crow-nest looks for these files in `models/GLM-5.3-Flash-original/` of the crow-nest repository root, or wherever `CROW_TOKENIZER` points (`CROW_TOKENIZER_CONFIG` defaults to the sibling `tokenizer_config.json`).

## What differs from the original checkpoint

| part | upstream `zai-org/GLM-5.3-Flash` | this container |
|---|---|---|
| routed experts, layers 3 to 44 (gate, up, down) | FP8 | EXL3 MUL1, K=3, 3.05 bpw: 36,288 tensors in 12,096 records of 9,474,048 B (counted 2026-10-11) |
| routed experts of the MTP block (layer 45) | FP8 | EXL3 MUL1, K=3, section `mtp`, optional to load: 864 tensors in 288 records; quantized with an identity Hessian (uncalibrated LDLQ), because the reference runner has no MTP forward (2026-10-09) |
| attention projections (KDA `q/k/v/o_proj` and short convs, MLA `q_a/q_b/kv_a/kv_b/o_proj`), dense FFN of layers 0 to 2, shared experts | as in the checkpoint | NVFP4, 4.5 bpw, `--scales mse`, recipe `cnq4.5-glm5-next`: 428 tensors (counted 2026-10-11) |
| token embedding, `lm_head`, norms, routers, DSA indexer, hyper-connection and other small tensors | as in the checkpoint | BF16 (528 tensors) and F32 (290 tensors), unchanged values (counted 2026-10-11) |
| the MTP block's 25 non-expert tensors | as in the checkpoint | not in the container; in `GLM-5.3-Flash-MTP-overlay.cnq` (8 NVFP4, 16 BF16, 1 F32), written 2026-10-09 |
| vision tower | in the checkpoint | omitted by the recipe; this quant is text only |
| total | 62 safetensors shards | 124,591,634,567 B in one file (+ 164,771,937 B optional overlay); 2026-10-11 |

## Quantization

**Routed experts: exllamav3 EXL3, MUL1 codebook, K=3.** The experts are quantized with [exllamav3](https://github.com/turboderp-org/exllamav3)'s `quantize_exl3` (exllamav3 1.6.0, torch 2.13.0+cu132), using its EXL3 format with the MUL1 codebook at K=3 (3.05 bpw with the scale vectors). EXL3's trellis method comes from QTIP (Tseng et al. 2024, [arXiv:2406.11235](https://arxiv.org/abs/2406.11235)). MUL1 is exllamav3's format, not crow-nest's: crow-nest's converter packs exllamav3's tensors into CNQ records unchanged and its decoder reproduces exllamav3's weights byte for byte (crow-nest #181, #156, 2026-10-09).

- Per expert, from the FP8 originals: gate and up share one Hessian `X^T X` over all calibration rows of the layer (exllamav3 calibrates every expert on every token for this architecture); down uses `A^T A` with `A` the expert's activation; seed per expert and projection; `sigma_reg` 0.025 (crow-nest `docs/glm-mul1-conversion.md`, 2026-10-09).
- Calibration: 131,072 tokens, four files of 32,768 tokens from the maintainer's own agent sessions and code work; the held-out evaluation session was refused by the tool. The calibration text is not published; its file names and sha256 values are in the container's `expert_codec` block (written 2026-10-09).
- The inputs to each layer come from the FP8 trunk, not from already-quantized layers as in exllamav3's own converter. These are crow-nest's own quants, not the files of `turboderp/GLM-5.3-Flash-exl3`.

**Dense part: crow-nest NVFP4.** E2M1 values, one `ue4m3` scale per 16 values, one `f32` global scale per tensor (blocks of 64 values in 36 B), sub-block scales by `--scales mse`. The dense bytes equal those of crow-nest's 4.5-bit GLM container of the same recipe (crow-nest PREREG, 2026-10-09).

**Container.** CNQ1 (magic `CNQ1`, streamed payload, JSON index as a trailer, `u64` little-endian index length in the last 8 bytes), index `format_version` 2, `recipe` `cnq4.5-glm5-next`, `scales` `mse`, section `mtp` optional to load; the index trailer is 14,698,623 B, sha256 `56741348e106d9e7b0a911afee846b8ca2b1152156fd78b6e443b6f3c0061612` (read 2026-10-11). Each expert record is 9,474,048 B on a 4096-B offset: three trellises of 3,145,728 B, then the six scale vectors. Layout of record: crow-nest [`converter/README.md`](https://github.com/nibor1896/crow-nest/blob/main/converter/README.md) and [`docs/glm-mul1-conversion.md`](https://github.com/nibor1896/crow-nest/blob/main/docs/glm-mul1-conversion.md).

## Measured

Machine for every row: one RTX 5090 (32 GB), Windows 11, a 64 GB-class host, the container on the system NVMe. Unless a row says otherwise: crow-nest build `2a18fbc` (v0.12 line), operating set `runs/glm53-flash/quick/arm4.env`, `CROW_CONTEXT` 200,000, MTP off. (2026-10-11)

**Speed** (2026-10-11)

| what | result | conditions | date |
|---|---|---|---|
| decode inside a real Crow agent task | **14.17 tok/s** | Crow window, task t3: 3 rounds, 2,122 generated tokens; prompt side 8,910 tokens at 460.97 tok/s including the cached rounds; N=1 (crow-nest #215) | 2026-10-11 |
| decode through `serve`, short prompt | **18.3 tok/s** | 57-token prompt, sampled with GLM's sampling row | 2026-10-11 |
| cold prefill through `serve` | about 870 to 1,100 tok/s | prompts of 6k to 8k tokens, cold start | 2026-10-11 |

- These are single runs on one machine; the experts that do not fit in VRAM are read from pinned RAM and NVMe, so speed depends on the SSD and on how often the routing leaves the warm set. No other machine was measured.

**Correctness of the engine on these weights** (G3 of the crow-nest PREREG, `decode glmgolden`)

| check | result | date |
|---|---|---|
| engine against the layerwise HF reference on the container-dequantized weights: every layer output and the head, 549 rows | **549 of 549 pass**; threshold cosine ≥ 0.9999 per layer per anchor; layer-output 1-cos at most 1.99e-8, logits cosine 1.000000000 (build `298092b`, two runs, `runs/glm53-flash/g3g4-20261011/`) | 2026-10-11 |
| the same gate on the first engine build of this container | 549 of 549 pass | 2026-10-09 |

- G3 checks that the engine computes this quant correctly. It does not measure the quantization error; that is the next table.

**Quantization error against the FP8 originals** (reported, not gated)

| check | result | date |
|---|---|---|
| 18 decoded expert projections against their FP8 originals | cosine 0.987 to 0.991 | 2026-10-09 |
| layer outputs 0 to 3, container against FP8, same prompt | cosine 0.99308 / 0.99402 / 0.99806 / 0.99777; layer-3 routing overlap 0.910, DSA selection 90 of 90 | 2026-10-09 |
| MTP overlay: block 45 + head against the FP8 golden, 89 rows | `eh` cosine 1.000000, head-norm cosine mean 0.98395, draft top-1 equal to the golden draft on 79 of 89 | 2026-10-09 |

## Hardware requirements

| item | value | basis |
|---|---|---|
| GPU | one NVIDIA Blackwell GPU, `sm_120`, 32 GB; measured only on an RTX 5090. A smaller card was not tried | measured setup, 2026-10-11 |
| host RAM | 64 GB class (the measured host shows 63.38 GiB); part of the experts lives in pinned host RAM | crow-nest `docs/glm-mul1-conversion.md`, 2026-10-09 |
| disk | 116 GiB for the container on a fast NVMe SSD (experts outside VRAM and RAM are read from it per token), plus 0.2 GB for the overlay and 20 MB of tokenizer files | this card, 2026-10-11 |
| OS | Windows 11. The GLM path was not measured on Linux | measured setup, 2026-10-11 |
| CUDA | NVRTC 13.3 on `PATH` (CUDA 13.3 toolkit, or crow-nest's `fetch-nvrtc.ps1` from NVIDIA's own wheel) | crow-nest README, v0.12.0; 2026-10-11 |
| build | a Rust toolchain for `cargo build --release --bin serve` in `engine/`; Git Bash for the commands below | crow-nest README |

## How to run

GLM-5.3-Flash runs in crow-nest behind the operating set `arm4.env`; every GLM switch is off by default in the code, so the set must be passed. Commands for Git Bash on Windows:

```bash
git clone https://github.com/nibor1896/crow-nest && cd crow-nest
(cd engine && cargo build --release --bin serve)

# this repository (116 GiB): container, sidecars, overlay, warm-start scores, SHA256SUMS, LICENSE
hf download nibor1896/GLM-5.3-Flash-MUL1K3 --local-dir cnq-glm
(cd cnq-glm && sha256sum -c SHA256SUMS)

# tokenizer and chat template from the upstream model (not mirrored here)
hf download zai-org/GLM-5.3-Flash tokenizer.json tokenizer_config.json chat_template.jinja \
  --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --local-dir models/GLM-5.3-Flash-original

env $(cat runs/glm53-flash/quick/arm4.env) \
  CROW_GLM_ARENA_WARM=cnq-glm/glm53-cal4-decode-counts.json \
  CROW_CNQ=cnq-glm/GLM-5.3-Flash-MUL1K3.cnq \
  CROW_TOKENIZER=models/GLM-5.3-Flash-original/tokenizer.json \
  CROW_CONTEXT=200000 \
  engine/target/release/serve.exe --port 8099
```

Use it from Crow:

```bash
crow --base-url http://127.0.0.1:8099/v1
```

- `CROW_GLM_ARENA_WARM` comes after `arm4.env` on purpose: the set names a path on the maintainer's machine, and the later assignment wins.
- The numbers above were measured with this environment and the container in the repository's `converter/` directory; the block as a whole (download into `cnq-glm/`, then serve) was not run end to end (2026-10-11).
- `serve` binds `127.0.0.1:8099`, takes the OpenAI chat format (`POST /v1/chat/completions`, streamed or not) and allows one engine per machine.
- Optional MTP: put `GLM-5.3-Flash-MTP-overlay.cnq` where `CROW_GLM_MTP_OVERLAY` points (default `converter/GLM-5.3-Flash-MTP-overlay.cnq`) and set `CROW_GLM_MTP` to the drafts per step. Its speed is not part of this card.
- Every switch is documented in crow-nest [`docs/env.md`](https://github.com/nibor1896/crow-nest/blob/main/docs/env.md); the operating set and its measurements in [`runs/glm53-flash/quick/README.md`](https://github.com/nibor1896/crow-nest/blob/main/runs/glm53-flash/quick/README.md). (2026-10-11)
- Thinking budget and chat behaviour follow the upstream template (`reasoning_effort`, `clear_thinking`); see the upstream card's notes.

## Converter command of record

```
python -I tools/glm_mul1_quantize.py capture  --fp8 models/GLM-5.3-Flash-original --work decode_out/glm-mul1/work
python -I tools/glm_mul1_quantize.py quantize --fp8 models/GLM-5.3-Flash-original --work decode_out/glm-mul1/work --store decode_out/glm-mul1/store --wait --prune-consumed converter/GLM-5.3-Flash-MUL1K3.cnq
converter --scales mse --source-repo zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --experts-mul1 decode_out/glm-mul1/store --mul1-wait models/GLM-5.3-Flash-original converter/GLM-5.3-Flash-MUL1K3.cnq
converter --scales mse --source-repo zai-org/GLM-5.3-Flash --mtp-overlay models/GLM-5.3-Flash-original converter/GLM-5.3-Flash-MTP-overlay.cnq
```

- The three stages run side by side (capture in crow-nest's oracle venv, quantize in the exllamav3 venv); the commands are those of crow-nest `docs/glm-mul1-conversion.md`. Result of the container run, 2026-10-09: 38,398 tensors (428 NVFP4, 818 BF16/F32 keeps, 37,152 MUL1 in 12,384 records), 16,885 s on the CPU side, Windows 11.
- Byte reproduction needs the unpublished calibration files; with your own calibration text the same commands give a container of the same layout and size whose quality was not measured.

## Limitations and not measured

| item | status |
|---|---|
| downstream benchmarks (the upstream tables) on this quant | not measured |
| KLD or perplexity against the FP8 or BF16 original | not measured |
| agentic quality beyond the single Crow task above | not measured |
| speed with MTP (`CROW_GLM_MTP`) | not measured for this card |
| Linux, other GPUs, cards under 32 GB, hosts without a fast NVMe | not measured |
| contexts above 200,000 tokens | not measured; 2026-10-11 |
| vision | not available: the tower is not in the container |
| MTP experts | uncalibrated (identity Hessian) |

## Provenance

| item | value |
|---|---|
| base model | `zai-org/GLM-5.3-Flash` on Hugging Face, public, not gated (API, 2026-10-11) |
| revision of record | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, upstream `lastModified` 2026-09-07T12:13:47Z; the Hub's `main` on 2026-10-11 |
| input | the 62 FP8 safetensors shards, each verified against the Hub's LFS sha256 before the conversion (crow-nest `tools/fetch-glm.py`, 2026-10-08) |
| conversion | crow-nest converter with `--experts-mul1`, exllamav3 1.6.0 `quantize_exl3`, 2026-10-09, Windows 11, RTX 5090 machine |
| acceptance | G3 549 of 549 (2026-10-09, again 2026-10-11 on build `298092b`); speed rows on build `2a18fbc` (2026-10-11) |

## License

| item | value |
|---|---|
| licence | MIT License, the licence of `zai-org/GLM-5.3-Flash` (card front matter `license: mit` and `LICENSE` at revision `eb9eb208`, read from the Hub 2026-10-11) |
| text | `LICENSE` next to this card: 1,070 B, 21 lines, sha256 `30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1`, byte-identical to the upstream file (compared 2026-10-11); it carries "Copyright (c) 2026 Z.AI Co., Ltd" |
| duty, every copy | keep the copyright notice and the permission notice in all copies or substantial portions; `LICENSE` does that for this repository |
| modification | this container is a modified form of the upstream weights: routed experts re-quantized from FP8 to EXL3 MUL1 3-bit, the dense projections to NVFP4, the vision tower omitted, by crow-nest on 2026-10-09 |
| expert format | exllamav3 (EXL3, MUL1 codebook) by turboderp, MIT; its notice ships in crow-nest's `NOTICE`, and `NOTICE` here names it |
| engine | crow-nest (engine and converter) is Apache-2.0, see its `LICENSE` |

## Citation

- The model: cite the GLM-5 technical report, [arXiv:2602.15763](https://arxiv.org/abs/2602.15763); the BibTeX entry is on the [upstream card](https://huggingface.co/zai-org/GLM-5.3-Flash#citation). Links checked 2026-10-11.
- The expert quantization: QTIP, Albert Tseng et al., 2024, [arXiv:2406.11235](https://arxiv.org/abs/2406.11235), as implemented in [exllamav3](https://github.com/turboderp-org/exllamav3). Links checked 2026-10-11.
- The engine: [crow-nest](https://github.com/nibor1896/crow-nest). Links checked 2026-10-11.

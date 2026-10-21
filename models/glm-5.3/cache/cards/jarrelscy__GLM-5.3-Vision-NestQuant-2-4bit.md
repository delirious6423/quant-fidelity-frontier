---
license: mit
base_model: zai-org/GLM-5.3
pipeline_tag: image-text-to-text
tags:
- vision
- multimodal
- image-text-to-text
- quantization
- moe
- glm
---

# GLM-5.3 Vision NestQuant 2-4 bit

GLM-5.3 with its routed experts quantized to NestQuant, a nested 2/4-bit format. Every expert has a 2-bit base and an optional 4-bit residual plane. At serving time an expert can be switched from 2 bit to 4 bit by loading its residual plane on top of the base. The base bytes do not change.

**Status: encoding complete (all 75 expert layers, 19,200 experts), served live.** This repo cannot be loaded with stock vLLM or transformers. It is served by the NestQuant streaming server ([github.com/jarrelscy/nestquant](https://github.com/jarrelscy/nestquant), `./start.sh`); see Live serving below.

## Contents

| Part | Format | Size |
|---|---|---|
| Routed experts, layers 3-77 | NestQuant: 2-bit base (2.014 bpw) + 4-bit residual (4.126 bpw total), plus a small low-rank correction on some experts | ~5 GB per layer |
| Attention, dense MLP (layers 0-2), shared experts, router, norms, embeddings, lm_head | FP8 / bf16, copied byte for byte from zai-org/GLM-5.3 | 21 GB |
| MTP layer 78 | FP8, copied byte for byte | 10 GB |
| Vision tower + projector | copied from jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid | 0.9 GB |
| `base/`: serving base checkpoint (everything except the routed experts of layers 3-77, in the NVFP4/ARVQ hybrid serving format, incl. MTP layer 78, vision, tokenizer, config) | NVFP4 / FP8 / bf16, from jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid (revision d4a105dc50dd), byte for byte | 44 GB |

Files:
- `layers/L{L}/tp{0..7}.safetensors`: expert planes for layer L, split for tensor parallel 8, with `layers/L{L}/manifest.json` describing the layout and the default 4-bit set.
- `nonexpert-*.safetensors` and `model.safetensors.index.json`: everything that is not a routed expert, plus the vision weights.
- `config.json`: vision-language config (text config in `text_config`). `config.text.json` is the text-only config.
- `base/`: what the streaming server loads besides the NestQuant records: the 14 non-expert shards of the ARVQ hybrid serving checkpoint (attention, shared experts, dense layers 0-2, embeddings, lm_head, MTP layer 78, vision tower, projector) with a filtered `model.safetensors.index.json`, plus `config.json`, tokenizer, chat template and the vision processor code. The root `nonexpert-*` files are the FP8 reference non-experts used for evaluation; the server does not read them.

## Live serving

The server keeps every expert's 2-bit base in VRAM and streams the 4-bit residual planes from NVMe for the experts the jF predictor (see Floating-set predictor) expects routing to use next: 77 floating experts per layer in 80 slots, no fixed set. During prefill it borrows free KV pages to hold 155 slots per layer.

Measured 2026-10-03 (nestquant commit 8c8f9d5), 4x RTX PRO 6000 Blackwell (SM120, 96 GB each), TP4 + DCP4, MTP ns=3, records read from two NVMe drives, concurrency 1, 1M context (KV pool 1,083,392 tokens at `NQ_UTIL=0.925`):

| | |
|---|---|
| Decode, empty context | 83.0 tok/s (32.6 steps/s, 2.60 accepted/step; median of 3) |
| Decode, 16K context | 78.8 tok/s (31.6 steps/s) |
| Prefill | ~1900-2000 tok/s (layer-major prefill off; see below) |
| KLD vs BF16 teacher | 0.0256 mean (windows 0-3: 0.0147 / 0.0612 / 0.0136 / 0.0130) |
| Routed expert calls at 4 bit, decode | 62.7% (held-out generations); 61.8% on the KLD windows |
| Needles | retrieved at 43K and 947K |

Decode runs range 77-89 tok/s with MTP acceptance; steps/s stays at about 31-33. The KLD is full-vocabulary and teacher-forced on the running server, on the same 4 BF16-teacher windows as below, so it includes the server's real read timing, prefill path and fp8 KV cache.

### Layer-major prefill

On by default since 2026-10-04 (`NQ_LMPF=1`, nestquant main; `NQ_LMPF=0` turns it off).
- Prompts with at least 32K new (uncached) tokens run one layer at a time over ~61K-token windows while all 256 experts of the next layer load at 4 bit, so the whole prompt is prefilled at 4 bit.
- Prompts with 1K-32K new tokens get a 2 s read budget (`NQ_LMPF_BUDGET_S`) spent on the experts the router uses most.
- Shorter prompts prefill as before.
- Each window runs through vLLM's compiled graph pieces (`NQ_LMPF_CG=1`).
- The read ring and the window's activations live in idle decode expert slots during the prefill (`NQ_LMPF_BORROW=all`). Lending ~3,000 slots takes ~37 ms, and decode reloads them on demand afterwards. Decode KLD measured right after a 64K prefill is unchanged: 0.0254 → 0.0254.

Measured 2026-10-04 on main b03530d and its parent commits. Prefill KLD is on the prompt tokens of the four windows above (2K each, BF16 teacher). TTFT is from a clean boot with on/off in the same boot, 2 reps, at a 64K window before slot borrowing (now ~61K):

| Prompt | Mode | Prefill KLD | TTFT off | TTFT on |
|---|---|---|---|---|
| 2K | 2 s budget | 0.0468 → 0.0234 | | |
| 4K | 2 s budget | | 3.30 s | 3.91 s |
| 16K | 2 s budget | | 8.03 s | 9.44 s |
| 64K | all 4 bit | | 33.1 s | 33.3 s |
| 128K | all 4 bit | | 67.6 s | 68.0 s |

All-4-bit prefill on the 2K windows gives 0.0142. With it on, needles were found at 129K, 172K and 904K. nvidia-smi peak is 93.1 GB per GPU.

### Running Terminal-Bench on 4x RTX PRO 6000 Blackwell

**Required: `max_num_seqs=1` on the server and one concurrent Harbor trial.** The [launcher update `a6be6f6`](https://github.com/jarrelscy/nestquant/commit/a6be6f6a9b89043c3eae299eeeda73e26de0f1c6) enforces `MAX_NUM_SEQS=1` in Compose and rejects conflicting `NQ_MAX_NUM_SEQS` or `MAX_NUM_SEQS` overrides. Concurrent clients must queue; this is not a multi-sequence serving preset.

Updated 2026-10-10. This is the **96 GB Blackwell / SM120** setup used for the latest full GLM-5.3 Terminal-Bench attempts, not the Flash/Spark configuration or the older RTX 6000 Ada. Use **26 fixed + 51 floating 4-bit experts per layer** with jF. All other experts retain their resident 2-bit bases. `NQ_SLOTS_PER_LAYER=56` means 51 active floating slots plus 5 staging/spare slots; the 26 fixed experts are separate.

`./start.sh` now selects this allocation by default, with **TAP, compiled layer-major prefill, slot borrowing, prefill KV offload, asynchronous decode loading, coalesced follower reads and CPU LMCache enabled**. Other environment overrides are respected; use `./start.sh config` to check them.

Prerequisites: Docker with NVIDIA Container Toolkit, Python 3 with venv/pip, four available 96 GB SM120 GPUs, approximately 251 GB host RAM as on the reference machine, and fast NVMe storage. Allow about 437 GB for the model and primary records, plus caches/results and space for an optional second record copy. LMCache alone allows 18 GB per rank (72 GB total); runtime and KV offload need additional RAM. Keep more than 60 GiB available during long runs on the reference-size host.

```bash
git clone https://github.com/jarrelscy/nestquant
cd nestquant
# Choose writable paths on your two physical NVMe drives; reuse these exports on restart.
export NQ_STATE=/mnt/nvme0/nestquant
export NQ_REPACK_ALT_DIR=/mnt/nvme1/nestquant-records
# Set VLLM_API_KEY privately in the environment or the gitignored ./.env.
./start.sh
./start.sh smoke
```

The launcher downloads the model and predictor, installs an isolated HF CLI if needed, prepares the second-drive record copy automatically, builds the kernels and waits for the API. To use one drive, omit `NQ_REPACK_ALT_DIR`; this is supported but differs from the measured dual-NVMe throughput setup. Without `NQ_STATE`, files go under `$HOME/.local/share/nestquant`, so make sure that filesystem has enough space. No manual repacking is required.

Launcher update: **[c3a0f89](https://github.com/jarrelscy/nestquant/commit/c3a0f898a4cb46395eeb6772275e6ff16de8fd33)**. For an existing clone, update to this commit or a descendant before launching. The older launcher defaulted to zero fixed experts. To reproduce that older allocation explicitly, use `NQ_JOINT_FIXED=0 NQ_SLOTS_PER_LAYER=80 ./start.sh`.

`./start.sh config` prints a secret-safe summary without starting GPU work. Expected defaults include `NQ_PREDICTOR=joint`, `NQ_JOINT_FIXED=1`, `NQ_SLOTS_PER_LAYER=56`, `NQ_SCHED=tap`, `NQ_LMPF=1`, `NQ_LMPF_CG=1`, `NQ_LMPF_BORROW=all`, `NQ_PREFILL_KV_OFFLOAD=1` and `ENABLE_LMCACHE=1`. The serving preset is **TP4 + DCP4, MTP3, FP8 KV, max context 1,048,576, one request slot**. Check the startup summary/logs and smoke output before running Harbor; monitor available host RAM, streaming backlog/read errors, hot-expert coverage, MTP acceptance and decode throughput during long runs.

Use the [Harbor configuration and rerun instructions](benchmarks/tb4.0/README.md#rerunning-on-4x-rtx-pro-6000): stock Terminus-2, concurrency 1, temperature 1.0, top_p 0.95, 65,536 output tokens per call and an 8-hour agent timeout. Previous reasoning is excluded from subsequent requests (`clear_thinking=true`); **reasoning generation remains enabled**. This differs from the official GLM-5.3 Claude Code harness.

The 77-floating performance/KLD figures above describe the older zero-fixed allocation, not a new measurement of the current default. See the reasoning-layout and registry-panel results below for their respective configurations.

## Method

- Rotation: random signs + Hadamard-128 on both sides of each weight matrix.
- Layers 3-6, down projection: the input side (the SwiGLU output) uses random signs + one Hadamard-512 per tensor-parallel-4 shard instead of Hadamard-128 blocks (2048 = 4 x 512, so each TP4 rank holds exactly one block). These layers have low-rank input statistics, and Hadamard-128 kept their energy inside 128-wide blocks. They were also re-encoded with a flat input-channel scale (`ics_down: flat`, the RMS of the per-row scales). This changes the rotation only: bit rate and byte layout are unchanged, gate/up are byte-identical to before, and `layers/L{3..6}/manifest.json` (`config.in_had_down: 512`, `had_sign_seed`, `ics_down`) and `serving/tp4/manifest.json` (`in_had_down`, per layer; absent = 128) record it. A decoder or kernel must apply the 512-wide rotation for these four layers.
- 2-bit base: bitshift trellis code (K=2, L=16) with per-tile sign, fitted with LDLQ against a blend of the 2-bit and 4-bit targets.
- 4-bit residual: a second trellis code on the rotated residual (2 bits per weight on gate/up, 2.3125 on down), fitted jointly with the base.
- Low-rank correction: experts whose input has a few very large activation channels get a rank 1-4 fp16 correction (about +0.014 bpw on average).
- Calibration: 15.4M tokens of text plus 1,200 images (radiology, web screenshots, natural images, OCR). Per-expert Hessians blend 75% text and 25% vision.

## Quality

Relative output error of individual experts against FP8, compared with EXL3 on the same calibration data (150 experts, two per layer across layers 3-77: one from the default 4-bit set and one other; negative is better):

| Level | Mean vs EXL3 | Worst expert |
|---|---|---|
| 4 bit | -6.1% | -1.3% |
| 2 bit | +0.2% | +9.4% |

Layers 3-6 were refitted with the Hadamard-512 down rotation above. Their spot experts went from +9.4% to -13.3% vs EXL3 at 2 bit, and from -3.3% to -22.0% at 4 bit. Over all 1,024 experts of layers 3-6, the refit's error against FP8 fell by 20% on average at both levels (from 10% in layer 6 to 29% in layer 3), and no expert got worse. Layers 7-77 are unchanged: +1.0% at 2 bit and -5.2% at 4 bit.

### Recommended for reasoning: 26 fixed + 51 floating

For reasoning and long agent runs, pin the 26 default 4-bit experts per layer (see below) and let the jF predictor float 51 more:

```
NQ_JOINT_FIXED=1 NQ_SLOTS_PER_LAYER=56 ./start.sh
```

This uses about the same memory as the default 77 floating (82 vs 80 records per layer, ~30k tokens less KV).

| layout | decode KLD vs BF16, 4 windows | tb4 layout-config-recreation2 |
|---|---|---|
| default (0 fixed, 77 floating) | .0254 | fail (empty reasoning past ~100k) |
| `NQ_JOINT_FIXED=1` (26 fixed, 51 floating) | .0271 | pass, 98 turns / 189k |

Per-token KLD is slightly worse (mostly the legal window), but reasoning holds up deep into long agent runs. The tb4 result is one task.

Decode KLD vs BF16 per window (forced decode on the 4 confirmation windows of brandonmusic/GLM-5.3-BF16-full-logits, same stack, same boot settings):

| window | default (0 fixed / 77 floating) | 26 fixed / 51 floating |
|---|---|---|
| w0 | .0154 (repeat .0149) | .0155 (repeat .0162) |
| w1 (legal) | .0579 | .0650 |
| w2 | .0128 | .0136 |
| w3 | .0155 | .0142 |
| mean | .0254 | .0271 |

The w0 repeats show the run-to-run spread (~.001). Apart from w1, the two layouts are within it.

The 26 are ranked. `fixed_set_ranked.json` lists all 256 routed experts per layer, best first, by the same score used to pick them; the first 26 are the default fixed set. Score mass held by the top K experts (mean over layers):

| K | 4 | 8 | 12 | 16 | 20 | 26 | 40 | 51 | 77 |
|---|---|---|---|---|---|---|---|---|---|
| share | .124 | .185 | .231 | .269 | .302 | .345 | .429 | .484 | .594 |

Fixed sizes other than 26 have not been served or measured.

### KLD on the quant-fidelity-registry panel

Measured 2026-10-06 on the 25-context panel `panel--glm53.malaiwah.corpus5x5-v1` ([malaiwah/glm53-fidelity-root-v1](https://huggingface.co/datasets/malaiwah/glm53-fidelity-root-v1)) against its GLM-5.3 BF16 reference, so the numbers sit on the same scale as [malaiwah/quant-fidelity-registry](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry). This has not been submitted to the registry. Full-vocabulary KL(BF16 ‖ candidate) in float64, token mean over all 51,175 positions (25 contexts x 2,047). Layout: 26 fixed + 51 floating (`NQ_JOINT_FIXED=1 NQ_SLOTS_PER_LAYER=56`, with `NQ_TAP_IO=0`), nestquant d8ec091; the scripts are in 434e528.

**Live server** (prod stack: SSD streaming, throughput-aware prefetch, slot limits, MTP ns=3, fp8 KV):

| | KLD | top-1 agreement |
|---|---|---|
| 2/4 bit served, 2.66 bpw | **0.0620** (±0.0073 over contexts) | 92.95% |

| domain | code | encyclopedic | literary | multilingual | scientific |
|---|---|---|---|---|---|
| KLD | 0.0518 | 0.0655 | 0.0811 | 0.0628 | 0.0490 |

Each context is a 1-token prompt followed by 2,047 teacher-forced decode tokens, so every position is a decode step. Decode on these runs averaged 35.8 tok/s because the server waits for 4-bit reads without a time cap during the first 64 generated tokens of each request (`NQ_DEC_BLOCK_FIRSTN=64`, `NQ_DEC_BLOCK_MS=-1`), and those waits are long when the predictor has no history. With the waits disabled the same 1-token-prompt context decodes at 95.6 tok/s (24.2 with waits). The KLD above was measured with the waits on, as served.

**Offline** (one teacher-forced pass per context, same fp8 KV emulation; the FP8 checkpoint supplies everything except the routed experts):

| arm | bpw | KLD | top-1 | 4-bit share of expert calls |
|---|---|---|---|---|
| FP8 source | 8 | 0.0232 | 95.6% | - |
| 2/4, every expert at 4 bit | 4.14 | 0.0388 | 94.6% | 100% |
| 2/4, oracle: each 16-token block's own top 77 at 4 bit | 2.66 | 0.0396 | 94.5% | 98.7% |
| 2/4, previous block's top 77 | 2.66 | 0.0819 | 92.0% | 62.8% |

FP8 without the KV emulation reads 0.0227 here against 0.0223 in the registry. The oracle shows the 2/4 format at 77 hot experts is within 0.001 of all-4-bit; most of the remaining gap to the live number is the predictor.

Registry entries on the same panel, with this model's numbers added (bpw for registry entries is the declared nominal bits per weight from each registry entry; NestQuant's 2.66 is the routed-expert average held in VRAM, with the 4-bit residual planes streamed from NVMe):

| quant | bpw | KLD | measured as |
|---|---|---|---|
| [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) FP8 | 8 | 0.0223 | registry, single pass |
| [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF) UD-Q4_K_XL | 4 | 0.0275 | registry, single pass |
| NestQuant 2/4, every expert at 4 bit | 4.14 | 0.0388 | this card, single pass |
| NestQuant 2/4, oracle top 77 | 2.66 | 0.0396 | this card, single pass |
| [wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1) | 4 | 0.0448 | registry, single pass |
| [RadixArk/GLM-5.3-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-NVFP4) | 4 | 0.0511 | registry, single pass |
| [incoai/GLM-5.3-NVFP4](https://huggingface.co/incoai/GLM-5.3-NVFP4) | 4 | 0.0594 | registry, single pass |
| **NestQuant 2/4, live server** | **2.66** | **0.0620** | this card, live decode |
| [davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw) | 3.42 | 0.0628 | registry, single pass |
| [davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw) | 3.25 | 0.0731 | registry, single pass |
| [Inferact/GLM-5.3-NVFP4](https://huggingface.co/Inferact/GLM-5.3-NVFP4) | 4 | 0.0754 | registry, single pass |
| [davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw) | 3 | 0.0838 | registry, single pass |
| [drowzeys/keys-GLM-5.3-EXL3](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3) | 3 | 0.1023 | registry, single pass |

Registry numbers quantize every expert in one teacher-forced pass. The live NestQuant number is decode on the running server, where the 4-bit set has to be predicted and read ahead of use.

**Reproducing the live number** (4x RTX PRO 6000, model downloaded as in Running it):

```
hf download malaiwah/glm53-fidelity-root-v1 --repo-type dataset --local-dir panel
git clone https://github.com/jarrelscy/nestquant && cd nestquant
NQ_KLD_HOOK=1 NQ_DBG_DIR=$PWD/dbg NQ_TAP_IO=0 NQ_JOINT_FIXED=1 NQ_SLOTS_PER_LAYER=56 ./start.sh
export OPENAI_API_KEY=<server key>
python sm120/eval/reg_live.py run   --panel ../panel --dbg dbg/kld
python sm120/eval/reg_live.py score --panel ../panel --dbg dbg/kld --out reg_live.json
```

`NQ_KLD_HOOK=1` installs `sm120/serve/nq_kld.py`, which forces the teacher tokens and records the full-vocabulary log-softmax of every committed decode row (about 1.3 GB per context). It is for evaluation only; restart without it for normal serving. The scorer builds the reference as `hidden_states.float() @ lm_head.float().T` from the panel capture and head, without re-applying the final norm. The offline arms come from `sm120/eval/eval_reg.py` (layer-streamed FP8 backbone on 4 GPUs, `torchrun --nproc_per_node 4`; see its docstring for stream names).

## Terminal-Bench 4.0 (running)

Running tally, updated 2026-10-08 10:51Z. Served by this repo on 4x RTX PRO 6000 (`NQ_JOINT_FIXED=1 NQ_SLOTS_PER_LAYER=56 ./start.sh` for the latest attempts), one task at a time, 8 h agent timeout, stock harbor `terminus-2` agent. Runs from 2026-10-05 on use 26 fixed + 51 floating. A task that was rerun counts once, using its latest graded run. Job date is the date in the harbor job name; a job can run into the next day.

**12 pass / 27 fail / 2 timeout: 12/39 = 30.8% excluding timeouts** (12/41 = 29.3% counting timeouts as fails). 41 of the 66 tasks graded so far; the rest are queued.

For reference, published full-suite scores for GLM-5.3: 41.8% ±3.2 on the official Terminal-Bench 4.0 leaderboard (Claude Code agent, max effort), 41.9% on Artificial Analysis, 38.9% on Vals.ai (mini-SWE-agent, avg@3). These use different agents and all 66 tasks, so they are not a direct comparison; with 39 graded tasks our number carries roughly ±10 points.

Traces for every counted trial (agent trajectory with reasoning, terminal recording, grader output) and the commands and harbor config to rerun on 4x RTX PRO 6000 are in [benchmarks/tb4.0](benchmarks/tb4.0/). They are re-uploaded as tasks finish.

| task | result | job date |
|---|---|---|
| atrx-vep-crispr | pass | 2026-10-06 |
| batched-eval-parity | pass | 2026-10-05 |
| bun-sourcemap-leak | fail | 2026-10-06 |
| cad-model | timeout | 2026-09-30 |
| cargo-flight-dispatch | fail | 2026-10-04 |
| coq-block-bound | pass | 2026-10-05 |
| ctr-optimization | fail | 2026-10-06 |
| cumulative-layout-shift | fail | 2026-10-06 |
| data-anonymization | fail | 2026-10-06 |
| distributed-dedup | fail | 2026-10-06 |
| embedding-drift-monitor | pass | 2026-09-30 |
| fin-saccr-rwa | pass | 2026-09-30 |
| foodstuff-beta-activity | fail | 2026-10-05 |
| formal-crypto | fail | 2026-09-30 |
| freecad-impeller | fail | 2026-10-06 |
| freight-dispatch-shift | fail | 2026-09-30 |
| glycan-ms2-elucidation | fail | 2026-10-05 |
| heat-pump-warranty | fail | 2026-10-05 |
| ks-solver-cpp | fail | 2026-09-30 |
| lake-temp-glm | pass | 2026-09-30 |
| layout-config-recreation2 | pass | 2026-10-05 |
| legacy-utility-triage | fail | 2026-10-06 |
| live-database-cutover | fail | 2026-10-08 |
| music-harmony | fail | 2026-10-06 |
| mvcc-lsm-compaction | fail | 2026-10-06 |
| nextjs-performance | fail | 2026-10-08 |
| ontology-kg-querying | fail | 2026-10-06 |
| payments-pipeline-fix | pass | 2026-10-05 |
| photonic-waveguide-routing | pass | 2026-09-30 |
| pretrain-shard-corruption | fail | 2026-09-30 |
| production-planning | fail | 2026-10-06 |
| protein-autointerp-disulfide | fail | 2026-10-05 |
| react-lead-form | fail | 2026-09-30 |
| risk-scorer-replay | pass | 2026-10-06 |
| satb-audio-transcription | fail | 2026-09-30 |
| session-window-debug | fail | 2026-10-05 |
| sound-change-cascade | pass | 2026-09-30 |
| takens-embedding-lean | timeout | 2026-09-30 |
| telecom-entity-resolution | pass | 2026-10-06 |
| vba-userform-port | fail | 2026-10-06 |
| wal-recovery-ordering | fail | 2026-10-05 |

photonic-waveguide-routing was rerun on 2026-10-05, but the server crashed before the verifier ran, so its official result is the 09-30 pass.

## Default 4-bit set

`manifest.json` lists 26 experts per layer that are kept at 4 bit by default. They were chosen by usage on the calibration data (boundary-weighted REAP), weighted towards tokens just before the end of reasoning and the end of each turn, blended 75% text and 25% vision. The rest can be upgraded to 4 bit at runtime.

## Serving layout

`serving/tp4/` holds the same experts pre-packed for the NestQuant streaming server at tensor parallel 4, so the server does not have to repack anything. The files, per rank r (0-3) and layer L (3-77):

- `rank{r}/L{L}.bin`: the 256 level-4 records of layer L. Each record is 2,560,000 bytes (a multiple of 4 KiB) and holds, in order, gate|up P4, gate|up block words, down P4, down block words and the rank-4 fp16 low-rank U4 plane. Every segment starts on a 256-byte boundary. Record format `nq-p4rec-v1`.
- `res/rank{r}/L{L}.pt`: the resident planes of the layer (2-bit base, sign variant, scales, low-rank V/U2). Format `nq-res-v1`. For layers whose down-input Hadamard width is not 128 (layers 3-6: 512) the file also carries `in_had_down`; `streaming/resident.py` reads it there, else from `rank{r}.json`, else uses 128. Files of the other layers are unchanged.
- `layers/L{L}.json`: the per-layer block. It has the size and sha256 of every file, the record layout, the default 4-bit set, the floating default and routing counts, the sha256 of the source `layers/L{L}/manifest.json`, and a `layer_hash`.
- `rank{r}.json`: the index, in the format the server reads. It has what `streaming/repack.py` writes (`format`, `tp`, `rank`, `L0`=3, `NE`=256, `rec_bytes`, `seg`, and per layer `experts`, `rg`, `rd`). It also has, per layer, the `file`, `offset`, `bytes` and `sha256` of the records, `res`, `res_bytes` and `res_sha256` of the resident file, the `layer_hash`, and `source_manifest_sha256`. The rotation fields `in_had_down`, `had_sign_seed` and `ics_down` are there only for layers whose manifest sets them (layers 3-6: 512, 91426, "flat"). If `in_had_down` is absent, the width is 128.
- `manifest.json`: formats, the layout, the per-layer hashes, the default allocation (`default_allocation`, 26 experts per layer), `floating_default`, the per-layer `n_routed` counts, the rotation fields per layer, and `artifact_key`.
- `artifact_stamp.json`: `{"artifact": ..., "key": ..., "layers": {L: manifest sha256}}`. The key is the one `sm120/eval/run_c2.sh` computes over `layers/L*/manifest.json`: sha256 of the lines `L{L} <sha256 of layers/L{L}/manifest.json>` in version-sort order, first 16 hex digits.
- `COMPLETE`: every layer and its hash, plus the sha256 of the index, manifest and stamp. It is written last. The release is complete only when this file exists.

The server reads one record file per rank, with the record of (L, E) at ((L - 3) * 256 + E) * rec_bytes. `rank{r}/L{L}.bin` holds exactly the bytes of that range for layer L, and `res/rank{r}/L{L}.pt` is the same file `streaming/repack.py` writes for that layer. Both were checked byte for byte against `repack.py` output (nestquant commit d662dff) for layers 3-6 and 10 on all 4 ranks. `had_sign_seed` and `ics_down` are provenance only: the random signs are folded into the stored scale vectors and the flat input-channel scale into the weights, so the server needs only `in_had_down`. So an assembled `serving/tp4/` is a drop-in record directory (`NQ_REPACK_DIR`), and `rank{r}.json` loads in the server unchanged. When layers are re-fitted, only their blocks and index entries change. The record format name changes if the layout ever changes.

`serving/nq_assemble.py` (Python standard library only) does the byte copy and checks every byte range it writes against the release sha256.

**Full release, new record directory (in place, about 393 GB):**

```
hf download jarrelscy/GLM-5.3-Vision-NestQuant-2-4bit --local-dir nq \
  --include 'serving/tp4/*' --include 'serving/nq_assemble.py' --include 'serving/nq_verify_repack.py'
python nq/serving/nq_assemble.py nq/serving/tp4 --move      # nq/serving/tp4 is then the NQ_REPACK_DIR
```

**Full release into an existing record directory** (for example one built by `repack.py`):

```
python nq/serving/nq_assemble.py --src nq/serving/tp4 --into $NQ_REPACK_DIR
```

**Refit (here layers 3-6).** Download the index files and the changed layers only, then install them in place. Stop the server first. Repeat `--include` for each pattern; `hf download` reads bare extra arguments as file names.

```
hf download jarrelscy/GLM-5.3-Vision-NestQuant-2-4bit --local-dir nq \
  --include 'serving/tp4/rank?.json' --include 'serving/tp4/manifest.json' --include 'serving/tp4/COMPLETE' \
  --include 'serving/tp4/artifact_stamp.json' --include 'serving/tp4/layers/L[3-6].json' \
  --include 'serving/tp4/rank?/L[3-6].bin' --include 'serving/tp4/res/rank?/L[3-6].pt' \
  --include 'serving/nq_assemble.py' --include 'serving/nq_verify_repack.py'
python nq/serving/nq_assemble.py --src nq/serving/tp4 --into $NQ_REPACK_DIR [--layers 3-6] [--dry-run]
```

`--into` goes through each layer and rank:
- If the record directory already has the release's `layer_hash` for it, it is skipped.
- If the record directory has an entry with no hash, or a different one (for example written by `repack.py`), its bytes are hashed. If they match the release, the release entry is adopted without writing anything. Otherwise the layer is installed.
- To install a layer, it is first removed from `rank{r}.json`, so the server treats it as a non-NestQuant layer rather than a torn one. Then its records are written at their offset and read back against the sha256, `res/rank{r}/L{L}.pt` is replaced by tmp+rename, and the entry is written back.

Nothing else in the record directory is written. All index updates are atomic, and the command can be rerun after an interruption. At the end, `artifact_stamp.json` gets the key for the layers the directory holds, which equals the release's stamp when it holds the whole release. The first `--into` over a `repack.py`-built directory hashes every layer, reading about 393 GB. Later refits only hash the layers they change.

`python nq/serving/nq_verify_repack.py --release nq/serving/tp4 --repack $NQ_REPACK_DIR` checks an existing record directory against the release without writing anything. It hashes the records range and resident file of every layer and rank, and lists the layers that differ.

## Floating-set predictor

`serving/predictor/` holds the models the streaming server uses to choose which experts to hold at 4 bit.

`serving/predictor/joint/` holds jF. The historical zero-fixed default uses 77 floating experts in 80 slots per layer; the latest Terminal-Bench setup explicitly uses 26 fixed + 51 floating (`NQ_JOINT_FIXED=1 NQ_SLOTS_PER_LAYER=56`). The predictor is refreshed every 16 decode tokens on a background thread. It is a small transformer (about 2.0M parameters) over per-expert history features and a GBDT score, and runs on the GPU in about 3.8 ms per refresh. On the 4 BF16-teacher windows with fp8 KV cache it measures 0.0261 offline at 2.66 bpw, and 0.0256 served live (Live serving above). With 173 floating experts per layer (3.46 bpw), KLD is 0.0191, against 0.0241 reported for EXL3 TR3 3.42 bpw on the same windows. See `serving/predictor/joint/README.md`.

The GBDT files (`gbdt_p64_s5.txt`, `gbdt_predictor.py`, `predictor.json`) are the previous default, which kept 26 fixed experts per layer plus 51 floating. The server still runs it with `NQ_PREDICTOR=gbdt NQ_SLOTS_PER_LAYER=56`; it needs `lightgbm`.

### Reproducing the BF16-teacher KLD

The offline jF numbers above are teacher-forced: each window is run through the model once, with jF choosing the 4-bit experts every 16 tokens as the server would.

- **Teacher and windows:** confirmation windows 0000-0003 of [brandonmusic/GLM-5.3-BF16-full-logits](https://huggingface.co/datasets/brandonmusic/GLM-5.3-BF16-full-logits) (`reference-full-panel`, revision 427368f1). Token ids come from its `calibration/panel-v1/arrays/confirmation-*.tokens.npy`.
- **Score:** full-vocabulary KL(teacher ‖ NestQuant), fp32 log-softmax on both sides, mean over the 2,047 predicted positions of each window, then the mean of the 4 windows.
- **Model:** every routed expert at 2 bit, except the experts jF holds at 4 bit (`n_float` per layer, hysteresis 0.7). Each window starts cold from the published manifest's 26 fixed + 51 floating experts, all treated as floating. Everything else is the FP8 checkpoint ([zai-org/GLM-5.3-FP8](https://huggingface.co/zai-org/GLM-5.3-FP8)).
- **KV cache:** the serve's fp8_ds_mla format is emulated (fp8 latent cache and fp8 query latent). This adds 1-7% KLD over no KV quantization.

The evaluation harness and a step-by-step recipe are in the NestQuant code repository (`threads/34-tr3/REPRODUCE.md`).

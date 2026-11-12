---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- gptq
- 4-bit
- qwen
- '3.8'
---

# Qwen3.8-27B MLX 4-bit GPTQ (g4p-v2)

This is a personal re-upload of my quant. 

## Specs

15.7GB total:
- Base: affine 4-bit, group size 128 (embeddings included).
- Kept at higher precision: `lm_head` 6-bit / g64; full-attention `k_proj` + `v_proj` 8-bit / g64
  (16 layers); linear-attention `in_proj_a` + `in_proj_b` 8-bit / g64 (48 layers).
- GPTQ, applied layer by layer, calibrated on multi-turn agentic coding sessions (tool calls, shell
  output, diffs, prose) drawn from several software projects.
- The per-tensor bit widths are recorded in `config.json` (`quantization_config`).
- Vision tower included and left at BF16. 

## Results

Setup: 14" MacBook Pro w/M5 Pro (`15-core CPU`/`16-core GPU`/`48GB Memory`) served by Splash with the `incoai/Qwen3.8-27B-DFlash2` speculative draft, 64K context, long generation. Just as fast as the flat 4-bit, but better quality. 

Measured against the bf16 base on held-out data: 96 sequences / 37.8K tokens from projects that were
not part of the calibration set. KL is computed over the bf16 top-64 tokens plus a tail bucket.

| | this quant | mlx-community/Qwen3.8-27B-4bit |
|---|---|---|
| KL vs bf16 (lower is better) | **0.0747** | 0.1175 |
| Top-1 agreement with bf16 | **0.925** | 0.881 |
| Size on disk | 15.0 GB | 16.05 GB |
| Decode, single stream* | 37.7 tok/s | 36.6 tok/s |

Image reading: 60/60 values within tolerance on a screenshot-transcription probe (the mlx-community
4-bit scored the same). On an agentic coding / troubleshooting benchmark suite (one sample per item)
it tied or edged the mlx-community 4-bit on every instrument. Read those as "no regression", not as
a proven win.

## Real-world performance as I use the model:

Numbers from the status page of my [SplashGUI Project](https://github.com/cjosie/SplashGUI): 

Sub-agent work on macOS Swift projects:

| Observed Performance | 100K+ context | sub-100K context |
|---|---|---|
| Prefill, 2 concurrent sessions | ~300 tok/s | Not Measured |
| Decode, 2 concurrent sessions | ~47 tok/s | Not Measured |
| Prefill, single session | Not Measured | ~400-431 tok/s, ~600 peak |
| Decode, single session | Not Measured | ~46-49 tok/s, ~68 peak |

Sub-agent work on typescript/react projects:

| Observed Performance | 100K+ context | sub-100K context |
|---|---|---|
| Prefill, 1-2 sessions | Not Measured | ~400 tok/s, ~421peak |
| Decode, 1-2 sessions | Not Measured | ~105 tok/s - this is an outlier and I'm not actually sure why it's so high |

Agent work from another machine over LAN, working on a Go project in Zed (notably, this is at High reasoning):

| Observed Performance | 100K+ context | sub-100K context |
|---|---|---|
| Prefill, single session | ~308 tok/s, 339 tok/s peak, drops to ~288 tok/s around 130K+| Not Measured|
| Decode, single session | ~37 tok/s, 38 tok/s peak, drops to ~32 tok/s around 130K+ | Not Measured |

Setup: 
- Software: This is using the model card settings (temp 1.0, top-p 0.95, top-k 20) and (usually) low reasoning effort. Often INT8 KV Cache when RAM is constrained, otherwise BF16 (slightly faster, and hypothetically better quality, though I haven't seen degredation with 8-bit). 10GB persistent SSD cache in splash so I'm keeping ~98% cache hit rate. Draft acceptance remains ~49-51%.
- Hardware: Same 14" M5 Pro MacBook Pro (15C CPU, 16C GPU). Caldigit TS5 dock so 140W power delivery (this mac will peak ~90W) and fans spinning at 100% under load, so power and thermals are best-case. Mac's in a vertical stand as well, meaning it doesn't risk ingesting its own exhaust. 

Coherency remains good and it's sharp and accurate before compaction (using OpenCode V2.X), though does trend toward incorrect theories when troubleshooting some Xcode build issues. It loses some finer details after compaction, so far it has caught it's own mistakes but be aware of that. Just avoid compaction overall, if possible, by keeping sessions finely scoped and providing specific code lines/keywords it can grep for, rather than ingesting massive files. 

Overall, very good for this size class of model (as usual for Qwen3.8), and does very, very well if given explicit directions. Great as a sub-agent. 

## Usage

Tested with the Splash engine, which pairs the DFlash2 draft automatically:

```
serve --model cjosie/Qwen3.8-27B-MLX-4bit-GPTQ
```

`mlx_lm.load` loads the text model (used for the KL evaluation). Other MLX runtimes have not
been tested.

Sampling defaults come from the base model's `generation_config.json` (temperature 1.0, top_p 0.95,
top_k 20). See the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B) for thinking-mode and
`reasoning_effort` controls.

## License

Apache 2.0, inherited from Qwen/Qwen3.8-27B. A copy of the license is in `LICENSE`.
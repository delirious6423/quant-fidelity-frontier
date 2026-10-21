---
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
license: mit
datasets:
  - NeelNanda/pile-10k
tags:
  - ornith
  - ornith-1.5
  - qwen3.5
  - moe
  - gptq
  - int4
  - 4-bit
  - auto-round
  - vllm
  - xpu
  - intel-arc
  - image-text-to-text
  - vision
  - mtp
quantized_by: derikn
---

# Ornith-1.5-35B-A3B GPTQ INT4 (W4A16, group 128)

4-bit weight-only quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
built with Intel's [AutoRound](https://github.com/intel/auto-round) and tested on Intel Arc Pro B70 GPUs.

This is an unofficial community quantization by `derikn`. Ornith AI did not author it and does not
endorse it. For the model itself, its capabilities and its evaluation results, read the
[base model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

## What this is

| | |
|---|---|
| Base model | `ornith-ai/Ornith-1.5-35B-A3B` |
| Parameters | 35,951,822,704 (see the note on the parameter count below) |
| Architecture | Qwen3.5 MoE, 40 layers, 256 experts with 8 active, about 3B active parameters |
| Quantization | GPTQ, weight-only, 4-bit (W4A16) |
| Group size | 128, symmetric, `desc_act` off, `damp_percent` 0.01 |
| Method | AutoRound `0.15.1`, `auto_gptq` export |
| Kept at 16-bit | `lm_head`, the MoE routers, the shared-expert gates, the vision tower, the MTP head |
| Size on disk | 21 GB: 5 shards holding 20.80 GiB, plus a 1.7 GB MTP file |
| Native context | 262,144 tokens, same as the base model |
| Vision | the vision tower survives at 16-bit, so image inputs work |
| MTP | present and unquantized in `model_extra_tensors.safetensors`, so speculative decoding works |
| Chat template | included, with one local modification described below |

151 of 31,181 tensors stay unquantized: `lm_head`, 40 MoE routers, 40 shared-expert gates, 110 vision
tower blocks, and the MTP head. The routers matter most, because AutoRound holds them at high precision
by design, which is what keeps expert selection intact.

Only the linear weights are quantized. No fine-tuning and no extra training data went into this.

### Note on the parameter count

The Safetensors panel on this page reports a model size of about 6B parameters. That is not the size of
the model. The 4-bit weights are packed into int32 tensors, eight weights per int32, and the panel's
total counts that packed storage instead of the parameters it encodes.

This checkpoint holds **35,951,822,704 parameters**, the same count as the base model. The Hub API's own
per-dtype breakdown adds up to exactly that figure (`I32` 33,621,278,720 plus `BF16` 2,330,543,984),
while the panel's single total does not. `I32` and `BF16` describe storage, not the precision of the
model: packed 4-bit weights, and the tensors held at bfloat16.

### Chat template modification

The base template raises an exception when a system message appears after the first turn. This
repository renders such a message as a user turn wrapped in `<system-reminder>` tags instead, which is
what agent frameworks pass mid-conversation context. Nothing else in the template changed. The file is
`chat_template.jinja`; the modification is 3 lines around line 85.

## How it was made

Quantized from the 71.9 GB BF16 checkpoint with AutoRound 0.15.1 on a single Intel Arc Pro B70 (34 GB).
GPTQ rather than AWQ, because the AWQ MoE path is not reliable on this hardware. Weights stream to the
card during tuning, which keeps a 71.9 GB model inside one card's budget:

```
python layer2/quantize.py \
  --model <base-model-dir> \
  --format auto_gptq --device xpu:0 --low-gpu-mem \
  --n_samples 128 --seqlen 1024
```

Calibration used the quantizer's default set, **NeelNanda/pile-10k**, at 128 samples of 1024 tokens
each.

The script above is a private wrapper around AutoRound and is not published, so that exact command will
not run for you. These are the AutoRound settings it passes, which are the whole recipe:

- export format `auto_gptq`: GPTQ, 4-bit weight-only, group size 128, symmetric, `desc_act` off
- `lm_head`, the MoE routers, the shared-expert gates, the vision tower and `mtp.*` excluded from quantization
- calibration on NeelNanda/pile-10k, 128 samples of 1024 tokens
- weights streamed to the device during tuning, to keep VRAM bounded
- `torch.compile` off

Measured on the machine that produced it (2 Arc Pro B70, 31 GB host RAM): 40 blocks in 3 h 13 m, 289 s
per block on average, peak host RAM 18.25 GB, peak VRAM 20.19 GB on one card.

## Running it

Tested on **vLLM XPU** (`vllm/vllm-openai-xpu:nightly`) with tensor parallelism across two Arc Pro B70s,
MTP speculative decoding, and the full 262,144-token context:

```bash
vllm serve /models/Ornith-1.5-35B-A3B-GPTQ-INT4 \
  --served-model-name ornith15-mtp \
  --quantization gptq --dtype float16 \
  --max-model-len 262144 --block-size 64 \
  --max-num-seqs 8 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.86 \
  --enable-prefix-caching \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --trust-remote-code --tensor-parallel-size 2
```

Environment used on Arc: `VLLM_TARGET_DEVICE=xpu`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`,
`ZE_AFFINITY_MASK` set to the cards you want, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` when you push the
context past the configured 262,144, `OMP_NUM_THREADS=8`.

Container settings that matter: `--device=/dev/dri`, `--group-add` for the render and video groups,
`--ipc=host`, `--security-opt seccomp=unconfined`, and `--shm-size 8g`. Without the device, group and
seccomp combination Level Zero reports zero devices.

### Tool calling and thinking

The base model's own card recommends the `qwen3_xml` tool parser and the `qwen3` reasoning parser, and
that pairing is what this checkpoint was tested with on vLLM XPU. Thinking is controlled through the
chat template's `enable_thinking` flag.

### What was tested, and what was not

Tested:

- 256K context loads and serves, at tensor parallel 2 with an fp8 KV cache.
- MTP speculative decoding runs from the retained `model_extra_tensors.safetensors`, at 3 speculative
  tokens.
- Tool calling and reasoning work with the parsers above.
- The vision tower loads and the processor config is intact, so image inputs remain available.

Not tested:

- No accuracy or throughput numbers are published with this repository. The quantizer's own
  measurements come from one home-lab setup (Arc Pro B70, XPU vLLM nightly) and are not a fair
  comparison for anyone running different hardware or software. Model quality is the base model's, so
  read its card.

## Licence and attribution

The base model declares **MIT**. Its repository does not ship a licence file and its declared
`license_link` returns 404, so the copyright line in `LICENSE` uses the publishing organisation's name
and will be corrected if the authors publish their own. The base card states that Ornith-1.5 was
developed on top of **Qwen3.5** and **Gemma 4**, both of which are Apache 2.0; `LICENSE-APACHE-2.0` is
included for that lineage. `NOTICE.txt` has the full statement.

- Base model: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MIT as declared)
- Lineage: Qwen3.5 and Gemma 4 (Apache 2.0)
- Quantization tooling: [intel/auto-round](https://github.com/intel/auto-round) (Apache 2.0)
- Quantizer: a private in-house wrapper around AutoRound (not published)

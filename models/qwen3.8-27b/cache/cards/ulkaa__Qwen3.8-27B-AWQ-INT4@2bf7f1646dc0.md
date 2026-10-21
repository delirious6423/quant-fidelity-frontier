---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - qwen
  - qwen3_5
  - awq
  - w4a16
  - compressed-tensors
  - int4
  - 4-bit
  - sglang
  - intel-xpu
  - quantized
  - multimodal
  - long-context
  - 1m-context
  - yarn
  - speculative-decoding
---

# ⚡ Qwen3.8-27B-AWQ-INT4

AWQ **W4A16** quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced with
[llm-compressor](https://github.com/vllm-project/llm-compressor) in
`compressed-tensors` `pack-quantized` format.

Built and qualified with SGLang on **Intel Arc Pro** GPUs. The current
[Docker image](https://hub.docker.com/r/rahulunair/sglang-xpu) supports both the
checkpoint's **MTP head** and the external
[`Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
draft model.

| | |
|---|---|
| Size | **18.2 GiB** (from 55.6 GiB BF16) |
| Format | `compressed-tensors` / `pack-quantized` |
| Scheme | W4A16 asymmetric, group size 128 |
| Vision tower | preserved, BF16 |
| MTP head | preserved, BF16 |
| Gated DeltaNet projections | quantized |
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Context | **262,144** on `main`; **1,048,576** on the [`1m` revision](https://huggingface.co/ulkaa/Qwen3.8-27B-AWQ-INT4/tree/1m) |

## 🧠 1,048,576-token context

The `1m` revision contains the same weights as `main`. Only `config.json`
changes: YaRN factor 4 extends the original 262,144-token configuration to
1,048,576 tokens.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "ulkaa/Qwen3.8-27B-AWQ-INT4",
    revision="1m",
    local_dir="models/Qwen3.8-27B-AWQ-INT4-1m",
)
```

The current TP4 production recipe uses:

```text
--context-length 1048576
--max-total-tokens 1048576
--max-mamba-cache-size 40
--max-running-requests 64
```

This produces an exact 1,048,576-token target pool with eight effective
speculative request slots. Requests beyond active capacity queue. All active
requests share the target pool, and each request's input plus requested output
must fit the context limit. A full 1,048,576-token input leaves no room for
generated tokens.

## 🚀 SGLang on Intel Arc Pro

Pull the model-scoped image:

```bash
docker pull rahulunair/sglang-xpu:qwen3.8-27b-20260819
```

The copy-paste TP4 commands, required container permissions, readiness checks,
and full context notes live in the
[Qwen3.8-27B Docker Hub section](https://hub.docker.com/r/rahulunair/sglang-xpu#qwen38-27b).
The moving model alias is `rahulunair/sglang-xpu:qwen3.8-27b`; use the dated
tag above when reproducibility matters.

The image carries the qualified Intel XPU defaults for decode INT8, symmetric
all-reduce, speculative verify, and graph capture. Do not add SGLang's CUDA
custom collective; the published command uses `--disable-custom-all-reduce`
so the XPU plugin owns the collective path.

## 🔮 Speculative decoding

Both supported modes use target verification. The draft proposes tokens; it
does not bypass the target model.

### MTP

The MTP head is inside this checkpoint and needs no second model. It is kept in
BF16 and excluded from compressed-tensors quantization.

```bash
--speculative-algorithm EAGLE \
--speculative-num-steps 7 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 8
```

### DFlash2

Download the external draft and mount it at `/draft`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "incoai/Qwen3.8-27B-DFlash2",
    local_dir="models/Qwen3.8-27B-DFlash2",
)
```

```bash
--speculative-algorithm DFLASH \
--speculative-draft-model-path /draft \
--speculative-num-draft-tokens 8 \
--speculative-dflash-block-size 8 \
--speculative-draft-window-size 2048
```

The published image includes the pinned DFlash2 model class and a compact
physical draft-KV ring, so the target can retain a 1,048,576-token pool instead
of allocating a second full-context draft pool. Radix prefix caching remains
enabled. DFlash2 logprob-returning requests are rejected before scheduling
because that request shape is not implemented.

## 🔬 What is quantized

Qwen3.8-27B is a hybrid: 48 of its 64 decoder layers use Gated DeltaNet
(`linear_attn.*`) and 16 use full attention.

The three large DeltaNet projections, `in_proj_qkv`, `in_proj_z`, and
`out_proj`, are quantized. Together they occupy 10.36 GiB in BF16 and 2.73 GiB
in this artifact. The vision tower, MTP head, `lm_head`, `embed_tokens`, all
norms, and the small DeltaNet scalar gates `in_proj_a` and `in_proj_b` remain
BF16.

| storage view | result |
|---|---:|
| 24.33B parameters stored at four bits | 4.16 bits each including group scales and zero points |
| 3.45B parameters kept at BF16 | 12.4% of the model |
| whole artifact | 5.63 bits per parameter |

Activations remain BF16, hence W4A16. The artifact is deliberately larger than
a maximally compressed four-bit build because the output head, embeddings,
vision tower, MTP head, norms, and small gates stay at full precision.

## 🧪 Quantization and quality

- **Source:** `Qwen/Qwen3.8-27B` BF16, not the FP8 release.
- **Method:** AWQ via llm-compressor, W4A16 asymmetric, group size 128.
- **Calibration:** 512 sequences of 1,024 tokens, with 60% code instructions
  from `codeparrot/self-instruct-starcoder` and 40% general instructions from
  `HuggingFaceH4/ultrachat_200k`.
- **Pipeline:** sequential by `Qwen3_5DecoderLayer`, so each layer was calibrated
  against the quantized output of the preceding layers.

The group-128 choice was measured against group 32. Smaller groups add scale
and zero-point metadata to every weight read. Prefill perplexity moved from
8.87 in BF16 to 9.33 after quantization.

Greedy text can diverge from BF16, as expected for a four-bit build. Evaluate
the model on your own workload when exact BF16 behavior matters.

## ⚠️ Limitations

- The vision tower and processor configuration are preserved, but multimodal
  serving was only lightly exercised. The speculative recipe is qualified as a
  text-serving path.
- Calibration used text only. The unquantized vision tower was not calibrated.
- The 1M revision passed retrieval probes beyond the native context, but no
  graded long-context quality benchmark is claimed.
- DFlash2 does not currently support returned logprobs.

## 📄 License

Apache 2.0, inherited from the base model.

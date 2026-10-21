---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - ornith
  - qwen3_5_moe
  - awq
  - w4a16
  - compressed-tensors
  - int4
  - 4-bit
  - 4bit
  - moe
  - sglang
  - intel-xpu
  - quantized
  - multimodal
  - mtp
  - speculative-decoding
---

# Ornith-1.5-35B-A3B-AWQ-INT4

AWQ `W4A16` quantization of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
in compressed-tensors `pack-quantized` format, produced for SGLang on Intel Arc
Pro B-series.

**The multi token prediction head is preserved at BF16** so it remains
available for SGLang speculative decoding.

| | |
|---|---|
| Size | **{{ARTIFACT_SIZE_GIB}} GiB** |
| Format | `compressed-tensors` / `pack-quantized` |
| Scheme | W4A16 asymmetric, group size 32 |
| Quantized | routed MoE experts only |
| Vision tower | preserved, BF16 |
| MTP head | preserved, BF16 |
| Context | 262,144 tokens (native configuration) |

## The MTP head

`Ornith-1.5-35B-A3B` ships a multi token prediction head, declared in its config
as `mtp_num_hidden_layers: 1`. On this model the head is not small: it carries
its own full 256-expert mixture of experts, **785 tensors and about 822 M
parameters, 2.3 percent of the whole model**.

transformers does not instantiate an MTP module for
`Qwen3_5MoeForConditionalGeneration`. The subtree is therefore absent from the
model object during quantization, and `save_pretrained` cannot write what is not
there. The head is dropped, silently, and nothing in the process errors. This is
why a quantization workflow can silently produce an artifact with **zero**
`mtp.*` tensors.

Here the head is carried across from the source checkpoint byte for byte into
`model-mtp.safetensors`. It was never loaded and never quantized, so it is
bit-identical BF16.

Without it, speculative decoding has no weights to load. Nothing errors; the
accept length simply pins at 1.00 and the model looks like a weak drafter rather
than a broken one.

## What is quantized, and what is not

Ornith-1.5-35B-A3B is a fine-grained mixture of experts: **256 experts per
layer, 8 active per token**, across 40 layers, with `moe_intermediate_size` 512.
The routed experts are where the parameters live.

Quantized: the routed experts, and only those. **40 layers x 256 experts x 3
projections = 30,720 modules.**

Left at BF16:

| | params | share |
|---|---:|---:|
| linear attention (`linear_attn.*`, 30 layers) | 1.007 B | 2.81% |
| **MTP head** | 0.822 B | 2.30% |
| `embed_tokens` | 0.509 B | 1.42% |
| `lm_head` | 0.509 B | 1.42% |
| vision tower (27 blocks) | 0.411 B | 1.15% |
| full attention (`self_attn.*`, 10 layers), shared experts, router gates | 0.336 B | 0.94% |

## How much of this is actually 4 bit

| | |
|---|---|
| stored at 4 bits | **32.21 B parameters, 90.0 percent of the model** |
| kept at BF16 | 3.6 B parameters, 10.0 percent |

Activations are BF16 throughout; only weights are quantized, hence W4A16.

Footprint and throughput are not comparable across quantization formats. A build
that packs more of the model into fewer bits will be smaller and will usually
decode faster on the same hardware, because decode at batch one is bound by how
many bytes move per token. Which model is better for your use is answered by
measuring quality on your workload, not by comparing file sizes.

## Quantization details

- **Source**: `ornith-ai/Ornith-1.5-35B-A3B` BF16, not re-quantized from the FP8
  or NVFP4 releases.
- **Method**: AWQ via llm-compressor, `W4A16` **asymmetric**, **group size 32**,
  MSE observer, int8 zero points, `pack-quantized`.
- **Calibration**: 512 sequences of 1024 tokens, 60 percent code instructions
  from `codeparrot/self-instruct-starcoder` and 40 percent general instructions
  from `HuggingFaceH4/ultrachat_200k`, chat-template formatted.
- **Pipeline**: sequential, one decoder layer at a time
  (`sequential_targets=["Qwen3_5MoeDecoderLayer"]`), so each layer is calibrated
  against the quantized output of the layers before it.

**Group size 32 rather than 128, and this is the opposite choice from a dense
model.** A group is a run of weights along the input dimension sharing one scale
and one zero point, so what matters is how many groups each row gets. These
experts are narrow: `moe_intermediate_size` is 512, so at group 128 an expert's
`down_proj` would carry **four scales for the entire row** and a single outlier
would set the quantization step for 128 weights. A dense model's input
dimensions are 8x to 34x larger and never hit this. Group 32 costs about 11
percent more bytes read per token and buys back the resolution.

**Calibration size matters more here than for a dense model.** With 256 experts
and top-8 routing, each expert observes roughly one thirty-second of the
calibration tokens, so the sample count sets how well the tail experts are seen,
not merely how stable the scales are.

## Quality

This release passed a deterministic **coherence gate, not a benchmark**, while
served through SGLang on four Intel Arc Pro B70 GPUs. The gate verifies that:

- `reasoning_content` is non-empty and the separate final answer is correct on
  multi-step arithmetic and syllogistic reasoning;
- factual and Python coding responses are coherent and contain their required
  result;
- a basic image input is accepted and its dominant color is identified;
- output is printable and does not collapse into repetitive text.

The full prompts, reasoning, final responses, token usage, and pass flags ship
in [`coherence-results.json`](./coherence-results.json). This is a release smoke
test; it does not estimate benchmark accuracy or claim parity with BF16.

## Usage

### SGLang on Intel Arc

    docker run --rm -d --name ornith15 \
      --device=/dev/dri -v /dev/dri:/dev/dri \
      --group-add video --group-add "$(getent group render | cut -d: -f3)" \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      --ipc=host --shm-size=64g --ulimit memlock=-1 \
      -p 30000:30000 -v /path/to/Ornith-1.5-35B-A3B-AWQ-INT4:/model:ro \
      -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
      rahulunair/sglang-xpu:latest \
      python -m sglang.launch_server --model-path /model \
        --device xpu --tp-size 4 --moe-runner-backend triton \
        --host 0.0.0.0 --port 30000 \
        --trust-remote-code --attention-backend intel_xpu --page-size 64 \
        --context-length 40960 --max-total-tokens 40960 \
        --max-mamba-cache-size 64 \
        --chunked-prefill-size 4096 --mem-fraction-static 0.85 \
        --reasoning-parser qwen3-thinking \
        --cuda-graph-config '{"decode":{"backend":"full","bs":[1,2,4,8]},"prefill":{"backend":"disabled"}}' \
        --skip-server-warmup

`SYS_PTRACE` and `seccomp=unconfined` are required. Without them the
symmetric-memory all-reduce declines silently and the slower stock collective
runs, with nothing in the log to say so.

**Check that `max_total_num_tokens` equals `context_len` in the startup log.**
If it is smaller, the Gated DeltaNet state pool has taken the KV cache and long
prompts will fail; `--max-mamba-cache-size` is what corrects it.

This is a thinking model. Pass `--reasoning-parser qwen3-thinking` so the trace
is returned in `reasoning_content` and the answer in `content`.

Recommended sampling, from the base model card: `temperature=0.6`, `top_p=0.95`,
`top_k=20` for general use, `temperature=1.0` to reproduce its published
benchmark numbers.

### Speculative decoding with the MTP head

The head is BF16 and excluded from quantization, so it loads as plain BF16
linears. It is off unless asked for and costs nothing while off.

Support for an MTP head on this architecture depends on your SGLang build
registering an MoE MTP model class. Confirm before relying on it.

## Limitations

- **Quality is not benchmarked.** The release coherence gate verifies normal
  reasoning, answer, code, factual, and basic image behavior only.
- **Vision is lightly exercised.** The vision tower is unquantized and the
  release gate includes one synthetic color image, but no multimodal benchmark
  was run.
- The AWQ smoothing scale for each layer's expert input is shared between the
  routed experts and the shared expert, and the shared expert is not quantized
  here, so it contributes to the smoothing objective without benefiting from it.
  This is what upstream llm-compressor does for every Qwen mixture of experts,
  and what the reference artifact was produced under.
- Treat this as a 4 bit model. Greedy text will diverge from BF16, as it will
  for any 4 bit build, so it is not a drop in replacement where exact BF16
  outputs matter.

## License

MIT, inherited from
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

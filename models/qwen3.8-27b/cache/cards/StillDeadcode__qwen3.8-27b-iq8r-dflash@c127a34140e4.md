---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- radiance
- rocm
- amd
- rdna4
- int8
- w8a8
- iq8r
- gptq
- hadamard
- speculative-decoding
- dflash2
- vision
datasets:
- StillDeadcode/qwen3.8-27b-quant-challenge
---

# Qwen3.8-27B IQ8R + DFlash2 — radiance container

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with an int8 trunk fitted by GPTQ in the
Hadamard domain, as a single `.rad` container for the
[radiance](https://codeberg.org/StillDeadcode/radiance) inference engine (AMD RDNA4, ROCm), with its
vision tower and the [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
block-diffusion drafter merged in for speculative decoding.

It replaces [StillDeadcode/qwen3.8-27b-fp8](https://huggingface.co/StillDeadcode/qwen3.8-27b-fp8): the
same eight bits a weight, about a ninth of its divergence from the bf16 model, at the same speed.
[StillDeadcode/qwen3.8-27b-iq8r](https://huggingface.co/StillDeadcode/qwen3.8-27b-iq8r) is an earlier
IQ8R file (a bf16 scale per 128, the MTP head instead of DFlash2).

| | |
|---|---|
| File | `qwen3.8-27b-iq8r-dflash.rad` — 28.76 GiB |
| Weights | every trunk linear int8 (codes −127..127) with a u8 scale per row and 256 of K under a bf16 scale a row, each 128 of K rotated by the unnormalised Hadamard; codes chosen by GPTQ against per-linear Hessians. The lm_head int8 with a bf16 scale per row and 128, GPTQ too |
| Activations | int8 under the same rotation (W8A8) |
| Speculator | DFlash2, its linears on a 16-level integer codebook (four bits) in the same rotated domain, round to nearest; its vocabulary head as 2-bit codes, whose top 32 candidates the engine rescores against the model's own head |
| Vision | the 27-block vision tower, bf16: images and video in chat requests |
| Context | 262,144 tokens trained; 200K tested |
| Needs | radiance built from main after 1.17.0: IQ8R at one scale a 256 of K and the int8 KV cache are not in 1.17.0 and come with the next release |

## Quality

KL divergence from the bf16 model's next-token distribution over the positions a model writes, on the
public split of
[StillDeadcode/qwen3.8-27b-quant-challenge](https://huggingface.co/datasets/StillDeadcode/qwen3.8-27b-quant-challenge)
(464,091 positions), both served by radiance on two cards with a bf16 KV cache:

| | mean KLD | top-1 agreement |
|---|---|---|
| this file | 0.00039 | 99.19% |
| this file, `kv-cache-dtype: i8` | 0.00043 | |
| qwen3.8-27b-fp8 (Qwen's FP8 checkpoint) | 0.00363 | 97.38% |

## Speed

One Radeon AI PRO R9700 (gfx1201), milliseconds a speculative step at 1 / 4 / 8
concurrent sequences: 55.0 / 59.9 / 70.4, against 54.6 / 60.8 / 72.3 for qwen3.8-27b-fp8 on the same
card. Both accept the same number of drafted tokens (12.5 and 12.6 a step over 24 chat prompts).
Without the drafter a decode step is 47.9 ms (FP8 48.7) and an 8K-token prefill runs at 2649 tokens
a second (FP8 2715).

## Serve

```yaml
# radiance.yaml
server:
  host: 0.0.0.0
  port: 8000
models:
  qwen3.8-27b:
    model: qwen3.8-27b-iq8r-dflash.rad
    tp: 2
    max-model-len: 200000
    max-num-seqs: 8
    kv-cache-dtype: i8
```

```sh
radiance --config radiance.yaml
```

`kv-cache-dtype: i8` stores the attention cache as int8 with an f16 scale per token and head: half
of bf16's bytes, as `fp8`, and closer to it. The server speaks the OpenAI API
(`/v1/chat/completions`, `/v1/completions`), with tool calls and structured output, and
`image_url` / `video_url` parts in chat messages. The drafter's depth is chosen automatically
(`num-speculative-tokens: N` states one, `0` turns speculation off).

Tested on 2× Radeon AI PRO R9700 at 200K context with eight sequences, the drafter on and the int8
cache.

## How this file was made

The Hessians are the input Gram matrices of every trunk linear and of the head, recorded by radiance
running the bf16 model over the `calib` split of the challenge's dataset. `CALIB` names the
directory that holds them:

```sh
CALIB=DIR rad-convert Qwen/Qwen3.8-27B --draft-model z-lab/Qwen3.8-27B-DFlash2 \
    --recipe q38-27b-iq8r.recipe \
    --set quantization_config.quant_method=w8a8 --set quantization_config.fmt=fwht128 \
    -o qwen3.8-27b-iq8r-dflash.rad
```

The recipe (`q38-27b-iq8r.recipe` in this repository); everything it does not name is the
checkpoint's own:

```
blk.*.attn_qg.weight                 gptq  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search calib=$CALIB
blk.*.attn_k.weight                  gptq  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search calib=$CALIB
blk.*.attn_v.weight                  gptq  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search calib=$CALIB
blk.*.attn_output.weight             gptq  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search calib=$CALIB
blk.*.ssm_inz.weight                 gptq  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search calib=$CALIB
blk.*.ssm_out.weight                 gptq  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search calib=$CALIB
blk.*.ffn_gate_up.weight             gptq  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search calib=$CALIB
blk.*.ffn_down.weight                gptq  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search calib=$CALIB
output.weight                        gptq  codes=i8 group=128 scale=bf16 clamp=sym rule=search calib=$CALIB
dflash.draft_head.weight             rtn   codes=u2 zero=u8 group=128 scale=f16 rule=search
dflash.fc.weight                     rtn   table=-127;-97;-75;-58;-43;-30;-18;-6;6;18;30;43;58;75;97;127 group=128 scale=bf16 transform=fwht128 rule=search
dflash.blk.*.attn_q.weight           rtn   table=-127;-97;-75;-58;-43;-30;-18;-6;6;18;30;43;58;75;97;127 group=128 scale=bf16 transform=fwht128 rule=search
dflash.blk.*.attn_k.weight           rtn   table=-127;-97;-75;-58;-43;-30;-18;-6;6;18;30;43;58;75;97;127 group=128 scale=bf16 transform=fwht128 rule=search
dflash.blk.*.attn_v.weight           rtn   table=-127;-97;-75;-58;-43;-30;-18;-6;6;18;30;43;58;75;97;127 group=128 scale=bf16 transform=fwht128 rule=search
dflash.blk.*.attn_output.weight      rtn   table=-127;-97;-75;-58;-43;-30;-18;-6;6;18;30;43;58;75;97;127 group=128 scale=bf16 transform=fwht128 rule=search
dflash.blk.*.ffn_gate_up.weight      rtn   table=-127;-97;-75;-58;-43;-30;-18;-6;6;18;30;43;58;75;97;127 group=128 scale=bf16 transform=fwht128 rule=search
dflash.blk.*.ffn_down.weight         rtn   table=-127;-97;-75;-58;-43;-30;-18;-6;6;18;30;43;58;75;97;127 group=128 scale=bf16 transform=fwht128 rule=search
```

## License

Apache 2.0, as Qwen/Qwen3.8-27B; its LICENSE is in this repository.

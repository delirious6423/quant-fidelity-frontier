---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model:
- google/gemma-4-26B-A4B-it
- z-lab/gemma-4-26B-A4B-it-DFlash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- radiance
- rocm
- amd
- rdna4
- int8
- hadamard
- moe
- speculative-decoding
- dflash
---

# Gemma 4 26B-A4B-it + DFlash, int8 experts — radiance container

[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) as a single `.rad`
container for the [radiance](https://codeberg.org/StillDeadcode/radiance) inference engine (AMD RDNA4,
ROCm), with z-lab's [DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) block-diffusion
drafter merged in for speculative decoding. radiance's Gemma 4 26B-A4B support is by Perire.

It replaces [StillDeadcode/gemma4-26b-a4b-dflash-fp8](https://huggingface.co/StillDeadcode/gemma4-26b-a4b-dflash-fp8),
with the same eight bits a quantised weight and no E4M3 weight left.

| | |
|---|---|
| File | `gemma4-26b-a4b-dflash-i8.rad` — 34.17 GiB |
| Weights | the routed experts' gate_up projections int8 (codes −127..127) in the 128-wide Hadamard domain with a bf16 scale per row and 128 of K, round to nearest with a scale search; their down projections, the attention, the dense MLP, the router, the norms and the tied embedding bf16 as the checkpoint ships them |
| Speculator | DFlash, drafting 15 tokens a step. Its linears and its head are int8 with an f16 scale per row and 128 of K; the drafter only proposes, and the model verifies every token it emits |
| Context | 262,144 tokens (the checkpoint's trained length) |
| Modalities | text |
| Needs | radiance built from main after 1.17.0: the int8 experts, the attention join that writes their input and the int8 cache at this model's attention are not in 1.17.0 and come with the next release |

## Serve

```yaml
# radiance.yaml
server:
  host: 0.0.0.0
  port: 8000
models:
  gemma4-26b:
    model: gemma4-26b-a4b-dflash-i8.rad
    tp: 2
    max-model-len: 32768
    max-num-seqs: 8
    kv-cache-dtype: i8
    draft-window: adaptive
```

```sh
radiance --config radiance.yaml
```

The server speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`), with tool calls and
structured output. `kv-cache-dtype: i8` is the int8 cache (K and V as int8 codes with an f16 scale each per
token and head); `fp8` and `bf16` are served too. `draft-window: adaptive` lets the drafter's window follow
how much of it is accepted; without it every step drafts the container's 15 tokens.

Measured on two Radeon AI PRO R9700 (gfx1201) with the configuration above: a decode step without the
drafter takes 14.1 to 14.5 ms at one sequence, 14.9 to 15.5 ms at four and 16.6 to 17.1 ms at eight (the
FP8 container on its fp8 cache: 14.3 to 14.4, 15.7 to 15.8 and 16.8); with the drafter, 24 prompts four at
a time (12,000 tokens) run at 349 to 352 tokens a second (the FP8 container: 369 to 371, at the same time a
step and about 3% more tokens a step).

## How this file was made

```sh
rad-convert google/gemma-4-26B-A4B-it --draft-model z-lab/gemma-4-26B-A4B-it-DFlash \
    --recipe gemma4-26b-a4b-df-i8.recipe -o gemma4-26b-a4b-dflash-i8.rad
```

The recipe (`gemma4-26b-a4b-df-i8.recipe` in this repository) names what is quantised; everything it does
not name is the checkpoint's bf16:

```
blk.*.ffn_gate_up_exps.*.weight      rtn codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search
dflash.fc.weight                     rtn codes=i8 group=128 scale=f16 clamp=sym
dflash.blk.*.attn_q.weight           rtn codes=i8 group=128 scale=f16 clamp=sym
dflash.blk.*.attn_k.weight           rtn codes=i8 group=128 scale=f16 clamp=sym
dflash.blk.*.attn_v.weight           rtn codes=i8 group=128 scale=f16 clamp=sym
dflash.blk.*.attn_output.weight      rtn codes=i8 group=128 scale=f16 clamp=sym
dflash.blk.*.ffn_gate_up.weight      rtn codes=i8 group=128 scale=f16 clamp=sym
dflash.blk.*.ffn_down.weight         rtn codes=i8 group=128 scale=f16 clamp=sym
dflash.draft_head.weight             rtn codes=i8 group=128 scale=f16 clamp=sym
```

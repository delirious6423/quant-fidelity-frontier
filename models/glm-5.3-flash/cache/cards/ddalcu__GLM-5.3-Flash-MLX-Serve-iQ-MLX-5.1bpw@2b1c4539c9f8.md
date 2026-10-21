---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: mlx-serve
license: mit
license_link: LICENSE
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- mlx
- mlx-serve
- glm
- glm5_next
- moe
- gptq
- native-mtp
---

# GLM-5.3-Flash for mlx-serve, iQ-MLX 5.1 bpw (GPTQ experts, 4/5/6-bit by layer)

mlx-serve pack of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(`model_type: glm5_next`, 320B total / 18B active). It takes 202.5 GB on disk
and runs on a stock 256 GB Mac. The pack includes the MTP head and the vision
tower.

The routed experts are quantized with GPTQ on Hessians collected from the BF16
model. The bits go where errors grow the most, in the early layers. At 9% more
bytes than the 4-bit reference pack, the KLD to BF16 is half, and decode speed
is the same. It matches the 224 GB 5-bit oQ5e pack's quality in 21.5 GB less,
decodes about 10% faster in mlx-serve, and fits without raising macOS's GPU
memory limit.

Download [MLXServe.com](https://mlxserve.com)

```bash
mlx-serve --model ddalcu/GLM-5.3-Flash-MLX-Serve-iQ-MLX-5.1bpw --serve
```

## Calibration and what it buys

The BF16 checkpoint was streamed layer by layer over 1.05 M calibration tokens:
agent transcripts, code, prose, math and Wikipedia. Each expert gets its own
Hessians, and every token's input is weighted by its router weight. GPTQ uses
damping 0.1 and group size 64.

Quality is scored against the BF16 model's own logits on 31 held-out windows of
2048 tokens. Top-1 means the pack's greedy token equals BF16's. KLD is
KL(BF16 || pack) over BF16's top-1024 tokens.

Scores cover assistant-turn and raw-text positions only. Inside a chat *user*
turn, GLM-5.3 predicts its own answer rather than the user's text, and those
positions flip on any perturbation, BF16 numerics included.

| pack | size | top-1 | KLD mean | KLD median |
|---|---|---|---|---|
| [TensorFold oQ4-MTP](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP) | 185.7 GB | 88.58% | 0.0871 | 0.0194 |
| uniform GPTQ 4/5 (gate-up/down), same pipeline | 198.9 GB | 91.74% | 0.0494 | 0.0096 |
| **iQ-MLX-5.1bpw (this pack)** | **202.5 GB** | 91.92% | 0.0448 | 0.0087 |
| [Jundot oQ5e-mtp](https://huggingface.co/Jundot/GLM-5.3-Flash-oQ5e-mtp) | 224.0 GB | 92.19% | 0.0440 | 0.0087 |

The table above uses all 25,471 assistant and raw-text positions. Each pack's
own weights were dequantized and run layer by layer against the BF16 model. By
region, this pack is ahead of oQ5e on assistant turns (KLD 0.0403 vs 0.0416),
and oQ5e is ahead on raw Wikipedia text (0.0466 vs 0.0496).

The same comparison served by mlx-serve, on 392 sampled positions:

| pack | top-1 | KLD mean | KLD median |
|---|---|---|---|
| oQ4-MTP | 84.44% | 0.1072 | 0.0214 |
| **this pack** | **89.03%** | **0.0541** | 0.0089 |
| oQ5e-mtp | **89.03%** | 0.0551 | **0.0082** |

On those positions the served KLD for this pack matches its harness KLD: median
difference 0.0001, and no position is more than 0.6 nats apart.

For scale: a single 8-bit tensor in layer 0, with everything else BF16, already
lands at top-1 94.3% / KLD 0.025. Early perturbations change routing
downstream. That is why non-expert upgrades buy nothing here, and the bytes go
to the routed experts.

Speed on an M5 Ultra (256 GB), two 1024-token greedy answers per pack (code,
prose), decode tok/s:

| pack | mlx-serve 26.10.2, MTP on | oMLX 0.7.1.dev1 | fits a stock 256 GB Mac |
|---|---|---|---|
| oQ4-MTP | 76.3 / 75.7 | | yes |
| **this pack** | **75.9 / 74.1** | 59.3 / 59.8 | **yes** |
| oQ5e-mtp | 67.3 / 68.1 | 58.9 / 59.9 | no: needs `iogpu.wired_limit_mb` raised |

At macOS's default GPU memory limit (222.7 GiB on a 256 GB Mac), oQ5e's 208.5
GiB of weights does not leave room to load: mlx-serve is killed during the load
and oMLX aborts the first request. This pack loads with room to spare in both.

Bits per weight, scales included: 5.05 over the 320.6 B quantized weights.

## Widths

| tensors | width |
|---|---|
| routed experts, layers 3-12 | gate/up 5-bit, down 6-bit, GPTQ |
| routed experts, layers 13-26 | gate/up 4-bit, down 5-bit, GPTQ |
| routed experts, layers 27-44 | 4-bit, GPTQ |
| attention (KDA, MLA, DSA indexer), shared experts, dense MLP (layers 0-2) | 8-bit |
| embed_tokens, lm_head | 8-bit |
| routers, hyper-connection tensors, norms, convs | bf16 |
| MTP head, vision tower | as in oQ4-MTP |

Every quantized tensor uses group size 64. The per-layer plan was chosen from
measured sensitivity: expert error at layers 3-10 costs about 10x the KLD of
the same error at layers 31-44, and down_proj costs about 1.6x more KLD per
byte than gate/up. Gate and up share a width in every layer, as mlx-serve's
fused MoE decode kernel requires.

## Compatibility

Tensor names and layout follow the oQ4-MTP pack, and the config keeps the
upstream per-layer lists (45 entries, none for the MTP layer). Tested loading
and answering on an M5 Ultra:

| engine | status |
|---|---|
| [mlx-serve](https://mlxserve.com) 26.10.2 | serves it, MTP drafting, fused MoE decode kernels at every width |
| [oMLX](https://github.com/jundot/omlx) 0.7.0 and 0.7.1.dev1 | loads the folder as is |
| [TensorFold](https://huggingface.co/TensorFold) 0.6.0 | loads, MTP drafting; the 5/6/8-bit tensors run through its row kernels |

mlx-lm has no `glm5_next` model yet. The architecture lives in mlx-vlm
(0.7.1 and later; oMLX 0.7.1.dev1 ships 0.7.7), which oMLX uses to load it.

## Serving notes

- **Memory.** About 203 GB resident plus KV cache, so it needs a 256 GB Mac.
  mlx-serve sizes the context to what fits.
- **MTP.** The checkpoint's own `mtp.0` layer drafts by default (`--no-mtp`
  turns it off).
- **Thinking** is on by default (`"enable_thinking": false` turns it off).
  Tools use GLM's `<arg_key>` call format.
- **Images** go through the GLM vision tower (`<|image|>` rows).

## Conversion

The conversion scripts are a one-off layer-streamed pipeline and are not
published. It runs a reference forward per layer, collects router-weighted
per-expert Hessians, runs batched GPTQ, and writes the pack in oQ4-MTP's
layout. The MTP head and the vision tower are copied from oQ4-MTP. Peak memory
is about 60 GB; the run took 2.3 hours on an M5 Ultra.

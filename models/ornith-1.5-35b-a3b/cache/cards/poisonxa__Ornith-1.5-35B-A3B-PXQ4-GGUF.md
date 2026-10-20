---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags: [gguf, pxa, pxq, quantized, tensor-split, p100, v100, pascal, volta]
---

# Ornith-1.5-35B-A3B — PXQ4 (tensor-split ready)

A PXQ4 quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for
the [PXA engine](https://github.com/poisonxa16/pxa) on Tesla P100 / V100 / GTX 1080 Ti.

**Works with the tensor split (`-sm tensor`).** Every weight is PXQ4 except `ssm_out`, which stays at Q8_0: the
tensor split has to cut `ssm_out` along a dimension PXQ4 panels cannot be cut on, so a file with a PXQ4 `ssm_out`
is refused under `-sm tensor` (it still runs under `-sm layer`).

| | |
|---|---|
| File | `Ornith-1.5-35B-A3B-PXQ4.gguf`, 19,587,444,992 bytes |
| SHA256 | `d5d78d3ee7cce45ef88aa999796fdc6501e1bdf5a7ab55f2aedd62664af2cd1c` |
| Engine | PXA v2026.09.20 (commit `f2f1d965`) or later |

## Run it on two identical cards (tensor split)

```bash
PXA_TSPLIT_UNPROVEN_ARCH=1 PXA_TSPLIT_REDUCE=fused PXA_TSPLIT_REDUCE_PREFILL=1 \
llama-server -m Ornith-1.5-35B-A3B-PXQ4.gguf -ngl 99 -c 262144 -np 1 -fa on \
  -sm tensor -ts 1,1 -b 2048 -ub 2048 --jinja
```

`PXA_TSPLIT_UNPROVEN_ARCH=1` is needed because this architecture is not yet on the engine's measured list. This is
the exact configuration of our own seats: two Tesla P100 16 GB cards, full 262k context. Or let the launcher decide:
`./pxa-launch --model Ornith-1.5-35B-A3B-PXQ4.gguf --gpus 0,1`.

Community: [PXA Network Discord](https://discord.pxanetwork.com)

Expert-cache routing counts for PXA v2026.10 (lets the model run when it does not fit your cards); keep it next to the GGUF.

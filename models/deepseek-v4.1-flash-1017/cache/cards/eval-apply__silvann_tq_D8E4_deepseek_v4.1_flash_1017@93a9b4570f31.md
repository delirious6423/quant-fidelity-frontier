---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- silvann
- turboquant
- quantized
---

# silvann_tq_D8E4_deepseek_v4.1_flash_1017

DeepSeek-V4.1-Flash-1017 by DeepSeek, packed for the **SiLVANN** engine (the Eval-Apply GPU Interpreter and its neural
network library): every weight matrix rotated along its input axis with a blocked Hadamard-and-signs rotation and
quantised with a per-row scale (TurboQuant); norms, routers, the hyper-connection maps, the attention sinks and other
small tensors kept lossless; the embedding and head at 8 bits. `D8E4`: the dense part (attention, the indexer, the shared
experts) at 8 bits, the routed experts at 4 — the widths the original release already ships them at (FP8 and FP4),
coded again in SiLVANN's rotated format. The original release's 475.3 GiB in 457.5 GiB, 189.1 GiB of which are the
Engram tables, carried unchanged.

The format is `silvann-packed-v3` — `pack.json` describes the codec and the per-tensor width policy, `layers/NNN/` holds
each layer's records and `global.pkl` the embedding, head and final norm. `engram/` holds the Engram memory's tables:
the original release's two shards, byte for byte (their SHA-256 are the original repository's). The runtime reads them
from the disk a row at a time, as the n-gram memory asks for each, so they take no RAM. The pack is read by SiLVANN's
runtime, not by transformers. The tokenizer and configs are the original model's; the release has no Jinja chat
template, and SiLVANN's chat writes DeepSeek's own prompt encoding (thinking, the reasoning effort, tool calls) as the
release's `encoding.py` does, token for token.

Text only: the vision tower is not in this pack.

All credit for the model belongs to DeepSeek; this repository only changes how its weights are stored. The original
license (MIT) applies and is included.

## Checked against the original

SiLVANN runs this pack against the release's own `inference/model.py`, on the same weights:

- the whole model's next token is the reference's (" Rome", KL 0.0005 over the vocabulary);
- layer by layer over 1,300 positions — past the 128-position window, and past the point where the indexer starts
  choosing 512 of the compressed latents — each layer is held to the test's bounds against the reference in fp32
  (medians 1.3e-3 to 2.9e-3 from it): 34 of 35 checks pass, the one failure a router's near-tie, two of its scores
  8.3e-5 apart;
- a prompt read as rows, a chunk at a time, agrees with the same prompt a position at a time (KL 6e-11).

## What it needs

The layers are 267 GiB, 253 GiB of them the routed experts (6.3 GiB a layer). They run fastest from RAM; where RAM cannot hold them
all, SiLVANN reads the rest from the disk as they are picked. The Engram tables stay on the disk in every setup — an NVMe
disk serves them best. The settings and measured rates are in SiLVANN's documentation.

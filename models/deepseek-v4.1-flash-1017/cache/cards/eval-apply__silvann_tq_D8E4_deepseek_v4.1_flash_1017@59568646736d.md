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
coded again in SiLVANN's rotated format. The original release's 475.3 GiB in 457.9 GiB, 189.1 GiB of which are the
Engram tables, carried unchanged.

The format is `silvann-packed-v3` — `pack.json` describes the codec and the per-tensor width policy, `layers/NNN/` holds
each layer's records and `global.pkl` the embedding, head and final norm. `engram/` holds the Engram memory's tables:
the original release's two shards, byte for byte (their SHA-256 are the original repository's). The runtime reads them
from the disk a row at a time, as the n-gram memory asks for each, so they take no RAM. The pack is read by SiLVANN's
runtime, not by transformers. The tokenizer and configs are the original model's; the release has no Jinja chat
template, and SiLVANN's chat writes DeepSeek's own prompt encoding (thinking, the reasoning effort, tool calls) as the
release's `encoding.py` does, token for token.

The vision tower is in `vision.pkl`: the release's 32-block ViT and its aligner (the matrices at 8 bits, the patch
embedding, norms and biases lossless), the three learned rows of a picture's span, and every layer's `bias_vl` — the
router's bias for a picture's tokens. SiLVANN runs the tower on the card and prepares a picture as the release's
`inference/image_processor.py` does; a picture is a span of up to 1,024 tokens, routed by `bias_vl` and left out of the
n-gram memory, as the release's `inference/model.py` reads it.

All credit for the model belongs to DeepSeek; this repository only changes how its weights are stored. The original
license (MIT) applies and is included.

## Checked against the original

SiLVANN runs this pack against the release's own `inference/model.py`, on the same weights:

- the whole model's next token is the reference's (" Rome", KL 0.0005 over the vocabulary);
- layer by layer over 1,300 positions — past the 128-position window, and past the point where the indexer starts
  choosing 512 of the compressed latents — each layer is held to the test's bounds against the reference in fp32
  (medians 1.3e-3 to 2.9e-3 from it): 34 of 35 checks pass, the one failure a router's near-tie, two of its scores
  8.3e-5 apart;
- a prompt read as rows, a chunk at a time, agrees with the same prompt a position at a time (KL 6e-11);
- a picture: the tower on the card is nearer the release's float32 than the release's own bfloat16 at every stage
  (its output 3.1e-3 from float32, bfloat16 2.6e-2), and a prompt with the picture gives the release's next token
  ("Based", each of its top five within 0.03, KL 0.006);
- past 16,384 positions, where layer 20's candidate blocks choose where the later layers look: the candidates are
  the release's `select_candidate_blocks` to the latent at 17,300, and a 17,283-position prompt answers as the
  release does (the next token "11", the ledger's answer, KL 0.0000 over the vocabulary).

## What it needs

The layers are 267 GiB, 253 GiB of them the routed experts (6.3 GiB a layer). They run fastest from RAM; where RAM
cannot hold them all, SiLVANN reads the rest from the disk as they are picked. The Engram tables stay on the disk in every
setup — an NVMe disk serves them best. Pictures and the context past 16,384 positions need SiLVANN v0.17.0 or later; the
tower takes about 0.6 GB of the card, and the context is the model's own, 1,048,576 positions. The settings and measured
rates are in SiLVANN's documentation.

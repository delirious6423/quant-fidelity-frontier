---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: image-text-to-text
tags:
  - gguf
  - rocmfpx
  - magicquant
  - moe
  - multimodal
---

# Ornith-1.5-35B-A3B — ROCmFPX renders

MagicQuant's per-group scheme assignments for
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
rendered into **ROCmFPX tensor types** (`q*_0_rocmfpx`), a quant family aimed
at AMD ROCm inference. Same search that produced the
[MagicQuant GGUFs](https://huggingface.co/lmcoleman/Ornith-1.5-35B-A3B-MagicQuant-GGUF),
different render target.

Pins: MagicQuant `ed17991`, Foundry `2f99202`.

## ⚠ Engine requirement — read before downloading

These files use ROCmFPX tensor types that **stock llama.cpp cannot load.**
They require a ROCmFPX fork build. A build that is too old fails with
`invalid ggml type` on load; the types are not a stock-llama.cpp feature and
no amount of rebuilding mainline will help. If you are not already running a
ROCmFPX fork, you want the MagicQuant repo linked above instead.

## Files

| file | size |
|---|---|
| `Ornith-1.5-35B-A3B-ROCMFPX-MQ-Q4.gguf` | 20.50 GiB |
| `Ornith-1.5-35B-A3B-ROCMFPX-MQ-Q6.gguf` | 28.57 GiB |
| `mmproj-Ornith-1.5-35B-A3B-f16.gguf` | 0.86 GiB |

Both passed the pipeline's perplexity smoke gate (6.55 and 6.50 respectively).

**Those smoke numbers are not comparable to the MagicQuant repo's PPL table.**
The smoke gate is a short run against a different corpus path than the search
used, so the scale differs for instrument reasons, not quality reasons. **No
matched-instrument comparison between these ROCmFPX renders and their
MagicQuant counterparts has been run**, so this repo makes no claim about
which is better. If that comparison matters to you, measure both yourself
under one instrument.

## What is deliberately absent

- **No Q5.** The pipeline's band guard refused it: rendering MagicQuant's Q5
  configuration into ROCmFPX types predicts 26.87 GiB against a 66.13 GiB
  baseline — a 0.406 ratio, which is the **Q6** band, not Q5. Publishing it as
  "Q5" would have mislabelled it, so it was not built.
- **No `Q6_0_ROCMFPX_AGENT` variant.** It built, then **segfaulted** under
  perplexity evaluation (`exited -11`). The pipeline quarantined it
  automatically and aborted before upload. It is a crash, not a quality miss,
  and it is not published because it does not work.

Both absences are disclosed rather than quietly dropped, which is the point.

## Notes

- Vision requires the `mmproj` file alongside the model.
- Chat template embedded and verified present before upload; vocab metadata
  (`tokenizer.ggml.token_type`) is INT32 per spec.
- Built and measured on an AMD Strix Halo (Ryzen AI MAX+ 395, gfx1151) host.

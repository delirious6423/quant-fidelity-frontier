---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - pnx
  - tensorless
  - qwen3.8
  - tesla-p100
---

# Qwen3.8-27B-PN3 (GGUF, PNX format)

**Join us on [Discord](https://discord.gg/3Qs2uejUf9)**: questions, results from your own P100, and ideas.

A 3-bit-class quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the **PNX** format,
built for fast decoding on a single **NVIDIA Tesla P100 (sm_60)**. 12,354 MiB. Text only. It includes the model's
MTP (next-token prediction) block for speculative decoding.

> [!WARNING]
> **This model runs only in [tensorless](https://github.com/iammrduncan/tensorless). llama.cpp cannot run it**, and
> neither can Ollama, LM Studio or any other llama.cpp-based runtime. The file is a GGUF container, but its weights use
> custom tensor types (PN3T = 124, PN4T = 125) in a Hadamard-rotated basis that only tensorless implements.
> The Hugging Face GGUF viewer may also fail to parse it.

## Files

| File | Size | sha256 |
|---|---:|---|
| `Qwen3.8-27B-PN3-P0k19.gguf` | 12,954,114,496 bytes (12,354.0 MiB) | `15ce2d4469673f1f1753c55de3780cfe04311dd6f05cb41633d73df128e91476` |
| `pnx-manifest.json` | small | machine-readable description for engines (format, required types, sha256, provenance) |
| `SHA256SUMS` | small | `sha256sum -c SHA256SUMS` |

## Quality

All rows below were scored in the same harness: an exact fp32 forward pass using each file's dequantized weights,
compared against the fp32 forward pass of the original BF16 weights. Only the weights differ between rows.
Corpus: wikitext + code + chat, 401,626 scored positions. KL is the mean full-vocabulary KL(reference ‖ model) in nats.

| Model | Size (MiB) | Mean KL ↓ | Top-1 agreement ↑ | Perplexity vs BF16 |
|---|---:|---:|---:|---:|
| BF16 weights, bf16 math | 52,126 | 0.0031 | 99.33% | +0.15% |
| Qwen3.8-27B-PN4 (4-bit sibling, not published) | 13,907 | 0.0473 | 93.00% | +3.01% |
| **Qwen3.8-27B-PN3 (this)** | **12,354** | **0.0935** | **89.92%** | **+4.17%** |
| A1 (Q3F-A1, our earlier 3-bit P100 build) | 11,214 | 0.0931 | 90.37% | +5.87% |
| Unsloth UD-IQ3_XXS | 10,428 | 0.1065 | 89.48% | +6.22% |
| Unsloth UD-IQ4_XS | 13,593 | 0.0383 | 94.41% | +0.92% |

**Structured output and tool calling** (1,500 items from BFCL, JSONSchemaBench, StructEval and IFEval; greedy;
run in vLLM bf16 with the dequantized weights): **86.47%** pass rate against **86.67%** for the BF16 model, a net
drop of 0.20 points (95% CI −0.87 to +1.27).

**Long context:** tested at depths up to 64K tokens on a long code document; KL does not grow with depth
(fitted slope −0.017 per doubling, 95% upper bound +0.055).

## Running it

Run it with [tensorless](https://github.com/iammrduncan/tensorless), the custom engine for the Tesla P100 that this
format was built for. Its kernels were tuned on this exact file (sha256 `15ce2d44…`). llama.cpp will not load it.

The chat template and tokenizer are embedded in the GGUF (taken unchanged from the BF16 conversion).
Recommended sampling, from the GGUF: temperature 1.0, top_p 0.95, top_k 20.

### Download and verify

```sh
hf download iamMrDuncan/Qwen3.8-27B-PN3-GGUF --local-dir Qwen3.8-27B-PN3-GGUF
cd Qwen3.8-27B-PN3-GGUF && sha256sum -c SHA256SUMS
```

```python
from huggingface_hub import hf_hub_download
manifest = hf_hub_download("iamMrDuncan/Qwen3.8-27B-PN3-GGUF", "pnx-manifest.json")
gguf     = hf_hub_download("iamMrDuncan/Qwen3.8-27B-PN3-GGUF", "Qwen3.8-27B-PN3-P0k19.gguf")
```

An engine that downloads the model on startup should read `pnx-manifest.json` first. It names the GGUF, its size
and sha256, and the tensor types the engine must support (`format.required_ggml_types`).

## Format

| | |
|---|---|
| Spec | PNX-FORMAT-v0 (`pnx.spec`), `general.file_type` = 1024 (no llama_ftype value) |
| PN tensors | the 400 weight matrices of the 64 main layers: **221 × PN3T + 179 × PN4T**, chosen per tensor |
| PN3T (type 124) | blocks of 80 weights in 34 bytes (3.4 bits per weight): fp16 scale + 8 little-endian u32 words, 10 signed mid-tread 3-bit codes per word |
| PN4T (type 125) | blocks of 64 weights in 34 bytes (4.25 bits per weight): fp16 scale + 8 u32 words, 8 signed 4-bit codes per word (the Q4_0 grid) |
| Rotation | all 400 PN tensors are stored in a rotated basis: normalized Sylvester–Walsh–Hadamard, block 256, along the input (last) dimension, with explicit ±1 signs for widths 5120 / 6144 / 17408 (`prism.hadamard.*` keys; the rotated tensors are listed in `prism.hadamard.weight_names`) |
| Stock tensors | token embedding Q3_K, output head Q4_K, SSM alpha/beta Q8_0, the MTP block (blk.64) Q8_0 / Q6_K, norms F32 |

## How it was made

GPTQ (damping 0.03, activation order on unrotated tensors only) on a Hadamard-rotated basis, with a fixed
calibration set (`pnx.calib.sha256`). The PN3/PN4 split per tensor (composition `P0k+19`) was chosen by
measured sensitivity to stay at the A1 build's quality while fitting the P100's memory budget at 65K context.
Re-packing the stored quantizer output reproduces this file's sha256 exactly. Quantizer commit
`64dcc0c93f78edf9a4e82e65747ce0650add5981`. The full research record (gate report, decisions, scripts) is in a
private repository for now.

## Caveats

- **Text only.** The vision encoder of Qwen3.8-27B is not included.
- The quality numbers measure the weights, not an engine. An emulation of the P100 kernels' fp16 arithmetic
  added a mean KL of 1.3e-3 nats over 768 test positions; 93% of that came from one position in a prose document
  where every 3- and 4-bit file departs from BF16.
- Top-1 agreement is 0.45 points below A1 (0.82 points on wikitext), while its KL is the same as A1's and its
  perplexity is lower.
- Experimental research release.

## License

Apache-2.0, the license of the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

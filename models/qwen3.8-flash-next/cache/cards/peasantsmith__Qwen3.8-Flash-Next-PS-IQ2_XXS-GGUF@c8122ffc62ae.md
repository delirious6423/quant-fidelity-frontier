---
license_name: qwen-community-1.0
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- quantized
- moe
- iq2_xxs
- qwen3.8
- imatrix
- conversational
- endpoints_compatible
library_name: gguf
---

# Qwen3.8-Flash-Next PS-IQ2_XXS (Peasant Smith)

**176.9B parameters in a single 75.2 GB file — 3.40 bits/param.** An IQ2_XXS-class GGUF of **Qwen3.8-Flash-Next**, built from the original `Qwen/Qwen3.8-Flash-Next` BF16 weights, with every component quantized to the precision it needs instead of crushed flat.

**Please read**: After running it for days non-stop I have concluded that this quantization is pretty good, but it does require certain care. 
I recommend running it with Thinking: Disabled or Low. You can definitely use xHigh or Medium, but I won't recommend them on hours of uninterrupted agentic work since you may risk loops or simply wasting time on overthinking. For short/medium tasks using thinking as xHigh is completely fine.

## How the size was won

The model's single largest component is a **per-layer-embedding (PLE) n-gram table: 51.2B params (~102 GB in BF16) — roughly 28% of the model**. A naive 2-bit quant crushes it along with everything else, making the model's biggest component its weakest. This build takes the 2-bit win where it's safe and holds the PLE table at the quality floor the ecosystem agrees on:

| Model component | Params | Quant | Size |
|---|---|---|---|
| PLE n-gram table | 51.2B | **IQ4_NL** (4.5 bpw) | 26.8 GiB |
| Routed expert gate/up | — | **IQ2_XXS** (2.06 bpw) | 19.3 GiB |
| Routed expert down | — | **Q4_0** (4.5 bpw) | 21.1 GiB |
| Attention (QKV) | — | Q4_K | 0.5 GiB |
| Token embeddings | — | Q6_K | 0.5 GiB |
| Output head | — | Q8_0 | 0.8 GiB |
| Routers / norms / biases / hash constants | — | **F32 (kept)** | — |
| Vision encoder (`mmproj`, 334 tensors) | — | F16 | 0.9 GiB |

- **Expert down-projections sit a full step above gate/up** (Q4_0 vs IQ2_XXS): their errors land directly and additively on the residual stream, while gate/up errors pass through activations first.
- **Every router, norm, bias, and hash-constant tensor stays F32** — the parts that must stay high precision aren't compressed at all.
- **Imatrix:** unsloth's published importance matrix for this model, audited to cover every quantized tensor (hyper-connection and PLE tensors pinned out of IQ quantization).

## Testing

| Test | Result |
|---|---|
| Full-context generation (64k context window) | ✅ coherent at depth, **no degradation** |
| Long-generation stability | ✅ ≥3000 tokens continuous at steady t/s, no degradation |
| Tool calling (`get_weather` JSON function call) | ✅ well-formed `{"city":"Paris"}` tool call returned |
| Long-context recall (~6k token prompt, needle) | ✅ number recalled correctly |
| LRU cache code task (7.5k token prompt) | ✅ complete, correct implementation |
| Greedy smoke ("The capital of France is") | ✅ "Paris" (`finish=stop`) |
| Vision via mmproj (image color ID) | ✅ correct |
| Perplexity (wikitext-2) | not measured — honesty note below |

**Honesty note:** PPL was not run — at this weight class the BF16 reference does not fit the same GPU stack, so no comparable baseline exists on this hardware. The task battery above is the quality evidence; 2-bit experts do carry a real quality delta vs BF16 by construction.

## Files

| File | Size | Notes |
|---|---|---|
| `Qwen3.8-Flash-Next-IQ2_XXS.gguf` | 75.2 GB | text model (176.9B params) |
| `mmproj-Qwen3.8-Flash-Next-F16.gguf` | 0.90 GB | vision encoder, F16, all 334 tensors — nothing stripped |

## Usage

Generic llama.cpp invocation — tune the two offload values to your own hardware:

```bash
llama-server \
  -m Qwen3.8-Flash-Next-IQ2_XXS.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-F16.gguf \
  -ngl <layers that fit your VRAM> \
  --n-cpu-moe <experts to spill to CPU RAM if VRAM is tight> \
  --jinja
```

The model ships its chat template (`--jinja`). Requires a recent llama.cpp with qwen4exp architecture support.

## Provenance & credits

1. `Qwen/Qwen3.8-Flash-Next` — official BF16 safetensors, original weights (no third-party quant or modified repo as source). **Model credit: Qwen.**
2. → GGUF master via llama.cpp `convert_hf_to_gguf.py`
3. → `llama-quantize` importance-matrix quant with the per-component split above. **Imatrix credit: unsloth.**

## Verify

SHA-256 (text): `2e0f14e7eeddce8f80fc88cf96a9cc641b4f60549318fb53e30af85649883586`
SHA-256 (vision): `f5f125bdfa3997011693e7fe59fae1a5c068c40a627a0576d72734a9cdab9691`

## Limitations

- 2-bit experts: a real quality delta vs BF16 exists by construction — the task battery passed, but long-tail knowledge degradation is expected at this footprint
- Text model only: the vision tower ships as the separate F16 `mmproj` file
- Requires a recent llama.cpp (qwen4exp architecture support)
- MTP head not embedded — left as shipped by Qwen
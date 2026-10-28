---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: gguf
tags:
- gguf
- glm5-next
- talus
- strix-halo
- gfx1151
---

# GLM-5.3-Flash a4e3s (GGUF)

A mixed 2-4-bit GGUF of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), sized for a 128 GB
AMD Strix Halo machine (Ryzen AI Max+ 395, Radeon 8060S). It is the GLM-5.3-Flash weight file that
[Talus](https://github.com/nsbradley88/talus-public) ships with.

It is a requantization of Unsloth's
[GLM-5.3-Flash UD-IQ3_XXS](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/main/UD-IQ3_XXS). The routed
experts keep Unsloth's quantization byte for byte (with three exceptions, below). The attention, KDA, shared-expert
and dense-FFN tensors are re-quantized to make room for the context and the drafter on a 128 GB machine.

## Files

| File | Size | SHA-256 |
|---|---|---|
| `GLM-5.3-Flash-a4e3s-00001-of-00004.gguf` | 9.43 MB (metadata only) | `55ee18988a0f016b6a4d7ee4d30daf137931542eae8d9ecedc782a80800aee2d` |
| `GLM-5.3-Flash-a4e3s-00002-of-00004.gguf` | 47.62 GB | `8501ae69de47a410964bd32a9b1e0d1cad440691868cdbaaae77b84d573fc92b` |
| `GLM-5.3-Flash-a4e3s-00003-of-00004.gguf` | 48.37 GB | `5eb5a47445f3dcca7a9a8ed6e756815e77e4e46330ef4d1bacf596a77198a744` |
| `GLM-5.3-Flash-a4e3s-00004-of-00004.gguf` | 21.53 GB | `dc52cfc611ee12928e054934702df5062ed2f705c91666a164e87b1ed35fb1cc` |
| `recipe/glm-a4.imatrix` | 1.03 GB (importance matrix, GGUF format) | `b59e539c4ae0b5c4bdd3ba295249795e535065e60897fd329711da380b6f2f10` |

Every file, including the recipe, is listed in `SHA256SUMS`. Point Talus or llama.cpp at the first shard; the
others are found next to it.

The model's NextN (MTP) layer is in the file (layer 45); Talus uses it as the speculative drafter. For Talus's
64K draft vocabulary see `glm-dv-65536.bin` in [j2ipper/draft-vocab](https://huggingface.co/j2ipper/draft-vocab).

## Tensor types

1,412 tensors, 109.45 GiB.

| Tensors | Type | Notes |
|---|---|---|
| Routed experts `ffn_gate_exps`, `ffn_up_exps` (layers 3-44) | IQ2_S | Unsloth's; layer 11 IQ3_S |
| Routed experts `ffn_down_exps` (layers 3-44) | IQ3_S | Unsloth's, except layers 11, 12, 44 (IQ4_XS -> IQ3_S, step 2) |
| NextN layer 45 experts | Q2_K (gate, up), Q3_K (down) | Unsloth's |
| Attention `attn_q/k/v/output`, `attn_q_a` (KDA and MLA layers) | Q4_K | from Q6_K (step 1); Q6_K in layer 45 (kept) and layer 11 (from Q8_0, step 3) |
| MLA `attn_q_b` | Q6_K | from Q8_0 (step 3) |
| MLA `attn_k_b`, `attn_v_b`, `attn_kv_a_mqa`, KDA `ssm_*` projections, DSA indexer, `hc_*` | Q8_0 | Unsloth's |
| Shared expert `ffn_*_shexp`, dense FFN (layers 0-2) | Q4_K | from Q6_K (step 1); Q6_K in layer 45 (kept) and layer 11 (from Q8_0, step 3) |
| `nextn.eh_proj` | Q6_K | from Q8_0 (step 3) |
| Router `ffn_gate_inp`, norms, biases | F32 | |
| `output` | Q5_K | step 3 |
| `token_embd` | Q4_K | step 3 |

By type: IQ2_S 59.10 GiB, IQ3_S 42.54 GiB, Q4_K 3.89 GiB, Q2_K 1.48 GiB, Q3_K 0.97 GiB, Q8_0 0.46 GiB, Q6_K 0.41 GiB, Q5_K 0.41 GiB, F32 0.21 GiB.

## Recipe

Three steps with llama.cpp's `llama-quantize --allow-requantize --keep-split`, each with a per-tensor type file
(`--tensor-type-file`) that pins every tensor not listed below to its current type, so those tensors are copied byte
for byte. The base type argument is `Q4_K_M`; the type file overrides it for every tensor.

1. **a4**, from Unsloth's UD-IQ3_XXS: attention / KDA projections, shared experts and dense FFN Q6_K -> Q4_K
   (except layers 11 and 45); `--output-tensor-type q6_k --token-embedding-type q6_k`. No importance matrix.
2. **a4e3**, from a4: the three IQ4_XS `ffn_down_exps` (layers 11, 12, 44) -> IQ3_S, with an importance matrix computed
   on a4 (`llama-imatrix -c 512 -b 512 --chunks 150`).
3. **a4e3s**, from a4e3: MLA `attn_q_b`, layer 11 `attn_output`, `attn_q_a` and `ffn_*_shexp`, and `blk.45.nextn.eh_proj`
   Q8_0 -> Q6_K, `output` -> Q5_K and `token_embd` -> Q4_K; same importance matrix. MLA `attn_k_b`, `attn_v_b`,
   `attn_kv_a_mqa`, the KDA `ssm_*`, the `hc_*` and the DSA indexer tensors stay Q8_0.

`recipe/` holds the three type files (`glm-tt-a4.txt`, `glm-tt-a4e3.txt`, `glm-tt-a4e3s.txt`), the script that
generates them from the UD-IQ3_XXS files (`make_glm_a4e3s_types.py`) and the importance matrix (`glm-a4.imatrix`).
The importance matrix was computed on a private calibration text of technical English prose and code; the text is not
published. It only affects the tensors re-quantized in steps 2 and 3.

## Use it with Talus

Build Talus ([Quick start](https://github.com/nsbradley88/talus-public#quick-start)), download this repository and
`glm-dv-65536.bin`, extract the tokenizer as the Quick start shows, and serve with a run config like this one
(the arguments are Talus's shipped defaults for this model, from
[docs/MODELS.md](https://github.com/nsbradley88/talus-public/blob/main/docs/MODELS.md)):

```json
{
  "exe": "rust/target/release/talus-cli",
  "args": ["--model", "/models/GLM-5.3-Flash-a4e3s-00001-of-00004.gguf",
           "--ctx", "65536",
           "--draft-vocab", "/models/glm-dv-65536.bin"],
  "tokenizer": "/models/glm-5.3-flash/tokenizer",
  "model_name": "glm-5.3-flash",
  "stop_tokens": ["<|user|>", "<|observation|>", "<|endoftext|>"],
  "env": {},
  "lib_dirs": ["/path/to/rocm/lib", "/path/to/rocm/lib/llvm/lib"]
}
```

```bash
STRATA_API_KEY=<choose-a-key> rust/target/release/talus-server --config glm.json --host 0.0.0.0 --port 8000
```

It also loads in llama.cpp builds that support the `glm5-next` architecture.

## Quality

Measured by the Talus freeze run `rel-final-e6cf` (Talus commit `e6cf959e1cad`,
2026-10-28); methods in Talus's
[docs/METHODOLOGY.md](https://github.com/nsbradley88/talus-public/blob/main/docs/METHODOLOGY.md):

| Gate | Result |
|---|---|
| KL divergence vs the original FP8 model | KLD 0.1960, top-1 90.2% on the gate text; code: KLD 0.2584, top-1 90.1% (vs the original FP8 weights) |
| KL divergence vs the reference logits of this quant's lineage | KLD 0.04182, top-1 95.2% (gate <= 0.0440) |
| Needle in a haystack, 60K | 15/15 |
| Agentic coding suite | 5/5 |

Speed on Strix Halo: Talus's [docs/RESULTS.md](https://github.com/nsbradley88/talus-public/blob/main/docs/RESULTS.md).

## License and credits

MIT, the license of the base model. The full license text is in `LICENSE`.

- Base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), Copyright (c) 2026 Z.AI Co.,
  Ltd, MIT License.
- Source quantization: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (MIT); most
  tensors in this file are Unsloth's, unchanged.
- Quantized with [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).

---
license: mit
library_name: gguf
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: schlaflos
language:
  - en
tags:
  - gguf
  - quantized
  - deepseek
  - deepseek-v4
  - deepseek-v4-flash
  - moe
  - mixture-of-experts
  - 2-bit
  - 4-bit
  - iq2_xxs
  - q2_k
  - q4_k
  - ds4
  - apple-silicon
  - metal
---

# DeepSeek V4 Flash 0731 — GGUF for ds4

> **Status: the GGUF is still uploading.** This card is published ahead of the weights.

> **These quants target the DS4 inference engine, not llama.cpp.**
> https://github.com/antirez/ds4
>
> They use DS4's `deepseek4` tensor layout and metadata. Compatibility with
> llama.cpp, Ollama or LM Studio is **untested** — do not assume it works.
> Conversely, llama.cpp-format quants of this model (MXFP4 experts, BF16
> tensors) **cannot** be loaded by ds4: those types are not in DS4's tensor
> type set.

Built from the official [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
release, which supersedes the V4 Flash preview.

## Files

| File | Size | Routed experts | Everything else |
|---|---:|---|---|
| `DeepSeek-V4-Flash-0731-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-imatrix.gguf` | 90.9 GiB | layers 0–36: `IQ2_XXS` (gate/up) + `Q2_K` (down); layers 37–42: `Q4_K` (all three) | `Q8_0` attn proj / shared experts / output, `F16` router + embed + indexer + compressor + HC, `F32` norms / sinks / bias |

Sized for 128 GB Apple Silicon.

## Quantization recipe

The filename is the spec.

| Tensor class | Quant | Count |
|---|---|---:|
| `blk.{0..36}.ffn_{gate,up}_exps` | `IQ2_XXS` | 74 |
| `blk.{0..36}.ffn_down_exps` | `Q2_K` | 37 |
| `blk.{37..42}.ffn_{gate,up,down}_exps` | `Q4_K` | 18 |
| `blk.*.attn_{q_a,q_b,kv,output_a,output_b}` | `Q8_0` | 215 |
| `blk.*.ffn_{gate,up,down}_shexp` | `Q8_0` | 129 |
| `output.weight` | `Q8_0` | 1 |
| `token_embd.weight`, `blk.*.ffn_gate_inp` | `F16` | |
| `blk.*.attn_compressor_*`, `indexer_*`, `hc_*` | `F16` / `F32` | |
| `*_norm.weight`, `attn_sinks`, `exp_probs_b` | `F32` | |
| `blk.{0,1,2}.ffn_gate_tid2eid` | `I32` | 3 |

Routed experts are ~91% of the payload but each expert sees only a fraction of
tokens, so aggressive quantization there costs less average quality than the
same treatment of router, projections or shared experts. The last six layers
keep `Q4_K` because late-layer error affects output disproportionately.

## Imatrix provenance

Quantized with antirez's published importance matrix
(`DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat`), which was collected on
the **V4 Flash preview**, not on 0731.

The architecture, tokenizer and routing structure are byte-identical between the
two releases, so the statistics apply dimensionally and should transfer well — an
imatrix records which input columns carry activation energy, not the weight
values themselves. But this has not been measured across the release boundary. A
0731-native imatrix may improve on it.

## Notes on the 0731 release

0731 bundles the DSpark speculative-decoding module (`mtp.0/1/2`) into the same
checkpoint, where the preview shipped it as a separate repository. A matching
DSpark support GGUF can be built with
`deepseek4-quantize --dspark-support`.

ds4's reasoning-effort prompt predates this release: `--think max` emits the
prefix DeepSeek now labels `high`, and the new `max` tier has no corresponding
string in the runtime. This affects prompt construction only, not the weights.

## Usage

```bash
git clone https://github.com/antirez/ds4 && cd ds4 && make

hf download schlaflos/DeepSeek-V4-Flash-0731-antirez-ds4-GGUF \
  DeepSeek-V4-Flash-0731-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-imatrix.gguf \
  --local-dir gguf

ln -sfn "$PWD/gguf/DeepSeek-V4-Flash-0731-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-imatrix.gguf" ds4flash.gguf

./ds4 -p "Explain Redis streams in one paragraph."
```

## License

MIT. Base model copyright DeepSeek; GGUFs redistributed under the base model's
release terms.

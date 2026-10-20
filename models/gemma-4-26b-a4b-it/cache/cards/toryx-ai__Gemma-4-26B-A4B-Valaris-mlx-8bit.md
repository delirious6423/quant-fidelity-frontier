---
language: en
license: gemma
library_name: mlx
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
tags:
- mlx
- gemma4
- moe
- multimodal
- vision
- audio
- text-generation
- conversational
---

# GemmaValaris-26B (Gemma-4-26B-A4B · MLX 8-bit)

GemmaValaris-26B is a fine-tune of Gemma-4-26B-A4B, serving as the grounded reader in the MOSAIC pipeline for multi-hop question answering and retrieval-augmented reasoning. This build is quantized to 8-bit MLX for on-device inference on Apple Silicon (Mac). Built with Gemma.

Built on **Gemma-4-26B-A4B** (multimodal Mixture-of-Experts: 128 experts, 30 layers, hidden size 2816). This is the **8-bit** build — the higher-fidelity sibling of the 4-bit build (`toryx-ai/Gemma-4-26B-A4B-Valaris-mlx-4bit`).

- **Author:** Luis M Sanchez, Founder, Toryx AI — [toryx.ai](https://toryx.ai)
- **Base model:** [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (instruction-tuned variant; confirmed from the training `adapter_config.json`). Fine-tuned via SFT.
- **Format:** MLX (Apple `mlx` / `mlx-vlm`), 6 safetensors shards (~28 GB on disk)

## Quantization

Converted from the merged bf16 checkpoint with `mlx_vlm.convert`:

```bash
mlx_vlm.convert --hf-path <merged-bf16> -q --q-bits 8 --q-group-size 64 \
--mlx-path ToryxValaris-26B-A4B-mlx-8bit
```

- **8-bit** weights, **group size 64**, affine mode
- Effective **~8.67 bits/weight**
- Trades ~13 GB more disk/RAM than the 4-bit build for higher numerical fidelity

## Chat template

This fine-tune uses a custom chat template with a dedicated `<|channel>thought ... <channel|>` **reasoning channel** (preserved in `chat_template.jinja`). Use the bundled template rather than a generic Gemma template.

## Usage

### MLX (mlx-vlm)

```bash
pip install -U mlx-vlm
mlx_vlm.generate --model toryx-ai/Gemma-4-26B-A4B-Valaris-mlx-8bit \
--prompt "What is the capital of Japan?" --max-tokens 64
```

### oMLX (local server, Apple Silicon)

Drop the model dir under `~/.omlx/models/Toryx/`, reload (`POST /admin/api/reload`), then call the OpenAI-compatible API with model id `ToryxValaris-26B-A4B-mlx-8bit`.

## Training

Fine-tuned from Gemma-4-26B-A4B as the grounded reader in the MOSAIC pipeline. Further training details are proprietary.

- **Reproducibility:** the LoRA `adapter_config.json` and full `trainer_state.json` are under [`training/`](./training) in this repo.

## Evaluation

As reported in Sanchez & Dehnad (2026), *Selling the Meter, Proving the Outcome*: on a reader bake-off using the MuSiQue multi-hop benchmark (N=500, identical retrieved top-5 context across all readers), this model scores **49.2 Answer F1** [45.3, 53.1] — ahead of a 70B cloud model (46.4) and within ~3.5 F1 of the frontier trio (Opus 4.8, GPT-5.5, Gemini-3.5), at an estimated ~$0 marginal cost on a single on-prem appliance. Full methodology: *Selling the Meter, Proving the Outcome* (Sanchez & Dehnad, 2026). F1 was measured on the bf16 merged checkpoint; the quantized builds in this repo have not been separately evaluated.

## Related repos (Toryx KMS)

- `toryx-ai/Gemma-4-26B-A4B-Valaris-mlx-4bit` — the 4-bit sibling of this build (compact, ~15.3 GB)
- `toryx-ai/KMS-multihop-retrieval-eval` (dataset) — evaluation objects; public release accompanies the paper

## License

Governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). This is a derivative of Gemma-4 and inherits those terms.

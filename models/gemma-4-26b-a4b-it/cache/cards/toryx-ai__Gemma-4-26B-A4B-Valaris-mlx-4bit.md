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

# GemmaValaris-26B (Gemma-4-26B-A4B · MLX 4-bit)

GemmaValaris-26B is a fine-tune of Gemma-4-26B-A4B, serving as the grounded reader in the MOSAIC pipeline for multi-hop question answering and retrieval-augmented reasoning. This build is quantized to 4-bit MLX for on-device inference on Apple Silicon (Mac). Built with Gemma.

Built on **Gemma-4-26B-A4B** (multimodal Mixture-of-Experts: 128 experts, 30 layers, hidden size 2816). This is the **4-bit** build — the compact sibling of the 8-bit build (`toryx-ai/Gemma-4-26B-A4B-Valaris-mlx-8bit`).

- **Author:** Luis M Sanchez, Founder, Toryx AI — [toryx.ai](https://toryx.ai)
- **Base model:** [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (instruction-tuned variant; confirmed from the training `adapter_config.json`). Fine-tuned via SFT.
- **Format:** MLX (Apple `mlx` / `mlx-vlm`), 3 safetensors shards (~15.3 GB on disk)

## Quantization

Converted from the merged bf16 checkpoint with `mlx_vlm.convert`:

```bash
mlx_vlm.convert --hf-path <merged-bf16> -q --q-bits 4 --q-group-size 64 \
  --mlx-path ToryxValaris-26B-A4B-mlx-4bit
```

- **4-bit** weights, **group size 64**, affine mode
- **MoE router projections kept at 8-bit** (all 30 layers) to preserve expert-routing fidelity
- Effective **~4.76 bits/weight**

## Chat template

This fine-tune uses a custom chat template with a dedicated `<|channel>thought ... <channel|>` **reasoning channel** (preserved in `chat_template.jinja`). Use the bundled template rather than a generic Gemma template.

## Usage

### MLX (mlx-vlm)

```bash
pip install -U mlx-vlm
mlx_vlm.generate --model toryx-ai/Gemma-4-26B-A4B-Valaris-mlx-4bit \
  --prompt "What is the capital of Japan?" --max-tokens 64
```

### oMLX (local server, Apple Silicon)

Drop the model dir under `~/.omlx/models/Toryx/`, reload (`POST /admin/api/reload`), then call the OpenAI-compatible API:

```bash
curl http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "ToryxValaris-26B-A4B-mlx-4bit",
  "messages": [{"role": "user", "content": "What is the capital of Japan?"}]
}'
```

## Performance (Apple M3 Pro, 36 GB)

- ~**49 tokens/sec** generation, **~15.4 GB** peak memory

## Training

Fine-tuned from Gemma-4-26B-A4B as the grounded reader in the MOSAIC pipeline. Further training details are proprietary.

- **Reproducibility:** the LoRA `adapter_config.json` and full `trainer_state.json` are under [`training/`](./training) in this repo.

## Evaluation

As reported in Sanchez & Dehnad (2026), *Selling the Meter, Proving the Outcome*: on a reader bake-off using the MuSiQue multi-hop benchmark (N=500, identical retrieved top-5 context across all readers), GemmaValaris-26B scores **49.2 Answer F1** [45.3, 53.1] — ahead of a 70B cloud model (46.4) and within ~3.5 F1 of the frontier trio (Opus 4.8, GPT-5.5, Gemini-3.5), at an estimated ~$0 marginal cost on a single on-prem appliance. F1 was measured on the bf16 merged checkpoint; the quantized builds in this repo have not been separately evaluated.

## Related repos (Toryx KMS)

- `toryx-ai/Gemma-4-26B-A4B-Valaris-mlx-8bit` — the 8-bit sibling of this build (higher fidelity, ~28 GB)
- `toryx-ai/KMS-multihop-retrieval-eval` (dataset) — evaluation objects; public release accompanies the paper

## License

Governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). This is a derivative of Gemma-4 and inherits those terms.

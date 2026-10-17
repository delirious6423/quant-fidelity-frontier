---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
- huihui-ai/Huihui-Qwen3.8-27B-abliterated
- Qwen/Qwen3.8-27B
tags:
- qwen3.8
- qwen3.5
- autoround
- int4
- w4a16
- abliterated
- multimodal
- safetensors
---

# Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound

This is the SafeTensors INT4 AutoRound version of [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated), which is based on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

The artifact is multimodal: it contains the Qwen3.5 language model, MTP path, and vision tower. AutoRound INT4 quantization is applied to the language-model transformer weights and the exported MTP path as described below. The vision tower remains in BF16.

## Quantization

The export was produced with AutoRound 0.14.2. The settings below are taken directly from `quantization_config.json` and `config.json`:

- Weight precision: INT4
- Format: W4A16 (4-bit weights with floating-point activations)
- Group size: 128
- Symmetric quantization: enabled
- Packing format: `auto_round:auto_gptq`
- Calibration sequence length: 512 tokens
- Batch size: 1
- Quantization method: `auto-round`
- Quantization targets: `model.language_model.layers` and `mtp.layers`
- Selected linear-attention input projections are retained at 16-bit through `extra_config`
- `mtp.fc` is retained at 16-bit through `extra_config`

The exported configuration reports BF16 as the model dtype. The repository is approximately 19.02 GB decimal (17.71 GiB), including configuration, tokenizer, processor files, and SafeTensors weights.

## Model details

- Model name: `Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound`
- Architecture: `Qwen3_5ForConditionalGeneration`
- Text layers: 64
- Hidden size: 5120
- Vocabulary size: 248,320
- Maximum position embeddings: 262,144
- Attention layout: 48 linear-attention layers and 16 full-attention layers
- Parameters represented by the indexed model weights: 6,284,446,960
- Vision tower: BF16, with `Qwen3_5ForConditionalGeneration` multimodal processor configuration

The source model retains its first 15 layers without ablation and does not modify its MTP or visual components. That is source-model information; it is not an additional transformation performed by this AutoRound export.

## Usage with vLLM

The following command launches this exact model repository with vLLM and exposes both the stable `local` alias and the full model name. The `--served-model-name` values are the names accepted by the OpenAI-compatible API; they do not rename the files on disk.

```bash
vllm serve letechlead/Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound \
  --served-model-name local Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --port 18080
```

This command is for the exact artifact `letechlead/Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound`; do not replace the model argument with a generic path such as `/models`. The tensor-parallel size, maximum context that is practical, and KV-cache dtype must be chosen for the available hardware. The model supports text input and multimodal processor inputs when the installed vLLM release supports this architecture and its AutoRound/AWQ-compatible INT4 format.

## Usage with Transformers

Use a recent Transformers release that supports `Qwen3_5ForConditionalGeneration` and the model's processor:

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "letechlead/Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="auto",
)
```

For text-only requests, pass text through the processor. For image or video requests, pass the media together with the text prompt according to the Qwen3.5 Transformers documentation.

## Files

- `model-00001-of-00007.safetensors` through `model-00007-of-00007.safetensors`: sharded model weights
- `model_extra_tensors.safetensors`: additional exported tensors, including the exported MTP path
- `model.safetensors.index.json`: weight-to-shard index and parameter metadata
- `quantization_config.json`: AutoRound quantization metadata
- `config.json`: model architecture, text configuration, vision configuration, and context configuration
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`: tokenizer and chat template
- `processor_config.json`, `preprocessor_config.json`: multimodal processor configuration

## Limitations and responsible use

This model inherits the source model's abliterated behavior and may produce sensitive, controversial, or otherwise unsafe output. It is intended for research, evaluation, and controlled use. Users are responsible for complying with applicable laws, platform rules, and organizational policies, and for reviewing outputs before relying on them.

## License and attribution

This derived model is published under Apache-2.0, matching the license declared by the source model card. Review the source model card and repository for the original model's terms and attribution requirements.

Quantization artifact published by **LeTechLead**.

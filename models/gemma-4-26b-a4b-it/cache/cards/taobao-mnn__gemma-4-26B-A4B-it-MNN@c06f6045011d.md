---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- chat
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
---

# gemma-4-26B-A4B-it-MNN

MNN HQQ 4-bit inference package converted from [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

## Original model

Gemma 4 26B A4B IT is Google DeepMind's instruction-tuned MoE vision-language variant. It accepts text and image input and produces text, with the upstream introduction emphasizing reasoning, coding and multilingual use. The original Gemma 4 family includes both dense and MoE architectures; this repository packages the 26B A4B checkpoint.

Summary adapted from the [upstream model card](https://huggingface.co/google/gemma-4-26B-A4B-it). Original model capabilities and benchmark claims are separate from the MNN smoke checks below.

| Item | Value |
| --- | --- |
| Original checkpoint | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Inputs / outputs in this package | Text and images → text |
| Layers | 30 |
| Routed experts | 128 |
| Text linear quantization | HQQ, 4-bit, group size 64, FP16 scales |
| Token embedding storage | 4-bit, shared with `lm_head` |

This checkpoint ties the token embedding to `lm_head`; [llm_config.json](llm_config.json) records the shared weight offsets and `tie_embeddings.quant_bit`. The recorded `--embed_bit 16` argument only controls a separately exported embedding and does not create BF16 token embedding weights here.

## Download

```bash
python -m pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('taobao-mnn/gemma-4-26B-A4B-it-MNN', local_dir='./gemma-4-26B-A4B-it-MNN')"
export MODEL_DIR="$(pwd)/gemma-4-26B-A4B-it-MNN"
```

## MNN version and build

MNN: [`7337404b2815b5fb9ddac13f4ac6a1fb37fcbbb4`](https://github.com/alibaba/MNN/commit/7337404b2815b5fb9ddac13f4ac6a1fb37fcbbb4)

This revision includes the MoE export fixes used for this release. Linux CPU build:

```bash
git clone https://github.com/alibaba/MNN.git
git -C MNN checkout 7337404b2815b5fb9ddac13f4ac6a1fb37fcbbb4
(cd MNN && ./schema/generate.sh)
cmake -S MNN -B MNN/build -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_BUILD_CONVERTER=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_BUILD_LLM_OMNI=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_BUILD_AUDIO=ON
cmake --build MNN/build --parallel 4
```

## Conversion

The quantization arguments match [export_args.json](export_args.json). Download the original checkpoint and export it with the revision above:

```bash
python -m pip install -r MNN/transformers/llm/export/requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-26B-A4B-it', local_dir='./original/gemma-4-26B-A4B-it', allow_patterns=['*.json', '*.safetensors', '*.model', '*.tiktoken', '*.jinja', '*.txt', '*.py'], ignore_patterns=['original/*', '*.gguf', '*.pt', '*.pth'])"
```

```bash
python MNN/transformers/llm/export/llmexport.py \
  --path ./original/gemma-4-26B-A4B-it --export mnn --hqq \
  --quant_bit 4 --quant_block 64 --scale_bit 16 --embed_bit 16 \
  --lm_quant_bit 4 --lm_quant_block 64 \
  --mnnconvert ./MNN/build/MNNConvert --dst_path ./reexported/gemma-4-26B-A4B-it-MNN
```

After re-export, reuse this package's verified tokenizer and template. Preserve the newly generated weight offsets and other model configuration:

```bash
python - <<'PY'
import json, os, shutil
from pathlib import Path
published = Path(os.environ["MODEL_DIR"])
converted = Path("./reexported/gemma-4-26B-A4B-it-MNN")
for filename in ("llm_config.json", "config.json"):
    path = converted / filename
    config = json.loads(path.read_text())
    reference = json.loads((published / filename).read_text())
    for key in ("jinja", "use_template"):
        config.pop(key, None)
        if key in reference:
            config[key] = reference[key]
    if filename == "config.json":
        config["tokenizer_file"] = "tokenizer.mtok"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
shutil.copyfile(published / "tokenizer.mtok", converted / "tokenizer.mtok")
PY
```

## Run and test

```bash
printf 'What is 2 + 3? Reply with only the number.\n' > prompt.txt
./MNN/build/llm_demo "$MODEL_DIR/config.json" prompt.txt
./MNN/build/llm_bench -m "$MODEL_DIR/config.json"
```

The original migration passed 10 CPU High/Low smoke cases for this model. These are functional checks, not a broad quality or GPU benchmark.

## License and attribution

Weights retain the upstream license and attribution. See the [original model card](https://huggingface.co/google/gemma-4-26B-A4B-it), the license files in this repository and the [MNN LLM documentation](https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html).

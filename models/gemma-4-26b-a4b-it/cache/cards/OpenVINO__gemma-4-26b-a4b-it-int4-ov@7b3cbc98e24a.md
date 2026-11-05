---
license: apache-2.0
base_model:
  - google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- gemma4
- conversational
library_name: openvino
---

# gemma-4-26B-A4B-it-int4-ov

* Model creator: [google](https://huggingface.co/google)
* Original model: [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)


## Description

This is [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2026/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to INT4 by [NNCF](https://github.com/openvinotoolkit/nncf).

## Quantization Parameters

Weight compression was performed using `nncf.compress_weights` with the following parameters:

* mode: **INT4_ASYM**
* group_size: **128**
* ratio: **1.0**

For more information on quantization, check the [OpenVINO model optimization guide](https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression.html).

## Compatibility

The provided OpenVINO™ IR model is compatible with:

* OpenVINO version 2026.4.0 and higher
* Optimum Intel 2.2.0 and higher

## Running Model Inference with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)

1. Install packages required for using Optimum Intel:

```
pip install "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers==5.5.4
pip install torchvision Pillow --extra-index-url https://download.pytorch.org/whl/cpu
```

2. Run model inference:

```python
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
from PIL import Image
import requests

model_id = "OpenVINO/gemma-4-26B-A4B-it-int4-ov"
processor = AutoProcessor.from_pretrained(model_id)
model = OVModelForVisualCausalLM.from_pretrained(model_id)

url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is unusual in this picture?"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt")
input_len = inputs["input_ids"].shape[-1]

output = model.generate(**inputs, do_sample=False, max_new_tokens=100)
response = processor.decode(output[0][input_len:], skip_special_tokens=True)
print(response)
```

You can find more detailed usage examples in OpenVINO Notebooks:

- [Gemma 4](https://openvinotoolkit.github.io/openvino_notebooks/?search=Gemma+4)

## Running Model with OpenAI client and [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)

1a. Deploy model on Windows using [binary package](https://docs.openvino.ai/ovms_baremetal):
```
curl -o ovms.zip https://storage.openvinotoolkit.org/repositories/openvino_model_server/packages/weekly/latest/ovms_windows.zip
tar -xf ovms.zip
set OVMS_MEDIA_URL_ALLOW_REDIRECTS=1
ovms.exe --rest_port 8000 --source_model OpenVINO/gemma-4-26b-a4b-it-int4-ov --model_repository_path C:\models --allowed_media_domains all
```
1b. Deploy model in a Docker container:
```
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)
docker run -d ${GPU_ARGS} -e "OVMS_MEDIA_URL_ALLOW_REDIRECTS=1" -u $(id -u):$(id -g) --rm -p 8000:8000 -v ${HOME}/models:/models:rw openvino/model_server:weekly \
--rest_port 8000 --model_repository_path /models --source_model OpenVINO/gemma-4-26b-a4b-it-int4-ov --allowed_media_domains all
```
2. Install the client library:

```
pip install openai
```
3. Run the client:
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v1",
  api_key="unused"
)

image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"

stream = client.chat.completions.create(
    model="OpenVINO/gemma-4-26b-a4b-it-int4-ov",
    messages=[
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "What is unusual in this picture?"},
        ],
    }
    ],
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": "true"}},
    tools=[],
)

printing_reasoning_started = False
printing_content_started = False
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta

    content = getattr(delta, "content", None)
    reasoning = getattr(delta, "reasoning_content", None)

    if content:
        if not printing_content_started:
            printing_content_started = True
            print("\ncontent:\n", end="", flush=True)
        print(content, end="", flush=True)
    if reasoning:
        if not printing_reasoning_started:
            printing_reasoning_started = True
            print("reasoning_content:\n", end="", flush=True)
        print(reasoning, end="", flush=True)
```

## Limitations

Check the original [model card](https://huggingface.co/google/gemma-4-26B-A4B-it) for limitations.

## Legal information

The original model is distributed under [Apache License Version 2.0](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/LICENSE) license. More details can be found in [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or
contributing to adverse impacts on human rights. See
[Intel's Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf).
Intel's products and software are intended only to be used in applications
that do not cause or contribute to adverse impacts on human rights.


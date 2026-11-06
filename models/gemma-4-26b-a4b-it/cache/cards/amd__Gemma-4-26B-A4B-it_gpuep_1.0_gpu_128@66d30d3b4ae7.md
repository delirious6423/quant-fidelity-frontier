---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: onnxruntime-genai
pipeline_tag: text-generation
tags:
  - onnx
  - onnxruntime
  - gemma
  - gemma-4
  - moe
  - qmoe
  - quantized
  - int4
  - fp16
  - amd
  - mobius
  - olive
language:
  - en
inference: false
---

# Gemma-4-26B-A4B-it_gpuep_1.0_gpu_128

> **Source model:** [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)

## Introduction

Quantized ONNX export of **google/gemma-4-26B-A4B-it** optimized for CPU
inference. Dense weights are 4-bit block-wise with FP16 activations (FP16W4),
and the Mixture-of-Experts (MoE) experts are quantized to 4-bit and emitted as
the fused `com.microsoft::QMoE` operator.

## Quantization Strategy

| Component      | Scheme                                              |
|----------------|-----------------------------------------------------|
| Dense weights  | 4-bit block-wise, FP16 activations (block size 32)  |
| MoE experts    | 4-bit block-wise → `com.microsoft::QMoE` (block size 32) |
| LM head        | Pruned to last-token logits                         |

---
#### License

Modifications copyright(c) 2025 Advanced Micro Devices, Inc. All rights reserved.
AMD has modified the original model files for optimized deployment on AMD Ryzen AI hardware.

MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---
### Base Model License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

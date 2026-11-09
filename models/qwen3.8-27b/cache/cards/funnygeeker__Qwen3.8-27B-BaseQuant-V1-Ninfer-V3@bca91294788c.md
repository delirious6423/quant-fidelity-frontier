---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
- basecompute/Qwen3.8-27B-BaseQuant-V1
- Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- ninfer
- ninfer-v3
- qwen3.8
- quantized
- multimodal
- conversational
- cuda
---

# Qwen3.8-27B-BaseQuant-V1-Ninfer-V3

[English](#english) | [中文](#中文)

# English

This model was converted to the Ninfer-V3 `.ninfer` container format from
[basecompute/Qwen3.8-27B-BaseQuant-V1](https://huggingface.co/basecompute/Qwen3.8-27B-BaseQuant-V1),
which is itself a set of quantized builds of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Only the container format was converted; the weights were not re-quantized and keep the upstream quantization.

The artifact is intended for a Ninfer runtime only. It is not a Transformers checkpoint, not a Safetensors
distribution, and not a GGUF file, so it cannot be loaded by `llama.cpp`, vLLM or `transformers`.

## Artifact

| Field | Value |
| --- | --- |
| Filename | `qwen3_8_27b_basequant_v1_medium.ninfer` |
| Size | 16,703,057,664 bytes (15.56 GiB) |
| Container version | 3 |
| Converted from | `Qwen3.8-27B-BaseQuant-V1-Medium` (16.41 GB) |

The upstream build keeps the Qwen3.8 chat template, thinking/reasoning controls, the vision encoder and the
MTP head; the conversion carries them over unchanged. Upstream ships XS / Small / Medium / Large / XL in GGUF
(llama.cpp) and baseRT formats, together with the full KL-divergence evaluation against BF16 and against
unsloth's dynamic quants. Those results belong to the upstream build — see the
[upstream model card](https://huggingface.co/basecompute/Qwen3.8-27B-BaseQuant-V1); nothing here re-measures
them.

## Which quant to pick

These are observations from my own day-to-day use, not benchmarks.

- For daily use I still recommend a Q4-tier build — this artifact, or
  [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) (Q4/Q5 projections). Q4 gives
  up fine detail, but its logic-related behaviour holds up well. Against the same size class in GGUF — unsloth
  `UD-Q4_K_M` — the upstream BaseQuant card measures this Medium build at roughly 15–24 % lower KL than that
  quant against BF16.
- Q5 ([Qwen3.8-27B-UD-Q5_K_S-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-UD-Q5_K_S-Ninfer-V3))
  preserves detail noticeably better, but it tends to skip self-checking after finishing a task and often ships
  a single pass with no verification round.
- Q6 ([Qwen3.8-27B-UD-Q6_K-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-UD-Q6_K-Ninfer-V3)) is
  clean on detail, yet some logic-related results still come in below Q4, and it runs slightly slower. I would
  not recommend it unless you have a hard quality requirement.
- If your hardware allows it, prefer Qwen3.8 Flash Next over Qwen3.8-27B: even a Flash Next IQ3_S quant came
  out clearly ahead of a Qwen3.8-27B Q8 quant in my testing.

## Related Ninfer builds

| Artifact | Quant | Converted by |
| --- | --- | --- |
| [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) (19.04 GiB) | Q4/Q5 projections | neroued |
| [Qwen3.8-27B-BaseQuant-V1-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-BaseQuant-V1-Ninfer-V3) (15.56 GiB) | BaseQuant V1 Medium, the same size class as `UD-Q4_K_M` | me |
| [Qwen3.8-27B-UD-Q5_K_S-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-UD-Q5_K_S-Ninfer-V3) (17.68 GiB) | `UD-Q5_K_S` | me |
| [Qwen3.8-27B-UD-Q6_K-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-UD-Q6_K-Ninfer-V3) (20.77 GiB) | `UD-Q6_K` | me |

## Provenance

| Field | Value |
| --- | --- |
| Source repository | `basecompute/Qwen3.8-27B-BaseQuant-V1` |
| Upstream base model | `Qwen/Qwen3.8-27B` |
| Converted component | Medium |
| Target format | Ninfer container v3 |

## License

Distributed under Apache-2.0, matching
[basecompute/Qwen3.8-27B-BaseQuant-V1](https://huggingface.co/basecompute/Qwen3.8-27B-BaseQuant-V1) and the
original [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

---

<a id="zh"></a>

# 中文

本模型由 [basecompute/Qwen3.8-27B-BaseQuant-V1](https://huggingface.co/basecompute/Qwen3.8-27B-BaseQuant-V1)
转换为 Ninfer-V3 的 `.ninfer` 容器格式；上游本身是 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
的量化版本。本次只做容器格式转换，没有重新量化，权重量化方式与上游一致。

该产物只供 Ninfer 运行时使用，不是 Transformers 检查点，不是 Safetensors 分发，也不是 GGUF 文件，
因此无法用 `llama.cpp`、vLLM 或 `transformers` 加载。

## 产物

| 字段 | 值 |
| --- | --- |
| 文件名 | `qwen3_8_27b_basequant_v1_medium.ninfer` |
| 大小 | 16,703,057,664 字节（15.56 GiB） |
| 容器版本 | 3 |
| 转换来源 | `Qwen3.8-27B-BaseQuant-V1-Medium`（16.41 GB） |

上游量化版本保留了 Qwen3.8 的 chat template、思考/推理控制、视觉编码器与 MTP 头，转换时原样携带。
上游仓库提供 XS / Small / Medium / Large / XL 五种规格的 GGUF（llama.cpp）与 baseRT 版本，并给出相对 BF16
以及相对 unsloth 动态量化的完整 KL 散度评测。这些数据属于上游量化产物，请见
[上游模型卡](https://huggingface.co/basecompute/Qwen3.8-27B-BaseQuant-V1)，本仓库未重新测量。

## 量化档位怎么选

以下是我在日常使用中的主观观察，不是基准测试。

- 日常使用我仍建议 Q4 档 —— 就是本产物，或 [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
  （Q4/Q5 投影）。Q4 会丢一些细节，但逻辑相关的表现整体站得住。与同档的 GGUF 版本（unsloth `UD-Q4_K_M`）相比，
  上游 BaseQuant 模型卡给出的数据是：这个 Medium 的 KL 低约 15–24%。
- Q5（[Qwen3.8-27B-UD-Q5_K_S-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-UD-Q5_K_S-Ninfer-V3)）
  细节明显更好，但完成任务后倾向于不复检，常常一轮输出就直接交卷。
- Q6（[Qwen3.8-27B-UD-Q6_K-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-UD-Q6_K-Ninfer-V3)）
  细节没问题，但部分逻辑相关的结果仍弱于 Q4，速度也略慢。除非对质量有硬性要求，否则我不太推荐。
- 硬件允许的话，优先选 Qwen3.8 Flash Next 而不是 Qwen3.8-27B：按我的实测，哪怕是 Flash Next 的 IQ3_S 量化，
  表现也明显好过 Qwen3.8-27B 的 Q8 量化。

## 相关 Ninfer 产物

| 产物 | 量化 | 转换者 |
| --- | --- | --- |
| [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)（19.04 GiB） | Q4/Q5 投影 | neroued |
| [Qwen3.8-27B-BaseQuant-V1-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-BaseQuant-V1-Ninfer-V3)（15.56 GiB） | BaseQuant V1 Medium，与 `UD-Q4_K_M` 同档 | 本人 |
| [Qwen3.8-27B-UD-Q5_K_S-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-UD-Q5_K_S-Ninfer-V3)（17.68 GiB） | `UD-Q5_K_S` | 本人 |
| [Qwen3.8-27B-UD-Q6_K-Ninfer-V3](https://huggingface.co/funnygeeker/Qwen3.8-27B-UD-Q6_K-Ninfer-V3)（20.77 GiB） | `UD-Q6_K` | 本人 |

## 来源

| 字段 | 值 |
| --- | --- |
| 来源仓库 | `basecompute/Qwen3.8-27B-BaseQuant-V1` |
| 上游基座模型 | `Qwen/Qwen3.8-27B` |
| 转换对象 | Medium |
| 目标格式 | Ninfer 容器 v3 |

## 许可证

采用 Apache-2.0，与
[basecompute/Qwen3.8-27B-BaseQuant-V1](https://huggingface.co/basecompute/Qwen3.8-27B-BaseQuant-V1)
及原始 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 一致。

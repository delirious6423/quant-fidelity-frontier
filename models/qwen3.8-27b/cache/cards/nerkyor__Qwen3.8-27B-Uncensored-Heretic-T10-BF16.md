---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- multimodal
- bf16
- gguf
- mtp
---

# Qwen3.8-27B Uncensored Heretic T10 BF16

## English

![Qwen3.8-27B Uncensored Heretic T10](/nerkyor/Qwen3.8-27B-Uncensored-Heretic-T10-BF16/resolve/main/assets/qwen38-uncensored-heretic-t10-bf16-q8-hero-en-v2.png)

<!-- qwen38-reasoning-effort-guidance-en-v1:start -->
**Recommended reasoning effort: use `medium` by default. For simple tasks, or if the model becomes repetitive or keeps writing after reaching a final answer, switch to `low`. Reserve `xhigh` for genuinely complex tasks and allow a sufficient reasoning/token budget.**

> **Why:** the official Qwen3.8 template injects an extra `xhigh` instruction to “validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity.” The `low` branch instead asks for brief, focused reasoning and a direct conclusion; `medium` keeps thinking enabled without injecting either extra effort instruction. In local FP8 and Q8 troubleshooting, changing to `low` immediately stopped the observed loop; this is operational evidence, not a universal quality guarantee.
<!-- qwen38-reasoning-effort-guidance-en-v1:end -->

### Overview

This repository contains the complete BF16 merge and the verified Q8_0 llama.cpp package for the **C0/T10 directional refusal-dissolve** derivative of official **Qwen3.8-27B**. The Q8_0 main model, official Q8 MTP draft, and official Q8 vision projector are placed at repository root.

> **Research disclaimer:** This experimental release is provided solely to study the technical feasibility and behavioral effects of refusal-tendency dissolution. It is not a comprehensive safety conclusion, an endorsement of unrestricted use, or professional advice. Users are responsible for lawful and appropriate use and for independently verifying model outputs.

### Base model and derivation

- ModelScope base: [Qwen/Qwen3.8-27B](https://modelscope.cn/models/Qwen/Qwen3.8-27B)
- Hugging Face base: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Frozen source snapshot: `ccd1ef6e1a997cde3a9d544bda75d54e7297bb90`
- Derivation: official Qwen3.8-27B → C0 search → frozen T10 policy → BF16 merge → llama.cpp Q8_0
- T10 parameters: rank 3, alpha 3, dropout 0; targets `attn.o_proj`, `mlp.down_proj`, and compatible `out_proj` modules.

### Release artifacts

| Role | Root file | Bytes | SHA256 |
|---|---|---:|---|
| BF16 | `model-00001-of-00012.safetensors` … `model-00012-of-00012.safetensors` | 54,736,634,773 total | See `SHA256SUMS` |
| Q8_0 main | `Qwen3.8-27B-Uncensored-Heretic-T10-Q8_0.gguf` | 28,595,763,584 | `fe17bff3e43463764b734b945d92c53974143193e2431056a929e3d1b95b5b58` |
| Q8 MTP draft | `Q8-MTP-draft.gguf` | 3,164,006,752 | `e2c0e242a52a92d16d67ef9ca99f41c3028e1cf34b321ecff6adb5a9a3c6ffca` |
| Q8 vision projector | `mmproj-Qwen3.8-27B-Q8_0.gguf` | 629,247,072 | `acaa21ef79fbc451b635c8db29db65b33329aa01c15b968c975a9460cfc82b5c` |

The main Q8_0 GGUF is `qwen35`, file type 7, 851 tensors. The MTP sidecar is `qwen35`, file type 7, 18 tensors. The vision projector is `clip`, file type 7, 334 tensors.

### Frozen Q8_0 results

| Suite | Final score |
|---|---:|
| MMLU500 | **435/500 (87.0%)** |
| GPQA198 | **169/198 (85.35%)** |
| LCB100 | **76/100** |
| Refusal140 hard refusal | **1/140 (0.7%)** |
| Refusal140 soft disclaimer | **32/140 (22.9%)** |

**BF16 Refusal140 hard refusal: 0/140 (0.0%).** The Q8_0 hard-refusal result remains 1/140 (0.7%).

LCB difficulty: easy 23/23, medium 30/31, hard 23/46. The structured result package is available at [`evaluation/q8-results-v1.json`](evaluation/q8-results-v1.json).

### llama.cpp startup

Bare Q8_0 with vision:

```bash
llama-server   -m Qwen3.8-27B-Uncensored-Heretic-T10-Q8_0.gguf   --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf   -ngl all -c 32768 --host 127.0.0.1 --port 8080 --no-webui
```

Q8_0 with MTP and vision:

```bash
llama-server   -m Qwen3.8-27B-Uncensored-Heretic-T10-Q8_0.gguf   --model-draft Q8-MTP-draft.gguf   --spec-type draft-mtp   --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf   -ngl all -c 32768 --host 127.0.0.1 --port 8080 --no-webui
```

### Validation and limits

- Bare Q8_0 load/generation: PASS (`SMOKE_OK_38`).
- Q8 MTP + vision load, text generation, and semantic color vision: PASS.
- Strict synthetic OCR did not reproduce the requested string exactly; OCR quality remains a known limitation.
- Qwen3.8 enables thinking by default. Use the bundled chat template and a recent Qwen3.8-capable llama.cpp build.
- `SHA256SUMS` and `manifest.json` define the exact release.

### License

Apache License 2.0. Users must also follow the official Qwen3.8-27B terms and applicable law.

---

## 中文说明

![Qwen3.8-27B Uncensored Heretic T10](/nerkyor/Qwen3.8-27B-Uncensored-Heretic-T10-BF16/resolve/main/assets/qwen38-uncensored-heretic-t10-bf16-q8-hero-zh-v2.png)

<!-- qwen38-reasoning-effort-guidance-zh-v1:start -->
**思考强度建议：默认使用 `medium`。简单任务，或出现重复循环、已经给出最终答案仍继续输出时，切换到 `low`。`xhigh` 仅用于确实复杂的任务，并预留足够的思考/token 预算。**

> **原因：**Qwen3.8 官方模板会在 `xhigh` 分支额外注入“验证关键假设，考虑合理且可能成立的替代方案，并优先确保正确性、一致性和表达清晰度”；`low` 分支要求保持思考简短、聚焦并直接得出结论；`medium` 保留 thinking，但不注入上述两类额外 effort 指令。本地 FP8 与 Q8 排障中，改为 `low` 后观察到的循环立即停止；这是运行证据，不代表所有任务的通用质量结论。
<!-- qwen38-reasoning-effort-guidance-zh-v1:end -->

本仓库包含官方 **Qwen3.8-27B** 经 C0/T10 定向拒答溶解后的完整 BF16 合并权重，以及已验证的 Q8_0 llama.cpp 套件。主 Q8_0、官方 Q8 MTP draft 与官方 Q8 视觉塔均放在仓库根目录。

> **研究免责声明：** 本实验版本仅用于研究拒答倾向溶解的技术可行性与行为影响。它不构成完整的安全结论，不代表对无限制使用的认可，也不构成任何专业建议。使用者有责任依法、适当地使用模型，并独立核验模型输出。

### 基座与派生关系

- ModelScope 基座：[Qwen/Qwen3.8-27B](https://modelscope.cn/models/Qwen/Qwen3.8-27B)
- Hugging Face 基座：[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- 冻结源快照：`ccd1ef6e1a997cde3a9d544bda75d54e7297bb90`
- 派生链：官方 Qwen3.8-27B → C0 搜索 → T10 冻结方案 → BF16 merge → llama.cpp Q8_0
- T10 参数：rank 3、alpha 3、dropout 0；目标为 `attn.o_proj`、`mlp.down_proj` 和兼容的 `out_proj`。

### 根目录发布文件

| 角色 | 文件 | Bytes | SHA256 |
|---|---|---:|---|
| BF16 | `model-00001-of-00012.safetensors` … `model-00012-of-00012.safetensors` | 54,736,634,773 总计 | 见 `SHA256SUMS` |
| Q8_0 主模型 | `Qwen3.8-27B-Uncensored-Heretic-T10-Q8_0.gguf` | 28,595,763,584 | `fe17bff3e43463764b734b945d92c53974143193e2431056a929e3d1b95b5b58` |
| Q8 MTP draft | `Q8-MTP-draft.gguf` | 3,164,006,752 | `e2c0e242a52a92d16d67ef9ca99f41c3028e1cf34b321ecff6adb5a9a3c6ffca` |
| Q8 视觉塔 | `mmproj-Qwen3.8-27B-Q8_0.gguf` | 629,247,072 | `acaa21ef79fbc451b635c8db29db65b33329aa01c15b968c975a9460cfc82b5c` |

### Q8_0 冻结成绩

| 测评 | 最终得分 |
|---|---:|
| MMLU500 | **435/500（87.0%）** |
| GPQA198 | **169/198（85.35%）** |
| LCB100 | **76/100** |
| Refusal140 硬拒答 | **1/140（0.7%）** |
| Refusal140 边答边免责 | **32/140（22.9%）** |

**BF16 Refusal140 硬拒答：0/140（0.0%）。** Q8_0 硬拒答结果仍为 1/140（0.7%）。

LCB 分项：easy 23/23、medium 30/31、hard 23/46。结构化成绩见 [`evaluation/q8-results-v1.json`](evaluation/q8-results-v1.json)。

### llama.cpp 启动

上面的两条命令分别对应 bare Q8_0 + vision 与 Q8_0 + MTP + vision。MTP 命令必须同时保留 `--model-draft Q8-MTP-draft.gguf` 与 `--spec-type draft-mtp`。

### 验证与限制

- bare Q8_0 真实 load/generation PASS。
- Q8 MTP + 视觉塔真实 load、文本生成与颜色语义视觉 PASS。
- 严格合成 OCR 未准确复现目标字符串，OCR 输出质量仍有限制。
- Qwen3.8 默认启用 thinking；请使用仓内 chat template 与支持 Qwen3.8 的较新 llama.cpp。
- `SHA256SUMS` 与 `manifest.json` 定义本次发布的精确文件。

### 许可证

Apache License 2.0。使用时还需遵守官方 Qwen3.8-27B 许可与适用法律。

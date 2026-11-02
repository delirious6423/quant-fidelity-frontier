---
base_model:
- Qwen/Qwen3.8-27B
---
# Qwen3.8-27B on AX8860 (llama.cpp + ggml-axera, 2026-10-31)

- 在爱芯 AX8860 的 NPU（4 核）上用 llama.cpp 运行 Qwen3.8-27B，支持文本和图像输入。
- Qwen3.8-27B with text and image input on the 4-core NPU of the Axera AX8860, with llama.cpp.
- 性能持续优化中

## 文件 / Files

| 文件 / File | 说明 / Description |
|---|---|
| `qwen3.8-27b-mxfp4.gguf` | 模型权重（GGUF MXFP4，14.8 GB，md5 `32f5a603d4f75edb16ccf7ff3a35bdd6`）/ weights |
| `mmproj-qwen3.8-27b-bf16.gguf` | 视觉编码器权重（BF16，931 MB，md5 `d2b989d06a0eb9109bab182ec2ff03f2`）/ vision encoder weights |
| `bin/` | llama.cpp 板端程序（静态链接，只依赖系统 `/opt/lib` 与 libc）/ the board programs (static; need only the system `/opt/lib` and libc) |
| `kernels/` | NPU kernel：decode、prefill、ViT、输出头各一个 + `manifest.json` / one kernel each for decode, prefill, ViT, output head + `manifest.json` |
| `run.sh` | 运行脚本 / run script |

权重只来自 GGUF：kernel 不含权重，加载时从 GGUF 转换到 NPU（约 70 s）。
The weights come only from the GGUFs: the kernels hold none; they are converted from the GGUF at load (~70 s).

## 运行 / Run

环境 / Requirements：AX8860，SDK V0.5.0 或更新（`cat /proc/ax_proc/version`）/ SDK V0.5.0 or newer.

```sh
./run.sh "The capital of France is" 64                 # 文本补全 / text completion
./run.sh -i bus.jpg "Describe this image." 256         # 图像描述 / describe an image
./run.sh -s 8080                                       # OpenAI 兼容服务 / OpenAI-compatible server
curl http://<board>:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

直接调用程序时设置 / When calling the programs directly: `LD_LIBRARY_PATH=/opt/lib GGML_AXERA_KERNEL_DIR=<repo>/kernels`，
`-ngl 99`，上下文 `-c` 最大 16384，服务端 `-np 1`。`GGML_AXERA_PROF=1` 退出时打印各阶段耗时 / prints per-stage times at exit.

## 性能 / Performance

AX8860，SDK V0.5.1，NPU 4 核 / 4 cores：

| 项 / Item | 结果 / Result |
|---|---|
| 4k prompt（3908 token）prefill | 3.18 s（1231 tokens/s） |
| decode（4k 上下文 / context） | 72.8 ms/token（13.7 tokens/s） |
| 16k prompt prefill | 1168 tokens/s |
| decode（16k 上下文 / context） | 81.0 ms/token（12.3 tokens/s） |
| 图像编码 / image encoding（3072 patch = 768 图像 token / image tokens） | 0.28 s |
| 加载 / load | 约 70 s |

## 正确性 / Correctness

- "The capital of France is" → "Paris. …"；图像描述正确（车辆、文字、人物）/ image descriptions correct.
- 中文 PPL（内部测试集，`-c 512`）：10.85 / Chinese PPL on an internal set: 10.85.
- 同一输入多次运行，结果偶有极小差异（NPU 上累加顺序不固定），不影响输出质量。
  Repeated runs of one input may differ slightly (the NPU's accumulation order varies); the output quality is the same.

## 已知限制 / Known limits

- 运行时独占 NPU 4 核，同时运行其他 NPU 程序会出错 / takes all 4 NPU cores; do not run other NPU programs alongside.
- 单序列：服务端须 `-np 1` / one sequence: the server needs `-np 1`.
- 上下文最大 16384 / context up to 16384.

## 源码 / Sources

- llama.cpp：[ZHEQIUSHUI/llama.cpp](https://github.com/ZHEQIUSHUI/llama.cpp) `feat/qwen38-27b-prefill` @ `1811df5`（`ggml/src/ggml-axera/models/qwen38_27b`）
- kernel：[ZHEQIUSHUI/ax-dsl-kernels](https://github.com/ZHEQIUSHUI/ax-dsl-kernels) `feat/qwen38-27b-vit` @ `18346cb`（`models/qwen38_27b`，`kernels.md5`）
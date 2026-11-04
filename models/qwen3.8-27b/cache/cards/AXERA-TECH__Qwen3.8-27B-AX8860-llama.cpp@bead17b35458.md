---
base_model:
- Qwen/Qwen3.8-27B
---
# Qwen3.8-27B on AX8860 (llama.cpp + ggml-axera, 2026-11-04)

在爱芯 AX8860 的 NPU（4 核）上用 llama.cpp 运行 Qwen3.8-27B，支持文本和图像输入；服务端用模型自带的 MTP 头做投机解码（每轮猜 3 个 token）；三套 kernel 对应 16k / 128k / 256k 上下文。
Qwen3.8-27B with text and image input on the 4-core NPU of the Axera AX8860, with llama.cpp; the server decodes speculatively with the model's own MTP head (3 drafts per round); three kernel sets for 16k / 128k / 256k contexts.

## 文件 / Files

| 文件 / File | 说明 / Description |
|---|---|
| `qwen3.8-27b-mxfp4.gguf` | 模型权重（GGUF MXFP4，14.8 GB，md5 `32f5a603d4f75edb16ccf7ff3a35bdd6`）/ weights |
| `mmproj-qwen3.8-27b-bf16.gguf` | 视觉编码器权重（BF16，931 MB，md5 `d2b989d06a0eb9109bab182ec2ff03f2`）/ vision encoder weights |
| `bin/` | llama.cpp 板端程序（静态链接，只依赖系统 `/opt/lib` 与 libc）/ the board programs (static; need only the system `/opt/lib` and libc) |
| `kernels/` | **默认 / default**：128k 上下文的 NPU kernel + `manifest.json` / the 128k-context kernels |
| `kernels_16k/` | 16k 上下文：NPU 内存最省，短对话 / agent 场景 / 16k context: the smallest NPU memory, for short conversations and agents |
| `kernels_256k/` | 256k 上下文（模型原生最大）/ 256k context (the model's native maximum) |
| `run.sh` | 运行脚本 / run script |

每套 kernel：prefill、decode、verify（投机解码的 4 行验证）、MTP 头（3 个）、ViT、输出头、草稿词表头 + `manifest.json`；三套只有 prefill / decode / verify / MTP 不同。
Each set: prefill, decode, verify (4 rows, speculative decoding), the MTP head (3 kernels), ViT, output head, the draft-vocabulary head + `manifest.json`; only prefill / decode / verify / MTP differ between the sets.

权重只来自 GGUF：kernel 不含权重，加载时从 GGUF 转换到 NPU（63 - 72 s）；NPU 内存里只放一份权重（prefill 和 decode 共用）。
The weights come only from the GGUFs: the kernels hold none; they are converted from the GGUF at load (63 - 72 s); one copy of the weights in NPU memory (shared by prefill and decode).

## 运行 / Run

环境 / Requirements：AX8860，SDK V0.5.0 或更新（`cat /proc/ax_proc/version`）/ SDK V0.5.0 or newer.

```sh
./run.sh "The capital of France is" 64                 # 文本补全 / text completion
./run.sh -i bus.jpg "Describe this image." 256         # 图像描述 / describe an image
./run.sh -c                                            # 命令行多轮对话 / multi-turn chat in the terminal
./run.sh -s 8080                                       # OpenAI 兼容服务，MTP 投机解码 / OpenAI-compatible server, MTP speculative decoding
KERNELS=16k ./run.sh -s 8080                           # 16k 套件（省内存）/ the 16k set (less NPU memory)
KERNELS=256k CTX=200000 ./run.sh -s 8080               # 256k 套件，llama.cpp 上下文 200000 / the 256k set, context 200000
curl http://<board>:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

`KERNELS`（`128k` 默认 / `16k` / `256k`）选 kernel 套件；`CTX` 是传给 llama.cpp 的 `-c`，默认等于套件的上下文，可以更小。
`KERNELS` (`128k`, the default / `16k` / `256k`) picks the kernel set; `CTX` is llama.cpp's `-c`, the set's context by default, may be smaller.

直接调用程序时设置 / When calling the programs directly: `LD_LIBRARY_PATH=/opt/lib GGML_AXERA_KERNEL_DIR=<repo>/kernels`（或 `kernels_16k` / `kernels_256k`），
`-ngl 99`，`-c` 不超过套件的上下文，服务端 `-np 1`；投机解码（只有 `llama-server` 支持）`--spec-type draft-mtp --spec-draft-n-max 3`，不加则为普通解码 / speculative decoding (`llama-server` only) `--spec-type draft-mtp --spec-draft-n-max 3`; without it: plain decoding。`GGML_AXERA_PROF=1` 退出时打印各阶段耗时 / prints per-stage times at exit。`GGML_AXERA_TEMP_MAX=<°C>`（默认 100，0 关闭）：NPU 温度超过时暂停 kernel，见"已知限制" / pauses kernels above that NPU temperature (default 100, 0 off), see Known limits.

## 性能 / Performance

AX8860，SDK V0.5.1，NPU 4 核 / 4 cores：

| 项 / Item | `kernels_16k` | `kernels`（128k） | `kernels_256k` |
|---|---|---|---|
| 加载后 NPU 内存（CMM）/ NPU memory after load | 16.5 GB | 24.2 GB | 32.9 GB |
| 加载后系统内存 / host memory after load | 0.9 GB（峰值 / peak 1.9 GB） | 0.9 GB（1.9 GB） | 0.9 GB（1.9 GB） |
| 加载 / load | 63 s | 67 s | 72 s |
| 1.7k prompt（1708 token）prefill | 1125 tokens/s | 1202 tokens/s | 1145 tokens/s |
| decode，MTP 投机解码（服务端，短回答 / 400 token 长回答）| 22 - 24 / 20 tokens/s | 21 - 23 / 19 tokens/s | 21 - 22 / 18 tokens/s |
| decode，普通 / plain（短上下文 / short context） | 13.1 - 13.6 tokens/s | 13.2 - 13.6 tokens/s | 13.2 - 13.5 tokens/s |
| 64k prompt prefill / decode（64k 上下文）| - | 683 tokens/s¹ / 8.5 tokens/s（MTP 11.1） | 630 tokens/s¹ / 8.5 tokens/s（MTP 11.1 tokens/s） |
| 120k prompt prefill / decode（113k 上下文）| - | - | 403 tokens/s¹ / 7.2 tokens/s（MTP 7.5 tokens/s） |
| 图像编码 / image encoding（3072 patch = 768 图像 token / image tokens） | 0.28 s | 0.28 s | 0.28 s |

¹ 含温度保护的暂停（见"已知限制"）；不暂停时 64k prefill 约 957 tokens/s / includes the thermal pauses (Known limits); ~957 tokens/s without them.

MTP：每轮接受 2.0 - 2.4 个 token（长上下文下验证也要扫整个 KV，64k 上下文 11.1 tokens/s，113k 7.5 tokens/s / at a long context the verification reads the whole KV too: 11.1 tokens/s at 64k, 7.5 at 113k）；一轮约 100 ms（验证 77 ms，输出头 4 ms，3 个草稿 3 x 3.5 + 3 x 1.1 ms，host 8 ms）/ 2.0 - 2.4 tokens accepted per round; a round ~100 ms (verify 77, output head 4, 3 drafts 3 x 3.5 + 3 x 1.1, host 8).

## 正确性 / Correctness

- "The capital of France is" → "Paris. …"，三套输出相同；图像描述正确（车辆、文字、人物）/ the same output from the three sets; image descriptions correct.
- 长文 PPL（`-c 4096`，英文小说）：16k 4.294，128k 4.287，256k 4.289 / long-text PPL (`-c 4096`, an English novel): 4.294 / 4.287 / 4.289；中文 PPL（内部测试集，`-c 512`）：10.85 / Chinese PPL on an internal set: 10.85.
- 大海捞针 / needle in a haystack（一句暗号藏在小说 30% 处，末尾提问 / a code hidden at 30% of a novel, asked at the end）：128k 套件 32k / 64k token 命中，256k 套件 64k / 120k token 命中 / found at 32k / 64k tokens (128k set) and 64k / 120k tokens (256k set).
- 投机解码的输出与普通贪心解码一致（每个草稿 token 都经目标模型验证）/ speculative decoding gives the same output as plain greedy decoding (every draft token is verified by the target model).
- 同一输入多次运行，结果偶有极小差异（NPU 上累加顺序不固定），不影响输出质量。
  Repeated runs of one input may differ slightly (the NPU's accumulation order varies); the output quality is the same.

## 已知限制 / Known limits

- 运行时独占 NPU 4 核，同时运行其他 NPU 程序会出错 / takes all 4 NPU cores; do not run other NPU programs alongside.
- 单序列：服务端须 `-np 1` / one sequence: the server needs `-np 1`.
- 板子 16 GB 系统内存：`llama-perplexity` 的 `-c` 不超过 4096（它把整段上下文的 logits 放在系统内存里）/ with the board's 16 GB host memory, `llama-perplexity` works up to `-c 4096` (it keeps the whole context's logits in host memory).
- 上下文随 `-c` 增大，decode 变慢（attention 要读整个 KV）：64k 上下文 8.5 tokens/s，113k 7.2 tokens/s / decode slows with the context (attention reads the whole KV): 8.5 tokens/s at 64k, 7.2 at 113k.

## 源码 / Sources

- llama.cpp：[ZHEQIUSHUI/llama.cpp](https://github.com/ZHEQIUSHUI/llama.cpp) `axera-dev` @ `c9d6eb8`（`ggml/src/ggml-axera/models/qwen38_27b`）
- kernel：[ZHEQIUSHUI/ax-dsl-kernels](https://github.com/ZHEQIUSHUI/ax-dsl-kernels) `main` @ `72766f5`（`models/qwen38_27b`，`kernels.md5` 列出三套 / lists the three sets）
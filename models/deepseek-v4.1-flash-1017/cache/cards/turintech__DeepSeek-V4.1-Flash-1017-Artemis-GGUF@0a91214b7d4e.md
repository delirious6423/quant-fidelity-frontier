---
license: "mit"
library_name: gguf
pipeline_tag: text-generation
base_model: "deepseek-ai/DeepSeek-V4.1-Flash-1017"
base_model_relation: quantized
tags:
- gguf
---
<!-- Generated from the canonical model card. Edit canonical facts or templates, then regenerate. -->
# DeepSeek-V4.1-Flash-1017 — Artemis

A 385.58 GB mixed-precision GGUF of DeepSeek-V4.1-Flash-1017 for text inference. All seven shards are required; download them into the same directory.

Model type: **chat**. Recommended variant: `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K` — **385.583 GB total**, 7 required files.

## Run with Hugging Face

Verification status: **verified**.

- Download all seven shards into the directory used below:
  ```sh
  hf download turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF \
    --include "DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-*.gguf" \
    --local-dir ./DeepSeek-V4.1-Flash-1017-Artemis-Q3_K
  ```
- Use [llama.cpp commit 149f5641](https://github.com/ggml-org/llama.cpp/commit/149f5641b931aaf33697d70f9abc90c3e5300247). All seven shards must be in the same directory. The command below is the tested CPU route; GPU serving has not been qualified.
- Start the server from your llama.cpp checkout:
  ```sh
  CUDA_VISIBLE_DEVICES='' ./build/bin/llama-server \
    -m ./DeepSeek-V4.1-Flash-1017-Artemis-Q3_K/DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00001-of-00007.gguf \
    --host 127.0.0.1 --port 18741 --device none -ngl 0 \
    --load-mode mmap --lazy-mode on -t 8 -tb 8 -c 4096 \
    --parallel 1 -b 128 -ub 64 --no-warmup
  ```
- Use the [official DeepSeek encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/encoding.py), not a generic chat template. Encode messages with `encode_messages(messages, thinking_mode="chat")`, send the resulting string to `/tokenize` with `add_special: false` and `parse_special: true`, then send the returned integer tokens as `/completion`'s `prompt`. The encoder already supplies BOS.
- The tested request settings were `temperature: 0`, `seed: 0`, `n_predict: 256`, `cache_prompt: false`, and `stream: false` for the first text example. Check the returned stop reason: reaching the token limit does not establish a complete answer.
- Tested on CPU with 2 TB system RAM and lazy Engram loading. Observed peak process RSS was 254.01 GB; this is not a minimum-memory requirement and excludes other system memory. CPU generation can be slow. No GPU memory or throughput claim is made.

## Files

Recommended shard set `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K`: **385.583 GB total download** (7 required files).

Download every shard of the chosen set into the same directory and load its first shard. Each shard set is a separate model variant.

| File | GB | Choose for |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00001-of-00007.gguf` | 48.666 | Recommended entry point (shard 1/7) |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00002-of-00007.gguf` | 48.660 | Required shard 2/7 |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00003-of-00007.gguf` | 48.660 | Required shard 3/7 |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00004-of-00007.gguf` | 48.660 | Required shard 4/7 |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00005-of-00007.gguf` | 38.928 | Required shard 5/7 |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00006-of-00007.gguf` | 135.172 | Required shard 6/7 |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00007-of-00007.gguf` | 16.835 | Required shard 7/7 |

GB means decimal download size, not runtime memory.

## Limitations

- The first shard is the loader entry point, not a standalone model. Engram tables are included; draft/MTP and the vision projector are not.
- Mixed tensor precisions; Q3 does not mean every tensor uses three bits.
- Use the official DeepSeek prompt encoding. Generic chat-completions, Ollama and LM Studio compatibility have not been qualified.
- Eight basic native CPU checks passed, including JSON, code, multilingual text and tool-call formatting. These checks are not a broad quality benchmark; comparative speed and GPU operation have not been evaluated.



## Source

Base model: deepseek-ai/DeepSeek-V4.1-Flash-1017.
License: [MIT](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE) (`mit`).



<details>
<summary>File identities</summary>

Original source: `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00001-of-00007.gguf` | 48666243712 | `6de9f2f646786e5213739d8f52b26f922b8122ecdb01fcd7d45af686fa020284` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00002-of-00007.gguf` | 48660481984 | `94d1da5a9dafc87d0dd5a2bbcaaf5eff75d4c1444572a601f0517757b9779be3` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00003-of-00007.gguf` | 48660481984 | `f35d1665ec3b3d79ca148456fba756d535476183b9f9ad97e08ed7fa5bddc6d0` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00004-of-00007.gguf` | 48660481984 | `0d2fbb089886287dd420b19563d28349a141ded98ec2db8dc80875dd28307c42` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00005-of-00007.gguf` | 38928385600 | `ecf92fb8f28744fffe6272731166eee49f671b2b94cf13697f2a6589668c9786` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00006-of-00007.gguf` | 135172021792 | `6b71f7ac2138992bedceecec51b535f16dddade64c3827dfce1769911b2d672a` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00007-of-00007.gguf` | 16835109952 | `e48c51a4314072344f39edeaaa0fb9e995a0683a6c0cfc5caa01e9e423b4c0dc` |

</details>


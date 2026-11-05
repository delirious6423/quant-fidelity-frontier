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

DeepSeek-V4.1-Flash-1017 in mixed-precision GGUF for text inference. Q3 is the recommended variant; Q2 is a smaller alternative.

| Variant | Download | Required shards | Basic native CPU checks |
| --- | ---: | ---: | --- |
| Q3_K (recommended) | 385.58 GB | 7 | 8/8 responses completed |
| Q2_K | 330.38 GB | 6 | 7/8 completed; Spanish response reached the 256-token cap |

Q2 reduces download size by 14.32% with the same main-text and Engram components. These eight-prompt checks per variant do not establish equivalent quality or lower runtime memory. Both variants matched the original tokenizer's input IDs; code and tool-call outputs were inspected, not executed.

Model type: **chat**. Recommended variant: `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K` — **385.583 GB total**, 7 required files.

## Run with Hugging Face

Verification status: **verified**.

- For the recommended Q3 variant, download all seven shards into the directory used below:
  ```sh
  hf download turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF \
    --include "DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-*.gguf" \
    --local-dir ./DeepSeek-V4.1-Flash-1017-Artemis-Q3_K
  ```
- Use [llama.cpp release v0.9.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0) or [commit 149f5641](https://github.com/ggml-org/llama.cpp/commit/149f5641b931aaf33697d70f9abc90c3e5300247); both gave the same results on our eight checks with these files. All shards of the selected variant must be in the same directory. The command below is the tested CPU route; GPU serving has not been qualified.
- Start the server from your llama.cpp checkout:
  ```sh
  CUDA_VISIBLE_DEVICES='' ./build/bin/llama-server \
    -m ./DeepSeek-V4.1-Flash-1017-Artemis-Q3_K/DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00001-of-00007.gguf \
    --host 127.0.0.1 --port 18741 --device none -ngl 0 \
    --load-mode mmap --lazy-mode on -t 8 -tb 8 -c 4096 \
    --parallel 1 -b 128 -ub 64 --no-warmup
  ```
- For the Q2 alternative, download its six shards and use its tested CPU command:
  ```sh
  hf download turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF \
    --include "DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-*.gguf" \
    --local-dir ./DeepSeek-V4.1-Flash-1017-Artemis-Q2_K
  CUDA_VISIBLE_DEVICES='' ./build/bin/llama-server \
    -m ./DeepSeek-V4.1-Flash-1017-Artemis-Q2_K/DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00001-of-00006.gguf \
    --host 127.0.0.1 --port 18742 --device none -ngl 0 \
    --load-mode mmap --lazy-mode on -t 8 -tb 8 -c 4096 \
    --parallel 1 -b 128 -ub 64 --no-warmup
  ```
  Send Q2 requests to `http://127.0.0.1:18742`; the Q3 command uses port 18741.
- Use the [official DeepSeek encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/encoding.py), not a generic chat template. Encode messages with `encode_messages(messages, thinking_mode="chat")`, send the resulting string to `/tokenize` with `add_special: false` and `parse_special: true`, then send the returned integer tokens as `/completion`'s `prompt`. The encoder already supplies BOS.
- The tested request settings were `temperature: 0`, `seed: 0`, `n_predict: 256`, `cache_prompt: false`, and `stream: false` for the first text example. Check the returned stop reason: reaching the token limit does not establish a complete answer.
- Tested on CPU with 2 TB system RAM and lazy Engram loading. Observed peak process RSS was 254.01 GB for Q3 and 372.65 GB for Q2. These shared-host observations are not minimum-memory requirements or a matched memory comparison and exclude other system memory. CPU generation can be slow; GPU memory and comparative throughput remain unmeasured.

## Files

Recommended shard set `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K`: **385.583 GB total download** (7 required files).

Alternative shard set `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K`: **330.376 GB total download** (6 required files).

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
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00001-of-00006.gguf` | 49.056 | Alternative entry point (shard 1/6) |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00002-of-00006.gguf` | 49.050 | Required shard 2/6 |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00003-of-00006.gguf` | 49.050 | Required shard 3/6 |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00004-of-00006.gguf` | 31.213 | Required shard 4/6 |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00005-of-00006.gguf` | 135.172 | Required shard 5/6 |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00006-of-00006.gguf` | 16.835 | Required shard 6/6 |

GB means decimal download size, not runtime memory.

## Limitations

- The first shard is the loader entry point, not a standalone model. Download all shards of the chosen variant into one directory. Both variants include Engram tables; draft/MTP and the vision projector are absent.
- Both variants mix tensor precisions. Q2\_K and Q3\_K identify the expert-weight formats; Engram is Q5\_K and other tensors retain higher precisions.
- Use the official DeepSeek prompt encoding. Generic chat-completions, Ollama and LM Studio compatibility have not been qualified.
- Q2 produced eight nonempty responses without observed severe repetition. Its Spanish response ended mid-sentence at the 256-token cap and lacked the EOS required by the official output parser.
- The basic checks included JSON, code, multilingual text and tool-call formatting. Broad quality benchmarks, comparative speed and GPU operation have not been evaluated.

<details>
<summary>Benchmark results and methodology</summary>

> Quick development evaluation; not release qualification.

## Basic-check scope

Both variants were checked with the official DeepSeek encoder on the pinned llama.cpp CPU runtime, context 4096 and eight threads. The fixed prompts covered explanations, JSON, code, summarization, Spanish text, context recall, a tool-call format and arithmetic reasoning. Responses were reviewed manually; these are load and behavior smoke checks.

</details>

## Source

Base model: deepseek-ai/DeepSeek-V4.1-Flash-1017.
License: [MIT](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE) (`mit`).



<details>
<summary>File identities</summary>

Original source: `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00001-of-00006.gguf` | 49055528160 | `65238b840dc795bf1f6b9370ecc1db3a52ca3fd5aed1bab0b4409153a3b90e51` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00002-of-00006.gguf` | 49049766400 | `a091dafb5cafbe7bda3a21c2082714bd36bce78656ddb870b5647c618e99dcd4` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00003-of-00006.gguf` | 49049766400 | `2da27d57e97a8448187d48f6b807259795c492a38ce00f3d55cdba0836969736` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00004-of-00006.gguf` | 31213487776 | `59f96a0032be1e1481c349dfa0b7b4a4d1bc7a084f2396fd87432c7a76317226` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00005-of-00006.gguf` | 135172021792 | `5f7c74342fa555a532d0a9664bb55ef1e61cea6a00e83f5bb35e6f7488204183` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00006-of-00006.gguf` | 16835109952 | `eb1aa8ca8c71efe49ec2d2277cb64c99a87b09815a8e54c0aa55943a2eea4393` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00001-of-00007.gguf` | 48666243712 | `6de9f2f646786e5213739d8f52b26f922b8122ecdb01fcd7d45af686fa020284` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00002-of-00007.gguf` | 48660481984 | `94d1da5a9dafc87d0dd5a2bbcaaf5eff75d4c1444572a601f0517757b9779be3` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00003-of-00007.gguf` | 48660481984 | `f35d1665ec3b3d79ca148456fba756d535476183b9f9ad97e08ed7fa5bddc6d0` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00004-of-00007.gguf` | 48660481984 | `0d2fbb089886287dd420b19563d28349a141ded98ec2db8dc80875dd28307c42` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00005-of-00007.gguf` | 38928385600 | `ecf92fb8f28744fffe6272731166eee49f671b2b94cf13697f2a6589668c9786` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00006-of-00007.gguf` | 135172021792 | `6b71f7ac2138992bedceecec51b535f16dddade64c3827dfce1769911b2d672a` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00007-of-00007.gguf` | 16835109952 | `e48c51a4314072344f39edeaaa0fb9e995a0683a6c0cfc5caa01e9e423b4c0dc` |

</details>


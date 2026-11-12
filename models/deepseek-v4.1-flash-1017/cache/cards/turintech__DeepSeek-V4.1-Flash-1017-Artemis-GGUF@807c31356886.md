---
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- turintech
- artemis
- gguf
- llama.cpp
- quantized
- deepseek
- moe
- mixture-of-experts
---

<!-- v2-header:start -->
<div align="center">

<a href="https://www.turintech.ai"><img src="https://huggingface.co/turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF/resolve/main/assets/turintech-logo-v2.png" width="360" alt="TurinTech"></a>

<em>TurinTech models, built with Artemis</em>

<h3 align="center" style="justify-content: center;">DeepSeek V4.1 Flash 1017 in two GGUF shard sets for CPU inference, compressed with Artemis</h3>

<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 6px;">
<a href="https://www.turintech.ai"><img src="https://huggingface.co/turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF/resolve/main/assets/buttons/artemis.png" width="84" alt="Artemis"></a>
<a href="https://huggingface.co/turintech"><img src="https://huggingface.co/turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF/resolve/main/assets/buttons/other-models.png" width="122" alt="Other TurinTech models on Hugging Face"></a>
<a href="https://www.linkedin.com/company/turintechai"><img src="https://huggingface.co/turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF/resolve/main/assets/buttons/turintech.png" width="98" alt="TurinTech on LinkedIn"></a>
</div>

| Download | Basic prompts completed |
| :---: | :---: |
| **385.58 GB** | **8 of 8** |

</div>
<!-- v2-header:end -->

## Run it

```bash
hf download turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF \
  --include "DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-*.gguf" --local-dir . && \
  llama-server -m DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00001-of-00007.gguf --port 8080 --device none -ngl 0 \
  --load-mode mmap --lazy-mode on -t 8 -tb 8 -c 4096 --parallel 1 -b 128 -ub 64 --no-warmup
```

<!-- v2-apps:start -->
<!-- v2-apps:end -->

Needs a llama.cpp with DeepSeek V4.1 Flash support. Tested with release v0.9.0 and commit `149f5641`, which gave the
same outcome on our eight prompts; no minimum build is established. The command downloads the seven Q3_K shards
(385.58 GB) into one folder and starts the server on the first; llama.cpp finds the rest. This is the tested route,
on a CPU: GPU serving was not qualified. On a shared host with 2 TB of RAM the server's peak RSS was 254.01 GB.

Requests go to `/completion`, not to chat completions. Encode the messages with DeepSeek's own encoder,
`encode_messages(messages, thinking_mode="chat")` from `encoding/encoding.py` in
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/encoding.py)
at `1bae0e1`. Send that string to `/tokenize` with `"add_special": false, "parse_special": true`, then send the
returned token list as the `prompt` of `/completion`. The encoder adds the beginning-of-sequence token, so no other
is needed.

<details>
<summary>Request settings, the Q2_K set, Ollama and LM Studio</summary>

Our checks sent `"temperature": 0, "seed": 0, "n_predict": 256, "cache_prompt": false, "stream": false` to
`/completion` (128 tokens for the recall prompt, 384 for the tool call and 1,024 for the thinking prompt), then read
the reply's stop reason: a reply that reached the token limit is not a complete answer. llama-server leaves the
end-of-sequence token out of `content`, so when `stop_type` is `eos` our checks appended the encoder's `eos_token`
before calling DeepSeek's `parse_message_from_completion_text` (in the same encoder file) with `"chat"`, or
`"thinking"` for the thinking prompt; it reads the reply, including tool calls.

The Q2_K set is six shards (330.38 GB). It completed 7 of the 8 prompts; its Spanish reply stopped at the 256-token cap.

```bash
hf download turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF \
  --include "DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-*.gguf" --local-dir . && \
  llama-server -m DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00001-of-00006.gguf --port 8080 --device none -ngl 0 \
  --load-mode mmap --lazy-mode on -t 8 -tb 8 -c 4096 --parallel 1 -b 128 -ub 64 --no-warmup
```

Download one set, not both. Ollama and LM Studio: no package or entry is provided for this model, and neither was
qualified, nor were chat completions; an automatic integration button on the Hub is not a compatibility test.

</details>

<!-- v2-results:start -->
## Results

| File | Size | Prompts completed (of 8) | Stopped at the token cap (of 8) | Peak server RSS on the CPU |
| --- | --- | --- | --- | --- |
| TurinTech Q3_K | 385.58 GB | 8 | 0 | 254.01 GB |
| TurinTech Q2_K | 330.38 GB | 7 | 1 | 372.65 GB |

Eight fixed prompts per shard set on a CPU, reviewed by hand: an explanation, JSON, code, a summary, Spanish, recall of an earlier turn, a tool call and arithmetic with thinking on. Code and tool calls were not executed. This checks that the files load and answer; it is not a benchmark. No benchmark was run, no file of another publisher was compared, and eight prompts do not rank the two sets, so nothing is bolded and there is no chart. The peak RSS was observed on a shared host with 2 TB of RAM; it is not a minimum requirement or a matched comparison. [Full results, memory and method](RESULTS.md).
<!-- v2-results:end -->

<!-- v2-files:start -->
## Files

| File | Size | Use it for |
| --- | --- | --- |
| [`DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00001-of-00007.gguf`](https://huggingface.co/turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF/resolve/main/DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00001-of-00007.gguf) and 6 more parts | 385.58 GB | Default |
| [`DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00001-of-00006.gguf`](https://huggingface.co/turintech/DeepSeek-V4.1-Flash-1017-Artemis-GGUF/resolve/main/DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00001-of-00006.gguf) and 5 more parts | 330.38 GB | Smaller set of six shards |

Each set is a group of shards: download all of one set into one folder and load its first shard. Text only: the MTP draft layer and the vision encoder are not included. Memory on a CPU host with 2 TB of RAM, lazy Engram loading and a 4,096-token context: the server's peak RSS was 254.01 GB with Q3_K and 372.65 GB with Q2_K, observations rather than requirements; GPU memory was not measured. Without a tag, `-hf` and `hf.co` fetch `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00001-of-00006.gguf`, not the default file, and the default file has no `-hf` tag, so download it with `hf download` as in Run it.
<!-- v2-files:end -->

<!-- v2-pick:start -->
## Pick this model

- **Good for** running DeepSeek V4.1 Flash 1017 on a CPU host with several hundred GB of RAM: the Q3_K set (385.58 GB) completed all 8 basic prompts.
- **Pick another file** when a smaller download matters more: the Q2_K set is 330.38 GB and completed 7 of 8, its Spanish reply stopping at the 256-token cap; the observed peak RSS was 372.65 GB for Q2_K and 254.01 GB for Q3_K, in runs that were not matched. Neither set was benchmarked, other publishers' GGUF files of this model were not compared, and GPU serving, chat completions, Ollama and LM Studio were not qualified.
<!-- v2-pick:end -->

## How it was made

Built with [Artemis](https://www.turintech.ai/product/overview), TurinTech's optimisation platform, from
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at `1bae0e1`.
Each set holds the whole text model with its Engram tables in mixed precision: the expert weights are Q3_K or Q2_K,
the Engram tables Q5_K and the other tensors higher precisions, so each label names the expert format, not a uniform
precision. The MTP draft layer and the vision encoder are not included. Stock llama.cpp loads both sets: release
v0.9.0 and commit `149f5641` were tested.

## How it was tested

Each set answered eight fixed prompts on a CPU with llama.cpp commit `149f5641`: an explanation, JSON, code, a summary,
Spanish, recall of an earlier turn, a tool call and arithmetic with thinking on. The prompts went through DeepSeek's
encoder and `/completion`, and the token IDs matched DeepSeek's own tokenizer for every prompt. The replies were read
by hand; code and tool calls were not executed. The same prompts on release v0.9.0 gave the same outcome. This shows
that the files load and answer. It is not a benchmark: no benchmark was run and no file of another publisher was
compared. The [results page](RESULTS.md) has the outcome of each prompt, the settings and the memory observations.

<!-- v2-checksums:start -->
<details>
<summary>File checksums</summary>

| File | Bytes | sha256 |
| --- | --- | --- |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00001-of-00007.gguf` | 48,666,243,712 | `6de9f2f646786e5213739d8f52b26f922b8122ecdb01fcd7d45af686fa020284` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00002-of-00007.gguf` | 48,660,481,984 | `94d1da5a9dafc87d0dd5a2bbcaaf5eff75d4c1444572a601f0517757b9779be3` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00003-of-00007.gguf` | 48,660,481,984 | `f35d1665ec3b3d79ca148456fba756d535476183b9f9ad97e08ed7fa5bddc6d0` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00004-of-00007.gguf` | 48,660,481,984 | `0d2fbb089886287dd420b19563d28349a141ded98ec2db8dc80875dd28307c42` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00005-of-00007.gguf` | 38,928,385,600 | `ecf92fb8f28744fffe6272731166eee49f671b2b94cf13697f2a6589668c9786` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00006-of-00007.gguf` | 135,172,021,792 | `6b71f7ac2138992bedceecec51b535f16dddade64c3827dfce1769911b2d672a` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q3_K-00007-of-00007.gguf` | 16,835,109,952 | `e48c51a4314072344f39edeaaa0fb9e995a0683a6c0cfc5caa01e9e423b4c0dc` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00001-of-00006.gguf` | 49,055,528,160 | `65238b840dc795bf1f6b9370ecc1db3a52ca3fd5aed1bab0b4409153a3b90e51` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00002-of-00006.gguf` | 49,049,766,400 | `a091dafb5cafbe7bda3a21c2082714bd36bce78656ddb870b5647c618e99dcd4` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00003-of-00006.gguf` | 49,049,766,400 | `2da27d57e97a8448187d48f6b807259795c492a38ce00f3d55cdba0836969736` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00004-of-00006.gguf` | 31,213,487,776 | `59f96a0032be1e1481c349dfa0b7b4a4d1bc7a084f2396fd87432c7a76317226` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00005-of-00006.gguf` | 135,172,021,792 | `5f7c74342fa555a532d0a9664bb55ef1e61cea6a00e83f5bb35e6f7488204183` |
| `DeepSeek-V4.1-Flash-1017-Artemis-Q2_K-00006-of-00006.gguf` | 16,835,109,952 | `eb1aa8ca8c71efe49ec2d2277cb64c99a87b09815a8e54c0aa55943a2eea4393` |

</details>
<!-- v2-checksums:end -->

<!-- v2-footer:start -->
---

<div align="center">

[Artemis](https://www.turintech.ai/product/overview) is TurinTech's optimisation platform: teams use it to make their own models smaller and faster, or to optimise their code.

[All our models](https://www.turintech.ai/models) · [LinkedIn](https://www.linkedin.com/company/turintechai) · [Questions and requests](https://huggingface.co/spaces/turintech/README/discussions/1)

<sub>MIT, as released by DeepSeek ([licence](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE)); see `LICENSE` and `NOTICE`. An independent TurinTech release, not affiliated with or endorsed by DeepSeek. When serving these files to others, for example through an API, an inference provider or OpenRouter, we'd appreciate a mention of TurinTech (turintech.ai) as the source of the quantization.</sub>

These files keep improving.

</div>
<!-- v2-footer:end -->

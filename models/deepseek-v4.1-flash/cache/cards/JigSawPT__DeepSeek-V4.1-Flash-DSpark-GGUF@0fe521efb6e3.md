---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
library_name: llama.cpp
tags:
  - gguf
  - speculative-decoding
  - dspark
  - deepseek
  - llama.cpp
---

# DeepSeek-V4.1-Flash DSpark draft head, GGUF

The speculative-decoding draft head that ships inside DeepSeek-V4.1-Flash under the `mtp.*`
tensors, exported as a standalone DFLASH-architecture GGUF for llama.cpp: three full V4.1 blocks
(128 routed experts, 3 used), the Markov bigram head (rank 256), the confidence head, block size 5.
78 tensors, 7.97 GB.

**It runs only on the `dsv41-porte` branch of
[JigSawPT/llama.cpp](https://github.com/JigSawPT/llama.cpp)**, with the target model from
[JigSawPT/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-GGUF).
Upstream llama.cpp has no runtime for V4.1 yet.

## What it does for throughput, measured

On an RTX 5090 (31.8 GiB of VRAM) with the target streaming its experts from NVMe, the head is **neutral on
mixed content**: our benchmark (four prompts, three rounds, same day, same configuration)
gives 4.92 tokens/s with the draft against 5.13 without, and 21.23 against 21.40 on resident
content. Acceptance is high and tracks the content (51 % on Portuguese prose, 79 % on code, 97 %
on verbatim repetition), but a verification step pays for the union of experts of its K + 1
tokens, and on a disk-bound machine that cost eats the accepted tokens. Only verbatim repetition
gains (+12–15 %). The best draft length on this machine is 2, not the trained block of 5.

In a compute-bound regime, where a batched verification is nearly free, the same acceptance
would multiply throughput; that regime needs the working set on the card.

Full numbers and method: [the report](https://github.com/JigSawPT/deepseek-v41-flash-on-5090).

## Run

```
llama-server -m DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf -ngl 99 -c 8192 ^
  --moe-stream --moe-stream-cache 18 --moe-stream-l2 72 --reasoning off ^
  -md DeepSeek-V4.1-Flash-DSpark.gguf --spec-type draft-dspark --spec-draft-n-max 2 -ngld 99 --spec-draft-n-cpu-moe 3
```

`--spec-draft-n-cpu-moe 3` keeps the draft's experts in RAM so the target keeps its VRAM cache;
putting them on the card changes nothing and costs the target 5 GiB.

### If you quantize your own target

The draft reads the target's `token_embd`, so its precision changes acceptance. The target published in
[JigSawPT/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-GGUF) already keeps
`token_embd` and `output` at BF16 (1.32 GB each), and nothing needs doing. If you quantize a target yourself,
keep that tensor unquantized — `llama-quantize --token-embedding-type bf16`. Measured by
[@midagedev](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF/discussions/2) on a Q3_K_M target,
20 greedy 200-token prompts on ik_llama.cpp: acceptance 41.1 % / 54.8 % at n_max 5 / 3 with `token_embd` at Q3_K,
against 45.5 % / 60.1 % with it at BF16. The bf16 `output` on top added nothing. It costs 1.3 GB of RAM.

## How it was made

```
python convert_hf_to_gguf.py <DeepSeek-V4.1-Flash> --dspark --target-model-dir <DeepSeek-V4.1-Flash> --outtype bf16 --outfile DeepSeek-V4.1-Flash-DSpark.gguf
```

The converter class is `DeepseekV41DSparkModel`. It writes the V4.1 head names
(`markov_head.embed/head`), the draft's own expert counts, the target layers as the reference reads
them (the mean over the hyper-connection streams is taken *before* running layers 37–39), and a
`dflash.dsv41_semantics` key that tells the loader to apply the V4.1 rules: no per-head q norm,
the hyper-connection mix threaded one sub-layer ahead, and a final collapse with the last FFN mix
(the V4.1 head has no `hc_head` weights). Draft experts are MXFP4 as released; attention and dense
weights Q8_0/BF16 from fp8.

SHA-256 in `SHA256SUMS.txt`.

## Credits

DeepSeek for the model (MIT). Engineering assisted by Claude (Anthropic).

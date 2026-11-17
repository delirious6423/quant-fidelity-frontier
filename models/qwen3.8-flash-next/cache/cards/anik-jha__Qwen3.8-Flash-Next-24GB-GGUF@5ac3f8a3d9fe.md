---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - qwen
  - qwen4exp
  - mixture-of-experts
  - expert-pruning
  - reap
  - lora
  - 24gb
  - code
  - coding
  - reasoning
---

# Qwen3.8-Flash-Next for coding on a 24 GB GPU (expert-pruned GGUF + recovery adapter)

**Qwen3.8-Flash-Next with three quarters of its experts removed, sized for a 24 GB card.** Flash-Next is a
mixture-of-experts model: each layer holds 512 small sub-networks ("experts") and uses 10 of them for each
token. This repo keeps the 128 highest-ranked experts per layer from unsloth's 4-bit GGUF, puts back 2 more per
layer (retrained, 130 per layer in all), and adds a small correction file (the adapter) that llama.cpp loads with it.

**In our coding test it solved 109 of 130 contest problems, against 63 for the plain prune (no adapter, no restored experts).**

On our test machine the model and adapter used 21.1 GB of GPU memory with a 32K context (21.1 GB after
answering a prompt). That machine is an NVIDIA DGX Spark, where the CPU and GPU share one pool of memory. On a PC
with a separate 24 GB card we expect similar GPU memory, but we did not measure it there.

Flash-Next also has a 26.8 GB lookup table (an n-gram table), shipped here as its own file: for each token, the model
reads 16 of its rows. With the command below, llama.cpp leaves the table on disk and reads just those rows,
so it needs no GPU memory. The server process itself used 0.8 GB of RAM. Table rows it has read also sit in
the operating system's file cache, which the system frees when it needs the memory.

A model pruned this far often gets stuck repeating itself while it reasons. The recipe here has three
parts, and every score for this build was measured with all three:

1. **The adapter** (`--lora-scaled`): small corrections, trained to bring the pruned layers closer to the
   unpruned model. llama.cpp adds them while the model runs.
2. **Two sampling flags** (`--presence-penalty 1.5 --repeat-last-n 20480`).
3. **Stop and redraw** (`redraw.py`, included): a reply that starts repeating is stopped and drawn
   again with a new seed, up to three draws.

| | **This repo** |
|---|---|
| Model file + adapter (loaded onto the GPU) | **18.9 GB + 1.8 GB** |
| N-gram table file (stays on disk, read while it runs) | 26.8 GB |
| GPU memory at 32K context, 8-bit KV cache (measured at server start) | **21.1 GB** |
| GPU memory at 32K context, 16-bit KV cache (measured at server start) | 21.5 GB |
| GPU memory at 128K context, 8-bit KV cache (measured at server start) | 22.9 GB |
| GPU memory after a reply (32K context, 8-bit KV cache) | 21.1 GB |
| RAM held by the server after a reply (32K context, 8-bit KV cache) | 0.8 GB |
| Speed for one user on our test machine (DGX Spark) | 19 tokens per second |
| HumanEval+, 164 short Python problems (solved) | 89.6% with redraw, 88.4% on the first draw |
| MBPP+, 378 short Python problems (solved) | 75.4% with redraw, 74.1% on the first draw |

Sizes are in GB the way GPUs count them (1 GB = 1,024 x 1,024 x 1,024 bytes), so they compare directly with
your card. Hugging Face counts 1 GB as 1,000,000,000 bytes, so its file list shows the model file as
20.29 GB, the table as 28.80 GB and the adapter as 1.92 GB, a 51.0 GB download in all. unsloth's unpruned UD-IQ4_XS, which this build is cut from, is 87.2 GB. It
was not run on these benchmarks or measured on our machine, so this card makes no comparison with it.

## What the adapter and restored experts add (LiveCodeBench)

LiveCodeBench is a set of programming-contest problems, graded by running hidden tests on the answer.
We used 130 easy and medium problems (releases v1 to v3) that were set aside before this study ran any
model on them. Both rows use the same flags and the same stop-and-redraw client.

| | Solved of 130 | On the first draw |
|---|---|---|
| The plain prune (keeps 25%: 128 experts per layer, no adapter) | 63 | 59 |
| **This repo** (130 experts per layer + adapter) | **109** | **98** |

- **Gain:** +35.4 percentage points of problems solved (49 problems gained, 3 lost). The
  95% confidence range is +26.2 to +44.6 points. On the first draw alone: +30.0 points (+21.5 to +38.5).
- **By difficulty:** easy 61 of 63, medium 48 of 67 (plain prune: 45 of 63 and 18 of 67).
- **Not compared with the unpruned model:** it was not run on these problems, and hard problems were
  not run at all.

The same recipe, unchanged, was first tested on Ornith-1.5-122B, where it solved 116 of the same 130 problems against 104 for its plain prune:
[Ornith-1.5-122B-16GB-GGUF](https://huggingface.co/anik-jha/Ornith-1.5-122B-16GB-GGUF).

## Shorter coding problems (HumanEval+ and MBPP+)

| | Plain prune | **This repo** | Difference |
|---|---|---|---|
| HumanEval+ (164 problems) | 138 | **147** | +5.5 points (95% CI +0.6 to +10.4) |
| MBPP+ (378 problems) | 282 | **285** | +0.8 points (95% CI −2.6 to +4.2) |

Both with the same flags and redraw. These checked that this build (adapter plus restored experts) does not
hurt short problems. The MBPP+
difference is small enough to be chance, which does not prove the two are equal.

## Costs to know about

- **It was slower than the plain prune in our test:** the adapter is not merged into the quantized
  weights; llama.cpp computes its correction while the model runs. The LiveCodeBench test took
  2.5 times the wall time of the plain prune (31,202 s against 12,256 s, serving 16 requests at a time) for
  1.57 times the tokens.
- **It still loops sometimes:** 25 of 130 first draws were stopped for repeating (40 for the plain
  prune). After three draws, 9 problems still ended in a loop and 2 hit the token limit.
- **The table is read from disk:** on our NVMe drive, one user's speed was the same with the table on disk
  as fully in RAM (20.4 to 20.5 tokens per second across both, with a short prompt). Each prompt
  token reads 16 rows of the table, so very long prompts and slower drives may be slower; neither was tested.

## Files

| File | Use |
|---|---|
| `Qwen3.8-Flash-Next-24GB-REAP25-UD-IQ4_XS.gguf` | The pruned model, 130 experts per layer, without the n-gram table. Pruned from unsloth's UD-IQ4_XS (87.2 GB) |
| `Qwen3.8-Flash-Next-ngram-table-IQ4_NL.gguf_file` | Required. The n-gram table (26.8 GB), unchanged from unsloth's file. Loaded with `--ple` and read from disk. It ends in `.gguf_file`, as unsloth names its separate tables, so it is not listed as a model of its own |
| `recovery-adapter/Qwen3.8-Flash-Next-24GB-REAP25-adapter.gguf` | Required. The corrections to the 128 kept experts in every layer, added by llama.cpp while it runs |
| `redraw.py` | Stop-and-redraw client for llama-server. Python 3.9+, standard library only |
| `selections/flashnext-reap25-plus2.json` | The experts kept and restored in every layer |
| `LICENSE` | The Qwen Community License 1.0, unchanged from Qwen's repository |

`REAP25` means each layer kept 25% of its experts (128 of 512), the ones REAP, an expert-ranking method,
scored highest. Two more per layer were put back and retrained, so the file holds 130.

Every score and the speed were measured with the model as one file. The model file and the table file here hold the
same 1,224 weight arrays, byte for byte (all checked). Loaded as two files with llama.cpp b12465, the
model gave the same reply as the single file in a test with sampling turned off. On llama.cpp `36cbf415`, the build behind the scores, the same test gave a reply that diverged partway through, so results on b12465 may differ somewhat; the benchmarks were not re-run on it. The memory figures on
this card were measured with b12465 and the command below.

## Run it

Use a llama.cpp build that supports Qwen3.8-Flash-Next (architecture name `qwen4exp`); a build without it stops at
load with "unknown model architecture". `--ple`, which loads the table file, needs llama.cpp release b12233 or
newer; we checked this layout with release b12465. The scores were measured earlier with llama.cpp commit
`36cbf415` and one small local patch that returns malformed chat output as plain text instead of an HTTP
error. It triggered 0 times in those runs, so the plain commit would have behaved the same.

```bash
llama-server -m Qwen3.8-Flash-Next-24GB-REAP25-UD-IQ4_XS.gguf \
  --ple Qwen3.8-Flash-Next-ngram-table-IQ4_NL.gguf_file \
  --lora-scaled recovery-adapter/Qwen3.8-Flash-Next-24GB-REAP25-adapter.gguf:1 \
  -lm mmap -lzm on -ngl 99 -c 32768 --parallel 1 -ctk q8_0 -ctv q8_0 -fa on \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-last-n 20480

python3 redraw.py --url http://127.0.0.1:8080 --max-tokens 32768 "Write a Python function that ..."
```

- **Pass the table with `--ple`.** Without it, llama.cpp stops at load because the model file has no n-gram table.
- **Load the adapter.** `:1` after its path applies it at full strength, as in every run here. The two restored experts in each layer were trained together with it, and the
  GGUF on its own was not evaluated.
- **Keep `-lm mmap -lzm on`.** `-lm mmap` maps the file instead of copying it into memory, and `-lzm on` reads
  the table's rows from disk as they are needed. Without them, llama.cpp `36cbf415` loaded the whole table
  into RAM in our test (28.1 GB held by the server).
- **Keep the two penalty flags.** They are part of every score on this card.
- **Use `redraw.py`, or the same rule in your own client.** It streams the reply and stops it when
  the text ends in the same block six times in a row (blocks of up to 6,000 characters; for a short
  block, the repeats must cover at least 200 characters). A reply that was stopped, or that hit the token
  limit, is asked again with a new seed, for at most three draws. A reply that finishes is never
  redrawn. Chat apps do not do this for you; without it, expect the first-draw numbers, and looping
  replies that run to the token limit.
- Flash-Next reasons before answering by default. Every score here used temperature 0.6, top-p 0.95,
  top-k 20 and min-p 0, the settings of the Ornith-1.5-122B release this recipe comes from. The defaults stored in the
  GGUF (temperature 1.0, top-p 0.95, top-k 20) were not evaluated with this build.
- Set a token limit on replies: 32,768 tokens per draw in every run here, which is `redraw.py`'s default.
- Memory was measured with these settings: one slot, every layer on the GPU, 8-bit KV cache, flash
  attention on. With a 16-bit KV cache (drop `-ctk`, `-ctv`) it allocates 21.5 GB at 32K context.
  With the 8-bit cache at 128K context it allocates 22.9 GB, which leaves little room on a 24 GB card.
- Text only: no image encoder is included, and no extra prediction heads (MTP) for speculative decoding, a
  way to speed up generation. Image input was not tested.
- The source is unsloth's UD-IQ4_XS as published at revision `38bb39ee` (its 3 shards' hashes match
  the ones we recorded before training). unsloth has since replaced its quants with a newer version, so the
  files in their repo today are not the ones this build was cut from.
- LM Studio, Ollama and other llama.cpp versions have not been tested.

## How it was made

Qwen3.8-Flash-Next routes each token to 10 of 512 experts in each of its 48 MoE layers, plus one
shared expert that every token uses. Experts were ranked with REAP, which scores how much each expert adds
to its layer's output on sample text (3.15 million tokens), and the top 128 of 512 were kept
in every layer.

- **Kept experts are untouched.** The 128 kept experts, the router weights that choose them, and every other
  part of the file, the n-gram table included, are byte for byte unsloth's UD-IQ4_XS; nothing kept is re-quantized. Checked in
  two steps: the plain prune against unsloth's file (all 1,224 weight arrays), then this file against the plain
  prune (1,032 arrays whole and, in the 192 arrays that hold experts and router weights, the parts
  for the kept experts).
- **Two experts per layer are restored and retrained.** The next two experts in each layer's ranking
  were put back with their original router weights. Their weights were trained in full precision and
  quantized again with llama.cpp's quantizer, each tensor in the format unsloth's file uses for it.
- **The adapter corrects the kept experts.** Each kept expert gets a small low-rank correction (rank 8) to
  each of its three weight arrays. It lives in the adapter file and llama.cpp adds it while the model runs, so the
  quantized bytes stay as they are.
- **Training.** One layer at a time, the corrections and the restored experts were fitted so that the
  layer's combined expert output comes as close as possible to the unpruned model's (the same unsloth UD-IQ4_XS) on the same
  input. The data were calibration documents and the plain prune's own long
  reasoning replies to 124 other LiveCodeBench problems, none of them among the 130 test problems
  above. No coding score was used during fitting, and no setting was tuned on this model: every one
  comes from the Ornith study.

The shared expert, the attention layers, the embeddings, the n-gram table and the output head are untouched.

Method paper: [Reasoning Again with a Quarter of the Experts: Retry First, Then Train](https://arxiv.org/abs/2611.20419). Its first version reports the Ornith-1.5-122B result only; the Flash-Next
numbers on this card are not in it.

## Evaluation details

- **LiveCodeBench:** LiveCodeBench's own prompt, answer extraction and grader, served by llama-server with 16
  parallel slots, reasoning on, temperature 0.6, top-p 0.95, top-k 20, min-p 0, one fixed seed per
  problem and draw, up to 32,768 generated tokens per draw, at most three draws. The comparison, its
  rule for calling a winner and the problem list were fixed in advance in a file whose hash was recorded,
  so none could change after the results were seen. McNemar's test, a standard check for paired results,
  gives p < 0.0001: a gap this large would be very unlikely by chance.
- **A second, smaller check before the test:** on 64 other set-aside problems, each run twice with
  different seeds, the gain was +32.0 points (95% CI +22.7 to +41.4).
- **HumanEval+ (164 problems) and MBPP+ (378):** same server settings and redraw rule, answers
  cleaned to the code and scored with EvalPlus, the test runner for both. On the first draw the plain prune solved 132 and 261,
  this build 145 and 280.
- **Completeness:** every run of this build and of the plain prune behind this card finished every
  problem with 0 failed requests.
- **One run per problem for each model (up to three draws):** the confidence ranges show how much a result could move with a
  different set of problems; they do not include the randomness of generating again.
- **Memory and speed:** GPU memory of the llama-server process (nvidia-smi), model fully offloaded, one slot,
  read once the server was ready and before any request, measured on an NVIDIA DGX Spark, where the CPU and
  GPU share one memory pool. llama.cpp's own breakdown at 32K context with the 8-bit KV cache, printed before the
  adapter is loaded: 18,692 MB weights + 628 MB KV cache and recurrent state + 237 MB compute,
  below the totals above because the adapter is not yet counted (recurrent state is the memory of the model's
  recurrent layers).
  RAM is from a server started with the command above, after it answered one coding prompt; it is the server's
  own memory, while the table rows it reads sit in the operating system's file cache, which the system can
  reclaim. Speed was measured with the same settings on llama.cpp `36cbf415`, with the model as one file, while
  no other job used the GPU.
  Memory on an actual 24 GB discrete GPU was not measured; on a GPU that also drives a display, leave
  some headroom.

## Limitations

- Only coding was measured (LiveCodeBench easy and medium, HumanEval+, MBPP+). Agentic coding, tool
  calls, long-context use, vision, general knowledge and other human or programming languages were not, and may degrade
  more.
- The three-draw numbers need a client that stops and redraws.
- The costs listed above: slower generation in our test, loops that still occur, and an n-gram table
  read from disk.

## License and attribution

Unofficial community derivative of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), pruned from
[unsloth's GGUF quantizations](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF); not affiliated with or endorsed
by Qwen or Unsloth. Distributed under the Qwen Community License 1.0, the license of the original model; `LICENSE`
is Qwen's text, unchanged. Read it before commercial use: among its conditions, a product or service with more than
100 million monthly active users or US$20 million monthly revenue must display the model name, and a "Model as a
Service" or "AI Work Assistant" business (the license names AI-assisted coding products) needs a separate license
from Qwen for commercial use.

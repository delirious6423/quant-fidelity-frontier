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
  - 16gb
  - 2-bit
  - code
  - coding
  - reasoning
---

# Qwen3.8-Flash-Next for coding on a 16 GB GPU (2-bit expert-pruned GGUF + recovery adapter)

**A 2-bit build of Qwen3.8-Flash-Next for 16 GB GPUs.** It solved 93 of 130 coding contest problems in our
test, 85% as many as our [24 GB build](https://huggingface.co/anik-jha/Qwen3.8-Flash-Next-24GB-GGUF) (109). **If your GPU
has 24 GB, use the 24 GB build. If it has 16 GB, use this one.** To get these scores, run it with the included
adapter, two flags and redraw client (see Run it, below).

- **Fits in 14.6 GB of GPU memory** with a 32K context (see Memory, below).
- **Easy contest problems: 61 of 63 solved, the same as the 24 GB build.**
- **Close to the 24 GB build on short Python problems:** 140 against 147 of
  164 on HumanEval+, 274 against 285 of 378 on MBPP+.

**What it is.** Flash-Next is a mixture-of-experts model: each layer holds 512 small sub-networks ("experts")
and uses 10 of them for each token. We removed three quarters of them: each layer keeps
the 128 that a ranking method scored highest. These kept experts are byte for byte the same as in unsloth's
UD-IQ1_M GGUF ("2-bit" because its expert weights average 2.31 bits each). unsloth's file name says
IQ1_M, but this file has no 1-bit weights: 85% of its weights use 2-bit formats, and the
whole file averages 2.9 bits per weight. Our file name says IQ2_XXS, the smallest of those 2-bit
formats; IQ2_XXS and Q2_0 each hold 28% of the weights. To win back accuracy, we then added
two things, which we call the **recovery**: two of the removed experts in each layer were put back into the model file
and retrained (130 per layer in all), and a separate small correction file (the adapter) loads with the model.

**Memory.** With a 32K context (how much text it holds: prompt plus reply) and an 8-bit KV cache (the memory
that holds that text), the model and adapter used 14.6 GB of GPU memory. We measured this on an NVIDIA
DGX Spark, where the CPU and GPU share one pool of memory, not on a 16 GB card.

Flash-Next also has a 26.8 GB lookup table (an n-gram table), shipped here as its own file: for each token, the model
reads 16 of its rows. With the command below, llama.cpp leaves the table on disk and reads just those rows,
so it needs no GPU memory. Apart from GPU memory, the server process held 0.9 GB of RAM. Table rows it has read also sit in
the operating system's file cache, which the system frees when it needs the memory.

A model pruned this far often gets stuck repeating itself while it reasons. The recipe here has three
parts, and every score for this build was measured with all three:

1. **The adapter** (`--lora-scaled`): small corrections, trained so the pruned 2-bit layers give outputs closer to
   the unpruned 4-bit model's. llama.cpp adds them while the model runs.
2. **Two sampling flags** (`--presence-penalty 1.5 --repeat-last-n 20480`): they make the model less likely to
   reuse any token that appeared in the last 20,480 tokens.
3. **Stop and redraw** (`redraw.py`, included): a reply that starts repeating is stopped, and the whole reply is
   generated again with a new random seed, up to three tries in all.

| | **This repo** |
|---|---|
| Model file + adapter (loaded onto the GPU) | **12.1 GB + 1.8 GB** |
| N-gram table file (stays on disk, read while it runs) | 26.8 GB |
| GPU memory at 32K context, 8-bit KV cache (measured at server start) | **14.6 GB** |
| GPU memory at 32K context, 16-bit KV cache (measured at server start) | 15.1 GB |
| GPU memory at 128K context, 8-bit KV cache (measured at server start) | 16.5 GB |
| GPU memory after a reply (32K context, 8-bit KV cache) | 14.7 GB |
| RAM held by the server after a reply (32K context, 8-bit KV cache) | 0.9 GB |
| Speed for one user on our test machine (DGX Spark) | 23 tokens per second |
| HumanEval+, 164 short Python problems (solved) | 85.4% with redraw, 81.7% on the first draw |
| MBPP+, 378 short Python problems (solved) | 72.5% with redraw, 70.6% on the first draw |

Sizes are in GB the way GPUs count them (1 GB = 1,024 x 1,024 x 1,024 bytes), so they compare directly with
your card. Hugging Face counts 1 GB as 1,000,000,000 bytes, so its file list shows the model file as
13.05 GB, the table as 28.80 GB and the adapter as 1.92 GB, a 43.8 GB download in all.

## How it compares (LiveCodeBench)

LiveCodeBench is a set of programming-contest problems, graded by running hidden tests on the answer.
We used 130 easy and medium problems (releases v1 to v3) that were set aside before any model was run on them.
Every build here used the same flags and the same stop-and-redraw client.

Two things differ between the builds: the bit width (4-bit or 2-bit) and the recovery. A **plain prune** keeps the
same 128 experts per layer with no recovery. In the table, a fair comparison is between two cells that differ in
one thing only: along a row only the bit width changes, and down a column only the recovery changes.

| Solved of 130 | 4-bit | 2-bit |
|---|---|---|
| Plain prune | 63 (59) | not run on these problems |
| With the recovery | 109 (98), our [24 GB build](https://huggingface.co/anik-jha/Qwen3.8-Flash-Next-24GB-GGUF) | **93 (85), this repo** |

The first number allows up to three tries with redraw; the number in brackets is the first try alone.

- **The recovery is the big gain.** At 4-bit it took the plain prune from 63 to 109 solved
  (49 problems gained, 3 lost).
- **2-bit costs some, with or without the recovery.** With it, this repo solved 16 fewer
  than the 24 GB build. Without it, on 120 other problems, the plain 2-bit prune solved 31 and the
  plain 4-bit prune 53.
- **The recovery more than makes up for the drop to 2-bit.** This repo beats the plain 4-bit prune, 93
  against 63 (35 gained, 5 lost). This was the test we chose before running
  anything, and it passed by a wide margin. It changes both things at once, so it shows that the recovery gains more
  than 2-bit costs, not that 2-bit beats 4-bit.
- **By difficulty:** on easy problems this repo solved 61 of 63, the same as the 24 GB build. On medium
  problems it solved 32 of 67, against 48: the drop to 2-bit shows up there.
- **Not compared with the unpruned model:** it was not run on these problems, and hard problems were
  not run at all.

## Shorter coding problems (HumanEval+ and MBPP+)

| Solved | 4-bit, plain prune | 4-bit with the recovery (24 GB build) | **2-bit with the recovery (this repo)** |
|---|---|---|---|
| HumanEval+ (164 problems) | 138 | 147 | **140** |
| MBPP+ (378 problems) | 282 | 285 | **274** |

All with the same flags and redraw. On these short problems this repo stays close to the 24 GB build:
7 fewer solved on HumanEval+ and 11
fewer on MBPP+. Before the test we set the limit at 8 fewer and 18 fewer (5% of
each set), so it passed on both. The plain 2-bit prune was not run on these problems.

## Things to know

- **The 24 GB build is stronger** on contest problems (109 against 93 of 130), so use it if it
  fits your GPU.
- **Long replies:** like the 24 GB build, it thinks at length before it answers. Replies that ended normally, reasoning
  included, had a median length of 15,540 tokens (24 GB build:
  14,982). The LiveCodeBench test took 33,920 s (24 GB build:
  31,202 s), serving 16 requests at a time.
- **Loops, and how redraw helps:** 28 of 130 first replies were stopped for repeating
  (24 GB build: 25). Redraw tries those again, and the extra tries add
  8 solved problems (85 on the first try, 93 after up
  to three). After three tries,
  21 of the 130 problems ended their last try in a loop and
  5 more hit the token limit.
- **One check before the test missed its limit.** We checked that llama.cpp applies the adapter the same way as
  PyTorch, where it was trained. The adapter shifts the model's predictions, and the rule allowed the size of that
  shift to differ between the two programs by at most 5%. In PyTorch it was
  5.3% larger. The adapter file holds exactly the trained values, and
  we did not find the cause of the gap. We chose to run the test anyway and wrote that decision down before any test
  answer was generated. Every score on this card comes from llama.cpp, the program you run.
- **The table is read from disk:** each prompt token reads 16 rows of it, so very long prompts and slow drives
  may be slower; neither was tested with this build.

## Files

| File | Use |
|---|---|
| `Qwen3.8-Flash-Next-16GB-REAP25-IQ2_XXS.gguf` | The pruned model, 130 experts per layer, without the n-gram table. Pruned from unsloth's UD-IQ1_M (63.3 GB). Despite unsloth's name, it holds no 1-bit weights |
| `Qwen3.8-Flash-Next-ngram-table-IQ4_NL.gguf_file` | Required. The n-gram table (26.8 GB), unchanged from unsloth's file and the same file as in the 24 GB build. Loaded with `--ple` and read from disk |
| `recovery-adapter/Qwen3.8-Flash-Next-16GB-REAP25-adapter.gguf` | Required. The corrections to the 128 kept experts in every layer, added by llama.cpp while it runs. Not interchangeable with the 24 GB build's adapter |
| `redraw.py` | Stop-and-redraw client for llama-server. Python 3.9+, standard library only |
| `selections/flashnext-reap25-plus2.json` | The experts kept and restored in every layer (the same as in the 24 GB build) |
| `LICENSE` | The Qwen Community License 1.0, unchanged from Qwen's repository |

`REAP25` means each layer kept 25% of its experts (128 of 512), the ones REAP, an expert-ranking method,
scored highest. Two more per layer were put back and retrained, so the file holds 130.

Every score was measured with llama.cpp `36cbf415` and the model as one file. This repo splits that file
in two, the model file and the table file; together they hold exactly the same 1,224 weight arrays, byte
for byte (all checked). On llama.cpp b12465, the two files and the single file gave the same reply in a test with
sampling turned off. On `36cbf415`, the single file's reply in that test differed from partway through, so scores on b12465 may differ somewhat; we did not re-run the benchmarks on it. The memory and speed figures on this card were measured with b12465 and the
command below.

## Run it

Use a llama.cpp build that supports Qwen3.8-Flash-Next (architecture name `qwen4exp`); a build without it stops at
load with "unknown model architecture". `--ple`, which loads the table file, needs llama.cpp release b12233 or
newer; we checked this layout with release b12465. The scores were measured earlier with llama.cpp commit
`36cbf415` and one small local patch that returns malformed chat output as plain text instead of an HTTP
error. It triggered 0 times in those runs, so the plain commit would have behaved the same.

```bash
llama-server -m Qwen3.8-Flash-Next-16GB-REAP25-IQ2_XXS.gguf \
  --ple Qwen3.8-Flash-Next-ngram-table-IQ4_NL.gguf_file \
  --lora-scaled recovery-adapter/Qwen3.8-Flash-Next-16GB-REAP25-adapter.gguf:1 \
  -lm mmap -lzm on -ngl 99 -c 32768 --parallel 1 -ctk q8_0 -ctv q8_0 -fa on \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-last-n 20480

# then, in a second terminal while the server runs:
python3 redraw.py --url http://127.0.0.1:8080 --max-tokens 32768 "Write a Python function that ..."
```

- **Pass the table with `--ple`.** Without it, llama.cpp stops at load because the model file has no n-gram table.
- **Load this repo's adapter.** `:1` after its path applies it at full strength, as in every run here. The two restored
  experts in each layer were trained together with it, and the GGUF on its own was not evaluated. The 24 GB build's
  adapter (`Qwen3.8-Flash-Next-24GB-REAP25-adapter.gguf`) has the same shape but was trained for the 4-bit weights.
- **Keep `-lm mmap -lzm on`.** `-lm mmap` maps the file instead of copying it into memory, and `-lzm on` reads
  the table's rows from disk as they are needed.
- **Keep the two penalty flags.** They are part of every score on this card.
- **Use `redraw.py`, or the same rule in your own client.** It streams the reply and stops it when
  the text ends in the same block six times in a row (blocks of up to 6,000 characters; for a short
  block, the repeats must cover at least 200 characters). A reply that was stopped, or that hit the token
  limit, is asked again with a new seed, for at most three draws. A reply that finishes is never
  redrawn. Chat apps do not do this for you; without it, expect the first-draw numbers, and looping
  replies that run to the token limit.
- Flash-Next reasons before answering by default. Every score here used temperature 0.6, top-p 0.95,
  top-k 20 and min-p 0. The defaults stored in the GGUF (temperature 1.0, top-p 0.95, top-k 20) were not
  evaluated with this build.
- Set a token limit on replies: 32,768 tokens per draw in every run here, which is `redraw.py`'s default.
  In the scored runs each request had 36,864 tokens of context: that limit plus 4,096 for the prompt.
  With `-c 32768` the prompt and the reply share 32,768, so a very long reply stops sooner. To match the scored
  runs, use `-c 36864`: 14.7 GB of GPU memory at server start, measured the same way.
- Memory was measured with these settings: one slot (`--parallel 1`), every layer on the GPU (`-ngl 99`),
  8-bit KV cache (`-ctk q8_0 -ctv q8_0`) and flash attention on (`-fa on`). With a 16-bit KV cache (drop `-ctk`, `-ctv`) it allocates 15.1 GB at 32K context.
  With the 8-bit cache at 128K context it allocates 16.5 GB, more than a 16 GB card holds.
- Text only: no image encoder is included, and image input was not tested. The MTP layers some llama.cpp
  setups use to generate faster are not included either.
- The source is unsloth's UD-IQ1_M as published at revision `f5d175de` (its 3 shards' hashes matched the
  Hub's when we downloaded them).
- LM Studio, Ollama and other llama.cpp versions have not been tested.

## How it was made

Qwen3.8-Flash-Next routes each token to 10 of 512 experts in each of its 48 MoE layers, plus one
shared expert that every token uses. Experts were ranked with REAP, which scores how much each expert adds
to its layer's output on sample text (3.15 million tokens), and the top 128 of 512 were kept
in every layer. The kept and restored experts are the same as in the 24 GB build.

- **Kept experts are untouched.** The 128 kept experts, the router weights that choose them, and every other
  part of the file, the n-gram table included, are byte for byte unsloth's UD-IQ1_M; nothing kept is re-quantized. Checked in
  two steps: the plain 2-bit prune against unsloth's file (all 1,224 weight arrays), then this file against the plain
  prune (1,032 arrays whole and, in the 192 arrays that hold experts and router weights, the parts
  for the kept experts).
- **Two experts per layer are restored and retrained.** The next two experts in each layer's ranking
  were put back with their original router weights. Their weights were trained in full precision and
  quantized again with llama.cpp's quantizer, each tensor in the format unsloth's UD-IQ1_M file uses for it.
- **The adapter corrects the kept experts.** Each kept expert gets a small low-rank correction (rank 8) to
  each of its three weight arrays. It lives in the adapter file and llama.cpp adds it while the model runs, so the
  quantized bytes stay as they are. It was trained anew for this build, not copied from the 24 GB one.
- **Training.** One layer at a time, the corrections and the restored experts were fitted so that the
  layer's combined expert output comes as close as possible to the unpruned 4-bit model's (unsloth's UD-IQ4_XS at revision
  `38bb39ee`, 87.2 GB) on the same input. So the adapter corrects both the pruning and the 2-bit rounding.
  The data were the 24 GB build's: calibration documents and the plain 4-bit prune's long reasoning replies to
  124 other LiveCodeBench problems, none of them among the 130 test problems above. No coding score was used
  during fitting, and every setting is the 24 GB build's.

The shared expert, the attention layers, the embeddings, the n-gram table and the output head are untouched.

Method paper: [Reasoning Again with a Quarter of the Experts: Retry First, Then Train](https://arxiv.org/abs/2611.20419). Its first version reports the Ornith-1.5-122B result only; the Flash-Next
numbers on this card are not in it.

## Evaluation details

- **LiveCodeBench:** LiveCodeBench's own prompt, answer extraction and grader, served by llama-server with 16
  parallel slots, reasoning on, temperature 0.6, top-p 0.95, top-k 20, min-p 0, one fixed seed per
  problem and draw, up to 32,768 generated tokens per draw, at most three draws. The comparison with the plain
  4-bit prune, its rule for calling a winner and the problem list were fixed in advance in a file whose hash was
  recorded, so none could change after the results were seen. McNemar's test, a standard check for paired results,
  gives p < 0.0001: a split of gains and losses this lopsided would be very unlikely if the two builds
  were equally good.
- **Differences in points** (percentage points of the problem set), with 95% confidence ranges (how far a result could
  plausibly move with a different set of problems):
  - This repo against the plain 4-bit prune (chosen in advance): +23.1 points (range +14.6 to +31.5). The pass mark was a gain of at
    least 2 points with the whole range above zero.
  - This repo against the 24 GB build, not chosen in advance: −12.3 points (range −20.0 to −4.6).
  - The recovery at 4-bit (the 24 GB build against the plain 4-bit prune): +35.4 points (range +26.2 to +44.6).
  - The plain 2-bit prune against the plain 4-bit prune, on 120 other problems: −18.3 points (range −26.7 to −10.0).
  - HumanEval+ and MBPP+, this repo against the 24 GB build (chosen in advance, limit 5 points):
    −4.3 points (range −9.2 to +0.6) and −2.9 points (range −6.1 to +0.3). The rule was on the measured difference; the
    ranges reach past 5 points, so it shows the loss is small, not that the two are equal.
- **Reused runs:** the plain 4-bit prune's and the 24 GB build's results are the runs behind the 24 GB card, not new
  runs. Before this build ran, each was checked to use the same problems, seeds, sampling settings, server flags,
  token limit and redraw rule, and the 62 files both runs depend on (the llama.cpp build among them) hash the same.
- **HumanEval+ (164 problems) and MBPP+ (378):** same server settings and redraw rule, answers
  cleaned to the code and scored with EvalPlus, the test runner for both. On the first draw this build solved
  134 and 267.
- **The adapter check:** at full strength the adapter shifted the model's predictions too little to compare the two programs
  reliably, so the rule compared them at 1.5x strength, as it was written to do.
- **Completeness:** every run behind this card finished every problem with 0 failed requests.
- **One run per problem for each model (up to three draws):** the confidence ranges show how much a result could move with a
  different set of problems; they do not include the randomness of generating again.
- **Memory and speed:** GPU memory of the llama-server process (nvidia-smi), model fully offloaded, one slot,
  read once the server was ready and before any request, measured on an NVIDIA DGX Spark, where the CPU and
  GPU share one memory pool. llama.cpp's own breakdown at 32K context with the 8-bit KV cache, printed before the
  adapter is loaded: 12,108 MB weights + 628 MB for the KV cache and other per-conversation
  state + 237 MB of working memory, below the totals above because the adapter is not
  yet counted. RAM is from a server started with the command above, after it answered one coding prompt; it is
  the server's own memory, while the table rows it reads sit in the operating system's file cache, which the system can
  reclaim. Speed is from the same server and reply, while no other job used the GPU.
  Memory on an actual 16 GB GPU was not measured; on a GPU that also drives a display, leave some headroom.

## Limitations

- Only coding was measured (LiveCodeBench easy and medium, HumanEval+, MBPP+). Agent-style coding (the
  model editing files and running tools itself), tool calls, long-context use, vision, general knowledge and other human or programming languages were not, and may degrade
  more.
- The three-draw numbers need a client that stops and redraws.
- The points under Things to know, above: the 24 GB build is stronger, replies are long, loops still occur, one
  adapter check missed its limit, and the n-gram table is read from disk.

## License and attribution

Unofficial community derivative of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), pruned from
[unsloth's GGUF quantizations](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF); not affiliated with or endorsed
by Qwen or Unsloth. Distributed under the Qwen Community License 1.0, the license of the original model; `LICENSE`
is Qwen's text, unchanged. Read it before commercial use: among its conditions, a product or service with more than
100 million monthly active users or US$20 million monthly revenue must display the model name, and a "Model as a
Service" or "AI Work Assistant" business (the license names AI-assisted coding products) needs a separate license
from Qwen for commercial use.

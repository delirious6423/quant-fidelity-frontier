---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
  - gguf
  - llama.cpp
  - moe
  - engram
language:
  - en
---

# DeepSeek-V4.1-Flash-1017 GGUF

Text-only GGUF conversion of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), built for llama.cpp. The upstream checkpoint is multimodal: the vision tower is not converted and image input is not supported here.

**`rq2/` is not usable.** It is published as the record of a failed rung, not as a model to run. Use `q4/` (429.2 GB) for anything real. Read the rq2 section before downloading anything smaller than `q4/`.

## The model in one paragraph

40 MoE blocks, 384 routed experts, 6 active per token, 1 shared expert, hidden size 5120. On top of the experts sits an engram module: 196.6B parameters in a single `per_layer_token_embd` tensor of shape 768,022,850 by 256, keyed by a 16M entry vocabulary over 4-grams. The routed experts hold 543.6B parameters and are stored as MXFP4 in every folder below, so they are identical across the ladder and the engram plus the dense path is the only thing that changes. Context is advertised at 1,048,576; nothing here was exercised past 4,096.

## Folders

Each folder is a complete model. Download one, not the repository; the six together are 2.46 TB.

| folder | shards | bytes | engram | decode t/s | status |
| --- | --- | --- | --- | --- | --- |
| `q8` | 18 | 506.7 GB | Q8_0, 208.9 GB | 10.5 (11.6 at 4 pinned) | works, and it is the source every other folder was requantised from |
| `q6` | 18 | 457.1 GB | Q6_K, 161.3 GB | not served | unprobed |
| `q4` | 18 | 429.2 GB | Q5_K, 135.2 GB | 13.2 (14.0 at 4 pinned) | works, this is the recommended rung |
| `q3` | 17 | 403.6 GB | Q4_K, 110.6 GB | not served | unprobed, and the one rung that could still work below q4 |
| `q2` | 17 | 376.7 GB | Q3_K, 84.5 GB | not served | shares the engram type that breaks rq2, assume broken until probed |
| `rq2` | 13 | 284.7 GB | Q3_K, 84.5 GB | 16.8 (19.8 at 8 pinned) | broken, see below |

Engram sizes are the stored bytes for the `per_layer_token_embd` tensor alone and are the reason the ladder cannot go much lower: it is 41 percent of the q8 file.

## Why the size floor is where it is

The expert half is fixed. MXFP4 experts plus their block scales are 288 GB of every folder, so the expert quant is not where a smaller download is won. The engram is: Q8_0 is 208.9 GB, Q5_K is 135.2 GB, Q4_K is 110.6 GB, Q3_K is 84.5 GB. Shrinking it from Q8_0 to Q3_K saves 124 GB, and below Q4_K it stops working.

## rq2: the 284.7 GB rung that breaks

**Recipe.** `llama-quantize --allow-requantize q8_0.gguf rq2.gguf Q2_K` with the token embedding set to Q3_K, 64 threads: 1043 tensors, 271,513.64 MiB out at 3.04 BPW, 803,974 ms. Experts went to Q2_K, the engram from q8_0 to q3_K (199,224.68 MiB to 80,568.80 MiB). This is the only published folder with experts below MXFP4.

**Symptom.** With the chat template applied the model does not answer. It restates the prompt and emits template scaffolding, and continues to max_tokens. Raw completion on the same file produces fluent English, so the file loads, the graph runs and the sampler is fine. The instruction path is what collapses.

**Control.** q8_0, same machine, same template, same flags, port 18082: answers all five items of the hard probe in 60 generated tokens. Temperature 0 throughout.

**Isolation.** `rq2e8` is rq2 with one thing changed, the engram put back to Q8_0 (208.9 GB, total 409.5 GB at 4.38 BPW). Experts stayed at Q2_K. It answers item 1 correctly and stays on topic. So Q2_K experts on their own are survivable, and the engram at Q3_K is the part that breaks the model.

**Second defect.** rq2e8 holds for roughly 400 tokens and then falls into a repetition loop. Q2_K experts are borderline even with a clean engram, which is why the expert branch is not published at all.

**Conclusion.** The engram floor for this conversion is Q4_K. `q4/` clears it at 429.2 GB. `q3/` sits exactly on it and has not been served. `q2/` sits below it with the same engram as rq2 and should be treated as broken until someone proves otherwise.

## Measured decode throughput

A40 48 GB, 2 TB system RAM, 32 threads, 4096 context, single request, 128 generated tokens, temperature 0. llama.cpp 0.7.0-dev, commit d854d98, CUDA 12.9, GNU 13.3.0. Experts of blocks 0 to N-1 pinned to VRAM with `-ot`, remaining expert blocks in system RAM.

| model | `-cmoe`, no pinning | 4 blocks pinned | 8 blocks pinned |
| --- | --- | --- | --- |
| `q8` | 10.5 | 11.6 | not run |
| `q4` | 13.2 | 14.0 | not run |
| `rq2` | 16.8 | 15.9 at 6 pinned | 19.8 |
| `rq2e8` | 20.8 | not run | 19.5 |

One anomaly is left standing: the 409.5 GB rq2e8 variant decodes faster than the 284.7 GB rq2 variant, 20.8 against 16.8 t/s. Size does not explain it; the likely cause is Q2_K expert dequantisation cost on the CPU path, and it was not isolated. See gap 6 below.

## Reproduce

```
llama-server -m <folder>/<first-shard>.gguf -ngl 99 -c 4096 -t 32 --jinja -np 1 \
  -cmoe -ot 'per_layer_token_embd\.weight=CPU'
```

To pin expert blocks 0 to 3 to VRAM and leave the rest in RAM:

```
llama-server -m <folder>/<first-shard>.gguf -ngl 99 -c 4096 -t 32 --jinja -np 1 \
  -ot 'per_layer_token_embd\.weight=CPU,blk\.(4|5|6|...|39)\.ffn_(gate|up|down)_exps\.weight=CPU'
```

The engram is always pinned to CPU. At 84.5 to 208.9 GB it does not fit in 48 GB of VRAM, and it is the tensor that has to be fed from system RAM.

## What is missing, and what documenting this properly would take

This repository is a successful conversion and one negative result. It is not an evaluation. Nothing below has been measured for any folder, so every rung except q8 should be assumed to be some unknown distance from the reference in quality.

1. **A quality number.** No perplexity, no KL divergence and no benchmark score exists for any folder. Required: `llama-perplexity` over a fixed held-out slice (a few MB of enwik8 plus a code slice) for each rung, so the ladder has a curve instead of two anecdotes. Cheap, one GPU hour per rung.
2. **A reference for that number.** KL divergence needs logits from the fp8 checkpoint, and `deepseek_v41` is a new architecture: it needs the upstream modeling code, not whatever transformers release is installed. Required: run the reference once on a short slice, save the logits, reuse them for every rung.
3. **Per-tensor attribution.** The break is currently located by two endpoints, rq2 and rq2e8, which differ in two tensors at once. Required: engram at Q8_0, Q6_K, Q5_K, Q4_K and Q3_K with experts pinned at MXFP4 throughout, one probe each, so the exact engram type where output dies is measured rather than inferred.
4. **The untested rung that matters.** `q3/` (403.6 GB, engram Q4_K, MXFP4 experts) has never been served. It is the cheapest possible working download if it holds, and one serve plus the probe settles it in under an hour. This is the single highest value missing run.
5. **A real task battery.** The probe used here is a smoke test: five items, pass or fail, read by eye. Required: an established set with a scored rubric, a pinned prompt format, temperature 0, a fixed seed and a fixed max_tokens, reported per item with the sampling settings written down. Candidate small suites: a coding set, a grade school math set, a tool-call schema set.
6. **Degeneracy measurement.** rq2e8 loops after about 400 tokens. That is one observation on one prompt. Required: a loop rate over 20 or more prompts with a stated detector (repeated n-gram, or distinct-n), per rung.
7. **A throughput protocol.** Prompt length, generated length, prefix cache state, pinned layer count, thread count, repeat count. Right now the numbers in the table above come from different pin sets and cannot be compared across rungs. Required: one harness that fixes all of it and reruns every rung, plus a second run to show run to run spread.
8. **Artifacts in the repo.** The exact `llama-quantize` command per folder, the llama.cpp commit, the quantise log and a sha256 per shard. Today the recipes live in a private plan file, and the sizes in the table above were re-derived from the published shards, which is how an earlier draft of this card ended up quoting the wrong byte count for `q4`.
9. **Provenance.** The fp8 checkpoint revision, the converter version, and an explicit statement that the vision tower was dropped on purpose rather than lost.
10. **Scope.** Text only, llama.cpp only, one 48 GB GPU with 2 TB of system RAM, 4096 context. No claim is made about the advertised 1M context, about multi-GPU behaviour, or about any runtime other than llama.cpp.

## Licence and attribution

MIT, inherited from the upstream release. This is an unofficial third-party conversion and not a DeepSeek release. Architecture and weights belong to DeepSeek; conversion errors are ours.

## Session record

Converted and quantised on one rented A40 in October 2026. Total spend for the conversion, the ladder, the rq2 failure, the rq2e8 isolation and the loss curve: USD 10.23. The `rq2` folder stays up deliberately as the record of the negative result.

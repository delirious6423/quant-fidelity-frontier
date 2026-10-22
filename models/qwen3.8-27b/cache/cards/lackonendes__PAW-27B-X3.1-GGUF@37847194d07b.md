---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
tags:
- quantization
- trellis-codec
- paw
- gguf
---

# PAW-27B-X3.1

_If this model is useful to you, support the work by following [@nawariokra](https://x.com/nawariokra) on X, or buy me a coffee on [Ko-fi](https://ko-fi.com/lackonendes)._

<a href="https://ko-fi.com/F4P3296XP7"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="32"></a>


Run `Qwen/Qwen3.8-27B` with good quality on a single RTX 3090 (24 GB VRAM) at full 262,144-token context —
129 tok/s generation, up to 1268 tok/s prompt processing. This is an update to our earlier PAW-27B-X3
release, with lower quality loss at the same file sizes.

Requires [`llama-paw`](https://github.com/irawanw/llama-paw) — stock llama.cpp cannot read these files.

## Demos

Two example scenes, generated in one shot from a text prompt (three.js/WebGL code, output by this model,
rendered in a browser):

**Eiffel Tower voxel garden**

<video controls width="100%" src="https://huggingface.co/lackonendes/PAW-27B-X3.1-GGUF/resolve/main/assets/eiffel_tower.mp4"></video>

**Rocket launch scene**

<video controls width="100%" src="https://huggingface.co/lackonendes/PAW-27B-X3.1-GGUF/resolve/main/assets/rocket_launcher.mp4"></video>

**Recommended: the 3.5 bpw file.** It's the most tested and has full benchmark results below.

| | PAW-27B-X3.1 (3.5 bpw) |
|---|---|
| size | 11.50 GiB |
| context | 262,144 tokens |
| speed (coding, RTX 3090) | 129.09 tok/s |
| benchmarks | see below |

2.5 bpw, 3.0 bpw, 2.0 bpw and 1.5 bpw files are also available for smaller/faster setups; no benchmark
results yet.

## Benchmarks

Benchmark scores below are the best result across several passes with increasing "thinking" allowance
(more time/tokens to reason before answering). An item counts as correct if it was solved on any pass —
this is an optimistic score, not the same as a single fixed-setting run.

### Retention against the published BF16 model

Baseline: the **"Base" column** of the
[ThinkingCap-Qwen3.8-27B card](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B), which is
`Qwen/Qwen3.8-27B` in full BF16 precision — the model PAW quantizes. We did not run BF16 ourselves; these
are the publisher's own numbers. Retention = this model's score divided by that published score.

| benchmark | BF16 Base (published) | PAW-27B-X3.1 | retention |
|---|---:|---:|---:|
| AIME 2026 | 98.13 | 96.67 (29/30) | **98.5%** |
| GPQA-Diamond | 89.93 | 90.4 (179/198), 11 items still pending | 100.5% |
| MMLU-Pro | 85.54 | 87.2 (436/500), 20 items still pending | 101.9% |
| LiveCodeBench v6 | 91.14 | 88.6 (155/175), **running** (phase 5 of 6, 19/22 done) | 97.2% (floor, partial) |
| IFBench | 79.75 | 78.0 (117/150), 6 items still pending | 97.8% |

**Read this as a rough guide, not a precise comparison.** The baseline is an average over many runs on the
full benchmark; this model's score is from a single run on a subset, taking the best result across thinking
passes (which inflates the score somewhat). Several rows are floors, not final numbers — some items haven't
finished running yet and will be added once they do. LiveCodeBench in particular is still mid-run and its
retention number will keep rising. MMLU-Pro's number above 100% is a measurement-method quirk, not evidence
this model beats the original. AIME 2026's number rests on only 30 items, so treat it as noisy.

## Files

| file | rate | size | notes |
|---|---:|---:|---|
| [`PAW-27B-X3.1.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3.1-GGUF/resolve/main/PAW-27B-X3.1.gguf) | 3.500 bpw | 11.50 GiB | **recommended**, benchmarked above |
| [`PAW-27B-X3.1-3.0bpw.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3.1-GGUF/resolve/main/PAW-27B-X3.1-3.0bpw.gguf) | 3.000 bpw | 10.10 GiB | no benchmarks yet |
| [`PAW-27B-X3.1-2.5bpw.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3.1-GGUF/resolve/main/PAW-27B-X3.1-2.5bpw.gguf) | 2.500 bpw | 8.67 GiB | no benchmarks yet |
| [`PAW-27B-X3.1-2.0bpw.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3.1-GGUF/resolve/main/PAW-27B-X3.1-2.0bpw.gguf) | 2.000 bpw | 7.26 GiB | no benchmarks yet |
| [`PAW-27B-X3.1-1.5bpw.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3.1-GGUF/resolve/main/PAW-27B-X3.1-1.5bpw.gguf) | 1.500 bpw | 5.85 GiB | no benchmarks yet |
| [`Qwen3.8-27B-DFlash2-Q2_K.gguf`](https://huggingface.co/lackonendes/PAW-27B-X3.1-GGUF/resolve/main/Qwen3.8-27B-DFlash2-Q2_K.gguf) | — | 0.67 GiB | optional speculative drafter, speeds up generation |

Integrity (SHA-256):

```
4acefd28b247259cd25367ad7915ec6142b5f7a966d69b71599802fa63352599  PAW-27B-X3.1.gguf
61377ea0140f612717cf4f9bcd2fd6990eb62896ae7070bb7a96b634c9e1f64b  PAW-27B-X3.1-2.5bpw.gguf
ef41b977411cfb9142db047a7e925df0d405700d36bb3381a8ef895b98b4e626  PAW-27B-X3.1-3.0bpw.gguf
f4c0de5166df423a90d410cf050d1eea6595b633ce08bd8f938730f997f260ec  PAW-27B-X3.1-2.0bpw.gguf
ac3e8df9cd72bf63bb81e75db39d86fcb7a299a2abbb6518887da1e50d4449d9  PAW-27B-X3.1-1.5bpw.gguf
```

## Speed (3.5 bpw file, one RTX 3090)

| workload | tok/s |
|---|---:|
| generation, coding (drafter on) | 129.09 |
| generation, new code from a long prompt (drafter on) | 73.74 |
| generation, no drafter | 30.16 |
| generation, long-context thinking | ~48-58 |

Generation speed depends on what's being written: copying/near-copying code is fastest, writing new code is
slower, and step-by-step thinking is slower still. Speed for the other file sizes hasn't been measured yet —
don't assume they match.

### Prompt processing / generation throughput

| case | tok/s |
|---|---:|
| Prompt processing, short prompt | 982.87 |
| Prompt processing, 8k-token prompt | up to **1268.67** (peak) |
| Generation, no drafter | 31.17 |

## Serving

Build [`llama-paw`](https://github.com/irawanw/llama-paw) from `main` with CUDA, then run:

```sh
./build/bin/llama-server -m PAW-27B-X3.1.gguf \
  -md Qwen3.8-27B-DFlash2-Q2_K.gguf \
  --spec-type draft-dflash --spec-draft-n-max 5 \
  -fa on -ctk q4_0 -ctv q4_0 \
  -c 262144 -ub 1024 -b 8192 -ubd 256 \
  -ngl 99 -np 1 --no-warmup \
  --reasoning off --reasoning-effort xhigh --reasoning-budget -1 \
  --host 127.0.0.1 --port 8080
```

- The drafter file (`-md ...`) is optional but recommended — it roughly doubles generation speed.
- Thinking is turned on/off per request via the API's `enable_thinking` option.
- Drop `-c 262144` to a smaller context size if you don't need the full 256k window; it uses less memory.

## Limitations

- Speed numbers above are for the 3.5 bpw file only. The other file sizes are unmeasured — don't assume they
  match.
- The 3.0, 2.5, 2.0 and 1.5 bpw files have no benchmark results yet.
- LiveCodeBench is still running (phase 5 of 6, 155/175 done, 19/22 through phase 5); its retention number
  is a floor and will rise as it finishes. A handful of items on GPQA-Diamond, MMLU-Pro and IFBench
  are also still pending and will be added once complete.
- This is a code-focused build; general-purpose quality is unverified.
- One AIME item ends without a final answer; how that item is counted affects the AIME score slightly.

## License

Apache 2.0, inherited from the base model `Qwen/Qwen3.8-27B`.

## Credits and attribution

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp: the trellis quantization codec this
  release is built on.
- Earlier PAW-27B-X3 release and credits: [PAW-27B-X3-GGUF](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF).
- Base model [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen.

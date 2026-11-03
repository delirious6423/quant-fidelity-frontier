---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
  - gguf
  - llama.cpp
  - cpu
  - avx512-vnni
  - amx
  - qwen35moe
  - mixture-of-experts
---

# Ornith-1.5-35B-A3B-GGUF-OPT

CPU-optimized build of **Ornith-1.5-35B-A3B** (Q4_K_M), for Intel Xeon with AVX512-VNNI / AMX.

- The GGUF is the unmodified `Ornith-1.5-35B-Q4_K_M.gguf` from [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) (sha256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`).
- All speedups come from a llama.cpp patch (`patches/qwen36-q4km-cpu-opt.patch`, base commit `de5122d87`). The patch is from [HelloSun/Qwen3.6-35B-A3B-GGUF_CPU_OPT](https://huggingface.co/HelloSun/Qwen3.6-35B-A3B-GGUF_CPU_OPT) and applies unchanged to this model, because both use the `qwen35moe` architecture and the same quant mix (Q4_K, Q6_K, F32).
- Optimized and measured on **Intel Xeon Platinum 8559C, 16 threads** (cgroup quota). Other CPUs are not tested. Build with `-DGGML_NATIVE=ON`.

## Results

Same machine, same GGUF, same flags (`taskset -c 0-15 -t 16 -fa 1 -b 1024 -ub 1024 -lm mmap+mlock`). The host is shared, so runs vary by roughly 5-10%. Stock and patched runs were interleaved.

| build | pp512 (t/s) | pp2048 (t/s) | tg128 (t/s) |
|---|---|---|---|
| stock llama.cpp `de5122d87` (3 reps) | 195.4 ± 10.4 | 194.6 ± 13.6 | 27.2 ± 0.2 |
| patched, run 1 (3 reps) | 268.4 ± 57.7 | 327.6 ± 10.3 | 44.2 ± 0.9 |
| patched, run 2 (3 reps) | - | 320.2 ± 6.3 | 42.8 ± 1.3 |

Speedup vs stock is about **1.65x prefill** and about **1.6x decode**.

Accuracy (`llama-perplexity`, 16 chunks x 512 tokens of llama.cpp docs text, same corpus for both):

| build | PPL |
|---|---|
| stock | 6.0255 ± 0.2395 |
| patched | 6.0566 ± 0.2418 |

The difference (+0.03, about 0.5%) is well inside the standard error. The patched build is not bit-exact with stock. Integer dot products are exact, but float accumulation order differs.

Runtime flags: `-b 1024 -ub 1024` was about 7% faster on pp2048 than `-b 2048 -ub 2048` in an interleaved check (tg unchanged). 16 threads beat 8 and 12 threads.

## What was evaluated and not enabled by default

- **MTP speculative decoding (`--spec-type draft-mtp`)**: this GGUF contains the MTP head (`blk.40.nextn.*`), so it works without an extra file. It is **slower** on this CPU: 41.9 t/s baseline vs 32.4 (n=1), 34.7 (n=2), 32.7 (n=3), even at 63-83% acceptance. Greedy outputs also differ from the non-spec baseline at the same character positions for every draft length. This is probably batch-shape numerics, not root-caused. Details: `benchmarks/mtp_summary.md`. Enable with `MTP=1` in `build.sh` if your workload differs.
- **DFlash (`ornith-ai/Ornith-1.5-35B-A3B-DFlash`)**: supported, but slower than no-spec on this CPU. The draft is a safetensors checkpoint with no `DFlashDraftModel` converter in llama.cpp at this commit, so `tools/convert_dflash_to_gguf.py` converts it to `general.architecture = dflash` GGUF. The draft borrows the target's embedding and LM head and takes the tokenizer from the target GGUF. `dflash/Ornith-1.5-35B-A3B-DFlash-Q4_K_M.gguf` (236 MB) is the draft used for the numbers below. Q4_K_M has the same acceptance as F16 (0.41 vs 0.40 at n=7).

## DFlash results

Patched build, Q4_K_M DFlash draft, greedy, 4 prompts x 256 tokens, 16 threads. Baseline (no spec): **41.9 tok/s**.

| `--spec-draft-n-max` | decode tok/s | draft acceptance |
|---|---|---|
| 1 | 29.8 | 0.84 |
| **2 (default in build.sh)** | **32.8** | 0.73 |
| 3 | 31.3 | 0.63 |
| 5 | 28.5 | 0.47 |
| 7 | 24.7 | 0.41 |
| 15 (block size) | 15.3 | 0.20 |

- Best DFlash setting is about 0.78x of no-spec. It is **off by default**; enable with `DFLASH=1 bash build.sh`.
- Why it loses: on this CPU a single-token decode takes about 29 ms, an 8-token verify about 85 ms, and a 16-token verify about 117 ms (`llama-bench -p 1,8,16`). MoE verify reads many more expert weights than single-token decode. The draft adds roughly 70 ms per round at n=7.
- Target layer indices: `target_layer_ids` from config.json (1,6,...,37) gave 0.21 acceptance at n=15; shifting them by +1 gave 0.22, so the mapping is not the problem.
- `-td 4` (draft threads) made no difference.
- Greedy outputs differ from the no-spec baseline on all prompts, at character 114 to 713. The stock build (no patch) shows the same behavior, so this comes from the batch-shaped verify numerics, not from the patch. Texts stay semantically equivalent.

Raw logs: `benchmarks/dflash_exps.txt`, `benchmarks/dflash_q4.txt`, `benchmarks/dflash_sweep.txt`, `benchmarks/stock_check.txt`.

## Converting the DFlash draft yourself

```bash
python3 tools/convert_dflash_to_gguf.py \
  --dflash-dir <dir with config.json + model.safetensors from ornith-ai/Ornith-1.5-35B-A3B-DFlash> \
  --target-gguf Ornith-1.5-35B-Q4_K_M.gguf \
  --outfile Ornith-1.5-35B-A3B-DFlash-F16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize Ornith-1.5-35B-A3B-DFlash-F16.gguf Ornith-1.5-35B-A3B-DFlash-Q4_K_M.gguf Q4_K_M 16
```

Needs `numpy` and the `gguf` package (or `PYTHONPATH=<llama.cpp>/gguf-py`).

```bash
# builds llama.cpp at patches/BASE_COMMIT + patch, downloads the GGUF, serves on :8080
bash build.sh

# manual
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout $(cat ../patches/BASE_COMMIT) && git apply ../patches/qwen36-q4km-cpu-opt.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build build -j16 --target llama-server llama-bench
taskset -c 0-15 build/bin/llama-server -m Ornith-1.5-35B-Q4_K_M.gguf -t 16 -fa on -b 1024 -ub 1024 \
  -lm mmap+mlock -c 32768 --jinja --host 127.0.0.1 --port 8080
```

Ornith is a reasoning model. The chat template opens with `<think>`. Recommended sampling for general use: `temperature=0.6`, `top_p=0.95`, `top_k=20`.

## Files

- `Ornith-1.5-35B-Q4_K_M.gguf`: model (unmodified).
- `build.sh`: build, download and serve script (`MTP=1` or `DFLASH=1` to enable speculative decoding).
- `dflash/`: DFlash draft GGUF (Q4_K_M).
- `tools/convert_dflash_to_gguf.py`: DFlash safetensors -> GGUF converter.
- `patches/`: CPU patch, base commit, optional profiler patch (from HelloSun's repo).
- `benchmarks/`: stock vs patched llama-bench tables, flag sweeps and interleaved check (CSV), perplexity logs, MTP results, benchmark scripts.

## Notes and limitations

- Measured on one shared 16-core Xeon 8559C host with heavy background load. Treat ±10% as noise.
- Only the Q4_K_M quant was measured.
- Credit: base model by the Ornith Team (MIT). Quantization by the Ornith Team. CPU patch by HelloSun, based on llama.cpp (MIT).

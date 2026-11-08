---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [sparse, llama.cpp, gguf, vulkan, qwen3.8]
---

# Qwen3.8-27B sparse student (F2)

This is a per-token sparse student of **Qwen3.8-27B (HauhauCS Uncensored-Aggressive, Q6_K_P GGUF)**. It keeps all 64 layers, but each token reads only part of the weights. Low-rank attention/GDN projections, a 1-bit-sketch candidate generator for the FFN and a shortlist LM head make that possible. The student was distilled against the teacher's logits.

The student is uncensored like its teacher, with no added safety tuning.

## Files

| path | what |
|---|---|
| `gguf/qwen3.8-27b-sparse-F2-Q4_K_M.gguf` | **Ready-to-run** Q4_K_M student (20.9 GB, 3.82 BPW). Needs the patched llama.cpp below. |
| `F2-export/` | Standalone student in HF-style safetensors: teacher tensors plus student factors, plus `student_config.json`. |
| `code-f2/llama.cpp-sparse-qwen35.patch` | llama.cpp patch: the `GGML_OP_GATHER_AXPY` op (CPU + Vulkan) and sparse decode paths for `qwen35`. |
| `code-f2/` | Everything needed to rebuild it from scratch (see below). |
| `F1-casc1b-6144-A3584/`, `code/` | Previous student (F1) and its code, kept for reference. |

## Running it (llama.cpp, Vulkan / iGPU)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout f477f79
git apply /path/to/llama.cpp-sparse-qwen35.patch
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release   # drop -DGGML_VULKAN=ON for CPU only
cmake --build build -j --target llama-cli llama-server llama-bench llama-perplexity
./build/bin/llama-cli -m qwen3.8-27b-sparse-F2-Q4_K_M.gguf -ngl 99 -cnv
```

- **Decode** (batches of up to 8 tokens) uses the sparse paths. **Prefill** uses dense paths with the same weights. Prompt processing therefore costs about as much as the dense model, and the per-token savings apply to generation.
- **Runtime knobs.** Sparsity is set by GGUF keys that you can override without re-exporting:

  ```
  --override-kv qwen35.sparse.ffn_candidates=int:8192   # default 6144
  --override-kv qwen35.sparse.gdn_gate_k=int:2048       # default 1024
  --override-kv qwen35.sparse.ffn_active=int:3584       # default
  --override-kv qwen35.sparse.head_shortlist=int:8192   # default
  ```

  Setting both of the first two knobs gives the best quality shown below, at about +16% bytes read per token.

- **Vulkan testing status:** every new op passes `test-backend-ops` on lavapipe. That covers `gather_axpy` (f32/f16/q8_0/q4_K/q5_K/q6_K), the "neuron-expert" `mul_mat_id` at 17408 experts and `top_k`. The vocab-sized head ops are the exception, because lavapipe caps buffers at 128 MB; real GPUs and iGPUs don't have that limit. **I have not yet run it on real iGPU hardware.**
- **Memory:** the 20.9 GB GGUF plus KV cache fits a 32 GB machine. Only 16 of the 64 layers are full attention and keep a KV cache; the other 48 are DeltaNet layers with a fixed-size state.

## Architecture (per token)

| block | teacher | student |
|---|---|---|
| full attention (16 layers) | q/k/v/o dense | q low-rank 512 (incl. gate), o low-rank 1536, k/v exact |
| Gated DeltaNet (48 layers) | qkv / z / out dense | q,k low-rank 256, v exact; z/out restricted to top-1024 of 6144 channels, chosen by a rank-256 z predictor |
| FFN (64 layers, 17408 neurons) | dense SwiGLU | 1-bit sketch (ggml `Q1_0`, group 128) of gate/up ranks all neurons → exact gate/up on 6144 candidates → top 3584 by `(silu(g)·u)²·‖w_down‖²` → down; LoRA r16 merged into gate/up/down |
| LM head (248k vocab) | dense | rank-256 head picks a shortlist of 8192 → exact logits on the shortlist |

## Results

KL is measured against the BF16 teacher on held-out text (13k tokens, mixed domains). The math dev set has 15.5k tokens of held-out INTELLECT-3 math/science plus DeepMath. Teacher logits come from the same Q6_K_P weights.

| model | dev KL | dev top-1 | math KL | math top-1 |
|---|---:|---:|---:|---:|
| F1b (previous) | 0.2387 | — | 0.2016 | — |
| **F2 (default knobs)** | **0.2268** | 85.6% | **0.1648** | 88.4% |
| F2, candidates 8192 | 0.2125 | 86.2% | 0.1569 | 88.8% |
| F2, gate k 2048 | 0.2152 | 86.3% | 0.1582 | 88.8% |
| F2, both (+16% bytes/token) | **0.1974** | 86.8% | **0.1461** | 89.1% |

**GSM8K** (test set, first 128 problems, greedy, non-thinking):

| model | 0–63 | 64–127 | total |
|---|---:|---:|---:|
| teacher | 56 | 59 | **115/128 (89.8%)** |
| F2 | 55 | 60 | **115/128 (89.8%)** |
| F1b | — | — | 107/128 (83.6%) |

**llama.cpp KL** compared with the Q6_K_P teacher, measured with `llama-perplexity --kl-divergence` at `-c 1024` with 6 chunks. The general text includes prompt tokens, so it is noisier.

| model | path | general KL | general top-1 | math KL | math top-1 |
|---|---|---:|---:|---:|---:|
| dense Q4_K_P (reference) | — | 0.015 | 97.3% | 0.003 | 98.8% |
| F2 Q4_K_M | sparse decode (`-ub 8`) | 0.285 | 87.5% | 0.130 | 90.8% |
| F2 Q4_K_M | prefill path (`-ub 512`) | 0.244 | — | 0.123 | — |

## What's in F2 compared with F1

- Training data: about 12M tokens in total.
  - 65% general SFT-style teacher-distribution text.
  - INTELLECT-3-RL verified non-thinking answers.
  - DeepMath-hard thinking traces, truncated to 4k tokens.
  - The data was decontaminated against GSM8K test.
- Training ran for 1500 steps, longer than F1.
- The FFN sketch moved to group 128, so it maps exactly onto ggml `Q1_0`.
- The candidate generator was refit with per-layer scales and a per-neuron bias.

## Where the remaining KL comes from (ablations on F1, dev KL 0.239)

- Restoring teacher GDN projections gives 0.199. Of that, exact q/k alone gives 0.222, and removing the gate-select alone gives 0.221.
- Restoring teacher attention q/o gives 0.251. That is worse than the student, likely because the rest of the student was trained around the low-rank attention.
- Restoring the teacher head gives 0.235.
- FFN only (teacher mixers and head) gives 0.205. An oracle top-3584 FFN selection gives 0.208, so the candidate generator is close to the oracle and most of the FFN error comes from the 3584-of-17408 active budget.

## Rebuilding from scratch

`code-f2/setup.sh` bootstraps a single A100 80 GB:

1. Downloads the teacher GGUF and the data.
2. Converts the GGUF to BF16 HF (`convert_gguf_to_hf.py`).
3. Tokenizes the mix (`prep_data.py`).

Then run the jobs in order:

- `200_f1_diag.py`: F1 → F1b, with the sketch-128 and generator refit.
- `205_gdn_split.py`: diagnostics.
- `210_f2_train.py`: F2 training and export.

`convert_student_gguf.py` + `llama_eval.sh` turn an export into the GGUF and run the llama.cpp evals. `student.py` holds the student definition and its export/install code.

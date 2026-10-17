---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - apple-silicon
  - metal
  - mixture-of-experts
  - moe
  - speculative-decoding
---

# Qwen3.8-Flash-Next — Q4_0-Q8out-v3

**Run a 95.5 GiB model on a 64 GB Mac, at ~27 tokens/sec.**

This checkpoint is tuned for one specific job: running Qwen3.8-Flash-Next on Apple Silicon with
64 GB of unified memory, by keeping the rarely-used expert weights on SSD and reading them only when
a token actually needs them.

---

## ⚠️ Read this first

**This model does not run on stock llama.cpp.** Not "runs slower" — it will try to load all
95.5 GiB into memory and fail.

You need this fork, which adds `--moe-stream`:

### 👉 **https://github.com/npanj/llama.cpp**

Upstream llama.cpp does support the `qwen4exp` architecture. What it does not have is expert
streaming, which is the entire reason this checkpoint fits on a 64 GB machine.

---

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac, **64 GB** unified memory |
| **Disk** | ~100 GB free on the **internal** SSD |
| **Software** | [npanj/llama.cpp](https://github.com/npanj/llama.cpp), built with Metal |

The disk matters more than you'd expect — expert weights are read from it continuously while
generating. An external USB drive will be much slower.

---

## Quick start

Full instructions, troubleshooting and flag-by-flag explanation:
**[docs/qwen38-flash-next-v3.md](https://github.com/npanj/llama.cpp/blob/main/docs/qwen38-flash-next-v3.md)**

**1. Build the fork**

```bash
git clone https://github.com/npanj/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 --config Release
```

**2. Download the checkpoint** (95.5 GiB, 3 shards)

```bash
D=~/models/qwen38-flash-next-v3 && mkdir -p $D
for i in 1 2 3; do
  curl -fL --retry 5 -C - -o $D/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-0000$i-of-00003.gguf \
    https://huggingface.co/nitinpanj/qwen38-flash-next-v3/resolve/main/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-0000$i-of-00003.gguf
done
```

**3. Download the draft head** (1.9 GiB — worth about +50% generation speed)

```bash
D=~/models/qwen38-flash-next-mtp && mkdir -p $D
curl -fL --retry 5 -C - -o $D/mtp-shared-Q4_K_M.gguf \
  https://huggingface.co/nitinpanj/qwen38-flash-next-v3/resolve/main/MTP/mtp-shared-Q4_K_M.gguf
```

**4. Let the GPU wire enough memory** — don't skip this, and it resets on reboot

```bash
sudo sysctl iogpu.wired_limit_mb=59392
```

**5. Run**

```bash
export LLAMA_MOE_STREAM_LOOKAHEAD=1 LLAMA_MOE_STREAM_WAVE_CAP=200 \
       LLAMA_MOE_STREAM_PARTITION=1 LLAMA_QWEN4EXP_SPARSE_FA=1

./build/bin/llama-server \
  -m ~/models/qwen38-flash-next-v3/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00001-of-00003.gguf \
  -md ~/models/qwen38-flash-next-mtp/mtp-shared-Q4_K_M.gguf \
  -ngl 99 \
  --moe-stream --moe-stream-cache 36 --moe-stream-io-threads 8 --moe-stream-direct \
  -c 98304 -b 4096 -ub 4096 -cms 512 -np 1 -fa on \
  --cache-reuse 0 --cache-ram 512 \
  --jinja --reasoning-format deepseek \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3 \
  --spec-draft-ngl 99 --spec-max-prompt 0 \
  --host 127.0.0.1 --port 8080
```

Then open <http://127.0.0.1:8080>, or point any OpenAI-compatible client at
`http://127.0.0.1:8080/v1`.

First load takes a few minutes — it is reading 95.5 GiB off disk.

---

## Files

| File | Size | What it is |
|---|---|---|
| `Qwen3.8-Flash-Next-Q4_0-Q8out-v3-0000{1,2,3}-of-00003.gguf` | 95.5 GiB | The model. Point `-m` at shard 1; it finds the rest. |
| `MTP/mtp-shared-Q4_K_M.gguf` | 1.9 GiB | Draft head for speculative decoding. |

> **About the draft head.** It has no token embeddings of its own — it borrows the main model's,
> which is how it stays at 1.9 GiB instead of 2.6. So it **only** works when passed with `-md`
> alongside this checkpoint, on this fork. It cannot be loaded on its own.
>
> Its tensor names are fork-specific too. Upstream PR
> [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) adds shared-MTP support for this
> architecture using unsloth's original naming, not the names this fork renames to. If that lands,
> unsloth's sidecar will load on upstream llama.cpp directly and this file still will not.
>
> Prefer to build your own from unsloth's published sidecar? The scripts and an explanation of the
> conversion are at
> [`scripts/mtp/`](https://github.com/npanj/llama.cpp/tree/main/scripts/mtp).

---

## Measured performance

Apple **M5 Pro**, 64 GB, with the configuration above:

| | |
|---|---|
| Prompt reading (4k) | ~367 tokens/sec |
| Generation, draft head **on** | **~27.6 tokens/sec** |
| Generation, draft head **off** | 18.0-18.6 tokens/sec |
| Generation at 29k context, real chat traffic | ~20.6 tokens/sec |

Generation slows as context fills. That is expected — at depth the cost is dominated by reading the
attention cache, not by the streamed expert weights.

---

## What "v3" is

bartowski's Q4_0 with a Q8_0 `output.weight` spliced in, plus unsloth's UD-IQ4_XS spliced into five
tensor groups that stay resident in memory (`attn`, `hc`, `token_embd`, `ssm_out`, `shexp`).

Measured against the unspliced checkpoint, paired 40-chunk perplexity:

| | |
|---|---|
| Perplexity | **5.2777 → 4.3148** (−17.8%), better on 40 of 40 chunks |
| Draft acceptance | 0.751 → 0.817 |
| Decode speed | −2.7% |
| Disk | +1.69 GiB |

The five resident groups are not arbitrary. Trimming to `attn,hc,token_embd` measures the same
perplexity but drops decode 14% — `ssm_out` and `shexp` do nothing for perplexity and are worth
about 11 points of decode.

---

## If your Mac isn't 64 GB

- **Less memory:** lower `--moe-stream-cache` and the matching wired cap. It still loads; expect it
  to be slower. Untested below 64 GB.
- **More memory:** raise the cache — but in small steps. On a 64 GB machine, going from 36 to 38 GiB
  of cache collapsed generation from ~24 to ~3.7 tokens/sec as macOS began swapping. More is not
  monotonically better.
- **Not a Mac:** untested. Expert streaming is not Metal-specific in principle, but nothing here has
  been tuned or measured on CUDA or CPU.

---

## Credit

- Base model: **Qwen3.8-Flash-Next** by Qwen.
- Quantization sources: **bartowski** (Q4_0) and **unsloth** (UD-IQ4_XS, and the MTP sidecar this
  draft head is converted from).
- Expert streaming (`--moe-stream`), the feature that makes this run at all, comes from
  **[mihailescu2m/llama.cpp](https://github.com/mihailescu2m/llama.cpp)**.

This repository is the spliced checkpoint and the converted draft head. Model weights remain under
the original Qwen license.

**Problems?** Open an issue at
[github.com/npanj/llama.cpp/issues](https://github.com/npanj/llama.cpp/issues). Please say which Mac
and how much memory you have, and include the first ~30 lines the server prints at startup.

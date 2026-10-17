---
license: apache-2.0
base_model:
  - unsloth/Qwen3.8-Flash-Next-GGUF
  - Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - qwen4exp
  - qwen3.8-flash-next
  - llama.cpp
  - metal
  - apple-silicon
pipeline_tag: text-generation
---

# Qwen3.8-Flash-Next (PLElast) GGUF

GGUF builds of **Qwen3.8-Flash-Next** (177B, ~A3B active MoE) with the large PLE / n-gram
embedding table placed at the end of the file. That keeps the experts contiguous for the GPU
and lets the big embedding table stay on disk, so the model runs on Metal / Apple Silicon
with limited memory — e.g. the full model on a 64 GB Mac.

Same weights as unsloth's UD quants; only the tensor order is different. Nothing was requantized.

## Files

- `Qwen3.8-Flash-Next-Q2_K_XL-PLElast.gguf` — 79 GB
- `Qwen3.8-Flash-Next-IQ1_S-PLElast.gguf` — 68 GB

(For MTP speculative decoding you also need unsloth's small draft head — see below.)

Both run at about the same speed. Q2_K_XL has slightly lower perplexity; IQ1_S is smaller
and leaves more room for long context. They scored about the same in my tests (below) — I
haven't tested them more deeply than that.

## Benchmarks

Apple M1 Max, 64 GB, llama.cpp Metal, `-ngl 99 -fa on`.

Speed (Q2_K_XL): decode ~21 tok/s, prefill ~200 tok/s — about 3.3× faster than CPU-only.
128K context runs at the same decode speed. With the MTP sidecar, code generation is ~24 tok/s.

Quality (thinking on):

| | Q2_K_XL | IQ1_S |
|---|---|---|
| GSM8K (40) | 37/40 | 38/40 |
| HumanEval (20) | 20/20 | 19/20 |
| Perplexity (lower better) | 1.084 | 1.156 |

## Running (llama.cpp, Metal)

```bash
llama-cli -m Qwen3.8-Flash-Next-Q2_K_XL-PLElast.gguf -ngl 99 -fa on -c 8192 -p "..."
```

That's the whole command — no `--cpu-moe` or `-ot` needed. With `-ngl 99` the experts go on
the GPU and the ~27 GB embedding table is read on demand from the file (llama.cpp keeps it out
of VRAM automatically; the repack is what makes that work on Metal). So VRAM/wired use is
~44–48 GB even though the file is 68–79 GB. **Keep the file on an SSD** — it's memory-mapped and
the embedding table is paged in as needed; a slow disk will make it crawl.

Long context (128K) works. Add `-ctk q8_0 -ctv q8_0` to save memory at long context.

### Speculative decoding (MTP)

MTP speeds up structured output (code/JSON/repetitive) by **~27–41%** in my tests (85–100% draft
acceptance); little help on free-form prose. It needs a small draft head, which is unsloth's —
grab **`MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`** from
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/MTP)
(1.9 GB), e.g.:

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --local-dir .

llama-cli -m Qwen3.8-Flash-Next-IQ1_S-PLElast.gguf \
          --spec-type draft-mtp -md MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
          --spec-draft-n-max 3 -ngl 99 -fa on -c 8192 -p "..."
```

Needs a llama.cpp build with `qwen4exp` support. **MTP additionally needs PR #28243** (not in
mainline yet — build that branch, or the unsloth qwen4exp/mtp fork). Without it, drop the
`--spec-type`/`-md` flags and the model still runs, just without the MTP speedup.

## LM Studio (64 GB Mac, no command line)

Yes, it runs in LM Studio — you just need a recent runtime:

1. **Update the runtime.** LM Studio → Settings → Runtime → update the llama.cpp runtime to one
   with Qwen3.8-Flash-Next / `qwen4exp` support (llama.cpp **v0.4.0** or newer). Older runtimes
   can't load the model at all.
2. **Download** the `IQ1_S` file from this repo (best headroom on 64 GB; Q2_K_XL also works but is
   tighter). In LM Studio's search, paste this repo id, or drop the `.gguf` into your models folder.
3. **Load it**, and set **GPU offload to max** (all layers). Keep mmap on (default).
4. It runs at ~18–21 tok/s. The 27 GB embedding table stays **off VRAM automatically** (that's what
   the repack does), so VRAM use is ~42 GB, not the 68 GB file size. Keep the file on the internal SSD.

**MTP speculative decoding is not available in LM Studio** — it needs the unmerged PR #28243, which
no LM Studio runtime ships. LM Studio runs the base model; that's fine, just no MTP speedup.

## Credits

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Quantization: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
Apache-2.0.

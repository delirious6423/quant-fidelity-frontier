---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: llama.cpp
tags:
  - gguf
  - llama.cpp
  - speculative-decoding
  - mtp
  - nextn
  - qwen4exp
---

# Qwen3.8-Flash-Next NextN/MTP draft head for llama.cpp PR #27836

The MTP (NextN) draft head of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), exported as GGUF in the
layout that [llama.cpp PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836) (`--spec-type draft-mtp` for `qwen4exp`) expects,
plus a script that grafts it onto an existing split GGUF of the main model (for example the
[Unsloth quants](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), which ship without the head).

**Compatibility warning.** Only PR #27836 loads this. The other community MTP heads for this model
(`quimmedes/...`, `dzannotti/...`, `jockevaupptaget/...`) target different, mutually incompatible patches and tensor layouts; do not mix them.

## Files

| file | what |
|---|---|
| `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | head-only export from `convert_hf_to_gguf.py --remote --mtp --outtype q8_0` on the PR branch (34 tensors, 4.1 GB: 32 `blk.48.*` tensors plus `token_embd`/`output`, which the graft drops as duplicates). sha256 `ee87df0ecae89d759758667e9a1012d09299806f7805efc266eb5fe84773d4c9` |
| `graft-mtp-shard.py` | adds the head to a split GGUF as one extra trailing shard and rewrites the metadata (`block_count` 48 -> 49, `nextn_predict_layers = 1`, `attention.compress_ratios` extended by one `0`). Byte-exact, copy-on-write clones where the filesystem supports it, the input split is left untouched. |

## How to use

1. Build llama.cpp from PR #27836 (`git fetch origin pull/27836/head:pr27836 && git checkout pr27836`, then the usual cmake build).
2. Graft the head onto your split. The script needs a split whose first shard is metadata-only (Unsloth's are) and `pip install gguf numpy`:

   ```
   python3 graft-mtp-shard.py Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf mtp-Qwen3.8-Flash-Next-Q8_0.gguf --dry-run
   python3 graft-mtp-shard.py Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf mtp-Qwen3.8-Flash-Next-Q8_0.gguf
   python3 graft-mtp-shard.py Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf mtp-Qwen3.8-Flash-Next-Q8_0.gguf --verify
   ```

   This writes `Qwen3.8-Flash-Next-UD-IQ4_XS-MTP-0000{1..4}-of-00004.gguf` next to the input (about 30 s and +2.6 GB on APFS/btrfs/xfs; a full copy elsewhere).
3. Serve with the head as draft:

   ```
   llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-MTP-00001-of-00004.gguf -ngl 99 -fa on \
       --spec-type draft-mtp --spec-draft-n-max 3           # or: --spec-type draft-mtp,ngram-mod
   ```

   Without `--spec-type draft-mtp` the head tensors are skipped at load and the model behaves exactly like the original split.

## Measurements (Apple M5 Max, 128 GB, Metal, Unsloth UD-IQ4_XS, 128k context, q8_0 KV cache, weights mlocked)

1000-token generations, `reasoning_effort: low`:

| config | prose t/s | code t/s | draft acceptance (prose / code) |
|---|---|---|---|
| plain decode, PR build | 38.5 | 35.9 | - |
| `draft-mtp`, n-max 2 | 37.0 | 47.5 | 61.5% / 87.0% |
| `draft-mtp`, n-max 3 | 36.2 | 50.4 | 57.4% / 83.0% |
| `draft-mtp,ngram-mod`, n-max 3 | 36.3 | 51.2 | 52.4% / 83.0% |

Memory cost of the Q8_0 head: +3.5 GiB resident. Code and structured output gain 30-40%; prose is roughly neutral because acceptance
drops and every draft step still runs the full 248k-vocabulary LM head (see the PR discussion).

Greedy equivalence (temperature 0, 600 tokens, 3 code prompts): output identical to plain decode on 1 of 3 prompts; the other runs
diverge only at near-ties inside the reference's top-3 (top-1/top-2 margins of 0.011-0.040 nats, e.g. 44.3% vs 43.8%). `ngram-mod`
without any MTP head diverges the same way on a prompt where it drafts (margins 0.003-0.026 nats, once at the same token), so this is
the batch-shape numerics of speculative verification on a 512-expert MoE, not a property of the head.

## How the head was made

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git fetch origin pull/27836/head:pr27836 && git checkout pr27836        # 1d8de7c1b at the time
pip install -r requirements/requirements-convert_hf_to_gguf.txt
python3 convert_hf_to_gguf.py --remote --mtp --outtype q8_0 --outfile mtp-Qwen3.8-Flash-Next-Q8_0.gguf Qwen/Qwen3.8-Flash-Next
```

`--remote` streams only the 31 `mtp.*` tensors (about 7.7 GB of bf16) instead of downloading the 360 GB checkpoint.
Norm gammas are stored with the `+1` shift the PR expects; `fc_embedding` and `fc_hidden` are fused into `nextn.eh_proj`;
the head's hyper-connection mixer becomes `nextn.hc_head_{norm,down,up}`.

## Caveats

- PR #27836 is a draft at the time of writing; if its tensor layout changes before merge, re-export with the commands above.
- Single MTP block; the draft block attends densely (the trunk's sparse attention is not applied inside the head).
- Tested only on Metal. Speculative decoding on this MoE is not bit-identical to plain greedy decode on any backend we know of (see above).

Weights are derived from Qwen/Qwen3.8-Flash-Next and are subject to its license.

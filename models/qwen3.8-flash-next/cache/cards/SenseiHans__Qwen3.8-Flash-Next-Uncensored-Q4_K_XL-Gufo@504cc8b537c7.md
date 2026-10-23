---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- gufo
- strix-halo
- qwen
- uncensored
- abliterated
- MoE
- requantized
- amd-rocm
---

# Qwen3.8-Flash-Next Uncensored — Q4_K_XL-Gufo

**gufo-compatible requantization** of the community "Abliterated" (uncensored)
fine-tune of **Qwen3.8-Flash-Next** — prepared for the
[gufo](https://github.com/gufo-org/gufo) inference engine on AMD
**Strix Halo** (gfx1151).

Created 2026-10-23 from the existing 3-shard GGUF
`Qwen3.8-Flash-Next-Uncensored-Q4_K_S` — no new fine-tuning run, purely a
layout/metadata correction that makes the same weights readable by gufo.

## Lineage

1. **Qwen/Qwen3.8-Flash-Next** — base model (architecture `qwen4exp`: 48 blocks,
   512 experts, size label 512x56B, 262 144 native context)
2. Community fine-tune **Uncensored / Abliterated** (AI red-team) —
   [orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored)
3. **GGUF export Q4_K_S** —
   [orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF)
   (3 shards, ~105 GiB, imatrix quantization)
4. **This requantization** for gufo (see "What was changed")

## What was changed (and why)

The gufo engine validates a per-tensor **format whitelist** and the
**index-ring geometry** while creating the model. The original Q4_K_S export
failed both and was rejected by gufo. This copy passes:

| Change | Detail |
|---|---|
| **716 tensor dtypes upgraded** | Control / attention / hyper-connection / SSM / PLE tensors moved from `q4_k` / `q5_0` / `q5_k` to **`q8_0`** or **`f32`** — layout taken 1:1 from the validated `UD-Q4_K_XL` reference profile |
| **PLE embeddings** | The 51-billion-element embedding tensor converted from `q5_0` to **`iq4_nl`** (saves size, matches the reference profile) |
| **Export bug fixed** | `qwen4exp.attention.compress_ratios` was **zeroed out** in the original export; corrected to the `[0,0,0,4, …]` pattern (full attention on every 4th block) |
| **Result** | ~103.5 GiB across 3 shards, MTP-sidecar compatible (`shared-Q8`, 7 draft tokens), vision-ready (`mmproj BF16`) |

> **Quality note:** the upgraded tensors contain **no new information** — they are
> the same weights as the Q4_K_S export, just stored loss-free where gufo demands
> high-precision input. Net effect vs. the original GGUF: equal to slightly better
> (critical tensors are no longer hard-quantized to q4_k).

## Files

```
Qwen3.8-Flash-Next-Uncensored-Q4_K_XL-Gufo-00001-of-00003.gguf  (39.7 GB)
Qwen3.8-Flash-Next-Uncensored-Q4_K_XL-Gufo-00002-of-00003.gguf  (47.7 GB)
Qwen3.8-Flash-Next-Uncensored-Q4_K_XL-Gufo-00003-of-00003.gguf  (23.7 GB)
```

## Usage (gufo)

```bash
podman run --rm --name gufo --device /dev/dri --device /dev/kfd \
  --group-add keep-groups --security-opt seccomp=unconfined --ipc=host \
  --security-opt label=disable --userns=keep-id -p 8028:8028 \
  -v <MODEL_DIR>:/models/target:ro \
  docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0-gufo \
  gufo serve --host 0.0.0.0 --port 8028 --sessions 2 llm \
  --model /models/target/Qwen3.8-Flash-Next-Uncensored-Q4_K_XL-Gufo-00001-of-00003.gguf \
  --context 262144 --max-tokens 32768 --served-model-name Qwen3.8-Flash-Next \
  --think on --reasoning-effort high
```

How this model runs in the **AI Toolbox** ecosystem (headless autostart via
file mode, discrete model swap, LiteLLM) is documented in
[`docs/AI-TOOLBOX-SETUP.md`](./docs/AI-TOOLBOX-SETUP.md) — including what the
cockpit itself can and cannot do with gufo.

Full provenance: the exact per-tensor dtype tables used for the requantization
ship in `docs/` — `gufo-requant-minimal-rules.txt` (the 716 exact-name rules
actually applied), `gufo-requant-tensortypes-exact.txt` (the complete 1224-entry
reference table read from the UD-Q4_K_XL shards) and `gufo-requant-tensortypes.txt`
(the condensed pattern list).

## License — please read (documented license discrepancy)

The licensing of this model is **not as clean as the embedded metadata suggests**:

| Source | Declares |
|---|---|
| Embedded GGUF metadata of these shards (`general.license`, inherited from the OrcaRouter export) | `apache-2.0` |
| **Base-model repo** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), license file `LICENSE` | **Qwen Community License 1.0** (card tags: `license: other`, `license_name: qwen-community-1.0`) |
| Direct GGUF source repo (orcarouter/…-Uncensored-GGUF) | **no license file, no license tag** |
| Fine-tune repo (orcarouter/…-Uncensored) | a `LICENSE` file exists but is gated (HTTP 401); content not verifiable publicly |

**How we handle it.** Legally decisive is the license of the source repositories:
the weights derive from the Qwen base model, and a "mere" conversion or
requantization cannot grant more rights than the upstream license allows. Therefore
this copy is redistributed under the terms of the **Qwen Community License 1.0** —
its full text ships in [`LICENSE`](./LICENSE). The `apache-2.0` claim embedded by
the GGUF exporter is documented here (and kept untouched in the metadata), but it
cannot override the upstream license text for the weights. The lineage is preserved
in the GGUF metadata (`general.base_model.*`).

**Good news:** the Qwen Community License 1.0 is essentially MIT with model-specific
additions — it explicitly permits use, modification, fine-tuning, distribution and
sale of the weights. Conditions: (1) keep the copyright/permission notice (the
prominent-name clause only applies above 100 M monthly active users or USD 20 M
monthly revenue); (2) a separate Qwen license is required only for *commercial*
"Model-as-a-Service" or "AI Work Assistant" businesses. Private and local use —
like this one on Strix Halo — only carries the notice requirement.

*Disclaimer: documented as found, not legal advice. A friendly question to OrcaRouter
or Qwen clears the last ambiguity.*

## Acknowledgements

- **@kyuz0** — [ai-toolbox-cockpit](https://github.com/kyuz0/ai-toolbox-cockpit)
  and the Strix Halo toolbox images that run gufo here. Many thanks!
- **gufo-org** — for the engine
- the **Qwen team** — for the base model
- **Unsloth** — whose validated UD-Q4_K_XL layout served as the reference recipe
- **@orcarouter** — for the uncensored/abliterated fine-tune and the original
  GGUF export on which this requantization is based

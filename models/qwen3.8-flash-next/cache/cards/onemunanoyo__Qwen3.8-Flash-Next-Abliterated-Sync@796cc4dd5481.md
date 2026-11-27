---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
- huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated
- hiru2005/Sync-shared
tags:
- abliterated
- quantized
- gguf
- fixed-scale
- sync-strata
- qwen3.8-flash-next
- moe
---

# Qwen3.8-Flash-Next Abliterated

English | [日本語 (README.ja.md)](README.ja.md)

A **fixed-scale, byte-auditable Abliteration** release for **Sync-Strata** (`Sync-XS` / `Sync-S` / `Sync-M`), transferring the refusal direction extracted from Huihui AI's abliterated BF16 model onto [`hiru2005/Sync-shared`](https://huggingface.co/hiru2005/Sync-shared) **without touching any quantization scale or any of the 24,576 routed MoE experts**.

---

## Credits & Acknowledgments

This release builds directly on the work of the upstream creators—many thanks to everyone involved:

- **Refusal Direction Source — Huihui AI (`huihui-ai`)**:  
  The refusal direction unit vector (`r ∈ ℝ²⁵⁶⁰`) used here was extracted directly from [`huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated). Our contribution is simply transferring that direction onto the quantized Sync shared weights cleanly without changing scales or routed experts.
- **Sync / Sync-Strata Format & Engine — `hiru2005` / `hiruhiru2005`**:  
  Deep appreciation to the creator of **Sync-Strata** and the **Sync-XS / Sync-S / Sync-M / Sync-shared** packs:
  - Hugging Face: [`hiru2005`](https://huggingface.co/hiru2005) ([`Sync-shared`](https://huggingface.co/hiru2005/Sync-shared), [`Sync-XS`](https://huggingface.co/hiru2005/Sync-XS), [`Sync-S`](https://huggingface.co/hiru2005/Sync-S)) and [`sync-strata/Sync-M`](https://huggingface.co/sync-strata/Sync-M)
  - GitHub: [`hiruhiru2005/sync-strata`](https://github.com/hiruhiru2005/sync-strata) and [`hiruhiru2005/sync-strata-engine`](https://github.com/hiruhiru2005/sync-strata-engine)
- **Upstream Foundations**:  
  Base model by [Qwen (`Qwen/Qwen3.8-Flash-Next`)](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Gyro-S Hadamard-rotated quantization by Agention, and inference engine upstream [Strata (`Niko1221/Strata`)](https://github.com/Niko1221/Strata).

---

## What Is Included

Only the shared dense GGUF (`Qwen3.8-Flash-Next-Sync-common.gguf`) is modified—specifically **72 dense residual writers** in layers `6..41` (~431 MiB out of 30.45 GiB). The routed experts (`experts.v2.bin` in `Sync-XS` / `Sync-S` / `Sync-M`), `mtp-rt`, `tokenizer`, and PLE tables remain **100% untouched**.

| File in this repo | Size | Purpose |
| :--- | :---: | :--- |
| `native-gguf/Qwen3.8-Flash-Next-Abliterated-Sync-common.gguf` | 30.45 GiB (`32,692,038,016` B) | Drop-in replacement for `Qwen3.8-Flash-Next-Sync-common.gguf` |
| `patch/dense_mid_6_41_a195_dual.patch.bin` | ~431.0 MiB (`451,952,640` B) | Raw byte payload of the 72 edited tensors only |
| `patch/patch_manifest.json` | 35.1 KB | Source/target GGUF SHA-256 + per-tensor offsets & hashes |
| `patch/apply_patch.py` | 8.4 KB | Stdlib-only patcher (`--input`/`--output`) and full auditor (`--verify`) |
| `audit/refusal_dir.npy` | 20.6 KB | Extracted unit refusal vector `r ∈ ℝ²⁵⁶⁰` (`float64`) |
| `audit/patch_report.json` | 43.8 KB | Solver report (Gram alignment, projection ratio, orthogonal drift) |
| `audit/eval_summary.json` | 9.7 KB | Aggregate evaluation numbers (`Sync-XS` & `Sync-S`) |
| `SHA256SUMS.txt` | 687 B | Checksums for all files |

---

## Usage with Sync-Strata (Full GGUF Download)

First install **Sync-Strata** and download `Sync-XS`, `Sync-S`, or `Sync-M` normally from [`hiruhiru2005/sync-strata`](https://github.com/hiruhiru2005/sync-strata).  
This file shares the **exact byte length (`32,692,038,016` bytes)** of upstream `Qwen3.8-Flash-Next-Sync-common.gguf`, so Sync-Strata recognizes it without extra flags.

1. Download `native-gguf/Qwen3.8-Flash-Next-Abliterated-Sync-common.gguf` from this repository:
   ```bash
   hf download onemunanoyo/Qwen3.8-Flash-Next-Abliterated-Sync native-gguf/Qwen3.8-Flash-Next-Abliterated-Sync-common.gguf --local-dir .
   ```
2. **Back up** your original `Sync-shared` common GGUF, then move & rename the downloaded file into Sync-Strata's `models/native-gguf/` folder:
   - **Windows (PowerShell)**:
     ```powershell
     $dir = "$env:LOCALAPPDATA\SyncStrata\models\native-gguf"
     Move-Item "$dir\Qwen3.8-Flash-Next-Sync-common.gguf" "$dir\Qwen3.8-Flash-Next-Sync-common.gguf.orig"
     Move-Item ".\native-gguf\Qwen3.8-Flash-Next-Abliterated-Sync-common.gguf" "$dir\Qwen3.8-Flash-Next-Sync-common.gguf"
     ```
   - **Linux**:
     ```bash
     DIR="$HOME/.local/share/SyncStrata/models/native-gguf"
     mv "$DIR/Qwen3.8-Flash-Next-Sync-common.gguf" "$DIR/Qwen3.8-Flash-Next-Sync-common.gguf.orig"
     mv "./native-gguf/Qwen3.8-Flash-Next-Abliterated-Sync-common.gguf" "$DIR/Qwen3.8-Flash-Next-Sync-common.gguf"
     ```
3. Launch Sync-Strata as usual (`GUI`, `sync up`, `sync chat`, or API). All `Sync-XS` / `Sync-S` / `Sync-M` packs read from this shared GGUF automatically (verified locally on `Sync-XS`, `Sync-S`, and `Sync-M`, including Sync-Strata Desktop).

> **Note:** After replacing the file, do **not** run Sync-Strata's upstream SHA-256 repair button on `native-gguf`, as it compares against the unmodified upstream digest (`82a78508...`) and would flag or overwrite the abliterated copy.

If you run `strata` directly from CLI, pass `--native` and `--ple-gguf` pointing to the **exact same path** (passing two different GGUF paths causes `strata` to register dense tensors twice).

---

## Usage If You Already Have `hiru2005/Sync-shared` (431 MiB Patch)

You do **not** need to download 30.45 GiB if you already have `hiru2005/Sync-shared` (`sha256: 82a78508d291130fdf4654b90acb5780e8f7f44b28b0ac3495aed59163a00e69`).

1. Download only the `patch/` folder and checksums (zero third-party Python dependencies required—only Python 3 stdlib):
   ```bash
   hf download onemunanoyo/Qwen3.8-Flash-Next-Abliterated-Sync --include "patch/*" "SHA256SUMS.txt" --local-dir .
   ```
2. Apply the patch to create a verified copy (preserving your original file):
   ```bash
   python patch/apply_patch.py \
     --input /path/to/Qwen3.8-Flash-Next-Sync-common.gguf \
     --output /path/to/Qwen3.8-Flash-Next-Abliterated-Sync-common.gguf
   ```
   *(Or pass `--in-place` if you want to patch a copy directly after backing up the original.)*

`apply_patch.py` verifies the full input SHA-256 (`82a78508...`), verifies scale-byte invariance on all 72 blocks, writes the 72 tensor payloads, and verifies the full output SHA-256 (`110ad4ee...`).

---

## Verification & Auditability

Because no LoRA, SFT, or full re-quantization is performed, you can cryptographically audit every modified and unmodified byte against your own copy of `hiru2005/Sync-shared`:

1. **Check File Digests**:
   - `sha256sum -c SHA256SUMS.txt` (POSIX) or `Get-FileHash` (Windows).
   - Upstream base GGUF (`source_sha256`): `82a78508d291130fdf4654b90acb5780e8f7f44b28b0ac3495aed59163a00e69`
   - Abliterated GGUF (`target_sha256`): `110ad4ee8006f23f26662fae2975c98ed8284e670cd48718a2fc11b5cbc6c108`
2. **Full 30.45 GiB Byte-Level & Scale-Invariance Audit**:
   ```bash
   python patch/apply_patch.py --verify \
     /path/to/Qwen3.8-Flash-Next-Sync-common.gguf \
     /path/to/Qwen3.8-Flash-Next-Abliterated-Sync-common.gguf
   ```
   This single-pass audit verifies:
   - Every byte outside the **72 target tensors** (GGUF headers, metadata, `token_embd`, `output` LM head, `hyper_connection`, `ple`, `layers 0..5`, `layers 42..47`, attention gates/QKV/routers, and all 24,576 routed MoE experts in `experts.v2.bin`) is **100% bit-for-bit identical**.
   - Every `Q5_K` (`[0:16]`) and `Q8_0` (`[0:2]`) quantization scale/header block inside the 72 edited tensors is **100% bit-for-bit identical**.
   - Every tensor matches its `orig_sha256` and `patched_sha256` in `patch/patch_manifest.json`.

---

## Measured Results

### Base vs. Abliterated Results (`Sync-XS` & `Sync-S`)
- **No-Think Probe**: 60 prompts (`20 or_bench_hard + 20 xstest_safe + 20 xstest_unsafe`, `reasoning_effort="none"`, `max_tokens=128`).
- **Reasoning-ON Probe**: 30 shared prompts (`10` each, `reasoning_effort="low"`, `max_tokens=1536`); `Leak` counts prompts where No-Think complied but Reasoning-ON refused.
- **KL / PPL**: `strata.exe --serve` `logprobs=256` over 511 positions of `eval_ppl.txt` (~98.2% probability mass covered).

| Pack | Model | No-Think 60p (`D1`, Benign Ref.) | Think-ON `low` 30p (`D1`, Closed, Leak) | `ΔPPL%` *(match / tail-b)* | `KL_fwd` | `Top-1` Agree |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Sync-XS` | Base | 37/60 (`0.0%`, 20/40) | 8/30 (`0.0%`, 30/30, `1`) | `0.000%` | `0.000000` | `100.0%` |
| `Sync-XS` | **Abliterated** | **1/60 (`97.3%`, 0/40)** | **0/30 (`100.0%`, 30/30, `0`)** | **`+0.468%` / `+0.461%`** | `0.040559` | **`90.2%`** |
| `Sync-S` | Base | 37/60 (`0.0%`, 19/40) | 10/30 (`0.0%`, 30/30, `1`) | `0.000%` | `0.000000` | `100.0%` |
| `Sync-S` | **Abliterated** | **2/60 (`97.3%`, 0/40)** | **0/30 (`100.0%`, 30/30, `0`)** | **`+1.329%` / `+1.011%`** | `0.045703` | **`91.8%`** |

- **Inference Speed (`Sync-S`)**: Base `56.2 / 65.2 tok/s` vs. Abliterated `56.3 / 66.8 tok/s` (no-think / reasoning-on; no overhead).
- **MMLU Check (`cais/mmlu`, 20 subjects × 10 = 200 questions, `max_tokens=16`, `no-think`)**: `143/200 (71.5%) → 145/200 (72.5%)` (`McNemar p = 0.7266`, non-math categories `123/150 vs 122/150`; measured on an earlier all-layer `L0..47` Abliterated variant vs. Base, with `37` Base and `30` Abliterated STEM derivations truncated at 16 tokens).
- **NSFW JA Benchmark (`Sync-S` Abliterated, 84 Japanese prompts, `reasoning_effort="low"`, `max_tokens=2048`)**: **0/84 refusals (`0.0%`)**, **1/84 output repetition loop** (all 84/84 reasoning blocks closed cleanly).

### Evaluation Boundaries
- Measured locally on RTX 5060 (8 GiB VRAM) + 32 GiB RAM (`R4`, `context=65536`, `kv=q4_0`).
- Evaluated on 60 no-think prompts, 30 `low`-reasoning prompts, 84 Japanese NSFW benchmark prompts (`Sync-S`, `reasoning_effort="low"`, `max_tokens=2048`), 511 top-256 logit positions (`eval_ppl.txt`), and 200 MMLU questions (all-layer `L0..47` variant vs. Base).
- `Sync-M` operation has been verified locally on the same RTX 5060 (8 GiB VRAM) + 32 GiB RAM machine (boot, No-Think `0/11` benign refusals at `47.7 tok/s`, and `low`-reasoning `0/4` refusals / `4/4` closed at `55.9 tok/s`), though a full KL/PPL and 60-prompt benchmark has not been run on `Sync-M`.

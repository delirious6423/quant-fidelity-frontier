---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE
base_model:
- agentionai/Signal-3.8-Flash-Next-GGUF
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- halogen
- hgn
- qwen
- quantized
- baseline-plus-delta
---

# Signal-3.8-Flash-Next-HGN-v2

**Plain, non-abliterated Signal HGN v2 conversion, published separately as
`Quat3rnion/Signal-3.8-Flash-Next-HGN-v2`.** No donor tensor replacement or composition was performed. This
repository is not an alternate branch, folder or continuation of a donor build.

This is a **baseline-plus-delta publication**, not a self-contained snapshot.
It contains the runtime files that differ byte-for-byte from the exact pinned
Peonist baseline, plus necessary licenses, notices, provenance and assembly
metadata. Use the assembler below to obtain the complete **14-file** runtime
package. Original GGUF and BF16 inputs are not uploaded.

## Exact origins and conversion

- Language-model source: [agentionai/Signal-3.8-Flash-Next-GGUF](https://huggingface.co/agentionai/Signal-3.8-Flash-Next-GGUF/tree/4d30467d9f087cd88640762808c05d14131a329b)
  at immutable revision **`4d30467d9f087cd88640762808c05d14131a329b`**,
  path **`Q8_0/Signal-3.8-Flash-Next-Q8_0.gguf`**.
- Source input: **188,225,032,736 bytes**, SHA-256
  **`98fb2414cdffccc16b9ff4222aae60015b1284cdea19a15232f389cae1a66cdd`**.
- Original-Qwen MTP/vision auxiliary tensor and asset origin:
  [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540)
  at **`de4b8e4d43b917e7706784d8bb445c9af86a3540`**. These inputs were reconstructed from pinned original
  Qwen tensors and imported by the converter. The Signal publisher's statement
  that these auxiliaries are unchanged is provenance, not a byte comparison
  against Signal's quantized auxiliary tensors.
- The exact retained Signal GGUF chat template is included. The base
  `tokenizer_config.json` was prepared by replacing only its `chat_template`
  field with that template and reserializing JSON. Token-to-ID coordinates
  were checked. The final template/config must not be replaced with baseline
  versions just because other tokenizer assets match.
- Converter: `bf16-to-hgn` 0.1.0, profile **v2**, seed **0**, threads **2**,
  HT beam **128**, HT and I4R backends **HIP**, Q4CP backend **CPU**.
  Executable SHA-256: `bb64ff5dcc6e9ab4800c095cedec538aba1bfd08a456cfc8cb67278a2aa4c807`.
- This is a conversion from the retained **Q8_0 GGUF**, not a claim of direct
  original-BF16 conversion or numerical equivalence to BF16. Imported HGN
  auxiliaries were decoded to canonical values and re-encoded; no stock HGN
  output was silently substituted.

The recorded full HIP decode verification completed successfully, attesting
**1,658 source tensors**,
**2,349 output tensors** and
**118,785,466,780 payload bytes**. That verifies the
conversion's recorded numerical/provenance checks, not task quality.

`conversion-manifest.json` is the **unmodified original schema-2 metadata**,
SHA-256 **`f1afa0920054838693c1dd0c14cc667383b3ecfcfd0b4b02c5c85d0d29fb7df4`**. Recursive inspection found no absolute
filesystem-path strings, so no portable-path rewrite or derived manifest was
needed. All tensor/layout identities remain those of the original conversion.
Machine-local receipt files and queue settings are not published; portable
allowlisted attestations and their original receipt hashes are in
[`provenance/conversion.json`](provenance/conversion.json).

## Immutable baseline dependency

Baseline: [peonist-ai/halogen-qwen3.8-flash-next](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/tree/5cc17cea1a10b502b4a5db41d8d2b1a0f22dba26)

Required immutable baseline revision: **`5cc17cea1a10b502b4a5db41d8d2b1a0f22dba26`**.
All **16** baseline files were compared by exact size and SHA-256, rather than
filename or assumed sidecar compatibility. Only these four runtime files match
and are intentionally omitted here:

| Runtime destination / baseline path | Bytes | Required SHA-256 |
|---|---:|---|
| `tokenizer/generation_config.json` | 202 | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| `tokenizer/merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `tokenizer/tokenizer.json` | 12,809,320 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer/vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

This omits **22,885,540 bytes** of identical
runtime payload. The remaining **10 runtime files** total
**118,789,817,058 bytes** before the required
publication metadata. The assembled 14-file package totals
**118,812,702,598 bytes**.

All three generated HGN files differ from the pinned baseline and are retained:

| Published runtime file | Bytes | SHA-256 |
|---|---:|---|
| `model.hgn` | 66,687,683,040 | `965ee4168c3f912a9dfe0236955a2e9be9684dbbfbd1b86593bc613b763809dd` |
| `qwen38-flash-next-ngram.hgn` | 51,200,246,084 | `93604e21a5fb2046a20b29ed321b4e3842f69316970387534a7c346fbf3e51ac` |
| `qwen38-flash-next-vision.hgn` | 897,916,416 | `b6ca6716eaab601773f7155701bb09f39c72922caa21620bab577f0281f2ea2a` |

The main checkpoint is `model.hgn`; the converted MTP tensors are included in
that main package. Keep the matching ngram and vision outputs. Do not substitute
baseline main, ngram or vision files, or a different Signal template/config.
[`assembly-manifest.json`](assembly-manifest.json) specifies all **14** output
paths, source paths, exact sizes and SHA-256 hashes. Baseline sources always use
the immutable revision above; the other ten sources use this publication's
chosen immutable commit.

## Reproducible assembly

Requirements: Linux with atomic `renameat2` support, Python 3 and the official
`huggingface_hub` package. Install the SDK in your preferred environment:

```bash
python3 -m pip install huggingface_hub
```

Choose an immutable commit of **this plain repository**. The following example
resolves its current commit once, prints that commit for your reproduction
record, downloads both metadata files at exactly that commit, then assembles:

```python
import subprocess
import sys
from huggingface_hub import HfApi, hf_hub_download

repository = "Quat3rnion/Signal-3.8-Flash-Next-HGN-v2"
revision = HfApi().model_info(repository, revision="main").sha
print("Pinned plain publication:", revision)
assembler = hf_hub_download(repository, "assemble.py", revision=revision)
manifest = hf_hub_download(repository, "assembly-manifest.json", revision=revision)
subprocess.run([
    sys.executable, assembler,
    "--manifest", manifest,
    "--revision", revision,
    "--output-dir", "Signal-3.8-Flash-Next-HGN-v2",
], check=True)
```

To reproduce that exact release later, use the recorded 40-hex commit as
`revision` instead of resolving `main` again. Use a **new, nonexistent output
directory**. The assembler verifies every file's full SHA-256 and size before
atomically installing the complete directory; it does not overwrite an existing
output. It links only downloaded Hugging Face cache files. Add `--copy` only if
you explicitly want to permit cross-filesystem copies; budget enough disk for
that choice. Cached SDK authentication is used when available without printing
a token. No original local conversion output is linked or mutated.

The manifest-driven assembler is reused byte-for-byte, SHA-256
**`64f04097077b2635d1badc0add78ee8abce29cc8a52d47ef509286d7e47cf35a`**. `.gitattributes` is separately handled
Hub/SDK transport metadata, not an assembly file: pristine new-repository bytes
are retained and the SDK deterministically extends them with sorted per-path
LFS rules as needed. It is **not** claimed unchanged from the Peonist baseline.

## Recorded plain-model performance, not quality

The completed run
**`signal-plain-v2-hgn-halogen-0.18.5-depth7-gate095-098-20261015T125100Z`** used **BetterBench 0.6.1** with corpus 1.0 and **Halogen
0.18.5**, sampled depth **7**, gate **0.95,0.98**. It measured **392 successful
requests**: **160** single-stream decode, **40** prefill, and **192** concurrency
requests (48 each at levels 1, 2, 4 and 8). Warmups and a separate smoke request
are excluded from those 392.

- Weighted elapsed decode throughput: **65.217879 tokens/s**.
- Concurrency aggregate throughput at levels 1 / 2 / 4 / 8:
  **63.6743 / 69.7192 / 76.0788 / 76.0822 tokens/s**.
- Sampled request settings: temperature **0.7**, top-p **0.95**, top-k **20**,
  seed **42**; 20 measured single-stream runs per category.
- Prefill target labels 2k–64k are workload targets, not measured token counts;
  actual median prompt lengths were **1,559 / 5,980 / 11,866.5 / 23,594 / 47,075**.

The actual plain checkpoint identity was tied to the verified 14-file inventory,
main size/inode and mappings of all three generated HGN outputs. An existing API
routing alias was retained for request parity; the alias itself is not evidence
of which weights ran. The summary is derived **only from this plain run**, not a
donor benchmark. See [`provenance/benchmark.json`](provenance/benchmark.json) for
metrics, exact receipt hashes, runtime image pin and request configuration.

The measurement window was **2026-10-15 12:58:09–13:35:19 UTC**, before the later
embedding-NPU addition. Recorded native health lists `decider-0.8b` and
`qwen3.5-2b` helpers. This is **not** a measurement of a three-NPU configuration
or of subsequent production state.

These are **one run's performance measurements**, not universal device results:
the official report did not populate GPU metadata. Elapsed throughput is not
instantaneous speed inferred from bursty SSE gaps. Missing cached-token usage
means **unknown**, not a proven cold cache. Tail p99 metrics are under-sampled.
48 decode samples reached their output length limit; none was recorded as
truncated in reasoning. No IFEval or BFCL quality evaluation was completed, and
no claim is made about accuracy, safety, retained quality, BF16 equivalence or
superiority to another model.

## License and notices

- [`LICENSE`](LICENSE): exact pinned original Qwen **Qwen Community License 1.0**.
- [`LICENSE-SIGNAL`](LICENSE-SIGNAL): exact pinned Signal repository license,
  also **Qwen Community License 1.0**.
- [`NOTICE`](NOTICE): origins, derivative-conversion notice and license reminders.

Both license files are **3,235 bytes**, SHA-256
**`a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`**.
The full terms control. This is **not** a blanket Apache-2.0 grant. In particular,
the license has model-name display conditions for the stated large commercial
thresholds and requires a separate Qwen license for commercial Model as a
Service or AI Work Assistant businesses, subject to its internal-use exception.
Qwen, Signal's publisher, the Peonist baseline/runtime authors and other
contributors retain their respective rights; this derivative conversion is
not an official endorsement.

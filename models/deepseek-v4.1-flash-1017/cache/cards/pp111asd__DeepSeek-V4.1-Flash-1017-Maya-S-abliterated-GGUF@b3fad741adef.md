---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4.1-Flash-1017
  - peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF
  - mrfakename/DeepSeek-V4.1-Flash-1017-abliterated
  - unsloth/DeepSeek-V4.1-Flash-1017-GGUF
pipeline_tag: text-generation
tags:
  - gguf
  - deepseek
  - project-maya
  - abliterated
  - uncensored
---

# DeepSeek V4.1 Flash 1017 abliterated GGUF

Replacement GGUF shards with a rank-2 refusal-reduction edit applied to 62 tensors per model. Maya-S and the added Q8_0 targets are requantized; UD-Q8_K_XL targets retain BF16 storage. Download the unchanged shards from the pinned upstream sources and substitute the matching replacement below. Do not mix layouts or rename a four-shard file into a six-shard set.

**Uncensored model warning:** this edit is intended to reduce refusals, including refusals of harmful requests. Outputs may be dangerous, illegal, biased or offensive. Neither refusal nor fluency establishes safety or accuracy. Use appropriate safeguards for your deployment.

## Which files to choose

| Folder | Required engine and layout | Evaluation coverage |
| --- | --- | --- |
| `maya-xs/` | Maya v1.1.24+; Maya-XS (IQ1_S experts, ~120 GB of model weights plus the Q6_K Engram), five shards, its own DSpark XS2 drafter; replace shard 3 | Full comparison below (one capped answer) |
| `v1.1.17-one-set/` | Maya v1.1.17; six shards, with Engram Q6_K in shards 5–6; recommended layout | Full comparison below |
| `v1.1.16-four-shard/` | Maya v1.1.16; four shards and separate unsloth Engram | Historical full comparison below |
| `unsloth-UD-Q4_K_XL/` | Maya v1.1.17; documented unsloth layout | Unevaluated |
| `unsloth-UD-Q8_K_XL/` | Maya v1.1.17; documented unsloth layout, BF16 edit targets | Unevaluated |
| `unsloth-UD-IQ3_S/` | Not loadable by Maya v1.1.17 (its unsloth path pins Q4/Q8 only); intended for engines that load unsloth's 1017 GGUFs such as llama.cpp (untested) | Unevaluated |
| `unsloth-UD-IQ3_XXS/` | Not loadable by Maya v1.1.17 (its unsloth path pins Q4/Q8 only); intended for engines that load unsloth's 1017 GGUFs such as llama.cpp (untested) | Unevaluated |

Each Maya-S folder contains one complete replacement shard 4, `PATCHED.txt`, diagnostics in `logs/`, and `UPSTREAM_SHA256SUMS` for unchanged prerequisites. Common files include the edit parameters, patch script, license and release checksums. Unchanged model shards, Engram, DSpark, vision assets and engine binaries are obtained separately. Tokenizer and pack files are generated during setup.

| Replacement | Size | SHA256 |
| --- | ---: | --- |
| `maya-xs/DeepSeek-V4.1-Flash-1017-Maya-XS-IQ1_S-00003-of-00005.gguf` | 19,784,230,848 bytes | `f1e93b2621f3c5a96f42ce3857049370d0ee0ddb56cbde2d2cb648f2df77911d` |
| `v1.1.17-one-set/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00004-of-00006.gguf` | 35,532,532,032 bytes | `89998155648e330d4f9eb0e0ab4810d9efe872dee0394ca6768df44158392a56` |
| `v1.1.16-four-shard/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00004-of-00004.gguf` | 35,532,532,032 bytes | `59ae4308de27d99b8bcbcf7721d13080f8537d6b52b776dee3f602d0ecd94c9e` |

## Download and verify

Use an empty model directory and a separate directory for documentation. Set the release revision to an immutable commit for reproducible downloads, then choose one layout block.

```bash
set -euo pipefail
PATCH_REPO='Kiwimaker/DeepSeek-V4.1-Flash-1017-Maya-S-abliterated-GGUF'
PATCH_REV='main'  # Replace with a release commit to pin the download.
UPSTREAM='peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF'
DOCS="$PWD/abliteration-release-docs"
```

**Six-shard layout:** Maya v1.1.17, commit `8d9a9b5d705049a87da853717cdc59440252de30`.

```bash
LAYOUT='v1.1.17-one-set'
DEST="$PWD/maya-s-abliterated-v1117"
UPSTREAM_REV='276e0badcfb4a6d8f6c2b9d4da1b468cd339c221'
SHARD='DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00004-of-00006.gguf'
PATCH_SHA='89998155648e330d4f9eb0e0ab4810d9efe872dee0394ca6768df44158392a56'
hf download "$UPSTREAM" --revision "$UPSTREAM_REV" --local-dir "$DEST" \
  --include 'Maya-S/*-00001-of-00006.gguf' \
  --include 'Maya-S/*-00002-of-00006.gguf' \
  --include 'Maya-S/*-00003-of-00006.gguf' \
  --include 'Maya-S/*-00005-of-00006.gguf' \
  --include 'Maya-S/*-00006-of-00006.gguf' \
  --include 'dspark/dspark.bin' --include 'dspark/dspark.json'
```

**Four-shard layout:** Maya v1.1.16, commit `abc09c74ba692d808062dfa92d2b7c081831ac03`.

```bash
LAYOUT='v1.1.16-four-shard'
DEST="$PWD/maya-s-abliterated-v1116"
UPSTREAM_REV='26c6aeaf5ecce88fe47103f780279bcc6f1f0d9b'
SHARD='DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00004-of-00004.gguf'
PATCH_SHA='59ae4308de27d99b8bcbcf7721d13080f8537d6b52b776dee3f602d0ecd94c9e'
hf download "$UPSTREAM" --revision "$UPSTREAM_REV" --local-dir "$DEST" \
  --include 'Maya-S/*-00001-of-00004.gguf' \
  --include 'Maya-S/*-00002-of-00004.gguf' \
  --include 'Maya-S/*-00003-of-00004.gguf' \
  --include 'dspark/dspark.bin' --include 'dspark/dspark.json'
ENGRAM='DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00008-of-00009.gguf'
ENGRAM_CACHE=$(hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  "UD-Q8_K_XL/$ENGRAM" --revision c991b28e7c109eed10a55075ddb46f830b5b3059 --quiet)
mkdir -p "$DEST/Engram"
cp --reflink=auto "$ENGRAM_CACHE" "$DEST/Engram/$ENGRAM"
```

**Finish either layout:**

```bash
hf download "$PATCH_REPO" --revision "$PATCH_REV" --local-dir "$DOCS" \
  --include README.md --include LICENSE --include SHA256SUMS \
  --include abliteration-edit.npz --include apply_abliteration.py \
  --include "$LAYOUT/PATCHED.txt" --include "$LAYOUT/logs/*" \
  --include "$LAYOUT/UPSTREAM_SHA256SUMS"
(cd "$DOCS" && sha256sum --check --ignore-missing SHA256SUMS)
PATCH_CACHE=$(hf download "$PATCH_REPO" "$LAYOUT/$SHARD" --revision "$PATCH_REV" --quiet)
cp --reflink=auto "$PATCH_CACHE" "$DEST/Maya-S/$SHARD"
test "$(stat -c '%s' "$DEST/Maya-S/$SHARD")" = 35532532032
printf '%s  %s\n' "$PATCH_SHA" "$DEST/Maya-S/$SHARD" | sha256sum --check -
(cd "$DEST" && sha256sum --check "$DOCS/$LAYOUT/UPSTREAM_SHA256SUMS")
```

`--ignore-missing` verifies only downloaded documentation; the explicit check verifies the replacement and the prerequisite manifest verifies unchanged files. The six-shard model plus DSpark occupies about **347.4 GB decimal**, and the four-shard model plus separate Engram and DSpark about **388.9 GB**, before caches and pack outputs. Without reflinks, copies require another 35.5 GB for the replacement and 202.8 GB for separate Engram.

## Set up and run in Maya

Use a separate, fully verified pristine set of the chosen layout for Maya setup, including the original shard 4. Setup and automatic repair can replace edited files. Do not run them against the patched directory, and do not pass `--download-model`. Setup's size checks do not replace checksum verification. Back up existing Maya configuration files before setup; it writes `maya-ds4-hip.json`.

From a directory containing the selected Maya checkout as `maya/` and a verified pristine set as `normal-model/`, generate a normal configuration, then build a pack directly from the patched set:

```bash
MAYA="$(realpath maya)"
NORMAL="$(realpath normal-model)"
WORK="$PWD/abliterated-runtime"
mkdir -p "$WORK"
cd "$MAYA"
"$MAYA/.venv/bin/python" maya.py --backend hip --setup --plain \
  --model DeepSeek-V4.1-Flash-1017 --ds4-quant maya-s \
  --model-dir "$NORMAL/Maya-S" --gpus 0,1 --context 131072 \
  --no-vision --no-start --no-speed-calibration
cp "$MAYA/maya-ds4-hip.json" "$WORK/setup-normal.json"
OUT_PACK="$WORK/abliterated-pack"
ABSENT_VISION="$WORK/no-vision-assets"
test ! -e "$ABSENT_VISION"
"$MAYA/.venv/bin/python" "$MAYA/tools/ds4/pack_gguf.py" \
  --model "$DEST/Maya-S" --out "$OUT_PACK" \
  --drafter "$DEST/dspark/dspark.json" --vision "$ABSENT_VISION"
```

Inspect the selected version's CLI help and adapt device selection to your deployment. `--no-start` alone permits speed calibration; `--no-speed-calibration` disables it. The six-shard pack reads Engram from shards 5–6; the four-shard pack discovers sibling `Engram/`. Check that `ds4.txt` and `tensors.txt` reference the patched directory and every required shard. Never patch a hard link to a pristine shard: the script writes in place.

Copy `setup-normal.json` to `ds-abliterated.json`. In that copy, set the value after `--pack` in `args` to `OUT_PACK`, `tokenizer` to `OUT_PACK/tokenizer`, and `installer.model_dir` to `DEST/Maya-S`. Update `installer.models_dir` if present, and choose distinct `model_name`, `log` and port values. Preserve the executable, backend, context and appropriate device configuration. Changing installer metadata alone does not redirect the engine's pack. Remove `MAYA_DEBUG` and `STRATA_DEBUG` from the config's `env`; even `0` enables completion logging.

```bash
env -u MAYA_DEBUG -u STRATA_DEBUG "$MAYA/.venv/bin/python" \
  "$MAYA/serve/server.py" --engine strata --config "$WORK/ds-abliterated.json" \
  --host 127.0.0.1 --port 8094
```

For the four-shard comparison settings, use `--max-context 131072 --threads on --prefill-chunk 8192 --expert-profile none --dspark none`, HIP devices `0,1`, vision off and no speed calibration. Its 256-value expert geometry falls back from requested TP=2 to TP=1/layer splitting; explicit threading retains batched prefill. The six-shard v1.1.17 full comparison used DSpark on, matching its normal deployment settings. Select DSpark explicitly or calibrate a pristine copy; results below distinguish the two configurations.

## Maya-XS download

Use an empty destination. Set `PATCH_REV` to a release commit for a fixed revision.

```bash
set -euo pipefail
PATCH_REPO='Kiwimaker/DeepSeek-V4.1-Flash-1017-Maya-S-abliterated-GGUF'
PATCH_REV='main'
UPSTREAM='peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF'
UPSTREAM_REV='c273fcbdda76a1db074c2f414cb70edbfca2f2ef'
DEST="$PWD/maya-xs-abliterated"
DOCS="$PWD/abliterated-docs"
hf download "$PATCH_REPO" --revision "$PATCH_REV" --local-dir "$DOCS" \
  --include README.md --include LICENSE --include SHA256SUMS \
  --include maya-xs/PATCHED.txt --include maya-xs/UPSTREAM_SHA256SUMS
SHARD=$(awk '/^Replacement:/ {print $2}' "$DOCS/maya-xs/PATCHED.txt")
PATCH_SHA=$(awk '/^SHA256:/ {print $2}' "$DOCS/maya-xs/PATCHED.txt")
hf download "$UPSTREAM" --revision "$UPSTREAM_REV" --local-dir "$DEST" \
  --include 'Maya-XS/*.gguf' --exclude "Maya-XS/$SHARD" \
  --include dspark/dspark-xs2.bin --include dspark/dspark-xs2.json
hf download "$PATCH_REPO" "maya-xs/$SHARD" --revision "$PATCH_REV" \
  --local-dir "$DOCS"
mkdir -p "$DEST/Maya-XS"
cp --reflink=auto "$DOCS/maya-xs/$SHARD" "$DEST/Maya-XS/$SHARD"
(cd "$DOCS" && sha256sum --check --ignore-missing SHA256SUMS)
printf '%s  %s\n' "$PATCH_SHA" "$DEST/Maya-XS/$SHARD" | sha256sum --check -
(cd "$DEST" && sha256sum --check "$DOCS/maya-xs/UPSTREAM_SHA256SUMS")
```

## Maya v1.1.24+

Keep a complete pristine upstream set for setup. Setup and automatic repair can restore original shards; use them only on that pristine directory. Build the abliterated pack directly from the replacement set.

From a directory containing a Maya checkout as `maya/` and the pristine set as `normal-model/`:

```bash
MAYA="$(realpath maya)"
NORMAL="$(realpath normal-model)"
WORK="$PWD/abliterated-runtime"
mkdir -p "$WORK"
cd "$MAYA"
"$MAYA/.venv/bin/python" maya.py --setup --plain \
  --model DeepSeek-V4.1-Flash-1017 --ds4-quant maya-xs \
  --model-dir "$NORMAL/Maya-XS" --context 131072 \
  --no-vision --no-start --no-speed-calibration
OUT_PACK="$WORK/abliterated-pack"
test ! -e "$WORK/no-vision-assets"
"$MAYA/.venv/bin/python" "$MAYA/tools/ds4/pack_gguf.py" \
  --model "$DEST/Maya-XS" --out "$OUT_PACK" \
  --drafter "$DEST/dspark/dspark-xs2.json" --vision "$WORK/no-vision-assets"
```

Copy the generated normal configuration to a separate configuration. Set its `args` value after `--pack` to `OUT_PACK`, `tokenizer` to `OUT_PACK/tokenizer`, and `installer.model_dir` to `DEST/Maya-XS`; update `installer.models_dir` if present. Choose a separate model name, log file and port. Keep DSpark enabled with the XS2 drafter. Remove `MAYA_DEBUG` and `STRATA_DEBUG` from the configuration environment, including values of `0`.

Use Maya's documentation for platform-specific setup and serving options. Other layouts require their corresponding upstream revision and complete shard set; `PATCHED.txt` identifies the replacement, and `UPSTREAM_SHA256SUMS` identifies unchanged prerequisites.

## Additional unsloth layouts

These four builds apply the same edit and are **unevaluated**: no quality, refusal or speed comparison was run. Only Maya-S has the measured results below. Model sizes are total files on disk, not RAM requirements. Each folder supplies a replacement for just the named shard; all other shards must come from the exact pinned source.

| Folder | Whole model | Replacement shard | Replacement bytes | Coverage |
| --- | ---: | --- | ---: | --- |
| `unsloth-UD-Q4_K_XL/` | 500.7 GB | `DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00009-of-00009.gguf` | 9,165,923,392 | Documented Maya option; unevaluated |
| `unsloth-UD-IQ3_S/` | 423.6 GB | `DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00007-of-00008.gguf` | 15,235,610,176 | GGUF edit only; incompatible with inspected Maya v1.1.17; unevaluated |
| `unsloth-UD-IQ3_XXS/` | 394.7 GB | `DeepSeek-V4.1-Flash-1017-UD-IQ3_XXS-00006-of-00007.gguf` | 36,893,972,640 | GGUF edit only; incompatible with inspected Maya v1.1.17; unevaluated |
| `unsloth-UD-Q8_K_XL/` | 508.4 GB | `DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00009-of-00009.gguf` | 16,835,109,952 | Documented Maya option; unevaluated |

UD-Q4_K_XL and UD-Q8_K_XL are documented Maya v1.1.17 options (`gguf-q4` and `gguf-q8`). **Both IQ3 builds are incompatible with the inspected Maya v1.1.17**, despite packer type acceptance: their Q6_K Engram projection matrices are unsupported by its dense loader. IQ3_S also has a Q6_K output matrix; IQ3_XXS additionally has IQ2_XS experts with no supported engine path. These are GGUF edits only, requiring another loader with independently validated support. Smaller IQ1/IQ2 and Q2 variants were not built because their edit targets include unsupported Q5_K/Q6_K storage. Maya-M has no files in the inspected upstream snapshots.

Choose the Q4 or Q8 row for Maya and set `QUANT` and `SHARD` to its values. Use an empty directory. Fetch every unchanged upstream shard, then put the replacement under the same original basename in `DEST/QUANT/`. Keep each complete split separate. IQ3 files can be assembled in the same arrangement for a different compatible loader; the Maya instructions below do not apply to them.

```bash
set -euo pipefail
QUANT='UD-Q4_K_XL'  # Or UD-Q8_K_XL for Maya.
SHARD='DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00009-of-00009.gguf'  # Match the chosen row.
LAYOUT="unsloth-$QUANT"
DEST="$PWD/unsloth-abliterated"
DOCS="$PWD/abliteration-release-docs"
PATCH_REPO='Kiwimaker/DeepSeek-V4.1-Flash-1017-Maya-S-abliterated-GGUF'
PATCH_REV='main'  # Replace with the release commit.
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  --revision c991b28e7c109eed10a55075ddb46f830b5b3059 --local-dir "$DEST" \
  --include "$QUANT/*.gguf" --exclude "$QUANT/$SHARD"
hf download "$PATCH_REPO" --revision "$PATCH_REV" --local-dir "$DOCS" \
  --include README.md --include LICENSE --include SHA256SUMS \
  --include abliteration-edit.npz --include apply_abliteration_v2.py \
  --include "$LAYOUT/PATCHED.txt" --include "$LAYOUT/UPSTREAM_SHA256SUMS" \
  --include "$LAYOUT/logs/*"
(cd "$DOCS" && sha256sum --check --ignore-missing SHA256SUMS)
PATCH_CACHE=$(hf download "$PATCH_REPO" "$LAYOUT/$SHARD" --revision "$PATCH_REV" --quiet)
cp --reflink=auto "$PATCH_CACHE" "$DEST/$QUANT/$SHARD"
PATCH_SHA=$(awk '/^Patched SHA256:/ {print $3}' "$DOCS/$LAYOUT/PATCHED.txt")
printf '%s  %s\n' "$PATCH_SHA" "$DEST/$QUANT/$SHARD" | sha256sum --check -
(cd "$DEST" && sha256sum --check "$DOCS/$LAYOUT/UPSTREAM_SHA256SUMS")
```

For Maya v1.1.17, pack the fully assembled directory directly. The Engram tensors are already in this complete split; no separate `Engram/` is needed. This example creates a text-only pack without DSpark. To enable DSpark, obtain and verify the pinned `dspark/dspark.bin` and `dspark/dspark.json` from peasantsmith revision `276e0badcfb4a6d8f6c2b9d4da1b468cd339c221`, then point `--drafter` at that JSON.

```bash
MAYA="$(realpath maya)"
WORK="$PWD/unsloth-runtime"
mkdir -p "$WORK"
test ! -e "$WORK/no-vision-assets"
test ! -e "$WORK/no-drafter.json"
OUT_PACK="$WORK/abliterated-pack"
"$MAYA/.venv/bin/python" "$MAYA/tools/ds4/pack_gguf.py" \
  --model "$DEST/$QUANT" --out "$OUT_PACK" \
  --drafter "$WORK/no-drafter.json" --vision "$WORK/no-vision-assets"
```

Use the configuration procedure above with `--ds4-quant gguf-q4` or `gguf-q8` on a pristine supported set. Replace its `--model-dir "$NORMAL/Maya-S"` with `--model-dir "$NORMAL/$QUANT"`. Then set its engine `--pack` to this `OUT_PACK`, tokenizer to `OUT_PACK/tokenizer`, and installer model directory to `DEST/QUANT`; update installer models directory if present. For the no-drafter example set `--dspark none`. Avoid setup or automatic repair against edited shards. IQ3 builds cannot use this Maya version.

[apply_abliteration_v2.py](apply_abliteration_v2.py) extends the original script to F32/F16/BF16 and Q8_0. It computes the same float64 projection, rounds to the original floating storage format or requantizes Q8_0, and preserves shape, offsets and byte size. Logs include the subspace ratio against `abs(1-s)`, storage error, and decode/encode baseline drift. Original floating-to-quant error is unknown without the floating source; baseline drift is a different measurement. Q5_K/Q6_K are skipped because the inspected gguf-py has no implemented quantizers for them.

To reproduce an added shard, verify its pristine hash in `PATCHED.txt`, make an independent copy, and run the v2 script with `--require-all-supported`, first with `--dry-run` and then without. Each added build has **62 patched tensors, 0 skipped**. Full verification rehashes both source and output and confirms all bytes outside the targeted tensor ranges are identical. These builds used NumPy 2.5.4 and gguf-py from llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`; CPU storage checks do not establish runtime behavior.

## Measured results and limits

The recommended **Maya-S six-shard layout on v1.1.17** received the full comparison, completed on 2026-10-21, with **DSpark on**, context 131072 and vision off. The normal model ran first, followed by the edited model, with warmed caches and matched prompts/settings. All suites completed with zero request errors and no prompt or output-cap reductions. These measurements do not establish broad capability preservation or safety.

### Maya-XS on Maya v1.1.24, DSpark XS2 on

Same frozen protocol as the Maya-S runs below.

| Measure | Normal Maya-XS | Abliterated Maya-XS |
| --- | --- | --- |
| Corrected short-answer accuracy (25 questions asked twice) | 50/50 | 49/50 |
| Answers that hit the 6,000-token cap without a final answer | 0/50 | 1/50 |
| Needle retrieval near 60K and 120K tokens, 3 depths each | 6/6 | 6/6 |
| Harmful refusal screen, 64-token cap | 50/50 refused (100%) | 1/50 refused (2%) |
| Borderline lawful-intent refusal screen | 0/50 refused (0%) | 0/50 refused (0%) |
| Decode median, four probes | 40.4 tokens/s | 36.7 tokens/s |
| Prefill, same 7,282-token prompt | 645 tokens/s | 642 tokens/s |

The one miss is an easy factual question on which the abliterated model kept reasoning until the 6,000-token cap and gave no final answer; the normal model answered it correctly but also spent about 4,600 tokens. This counts against the predeclared "no new loops" criterion, so the XS result is not a clean pass. In a separate check of that question (four requests, 12,000-token cap) the abliterated model answered three times and ran to the cap once, so set a generous `max_tokens`; the Maya server's `thinking_budget` setting does not apply to the DeepSeek engine in v1.1.24. Exact harmless-answer agreement was 29/50. Only Maya-S was evaluated more extensively.

### Maya v1.1.17, six shards, DSpark on

| Measure | Normal Maya-S | Abliterated Maya-S |
| --- | --- | --- |
| Corrected / original-key accuracy, a 25-question short-answer set asked twice | 50/50 / 48/50 | 50/50 / 48/50 |
| Loop heuristic flags | 0/50 | 0/50 |
| Accuracy median request wall time | 2.8060 s | 3.2022 s |
| Decode median, 4 engine timing samples | 23.15 tokens/s | 26.50 tokens/s |
| Prefill, 7,282 prompt tokens evaluated, 0 cached | 238.4 tokens/s | 231.3 tokens/s |
| Isolated first-token latency | Unavailable | Unavailable |
| 60K needle retrieval, 10% / 50% / 90% depth | 3/3; actual 59,989 / 59,988 / 59,989 tokens | 3/3; same lengths and prompt hashes |
| 120K needle retrieval, 10% / 50% / 90% depth | 3/3; actual 119,998 / 119,998 / 119,997 tokens | 3/3; same lengths and prompt hashes |
| Harmless greedy answers | 50/50 measured; 4 capped | 50/50 measured; 4 capped |
| Paired exact / NFKC-whitespace normalized agreement | 26/50 (52%) / 26/50 (52%) | Same cross-model statistic |
| First-token KL | Unavailable: server exposes no logprobs | Unavailable |
| Harmful refusal screen, 64-token cap | 50/50 refused (100%); 48 capped | 2/50 refused (4%); 50 capped |
| Borderline lawful-intent refusal screen, 64-token cap | 0/50 refused (0%); 50 capped | 0/50 refused (0%); 50 capped |
| Refusal errors / empty answers, both categories | 0 / 0 | 0 / 0 |
| Warmup request wall time, engine already loaded | 2.7645 s | 1.5284 s |

The corrected accuracy difference was 0/50, loop flags were 0/50 each, matched needle retrieval was 6/6 each, and harmful refusal fell by **96 percentage points**. These satisfy the same prespecified screen as the historical comparison: lose no more than 2/50 corrected accuracy points, introduce no new loops, preserve needle retrieval at measured lengths, and reduce harmful refusal by at least 20 percentage points. All refusal answers were scorable, with no error/empty exclusions. Reduced refusal does not verify substantive harmful assistance.

All requests used temperature 0 and top_p 1. Accuracy used medium reasoning and a 6,000-token cap; four decode prompts used low reasoning/400 tokens; prefill used 40,000 characters of neutral prose and low reasoning/32 tokens; six needle tests used bounded length calibration and low reasoning/2,000 tokens. The 50 harmless prompts used thinking disabled/128 tokens. The 50 AdvBench-derived harmful and 50 original borderline lawful-intent prompts used thinking disabled/64 tokens. Only refusal flags and answer prefixes of at most 60 characters were retained.

Accuracy uses substring screening, correcting a known arithmetic key from 47 to 53; repeated questions are not independent observations. Loop and refusal labels are heuristics. Short completion caps limit interpretation: 98/100 normal and all 100 edited refusal-suite answers were capped. Manual review of harmless answers found no actual repetition loops or substantial new core factual error in this bounded set. Both models correctly answered the axial-tilt item and returned valid raw JSON on the item that previously exposed edited-model regressions. Both still over-expanded some answers, violated a one-line instruction, and hit the harmless cap on the same four items. This does not establish broad factual accuracy or instruction-following preservation.

Greedy output text varied between the two repetitions of **12/25 accuracy questions in each model**, despite unchanged sampling settings. This DSpark-enabled setup therefore did not yield fully repeatable greedy text. The historical DSpark-off setup also varied, so the variation cannot be attributed to DSpark alone. Cross-model agreement measures answer-text consistency, not semantic equivalence or KL.

The edited model's decode median was **14.5% higher** and prefill **3.0% lower** in this single run order. Four decode samples, one prefill request, different generated text and warmed caches do not establish a reproducible speed difference. Accuracy request wall times also include differing answer/reasoning lengths. Warmup followed health checks with the engine already loaded; it is neither cold startup nor isolated first-token latency. Repeated or reversed-order full comparisons remain unmeasured.

### Historical comparison: Maya v1.1.16, four shards, DSpark off

The following full comparison was completed on 2026-10-20 with the four-shard Maya-S layout and separate Engram. The normal model ran first, then the edited model, with warmed caches and matched prompts/settings. Its results apply to that configuration. Version, layout, Engram storage, expert-profile policy, threading defaults and DSpark differ from the six-shard comparison above, so changes between the two comparisons cannot be assigned to a single feature.

| Measure | Normal Maya-S | Abliterated Maya-S |
| --- | --- | --- |
| Corrected / original-key accuracy, a 25-question short-answer set asked twice | 50/50 / 48/50 | 50/50 / 48/50 |
| Loop heuristic flags | 0/50 | 0/50 |
| Accuracy median request wall time | 4.8116 s | 3.7947 s |
| Decode median, 4 engine timing samples | 24.25 tokens/s | 21.15 tokens/s |
| Prefill, 7,282 prompt tokens evaluated, 0 cached | 240.2 tokens/s | 231.1 tokens/s |
| Isolated first-token latency | Unavailable | Unavailable |
| 60K needle retrieval, 10% / 50% / 90% depth | 3/3; actual 59,989 / 59,988 / 59,989 tokens | 3/3; same lengths and prompt hashes |
| 120K needle retrieval, 10% / 50% / 90% depth | 3/3; actual 119,998 / 119,998 / 119,997 tokens | 3/3; same lengths and prompt hashes |
| Harmless greedy answers | 50/50 measured; 4 capped | 50/50 measured; 5 capped |
| Paired exact / NFKC-whitespace normalized agreement | 28/50 (56%) / 28/50 (56%) | Same cross-model statistic |
| First-token KL | Unavailable: server exposes no logprobs | Unavailable |
| Harmful refusal screen, 64-token cap | 50/50 refused (100%); 49 capped | 2/50 refused (4%); 50 capped |
| Borderline lawful-intent refusal screen, 64-token cap | 0/50 refused (0%); 50 capped | 0/50 refused (0%); 50 capped |
| Refusal errors / empty answers, both categories | 0 / 0 | 0 / 0 |
| Warmup request wall time, engine already loaded | 1.9478 s | 1.0476 s |

The corrected accuracy difference was 0/50, loop flags were 0/50 each, matched needle retrieval was 6/6 each, and harmful refusal fell by **96 percentage points**. These satisfy the prespecified screen: lose no more than 2/50 corrected accuracy points, introduce no new loops, preserve needle retrieval at measured lengths, and reduce harmful refusal by at least 20 percentage points. All refusal answers were scorable, with no error/empty exclusions. Reduced refusal does not verify substantive harmful assistance.

All requests used temperature 0 and top_p 1. Accuracy used medium reasoning and a 6,000-token cap; four decode prompts used low reasoning/400 tokens; prefill used 40,000 characters of neutral prose and low reasoning/32 tokens; six needle tests used bounded length calibration and low reasoning/2,000 tokens. The 50 harmless prompts used thinking disabled/128 tokens. The 50 AdvBench-derived harmful and 50 original borderline lawful-intent prompts used thinking disabled/64 tokens. No prompt or suite cuts were made.

Accuracy uses substring screening, correcting a known arithmetic key from 47 to 53; repeated questions are not independent observations. Loop and refusal labels are heuristics. Short completion caps limit interpretation, particularly with 49/50 normal harmful answers and all 100 edited refusal-suite answers capped. Harmless-answer inspection found a new factual error about Earth's roughly 23.5° axial tilt (relative to the orbital plane instead of its perpendicular), and a Markdown fence added around a requested JSON object. Changed verbosity also affected agreement; no new repetitive loop was found. Agreement is an output consistency statistic, not semantic equivalence or KL.

Decode declined **12.8%** and prefill **3.8%** in this single run order. Four decode samples and one prefill request do not establish statistically meaningful speed differences. Engine timings measure request throughput; warmup followed health checks with the engine already loaded, so its wall time is neither cold startup nor isolated first-token latency. Broader capability testing and repeated/reversed-order timing remain unmeasured.

Across the two configurations, corrected accuracy remained 50/50 per model, needle retrieval remained 6/6 per model, and harmful refusal remained 100% normal versus 4% edited. Exact/normalized harmless agreement was 56% in the historical comparison and 52% in the six-shard comparison. The earlier axial-tilt wording error and fenced-JSON answer were not observed in the corresponding current answers; this is a bounded observation, not evidence that those failure modes are eliminated.

## Edit math and reproduction

[apply_abliteration.py](apply_abliteration.py) applies the parameters in `abliteration-edit.npz`:

```text
W' = W - scales[L,k] * Q @ (Q.T @ W)
k=0: blk.L.attn_output_b.weight
k=1: blk.L.ffn_down_shexp.weight
Q: (5120, 2); scales: (40, 2)
W: (5120, input_features), residual-stream outputs as rows
```

It dequantizes eligible Q8_0 weights, computes the update in float64, casts to float32 and requantizes to Q8_0 without changing byte size. Zero-strength entries are untouched. Both Maya-S patch logs report **62 patched tensors, 0 skipped**. This applies the edit to quantized Maya-S dense weights; mrfakename's FP8 artifact is a different quantization. For orthonormal `Q`, the ideal subspace norm ratio is `abs(1-s)`; strengths above one reverse that component's sign. Requantization perturbs this ratio; per-tensor diagnostics report the measured ratio and error.

The operation is **not idempotent**. Inspect the script and run it only on an independent pristine copy, after verifying the matching upstream hash. Do not patch a published replacement again.

```bash
python3 -m venv patch-env
patch-env/bin/python -m pip install numpy gguf
patch-env/bin/python apply_abliteration.py abliteration-edit.npz original-shard4.gguf --dry-run
patch-env/bin/python apply_abliteration.py abliteration-edit.npz original-shard4.gguf
sha256sum original-shard4.gguf
```

Original dependency versions were not captured; installing current dependencies does not guarantee byte-identical reproduction. Verify counts, size, checksum and behavior. The parameters occupy 43,334 bytes, SHA256 `83273cdbe06e65ad2852dd09e8d976a0e8ce3209788938add325989c20b63b02`; the script SHA256 is `84f1e9fa9fa12dffe61f0abbc8742e34551b3bb15f618c8783791b96d4b76260`.

## Provenance

| Source | Immutable revision | Use |
| --- | --- | --- |
| [DeepSeek base](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) | `1bae0e1088993b9b82de08f59de5be5a34e42170` | Base model |
| [peasantsmith six-shard](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/tree/276e0badcfb4a6d8f6c2b9d4da1b468cd339c221) | `276e0badcfb4a6d8f6c2b9d4da1b468cd339c221` | Six-shard Maya-S and DSpark |
| [peasantsmith four-shard](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/tree/26c6aeaf5ecce88fe47103f780279bcc6f1f0d9b) | `26c6aeaf5ecce88fe47103f780279bcc6f1f0d9b` | Four-shard Maya-S and DSpark |
| [unsloth Engram](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/c991b28e7c109eed10a55075ddb46f830b5b3059) | `c991b28e7c109eed10a55075ddb46f830b5b3059` | Engram for four-shard layout and the added unsloth quantizations |
| [mrfakename edit](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated/tree/517efab9ccfef7f714dbd7d679f181a86f43ccf9) | `517efab9ccfef7f714dbd7d679f181a86f43ccf9` | Rank-2 subspace and layer strengths |

The pristine six-shard shard 4 hash is `d60dba35fac9601cd5c6026439dd980c457a5bb1c009192ea539b220108ea0db`; the pristine four-shard shard 4 hash is `506dd95a2ea4f17c933534d730ec72b2ef1b82eec4847770f4f23c38344f96ca`. Their sizes match, but their bytes differ. Prerequisite checksums match Maya's pinned download manifest. Full reads on 2026-10-20 reproduced both published replacement hashes and sizes. `SHA256SUMS` covers release files and replacement GGUF paths, excluding itself and upload instructions.

## Credits and license

[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) released the base model under MIT. [peasantsmith](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF) produced the Maya-S quantization, DSpark files and one-set layout in a repository with MIT license metadata. [unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) provides the added quantizations and the Engram GGUF used by the four-shard layout in a repository with MIT license metadata. [mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) supplied the edit parameters in an MIT repository. [Project Maya](https://github.com/mw00/project-maya) supplies the inference engine and Maya-S support. This derivative is not an official release or endorsement by an upstream author.

This derivative is released under **MIT**. Retain the copyright and permission notice in [LICENSE](LICENSE) when redistributing.

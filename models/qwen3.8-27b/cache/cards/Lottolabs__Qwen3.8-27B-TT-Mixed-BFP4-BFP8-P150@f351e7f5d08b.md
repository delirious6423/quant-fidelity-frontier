---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- tenstorrent
- blackhole
- p150
- ttnn
- tt-metal
- bfp4
- bfp8
- mixed-precision
- gptq
- speculative-decoding
- dflash
- prefix-caching
- vision
- long-context
- custom-runtime
---

# Qwen3.8-27B — TT-native mixed BFP4/BFP8 (GPTQ), single P150, native 262K context

A hardware-specific checkpoint (text, image and video input) and custom runtime derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, for one Tenstorrent Blackhole P150.

This is a community experimental release by **Lottolabs**, not an official Qwen or Tenstorrent release. It is not GGUF, AWQ, an AutoGPTQ/GPTQModel int4 checkpoint, bitsandbytes, NVFP4, or an ordinary Transformers checkpoint. `transformers.from_pretrained()` and hosted Inference Providers are not supported launch paths.

## What changed in this release (`s27v3-20261022`)

The checkpoint tensors are unchanged (manifest sha256 `47b781ab…`). The reload proofs were re-recorded on the two new runtime images. **Exact prefix caching now covers prompts with images and videos**, as vLLM's prefix cache does on a GPU; the previous release re-prefilled the whole conversation, and re-ran the vision tower, on every turn that contained an image or a video.

- **The resume key covers the images and videos.** A prompt resumes from a saved 2,048-token boundary only if its tokens and every image or video that starts before that boundary match: each item is keyed by vLLM's content hash of the item and its placeholder position (the role the multimodal hashes play in vLLM's block hashes). A different image of the same size never matches; text-only keys are unchanged.
- **Images and videos that end before the resume point are not re-encoded**; an item that crosses it is encoded and spliced from the boundary on.
- **Measured on the 128K profile** with five multimodal conversations of 4,142–4,900 tokens (an image first, an image after 1,500 tokens of text, a video, a same-size different image, and a new image added in the follow-up turn): the six follow-up turns resumed at 4,096 tokens and took 0.36–1.67 s instead of 3.8–5.2 s, the different image did not reuse the first image's state, and all 11 answers are token-identical to a caching-off server ([`evidence/s27v3/`](evidence/s27v3/)).
- **Two new runtime images** (the s27v2 images plus this change; `runtime-image.tar.gz` holds both):

  | Image | ID | Used by |
  |---|---|---|
  | `lottolabs/qwen27b-tt-p150:ctx5-s27v3` | `sha256:59f32023c9f35d9a5022406e79e07fca1514d75a95d0e695a4a3606f2b9ef55d` | 8K and 128K profiles (BF16 KV up to 64K, BFP8 above) |
  | `lottolabs/qwen27b-tt-p150:ctx6-kv4-s27v3` | `sha256:8fdadce02519a9a26984338d4151428c503b758e120f9dc32eb7cefba25917fe` | 262K profile (BFP8 K, 4-bit V) |

### Images and videos (`s27v2-20261022`, still included)

Image and video input follows the upstream model's processor and vLLM's defaults, as on a GPU vLLM server.

- **Any number of images and videos per request**, mixed, in any turn of the conversation (vLLM's default per-prompt limit). Images go in `image_url` parts and videos in `video_url` parts (data URLs or links); videos are decoded with OpenCV and sampled at 2 frames per second, as vLLM does for Qwen3-VL.
- **The checkpoint processor's size limits**: images up to 16,777,216 pixels (16,384 image tokens) and videos up to 25,165,824 pixels over all frames. Below a 16,384-token context (the 8K profile) both are capped at max-model-len × 1,024 pixels (8,388,608 at 8K, 8,192 tokens), so one item fits the context. `--max-pixels` sets another cap.
- **The tower runs per image and per video frame.** Patches attend only within their own frame upstream, so this is exact. Padding is masked through an otherwise zero head-dimension column instead of an S × S mask, and frames longer than 8,192 patch rows run their projections, attention query blocks and MLP in row chunks. A 16,384-token image (65,536 patches) fits beside the 128K and 262K KV pools: the tower takes 25.6 s for it and matches the BF16 reference tower with a Pearson correlation of 0.975, and the 128K profile answers in 42 s.
- **Fixed: vision rows could be overwritten during prefill.** The packed vision rows lived in a device buffer allocated after the prefill trace was captured, and the chunked prefill replays that trace between splices, so the replays could overwrite it. A 16,384-token image produced garbage. The rows now stay on the host, where the splice reads them anyway. In `s27v` (images of at most 1,024 tokens) this never showed in the checks, but the hazard was the same.
- **Text and single images are unchanged.** The 24-image set (8K), the 8-image set (262K) and their long descriptions are token-identical to `s27v`, and 5 greedy text prompts are token-identical to `s27` at the same speed.

**Image and video checks** (single P150, greedy, thinking off; [`evidence/s27v2/`](evidence/s27v2/)):

| Check | Result |
|---|---|
| Three images in one message, word and color of each | all words and colors right in the 8K, 128K and 262K profiles ("LANTERN 73" was read as "LANTERN 42") |
| An image in turn 1, a new image in turn 2, a question about both | right in every profile |
| A 4 s video: a square moving right, red for the first half, blue for the second | "red, blue, right" in every profile |
| An image and a video in one message | right in every profile |
| A 4096 × 4096 image (16,384 tokens), 128K and 262K profiles | "SUMMIT 96, yellow", 42 s end to end |
| The same card at 1,024–4,096 px, 128K profile | right at every size, except "SUMMT" for "SUMMIT" at 1,024 px |
| DFlash2 on vs off, the multi-image, multi-turn, video and mixed requests (8K) | token-identical |
| The same probe on an RTX 3090 (vLLM, W4A16 target, same limits) | all pass; the answers differ only in the digits read or left out |

### Vision layer (`s27v-20261021`, still included)

- **The upstream vision tower** is shipped as [`vision/model.safetensors`](vision/model.safetensors): the base revision's 333 `model.visual.*` tensors (460.7M parameters, BF16, 0.92 GB), byte-identical values. It runs on the same P150 next to the text model, with BFP8 weights (0.46 GB) and HiFi4 / FP32 accumulation. Its matmuls, attention and norms use small blocks so their L1 buffers fit beside the resident GDN state.
- **Fixed: the tower attended to padding.** Image patches are padded to a fixed length, and the upstream TT tower did not mask the padding out of its attention: its output matched an FP32 reference of the tower with a Pearson correlation of only 0.936. With the mask, it is 0.989 with BFP8 weights and 0.996 with BF16 weights and HiFi4.
- **Speculative greedy output on image prompts equals non-speculative output.** Image tokens shift the M-RoPE positions of the text after them; the captured verification step applies the request's offset, as decode already did.

Its checks ([`evidence/s27v/vision/`](evidence/s27v/vision/)): 23/24 synthetic images at 8K (the miss answered "ROCKET 91" for the word "ROCKET"; an RTX 3090 with vLLM gave the same answers up to capitalization and got that one), 8/8 at 128K and 262K, one image plus a 231,203-token prompt answered correctly at 262K (395 s), and image descriptions decode at about 44 tok/s (2.9 accepted tokens per DFlash2 step vs 3.9 on text).

### Changes in `s27-20261005` (still included)

- **Exact prefix caching** (on by default). A prompt that extends an earlier prompt resumes at the last 2,048-token prefill chunk boundary the earlier prompt crossed, and then runs the same chunks a full prefill would run.
- **Speculative greedy output is token-identical to non-speculative output.** The verification path rounds like single-row decode.
- **DFlash2 at 262K.** The 262K profile runs DFlash2 block-8 speculative decoding with a KV cache that stores K in BFP8 and V in 4-bit BFP4 after a Hadamard rotation (`bfp8_v4r`, 26,624 instead of 34,816 bytes per token). That frees the 2.15 GB the BFP8 drafter needs at 262,144 tokens. At 261,632 prompt tokens it decodes about 28% faster than MTP-3 on the same image.
- **On-device exact sampling and lossless speculative sampling.** Temperature, top-k (1–64), top-p, min-p and presence/frequency/repetition penalties run on device with a seeded Philox RNG and reproduce vLLM's sampler. Sampled requests are speculated too: DFlash2 draws its draft by the probabilistic selector walk and MTP by sampled chains, and the target accepts or resamples them by speculative rejection sampling, so the output distribution is the target model's.

## Quality

**Checkpoint.** These are teacher-forced device runs on the P150 (unchanged from the previous release; the tensors are identical). The eval set has 34 records: 14 multi-turn chats, 12 Python files and 8 instructions, with 42,129 scored positions. Chat and instruction records are scored on the assistant tokens only.

Against the TT all-BFP8 build (same runtime, every matrix in BFP8), mean KL divergence and top-1 agreement:

| Corpus | Round-to-nearest BFP4 (release dual-noc) KL / top-1 | GPTQ BFP4 (this checkpoint) KL / top-1 |
|---|---:|---:|
| Chat | 0.180 / 0.838 | **0.070** / **0.897** |
| Code | 0.053 / 0.940 | **0.047** / **0.943** |
| Instruct | 0.013 / 0.960 | **0.011** / **0.967** |
| All | 0.103 / 0.899 | **0.055** / **0.925** |

Against a CPU BF16 reference (Δ perplexity, top-1 agreement, KL):

| Corpus | TT all-BFP8 | Round-to-nearest | GPTQ (this checkpoint) |
|---|---:|---:|---:|
| Code | +0.59% / 0.952 / 0.037 | +4.94% / 0.925 / 0.091 | **+4.25%** / **0.931** / **0.086** |
| Instruct | −0.45% / 0.968 / 0.010 | +0.24% / 0.957 / 0.019 | **−0.09%** / **0.960** / **0.019** |

**The 4-bit V cache of the 262K profile.** Long-context teacher forcing of the 4-bit V cache against the BFP8 KV cache on the same runtime (ctx6-kv4, the base of the 262K image), 4,096 scored positions per record ([`evidence/s27/kv4/quality/tables.md`](evidence/s27/kv4/quality/tables.md)):

| Corpus (contexts) | ΔNLL ± SE vs BFP8 KV | top-1 agreement | mean KL |
|---|---:|---:|---:|
| Book (32K, 64K, 128K) | +0.0018 ± 0.0012 | 0.957 | 0.0074 |
| Book (262K) | +0.0042 ± 0.0023 | 0.937 | 0.0093 |
| Code (32K–262K) | +0.0010 ± 0.0008 | 0.981 | 0.0045 |
| Long chat (61K) | **+0.0071 ± 0.0010** | 0.966 | 0.0062 |

Book and code stay within 2 standard errors of the BFP8 KV cache. Needle retrieval (answer-token top-1) was 100% at 32K, 128K and 262K with both caches, and GSM8K-100 (greedy, DFlash2) scored 95 with the 4-bit V cache vs 94 with BFP8 KV. **The long chat costs +0.0071 nats per token, about 0.7% perplexity.** This is a known cost of the 262K profile only; the 8K and 128K profiles do not use the 4-bit V cache.

**Sampling.** GSM8K-100 with sampling on the 8K DFlash2 profile, one 100-question run each: coding preset 95, non-thinking preset 92 ([`evidence/s27/sampling/gsm-spec/`](evidence/s27/sampling/gsm-spec/)).

## Speed

Single P150, one sequence, 512 streamed output tokens, chat prompts. Decode tok/s is the p50 over 3 requests (distinct seeds for the sampled presets) and counts client-observed tokens after the first one ([`evidence/s27/sampling/`](evidence/s27/sampling/), `sb8k`, `sb128k`, `sb262k`, `mtp262kb`, `ns8k`, `base8k`). Presets: thinking = temperature 1.0, top-p 0.95, top-k 20, presence 1.5; coding = temperature 0.6, top-p 0.95, top-k 20; non-thinking = temperature 0.7, top-p 0.8, top-k 20, presence 1.5.

| Profile | Prompt tokens | Greedy | Thinking | Coding | Non-thinking |
|---|---:|---:|---:|---:|---:|
| 8K, DFlash2 (default) | 128 | 50.1 | 48.6 | 52.5 | 46.3 |
| 8K, DFlash2 (default) | 2,048 | 54.5 | 48.4 | 50.0 | 49.0 |
| 128K, DFlash2 | 128 | 49.0 | 49.0 | 50.3 | 46.7 |
| 128K, DFlash2 | 2,048 | 49.8 | 44.3 | 51.2 | 45.5 |
| 128K, DFlash2 | 32,768 | 47.2 | 43.8 | 43.2 | 42.2 |
| 262K, DFlash2 + 4-bit V | 65,024 | 44.9 | 38.5 | 42.1 | 41.8 |
| 262K, DFlash2 + 4-bit V | 261,632 | **30.2** | **28.6** | **31.4** | 29.2 |
| 262K, MTP-3 + 4-bit V (`--spec mtp`) | 65,024 | 35.5 | 32.9 | 36.5 | 33.3 |
| 262K, MTP-3 + 4-bit V (`--spec mtp`) | 261,632 | 23.7 | 23.4 | 24.6 | 23.7 |
| 8K, non-speculative (`--spec off`) | 128 / 2,048 | 20.2 / 20.0 | 20.2 / 20.0 | 20.2 / 20.1 | 20.2 / 20.0 |

Without the sampling layer (runtime ctx5, the same 8K launch otherwise), sampled requests ran on the host sampler at **14.2 tok/s** (thinking preset, 128 and 2,048 tokens) while greedy ran at 50.1 / 54.5 tok/s ([`evidence/s27/sampling/base8k/`](evidence/s27/sampling/base8k/)). A request that leaves every sampling field at its default (the checkpoint's `generation_config.json`) decoded at 52.8 / 46.9 tok/s at 128 / 2,048 tokens on the 8K profile.

In the previous release (`gptq-ctx2`), greedy decode was 49.1 / 48.1 tok/s at 128 / 2,048 tokens (8K DFlash2) and 24.5 tok/s at 261,632 tokens (262K MTP-3).

**Prefix caching**, measured on runtime ctx5, the base of the 8K/128K image ([`evidence/s27/spec-exact/ctx5-verify/`](evidence/s27/spec-exact/ctx5-verify/) `df128k/mt.json`; the caching-off times are from runtime ctx4, [`evidence/s27/prefix-cache/ctx4-verify/dfoff128k/`](evidence/s27/prefix-cache/ctx4-verify/dfoff128k/)): a second turn that extends a 32K, 64K or 127K conversation on the 128K profile started its answer in 3.0 / 3.8 / 5.3 s instead of 31.2 / 71.4 / 172.3 s without caching, and repeating that turn took 0.8 / 1.0 / 1.3 s. All texts equal the non-speculative, caching-off reference.

TTFT on a cold prompt is about 0.2 s at 128 tokens, 1.7 s at 2,048, 31 s at 32K, 71 s at 64K, 171 s at 127K and about 8 min at 258K.

## Download and serve

Prerequisites:

- Linux with a supported P150 driver and hugepage mounts at `/dev/hugepages` and `/dev/hugepages-1G`.
- Docker and Python 3.11 or newer.
- Exclusive ownership of one P150.
- Disk space for:
  - the ~22.1 GB checkpoint;
  - the 3.8 GB drafter and the 0.92 GB vision tower;
  - the 5.6 GB compressed runtime archive (both images) and the loaded images;
  - writable tensor/device caches (~23 GB per decoding mode and image).

```bash
curl --fail --location --output launch.py \
  https://huggingface.co/Lottolabs/Qwen3.8-27B-TT-Mixed-BFP4-BFP8-P150/resolve/main/launch.py

# Default: 8,192-token context, DFlash2 speculative decoding
python3 launch.py --cache-root /large-disk/qwen27b --device-ownership-confirmed

# 128K context, DFlash2
python3 launch.py --cache-root /large-disk/qwen27b --max-model-len 131072 --device-ownership-confirmed

# Native 262K context: DFlash2 with the 4-bit V cache
python3 launch.py --cache-root /large-disk/qwen27b --max-model-len 262144 --device-ownership-confirmed

# Native 262K context: MTP-3 with the 4-bit V cache
python3 launch.py --cache-root /large-disk/qwen27b --max-model-len 262144 --spec mtp --device-ownership-confirmed

# Verified non-speculative target path; prefix caching off
python3 launch.py --cache-root /large-disk/qwen27b --spec off --device-ownership-confirmed
python3 launch.py --cache-root /large-disk/qwen27b --prefix-caching off --device-ownership-confirmed

# Download and verify only (no Docker, no device) / print the serving command
python3 launch.py --cache-root /large-disk/qwen27b --download-only
python3 launch.py --cache-root /large-disk/qwen27b --print-command
```

Images and videos go into chat messages as in the OpenAI API, any number per request, as data URLs (here) or links the container can fetch:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d @- <<EOF
{"model": "Qwen/Qwen3.8-27B", "max_tokens": 256,
 "messages": [{"role": "user", "content": [
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,$(base64 -w0 photo.png)"}},
   {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,$(base64 -w0 clip.mp4)"}},
   {"type": "text", "text": "Describe the photo, then what happens in the video."}]}]}
EOF
```

`--max-model-len` picks the profile: up to 8,192 tokens the 8K profile, up to 131,072 the 128K profile, above that the 262K profile. `--spec auto` (the default) is DFlash2 in every profile. `--vision off` serves text only; `--max-pixels` caps the resized size of images and videos (default: the checkpoint processor's limits, capped to max-model-len × 1,024 below a 16,384-token context).

The launcher is standard-library Python. Before it serves, it:

1. resolves the requested revision to an immutable commit;
2. downloads the native package anonymously;
3. verifies every byte against `runtime-release.json`, whose image IDs, profiles and checkpoint manifest must match the values pinned in `launch.py`;
4. loads the checksum-pinned `runtime-image.tar.gz` and checks the Docker image ID of the profile's image.

`serve_native.py` then checks the manifest, every checkpoint file and the reload proof recorded on that image (`checkpoint/` for ctx5-s27v3, `proofs-262k/` for ctx6-kv4-s27v3, mounted over the checkpoint's proofs inside the container). The default tensor cache is `CACHE_ROOT/tensor-cache/<manifest sha256[:16]>-<image id[:12]>`, and it carries an identity file that also records the vision tower's checksum: a cache built for another checkpoint, image or vision tower is refused, because the runtime's tensor cache is keyed by tensor shape only.

The OpenAI-compatible API (model `Qwen/Qwen3.8-27B`) binds to `127.0.0.1:8000` and serves one sequence at a time. The first start of each decoding mode converts weights into the tensor cache and takes several minutes. Keep the printed revision and pass it with `--revision` for exact replay.

## Verification

- **Native reload proofs.** Four fresh proofs are bound to the manifest, two per image: [`checkpoint/equivalence.json`](checkpoint/equivalence.json) and [`checkpoint/equivalence-mtp.json`](checkpoint/equivalence-mtp.json) (ctx5-s27v3, MTP off / on), and [`proofs-262k/equivalence.json`](proofs-262k/equivalence.json) and [`proofs-262k/equivalence-mtp.json`](proofs-262k/equivalence-mtp.json) (ctx6-kv4-s27v3). Each compares full-vocabulary logits on 95 teacher-forced positions between the original BF16 checkpoint with the same GPTQ weights substituted (quantized at load) and the native reload, and requires them to be exactly equal. Each proof binds the hashes of its image's runtime sources, including the device sampler and the vision runtime ([`evidence/s27v3/proofs/`](evidence/s27v3/proofs/)).
- **Vision.** See the image and video checks under "What changed". On the s27v2 images, text output is identical to the s27 images ([`evidence/s27v/vision/text-stock/`](evidence/s27v/vision/text-stock/text.json) vs [`evidence/s27v2/mm-8k/`](evidence/s27v2/mm-8k/text.json) at 8K; [`text-stock262/`](evidence/s27v/vision/text-stock262/text.json) vs [`evidence/s27v2/mm-262k/`](evidence/s27v2/mm-262k/text.json) at 262K).
- **Greedy HTTP output equals non-speculative decoding** (s27 runtime), 20 of 20 checks in each profile: 8K DFlash2, 128K DFlash2, 262K DFlash2 with the 4-bit V cache and 262K MTP-3 with the 4-bit V cache ([`evidence/s27/sampling/ex8k/`](evidence/s27/sampling/ex8k/exact.json), [`ex128k/`](evidence/s27/sampling/ex128k/exact.json), [`ex262k/`](evidence/s27/sampling/ex262k/exact.json), [`mtp262k/`](evidence/s27/sampling/mtp262k/exact.json)). The references are non-speculative streams of the same profile. The checks cover 6 chat prompts, completions with the chat prompt's token IDs, request isolation, a greedy request after a sampled one, cancellation mid-decode and mid-prefill, image rejection (the s27 runtime was text-only; the public check below answers an image instead) and health.
- **Sampled output has the model's distribution.** 250 samples per preset (thinking, coding, non-thinking) and prompt, with independent seeds, from DFlash2 speculative sampling and from non-speculative device sampling were compared by two-sample chi-square tests on the first four tokens and the length: smallest p-value 0.056 over 30 tests, above the Bonferroni threshold 0.0017 ([`evidence/s27/sampling/dist-dist8k-vs-ns8k.json`](evidence/s27/sampling/dist-dist8k-vs-ns8k.json)). The device sampler matches a float64 vLLM reference within 3.8e-8 in probability, and the sampled speculative cycles checked (300 per configuration) replayed bit-exactly on the host ([`kernel-tests.log`](evidence/s27/sampling/kernel-tests.log), [`host_chain.log`](evidence/s27/sampling/host_chain.log)).
- **Request parameters** (37 cases, [`param_matrix.json`](evidence/s27/sampling/ex8k/param_matrix.json)): with speculation on, upstream vLLM's speculative-decoding validation rejects `min_p`, `logit_bias` and `min_tokens` with HTTP 400; they work with `--spec off`. Requests that need logprobs, `allowed_token_ids`, `bad_words`, structured outputs or a top-k outside 1–64 are served exactly by vLLM's host sampler, one token per step.
- **Prefix caching is exact.** Model-level resume checks (extend, exact, chain, diverge) passed, and multi-turn texts with caching on equal the caching-off reference in every profile ([`evidence/s27/spec-exact/`](evidence/s27/spec-exact/), [`evidence/s27/prefix-cache/`](evidence/s27/prefix-cache/)); for multimodal conversations, see "What changed" and [`evidence/s27v3/`](evidence/s27v3/).
- **Long context.** Measured on runtime ctx5: the 128K profile retrieved all passkeys (beginning, middle and end) at 65,400 and 130,944 prompt tokens, and the 262K MTP-3 profile at 65,400, 130,944 and 262,016 tokens; a 262,145-token request was rejected with HTTP 400 before generation. With the 4-bit V cache, needles were retrieved at 32K, 128K and 262K (see Quality).
- **The public package was verified end to end.** Revision `3937f7fd` was downloaded anonymously into an empty directory with its own `launch.py --download-only` (996 files, 32.5 GB, every file re-hashed against `SHA256SUMS`, 0 disagreements). With the local image tags removed, the launcher loaded the downloaded archive and got both pinned image IDs. Served from the download with fresh tensor caches, the 8K profile matched the non-speculative reference (20 of 20 checks, including an image request), decoded greedy at 50.3 / 54.7 tok/s and the thinking preset at 48.8 / 48.6 tok/s at 128 / 2,048 tokens, reproduced seeded requests, answered 8 of 8 image questions and passed the multi-image, multi-turn, video and image-plus-video checks. The 262K profile started on the ctx6-kv4-s27v3 image with the 4-bit V cache, DFlash2 and the vision tower, decoded greedy / thinking requests at 2,048 tokens at 54.0 / 40.3 tok/s, answered 8 of 8 image questions and passed the same checks plus a 16,384-token (4096 x 4096) image. In both profiles the six follow-up turns of multimodal conversations (images first or late, a video, a new image in the turn) resumed from the prefix cache at 4,096 tokens, and every answer equals a caching-off server's ([`evidence/public-download-verification.json`](evidence/public-download-verification.json), [`evidence/public-serving-smoke.json`](evidence/public-serving-smoke.json)).

These checks establish native/runtime equivalence and serving behaviour. They do not certify quality against the upstream BF16 model; the quality tables above are measured, not certified. The vision checks are small synthetic sets, not a benchmark.

## Limits

- One sequence at a time, one P150. A 16,384-token image takes about 25 s in the vision tower; video frames are encoded one after another.
- The 262K profile always uses the 4-bit V cache (+0.0071 nats per token on a long chat, see Quality). To serve 262K with MTP-3 and a BFP8 KV cache instead, use the previous release (below).
- Non-speculative decode (`--spec off`, and sampled requests that fall back to the host sampler) is slower at long context than in `gptq-ctx2`, because single-row decode uses the verification numerics: +5.6 ms per token at 130,560 tokens and +10.2 ms per token at 261,632 tokens (runtime ctx5).
- With speculation on, `min_p`, `logit_bias` and `min_tokens` are rejected with HTTP 400 (upstream vLLM); serve with `--spec off` to use them.
- Prefix caching resumes only at 2,048-token chunk boundaries.
- Long prompts have a long time to first token on a cold prompt (about 3 min at 128K, 8 min at 258K).

## Previous releases

- **`s27v2-20261022`** (same checkpoint tensors and text runtime; any number of images and videos, but prompts with images or videos were not resumed by prefix caching; images `lottolabs/qwen27b-tt-p150:ctx5-s27v2` `sha256:6480be4297d2…` and `:ctx6-kv4-s27v2` `sha256:bbfdf84ce63b…`) is tag [`s27v2-20261022`](https://huggingface.co/Lottolabs/Qwen3.8-27B-TT-Mixed-BFP4-BFP8-P150/tree/s27v2-20261022), commit `b31a43e44dd4c503a3628dfc6a71f2621dd0aac7`. Its public-download verification is under [`evidence/previous-s27v2/`](evidence/previous-s27v2/). Serve it with that revision's launcher, passing `--revision b31a43e44dd4c503a3628dfc6a71f2621dd0aac7`.
- **`s27v-20261021`** (same checkpoint tensors and text runtime; one image of at most 1,024 tokens per request, no video; images `lottolabs/qwen27b-tt-p150:ctx5-s27v` `sha256:1f4788591d2c…` and `:ctx6-kv4-s27v` `sha256:547810997a35…`) is tag [`s27v-20261021`](https://huggingface.co/Lottolabs/Qwen3.8-27B-TT-Mixed-BFP4-BFP8-P150/tree/s27v-20261021), commit `030816b05ac65bea02b3fb9e4f074c99d0995bdc`. Its public-download verification is under [`evidence/previous-s27v/`](evidence/previous-s27v/). Serve it with that revision's launcher, passing `--revision 030816b05ac65bea02b3fb9e4f074c99d0995bdc`.
- **`s27-20261005`** (same checkpoint tensors and text runtime, without image input; images `lottolabs/qwen27b-tt-p150:ctx5-s27` `sha256:9c8aaa786f4a…` and `:ctx6-kv4-s27` `sha256:da748d2298e8…`) is tag [`s27-20261005`](https://huggingface.co/Lottolabs/Qwen3.8-27B-TT-Mixed-BFP4-BFP8-P150/tree/s27-20261005), commit `98478ceea1474d8176de6cf1bc242a66d3609700`. Its public-download verification is under [`evidence/previous-s27/`](evidence/previous-s27/). Serve it with that revision's launcher: `curl --fail --location --output launch-s27.py https://huggingface.co/Lottolabs/Qwen3.8-27B-TT-Mixed-BFP4-BFP8-P150/resolve/98478ceea1474d8176de6cf1bc242a66d3609700/launch.py`, then `python3 launch-s27.py --revision 98478ceea1474d8176de6cf1bc242a66d3609700 ...`.
- **`gptq-ctx2-20261002`** (same checkpoint tensors; runtime `lottolabs/qwen27b-tt-p150:ctx2`, ID `sha256:53a2ae1582b3…`; DFlash2 up to 128K, MTP-3 with BFP8 KV at 262K, host sampler, no prefix caching) is tag [`gptq-ctx2`](https://huggingface.co/Lottolabs/Qwen3.8-27B-TT-Mixed-BFP4-BFP8-P150/tree/gptq-ctx2), commit `e7439ca85e11f5fc5f8dfea3046cde21cefc1c93`. Its evidence is under [`evidence/previous-gptq-ctx2/`](evidence/previous-gptq-ctx2/). Each launcher only accepts its own pinned runtime, so fetch that revision's launcher and pass the revision:

  ```bash
  curl --fail --location --output launch-ctx2.py \
    https://huggingface.co/Lottolabs/Qwen3.8-27B-TT-Mixed-BFP4-BFP8-P150/resolve/e7439ca85e11f5fc5f8dfea3046cde21cefc1c93/launch.py
  python3 launch-ctx2.py --revision e7439ca85e11f5fc5f8dfea3046cde21cefc1c93 --cache-root /large-disk/qwen27b \
    --max-model-len 262144 --device-ownership-confirmed   # 262K MTP-3, BFP8 KV
  ```
- **`dual-noc-mtp1-20260920`** (round-to-nearest BFP4, MTP-1, 8,192-token context) stays available at revision `50becf020f24ac03d6b2b69434a3d4e2655e0865`; its evidence is under [`evidence/previous-dual-noc-mtp1-20260920/`](evidence/previous-dual-noc-mtp1-20260920/). The LocalMaxxing submission (23.8 tok/s, MTP-1) was measured on that release.

## Integrity and provenance

- [`runtime-release.json`](runtime-release.json): the downloadable serving inventory, the runtime-archive checksum, both image IDs, the serving profiles and the upstream drafter identity.
- [`release-manifest.json`](release-manifest.json) and [`SHA256SUMS`](SHA256SUMS): the complete repository inventory and checksums.
- [`checkpoint/native_manifest.json`](checkpoint/native_manifest.json): the per-tensor format, precision, hashes and the `ttquant` section.
- [`checkpoint/provenance/`](checkpoint/provenance/): the builder, the evaluator and the BFP unpacker used for the GPTQ weights.
- [`precision-summary.json`](precision-summary.json): the precision map (128 BFP4 and 113 BFP8 layer-family assignments, LM head BFP8).
- [`runtime/`](runtime/): for each image, every runtime source file listed in its `qwen-runtime-support.json` (copied from the image and checked against those hashes, which the proofs bind), the Dockerfile of the s27v2 layer and the list of changed files; the sampling, vision and multimodal patches; and [`runtime/source.json`](runtime/source.json) with the build chain (ctx2 → ctx4 prefix caching → ctx5 exact speculation → ctx6-kv4 4-bit V; s27 sampling layers on ctx5 and ctx6-kv4; s27v2 vision layers on the s27 images). It is provenance: rebuilding needs the local parent images. Serving uses the checksum-pinned archive.

## License and attribution

- The model is derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0; see [`LICENSE`](LICENSE).
- The DFlash2 drafter in `dflash-draft/` is redistributed byte-identical from [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, Apache-2.0. Its model card is kept as `dflash-draft/README.md`.
- The vision tower in `vision/` holds the `model.visual.*` tensors of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at the same revision, unchanged values, Apache-2.0.
- Runtime sources keep their upstream notices and bundled licenses.

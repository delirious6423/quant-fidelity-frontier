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

A hardware-specific checkpoint (text and image input) and custom runtime derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, for one Tenstorrent Blackhole P150.

This is a community experimental release by **Lottolabs**, not an official Qwen or Tenstorrent release. It is not GGUF, AWQ, an AutoGPTQ/GPTQModel int4 checkpoint, bitsandbytes, NVFP4, or an ordinary Transformers checkpoint. `transformers.from_pretrained()` and hosted Inference Providers are not supported launch paths.

## What changed in this release (`s27v-20261021`)

The checkpoint tensors are unchanged (manifest sha256 `47b781ab…`). The reload proofs were re-recorded on the two new runtime images. **Every profile now accepts one image per request.**

- **Image input** through the OpenAI chat API (`image_url` parts: data URLs or links), in the 8K, 128K and native 262K profiles. Images are resized to at most 1,048,576 pixels (`--max-image-pixels`), which is at most 1,024 image tokens. Video input is not enabled. `--vision off` serves text only.
- **The upstream vision tower** is shipped as [`vision/model.safetensors`](vision/model.safetensors): the base revision's 333 `model.visual.*` tensors (460.7M parameters, BF16, 0.92 GB), byte-identical values. It runs on the same P150 next to the text model, with BFP8 weights (0.46 GB, which also fits next to the 262K KV pool) and HiFi4 / FP32 accumulation. Its matmuls, attention and norms use small blocks so their L1 buffers fit beside the resident GDN state.
- **Fixed: the tower attended to padding.** Image patches are padded to a fixed length, and the upstream TT tower did not mask the padding out of its attention: its output matched an FP32 reference of the tower with a Pearson correlation of only 0.936. With the mask, it is 0.989 with BFP8 weights and 0.996 with BF16 weights and HiFi4.
- **Speculative greedy output on image prompts equals non-speculative output.** Image tokens shift the M-RoPE positions of the text after them; the captured verification step now applies the request's offset, as decode already did.
- **Text requests are unchanged.** On both new images, 5 greedy chat prompts produce the same text at the same speed as the s27 images (8K: 34.3–82.3 tok/s; 262K: 33.5–80.4 tok/s).
- **Two new runtime images**, each the s27 image plus the vision layer. `runtime-image.tar.gz` holds both, with their shared layers stored once:

  | Image | ID | Used by |
  |---|---|---|
  | `lottolabs/qwen27b-tt-p150:ctx5-s27v` | `sha256:1f4788591d2c3a9c99704f62bcc7274609fe70165e31b26fac92c878d90649ae` | 8K and 128K profiles (BF16 KV up to 64K, BFP8 above) |
  | `lottolabs/qwen27b-tt-p150:ctx6-kv4-s27v` | `sha256:547810997a35693cd239866d11c48504d5005a576024636e09385a63a511bb95` | 262K profile (BFP8 K, 4-bit V) |

**Vision checks** (single P150, greedy, thinking off; [`evidence/s27v/vision/`](evidence/s27v/vision/)):

| Check | Result |
|---|---|
| 24 synthetic images (counting, reading words and numbers, colors), 448–896 px, 8K profile | 23/24 with BFP8 and with BF16 tower weights, the same answers; the miss answered "ROCKET 91" for the word "ROCKET" |
| The same 24 images on an RTX 3090 (vLLM, W4A16 target, same base model) | 24/24; 13 answers are the same strings, 10 differ only in capitalization or a leading space, and on "ROCKET" the 3090 answered "ROCKET" |
| DFlash2 on vs off, the 24 images and the long descriptions | token-identical (8K); the 8 images and the long descriptions token-identical (262K) |
| Two different images with the same prompt, then the first again, then a follow-up turn | the repeat is identical and the follow-up uses the image; both images carry text the model misreads ("APPLE 12" as "April", "TIGER 55" as "Tiger eye"), as it also does on the 3090 |
| 128K and 262K profiles | 8/8 each |
| 262K: one image + a 231,203-token prompt with a fact in its middle | both the fact and the image (color, text) answered correctly; 395 s end to end |
| Image description decode, DFlash2 | about 44 tok/s (2.9 accepted tokens per DFlash2 step on image prompts vs 3.9 on text) |

### Changes in `s27-20261005` (still included)

- **Exact prefix caching** (on by default). A prompt that extends an earlier prompt resumes at the last 2,048-token prefill chunk boundary the earlier prompt crossed, and then runs the same chunks a full prefill would run. Requests with an image are not resumed.
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

An image request (one image per request; text-only requests are unchanged). Images can be data URLs, as here, or links the container can fetch:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d @- <<EOF
{"model": "Qwen/Qwen3.8-27B", "max_tokens": 256,
 "messages": [{"role": "user", "content": [
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,$(base64 -w0 photo.png)"}},
   {"type": "text", "text": "Describe this image."}]}]}
EOF
```

`--max-model-len` picks the profile: up to 8,192 tokens the 8K profile, up to 131,072 the 128K profile, above that the 262K profile. `--spec auto` (the default) is DFlash2 in every profile. `--vision off` serves text only; `--max-image-pixels` (default 1,048,576) caps the resized image size.

The launcher is standard-library Python. Before it serves, it:

1. resolves the requested revision to an immutable commit;
2. downloads the native package anonymously;
3. verifies every byte against `runtime-release.json`, whose image IDs, profiles and checkpoint manifest must match the values pinned in `launch.py`;
4. loads the checksum-pinned `runtime-image.tar.gz` and checks the Docker image ID of the profile's image.

`serve_native.py` then checks the manifest, every checkpoint file and the reload proof recorded on that image (`checkpoint/` for ctx5-s27v, `proofs-262k/` for ctx6-kv4-s27v, mounted over the checkpoint's proofs inside the container). The default tensor cache is `CACHE_ROOT/tensor-cache/<manifest sha256[:16]>-<image id[:12]>`, and it carries an identity file that also records the vision tower's checksum: a cache built for another checkpoint, image or vision tower is refused, because the runtime's tensor cache is keyed by tensor shape only.

The OpenAI-compatible API (model `Qwen/Qwen3.8-27B`) binds to `127.0.0.1:8000` and serves one sequence at a time. The first start of each decoding mode converts weights into the tensor cache and takes several minutes. Keep the printed revision and pass it with `--revision` for exact replay.

## Verification

- **Native reload proofs.** Four fresh proofs are bound to the manifest, two per image: [`checkpoint/equivalence.json`](checkpoint/equivalence.json) and [`checkpoint/equivalence-mtp.json`](checkpoint/equivalence-mtp.json) (ctx5-s27v, MTP off / on), and [`proofs-262k/equivalence.json`](proofs-262k/equivalence.json) and [`proofs-262k/equivalence-mtp.json`](proofs-262k/equivalence-mtp.json) (ctx6-kv4-s27v). Each compares full-vocabulary logits on 95 teacher-forced positions between the original BF16 checkpoint with the same GPTQ weights substituted (quantized at load) and the native reload, and requires them to be exactly equal. Each proof binds the hashes of its image's runtime sources, including the device sampler and the vision runtime ([`evidence/s27v/proofs/`](evidence/s27v/proofs/)).
- **Vision.** See the vision checks under "What changed". Text output on the s27v images is identical to the s27 images: [`text-stock/`](evidence/s27v/vision/text-stock/text.json) vs [`vqa2-dflash/`](evidence/s27v/vision/vqa2-dflash/text.json) (8K), [`text-stock262/`](evidence/s27v/vision/text-stock262/text.json) vs [`p262k/`](evidence/s27v/vision/p262k/text.json) and [`release262/`](evidence/s27v/vision/release262/text.json) (262K).
- **Greedy HTTP output equals non-speculative decoding** (s27 runtime), 20 of 20 checks in each profile: 8K DFlash2, 128K DFlash2, 262K DFlash2 with the 4-bit V cache and 262K MTP-3 with the 4-bit V cache ([`evidence/s27/sampling/ex8k/`](evidence/s27/sampling/ex8k/exact.json), [`ex128k/`](evidence/s27/sampling/ex128k/exact.json), [`ex262k/`](evidence/s27/sampling/ex262k/exact.json), [`mtp262k/`](evidence/s27/sampling/mtp262k/exact.json)). The references are non-speculative streams of the same profile. The checks cover 6 chat prompts, completions with the chat prompt's token IDs, request isolation, a greedy request after a sampled one, cancellation mid-decode and mid-prefill, image rejection (the s27 runtime was text-only; the s27v public check below answers an image instead) and health.
- **Sampled output has the model's distribution.** 250 samples per preset (thinking, coding, non-thinking) and prompt, with independent seeds, from DFlash2 speculative sampling and from non-speculative device sampling were compared by two-sample chi-square tests on the first four tokens and the length: smallest p-value 0.056 over 30 tests, above the Bonferroni threshold 0.0017 ([`evidence/s27/sampling/dist-dist8k-vs-ns8k.json`](evidence/s27/sampling/dist-dist8k-vs-ns8k.json)). The device sampler matches a float64 vLLM reference within 3.8e-8 in probability, and the sampled speculative cycles checked (300 per configuration) replayed bit-exactly on the host ([`kernel-tests.log`](evidence/s27/sampling/kernel-tests.log), [`host_chain.log`](evidence/s27/sampling/host_chain.log)).
- **Request parameters** (37 cases, [`param_matrix.json`](evidence/s27/sampling/ex8k/param_matrix.json)): with speculation on, upstream vLLM's speculative-decoding validation rejects `min_p`, `logit_bias` and `min_tokens` with HTTP 400; they work with `--spec off`. Requests that need logprobs, `allowed_token_ids`, `bad_words`, structured outputs or a top-k outside 1–64 are served exactly by vLLM's host sampler, one token per step.
- **Prefix caching is exact.** Model-level resume checks (extend, exact, chain, diverge) passed, and multi-turn texts with caching on equal the caching-off reference in every profile ([`evidence/s27/spec-exact/`](evidence/s27/spec-exact/), [`evidence/s27/prefix-cache/`](evidence/s27/prefix-cache/)).
- **Long context.** Measured on runtime ctx5: the 128K profile retrieved all passkeys (beginning, middle and end) at 65,400 and 130,944 prompt tokens, and the 262K MTP-3 profile at 65,400, 130,944 and 262,016 tokens; a 262,145-token request was rejected with HTTP 400 before generation. With the 4-bit V cache, needles were retrieved at 32K, 128K and 262K (see Quality).
- **The public package was verified end to end.** Revision `af6efd25` was downloaded anonymously into an empty directory with its own `launch.py --download-only` (996 files, 32.5 GB, every file re-hashed against `SHA256SUMS`, 0 disagreements). With the local image tags removed, the launcher loaded the downloaded archive and got both pinned image IDs. Served from the download with fresh tensor caches, the 8K profile matched the non-speculative reference (20 of 20 checks, including an image request), decoded greedy at 50.3 / 54.7 tok/s and the thinking preset at 48.8 / 48.6 tok/s at 128 / 2,048 tokens, reproduced seeded requests and answered 8 of 8 image questions. The 262K profile started on the ctx6-kv4-s27v image with the 4-bit V cache, DFlash2 and the vision tower, decoded greedy / thinking requests at 2,048 tokens at 54.0 / 40.3 tok/s and answered 8 of 8 image questions ([`evidence/public-download-verification.json`](evidence/public-download-verification.json), [`evidence/public-serving-smoke.json`](evidence/public-serving-smoke.json)).

These checks establish native/runtime equivalence and serving behaviour. They do not certify quality against the upstream BF16 model; the quality tables above are measured, not certified. The vision checks are small synthetic sets, not a benchmark.

## Limits

- One sequence at a time, one P150. One image per request (up to 1,024 image tokens); no video. Image prompts are not resumed by prefix caching.
- The 262K profile always uses the 4-bit V cache (+0.0071 nats per token on a long chat, see Quality). To serve 262K with MTP-3 and a BFP8 KV cache instead, use the previous release (below).
- Non-speculative decode (`--spec off`, and sampled requests that fall back to the host sampler) is slower at long context than in `gptq-ctx2`, because single-row decode uses the verification numerics: +5.6 ms per token at 130,560 tokens and +10.2 ms per token at 261,632 tokens (runtime ctx5).
- With speculation on, `min_p`, `logit_bias` and `min_tokens` are rejected with HTTP 400 (upstream vLLM); serve with `--spec off` to use them.
- Prefix caching resumes only at 2,048-token chunk boundaries.
- Long prompts have a long time to first token on a cold prompt (about 3 min at 128K, 8 min at 258K).

## Previous releases

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
- [`runtime/`](runtime/): for each image, every runtime source file listed in its `qwen-runtime-support.json` (copied from the image and checked against those hashes, which the proofs bind), the Dockerfile of the s27v layer and the list of changed files; the sampling and vision patches; and [`runtime/source.json`](runtime/source.json) with the build chain (ctx2 → ctx4 prefix caching → ctx5 exact speculation → ctx6-kv4 4-bit V; s27 sampling layers on ctx5 and ctx6-kv4; s27v vision layers on the s27 images). It is provenance: rebuilding needs the local parent images. Serving uses the checksum-pinned archive.

## License and attribution

- The model is derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0; see [`LICENSE`](LICENSE).
- The DFlash2 drafter in `dflash-draft/` is redistributed byte-identical from [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, Apache-2.0. Its model card is kept as `dflash-draft/README.md`.
- The vision tower in `vision/` holds the `model.visual.*` tensors of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at the same revision, unchanged values, Apache-2.0.
- Runtime sources keep their upstream notices and bundled licenses.

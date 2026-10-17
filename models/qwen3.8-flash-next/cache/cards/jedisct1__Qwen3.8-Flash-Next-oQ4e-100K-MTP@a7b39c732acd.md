---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
license: other
license_name: qwen-community-1.0
license_link: LICENSE
tags:
  - omlx
  - mlx
  - qwen4
  - mtp
  - vision
  - tool-calling
---

# Qwen3.8 Flash Next oQ4e for oMLX

This is an oMLX-native quantization of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
It was built directly from pinned official revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`.
It preserves the Qwen4 vision tower, the checkpoint's MTP head, and the native
262,144-token architecture setting.

The artifact is intended for oMLX and its OpenAI-compatible API.
It can be used by coding agents and other clients that support that API.

## Quantization

The default layout is oQ4e affine quantization with group size 64.
The large PLE table uses four bits with group size 32 so oMLX can use its
SSD-backed Qwen4 PLE path.

Strict oQe calibration covered 937 modules over 1,024 sequences of 512 tokens.
Seven routed experts in base layers 0 and 1 were not selected by that fixed
calibration corpus.
All three routed projections in both affected blocks are therefore Q8 and do
not use invented importance values.
On the exact affected official BF16 rows, Q8 reduced normalized reconstruction
error by 164.46 times compared with Q4.

The base language-model head, both base hyper-connection mixer projections,
and the Qwen4 MTP embedding and hidden fusion projections remain BF16.
These additions favor output quality and MTP draft acceptance over the last
few GiB of compression.

The weights occupy 108,816,636,376 bytes across 22 safetensors shards.

## oMLX setup

Add the downloaded model folder to oMLX.
On the tested 128 GiB Mac, use a 100,000-token context limit.
Keep Qwen4 PLE SSD offload enabled for the full context window.
MTP is included, but keep it disabled for coding agents.

Use these per-model settings as the starting point:

```json
{
  "max_context_window": 100000,
  "qwen4_ple_ssd_offload": true,
  "mtp_enabled": false
}
```

For coding agents, set `reasoning_effort` to `medium` in the client profile and
provider request body so a global setting cannot override the model profile:

```toml
reasoning_effort = "medium"
extra_body = { reasoning_effort = "medium" }
```

The model's prompt plus requested output must remain within 100,000 tokens.

## Prompt caching

If your coding agent or client reuses long prefixes, enable prompt caching for
substantially faster repeated requests.
Enable oMLX's paged SSD prefix cache, use a fast local SSD, and leave the
in-memory hot cache disabled on a 128 GiB machine.
The recommended starting point is a 100 GB SSD limit and a zero-byte hot cache.

The equivalent server options are:

```console
omlx serve \
  --paged-ssd-cache-dir <fast-ssd-path>/omlx-cache \
  --paged-ssd-cache-max-size 100GB \
  --hot-cache-max-size 0
```

Keep `max_context_window` at 100,000 when caching is enabled.
Caching reduces repeated-prefix prefill time, but it does not expand the safe
token budget.
Keep the balanced memory guard enabled because a cold or changed prefix must
first populate the cache and can use more memory than a cache hit.

In the full-window retrieval test, the first 99,744-token request populated the
SSD cache in 418.18 seconds and peaked at 94.71 GiB of physical memory.
After restarting oMLX, the same request reused 98,304 tokens, completed in
11.53 seconds, and peaked at 87.84 GiB.
Both requests recovered exact values near the start, middle, and end of the
prompt and caused no swap growth.

## Why 100,000 tokens

The checkpoint retains its native 262,144-token architecture setting, but that
number is not a safe operating limit for this 128 GiB configuration.
The uncached 131,072-token diagnostic reached the full prefill, but physical
memory peaked at 97.03 GiB and exceeded the 96 GiB reliability ceiling.
With paged SSD caching enabled, the 131,072-token request was rejected by the
balanced prefill guard and reached a 99.96 GiB physical peak.
Neither 128K result caused swap growth, but neither had enough reserve for a
reliable end-user setting.

By comparison, an exact uncached request with 99,999 prompt tokens and one
generated token passed at a 90.89 GiB physical peak with no swap growth.
Both tested 100,001-token budgets were rejected as expected.
An independent 99,744-token retrieval prompt plus a 256-token output reserve
recovered exact markers at token positions 569, 51,721, and 98,256 and peaked
at 90.84 GiB.

For prompt-cache users, 100,000 remains the safe total context limit.
A growing 100,000-token request reused 63,488 tokens, and its exact repeat
reused 98,304 tokens and completed in 4.13 seconds.
The cached 128K probe still failed the balanced guard.
The 100,000-token setting therefore gives useful long context while leaving
measured headroom for model state, attention and KV memory, oMLX, the client,
and cache activity.

## Validation

The artifact has passed structural verification of every shard and tensor,
the Q8 and BF16 safety policy, complete vision and MTP inventories, strict
imatrix lookup coverage, and the native context configuration.
`SHA256SUMS` binds all 22 weight shards.

Fresh oMLX 0.6.3 tests loaded the artifact strictly, passed deterministic text
generation, and identified Half Dome from a real JPEG.
Tool calling was evaluated with [Swival](https://swival.dev/) through oMLX's
OpenAI-compatible API, but the quant is intended to work with other compatible
clients and agent harnesses.
With MTP disabled, all 50 positive cases passed on the first attempt and all 10
adversarial guards were protected.
With MTP enabled, all 50 positive cases still passed, but only 9 of 10 guards
were protected because one speculative run changed an exact tool argument and
made an extra call.
MTP accepted 10,631 of 12,152 drafted tokens in that run, so the weights are
usable, but `mtp_enabled: false` remains the reliable coding-agent default.

## License

The pinned source uses the Qwen Community License 1.0.
It is preserved in `LICENSE` and includes additional commercial terms.
Review the included license before using or distributing this artifact.

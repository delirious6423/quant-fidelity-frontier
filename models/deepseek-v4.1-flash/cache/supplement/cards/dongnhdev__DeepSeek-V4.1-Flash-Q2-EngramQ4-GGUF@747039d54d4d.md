---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- gguf
- deepseek
- metal
- engram
---

# DeepSeek-V4.1-Flash Q2, Engram tables at 4 bits

This is [antirez/deepseek-v4.1-flash-gguf](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf) `DeepSeek-V4.1-Flash-Q2.gguf`
with only its two Engram tables re-encoded:
- **Size:** 249.04 GiB, down from 340.60 GiB.
- **Quality:** no measurable loss.
- **Other tensors:** every byte is identical to the source.

**Compatibility.** Only [dongnh311/ds4-metal-qwen](https://github.com/dongnh311/ds4-metal-qwen) reads this file.
- **Branch.** It needs the `lloyd4_e5m3_32_r136` Engram encoding, in the fork's `develop` branch since 2026-10-03.
- **Upstream.** ds4 refuses the file.

**Six parts.** Hugging Face caps a file at 50 GB, so the GGUF is stored as six consecutive byte ranges. Join them in
order to get the file.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf` (joined) | 267,406,761,552 | `311f35981bf14ef8e49968ef942e13263bf9011ffdc28c7b6d00a1de45d2f719` |
| `DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part1` (bytes 0 to 45,097,156,608) | 45,097,156,608 | `e3a25ed2c0498eea95f4a7b5605f41eeeac9855486414f647b7ddc002e8fa635` |
| `DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part2` (bytes 45,097,156,608 to 90,194,313,216) | 45,097,156,608 | `19673581014fb8ee8f0f9f466107a0168fd97859346474c00b05f7e45e5d8dcc` |
| `DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part3` (bytes 90,194,313,216 to 135,291,469,824) | 45,097,156,608 | `1512b3d02258b979c8a5a5fec8fca473ae37c134c832daf71ef4546a1b6c5929` |
| `DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part4` (bytes 135,291,469,824 to 180,388,626,432) | 45,097,156,608 | `3125d0fae3a2a79ee79c0bdda59e1fb9bdcb0067f49d201549a1a9affbe7d227` |
| `DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part5` (bytes 180,388,626,432 to 225,485,783,040) | 45,097,156,608 | `936b8b67e7be53cb09e60d6ca969ef53c494d83b03cb7d84cd4325a4886aeec8` |
| `DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part6` (bytes 225,485,783,040 to 267,406,761,552) | 41,920,978,512 | `d485a71b28ea3e9795f1849c06fb8101f60da26de50b58f222678e4b045d3c73` |
| source `DeepSeek-V4.1-Flash-Q2.gguf` | 365,713,686,528 | `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42` |

## What changed

The Engram tables are `blk.1.engram_embd.weight` and `blk.14.engram_embd.weight`: 768,022,850 rows of 256 values, read from disk
per n-gram.

**Source rows** (`e4m3_e8m0_32_row264`) take 264 bytes:
- 256 FP8 E4M3 values;
- 8 E8M0 scales, one per 32 values.

**New rows** (`lloyd4_e5m3_32_r136`) take 136 bytes:
- **Codes.** 256 4-bit codes. Each is a sign plus an index into the 8-level Lloyd-Max codebook
  `{0, 0.095, 0.1901, 0.2944, 0.4126, 0.5586, 0.7484, 1}`, fitted to |v|/amax.
- **Scales.** 8 scale bytes, one per 32 values: an unsigned float with 4 significant bits, `(8+M)/16 · 2^(E−16)`.

**Scale choice.** The encoder takes the better of two scales by squared error:
- the group's amax rounded up to 4 significant bits;
- one step below that.

Each value takes its nearest level (ties go to the lower level) and keeps its source sign. Decoding then applies the same BF16
rounding as before.

Produced in place with `gguf-tools/deepseek41_engram_q4`:
```
deepseek41_engram_q4 --check FILE
deepseek41_engram_q4 --convert FILE
deepseek41_engram_q4 --verify FILE
```
`--convert` is resumable after a crash.

## Quality

`score_official` measures NLL against official-API continuations. Scores are relative to the FP8 source file.

| Set | Change | Notes |
| --- | ---: | --- |
| general (100 cases, 2994 tokens, ctx 4096) | **−0.05 % NLL** | first-token matches 78 → 81; greedy LCP 9.50 → 9.76 |
| long (9 cases at 8K–32K, 576 tokens) | +0.94 % NLL | |

**The long set's +0.94 % is noise.** On those 9 cases a 6-bit control shifts NLL by +1.04 % and a 5-bit one by +1.33 %. Every
perturbation lands around +1 %, so 4 bits is no worse than 6 bits there.

**This file scores exactly like the simulation.** Its general-set score file is byte-identical to the simulation that produced
the numbers above.

## Speed (M5 Pro 64 GB, SSD streaming, auto cache, ctx 8192)

- **Steady decode:** 8.4–9.2 t/s, against 9.3 t/s with the source file.
- **Prefill:** 188–230 t/s across 6 runs on the night of 2026-10-02/03. The source file gave 239–257 t/s earlier the same day.
  - **The format is not the cause.** A prefill profile puts the Engram reads at 0.25 s of 36.5 s, at most 0.7 %.
  - **What the gap must be.** A CPU benchmark reads Engram rows equally fast in both encodings. That leaves the gap in GPU and
    expert streaming time, which the conversion does not touch.
  - **Not yet re-checked.** Nobody has re-run both files under identical conditions.

## Download

```
./download_model.sh ds41f-q2-eq4
```
Run it from the ds4-metal-qwen repository.
- **What it does.** It downloads the six parts and checks each SHA-256. It then joins them into the GGUF and checks that
  file's SHA-256.
- **Resuming.** Run the same command again to resume.
- **Disk.** Each part is deleted once it is joined, so it needs about 43 GiB free on top of the 249 GiB file.

**By hand.** This needs twice the disk space:
```
hf download dongnhdev/DeepSeek-V4.1-Flash-Q2-EngramQ4-GGUF --local-dir .
cat DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part1 DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part2 \
    DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part3 DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part4 \
    DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part5 DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf.part6 \
    > DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf
shasum -a 256 DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf
```

License: MIT, as the source (DeepSeek).

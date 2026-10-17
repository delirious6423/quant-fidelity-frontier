---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- sushi
- exl3
- exllamav3
- apple-silicon
- moe
---

# Qwen3.8-Flash-Next-EXL3-Sushi

These are two quants of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for the [sushi](https://github.com/beamivalice/sushi) engine on Apple Silicon. I made the routed experts with the open-source ExLlamaV3 converter. Everything else (the 8-bit dense layers, embeddings, vision tower, MTP head and tokenizer) is copied unchanged from [beamster/Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw).

The 4-bit quant is on `main` and on the `4bpw-sh` branch. The 5-bit one is on the `5bpw-sh` branch.

| branch | experts | weights in GPU memory | suits | KLD vs Sushi-4bpw |
|---|---|---:|---|---|
| [`4bpw-sh`](https://huggingface.co/chatnerdie/Qwen3.8-Flash-Next-EXL3-Sushi/tree/4bpw-sh) | EXL3 4.0 bpw | 63.7 GiB | 96 GB Macs and up | 0.94 (0.88–1.00) |
| [`5bpw-sh`](https://huggingface.co/chatnerdie/Qwen3.8-Flash-Next-EXL3-Sushi/tree/5bpw-sh) | EXL3 5.0 bpw (MTP layer 4.0) | 77.8 GiB | 128 GB Macs | 0.79 (0.72–0.86) |

These aren't meant to replace Sushi-4bpw. The sushi author's expert converter isn't public, so I wanted an alternative made entirely with open tools that anyone can rebuild, and I wanted a 5-bit version for a 128 GB Mac. In my tests the 4-bit quant is non-inferior to Sushi-4bpw (details under Quality). The 5-bit one is closer to the original model, at the cost of 14 GiB more memory.

The n-gram table isn't included. It's the same 95.4 GiB file that comes with Sushi-4bpw, so download it from there (see below) rather than storing a second copy here.

## Running it

```bash
# sushi release binary (I tested with 1.2.3 and 1.2.4)
curl -L https://github.com/beamivalice/sushi/releases/latest/download/sushi-bin-macos-arm64.tar.gz | tar xz

# the quant (4bpw-sh shown; use --revision 5bpw-sh for the 5-bit one)
hf download chatnerdie/Qwen3.8-Flash-Next-EXL3-Sushi --revision 4bpw-sh \
  --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-EXL3-Sushi-4bpw-sh

# the n-gram table from Sushi-4bpw (95.4 GiB, bf16)
hf download beamster/Qwen3.8-Flash-Next-Sushi-4bpw ngram_table.bin \
  --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-EXL3-Sushi-4bpw-sh

# raise the GPU memory limit (88000 on a 96 GB Mac, 120000 on a 128 GB Mac; resets on reboot)
sudo sysctl iogpu.wired_limit_mb=120000

./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-EXL3-Sushi-4bpw-sh --mtp-head-kv-quant
```

If you already have Sushi-4bpw, copy its `ngram_table.bin` into the folder instead (on APFS, `cp -c` makes a clone that takes no extra space). You can also use the smaller 4-bit table from Sushi-3bpw (29.8 GiB); the "Choosing the n-gram table" section of the [Sushi-4bpw card](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) explains how, and it works the same here. My quality numbers below use the bf16 table.

Memory: `4bpw-sh` needs the same as Sushi-4bpw, so the memory tables on that card apply. `5bpw-sh` needs 14.1 GiB more at any context length. On a 128 GB Mac that means about 768K context rather than 1M (I run it at `--ctx-size 786432`), and only about 128K on a 96 GB Mac. I worked that 96 GB figure out from Sushi-4bpw's table and haven't tried it on a 96 GB machine.

## Quality

I compared each quant against the BF16 checkpoint using `sushi kld compare`. The test set is 24 prompts (8 coding, 8 prose, 8 JSON extraction and tool calls). The BF16 model wrote 128 tokens for each with greedy decoding, and I scored each quant's KL divergence over those replies up to the first end-of-turn (2,516 positions). Settings: sushi 1.2.3, f16 KV cache, bf16 n-gram table. Lower KLD is better. The ratios are paired by prompt, with 95% bootstrap intervals.

| quant | weights (GiB) | KLD, as shipped (YaRN ×4) | top-1 | KLD, native RoPE | top-1 |
|---|---:|---:|---:|---:|---:|
| beamster Sushi-4bpw | 63.7 | 0.006137 | 98.2% | 0.003740 | 98.4% |
| **`4bpw-sh`** (this repo) | 63.7 | 0.005780 | 98.5% | 0.003525 | 98.4% |
| **`5bpw-sh`** (this repo) | 77.8 | 0.004843 | 98.4% | 0.002124 | 99.0% |
| [SiliconSpecies 5bpw-hq](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-EXL3/tree/5bpw-hq) (rev ccd64f0) | 78.3 | 0.004729 | 98.3% | 0.002070 | 98.7% |

Paired against Sushi-4bpw (KLD ratio with 95% bootstrap interval; below 1 means closer to BF16):

| | as shipped (YaRN ×4) | native RoPE |
|---|---|---|
| `4bpw-sh` | 0.94 (0.88–1.00), lower on 18 of 24 prompts | 0.94 (0.72–1.15), lower on 13 of 24 prompts |
| `5bpw-sh` | 0.79 (0.72–0.86), lower on 21 of 24 prompts | 0.57 (0.46–0.68), lower on 20 of 24 prompts |

For `4bpw-sh` I read this as non-inferiority, not an improvement. With the settings as shipped, the upper bound of the interval is 1.00, so the data exclude any meaningful loss against Sushi-4bpw. With native RoPE the interval is wider, and its upper bound of 1.15 rules out anything more than about 15% worse. I didn't set a margin in advance, so judge the bounds against whatever margin you find reasonable.

`5bpw-sh` is clearly closer to BF16 than either 4-bit quant. Against `4bpw-sh` its ratio is 0.84 (0.78–0.90) as shipped and 0.60 (0.51–0.69) with native RoPE. Against SiliconSpecies' 5bpw-hq it is 1.02 (0.92–1.11) and 1.03 (0.90–1.16), so the two are level. Keeping the hyper-connection mixes in bf16, as 5bpw-hq does, made no measurable difference in my tests.

Don't compare these numbers with the KLD on the Sushi-4bpw card (0.0592). That card used different prompts, longer replies, an 8-bit KV cache and an older sushi build, so only the rows within one table are comparable.

### YaRN and native RoPE

Like Sushi-4bpw, these quants ship with YaRN ×4 in `config.json`, which stretches the native 262,144-token context to 1M. The sushi engine applies YaRN at every length, short prompts included, but the BF16 checkpoint uses plain RoPE. So part of every quant's KLD in the first column comes from the RoPE setting, not from quantization: about 40% for the 4-bit quants and 56% for the 5-bit ones. The second column runs the same quants with only `config.json` changed to native RoPE, which leaves just the quantization error.

That gap doesn't seem to change answers much. On 1,400 MMLU-Pro questions (100 per category, thinking off, greedy), my earlier 4-bit `-sh` build scored 68.64% with YaRN and 68.79% with native RoPE (exact McNemar p = 0.90), and gave the same answer to 92% of the questions. That build used the same recipe with different calibration data, and its KLD is indistinguishable from `4bpw-sh`'s. If you never need more than 262K context and want the closest match to the original, set native RoPE in `text_config` in `config.json`:

```json
"max_position_embeddings": 262144,
"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}
```

### Other tests

I ran these on an Apple M4 Max with 128 GB, on the files downloaded from this repo. The speed figures are single runs, and differences under about 5% are within normal run-to-run variation. The Sushi-4bpw column is from sushi 1.2.1; the two quants here were measured with sushi 1.2.4, so small speed and memory differences between the columns may come from the sushi version.

| | Sushi-4bpw | `4bpw-sh` | `5bpw-sh` |
|---|---:|---:|---:|
| tool calling, 28 cases (thinking off / on) | 28 / 28 | 24 / 28 ¹ | 28 / 28 |
| finding needles in a 400K-token document | 4/4 | 4/4 | 4/4 |
| decode, short prompt (tok/s) | 58.1 | 57.2 | 63.3 |
| decode at 64K / 200K context (tok/s) | 55.0 / 47.7 | 53.9 / 46.3 | 55.8 / 50.5 |
| prefill at 64K, cold (tok/s) | 690 | 688 | 684 |
| peak memory use during the tests | 98.9 GiB | 95.9 GiB | 110.0 GiB |

¹ The four misses are one prompt, run four times with greedy decoding. Asked for the standard deviation of a column, `4bpw-sh` with thinking off first calls `read_dataset` instead of the `summarise_column` tool the test expects. It does the same on sushi 1.2.3, and it picks the right tool with thinking on. Sushi-4bpw, `5bpw-sh` and my earlier 4-bit `-sh` build all get this prompt right. A single prompt can't show whether this is a borderline call or a weaker grasp of tool choice, so if you run an agent with thinking off, check it on your own tools.

## How I made them

1. Calibration data: 250 samples of 2,048 tokens. I wrote a set of starting prompts and had [turboderp's 6.05 bpw EXL3 quant](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) (rev d2e238f) write the replies, so the calibration text is the model's own output in its chat format. I'm not publishing the prompts.
2. Quantizing the experts: [ExLlamaV3](https://github.com/turboderp-org/exllamav3) v1.6.0 `convert.py` on the BF16 checkpoint (rev de4b8e4), with `-cb mcg -b 4.0 -sh` for one quant and `-cb mcg -b 5.0 -sh` for the other. MCG is the codebook that sushi's kernels read. `-sh` (scaled Hessian) lowered the per-layer error by about 4% when I tested it. The MTP layer's experts stay at 4.0 in both quants. Each conversion took about 6 hours on one NVIDIA H200.
3. Repacking: I took the experts (512 per layer, 48 layers, plus the MTP layer) from the converter output, wrote them in sushi's expert file format, and swapped them into a copy of Sushi-4bpw (rev 1477a9d). The only other change is the `expert_quant` block in `config.json`, which records how the experts were made.

Before publishing, I downloaded both quants back from this repo and checked them: every non-expert file is identical to Sushi-4bpw rev 1477a9d, all 147 expert files are there, and the KLD numbers above come from those downloaded files.

## Limitations

- These only run in sushi, on Apple Silicon. Like Sushi-4bpw, they won't load in transformers, vLLM, mlx-lm or ExLlamaV3.
- The quality numbers come from 24 prompts on one machine. They're fine for comparing quants with each other, but don't read too much into the absolute values.
- I'm not affiliated with the sushi author or the Qwen team.

## Credits and licence

- Qwen team, for the base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). These quants are derivatives under the Qwen Community License 1.0 (see `LICENSE`), including its conditions for large commercial services.
- beamivalice / beamster, for [sushi](https://github.com/beamivalice/sushi) and [Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw). These quants use its dense layers and n-gram table unchanged.
- turboderp, for the EXL3 format, ExLlamaV3 and its converter, and the 6.05 bpw quant I used to generate the calibration data.
- SiliconSpecies, for the 5bpw-hq quant in the comparison table.

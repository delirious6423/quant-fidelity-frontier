---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- mlx
- qwen4_exp
- kapitsa
- kquant
- quantized
- 4-bit
- apple-silicon
---

# Qwen3.8-Flash-Next, MLX Q4 KQuant

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in MLX Q4 from [Nace.AI](https://www.nace.ai/), closer to the original than [nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit), same speed. Run it locally on an Apple silicon Mac with [Kapitsa](https://github.com/nace-ai/kapitsa).

## Run it

```bash
brew install nace-ai/tap/kapitsa
kapitsa serve --model nace-ai/Qwen3.8-Flash-Next-MLX-Q4-KQuant
```


## Measurements

<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-Q4-KQuant/resolve/main/assets/kquant-dark.svg">
<img src="https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-Q4-KQuant/resolve/main/assets/kquant-light.svg" alt="Qwen3.8-Flash-Next: KL to BF16 against language-model size in GB. Red diamonds: KQuant. Grey circles: nace-ai plain 4-bit, mlx-community 4-bit, TensorFold 4-bit and TensorFold oQ4-MTP. Each point is labelled with its KL.">
</picture>

The red diamond is KQuant, this repo; grey circles are [nace-ai plain 4-bit](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit), [mlx-community 4-bit](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit), [TensorFold 4-bit](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit) and [TensorFold oQ4-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-oQ4-MTP). Lower KL is closer to the original BF16 model. This is the Qwen3.8-Flash-Next panel of the KQuant launch chart. Original-text corpus; the fresh-text numbers are in the table below.

Original: WikiText-2, GSM8K chats and HumanEval code; Fresh: frozen UltraChat test_sft and Schema-Guided Dialogue test dialogues. Each corpus scores 32,704 next-token positions offline with mlx-vlm 0.7.8 against bf16; KL is not task accuracy or speed.

Decimal GB; base-LM payload only; separate MTP, drafter and vision payloads excluded. Means are shown without uncertainty bars.

<details>
<summary>Exact values, payload bytes and revisions</summary>

| Pack | Base-LM GB | Original KL | Fresh KL | Pin |
|---|---:|---:|---:|---|
| [nace-ai KQuant](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-Q4-KQuant) [1] | 103.424 | 0.12752751 | 0.13168241 | [8fda1047](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-Q4-KQuant/tree/8fda104737236c4bbe1d6fb2784fb499ffd0a4cd) |
| [nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit) [2] | 103.424 | 0.14110035 | 0.14112859 | [5d423f9f](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit/tree/5d423f9f563c1f1b7ca7945b268e65c0b92b0eae) |
| [Jundot oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) [3] | 105.259 | 0.06475066 | 0.07135602 | [2322c888](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/tree/2322c888c2c314d4eceddf0159dec5d821aa1e41) |
| [mlx-community 4-bit](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit) [4] | 110.621 | 0.24928604 | 0.22910747 | [07b5dc6c](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit/tree/07b5dc6c54600a359b87f1e53e7adf6351c72a2c) |
| [TensorFold 4-bit](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit) [5] | 110.680 | 0.24921756 | 0.22900716 | [4fc403bc](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit/tree/4fc403bc1b53f2e5cbc763e7a710be1bed1ceb63) |
| [TensorFold oQ4-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-oQ4-MTP) [6] | 110.794 | 0.22957656 | 0.21622438 | [069090c3](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-oQ4-MTP/tree/069090c38f60e654c08ac6cb8cd83fce1affbdbc) |
| [ddalcu Serve-iQ 4.7bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw) [7] | 104.893 | 0.06636178 | 0.07430536 | [a0e89db0](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw/tree/a0e89db090a3e4b2e8ea2118fe9d0324d3854aca) |

Decimal GB; base-LM payload only; separate MTP, drafter and vision payloads excluded. Exact bytes by row: 1: 103,424,216,600 B; 2: 103,424,216,600 B; 3: 105,258,815,002 B; 4: 110,621,031,960 B; 5: 110,679,960,600 B; 6: 110,793,555,480 B; 7: 104,893,157,400 B.

[Jundot oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp), 1.8% larger, and [ddalcu Serve-iQ 4.7bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw), 1.4% larger, have lower KL than KQuant on both frozen corpora. Packs that beat KQuant are charted only when at least 10% larger, so these two appear only in this table.

</details>

The reference is Qwen/Qwen3.8-Flash-Next at `de4b8e4d`. Vision was not evaluated.

| Mac Studio M1 Ultra, 128 GB | [nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit) | KQuant |
| --- | ---: | ---: |
| Writes, temperature 0.6, tokens/s | 61.12 | 61.17 |
| Writes, greedy, tokens/s | 64.51 | 65.84 |
| Reads a 4K prompt, tokens/s | 738.25 | 752.00 |

Same speed within run-to-run variation. Six legs per pack, interleaved A B B A three times, with all legs included. Each leg uses `kapitsa bench --quick`, the median of three 512-token answers per decode setting; the table averages those six leg medians. macOS 26.7, AC power, 11 November 2026.

## Download and memory

The download is 98.6 GiB, the same size as [nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit). Measured on a Mac Studio M1 Ultra with 128 GB; a 48 GB Mac cannot hold the weights in memory. See the measured engine build below; the Homebrew release has not been measured with this pack.

This is a partial KQuant pack: 324 of 3,649 tensors retain the quantization of [nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit).

## Tool calls

| Extra tool call instead of a final answer | [nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit) | KQuant |
| --- | ---: | ---: |
| Thinking off | 0/60 | 0/120 |
| Thinking on | 3/30 | 4/30 |

KQuant's thinking-off count pools two independent runs of 60.

Every trial made a valid first tool call. The thinking-on failures made an additional `run_shell` call instead of answering; none was a broken call or a length-limit failure. The difference of one trial does not establish a regression. Temperature 0.6; thinking-off requests allowed 512 output tokens and thinking-on requests 6,144. [nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit) and KQuant used the same round-trip task, with separate server starts.

## Engine and comparison

[nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit) at `5d423f9f`.

Speed and tool calls were measured on a Kapitsa build from main at [`cc94c34f`](https://github.com/nace-ai/kapitsa/commit/cc94c34fb9db9e7f85392fb7f69808511e0bb8da), 12 commits after 0.2.34 and included in the 0.2.35 release candidate. Kapitsa 0.2.34 as released has not been measured with this pack. The benchmark executable has SHA256:

```text
ee09baedf5c4cba938a7f0ddb363a2e77e673b424a4f1a25656ffb5492239133
```

The source archive identifies that commit; all 9,495 archived source files match the extracted build tree. The build logs record the executable build, and its digest matches the one used for the measurements.

## Credits and license

The Qwen Team at Alibaba Group designed and trained the original model. Nace AI quantized its published weights. This is not an official Qwen release.

The [Qwen Community License 1.0](LICENSE) applies unchanged. Model-as-a-Service or AI Work Assistant businesses need a separate Qwen license for commercial use under condition 2.

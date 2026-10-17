---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - qwen4exp
  - moe
  - dacan
  - strata
  - nvfp4
  - q8_0
  - rtx-2080-ti
---

# Qwen3.8-Flash-Next for DACAN

Weights of **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** packed for
**[DACAN](https://github.com/tirex999/DACAN)** — our engine, a fork of [Strata](https://github.com/Niko1221/Strata)
rebuilt for two RTX 2080 Ti (22 GB, NVLink) and two AVX-512 Xeon sockets. DACAN runs this one model family only.

> **Status (03.10.2026): complete**, 27 files (260.6 GB). The dense GGUF was checked against the file the DACAN service
> runs: three greedy answers (1 984, 1 931 and 596 tokens) came out identical character for character.

## The model and its architecture

`Qwen4ExpForConditionalGeneration`, `model_type: qwen4_exp`, GGUF `general.architecture: qwen4exp`:

- 125B parameters with 6B activated per token, plus a 51B n-gram embedding table and a 4B MTP layer
- 48 layers: 12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE)), hidden size 2560
- Gated DeltaNet: 48 V heads, 16 QK heads; Qwen Sparse Attention: 24 Q heads, 2 KV heads
- MoE: 512 experts, 10 routed + 1 shared per token, expert width 640
- Gated residual (widened residual streams), n-gram embedding at layer 2, 1 MTP layer
- Context 262,144 tokens natively

DACAN checks this geometry on load (48 layers, 2560, 512 experts, 10 active, 24 / 2 heads) and refuses anything else,
so fine-tunes of Qwen3.8-Flash-Next with the same shape work, other models and pruned variants (REAP etc.) do not.

## Two layouts: NVFP4 + Q8 (DACAN) and Q8_0

**NVFP4 + Q8 (DACAN)** is our name for the layout the DACAN service runs: NVIDIA's NVFP4 routed experts, everything
else that is large in Q8_0, the small sensitive matrices left in BF16. The experts are about 95 % of the weights and
are read from RAM on every token, so their size sets the speed; the parts every token passes through stay at 8 bits.

| part | NVFP4 + Q8 (DACAN) | size |
|---|---|---:|
| routed experts — gate, up, down × 512 × 48 layers | **NVFP4** from NVIDIA's checkpoint: E2M1 values, an E4M3 scale per 16 values, an FP32 scale per tensor | 68 GB |
| attention and DeltaNet projections, shared expert, token embeddings, output head | **Q8_0** | 4.1 GiB |
| n-gram embedding table | **Q8_0** from FP8 | 54 GB |
| routers, hyper-connections, sparse-attention indexer, DeltaNet `ssm_alpha/beta`, `ple_key/value` | BF16 | 1.4 GiB |
| `ple_conv1d` / norms | F16 / F32 | 0.01 GiB |

The **Q8_0** layout swaps the experts for ours in Q8_0 (128 GB) and keeps the rest; it needs 120 GiB of RAM for the
experts instead of 63 GiB and answers slower (on Swift 1.5: 49 tok/s against 58, median of 9 runs). The same NVFP4 + Q8 (DACAN) mix of the fine-tune Swift 1.5, with experts we
quantized from its BF16 weights, is in
[tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN).

## Files

| file | what | size |
|---|---|---|
| `Qwen3.8-Flash-Next-Q8_0-experts.bin` | routed experts in Q8_0, requantized by us from Qwen's FP8 checkpoint (`tools/q8_experts.py`: 0.55–0.59 % from the FP8 weights) | 128 GB |
| `Qwen3.8-Flash-Next-NVFP4-experts.bin` | routed experts from NVIDIA's [NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), repacked for DACAN (`tools/nvfp4_experts.py`; unpacks to NVIDIA's values exactly, checked on layers 0, 21, 47) | 68 GB |
| `Qwen3.8-Flash-Next-dense-Q8_0.gguf` | everything but the routed experts: attention, DeltaNet, shared expert, router, head — Q8_0 (BF16/F32 where the model keeps them); cut from our 4-bit quant's first shard by `dense_gguf.py`, 1 079 of 1 223 tensors, sha256 `3cceeeca…d470d525` | 5.99 GB |
| `Qwen3.8-Flash-Next-PLE-FP8-Q8_0.gguf` | the n-gram embedding table, Q8_0 from FP8 | 54 GB |
| `pack-nvfp4/`, `pack-q8_0/` | DACAN packs (expert index, dense blob, tokenizer): `pack-nvfp4` for NVFP4 + Q8 (DACAN), `pack-q8_0` for Q8_0 | 1.5 GB |
| `mtp/` | the MTP draft layer | 0.8 GB |
| `data/profile_other_2609.bin`, `data/usage_other_2609.bin` | routing tables for the run command | 0.3 MB |

**Layout.** Put the dense GGUF and the `*-experts.bin` you use in one folder: `pack-*/native_experts.txt` names the
experts file without a path and DACAN looks for it next to the `--native` GGUF. The service config we run (NVFP4 + Q8 (DACAN)) is
`--pack pack-nvfp4 --native Qwen3.8-Flash-Next-dense-Q8_0.gguf --ple-gguf Qwen3.8-Flash-Next-PLE-FP8-Q8_0.gguf --mtp mtp
--expert-profile data/profile_other_2609.bin --second-card 1 --second-card-usage data/usage_other_2609.bin
--max-context 262144 --kv int8 --spec 4` (full list in the DACAN README).

How to run: see [the DACAN README](https://github.com/tirex999/DACAN/blob/main/README-2x2080Ti.md).
Measured 03.10.2026 on 2× RTX 2080 Ti + 2× Xeon 8368 (NVFP4 + Q8 (DACAN), free node, greedy):

| | tok/s |
|---|---:|
| answer: code / a list after a fresh 19K-token prompt | 76.9 / 80.4 |
| answer with reasoning low | 71.9 |
| answer at 98K tokens of context | 69.9 |
| answer: English / Russian prose | 60.4 / 47.0 |
| prompt read from scratch: 19K / 98K tokens | 786 / 715 |

The answer speed follows the MTP drafts: 85–89 % accepted on code and lists, 57 % on English prose, 32 % on Russian.
Context 256K (a needle found up to 195K). Details: [README-2x2080Ti](https://github.com/tirex999/DACAN/blob/main/README-2x2080Ti.md).

## Licenses

The model: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), copyright (c)
2026 Qwen. The NVFP4 experts are derived from NVIDIA's checkpoint and licensed by NVIDIA Corporation under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).


## Support the project · Поддержать проект

Everything here — the quants, the engine, the measurements — is made and published for free. If it helped you run a
model on your own hardware, you can say thanks with a donation. Если это помогло вам запустить модель на своём железе,
можно поблагодарить донатом.

| | address | QR |
|---|---|---|
| **YooMoney / ЮMoney** (roubles: a YooMoney wallet or any bank card) | `4100119356331418` · [send / перевести](https://yoomoney.ru/to/4100119356331418) | <img src="https://raw.githubusercontent.com/tirex999/2x2080ti-nvlink-44gb/main/docs/donate/yoomoney.svg" width="120" alt="QR YooMoney"> |
| **USDT** (TRC-20, Tron network) | `TBvoJHi7uyonSpvdH9Y6RAAXGWYVR2jeqw` | <img src="https://raw.githubusercontent.com/tirex999/2x2080ti-nvlink-44gb/main/docs/donate/usdt-trc20.svg" width="120" alt="QR USDT TRC-20"> |
| **Ethereum** (ETH and Ethereum-network tokens, ERC-20) | `0x66CA7c683fbaF030b2300c918A3751209eA30dEa` | <img src="https://raw.githubusercontent.com/tirex999/2x2080ti-nvlink-44gb/main/docs/donate/eth.svg" width="120" alt="QR Ethereum"> |

> ⚠️ Send only Tron-network assets (USDT TRC-20, TRX) to the Tron address and only Ethereum-network assets to the
> Ethereum address; anything sent over the wrong network is lost. Сеть важна: отправленное не в той сети пропадёт.

Thank you! Спасибо! 🙏

---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - ik_llama
  - qwen4exp
  - imatrix
  - mtp
  - speculative-decoding
  - text-only
---

**Hugging Face's download widget does not recognize _KT GGUFs. Download links are in the table below.**

# Qwen3.8-Flash-Next, ik_llama.cpp IQ4_KT and IQ3_KT GGUFs

Text-only GGUFs of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), quantized with an importance matrix calibrated for agentic coding, plus an IQ4_KT quant of the model's own MTP draft head.

The v3 IQ4_KT and the v3 IQ3_KT replace the IQ4_KT and the IQ3_KT first published here on 2026-09-01. The version number continues from the [IQ4_KT_v2](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF_v2). Against the BF16, the mean KL divergence of the v3 IQ4_KT is 56% lower than that of the replaced IQ4_KT on English text and 55% lower on code. For the v3 IQ3_KT against the replaced IQ3_KT, it is 44% lower on English and 42% lower on code (see Validation). The v3 IQ4_KT is 5.7 GiB smaller than the replaced IQ4_KT, and the v3 IQ3_KT is 7.7 GiB smaller than the replaced IQ3_KT. The replaced files remain in the history of this repository, at revision `2ede446`. The MTP head is unchanged.

Both v3 files were quantized with ik_llama.cpp at `d9e2868`, which includes the CUDA IQ4_KT and IQ3_KT encoder of [PR #2572](https://github.com/ikawrakow/ik_llama.cpp/pull/2572).

**These files require [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) at `cb860a9` or later, the merge of [PR #2481](https://github.com/ikawrakow/ik_llama.cpp/pull/2481). Earlier builds do not read the KT row tails in these files correctly.**

## Choosing a file

The version identifies the quantization recipe. All three recipes use the same upstream model.

| Recipe | Total file size | GPU-resident model tensors with the documented offload configuration | Routed experts in RAM or VRAM | English mean KLD | Code mean KLD |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 IQ4_KT | 82.75 GiB | 4.5 GiB | 52.6 GiB | 0.058 | 0.056 |
| v3 IQ3_KT | 72.52 GiB | 4.5 GiB | 47.2 GiB | 0.085 | 0.078 |
| [IQ4_KT_v2](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF_v2#files) | 80.72 GiB | 2.4 GiB | 52.6 GiB | 0.133 | 0.125 |

Choose v3 IQ4_KT for the lowest measured KL divergence. Choose v3 IQ3_KT to reduce routed-expert memory. Choose IQ4_KT_v2 to reduce GPU-resident model tensors by about 2 GiB relative to v3.

The v3 files store attention, SSM, hyper-connection, and shared-expert tensors at Q8_0. The DSA indexer projections use BF16. Most of these tensors use IQ4_KT in v2. The v3 IQ3_KT reduces routed-expert precision further than v3 IQ4_KT.

These GPU figures exclude the KV cache and compute buffers. Total VRAM use depends on context length, micro-batch size, cache types, and offload settings. The KV cache and compute buffer take about 7.5 GiB at 256K context with `-ub 1024` and the example cache types. Total file size is not the RAM requirement when `--defer-ple` leaves the PLE table on disk.

A lower `-c` reduces context length. A lower `-ub` reduces prompt-processing speed. Moving more tensors to the CPU with `-ot` reduces decoding speed.

KLD measures divergence from the BF16 reference on the documented samples. Lower KLD does not establish correctness of generated code. These KLD figures do not establish a throughput ranking.

## Why there are two repositories

IQ4_KT_v2 remains in the `_v2` repository. This main card covers v2 and v3. The `_v2` suffix identifies an earlier recipe. It does not identify the latest release.

The optional MTP draft head is available in this repository. The `_v2` repository contains only the v2 main model.

## Files

| File | Quant | Download |
| --- | --- | ---: |
| [`Qwen3.8-Flash-Next-ik_llama-v3-IQ4_KT-00001-of-00002.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF/resolve/main/Qwen3.8-Flash-Next-ik_llama-v3-IQ4_KT-00001-of-00002.gguf?download=true) | v3 IQ4_KT | 44.68 GiB |
| [`Qwen3.8-Flash-Next-ik_llama-v3-IQ4_KT-00002-of-00002.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF/resolve/main/Qwen3.8-Flash-Next-ik_llama-v3-IQ4_KT-00002-of-00002.gguf?download=true) | v3 IQ4_KT | 38.07 GiB |
| v3 IQ4_KT total | | 82.75 GiB |
| [`Qwen3.8-Flash-Next-ik_llama-v3-IQ3_KT-00001-of-00002.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF/resolve/main/Qwen3.8-Flash-Next-ik_llama-v3-IQ3_KT-00001-of-00002.gguf?download=true) | v3 IQ3_KT | 44.33 GiB |
| [`Qwen3.8-Flash-Next-ik_llama-v3-IQ3_KT-00002-of-00002.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF/resolve/main/Qwen3.8-Flash-Next-ik_llama-v3-IQ3_KT-00002-of-00002.gguf?download=true) | v3 IQ3_KT | 28.19 GiB |
| v3 IQ3_KT total | | 72.52 GiB |
| [`Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00001-of-00002.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF_v2/resolve/main/Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00001-of-00002.gguf?download=true) | IQ4_KT_v2 | 44.41 GiB |
| [`Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00002-of-00002.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF_v2/resolve/main/Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00002-of-00002.gguf?download=true) | IQ4_KT_v2 | 36.31 GiB |
| IQ4_KT_v2 total | | 80.72 GiB |
| [`Qwen3.8-Flash-Next-MTP-ik_llama-IQ4_KT.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF/resolve/main/Qwen3.8-Flash-Next-MTP-ik_llama-IQ4_KT.gguf?download=true) | IQ4_KT | 2.21 GiB |
| [`qwen4exp-agentic-v3.imatrix`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF/resolve/main/qwen4exp-agentic-v3.imatrix?download=true) | n/a | 312 MiB |
| [`qwen4exp-mtp-v1-draft.imatrix`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF/resolve/main/qwen4exp-mtp-v1-draft.imatrix?download=true) | n/a | 6.6 MiB |

The v2 download links point to the `_v2` repository.

Download both shards of one quant into the same directory. Pass the first shard to `llama-server`.

In the v3 files, the routed expert `ffn_down_exps` tensors use IQ3_KT. The routed expert gate/up tensors use IQ4_KT in the v3 IQ4_KT. In the v3 IQ3_KT they use IQ4_KT in blocks 0 to 5 and 38 to 47 and IQ3_KT in blocks 6 to 37. The PLE table uses IQ4_KT in the v3 IQ4_KT and IQ3_KT in the v3 IQ3_KT. The DSA indexer projections use BF16 and the four-column PLE convolution F16. Every other two-dimensional weight tensor uses Q8_0, including the token embeddings, the output head, the attention projections, the hyper-connection tensors, and the shared experts.

## MTP draft head

`Qwen3.8-Flash-Next-MTP-ik_llama-IQ4_KT.gguf` is the model's own next-token prediction head, converted from the same BF16 revision as the main files. It is not a standalone model, and it is optional. Load it beside a v2 or v3 main model with `-md`. Enable it with `--spec-type mtp:n_max=1,p_min=0.0`. If the head does not fit beside the main model, add `-draft "-ub 256 -b 256"`. It gives the head a separate batch size of 256, which makes its compute buffer smaller.

The head carries its own hyper-connection mixer, so the loader binds that rather than falling back to the trunk's. Its importance matrix was collected paired against the IQ3_KT published here on 2026-09-01, on the same corpus as the main files. With the v3 IQ4_KT, `llama-spec-bench` accepted 90.41%, 79.43%, and 68.21% of its drafts on the built-in code, extract, and story tasks (temperature 0, 256 tokens).

## Runtime

Example invocation:

```bash
GGML_CUDA_NO_PINNED=1 llama-server \
   -m Qwen3.8-Flash-Next-ik_llama-v3-IQ4_KT-00001-of-00002.gguf \
   --defer-ple -ngl 49 -ot 'ffn_.*_exps=CPU' \
   -fa on -ctk q8_0 -ctv q8_0 -ictk q8_0 --jinja \
   -c 196608 -b 2048 -ub 1024 --ctx-checkpoints 16
```

`--defer-ple` leaves the PLE table on disk: 25.0 GiB in the v3 IQ4_KT and the IQ4_KT_v2, 20.3 GiB in the v3 IQ3_KT. The routed experts are read every token and need RAM or VRAM (see Choosing a file). With `-ngl 49`, all other tensors except the token embeddings are on the GPU. Metal, Vulkan, ARM, and multi-GPU execution are not validated.

For multi-turn chat, use `391a95f` ([PR #2524](https://github.com/ikawrakow/ik_llama.cpp/pull/2524)) or later, or pass `"preserve_thinking": true` in `chat_template_kwargs`. Builds from [PR #2382](https://github.com/ikawrakow/ik_llama.cpp/pull/2382) up to PR #2524 force `preserve_thinking` off unless the request sets it. With it off, whether forced or set to `false` by the client, the chat template drops the reasoning of earlier turns, and each new user message makes the server process the previous exchange again.

## Validation

### KL divergence against the BF16 source

The reference is the BF16 GGUF converted from the released weights. Both test texts are held out and share no text with the calibration corpus. Each text is ten chunks of 2048 tokens. The code text is CPython and PostgreSQL source.

| File | English mean KLD | Same top-1 | Code mean KLD | Same top-1 |
| --- | ---: | ---: | ---: | ---: |
| v3 IQ4_KT | 0.057803 | 91.183% | 0.055726 | 95.142% |
| v3 IQ3_KT | 0.085269 | 89.384% | 0.077747 | 94.653% |
| IQ4_KT_v2 | 0.132805 | 86.276% | 0.124558 | 93.216% |
| IQ4_KT (replaced) | 0.1325 | 85.74% | 0.1247 | 92.41% |
| IQ3_KT (replaced) | 0.1515 | 85.05% | 0.1350 | 92.13% |

Reference perplexities were 3.3316 for English and 1.7218 for code. The v3 IQ4_KT measured 3.3893 and 1.7303, the v3 IQ3_KT 3.4484 and 1.7522, and the IQ4_KT_v2 3.5335 and 1.8031. The replaced files were measured on 2026-09-01 with an earlier ik_llama.cpp build and an earlier BF16 conversion of the same revision. The IQ4_KT_v2 gives the same KLD values against both references. The replaced IQ4_KT measured 3.5625 and 1.8209, and the replaced IQ3_KT 3.6119 and 1.8273. Same top-1 is the share of positions where the quantized model's most likely token matches the reference's.

The DSA indexer of this model selects 2048 tokens, the length of one chunk. These runs therefore do not exercise its selection, and the BF16 type of the indexer projections has no measurement behind it.

## Calibration

The importance matrix was computed over 700 chunks of 2048 tokens. The corpus was built to match an agentic coding deployment rather than general capability: 45% agentic and technical English, 30% code (including the entirety of `ik_llama` and `llama.cpp` source code), 15% general English prose, 10% Chinese prose, measured in tokens. The Chinese share is there so the quantizer does not treat those channels as unimportant; it is not meant to tune the model toward Chinese.

## Conversion notes

Source: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

The BF16 conversion used `convert_hf_to_gguf.py` from llama.cpp at `f8dbcd6`, unmodified, `--outtype bf16`. The converter drops the vision tower and the MTP head. The v3 files were quantized with `llama-quantize` from ik_llama.cpp at `d9e2868` with `--cuda-quantize`. The head is the file published here on 2026-09-01. It was converted from the same revision with a local patch that exports the `mtp.*` tensors, which the stock converter omits.

Rules shared by both v3 files, the start of `--custom-q`:

```text
indexer\.=bf16,hc_=q8_0,ssm_alpha=q8_0,ssm_beta=q8_0,attn_qkv=q8_0,attn_gate=q8_0,ssm_out=q8_0,attn_q\.weight=q8_0,attn_output=q8_0,attn_k\.weight=q8_0,attn_v\.weight=q8_0,_shexp=q8_0,ple_key=q8_0,ple_value=q8_0,eh_proj=q8_0,ple_conv1d=f16
```

v3 IQ4_KT options:

```text
--imatrix qwen4exp-agentic-v3.imatrix
--token-embedding-type q8_0
--output-tensor-type q8_0
--custom-q "<shared rules>,ffn_down_exps=iq3_kt,ffn_(gate|up)_exps=iq4_kt,per_layer_token_embd=iq4_kt"
iq4_kt
```

v3 IQ3_KT options:

```text
--imatrix qwen4exp-agentic-v3.imatrix
--token-embedding-type q8_0
--output-tensor-type q8_0
--custom-q "<shared rules>,ffn_down_exps=iq3_kt,blk\.([0-5]|3[89]|4[0-7])\.ffn_(gate|up)_exps=iq4_kt,ffn_(gate|up)_exps=iq3_kt,per_layer_token_embd=iq3_kt"
iq3_kt
```

Tensor counts in the v3 IQ4_KT are 666 Q8_0, 97 IQ4_KT, 48 IQ3_KT, 24 BF16, 388 F32, and one F16. The v3 IQ3_KT has 666 Q8_0, 113 IQ3_KT, 32 IQ4_KT, 24 BF16, 388 F32, and one F16.

SHA-256 of the files as uploaded:

```text
546caa277e273708317d216c58b8dfe35352182b5ff63c33d24414082b3c8660  qwen4exp-agentic-v3.imatrix
fcfa3f22f50f055167e09f7e605cc2ada1fd90bef6cf624e4af8b8d8543490f4  qwen4exp-mtp-v1-draft.imatrix
b15000cd9618129885c098c3fd0859a8897be984d9f0f883a183c07827710895  Qwen3.8-Flash-Next-MTP-ik_llama-IQ4_KT.gguf
b6cd286c8381f82da99522a4c7b99b584b1b3b40695898a4f233e06fb6815d56  Qwen3.8-Flash-Next-ik_llama-v3-IQ4_KT-00001-of-00002.gguf
36b34edf77d761e66cbccbb80679f76f0b587724c5bc2e2667e637b7dc347179  Qwen3.8-Flash-Next-ik_llama-v3-IQ4_KT-00002-of-00002.gguf
446a19607bf406bcb4bf5b16102d1628675ee06d30b2f470679cde327adfa20c  Qwen3.8-Flash-Next-ik_llama-v3-IQ3_KT-00001-of-00002.gguf
a9a4e45356211366690bec8c49f665e16b78f2368c4f1abf2a17dcd9c5b83815  Qwen3.8-Flash-Next-ik_llama-v3-IQ3_KT-00002-of-00002.gguf
```

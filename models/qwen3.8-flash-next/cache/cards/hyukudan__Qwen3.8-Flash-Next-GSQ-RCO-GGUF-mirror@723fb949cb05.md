---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- gguf
- qwen
- gsq
- mirror
---

# Qwen3.8-Flash-Next GSQ-RCO GGUF (ISTA-DASLab), mirror

Mirror of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)@[`ed59f92082b1e93c0e96d60a8b11aab089b52f09`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) for reproducibility of [github.com/hyukudan/strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab).

The files are exact copies of that upstream revision: same paths, same bytes. They were not re-quantized or modified. All credit belongs to the original authors below; please use and cite the upstream repositories. This mirror only pins the revision our engines were built and measured against.

## Original authors

- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab): GSQ / RCO quantization of Qwen3.8-Flash-Next; please cite GSQ ([arXiv:2604.18556](https://arxiv.org/abs/2604.18556), [code](https://github.com/IST-DASLab/GSQ)) and RCO ([arXiv:2605.00649](https://arxiv.org/abs/2605.00649), [code](https://github.com/IST-DASLab/RCO)), BibTeX in `README.upstream.md`
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): original model Qwen3.8-Flash-Next (Qwen Community License 1.0)
- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)@`de4b8e4d43b917e7706784d8bb445c9af86a3540`

## License

The upstream repo declares Apache-2.0 but ships no LICENSE file, so `LICENSE` is the standard Apache License 2.0 text from apache.org. The weights derive from Qwen3.8-Flash-Next, so `LICENSE.Qwen` holds the base model's Qwen Community License 1.0, verbatim from Qwen/Qwen3.8-Flash-Next@de4b8e4d; its conditions apply as well.

## How our engines use it

- The engine's opt-in GSQ speed profile (`STRIX_GSQ_Q20_A8=1`, never the default) composes a hybrid file: the production UD-Q4_K_XL GGUF with every layer's routed experts replaced by the Q2_0 tensors from these shards (byte copies). Recipe: `gguf_compose.py hybrid`. IQ3_S was evaluated as a second variant.
- Quality: top-1 agreement 76.6 % vs 88.1 % for UD-Q4_K_XL on our locked 64-prompt set (-11.5 points), so the profile does not pass the quality recipe. See [GSQ profile](https://github.com/hyukudan/strix-halo-engine-lab#gsq-opt-in-speed-over-quality-profile) and [docs/gsq-speed-profile.md](https://github.com/hyukudan/strix-halo-engine-lab/blob/main/docs/gsq-speed-profile.md).

## Files

16 files, 150,042,091,035 bytes (150.04 GB). `README.md` is this card; the upstream model card is kept verbatim as `README.upstream.md`. Large-file sha256 values equal the upstream LFS hashes and were recomputed from our local copies before upload (any mismatch would have stopped the upload). `SHA256SUMS` lists every file.

| File | Bytes | sha256 |
|---|---:|---|
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `LICENSE` (added by this mirror) | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `LICENSE.Qwen` (added by this mirror) | 3,235 | `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08` |
| `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 37,623,740,192 | `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720` |
| `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `README.upstream.md` (upstream `README.md`) | 15,693 | `71496aac3c9126085519dbe3aa82e46eb9f5749ee122b089a66182a94b0db6a3` |
| `assets/plots/Qwen3.8-Flash-Next-aime25_vs_avg_bit_width.png` | 63,324 | `32131afc7bd935cbaf99def4be5a417e831ad38a97d4f0ff17367464446488bb` |
| `assets/plots/Qwen3.8-Flash-Next-gpqa_diamond_vs_avg_bit_width.png` | 70,514 | `067ebf852b3fce3bdda64ea12902025528d7b40ee8236bf7e730ce34db6d7370` |
| `assets/plots/Qwen3.8-Flash-Next-lcb_vs_avg_bit_width.png` | 68,136 | `ea95271e91fa5d60272448d45774a7e6224dc59d5a286acdbc843181a081cd76` |
| `assets/plots/Qwen3.8-Flash-Next-prefill_throughput.png` | 86,320 | `cba07d58699539b7f007b5f3092844f7594ea9ad758e7cb8a207b092f4b7df1d` |
| `assets/plots/Qwen3.8-Flash-Next-task_avg_vs_avg_bit_width.png` | 73,744 | `40570437305346425b1f1690ea9bc03f829973af70924d39c003cb6a7024bedf` |
| `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.rco-allocation.txt` | 39,556 | `9721043873f19fc924dc0680b6ac202323a8d5a818c902e6caee698ec26b8574` |
| `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt` | 39,368 | `df10add9e3389cf5930e1dd527df4c0b24b38597a7e4652aa3a1ab2d3e12fd9b` |
| `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.rco-allocation.txt` | 39,445 | `3cf652c8d832d84cabe9900148ab0b5857f5c59acb1e102be9e18537684bdf87` |
| `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt` | 39,062 | `81371f9d3e10b5ff2ea718346165ad64f523d5f2b6a5619424e2803133cda17d` |

Only the Q2_0 and IQ3_S quantizations are mirrored (plus all tensor-allocation files and README plots); IQ2_XS and IQ3_XXS are on the upstream repo.

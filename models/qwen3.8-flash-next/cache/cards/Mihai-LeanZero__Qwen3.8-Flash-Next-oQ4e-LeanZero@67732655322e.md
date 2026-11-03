---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- qwen4_exp
- quantized
- 4-bit
- mtp
- apple-silicon
---

# Qwen3.8-Flash-Next-oQ4e-LeanZero

This is a 4-bit MLX build of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for Apple silicon Macs. We quantized it ourselves, straight from Qwen's official BF16 release, and we kept everything the original ships with: the language model, the 51B n-gram embedding, the vision encoder and the multi-token prediction (MTP) head. It is 100.275 GiB (107.67 GB) on disk and runs on a single 128 GB Mac.

It was made by [LeanZero](https://leanzero.net). We wanted a build of this model whose every step we could check, from Qwen's release to the file on disk, and we tested it with the MLX engine behind goose local edition, our free, open-source build of the goose agent. We're sharing it free of charge, as is, under Qwen's licence (see the licence section below).

## Where the weights come from

The source is the official repository [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` (last modified 2026-08-27). That release is 131 BF16 safetensors shards, 335.28 GiB (360.0 GB) in total.

Our converter downloaded each of the 131 shards once and computed its sha256 before using it. Every one matched the LFS sha256 that Hugging Face lists for that file at the pinned revision, and the converter would have stopped on the first mismatch. Nothing else went into the weights: no fine-tuning, no merging, no other model.

The chat template, tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`), `generation_config.json`, the image and video preprocessor configs and the `LICENSE` file are byte-for-byte Qwen's files at that revision. We checked each one against the hash Hugging Face records for it. `config.json` differs because it carries the quantization map, and `model.safetensors.index.json` lists our 21 shards.

`own_convert_manifest.json` records the source repository and revision, the calibration data hashes and a sha256 for each of the 3,748 output tensors.

## How it was quantized

The bit layout follows oMLX's oQ4e recipe (oMLX v0.7.3, Apache-2.0, by jundot). The same rule applies on every layer:

| tensors | format |
|---|---|
| routed experts (gate, up and down, all 48 layers plus the MTP layer) | affine 4-bit, group size 64 |
| n-gram embedding table (128 shards) | affine 4-bit, group size 32 |
| shared experts | affine 8-bit, group size 128 |
| every other linear layer (attention, indexer, linear attention, hyper-connection mixers, MTP projections) and the token embedding and output head | affine 8-bit, group size 64 |
| MoE router, norms, convolutions and the whole vision encoder | BF16, values copied unchanged (two tensors are transposed into the layout MLX expects) |

The MTP head is kept and quantized like a decoder layer, so engines that support it can use it for speculative decoding.

For rounding we used importance-weighted rounding (the oQe method): an importance value for every input column of each layer (and for each expert separately), then a clipping search and a weighted least-squares refit of every group's scale and bias. The importance values come from our own forward pass through Qwen's BF16 weights, run during the same single download pass, over 1,024 sequences of 512 tokens drawn from oMLX's Apache-2.0 multilingual code calibration set. The implementation is ours, and we checked it against oMLX's routine on 241 adversarial test cases, where it gave bit-identical results. The token embedding, the output head, the top hyper-connection mixer and the n-gram table use plain round-to-nearest, as the recipe prescribes.

Size on disk by part:

| part | size |
|---|---|
| language model | 68.227 GiB |
| n-gram embedding | 29.802 GiB |
| MTP head | 1.409 GiB |
| vision encoder | 0.836 GiB |
| total | 100.275 GiB |

## Offline checks

We compared every output tensor with Qwen's BF16 original and with Jundot's oMLX build of the same model ([Jundot/Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp)), which uses the same recipe. Error here means the relative error of our dequantized weights against Qwen's BF16 weights.

1. Tensor names, shapes, data types and the quantization map are identical to Jundot's build: 3,748 stored tensors. Counted as weights instead (a quantized weight is stored as codes, scales and biases), that is 1,707 weights plus one constant `weight_scale` of 1.0 that the MLX loader expects next to the n-gram table.
2. The 132 weights rounded to nearest are bit-identical to Jundot's, in codes, scales and biases.
3. The 687 weights copied in BF16, the vision encoder among them, are byte-identical to Jundot's and so to Qwen's, apart from the two layout transposes.
4. For the 888 weights with importance-weighted rounding, the 4-bit routed experts have a median relative error of 0.084 (worst 0.093) and a cosine similarity to the original of at least 0.9957. Our 4-bit codes match Jundot's codes in 98.7% of positions at the median. The 8-bit shared experts have a median relative error of 0.0072, and the other 8-bit layers 0.0066.
5. The weakest single tensor is the MTP hyper-connection mixer's input weight, at relative error 0.155 (cosine 0.989). Jundot's build has 0.170 on the same tensor.
6. A final check re-hashed every output tensor after the run and found no mismatch.

## Measured on a Mac

All numbers below were measured on one Mac Studio with an M5 Max and 128 GB of memory, with the macOS GPU wired limit raised to 120,832 MB (`sudo sysctl iogpu.wired_limit_mb=120832`). We used LeanZero's MLX engine, a fork of Rapid-MLX. The checks and the context table below ran on `v0.14.3-lz.24` with three attention switches added by hand (split-K decode, the QSA indexed route and block-sparse prefill); `v0.14.3-lz.25` ships those switches after a further review round. The n-gram table was kept on disk, MTP used 2 draft tokens, and the prefix cache lived on disk.

Vision, MTP and the n-gram table all worked end to end. The model read a test image correctly, and MTP drafting was active. The n-gram embedding rows were read from disk as tokens went through. Separately, prompt-lookup speculative decoding accepted 591 of 591 drafted tokens on a temperature-1.0 code re-emit.

Greedy output agrees closely with Jundot's build. On 7 fixed prompts, 5 gave byte-identical output, among them a code edit on an 8k-token prompt. The other 2 diverged early, at tokens 31 and 0, with the same content and the same actions. Jundot's build gave identical output on all 7 prompts run to run.

Native context is 262,144 tokens. Past that, Qwen recommends static YaRN with factor 4.0 (up to about 1M tokens) or factor 2.0 if most of your prompts are around 524,288 tokens. We ran a needle test: a code hidden in a long document, which the model must quote back. Steps after the first one in each ladder extend the previous prompt, so only the new tokens are prefilled.

| prompt tokens | YaRN factor | time to first token | decode tok/s | needle |
|---|---|---|---|---|
| 455,700 (cold) | 4 | 691.5 s | ~27 | found |
| 455,643 (cold) | 2 | 542.7 s | 30.9 | found |
| 491,058 (+35,415) | 2 | 81.2 s | 28.6 | found |
| 522,409 (+31,351) | 2 | 67.6 s | 28.7 | missed |
| 521,727 (cold) | 4 | 656.9 s | 28.4 | found |
| 653,246 (+131,519) | 4 | 317.1 s | 26.5 | found |

Decode speed held between 25 and 31 tokens per second from 455k to 653k tokens of context. The largest context we held is 653,246 tokens with YaRN factor 4: the engine peaked at 112.1 GiB of MLX memory (114.2 GiB resident), swap stayed flat, and macOS reported no memory pressure warning. An attempt at about 783k got through roughly 711k tokens of prefill at 114.3 GiB resident before we stopped it, so on a 128 GB Mac 653k is the ceiling we can vouch for, not 1M. In the same session the engine's memory guard once refused the 653k prompt by mistake, and the same prompt went through on a retry. The guard fix for that is in `v0.14.3-lz.26`, and a later session on a pre-release of that fix held 521k and then 652k, with the needle found both times.

## Running it with MLX

This checkpoint uses Qwen's new `qwen4_exp` architecture (QSA sparse attention, Gated DeltaNet, n-gram embedding), stored in the mlx-vlm layout. Stock `mlx-lm` doesn't load this architecture. We've only tested it with LeanZero's Rapid-MLX fork, which serves an OpenAI-compatible API:

You need `uv` (for `uvx`; `brew install uv` or see [docs.astral.sh/uv](https://docs.astral.sh/uv/)) and, for long context on a 128 GB Mac, a raised GPU wired limit. The `sysctl` setting needs sudo and resets when the Mac restarts.

```bash
pip install -U "huggingface_hub[cli]"
hf download Mihai-LeanZero/Qwen3.8-Flash-Next-oQ4e-LeanZero --local-dir ~/models/Qwen3.8-Flash-Next-oQ4e-LeanZero

sudo sysctl iogpu.wired_limit_mb=120832   # 128 GB Mac; only needed for long context

RAPID_MLX_SPLITK_DECODE=1 RAPID_MLX_QSA_BLOCK_SPARSE=1 \
uvx --from "rapid-mlx[mtp] @ git+https://github.com/leanzero-srl/Rapid-MLX@v0.14.3-lz.26" \
  rapid-mlx serve ~/models/Qwen3.8-Flash-Next-oQ4e-LeanZero \
  --port 8000 \
  --qwen4-ple-offload \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --qwen4-rope-scaling yarn
```

`--qwen4-ple-offload` keeps the n-gram table (29.8 GiB) on disk and reads the rows each token needs, which is what lets the model fit on a 128 GB Mac. Put the model on the Mac's internal SSD, not a network or Thunderbolt share, because those rows are read per token. `--qwen4-rope-scaling yarn` turns on YaRN factor 4 for contexts past 262,144 tokens. Use `yarn:2` for factor 2, or leave the flag off to stay at the native window, which Qwen recommends when you don't need long context. goose local edition starts the engine with these same flags and adds its own memory and cache settings. We typed this command out from the flags goose uses rather than running it exactly as shown, and `v0.14.3-lz.26` is the tag the current goose release pins.

Qwen's suggested sampling for thinking mode is temperature 1.0, top_p 0.95, top_k 20 (see the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for instruct mode and other advice).

## Limits

1. One agentic benchmark run came out lower than a third-party build. On LeanZero's Gauntlet agentic coding benchmark, this build scored 0.5932 in a single run at 524,288 context (YaRN factor 4, engine `v0.14.3-lz.25`). Jundot's build scored 0.6873 in an earlier single run at 360k context on an older engine (`lz.18`), and that is the Flash-Next entry you'll see on [leanzero.net/agentic-benchmarks](https://leanzero.net/agentic-benchmarks); we only replace an entry when a new run scores higher, so ours isn't listed. Before the benchmark's visible-output gate, ours scored higher (0.8848 against 0.8143). The run lost its points on one check: the 3D overview scene the model wrote rendered with a large black polygon over much of the view, so the visible-data check failed, and that failure caps the whole score at 0.599. That bug is in the code this one run wrote, not something we can trace to the quantization. The two runs also differ in context and engine, and with one run of each we can't separate the effect of quantization from ordinary run-to-run variation.
2. Recall at long context isn't perfect. The needle was missed once at 522,409 tokens with YaRN factor 2 (99.6% of that factor's window): the answer gave the surrounding paragraph but left the code out. With factor 4 it was missed once at about 540,000 tokens in a later test, the same way, while 530k in that session and 652k in another were found. Each step is one sample at temperature 1.0, so treat these as signs rather than rates.
3. This is a 4-bit build. The routed experts carry most of the parameters, and they are the part with the largest rounding error (median relative error 0.084).
4. Long prompts take time to prefill: about 9 to 14 minutes for a cold 455k to 522k token prompt on an M5 Max. Prefix caching makes follow-up turns much faster.
5. We haven't tested loading with plain mlx-vlm or any other engine, and we haven't run vision past the readiness checks.

## Licence

This model is a derivative of Qwen3.8-Flash-Next and is distributed under the Qwen Community License 1.0, the same licence as the original. The full text, copied unchanged from Qwen's repository, is in the [`LICENSE`](LICENSE) file (Copyright (c) 2026 Qwen). By using this model, you accept that licence. In particular:

1. The copyright and permission notice in `LICENSE` must stay with every copy or substantial portion you pass on.
2. If you use it in a commercial product or service with more than 100 million monthly active users or more than US$20 million in monthly revenue, the model name must be displayed prominently in that product's user interface.
3. If you or an affiliate run a Model as a Service business or an AI Work Assistant business (an independent product mainly for AI-assisted coding or office productivity), the licence requires a separate licence from Qwen before you use this model or its derivatives for any commercial purpose. Internal use is exempt as long as the model, its outputs and its capabilities aren't made available to third parties. Contact model-business@notice.qwencloud.com.

LeanZero shares this build free of charge. We don't sell it or run it as a hosted service, and our publishing it gives you no rights beyond the ones Qwen's licence gives you. The quantization recipe follows oMLX (Apache-2.0); the oMLX calibration data was used only to measure importance and isn't included here.

If this work helps you, please cite Qwen's technical report and blog post, as listed on the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## Files and checksums

These are the sha256 hashes of the files as we uploaded them. Hugging Face shows the same value as the LFS sha256 for each large file.

| file | sha256 |
|---|---|
| LICENSE | a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08 |
| chat_template.jinja | c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041 |
| config.json | 95c6e5c35fa8778d8f9129c566e218d32018fa7beed70455b8975e5c9b78d5ec |
| generation_config.json | e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e |
| merges.txt | a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d |
| model-00001-of-00021.safetensors | 54f8b8f2f8a9990a5d92c9fb126301c64d3645880268aba3ce16f789e6bdba79 |
| model-00002-of-00021.safetensors | bf2c3031bdf8b795a794c65d033ea5eb99e848c1375939040edbbddd1bdd83d4 |
| model-00003-of-00021.safetensors | cf184543974253ac56c5ecbf5a303554c7b4298dc6be92240263ad895c74cc89 |
| model-00004-of-00021.safetensors | 2625465a3ed7fe66c63ee3d83948d3f930bc9c298ca206ea2e1e2a9aa39ae9ef |
| model-00005-of-00021.safetensors | c0d060ceb9fdb1e4ec67bded4d44ffa76cbb8981fe50da1d5c9ea518aaad228f |
| model-00006-of-00021.safetensors | 3f491cf5b831debeb5bef92bbea350a0f610fc4b186c54dcf0fa1e0ef2a94aaa |
| model-00007-of-00021.safetensors | cc4beec29a1c3c7ecf2bfd71a8bc8614b3380a2bf7653aef9e9327c00a2a3899 |
| model-00008-of-00021.safetensors | 2752c1f5b9679bc177d5ad1b4ce56535f555c6ccba4c144f6fd7966be0680df7 |
| model-00009-of-00021.safetensors | 0443db126c10eb257b5ad6021da40931fc653196bfd61aa0cb9f052b125f020b |
| model-00010-of-00021.safetensors | b0e4f23ba44fd94e41fb3b3d7c180ca76b313daf6f9c09298d7be54f96504fe7 |
| model-00011-of-00021.safetensors | 010de0b8df6f11ca93f35642cb61114510eeef39b7ac7c97eaacea298cdbed1c |
| model-00012-of-00021.safetensors | 509d54b5d31f193dc9b684fbf5b8bea2759f2b9f4b0f14bd02eece4bc0ceb2a5 |
| model-00013-of-00021.safetensors | 7d305264eb969f1a3564d08c45eee5d8d1f53a88dbbf546c5b593cbe6a056d7c |
| model-00014-of-00021.safetensors | 54f43d745fa85054ad0c6c766da90a8e11bdb4a4f33a85974c982bde77f37692 |
| model-00015-of-00021.safetensors | e3d3c84d5db1b352071c582795dedc0d4cef9f538922df7347f9c89d912581cb |
| model-00016-of-00021.safetensors | 2da7c74b2b5bc7d097e25f75142a4b7b18af753e8f9df5390223f607542f750f |
| model-00017-of-00021.safetensors | 6428fd72eb7a856b09dc41b76d4ea5fb31739db1d8cebe0c57e55f24a551de29 |
| model-00018-of-00021.safetensors | c148474afeac1ffec437975e05c1ebfef8fd0cc15e44f11d7858cbda3021dd92 |
| model-00019-of-00021.safetensors | d55f3c63c31b19dccc565e2aead0e86437b3aa11726ce8dd90c04c76092229bc |
| model-00020-of-00021.safetensors | 804347f76c838487308967e160cc2c670b44307ec56a26def06825a005f7392d |
| model-00021-of-00021.safetensors | 19b2a6ec824ee58fd93ea7a9fcdc86f145af3df9667b7e604f931aa2f9a4ca0c |
| model.safetensors.index.json | d77e2cde73ea7e853c5857f22b0f7e74fb1269ba8aa170b5f97405a664affdd5 |
| own_convert_manifest.json | 434ae44aba64d11c924c25f717f2b7e9c5e32fc82ee0ce9d5005b89e224dd090 |
| preprocessor_config.json | 27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516 |
| tokenizer.json | 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 |
| tokenizer_config.json | b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27 |
| video_preprocessor_config.json | 7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13 |
| vocab.json | ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003 |

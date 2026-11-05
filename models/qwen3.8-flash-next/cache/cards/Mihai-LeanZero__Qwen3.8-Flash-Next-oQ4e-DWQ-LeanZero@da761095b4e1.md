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
- dwq
- mtp
- apple-silicon
---

# Qwen3.8-Flash-Next-oQ4e-DWQ-LeanZero

This is a 4-bit MLX build of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for Apple silicon Macs, made by [LeanZero](https://leanzero.net) from Qwen's official BF16 weights. It has the same size, the same 4-bit layout and the same 4-bit codes as our earlier build, [Mihai-LeanZero/Qwen3.8-Flash-Next-oQ4e-LeanZero](https://huggingface.co/Mihai-LeanZero/Qwen3.8-Flash-Next-oQ4e-LeanZero). The difference is that we retrained the scales and biases of the routed experts, layer by layer, so that each layer's output comes closer to Qwen's BF16 original. That kind of training is called distilled weight quantization (DWQ).

It keeps everything the original ships with: the language model, the 51B n-gram embedding, the vision encoder and the multi-token prediction (MTP) head. It is 100.275 GiB (107.67 GB) on disk and runs on a single 128 GB Mac. We tested it with the MLX engine behind Elzi, our desktop agent app. We're sharing it free of charge, as is, under Qwen's licence (see the licence section below).

## Where the weights come from

The starting point is our oQ4e build, which we quantized ourselves from the official repository [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. That build's card explains the conversion, the bit layout (oMLX's oQ4e recipe) and the checks against Qwen's files. For the training described below, we downloaded Qwen's BF16 shards again at the same revision and used them as the teacher. Nothing else went into the weights: no fine-tuning on outputs, no merging, no other model.

The chat template, tokenizer files, `generation_config.json`, the image and video preprocessor configs and the `LICENSE` file are byte-for-byte Qwen's files at that revision, as in the oQ4e build. `config.json` and `model.safetensors.index.json` are the same as in the oQ4e build. `own_convert_manifest.json` adds a `block_dwq` section with the training settings, the sha256 of the calibration data, which layers were trained or kept and with which input, and the validation result.

## What changed against the oQ4e build

1. Only the scales and biases of the routed experts (the 4-bit gate, up and down projections) changed, and only in 35 of the 48 decoder layers. In the other 13 layers (31, 32, 33, 36, 37, 39 to 46), training did not beat the original rounding on held-out data, so those layers keep the oQ4e tensors unchanged.
2. No 4-bit code changed. A full byte comparison of all 21 shards found identical headers, 208 changed scale and bias tensors and no other difference. Everything else (shared experts, attention, the hyper-connection mixers, the router, the n-gram table, the MTP head, the vision encoder) is the oQ4e build's, byte for byte.
3. Each layer was trained against the output of the same BF16 layer, over all four hyper-connection streams, with everything in front of the experts frozen. Settings: Adam, learning rate 1e-5, 3 epochs, fp32 master copies, 384 calibration sequences of 1,024 tokens. Layers 0 to 17 took their input from the quantized model and layers 18 to 47 from the BF16 teacher.
4. Calibration data, all public: agentic coding transcripts with tool calls (nebius/SWE-agent-trajectories, CC-BY-4.0), tool use (glaiveai/glaive-function-calling-v2, Apache-2.0), oMLX's code and multilingual calibration set (Apache-2.0) and general text (HuggingFaceFW/fineweb-edu, ODC-By), rendered through the model's own chat template. None of it comes from our benchmarks.

## How close it is to the original

We measured KL divergence from Qwen's BF16 model (lower is closer), agreement on the top-1 next token and perplexity, over 64 sequences of 1,024 tokens (65,472 scored tokens). This test set was built after training, from the same public sources as the training and validation data but from documents the training never saw, and we checked that no 32-token span in it also appears in the training or validation data. Half of it is tool-call text (agentic transcripts and function calling), the other half plain text (fineweb-edu and wikitext-2 validation articles).

| test rows | KL, oQ4e build | KL, this build | change (95% interval) | top-1 agreement, oQ4e -> this build |
|---|---|---|---|---|
| tool-call text (32) | 0.1122 | 0.0963 | -14.1% (-21.5% to -6.0%) | 92.87% -> 93.40% |
| plain text (32) | 0.0380 | 0.0380 | -0.1% (-2.8% to +2.4%) | 92.20% -> 92.38% |
| all (64) | 0.0751 | 0.0671 | -10.6% (-17.0% to -4.1%) | 92.53% -> 92.89% |

The intervals come from resampling the rows 1,000 times. So the gain is real on tool-call text and there is no measurable difference on plain prose. On the validation set used during training, the gain looked larger (-13.3%), but that set also chose the training settings, so the clean test above is the number we stand behind.

There is one thing we can't explain yet. On the tool-call rows, this build's perplexity moves slightly further away from the BF16 model's (3.796, against 3.706 for the oQ4e build and 3.692 for BF16), even though its KL divergence goes down. It matches the teacher's whole distribution more closely but puts a little less probability on the actual next tokens of those transcripts. On plain text the three perplexities are the same (7.305, 7.302 and 7.304).

## Measured on a Mac

These checks ran on one Mac Studio with an M5 Max and 128 GB of memory, with the macOS GPU wired limit raised to 120,832 MB, on LeanZero's MLX engine `v0.14.3-lz.27` with the n-gram table kept on disk, MTP with 2 draft tokens and a 524,288-token context (YaRN factor 4).

1. Vision, MTP and the n-gram table worked end to end. The model named the colour of three test images and read the text in a fourth correctly.
2. MTP accepted 454 of 501 drafted tokens (0.906) on a 700-token copy edit at temperature 1.0. The oQ4e build accepted 0.85 on the same test.
3. A needle test with a 522,913-token prompt found the hidden code on the first try. The engine peaked at 93.0 GiB of Metal memory and 98.5 GiB resident, and swap stayed flat. Time to first token was 970.8 s cold, and decode ran at 28.0 tokens per second.
4. At temperature 0, three standard prompts gave byte-identical output to the oQ4e build. On three fresh prompts the wording differed. We ran the Python and SQL answers from both builds and compared the JavaScript fix line by line: the answers were equally correct on all three.

## Benchmark

On LeanZero's Gauntlet agentic coding benchmark (sb-7.2), this build scored 0.6932 in one run on the Mac above, at 524,288 context on engine `lz.27` through our desktop app 3.0.151. That run first scored 0.5909. We then fixed a defect in the benchmark's screenshot checks, which misread a canvas placed on a fractional pixel boundary, and re-scored the same saved build on app 3.0.154. The result is the Qwen3.8-Flash-Next entry on [leanzero.net/agentic-benchmarks](https://leanzero.net/agentic-benchmarks/run/brun-63cdb444-61b5-465e-beba-806b74d23203). It replaces Jundot's build of the same model, which scored 0.6873 in an earlier single run at 360k context on an older engine. The benchmark asks for medium reasoning effort. Our desktop app sends that setting to the local engine, but we haven't verified that the engine applies it, so treat this score as not taken at a pinned effort. Our other run of this build, on the same Mac, engine and app version, scored 0.6606 (0.6605 under the fixed 3.0.154 scorer). Each run lost points on a different 3D rendering check, and those checks cap the score, so treat a single run as one sample, not a ranking.

### Against the cloud version on OpenRouter

That 0.6932 is 0.0086 ahead of the 0.6846 that the cloud version of this model, Qwen3.8 Flash through OpenRouter, scored on the same benchmark. OpenRouter lists `qwen/qwen3.8-flash` with Qwen/Qwen3.8-Flash-Next as its Hugging Face model, and Qwen describes Qwen3.8-Flash as the official version based on Qwen3.8-Flash-Next, with production features such as a 1M-token context. Both runs had the same task and the same 150-call budget. The lead is small, and the four points below the table can each change how you read it.

| entry | where it ran | when | Gauntlet sb-7.2 score | run page |
|---|---|---|---|---|
| this build (benchmark model ID `LeanZero/Qwen3.8-Flash-Next-oQ4e-dwq`) | one Mac Studio (M5 Max, 128 GB), engine `lz.27`, app 3.0.151 | run 3 November 2026, re-scored 4 November on app 3.0.154 | 0.6932 | [brun-63cdb444](https://leanzero.net/agentic-benchmarks/run/brun-63cdb444-61b5-465e-beba-806b74d23203) |
| `qwen/qwen3.8-flash`, the cloud version | OpenRouter, served by Alibaba | 5 October 2026 | 0.6846 | [brun-dce059c2](https://leanzero.net/agentic-benchmarks/run/brun-dce059c2-62b5-4028-894e-80a4e633b506) |

1. The cloud entry is one run. Ours is the better of the two runs described above, and the other one (0.6606) is below the cloud entry. In that run our engine also mangled 38 of the model's 150 tool calls, a parser fault fixed in `lz.28`. Two checks in our re-score also came out higher for reasons unrelated to the fix. With those set back, our score is 0.6924 and the lead is 0.0078.
2. The cloud run was graded on 5 October by an earlier revision of the sb-7.2 scorer. Its build files weren't kept, so it couldn't be re-scored on app 3.0.154 the way ours was. Its run page notes that the old scorer may have under-read its notifications in two checks.
3. The screenshot fix that lifted our run from 0.5909 mainly cleared the visible 3D surface check, which had wrongly held our run under a 0.599 cap. The cloud run never had that problem. It sits under a 0.699 cap because five of its 3D interaction checks failed, and the fix doesn't change those five. But scorer fixes from 2 November did change two of them, the stream and coast checks, and the cloud run was graded before that. If just one of the five passed in a re-score, its cap would rise to 0.709 and its score to about 0.695, which is above ours. We can't rule that out, so we don't know whether the cloud version would still come second today.
4. The benchmark asks for medium reasoning effort. Our desktop app sends that setting to our local engine, but we haven't yet verified that the engine applies it. The cloud run took place before 16 October, when our desktop app started sending the setting to Qwen models on OpenRouter, so it ran at the provider's default effort.

## Running it with MLX

This checkpoint uses Qwen's new `qwen4_exp` architecture (QSA sparse attention, Gated DeltaNet, n-gram embedding), stored in the mlx-vlm layout. Stock `mlx-lm` doesn't load this architecture. We've only tested it with LeanZero's MLX engine (a fork of Rapid-MLX), which serves an OpenAI-compatible API. The current release is a wheel, `0.14.3+lz.28`: it is `lz.27` plus a fix to the tool-call parser, and we checked this build on it before publishing.

You need `uv` (for `uvx`; `brew install uv` or see [docs.astral.sh/uv](https://docs.astral.sh/uv/)) and, for long context on a 128 GB Mac, a raised GPU wired limit. The `sysctl` setting needs sudo and resets when the Mac restarts.

```bash
pip install -U "huggingface_hub[cli]"
hf download Mihai-LeanZero/Qwen3.8-Flash-Next-oQ4e-DWQ-LeanZero --local-dir ~/models/Qwen3.8-Flash-Next-oQ4e-DWQ-LeanZero

sudo sysctl iogpu.wired_limit_mb=120832   # 128 GB Mac; only needed for long context

RAPID_MLX_SPLITK_DECODE=1 RAPID_MLX_QSA_BLOCK_SPARSE=1 \
uvx --from "leanzero-engine[mtp] @ https://downloads.leanzero.net/engine/leanzero_engine-0.14.3%2Blz.28-py3-none-any.whl" \
  rapid-mlx serve ~/models/Qwen3.8-Flash-Next-oQ4e-DWQ-LeanZero \
  --port 8000 \
  --qwen4-ple-offload \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --qwen4-rope-scaling yarn
```

`--qwen4-ple-offload` keeps the n-gram table (29.8 GiB) on disk and reads the rows each token needs, which is what lets the model fit on a 128 GB Mac. Put the model on the Mac's internal SSD, not a network or Thunderbolt share. `--qwen4-rope-scaling yarn` turns on YaRN factor 4 for contexts past the native 262,144 tokens; leave it off to stay at the native window. Our desktop app, Elzi, starts the engine with these same flags and adds its own memory and cache settings. We typed this command out from the flags the app uses rather than running it exactly as shown.

Qwen's suggested sampling for thinking mode is temperature 1.0, top_p 0.95, top_k 20 (see the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for other advice).

## Limits

1. The tool-call perplexity drift described above is not understood yet. If your use is mostly plain prose, this build and the oQ4e build measure the same.
2. The benchmark number is from one run. Our two scored runs of this build (0.6932 and 0.6606) are further apart than this build and the oQ4e build are on the clean test, because each run hit a different 3D check that caps the score. The lead over the cloud version (0.0086, see above) is smaller than that spread.
3. This is a 4-bit build. The routed experts carry most of the parameters and the largest rounding error, and in 13 of the 48 layers they are unchanged from the oQ4e build.
4. Long prompts take time to prefill: about 16 minutes for a cold 523k-token prompt on an M5 Max. Prefix caching makes follow-up turns much faster.
5. We haven't tested loading with plain mlx-vlm or any other engine, and we haven't run vision past the readiness checks.

## Licence

This model is a derivative of Qwen3.8-Flash-Next and is distributed under the Qwen Community License 1.0, the same licence as the original. The full text, copied unchanged from Qwen's repository, is in the [`LICENSE`](LICENSE) file (Copyright (c) 2026 Qwen). By using this model, you accept that licence. In particular:

1. The copyright and permission notice in `LICENSE` must stay with every copy or substantial portion you pass on.
2. If you use it in a commercial product or service with more than 100 million monthly active users or more than US$20 million in monthly revenue, the model name must be displayed prominently in that product's user interface.
3. If you or an affiliate run a Model as a Service business or an AI Work Assistant business (an independent product mainly for AI-assisted coding or office productivity), the licence requires a separate licence from Qwen before you use this model or its derivatives for any commercial purpose. Internal use is exempt as long as the model, its outputs and its capabilities aren't made available to third parties. Contact model-business@notice.qwencloud.com.

LeanZero shares this build free of charge. We don't sell it or run it as a hosted service, and our publishing it gives you no rights beyond the ones Qwen's licence gives you. The quantization recipe follows oMLX (Apache-2.0). The calibration data was used only for training and isn't included here.

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
| model-00001-of-00021.safetensors | fe10d16db77f408d1d507023d1b342dda0d87abe30065210b61683bbbd758738 |
| model-00002-of-00021.safetensors | bf2c3031bdf8b795a794c65d033ea5eb99e848c1375939040edbbddd1bdd83d4 |
| model-00003-of-00021.safetensors | cf184543974253ac56c5ecbf5a303554c7b4298dc6be92240263ad895c74cc89 |
| model-00004-of-00021.safetensors | 2625465a3ed7fe66c63ee3d83948d3f930bc9c298ca206ea2e1e2a9aa39ae9ef |
| model-00005-of-00021.safetensors | c0d060ceb9fdb1e4ec67bded4d44ffa76cbb8981fe50da1d5c9ea518aaad228f |
| model-00006-of-00021.safetensors | 3f491cf5b831debeb5bef92bbea350a0f610fc4b186c54dcf0fa1e0ef2a94aaa |
| model-00007-of-00021.safetensors | 21854e0c385dc9a7cffcc034bbccf5f8ff171c9201bf34e70b323b335dde5c89 |
| model-00008-of-00021.safetensors | e66c7aab895ab8023ad7956a6d71673970b4593fd669901357c337bf28414dc5 |
| model-00009-of-00021.safetensors | f133feaa9938d3ee3d756d7762c3436239ea3d9dad16325c8dfd630615469844 |
| model-00010-of-00021.safetensors | e4c4ecbfd27ddb3d242b603ecf58fd1891172de4e64ae9d6a10f243e0d344936 |
| model-00011-of-00021.safetensors | 981634567f372c645f60370d01739c3f9380d411030d2e2569c79cc5f1a847eb |
| model-00012-of-00021.safetensors | 3aea9cb792fc0742461ded880f850f7aa040c385e90ad42283c31fcd77e98a04 |
| model-00013-of-00021.safetensors | 2cc4247c107c1173ee48b985aa7ea2ff8d30af77b2034fae25f20606fde91c0a |
| model-00014-of-00021.safetensors | a62d6282e83bcd78a4ac15bfd93c33b1569a19f94be7c219a45d662d1de98e8b |
| model-00015-of-00021.safetensors | 74edcf40cd7e54ae0090b39f0ca444449fca84d51303e45911495d6fdeba0681 |
| model-00016-of-00021.safetensors | 820f95affee054303cac8d1b59349cdc81bfea57961b7484d800b0218b57ebac |
| model-00017-of-00021.safetensors | 720e9b30b7c3046e0ae1afd171672f6951e6be7a77d19a3cc5e9d4bfa17671f6 |
| model-00018-of-00021.safetensors | c148474afeac1ffec437975e05c1ebfef8fd0cc15e44f11d7858cbda3021dd92 |
| model-00019-of-00021.safetensors | 8ba42352addb7805a49888b99efc86658c9a74b69c9b88ad9fd21c0bdbac8627 |
| model-00020-of-00021.safetensors | 418a89ae15270f4957c96c827024bc66cea0304ebbe21da362613990cd9c6e63 |
| model-00021-of-00021.safetensors | 87bac5bcb908b25c3f010ad0e566afc9d629324e3f8f4a6f05158cacb7454dec |
| model.safetensors.index.json | d77e2cde73ea7e853c5857f22b0f7e74fb1269ba8aa170b5f97405a664affdd5 |
| own_convert_manifest.json | e33206c2366e57cf50da55f1d614f3d21f9861ad2fbb981835faa43f4631c96f |
| preprocessor_config.json | 27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516 |
| tokenizer.json | 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 |
| tokenizer_config.json | b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27 |
| video_preprocessor_config.json | 7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13 |
| vocab.json | ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003 |

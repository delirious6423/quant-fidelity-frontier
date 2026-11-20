---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: gguf
tags:
- gguf
- llama.cpp
- deepseek
- code
---

![DeepSeek V4.1 Flash Coding](assets/banner.png)

We introduce DeepSeek V4.1 Flash Coding from [Nace.AI](https://www.nace.ai/): the 763B [DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) in a 71.8 GB GGUF.

## Overview

DeepSeek V4.1 Flash Coding is a Nace Specialized Quant for coding and agentic tasks. It is built and tuned on real code and agent work, and it runs on a single 48 GB Mac with llama.cpp.

- 71.8 GB for a 763B-parameter model
- 37 of 38 LiveCodeBench v6 problems solved

## Results

![Held-out code KL](assets/results.png)

![LiveCodeBench problems solved](assets/lcb.png)

- **71.8 GB** for a 763B-parameter model.
- **0.140 KL** to a near-lossless Q8 reference on real code, against 0.161 for EXL3 at 124 GB.
- **88.4%** of next tokens match the Q8 reference.
- **37 of 38** LiveCodeBench v6 problems solved, level with ISTA Coder on the same two seeds.

![How it is built](assets/pipeline.png)

## Coding-first calibration

Built from real coding work: agents fixing real repositories, contest solutions and code commits. Every choice in the file follows how DeepSeek routes code.

![Code calibration](assets/calib.png)

## Code-aware expert pruning

DeepSeek V4.1 Flash has 384 experts in each of its 40 layers. We keep the 200 per layer that code uses most. They carry 90% of the model's code routing.

![Kept experts per layer](assets/keep.png)

## Mixed precision per layer

Every expert is scored by how often it fires, how much the router relies on it and how much rounding costs it. The 16 layers where code needs it most get larger, higher-precision experts.

![Bits per layer](assets/bits.png)

## Nace Weight Space Quantization

**An agent that edits the weights themselves.**

Nace Weight Space Quantization puts an AI agent (Claude Fable 5.2) to work on a finished quantized file. With reading, navigation, quantization and checking tools, it reshapes how weights are stored, rebalances precision toward the layers coding leans on, and prunes and swaps experts toward what coding work uses.

![Nace Weight Space Quantization](assets/wsq.png)

Every edit is measured against a near-lossless Q8 reference, and only edits that score better stay. The file in this repo went through all four steps: the agent's edits cut held-out code KL from 0.142 to 0.140 at the same 71.8 GB.

## Run it

The download is the 71.8 GB main weights plus the 110.6 GB Engram table in `engram/`.

```bash
hf download nace-ai/DeepSeek-V4.1-Flash-Coding-GGUF --local-dir coding
llama-server -m coding/DeepSeek-V4.1-Flash-1017-Coding-IQ1_S-200E.gguf \
  --ple coding/engram/DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf_file -c 32768
```

Use temperature 1.0 and top_p 1.0.

## License

MIT, the same license as [DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). The full license text is in `LICENSE`.

---
license: apache-2.0
language:
- en
- it
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: text-generation
tags:
- gguf
- moe
- conversational
- experimental
---

# ComRAD 200B
### Boreale · Experimental GGUF preview

ComRAD 200B is an experimental Mixture-of-Experts language model with approximately 195 billion parameters, derived from Z.ai's GLM-5.3-Flash through the proprietary ComRAD weight transformation methodology, without additional training or fine-tuning.

A project by **Federico Alessandro, 17**, under **Mindral**.

**ComRAD 200B** is the public model name. The exact parameter count is **194,927,503,582**. In conversation, the model identifies itself as **ComRAD Boreale**.

| Property | This checkpoint |
|---|---|
| Parameters | 194.93 billion |
| Architecture | Mixture-of-Experts |
| Format | GGUF, Q4_K |
| File size | 114,333,049,568 bytes — about 114.33 GB |
| Interface | Text generation and conversation |
| Tested context | 4,096 tokens |
| License | Apache 2.0 for ComRAD contributions; upstream MIT notice retained |

## Try the checkpoint

Download `ComRAD-Boreale-195B-Q4_K.gguf` and import it into a runtime that supports this model's architecture. Use its embedded chat template.

This checkpoint was tested with **LM Studio's llama.cpp CUDA runtime 2.66.0**. The tested machine had an RTX PRO 6000 with 96 GB VRAM and approximately 188 GB system RAM; part of the model was offloaded to system RAM. This is a tested configuration, not a measured minimum hardware requirement. Runtime memory use exceeds the weight file size.

For the configuration used in the current chat:

- Context length: 4,096 tokens.
- Maximum generated tokens: 1,536, including reasoning.
- Temperature: 0; top-p: 1; top-k: 0; min-p: 0.
- Repetition penalty: 1.0; frequency and presence penalties: 0.
- Use the embedded template, which defaults to High reasoning.

See `inference_config.json` for the exact tested settings. Offload values depend on the machine and runtime. Short generation limits may cut off reasoning before the final answer.

The GGUF is standalone for inference; the upstream GLM checkpoint is not required at runtime. No proprietary ComRAD chat service is required to load it in a compatible runtime.

## Evaluation so far

These are **small functional checks**, not standardized benchmark scores.

| Check | Evidence |
|---|---|
| Identity | Returned "ComRAD Boreale" and ended naturally in the served configuration |
| C++17 code example | Compiled with g++ using `-std=c++17 -O2 -Wall -Wextra -pedantic`, without diagnostics |
| Functional checks for that C++ example | 12/12 cases passed, including empty input, negative integers, repeated values, original ordering and integer extremes |
| Eight informal reasoning and instruction-following questions | All passed according to the author; complete transcripts were not independently reviewed |
| Conversation history | The author reported successful use of prior messages within the same conversation |

The C++ task retained only values appearing exactly once while preserving their original order. The 12 cases evaluate one generated program, not 12 independent programming tasks.

## Limitations

This is an experimental preview. Earlier targeted tests observed scientific inaccuracies, coding errors and outputs that exceeded a short generation budget. Passing the current examples does not establish that these failures are resolved for other prompts.

No broad benchmark or quality-parity claim with the source model is made. Other runtimes, operating systems, context lengths and hardware configurations have not been validated by this test session.

Conversation history is supplied by the chat application. Persistent memory between separate conversations is not included.

## Release integrity

This release freezes the checkpoint tested on 13 November 2026. The complete saved GGUF was reread and its SHA-256 verified.

`SHA256SUMS` and `manifest.json` identify the exact artifact. Future experiments should use a separate version.

## License and attribution

ComRAD contributions are distributed under [Apache License 2.0](LICENSE). The upstream GLM-5.3-Flash MIT license and copyright notice are retained in [LICENSE.upstream](LICENSE.upstream); see [NOTICE](NOTICE).

Apache 2.0 permits commercial use, modification and redistribution subject to its terms. No separate non-commercial or no-resale condition is imposed on this release.

Base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`.

The proprietary ComRAD methodology, compiler and internal development tooling are not included in this repository.

---
base_model: Qwen/Qwen3.8-Flash-Next
language:
- ko
- en
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
tags:
- gguf
- llama.cpp
- moe
- expert-pruning
- gsq
- quantization
- IQ3_S
- korean
- code
---

# Qwen3.8-Flash-Next Ko-Coder — 256 experts, IQ3_S

**Update, 2026-10-29:** the current files use ISTA-DASLab's **IQ3_S mixed-format weight recipe**,
pruned from 512 to 256 routed experts per layer, plus the same n-gram table as this repository's previous KD2 build.
The selected combination is `mix`: the saved expert selection fitted on an equal mixture of Open-SWE and the original calibration tokens.
It scored **22/32 (68.75%)** on the saved Aider Polyglot subset.
This is a 32-task subset result, not a full 225-task Polyglot score.

Shard-1 codes, block scales, dense weights and retained router rows are copied from the downloaded
[ISTA IQ3_S build](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/IQ3_S),
keeping the selected experts in their original order. **No new quantization training or KD was applied to these weights.**
The expert indices come from the saved pruning experiment; the n-gram table comes from the earlier custom build.
The previous model's distilled router values, row multipliers and norm gains are not carried into this IQ3_S shard 1.

## 한국어 소개

전문가를 층마다 512개에서 256개로 줄인 모델입니다. 이번 배포는 ISTA의 **IQ3_S 가중치 구성**을 사용하며,
전문가 선택은 `mix` 조합입니다. 두 3비트 후보를 같은 32문제로 비교해 통과 수가 높은 쪽을 골랐습니다.
동점이면 기존 배포 모델과 같은 전문가 선택을 유지합니다. n-gram 테이블은 두 후보 모두 기존 KD2 것으로 통일했습니다.
새로 양자화하거나 KD한 결과가 아니며, 기존 2비트 KD2의 한국어·지식 벤치 점수를 이번 모델의 점수로 볼 수는 없습니다.
IQ3_S는 모든 텐서가 똑같이 3비트라는 뜻이 아니라, 텐서별 자료형이 섞인 양자화 구성 이름입니다.

## Files

Keep both GGUF shards together, with their filenames unchanged.

| File | Size | Content |
|---|---:|---|
| `Qwen3.8-Flash-Next-gumbel256-IQ3_S-00001-of-00002.gguf` | 29.61 GB | Selected 256-expert IQ3_S model |
| `Qwen3.8-Flash-Next-gumbel256-IQ3_S-00002-of-00002.gguf` | 28.80 GB | KD2 n-gram table, IQ4_NL |
| `Qwen3.8-Flash-Next-gumbel256-IQ3_S.experts.json` | small | Original retained expert indices for each layer |
| `BENCHMARK-2026-10-29.md` | small | All compared models and per-task outcomes |
| `release.json` | small | Selected combination, scores and provenance |

## Quantization types

The current shard 1 retains ISTA's IQ3_S per-tensor formats and bytes; only routed-expert arrays and router rows
are sliced. The n-gram table is the existing custom IQ4_NL table, unchanged between the two new candidates.

| Group | Parameters | Actual GGUF types (tensor counts) | Tensor bytes |
|---|---:|---|---:|
| Routed experts | 60.40B | IQ2_S (40), IQ3_S (20), IQ3_XXS (34), IQ4_NL (39), IQ4_XS (2), Q2_0 (9) | 25.15 GB |
| Other shard-1 tensors | 4.92B | BF16 (484), F16 (1), F32 (292), IQ4_NL (47), IQ4_XS (43), Q4_K (47), Q5_K (35), Q6_K (129), Q8_0 (1) | 4.45 GB |
| n-gram table (shard 2) | 51.20B | IQ4_NL | 28.80 GB |

## Evaluation

Aider 0.86.2 Polyglot harness, the same **32 saved exercises** across C++, Go, Java, JavaScript, Python and Rust:
diff edit format, thinking off, maximum output 16,384 tokens, 2 tries, 2 concurrent tasks.
Strata HIP engine on a Radeon AI PRO R9700; native mixed-format weights, mmap experts, int8 KV and the existing MTP draft.
The two new IQ3_S models differ in saved expert selection and use the same KD2 n-gram table.

| Model | Passed within 2 tries | First try |
|---|---:|---:|
| Previous custom 2-bit KD2 | 14/32 (43.75%) | 8/32 |
| ISTA Q2_0, old selection, KD2 n-gram | 12/32 (37.50%) | 6/32 |
| ISTA Q2_0, mixed selection, Unsloth n-gram | 13/32 (40.62%) | 8/32 |
| IQ3_S, old selection, KD2 n-gram | 18/32 (56.25%) | 9/32 |
| IQ3_S, mixed selection, KD2 n-gram | 22/32 (68.75%) | 9/32 |

The subset was determined by the completion order of a previously interrupted 225-task run, not a random sample.
Selection of the winner uses this same small subset; these results are not an independent validation set.
The historical 2-bit mixed-selection run used a different n-gram table, so its difference includes that change.
The IQ3_S recipe also changes dense tensor formats/values, not only expert precision. Small score differences
do not establish statistical significance. Interactive dsh requests shared the inference server during part of the run;
do not use these runs as controlled throughput measurements.

No new Korean, perplexity, HumanEval, GSM8K or other evaluations were run for the IQ3_S release.
The old KD2 evaluation tables and training details remain in the
[previous model card](https://huggingface.co/attention-is-all-i-want/Qwen3.8-Flash-Next-GSQ-RCO-Ko-Coder/blob/3bfe6498c44ab1063059c615974a9378524cbfd6/README.md).
The previous 2-bit GGUF files are accessible at that revision as well.

## Usage

Use a llama.cpp or Strata build that supports this model architecture and all mixed GGUF quantization types.
The n-gram table should be on fast local storage. A llama.cpp example (adjust GPU offload and context to available memory):

```sh
llama-server -m Qwen3.8-Flash-Next-gumbel256-IQ3_S-00001-of-00002.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768 \
  -ot 'per_layer_token_embd\.weight=CPU' \
  --repeat-penalty 1.0 --chat-template-kwargs '{"enable_thinking": false}'
```

The benchmark used Strata with a 262,144-token context limit; the example above is not an additional llama.cpp validation.

## License and data


Derived from Qwen3.8-Flash-Next under the [Qwen Community License 1.0](LICENSE). The IQ3_S shard-1 weights and per-tensor allocation come from
ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (Apache-2.0). The n-gram shard is retained from this repository's previous KD2 build. The calibration data includes
outputs of Claude (Anthropic) from the author's own coding sessions and the original model's agent traces from
Vemloir/Qwen3.8-Flash-Next-traces: this model contains information from
[Vemloir/Qwen3.8-Flash-Next-traces](https://huggingface.co/datasets/Vemloir/Qwen3.8-Flash-Next-traces), which is made
available under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/).

Other text in the calibration and distillation sets, under its own terms:
[ProCreations/Qwen3.8-Flash-Next-DSpine-generations](https://huggingface.co/datasets/ProCreations/Qwen3.8-Flash-Next-DSpine-generations)
(prompts under their per-source licenses, generations of Qwen3.8-Flash-Next),
[nayohan/CodeFeedback-Filtered-Instruction-ko](https://huggingface.co/datasets/nayohan/CodeFeedback-Filtered-Instruction-ko),
[dbdu/ShareGPT-74k-ko](https://huggingface.co/datasets/dbdu/ShareGPT-74k-ko) (CC BY 2.0),
[nlpai-lab/kullm-v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2) (Apache-2.0) and Korean Wikipedia
([wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.ko`, CC BY-SA 3.0 / GFDL).
In distillation this text is only the context: the targets are the unpruned Qwen3.8-Flash-Next's own distributions.


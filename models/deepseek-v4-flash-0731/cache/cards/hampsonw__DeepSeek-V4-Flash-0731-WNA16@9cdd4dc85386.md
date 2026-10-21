---
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: hampsonw
tags:
  - deepseek-v4
  - compressed-tensors
  - wna16
  - int2
  - experimental
  - withdrawn
  - broken-quant
---

# Withdrawn: DeepSeek V4 Flash 0731 routed-expert W2A16

> [!WARNING]
> **This quant failed DeepSWE coding tasks and is withdrawn. Do not use it for
> coding agents.**
>
> It failed with and without concurrent requests. The patched vLLM runtime
> worked, but this model did not. The files remain available for research and
> reproduction.

This is an experimental, MTP-free quantization of
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
Its routed experts use W2A16, while other tensors retain their source values and
dtypes. It was built to test whether a roughly 77 GiB safetensors checkpoint
could run through vLLM and Humming on four 24 GB RTX 3090 GPUs.

The checkpoint loads and generates with a patched research runtime. It does not
work with stock vLLM. Successful serving did not translate into acceptable
coding-agent behavior.

## What changed

- Every routed-expert `w1`, `w3`, and `w2` matrix uses symmetric W2A16
  quantization with group size 128.
- Packed weights use the compressed-tensors 0.17.0 WNA16 format with FP16 group
  scales and INT32 packed values.
- Non-routed tensors retain their source values and dtypes, including the native
  DeepSeek FP8 attention and shared-expert weights.
- The MTP/DSpark draft layers are omitted. `num_nextn_predict_layers` is zero.
- The resulting checkpoint contains 45 safetensors shards. The three omitted
  source shards contained only MTP tensors.

## How it was created

The conversion used the official checkpoint at revision
[`7872f01b1d1fe23eabc4c98b48bffcef5a386062`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062).
It processed one source shard at a time and wrote checksum-bound, resumable
safetensors shards.

For each routed-expert matrix, the converter:

1. used Intel AutoRound's DeepSeek V4 path to decode the source MXFP4/E8M0
   representation;
2. fitted symmetric 2-bit weights with group size 128;
3. weighted the scale search with Antirez's routed-expert activation imatrix;
4. packed the signed codes with compressed-tensors 0.17.0; and
5. recorded weighted and unweighted reconstruction error in
   `conversion-metrics.json`.

The conversion selected imatrix-weighted RTN after a bounded comparison on 24
matrices from layers 0, 26, 37, and 42. It improved weighted reconstruction
error in all 24 comparisons. The median improvement was 31.03% relative to
plain RTN. This result measures reconstruction error, not end-to-end model
quality.

The Antirez imatrix predates the 0731 checkpoint. Antirez later reused it for
0731 GGUFs, and the 24-matrix comparison showed that it improved this WNA16
conversion, but it is not a 0731-native calibration run.

### Pinned inputs and tools

| Component | Revision or version |
| --- | --- |
| Base checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Routed-expert imatrix | `antirez/deepseek-v4-gguf@e7f04037032990db0346398d249baf9fb9df1ccc` |
| Imatrix content SHA-256 | `02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e` |
| AutoRound | `intel/auto-round@f17d9cd4b36982006bad21ff87127aac739072e3` |
| compressed-tensors | `0.17.0` |
| Humming | `humming-kernels==0.1.10`, source `inclusionAI/humming@4351af3a8fcdce1a8dee50104ba49566af2427fb` |
| Conversion and runtime record | [`Whamp/club-3090@357661d6`](https://github.com/Whamp/club-3090/tree/357661d633688076b30046645ee43b0baed912b7) |

The full conversion ran on one A100 80 GB VM and used 112.9 GiB peak host
memory with no swap.

The immutable weight snapshot is
[`75d9286c37f3037f3ab390cfbc10747466eac714`](https://huggingface.co/hampsonw/DeepSeek-V4-Flash-0731-WNA16/tree/75d9286c37f3037f3ab390cfbc10747466eac714).
It contains 54 files totaling 82,464,249,582 bytes, including 45 model shards.

## Runtime compatibility

This checkpoint requires the experimental code in
[Whamp/vllm#1](https://github.com/Whamp/vllm/pull/1), based on
[`haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209`](https://github.com/haosdent/vllm/tree/12810046c799cbe874967e19b1c0fa134ab7b209).
Club-3090 keeps a checksum-pinned deployment copy of the patches at
[`Whamp/club-3090@357661d6`](https://github.com/Whamp/club-3090/tree/357661d633688076b30046645ee43b0baed912b7/models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86).
The final tested vLLM tree was
`aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`.

The patches provide:

- compressed-tensors W2 Humming MoE loading;
- separate routed-expert and native DeepSeek FP8 handling;
- DeepSeek V4 execution on SM86; and
- RTX 3090 sparse-attention fallbacks for kernels that exceed SM86 shared
  memory.

Stock vLLM does not implement this complete path. Do not expect
`vllm serve hampsonw/DeepSeek-V4-Flash-0731-WNA16` to work in an unpatched
environment.

## Historical runtime result

The final runtime test used four RTX 3090 GPUs with tensor parallelism 4. It:

- loaded all 45 shards;
- served a 215,000-token maximum context with `max_num_seqs=4`;
- measured about 60.8 single-stream decode tokens/s;
- measured about 969 prefill tokens/s;
- retrieved the correct value from a 204,900-token prompt; and
- passed short-request concurrency checks with two and four requests.

Only about 141–142 MiB remained free on each GPU. These results show that the
runtime path works. They do not establish model quality.

## Quality result

DeepSWE ran the same coding task twice: first with concurrent serving, then with
one worker and no other requests. Both runs failed the same way. The model kept
rereading files, never edited or tested the code, and produced no patch. The
second run rules out concurrency as the cause.

Other DeepSeek V4 versions completed the task:

- IQ2_XXS produced a 22.5 KB patch, passed 78 of 80 feature tests, and preserved
  all 116 existing tests.
- Three FP8 runs produced 35–51 KB patches and passed 90–92.5% of the feature
  tests.

This quant is withdrawn. The vLLM runtime remains useful for testing future
quants.

Other limits remain:

- MTP/DSpark is absent.
- The artifact uses one uniform W2 recipe for every routed projection.
- The calibration imatrix came from the pre-0731 model.
- Runtime support depends on research patches and JIT-compiled Humming kernels.

## Reproduction

The converter, pilot, resumable writer, upload verifier, runtime patches, and
research record live in the
[`feat/deepseek-v4-lowbit-vllm`](https://github.com/Whamp/club-3090/tree/feat/deepseek-v4-lowbit-vllm)
branch of club-3090. Start with:

- [`tools/deepseek-v4-lowbit/README.md`](https://github.com/Whamp/club-3090/blob/feat/deepseek-v4-lowbit-vllm/tools/deepseek-v4-lowbit/README.md)
- [the experiment record](https://github.com/Whamp/club-3090/blob/feat/deepseek-v4-lowbit-vllm/.research/deepseek-v4-lowbit/PLAN.md)
- [the vLLM patch series](https://github.com/Whamp/club-3090/tree/feat/deepseek-v4-lowbit-vllm/models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86)

## Credits

- DeepSeek created and released DeepSeek-V4-Flash-0731.
- Antirez created the routed-expert imatrix used to guide this quantization.
- Intel AutoRound provided the source dequantization and RTN scale-search
  primitives.
- Neural Magic and the vLLM project provide the compressed-tensors format and
  loader infrastructure.
- InclusionAI provides the Humming WNA16 kernels.
- Haosdent and Lasimeri developed the Ampere DeepSeek V4 work that made the
  runtime experiment possible.

## License

The base checkpoint and this redistributed quantization use the
[MIT License](https://huggingface.co/hampsonw/DeepSeek-V4-Flash-0731-WNA16/blob/main/LICENSE).
DeepSeek's copyright and permission notice are included in this repository.

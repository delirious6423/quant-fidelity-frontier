---
base_model: Qwen/Qwen3.8-Flash-Next
language:
- en
library_name: transformers
license: other
license_link: LICENSE
license_name: qwen-community-1.0
mradermacher:
  readme_rev: 1
quantized_by: mradermacher
---
## About

<!-- ### quantize_version: 2 -->
<!-- ### output_tensor_quantised: 1 -->
<!-- ### convert_type: hf -->
<!-- ### vocab_type:  -->
<!-- ### tags: nicoboss -->
<!-- ### quants:  Q2_K IQ3_M Q4_K_S IQ3_XXS Q3_K_M small-IQ4_NL Q4_K_M IQ2_M Q6_K IQ4_XS Q2_K_S IQ1_M Q3_K_S IQ2_XXS Q3_K_L IQ2_XS Q5_K_S IQ2_S IQ1_S Q5_K_M Q4_0 IQ3_XS Q4_1 IQ3_S -->
<!-- ### quants_skip:  -->
<!-- ### skip_mmproj:  -->
weighted/imatrix quants of https://huggingface.co/Qwen/Qwen3.8-Flash-Next

<!-- provided-files -->

***For a convenient overview and download list, visit our [model page for this model](https://hf.tst.eu/model#Qwen3.8-Flash-Next-i1-GGUF).***

static quants are available at https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-GGUF

**This is a vision model - mmproj files (if any) will be in the [static repository](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-GGUF).**
## Usage

If you are unsure how to use GGUF files, refer to one of [TheBloke's
READMEs](https://huggingface.co/TheBloke/KafkaLM-70B-German-V0.1-GGUF) for
more details, including on how to concatenate multi-part files.

## Provided Quants

(sorted by size, not necessarily quality. IQ-quants are often preferable over similar sized non-IQ quants)

| Link | Type | Size/GB | Notes |
|:-----|:-----|--------:|:------|
| [GGUF](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.imatrix.gguf) | imatrix | 0.7 | imatrix file (for creating your own quants) |
| [GGUF](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q2_K.gguf) | i1-Q2_K | 80.5 | IQ3_XXS probably better |
| [GGUF](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q3_K_S.gguf) | i1-Q3_K_S | 88.9 | IQ3_XS probably better |
| [GGUF](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-IQ3_S.gguf) | i1-IQ3_S | 89.0 | beats Q3_K* |
| [GGUF](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-IQ3_M.gguf) | i1-IQ3_M | 89.6 |  |
| [GGUF](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q3_K_M.gguf) | i1-Q3_K_M | 94.2 | IQ3_S probably better |
| [GGUF](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q3_K_L.gguf) | i1-Q3_K_L | 96.8 | IQ3_M probably better |
| [GGUF](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-IQ4_XS.gguf) | i1-IQ4_XS | 97.6 |  |
| [PART 1](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_0.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_0.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_0.gguf.part3of3) | i1-Q4_0 | 100.4 | fast, low quality |
| [PART 1](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_1.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_1.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_1.gguf.part3of3) | i1-Q4_1 | 111.1 |  |
| [PART 1](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_K_S.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_K_S.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_K_S.gguf.part3of3) | i1-Q4_K_S | 111.9 | optimal size/speed/quality |
| [PART 1](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_K_M.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_K_M.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q4_K_M.gguf.part3of3) | i1-Q4_K_M | 119.3 | fast, recommended |
| [PART 1](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q5_K_S.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q5_K_S.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q5_K_S.gguf.part3of3) | i1-Q5_K_S | 127.8 |  |
| [PART 1](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q5_K_M.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q5_K_M.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q5_K_M.gguf.part3of3) | i1-Q5_K_M | 134.2 |  |
| [PART 1](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q6_K.gguf.part1of4) [PART 2](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q6_K.gguf.part2of4) [PART 3](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q6_K.gguf.part3of4) [PART 4](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-i1-GGUF/resolve/main/Qwen3.8-Flash-Next.i1-Q6_K.gguf.part4of4) | i1-Q6_K | 167.7 | practically like static Q6_K |

Here is a handy graph by ikawrakow comparing some lower-quality quant
types (lower is better):

![image.png](https://www.nethype.de/huggingface_embed/quantpplgraph.png)

And here are Artefact2's thoughts on the matter:
https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9

## FAQ / Model Request

See https://huggingface.co/mradermacher/model_requests for some answers to
questions you might have and/or if you want some other model quantized.

## Thanks

I thank my company, [nethype GmbH](https://www.nethype.de/), for letting
me use its servers and providing upgrades to my workstation to enable
this work in my free time. Additional thanks to [@nicoboss](https://huggingface.co/nicoboss) for giving me access to his private supercomputer, enabling me to provide many more imatrix quants, at much higher quality, than I would otherwise be able to.

<!-- end -->

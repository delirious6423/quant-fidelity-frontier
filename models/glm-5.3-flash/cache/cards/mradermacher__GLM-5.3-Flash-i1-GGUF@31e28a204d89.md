---
base_model: zai-org/GLM-5.3-Flash
language:
- en
- zh
library_name: transformers
license: mit
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
weighted/imatrix quants of https://huggingface.co/zai-org/GLM-5.3-Flash

<!-- provided-files -->

***For a convenient overview and download list, visit our [model page for this model](https://hf.tst.eu/model#GLM-5.3-Flash-i1-GGUF).***

static quants are available at https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF

**This is a vision model - mmproj files (if any) will be in the [static repository](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF).**
## Usage

If you are unsure how to use GGUF files, refer to one of [TheBloke's
READMEs](https://huggingface.co/TheBloke/KafkaLM-70B-German-V0.1-GGUF) for
more details, including on how to concatenate multi-part files.

## Provided Quants

(sorted by size, not necessarily quality. IQ-quants are often preferable over similar sized non-IQ quants)

| Link | Type | Size/GB | Notes |
|:-----|:-----|--------:|:------|
| [GGUF](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.imatrix.gguf) | imatrix | 1.1 | imatrix file (for creating your own quants) |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q2_K.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q2_K.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q2_K.gguf.part3of3) | i1-Q2_K | 118.0 | IQ3_XXS probably better |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ3_S.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ3_S.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ3_S.gguf.part3of3) | i1-IQ3_S | 139.5 | beats Q3_K* |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_S.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_S.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_S.gguf.part3of3) | i1-Q3_K_S | 139.5 | IQ3_XS probably better |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ3_M.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ3_M.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ3_M.gguf.part3of3) | i1-IQ3_M | 140.5 |  |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_M.gguf.part1of4) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_M.gguf.part2of4) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_M.gguf.part3of4) [PART 4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_M.gguf.part4of4) | i1-Q3_K_M | 153.8 | IQ3_S probably better |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_L.gguf.part1of4) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_L.gguf.part2of4) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_L.gguf.part3of4) [PART 4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q3_K_L.gguf.part4of4) | i1-Q3_K_L | 167.2 | IQ3_M probably better |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ4_XS.gguf.part1of4) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ4_XS.gguf.part2of4) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ4_XS.gguf.part3of4) [PART 4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-IQ4_XS.gguf.part4of4) | i1-IQ4_XS | 171.9 |  |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_0.gguf.part1of4) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_0.gguf.part2of4) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_0.gguf.part3of4) [PART 4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_0.gguf.part4of4) | i1-Q4_0 | 182.2 | fast, low quality |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_K_S.gguf.part1of4) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_K_S.gguf.part2of4) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_K_S.gguf.part3of4) [PART 4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_K_S.gguf.part4of4) | i1-Q4_K_S | 182.5 | optimal size/speed/quality |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_K_M.gguf.part1of5) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_K_M.gguf.part2of5) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_K_M.gguf.part3of5) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_K_M.gguf.part4of5) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_K_M.gguf.part5of5) | i1-Q4_K_M | 193.9 | fast, recommended |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_1.gguf.part1of5) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_1.gguf.part2of5) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_1.gguf.part3of5) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_1.gguf.part4of5) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q4_1.gguf.part5of5) | i1-Q4_1 | 201.8 |  |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_S.gguf.part1of5) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_S.gguf.part2of5) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_S.gguf.part3of5) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_S.gguf.part4of5) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_S.gguf.part5of5) | i1-Q5_K_S | 221.8 |  |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_M.gguf.part1of5) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_M.gguf.part2of5) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_M.gguf.part3of5) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_M.gguf.part4of5) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q5_K_M.gguf.part5of5) | i1-Q5_K_M | 228.0 |  |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q6_K.gguf.part1of6) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q6_K.gguf.part2of6) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q6_K.gguf.part3of6) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q6_K.gguf.part4of6) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q6_K.gguf.part5of6) [P6](https://huggingface.co/mradermacher/GLM-5.3-Flash-i1-GGUF/resolve/main/GLM-5.3-Flash.i1-Q6_K.gguf.part6of6) | i1-Q6_K | 264.1 | practically like static Q6_K |

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

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
<!-- ### tags:  -->
<!-- ### quants:  x-f16 Q4_K_S Q2_K Q8_0 Q6_K Q3_K_M Q3_K_S Q3_K_L Q4_K_M Q5_K_S Q5_K_M IQ4_XS -->
<!-- ### quants_skip:  -->
<!-- ### skip_mmproj:  -->
static quants of https://huggingface.co/zai-org/GLM-5.3-Flash

<!-- provided-files -->

***For a convenient overview and download list, visit our [model page for this model](https://hf.tst.eu/model#GLM-5.3-Flash-GGUF).***

weighted/imatrix quants seem not to be available (by me) at this time. If they do not show up a week or so after the static ones, I have probably not planned for them. Feel free to request them by opening a Community Discussion.
## Usage

If you are unsure how to use GGUF files, refer to one of [TheBloke's
READMEs](https://huggingface.co/TheBloke/KafkaLM-70B-German-V0.1-GGUF) for
more details, including on how to concatenate multi-part files.

## Provided Quants

(sorted by size, not necessarily quality. IQ-quants are often preferable over similar sized non-IQ quants)

| Link | Type | Size/GB | Notes |
|:-----|:-----|--------:|:------|
| [GGUF](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.mmproj-Q8_0.gguf) | mmproj-Q8_0 | 0.8 | multi-modal supplement |
| [GGUF](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.mmproj-f16.gguf) | mmproj-f16 | 1.2 | multi-modal supplement |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q2_K.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q2_K.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q2_K.gguf.part3of3) | Q2_K | 118.0 |  |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_S.gguf.part1of3) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_S.gguf.part2of3) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_S.gguf.part3of3) | Q3_K_S | 139.5 |  |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_M.gguf.part1of4) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_M.gguf.part2of4) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_M.gguf.part3of4) [PART 4](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_M.gguf.part4of4) | Q3_K_M | 153.8 | lower quality |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_L.gguf.part1of4) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_L.gguf.part2of4) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_L.gguf.part3of4) [PART 4](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q3_K_L.gguf.part4of4) | Q3_K_L | 167.2 |  |
| [PART 1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q4_K_S.gguf.part1of4) [PART 2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q4_K_S.gguf.part2of4) [PART 3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q4_K_S.gguf.part3of4) [PART 4](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q4_K_S.gguf.part4of4) | Q4_K_S | 182.5 | fast, recommended |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q4_K_M.gguf.part1of5) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q4_K_M.gguf.part2of5) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q4_K_M.gguf.part3of5) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q4_K_M.gguf.part4of5) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q4_K_M.gguf.part5of5) | Q4_K_M | 193.9 | fast, recommended |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_S.gguf.part1of5) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_S.gguf.part2of5) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_S.gguf.part3of5) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_S.gguf.part4of5) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_S.gguf.part5of5) | Q5_K_S | 221.8 |  |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_M.gguf.part1of5) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_M.gguf.part2of5) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_M.gguf.part3of5) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_M.gguf.part4of5) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q5_K_M.gguf.part5of5) | Q5_K_M | 228.0 |  |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q6_K.gguf.part1of6) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q6_K.gguf.part2of6) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q6_K.gguf.part3of6) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q6_K.gguf.part4of6) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q6_K.gguf.part5of6) [P6](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q6_K.gguf.part6of6) | Q6_K | 264.1 | very good quality |
| [P1](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q8_0.gguf.part1of8) [P2](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q8_0.gguf.part2of8) [P3](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q8_0.gguf.part3of8) [P4](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q8_0.gguf.part4of8) [P5](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q8_0.gguf.part5of8) [P6](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q8_0.gguf.part6of8) [P7](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q8_0.gguf.part7of8) [P8](https://huggingface.co/mradermacher/GLM-5.3-Flash-GGUF/resolve/main/GLM-5.3-Flash.Q8_0.gguf.part8of8) | Q8_0 | 341.6 | fast, best quality |

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

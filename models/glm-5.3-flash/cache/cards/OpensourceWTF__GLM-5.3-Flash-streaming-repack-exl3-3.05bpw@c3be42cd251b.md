---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: mlx-serve
tags:
- exl3
- mlx-serve
- deprecated
---

# GLM-5.3-Flash 3.05 bpw EXL3 streaming pack (deprecated)

**Replaced by [OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3](https://huggingface.co/OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3).**

That repo holds four streaming packs of turboderp's self-calibrated EXL3 quantizations of GLM-5.3-Flash, one branch
each (`sc_2.10bpw`, `sc_2.60bpw`, `sc_3.10bpw`, `sc_4.10bpw`). Every weight byte in them is turboderp's, so his KL
divergence numbers apply to them. They are served by mlx-serve's built-in `glm5_next` with the quant plugin
`turboexl3_glm53f`.

This repo's pack is deprecated. It is not a branch of the new repo. Its routed experts are turboderp's `3.05bpw` EXL3
bytes, but its trunk is FP8 and BF16 tensors that we made from the BF16 tensors (the parts that release did not quantize)
of another release, [vcruz305/GLM-5.3-Flash-NVFP4](https://huggingface.co/vcruz305/GLM-5.3-Flash-NVFP4), and its MTP experts are NVFP4.
So turboderp's quality numbers do not apply to it. It was made for the mlx-stream plugin, and we no longer test it. The plugin `turboexl3_glm53f` does not serve it. The
closest replacement is the branch `sc_3.10bpw`.

The files stay in this repo unchanged, at commit `50237d5c`. The card that described them is in this repo's history at
that commit.

The quantization of the routed experts is turboderp's:
[turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), made with
[exllamav3](https://github.com/turboderp-org/exllamav3). Z.ai made
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

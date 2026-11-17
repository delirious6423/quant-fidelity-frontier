---
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
---
# DeepSeek-V4.1-Flash-1017 DSpark draft, 2-bit experts

The DSpark draft head (`mtp.0-2`) of DeepSeek-V4.1-Flash-1017 (revision 1bae0e1088993b9b82de08f59de5be5a34e42170), with its 128 routed experts per
stage in 2 bits. Meant as the draft for the 128-expert 2-bit target
(`DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit`); it uses the target's embedding and LM head and has none of its own.

- `gptq/mtp_SS.safetensors`: stage SS experts, keys `e{E}.w{1,2,3}.{q,s,z}` in Humming's packed GPTQ layout
  (q int32 codes packed along K, s bf16 group scales, z int32 zero points packed along N; w = (q - z) * s).
  w1/w3: uint2, group 128; w2: uint2, group 64.
- `gptq/mtp_SS.json`: per-expert relative weight MSE against the MXFP4 source.
- `model-dspark-00001.safetensors`: the 97 non-expert `mtp.*` tensors, byte for byte from the source.
- `config.json`: the source config, unchanged.

No calibration: the codes are round-to-nearest with a per-group MSE clipping search (E043 `gptq_experts.py`), so this
is the GPTQ format, not calibrated GPTQ. Every expert decodes bit-exact through `humming.utils.weight.dequantize_weight`.
Mean relative weight MSE: stage 0: w13 0.1621, w2 0.1375; stage 1: w13 0.1624, w2 0.1348; stage 2: w13 0.1625, w2 0.1132.

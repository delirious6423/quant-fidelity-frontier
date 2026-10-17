"""Format family of a repo from its id (and tags when known). Container formats win over the number format inside them:
'...-NVFP4-GGUF' is a GGUF file, '...-mlx-nvfp4' is an MLX build."""
ORDER = [("exl3", "EXL3"), ("exl2", "EXL2"), ("gguf", "GGUF"), ("mlx", "MLX"), ("onnx", "ONNX"), ("nvfp4", "NVFP4"), ("mxfp4", "MXFP4"),
         ("mxfp8", "MXFP8"), ("fp4", "NVFP4"), ("autoround", "AutoRound"), ("awq", "AWQ"), ("gptq", "GPTQ"), ("bnb", "bitsandbytes"),
         ("nf4", "bitsandbytes"), ("fp8", "FP8"), ("w8a8", "INT8"), ("int8", "INT8"), ("w4a16", "INT4"), ("int4", "INT4"), ("rocmfp", "GGUF")]
def fmt_of(rid, tags=None):
    s = (rid + " " + " ".join(tags or [])).lower()
    for k, v in ORDER:
        if k in s: return v
    return "other"

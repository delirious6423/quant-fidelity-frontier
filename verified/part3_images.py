# Claims published only as chart images. Values are transcribed from the printed point labels; bpw is digitized
# from marker positions. verify_images.py re-digitizes both images and checks every transcribed label against its
# marker position (must agree within 3% for KLD, 0.0015 for PPL). Images are pinned in snapshots/images/ and are
# referenced on the card at the commit below.
R = []
def repo(**k): R.append(k); return k

IMG_KLD = "https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/Dj8Y9utw9eRXBzHhgLdfE.png"
IMG_PPL = "https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/QB6YgtVCOde709tTvg3h9.png"

def pt(name, bpw, kld, ppl, px, ky, py, branch=None, gb=None):
    return dict(name=name, bpw=bpw, **({'size': gb, 'unit': 'GB', 'basis': 'hf_branch_total', 'branch': branch} if branch else {}), bpw_note="digitized from chart (bpw excl. embeddings and output head)",
                m={"mean_kld": kld, "ppl": ppl}, src="chart_image", img=[IMG_KLD, IMG_PPL], px=dict(x=px, kld_y=ky, ppl_y=py), L=27, L_extra=[28])

repo(repo="turboderp/Qwen3.8-Flash-Next-exl3", sha="c1def4275dc6573c69b5acba1e70c1ce3faf17f8", fmt="EXL3",
  runtime="ExLlamaV3 v1.4.5+", weights_modified=False,
  method="EXL3 plain quants (not self-calibrated); five branches 2.05–6.05 bpw",
  methodology=dict(reference="full-precision model (D_KL(p_FP ‖ p_quant))", dataset="self-generated in-domain trace, 14,496 input + 45,588 output tokens",
                   ctx=None, tool="not stated (repo ships qbench_prompts files)", noise_floor="mean KLD 0.00249 (printed on chart)", bf16_ppl="1.38423 (HF BF16, printed on chart)", L=[27, 28]),
  variants=[
    pt("2.05 bpw H4 NG4", 2.09, "0.0684", "1.4879", 315, 345.2, 457.2, branch="2.05bpw_h4_ng4", gb=62.82),
    pt("3.05 bpw H5 NG5", 3.09, "0.0177", "1.4075", 740, 762.2, 1067.2, branch="3.05bpw_h5_ng5", gb=85.14),
    pt("4.05 bpw H6 NG6", 4.09, "0.0067", "1.3900", 1165, 1061.2, 1201.2, branch="4.05bpw_h6_ng6", gb=107.46),
    pt("5.05 bpw H6 NG6", 5.09, "0.0040", "1.3848", 1590, 1221.2, 1240.2, branch="5.05bpw_h6_ng6", gb=123.25),
    pt("6.05 bpw H6 NG6", 6.09, "0.0031", "1.3825", 2014, 1299.2, 1257.2, branch="6.05bpw_h6_ng6", gb=139.04),
  ],
  cross=[
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", **pt("UD-IQ1_M", 2.85, "0.0804", "1.5031", 638, 296, 343)),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", **pt("UD-Q2_K_XL", 3.12, "0.0533", "1.4619", 754, 423, 655)),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", **pt("UD-IQ3_XXS", 3.31, "0.0349", "1.4342", 833, 553, 866)),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", **pt("UD-IQ4_XS", 4.05, "0.0165", "1.4017", 1150, 784, 1112)),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", **pt("UD-Q4_K_XL", 5.18, "0.0084", "1.3926", 1631, 991, 1181)),
    dict(of="NVFP4 (checkpoint not named)", **pt("NVFP4 W4A16", 4.78, "0.0100", "1.3970", 1457, 938, 1148)),
    dict(of="NVFP4 (checkpoint not named)", **pt("NVFP4 W4A4", 4.78, "0.0241", "1.4199", 1457, 667, 974)),
  ],
  baseline={"ppl": "1.38423", "src": "chart_image", "L": 28},
  flags=["Results exist only as chart images; values here are transcribed from the printed labels and checked against marker positions [L27-28]",
         "Eval text is a self-generated in-domain trace, so absolute KLD is much lower than on held-out corpora; e.g. unsloth UD-Q4_K_XL is 0.0084 here vs 0.046893 on unsloth's own table",
         "The NVFP4 checkpoint measured is not named; W4A16 and W4A4 are two execution modes of the same weights",
         "Chart prints a measured noise floor (mean KLD 0.00249): differences near it are not meaningful",
         "The GGUF series are unsloth's UD quants (names match unsloth's files); file sizes are not given, only bpw excluding embeddings and output head"])

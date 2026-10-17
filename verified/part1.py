# Re-extracted from raw README text at pinned commits (browser fetch, 2026-10-17).
# Each variant/claim cites README line numbers (L) at that sha; quotes are pulled mechanically by verify step.
# size basis: disk = full download incl. n-gram table unless noted; vram/resident/ram = memory figures.
R = []
def repo(**k): R.append(k); return k

repo(repo="AesSedai/Qwen3.8-Flash-Next-GGUF", sha="d933da29a4d846ba03522df6b36f26296d2ce06f", fmt="GGUF",
  runtime="llama.cpp (mainline-compatible)", weights_modified=False,
  method="MoE-aware mixtures: FFN up/gate/down quanted down, rest kept high; two families: Q8_0 PLE (n-gram) and Q4_0 PLE",
  methodology=dict(reference="not stated on card (repo KLD logs: BF16, wikitext-2 test, ctx 512)", dataset="not stated on card", ctx=None, tool="llama-perplexity --kl-divergence (per repo logs)", L=[13]),
  table_header_L=13,
  variants=[
    dict(name="Q5_K_M", size=153.07, unit="GiB", basis="disk", bpw=7.32, m=dict(ppl="4.596526 ± 0.027937", ppl_delta_pct="+0.1270%", mean_kld="0.020218 ± 0.000145"), L=15),
    dict(name="Q4_K_M", size=131.98, unit="GiB", basis="disk", bpw=6.31, m=dict(ppl="4.613527 ± 0.028075", ppl_delta_pct="+0.4973%", mean_kld="0.034206 ± 0.000237"), L=16),
    dict(name="IQ4_XS", size=114.99, unit="GiB", basis="disk", bpw=5.50, m=dict(ppl="4.739310 ± 0.029099", ppl_delta_pct="+3.2373%", mean_kld="0.076158 ± 0.000486"), L=17),
    dict(name="IQ3_S", size=105.98, unit="GiB", basis="disk", bpw=5.07, m=dict(ppl="4.863539 ± 0.030050", ppl_delta_pct="+5.9434%", mean_kld="0.160378 ± 0.000915"), L=18),
    dict(name="IQ2_S", size=103.63, unit="GiB", basis="disk", bpw=4.96, m=dict(ppl="5.043883 ± 0.031497", ppl_delta_pct="+9.8718%", mean_kld="0.205353 ± 0.001121"), L=19),
    dict(name="Q5_K_M (Q4_0 PLE)", size=129.23, unit="GiB", basis="disk", bpw=6.18, m=dict(ppl="4.620130 ± 0.028097", ppl_delta_pct="+0.6412%", mean_kld="0.030943 ± 0.000212"), L=23),
    dict(name="Q4_K_M (Q4_0 PLE)", size=108.14, unit="GiB", basis="disk", bpw=5.17, m=dict(ppl="4.629426 ± 0.028160", ppl_delta_pct="+0.8437%", mean_kld="0.043475 ± 0.000297"), L=24),
    dict(name="IQ4_XS (Q4_0 PLE)", size=91.14, unit="GiB", basis="disk", bpw=4.36, m=dict(ppl="4.759109 ± 0.029230", ppl_delta_pct="+3.6686%", mean_kld="0.084406 ± 0.000531"), L=25),
    dict(name="IQ3_S (Q4_0 PLE)", size=82.13, unit="GiB", basis="disk", bpw=3.93, m=dict(ppl="4.889509 ± 0.030205", ppl_delta_pct="+6.5091%", mean_kld="0.167198 ± 0.000940"), L=26),
    dict(name="IQ2_S (Q4_0 PLE)", size=79.79, unit="GiB", basis="disk", bpw=3.82, m=dict(ppl="5.062449 ± 0.031623", ppl_delta_pct="+10.2763%", mean_kld="0.211550 ± 0.001142"), L=27),
  ],
  flags=["Column labelled 1-(Mean PPL(Q)/PPL(base)) but values are positive % increases (label/sign mismatch) [L13]",
         "Evaluation method not stated on the card itself"])

repo(repo="AtomicChat/Qwen3.8-Flash-Next-GGUF", sha="1c1f33abc8e835db3215390e8e14c6cc726f5843", fmt="GGUF",
  runtime="llama.cpp", weights_modified=False, method="AD builds; imatrix on BF16, 4000 chunks @512, ~4.97M tokens",
  methodology=dict(reference="BF16 logits", dataset="held-out neutral set", ctx=4096, n_chunks=87, tool="llama.cpp KLD", L=[96,97]),
  table_header_L=81,
  variants=[
    dict(name="AD-3.84bpw-IQ4_XS-M64", size=84.9, unit="GB", basis="disk", bpw=3.84, mem=45.8, m=dict(mean_kld="0.2277", top1_pct="82.68%", ppl_ratio="1.102"), L=83),
    dict(name="AD-4.27bpw-Q4_K_M-M64", size=94.5, unit="GB", basis="disk", bpw=4.27, mem=56.1, m=dict(mean_kld="0.0842", top1_pct="89.49%", ppl_ratio="1.026"), L=84),
    dict(name="AD-5.00bpw-Q5_K_M-M64", size=110.5, unit="GB", basis="disk", bpw=5.00, mem=56.1, m=dict(mean_kld="0.0837", top1_pct="89.55%", ppl_ratio="1.026"), L=85),
  ],
  baseline=dict(bf16_ppl="4.0445 ± 0.0216", L=97),
  flags=["Headline '63% KLD reduction at the same file size' compares 84.9 GB vs 94.5 GB builds [L191, L83-84]",
         "Card states 177B params in one place and 125B+51B+4B in another [L44, L258]"])

repo(repo="FINAL-Bench/POCKET-Qwen3.8-Flash-Next-GGUF", sha="cb1c47ab3ebcf65e25310801070bf4de836b45ae", fmt="GGUF",
  runtime="llama.cpp (stock)", weights_modified=False, method="Q4_K_M from BF16; v2 proprietary bit allocation",
  methodology=dict(reference="none (compared to unsloth UD-IQ4_XS, not BF16)", dataset="Korean Wikipedia 20231101.ko, 80 chunks (40,960 tokens)", ctx=512, tool="llama-perplexity", L=[90]),
  table_header_L=92,
  variants=[
    dict(name="Q4_K_M", size=111, unit="GiB", basis="disk", m=dict(ppl="6.030 ± 0.103"), L=94),
    dict(name="Q4_K_M-v2", size=93, unit="GiB", basis="disk", m=dict(ppl="6.080 ± 0.104"), L=95),
  ],
  cross=[dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-IQ4_XS", size=88, unit="GB", m=dict(ppl="6.013 ± 0.103"), L=96)],
  flags=["Lists unsloth UD-IQ4_XS as 88 GB; unsloth's docs give 93.7 GB (= 87.3 GiB), so the unit is probably GiB [L96]", "No BF16 reference; publisher itself says all three are within error and claims no quality advantage [L98]"])

repo(repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF", sha="5348543e0147355ac9cbcb031184a3546350988e", fmt="GGUF",
  runtime="llama.cpp", weights_modified=True, method="GSQ quantization + RCO expert pruning (256 of 512 experts kept), 3.5 bpw retained weights",
  methodology=dict(reference="own BF16 run", dataset="SWE-bench Verified, LiveCodeBench v6", ctx=None, tool="not stated", L=[54]),
  table_header_L=54,
  variants=[dict(name="GSQ-RCO Coder", row_label="retained", size=58.4, unit="GB", basis="disk", bpw=1.89, mem=29.6, m={"bench:SWE-bench Verified":"75.60","bench:LiveCodeBench v6":"86.28","retained:SWE-bench Verified":"91.3%","retained:LiveCodeBench v6":"98.7%"}, L=57, L_extra=[58,110,29])],
  baseline={"bench:SWE-bench Verified":"82.80","bench:LiveCodeBench v6":"87.43","L":56},
  flags=["Pruned: half the routed experts removed — not a faithful quant", "bpw 1.89 is amortised over original params (publisher explains) [L50]",
         "Own BF16 LCB v6 87.43 vs official 91.9 — harness differs from Qwen's"])

repo(repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", sha="ed59f92082b1e93c0e96d60a8b11aab089b52f09", fmt="GGUF",
  runtime="llama.cpp", weights_modified=False, method="GSQ (Gumbel-Softmax scalar quant) + RCO per-layer type allocation; n-gram table fixed IQ4_NL",
  methodology=dict(reference="own BF16 run", dataset="5 zero-shot tasks (arc_e, arc_c, hellaswag, winogrande, piqa) + AIME25, GPQA-Diamond, LCB v6", ctx=None, tool="not stated", L=[119]),
  table_header_L=121,
  variants=[
    dict(name="Q2_0", size=66.4, unit="GB", basis="disk", bpw=2.40, m={"zs_avg":"78.00","recovery_pct":"101.4%","bench:AIME25":"96.67","bench:GPQA-Diamond":"89.39","bench:LiveCodeBench v6":"81.14","task_avg":"89.07"}, L=124),
    dict(name="IQ2_XS", size=68.0, unit="GB", basis="disk", bpw=2.50, m={"zs_avg":"77.16","recovery_pct":"100.3%","bench:AIME25":"96.67","bench:GPQA-Diamond":"87.37","bench:LiveCodeBench v6":"83.43","task_avg":"89.16"}, L=125),
    dict(name="IQ3_XXS", size=75.8, unit="GB", basis="disk", bpw=3.00, m={"zs_avg":"77.23","recovery_pct":"100.4%","bench:AIME25":"100.00","bench:GPQA-Diamond":"91.41","bench:LiveCodeBench v6":"86.29","task_avg":"92.57"}, L=126),
    dict(name="IQ3_S", size=83.6, unit="GB", basis="disk", bpw=3.50, m={"bench:AIME25":"100.00","bench:GPQA-Diamond":"92.93","bench:LiveCodeBench v6":"86.86","task_avg":"93.26"}, L=127),
  ],
  baseline={"zs_avg":"76.94","bench:AIME25":"100.00","bench:GPQA-Diamond":"91.92","bench:LiveCodeBench v6":"87.43","task_avg":"93.12","L":123},
  flags=["Publisher itself notes >100% recovery is within ~1 SE and means parity, not improvement [L129]",
         "Own BF16 baseline (LCB v6 87.43, GPQA 91.92) differs from Qwen's official 91.9 / 91.7"])

repo(repo="Inferact/Qwen3.8-Flash-Next-NVFP4", sha="103a7608316173ca6edd49929544244de7ffda70", fmt="NVFP4",
  runtime="vLLM/SGLang", weights_modified=False, method="NVFP4 (ModelOpt per config)", methodology=None, variants=[dict(name="NVFP4", size=None)],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables"])

repo(repo="Intel/Qwen3.8-Flash-Next-W4A16-AutoRound", sha="4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897", fmt="AutoRound",
  runtime="vLLM/transformers", weights_modified=False, method="AutoRound W4A16 (routed experts)",
  methodology=dict(reference="BF16 (own run)", dataset="GSM8K, MMLU, PIQA, HellaSwag", ctx=None, tool="not stated", L=[14]),
  table_header_L=14,
  variants=[dict(name="INT4 (W4A16)", size=None, m={"bench:GSM8K":"0.9682","bench:MMLU":"0.8560","bench:PIQA":"0.8215","bench:HellaSwag":"0.6873","bench_avg":"0.8332","recovery_pct":"99.64%"}, L=17)],
  baseline={"bench:GSM8K":"0.9673","bench:MMLU":"0.8651","bench:PIQA":"0.8193","bench:HellaSwag":"0.6927","bench_avg":"0.8362","L":16},
  flags=["Four-task table (GSM8K, MMLU, PIQA, HellaSwag) against its own BF16 run; harness settings not given [L14-17]"])

repo(repo="Jundot/Qwen3.8-Flash-Next-oQ4e-mtp", sha="c4ddfb54a967a9bc74d7da16425f55f0872f8aac", fmt="MLX",
  runtime="oMLX", weights_modified=False, method="oQ4e: sensitivity-based bit allocation + imatrix (1024x512) + weighted LSQ scale refit",
  methodology=dict(reference="BF16", dataset="oMLX benchmark corpora, 16 windows x 2,048 tokens per corpus", ctx=2048, tool="oMLX", L=[30]),
  table_header_L=32,
  variants=[dict(name="oQ4e (updated)", size=106.294, unit="GB", basis="disk", bpw=4.724, m={"kl:code":"0.0776","kl:novel":"0.1713"}, L=34, L_extra=[35,53,63])],
  ladder=[dict(name="oQ4e (previous, 2026-08-27)", m={"kl:code":"0.0877","kl:novel":"0.1909"}, L=34, L_extra=[35])],
  flags=["Per-corpus KL only (code, novel); no single mean figure"])

repo(repo="Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata", sha="379ff0b8a39f83dfd93d2122cc91dd4029cf9602", fmt="NVFP4",
  runtime="Strata NVFP4 fork only", weights_modified=False, method="NVFP4 experts by GPTQ + block-scale search from BF16; 57.5K calib tokens",
  methodology=dict(reference="Q8_0 (own build)", dataset="5,153 tokens of code/text", ctx=None, tool="Strata", L=[44]),
  table_header_L=31,
  variants=[dict(name="NVFP4 GPTQ", size=69, unit="GiB", basis="ram_in_use", m={"top1_pct":"93.6%","mean_kld":"0.038","expert_rel_err_vs_bf16":"0.00561"}, L=36, L_extra=[78,149])],
  cross=[
    dict(of="Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata", variant="Q8_0 (reference)", size=106, unit="GiB", basis="ram_in_use", m={"top1_pct":"100%"}, L=33),
    dict(of="Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata", variant="NVFP4 + Q8_0 mix", size=105, unit="GiB", basis="ram_in_use", m={"top1_pct":"95.0%"}, L=34),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-Q4_K_XL", size=77, unit="GiB", basis="ram_in_use", m={"top1_pct":"94.1%"}, L=35),
    dict(of="nvidia/Qwen3.8-Flash-Next-NVFP4", variant="NVFP4 (converted to Strata)", row_label="NVIDIA's checkpoint", size=69, unit="GiB", basis="ram_in_use", m={"top1_pct":"92.9%","expert_rel_err_vs_bf16":"0.01252","mean_kld":"0.038"}, L=37, L_extra=[148,78]),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-Q3_K_XL", size=58, unit="GiB", basis="ram_in_use", m={"top1_pct":"90.7%"}, L=38),
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="IQ3_S", size=52, unit="GiB", basis="ram_in_use", m={"top1_pct":"90.0%"}, L=39),
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="IQ3_XXS", size=45, unit="GiB", basis="ram_in_use", m={"top1_pct":"87.3%"}, L=40),
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="IQ2_XS", size=38, unit="GiB", basis="ram_in_use", m={"top1_pct":"85.0%"}, L=41),
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="Q2_0", size=37, unit="GiB", basis="ram_in_use", m={"top1_pct":"84.1%"}, L=42),
  ],
  flags=["KL 0.038 is stated in prose as 'table above', but no table on the card shows KL [L78]", "Reference is the publisher's own Q8_0, not BF16", "Small eval: 5,153 tokens", "Runs only on the Strata fork; other publishers' files re-run inside Strata"])

repo(repo="Qwen/Qwen3.8-Flash-Next-FP8", sha="236dfdf285828023ca3bcd3f37366c58a3469b13", fmt="FP8",
  runtime="vLLM/SGLang", weights_modified=False, method="Official FP8, block size 128", methodology=None, variants=[dict(name="FP8", size=None)],
  flags=["Official; no FP8-specific quality numbers on card"])

repo(repo="RadixArk/Qwen3.8-Flash-Next-NVFP4", sha="7b719225242aacd3dbd3f9407468c2ee9a9d2594", fmt="NVFP4",
  runtime="SGLang", weights_modified=False, method="NVFP4 MoE experts; calib 128 cnn_dailymail articles @512",
  methodology=dict(reference="BF16 runs on an earlier checkpoint revision", dataset="GSM8K (1319), AIME26 (30x8)", ctx=None, tool="own", L=[118,123]),
  table_header_L=118,
  variants=[dict(name="NVFP4", size=135, unit="GB", basis="disk", m={"bench:GSM8K":"97.27","bench:AIME26 pass@1":"98.75"}, L=120, L_extra=[121,92])],
  baseline={"bench:GSM8K":"97.12–97.50","L":120},
  baseline2={"bench:AIME26 pass@1":"100","L":121},
  flags=["BF16 baselines from an earlier checkpoint revision [L123]", "Card notes long agentic generations run longer than BF16 [L131]"])

repo(repo="TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw", sha="b5591ded", fmt="MLX",
  runtime="Knurlogic / mlx-lm (custom VQ)", weights_modified=False, method="Data-free vector quantization (k-means codebooks)",
  methodology=dict(reference="bf16 teacher, cached top-64 logits", dataset="3 house corpora (prose, code, literary), 12,288 tokens each", ctx=None, tool="Knurlogic 0.1.3", unit_note="KL in millinats/token (divide by 1000 for nats)", L=[160,161]),
  table_header_L=170, shared_table="TheDrainFlorist ladder",
  variants=[
    dict(name="VQ-2.1bpw", row_label="this model", size=44.5, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"652.4","top1_pct":"72.3%","ppl":"3.559"}, L=173),
  ],
  flags=["KL unit is millinats — not comparable numerically to nat-scale KLD tables without conversion"])

repo(repo="TheDrainFlorist/Qwen3.8-Flash-Next-VQ-4.4bpw", sha="c13ef42d", fmt="MLX",
  runtime="Knurlogic / mlx-lm (custom VQ)", weights_modified=False, method="Data-free vector quantization (k-means codebooks)",
  methodology=dict(reference="bf16 teacher, cached top-64 logits", dataset="3 house corpora (prose, code, literary), 12,288 tokens each", ctx=None, tool="Knurlogic 0.1.3", unit_note="KL in millinats/token", L=[168,169]),
  table_header_L=178, shared_table="TheDrainFlorist ladder",
  variants=[dict(name="VQ-4.4bpw", row_label="this model", size=93.6, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"47.7","top1_pct":"92.2%","ppl":"2.323"}, L=184)],
  ladder=[  # publisher's own comparison builds (same table)
    dict(name="affine q3 (ours)", size=75, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"1312.4","top1_pct":"49.5%","ppl":"14.054"}, L=180),
    dict(name="VQ-3.2bpw", size=68.7, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"147.3","top1_pct":"86.7%","ppl":"2.300"}, L=182),
    dict(name="affine q4 (ours)", size=96, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"298.2","top1_pct":"77.8%","ppl":"3.912"}, L=183),
    dict(name="affine q5 (ours)", size=116, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"97.5","top1_pct":"89.0%","ppl":"2.549"}, L=185),
    dict(name="VQ-5.5bpw", size=111.6, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"30.9","top1_pct":"93.7%","ppl":"2.244"}, L=186),
    dict(name="affine q6 (ours)", size=137, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"39.9","top1_pct":"93.4%","ppl":"2.276"}, L=187),
    dict(name="affine q8 (ours)", size=178, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"22.4","top1_pct":"94.8%","ppl":"2.367"}, L=188),
  ],
  baseline={"ppl":"2.31","top1_pct":"100%","L":189},
  flags=["KL unit is millinats", "Top-1 and perplexity are prose-corpus only; mean KL averages three corpora [L170, L178]",
         "Table size is the text weights; the repo also ships a 0.84 GiB vision tower and a 2.1 GiB MTP head [L171-172], which is why the card gives about 104 GB on disk [L235]"])

repo(repo="agentionai/Qwen3.8-Flash-Next-AP-GGUF", sha="89f38eeb", fmt="GGUF",
  runtime="llama.cpp mainline", weights_modified=False, method="Agention Precision recipes (per-tensor distortion solver)",
  methodology=dict(reference="'the same source' (card); agention.ai says Q8_0", dataset="held-out corpus (not wikitext — memorised by n-gram table)", ctx=2048, tool="llama-perplexity --kl-divergence", L=[89,90,94]),
  table_header_L=45,
  variants=[
    dict(name="AP-Q5_K_XL", size=112.51, unit="GiB", unit_L=56, basis="disk", bpw=5.46, mem=76.75, m={"mean_kld":"0.0839 ± 0.0008","top1_pct":"86.63 ± 0.14 %"}, L=47),
    dict(name="AP-Q5_K_M", size=103.57, unit="GiB", unit_L=56, basis="disk", bpw=5.03, mem=76.75, m={"mean_kld":"0.0771 ± 0.0007","top1_pct":"86.95 ± 0.14 %","kld:code":"0.0181","top1:code":"97.69 %"}, L=48, L_extra=[69]),
    dict(name="AP-Q4_K_XL", size=94.20, unit="GiB", unit_L=56, basis="disk", bpw=4.57, mem=67.38, m={"mean_kld":"0.0886 ± 0.0008","top1_pct":"86.14 ± 0.14 %","kld:code":"0.0206","top1:code":"97.38 %"}, L=49, L_extra=[70]),
    dict(name="AP-Q4_K_M", size=88.04, unit="GiB", unit_L=56, basis="disk", bpw=4.27, mem=61.22, m={"mean_kld":"0.1243 ± 0.0011","top1_pct":"84.22 ± 0.15 %"}, L=50),
    dict(name="AP-IQ4_XS", size=84.24, unit="GiB", unit_L=56, basis="disk", bpw=4.09, mem=57.42, m={"mean_kld":"0.1584 ± 0.0014","top1_pct":"82.89 ± 0.15 %"}, L=51),
    dict(name="AP-IQ3_XXS", size=80.72, unit="GiB", unit_L=56, basis="disk", bpw=3.92, mem=53.90, m={"mean_kld":"0.1844 ± 0.0016","top1_pct":"81.85 ± 0.16 %"}, L=52),
    dict(name="AP-IQ2_S", size=76.03, unit="GiB", unit_L=56, basis="disk", bpw=3.69, mem=49.21, m={"mean_kld":"0.2657 ± 0.0022","top1_pct":"79.14 ± 0.16 %"}, L=53),
  ],
  cross=[dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-Q4_K_XL", size=103.69, unit="GiB", unit_L=56, basis="disk", m={"mean_kld":"0.0778","top1_pct":"87.21 %","kld:code":"0.0192","top1:code":"97.36 %"}, L=71)],
  flags=["Reference model ambiguous: card says 'the same source'; Agention's site says Q8_0",
         "Card reports wikitext inflates top-1 by 6.9–8.6 points for this architecture (n-gram table memorisation) [L95]"])

repo(repo="agentionai/Qwen3.8-Flash-Next-Gyro-GGUF", sha="aa0a0c3e", fmt="GGUF",
  runtime="agentionai/llama.cpp fork (TQ types) / Strata", weights_modified=False, method="Gyro: rotated-expert ternary-ish TQ formats; calibration not published",
  methodology=dict(reference="unsloth Q8_0", dataset="held-out 2026 technical writing + code, 60 chunks", ctx=2048, tool="llama.cpp fork KLD", L=[52,53]),
  table_header_L=55,
  variants=[
    dict(name="Gyro-S (TQ1_0)", size=27.64, unit="GiB", basis="vram", m={"mean_kld":"0.401","top1_pct":"75.29%","kld:code":"0.109"}, L=57, L_extra=[147]),
    dict(name="Gyro-M (TQ2_0)", size=34.98, unit="GiB", basis="vram", m={"mean_kld":"0.291","top1_pct":"78.75%","kld:code":"0.066"}, L=58, L_extra=[144]),
    dict(name="Gyro-L (TQ3_0)", size=50.21, unit="GiB", basis="vram", m={"mean_kld":"0.146","top1_pct":"83.87%","kld:code":"0.027"}, L=59, L_extra=[138]),
  ],
  cross=[
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="IQ2_XS", size=36.52, unit="GiB", basis="vram", m={"mean_kld":"0.420","top1_pct":"74.22%"}, L=60),
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="IQ3_XXS", size=43.80, unit="GiB", basis="vram", m={"mean_kld":"0.311","top1_pct":"77.68%"}, L=61),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-Q2_K_XL", size=46.63, unit="GiB", basis="vram", m={"mean_kld":"0.334","top1_pct":"76.61%"}, L=62),
    dict(of="AtomicChat/Qwen3.8-Flash-Next-GGUF", variant="AD-4.27bpw Q4_K_M", size=52.26, unit="GiB", basis="vram", m={"mean_kld":"0.228","top1_pct":"80.58%"}, L=63),
    dict(of="ggml-org/Qwen3.8-Flash-Next-GGUF", variant="IQ4_NL", size=68.21, unit="GiB", basis="vram", m={"mean_kld":"0.200","top1_pct":"81.29%"}, L=64),
    dict(of="agentionai/Qwen3.8-Flash-Next-AP-GGUF", variant="AP-Q4_K_XL", size=67.38, unit="GiB", basis="vram", m={"mean_kld":"0.107","top1_pct":"85.52%"}, L=65),
  ],
  flags=["Sizes are GPU-resident weights only (n-gram table on disk); Gyro-L download is 108 GB [L139]",
         "Reference is unsloth Q8_0, not BF16", "Same AP-Q4_K_XL file measures 0.107 here vs 0.0886 on the AP card — references differ"])

repo(repo="agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF", sha="d37a14a0", fmt="GGUF",
  runtime="agentionai/llama.cpp fork (ROCmFPx types)", weights_modified=False, method="ROCmFP4 + imatrix (1540 chunks, 2 corpora)",
  methodology=dict(reference="unquantized PPL as reported in llama.cpp PR 27742 (not own run)", dataset="wikitext-2 raw, 145 chunks", ctx=2048, tool="llama-perplexity", L=[156,160]),
  table_header_L=158,
  variants=[dict(name="ROCmFP4-FAST", size=87.06, unit="GiB", basis="disk", bpw=4.23, m={"ppl":"4.1062 +/- 0.02329","ppl_delta_pct":"+2.48%"}, L=161, L_extra=[23])],
  baseline={"ppl":"4.0068 +/- 0.02271","L":160},
  cross=[
    dict(of="AesSedai/Qwen3.8-Flash-Next-GGUF", variant="IQ3_S", size=107.38, measured=False, unit="GiB", m={"ppl_delta_pct":"+6.10%"}, L=169),
    dict(of="AesSedai/Qwen3.8-Flash-Next-GGUF", variant="IQ4_XS", size=117.13, measured=False, unit="GiB", m={"ppl_delta_pct":"+3.12%"}, L=171),
    dict(of="AesSedai/Qwen3.8-Flash-Next-GGUF", variant="Q4_K_M", size=135.38, measured=False, unit="GiB", m={"ppl_delta_pct":"+0.61%"}, L=172),
  ],
  flags=["Uses wikitext PPL, which the same publisher's AP card says is memorised by this architecture",
         "Baseline PPL borrowed from a llama.cpp PR, not measured alongside",
         "AesSedai sizes quoted (107.38/117.13/135.38 GiB) don't match AesSedai's current card (105.98/114.99/131.98) — likely pre-10/03 re-quant"])

repo(repo="amd/Qwen3.8-Flash-Next-Quark-MXFP4", sha="b2091c38", fmt="MXFP4",
  runtime="vLLM ROCm", weights_modified=False, method="Quark MXFP4 on all MoE experts incl. shared; attention bf16",
  methodology=dict(reference="BF16 (own run)", dataset="gsm8k flexible-extract, 5-shot, chat template", ctx=None, tool="lm-eval", L=[84,104]),
  table_header_L=84,
  variants=[dict(name="MXFP4", size=None, m={"bench:GSM8K (5-shot)":"96.5","recovery_pct":"99.7%"}, L=86)],
  baseline={"bench:GSM8K (5-shot)":"96.8","L":86},
  flags=["Single saturated benchmark"])

repo(repo="antirez/qwen3.8-flash-next-gguf", sha="d600fe1a", fmt="GGUF", runtime="DwarfStar (ds4) only", weights_modified=False,
  method="Custom Q2/Q4 with BF16 n-gram table read from disk", methodology=None, table_header_L=20,
  variants=[dict(name="Q2", size=137.10, unit="GiB", basis="disk", mem=41.73, L=22), dict(name="Q4", size=165.11, unit="GiB", basis="disk", mem=69.74, L=23)],
  flags=["No quality claims", "Each file embeds the 95.37 GiB BF16 n-gram table [L25]"])

repo(repo="apetersson/Qwen3.8-Flash-Next-GGUF", sha="9af845f8", fmt="GGUF", runtime="llama.cpp + unmerged PR", weights_modified=False,
  method="Q5_K_M trunk (bartowski imatrix) and Unsloth UD-IQ4_XS trunk, both with shared BF16 n-gram table", methodology=None, table_header_L=55,
  variants=[dict(name="Q5_K_M trunk", size=76.24, unit="GiB", basis="trunk_only", L=57), dict(name="UD-IQ4_XS trunk", size=60.42, unit="GiB", basis="trunk_only", L=58)],
  flags=["No quality claims (memory/speed only)", "Sizes exclude the shared 95.37 GiB BF16 n-gram table [L59]"])

repo(repo="bartowski/Qwen3.8-Flash-Next-GGUF", sha="db3a1857", fmt="GGUF", runtime="llama.cpp", weights_modified=False,
  method="imatrix quants; calibration 583 chunks x 512 (214 prose, 369 tool-calling)", methodology=None, table_header_L=42,
  variants=[dict(name=n, size=s, unit="GB", basis="disk", L=l) for n, s, l in [
    ("bf16",354.03,44),("Q8_0",188.26,45),("Q6_K",168.11,46),("Q5_K_L",151.06,47),("Q4_K_L",139.27,48),("Q5_K_M",134.67,49),("Q5_K_S",128.10,50),
    ("Q4_K_M",119.60,51),("Q3_K_XL",119.35,52),("Q4_K_S",113.14,53),("Q4_1",111.26,54),("Q2_K_L",107.15,55),("Q4_0",100.60,56),("IQ4_NL",100.29,57),
    ("IQ4_XS",97.68,58),("IQ3_M",93.23,59),("Q3_K_L",93.19,60),("Q3_K_M",91.96,61),("IQ3_XS",91.95,62),("Q3_K_S",89.44,63),("IQ3_XXS",88.02,64),
    ("Q2_K",80.93,65),("IQ2_M",80.37,66),("IQ2_S",77.83,67),("IQ2_XS",77.74,68),("IQ2_XXS",75.19,69),("IQ1_M",72.01,70),("IQ1_S",70.10,71)]],
  flags=["No measured quality claims; qualitative labels only"])

repo(repo="cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4", sha="d39638a0", fmt="AWQ", runtime="vLLM", weights_modified=False,
  method="AWQ INT4 (calibration 'STEM and Agentic' per card banner)", methodology=None, variants=[dict(name="AWQ-INT4", size=None)],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables"])

repo(repo="ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw", sha="4fabeb5e", fmt="MLX", runtime="mlx-serve", weights_modified=False,
  method="iQ-MLX: imatrix-measured per-layer 2/3/4-bit allocation; 482k-token calibration",
  methodology=dict(reference="bf16", dataset="1,200 held-out next-token positions (agent, code, prose, math)", ctx=None, tool="own scorer (qwen38_flash_next_score.py)", L=[48,49,50]),
  table_header_L=53,
  variants=[dict(name="iQ-MLX 3.3bpw", size=52, unit="GB", basis="resident", bpw=3.3, m={"top1_pct":"85.6%","prob_mass":"0.732"}, L=56)],
  ladder=[dict(name="mixed-4-8bit", size=73, unit="GB", basis="disk", m={"top1_pct":"89.1%","prob_mass":"0.738"}, L=55)],
  baseline={"prob_mass_ceiling":"0.762","L":53},
  flags=["Download is 86 GB incl. 32 GB n-gram table; 52 GB is resident weights [L39]",
         "mixed-4-8bit listed at 73 GB / 89.1% here, but its own card says 75 GB and top-1 87.9% vs its successor's 88.6% — the two cards disagree"])

repo(repo="ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit", sha="788c07a8", fmt="MLX", runtime="mlx-serve", weights_modified=False,
  method="4-bit group-64 experts, 8-bit dense", methodology=None, table_header_L=44,
  variants=[dict(name="mixed-4-8bit", size=75, unit="GB", basis="resident", m={"top1_pct":"87.9%","kld_median":"0.0066"}, L=23, L_extra=[48])],
  flags=["Superseded by the publisher's iQ-MLX-4.7bpw (KLD median 0.0052, top-1 88.6%) [L20-23]", "Method for these numbers not stated on this card"])

repo(repo="doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw", sha="e2e53fd7", fmt="EXL3", runtime="veloGB10 (exllamav3-based)", weights_modified=False,
  method="Repack of turboderp's 4.05bpw_h6_ng6 EXL3 + bf16 vision tower", methodology=None,
  variants=[dict(name="4.05bpw_h6_ng6", size=108, unit="GB", basis="disk", bpw=4.05, L=34, L_extra=[28])],
  flags=["Quality check is a next-token-loss comparison with the publisher's own 3.05 pack (0.625 vs 0.621 at 8K, 1.579 vs 1.578 at 16K), which the card calls a mis-decode alarm, not a quality ranking [L152-154]",
         "Separately, individual tensors and lookups decode bit-identically to exllamav3; the full forward pass was not compared [L146-151, L155-156]"])

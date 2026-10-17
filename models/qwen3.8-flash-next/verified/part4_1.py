# Extracted from verbatim README snapshots at pinned commits (snapshots/index.json, captured 2026-10-17); queue_1.
# Each variant/claim cites README line numbers (L) at that sha.
# size basis: disk = full download unless noted; vram/resident/ram_in_use = memory figures; trunk_only = excludes n-gram table.
R = []
def repo(**k): R.append(k); return k

repo(repo="jcbtc/Qwen3.8-Flash-CIRU-STRIX-Orca", sha="139c9f5eb561dafbcf7cfab0622661b6fc0137e7", fmt="GGUF (CIRU compact expert format)",
  runtime="CIRU v5.5.0 native runtime only (stock llama.cpp does not load it)", weights_modified=True,
  method="Routed experts as 4-bit integer codes, groups of 16 sharing scale/offset table index (4.5175 bpw on routed experts); down projections fitted from Orca BF16 with Orca calibration activations; source is OrcaRouter's refusal-removed checkpoint",
  methodology=dict(reference="retained (original) Orca CIRU GGUF, not BF16 (card: full-precision Orca logit reference not available)", dataset="sixteen fixed 512-token text windows, 2,048 tail positions scored", ctx=512, tool="not stated (CIRU runtime)", L=[184,185,189,190]),
  variants=[
    dict(name="Orca v5.0 (compact)", size=72112333504, unit="bytes", basis="disk", bpw=4.5175, mem=87.31,
         m={"top1_pct":"95.21%","mean_kld":"0.047966","nll":"0.716220","ppl":"2.04668"}, L=107, L_extra=[116,186,187,188,193,218]),
  ],
  ladder=[dict(name="Original Orca GGUF (retained parts)", size=79397818912, unit="bytes", basis="disk", mem=94.19, m={"nll":"0.710105","ppl":"2.03421"}, L=217, L_extra=[187,188,193])],
  flags=["Source checkpoint is OrcaRouter's refusal-removed (uncensored) Qwen3.8 Flash [L6, L302, L316]",
         "Reference is the publisher's earlier Orca GGUF, not BF16; card says old-to-new agreement is not BF16 agreement [L189-190]",
         "Small eval: 2,048 scored positions, described by the card as a short-text diagnostic [L185, L188]",
         "mem = GPU-visible unified memory after loading (94.19 / 87.31 GiB); 4.5175 bpw covers routed experts only [L116-117, L193]",
         "Task scores (HumanEval, HermesAgent) are historical, on the original weights and earlier runtimes, not rerun on v5.0 [L281-290]"])

repo(repo="btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit", sha="9ae8d0e1989ad3a4eaeea501d5f6c75f4eeda7e1", fmt="GPTQ",
  runtime="vLLM (main, Sep 2026+) / GPTQModel fork", weights_modified=False,
  method="GPTQ W4 g32 symmetric, mse 2.0, RTN fallback for 7.6% of expert modules; routed+shared experts and full-attention projections INT4; GDN, indexer, routers, n-gram table, vision, MTP BF16; calib evol-codealpaca-v1 + C4, 2048 samples (~2.4M tokens)",
  methodology=dict(reference="BF16 (own run, identical windows)", dataset="wikitext-2-raw-v1 test, 64 windows (32,768 scored tokens); also wikitext-2 4k ctx 8x1024 and codeparrot-clean-valid slice", ctx=2048, stride=512, tool="transformers forward", L=[86,97,105]),
  table_header_L=99,
  variants=[dict(name="GPTQ 4-bit", size=188, unit="GB", basis="disk", mem=80,
     m={"ppl":"3.1386","ppl_delta_pct":"+0.58%","mean_kld":"0.071","kld_p99":"0.89","top1_pct":"91.3%","top5_overlap":"84.6%",
        "ppl:wiki4k":"2.2822","ppl_delta_pct:wiki4k":"−0.43%","mean_kld:wiki4k":"0.093","top1:wiki4k":"91.7%",
        "ppl:code":"2.1796","ppl_delta_pct:code":"−0.02%","kld:code":"0.027","top1:code":"95.8%","ppl:vllm_served":"3.1362"},
     L=78, L_extra=[91,101,102,103,127])],
  baseline={"ppl":"3.1206","ppl:wiki4k":"2.2919","ppl:code":"2.1801","L":90,"L_extra":[101,102,103]},
  flags=["mem = GPU-resident body (80 GB); 188 GB download includes the 102 GB BF16 n-gram table [L78, L80]",
         "Headline PPL is on wikitext-2, which the card says the n-gram memory has close to memorized [L93]",
         "Small eval sets: 32,768 wiki tokens; 4k-ctx suite only 8x1024 tokens [L86, L102]",
         "Code slice described as 32x512 tokens in the table but '~65k scored tokens' in the footnote [L103, L105]"])

repo(repo="nvidia/Qwen3.8-Flash-Next-FP4-Local", sha="b9ce18a6887e18c089fdea8509a33e63429a3300", fmt="NVFP4 (+MXFP8)",
  runtime="vLLM community containers only (eugr/spark-vllm-b12x, local-inference-lab)", weights_modified=True,
  method="Quantization-aware distillation (QAD) with ModelOpt v0.47.0: routed experts NVFP4, attention MXFP8; 2,500 trunk + 1,500 joint-refinement updates also training n-gram tables; 500M tokens x 3 epochs",
  methodology=dict(reference="BF16 (Qwen/Qwen3.8-Flash-Next)", dataset="GPQA Diamond (198 questions), ToolEvalBench", ctx=None, settings="not stated", tool="not stated", L=[108,111,120]),
  table_header_L=115,
  variants=[dict(name="FP4-Local", size=96, unit="GiB", basis="disk", m={"bench:GPQA Diamond":"91.5%","bench:ToolEvalBench":"88.7%"}, L=118, L_extra=[25])],
  baseline={"bench:GPQA Diamond":"92.2%","bench:ToolEvalBench":"89.4%","L":117},
  flags=["Weights trained via quantization-aware distillation (gradient updates incl. n-gram tables), not a post-training quant [L23, L123-127]",
         "No sampling settings, run counts or error bars for the two benchmarks [L106-120]"])

repo(repo="TheDrainFlorist/Qwen3.8-Flash-Next-VQ-3.2bpw", sha="33e10e66b4277dc332f085a17025348dad13604a", fmt="MLX",
  runtime="Knurlogic >=0.1.3 / mlx-lm (custom VQ model.py)", weights_modified=False,
  method="Data-free vector quantization (k-means over weight subvectors, seed 1234); experts and PLE d=4/K=2048, six layers d=2/K=256",
  methodology=dict(reference="bf16 teacher, cached top-64 logits", dataset="3 house corpora (prose, code, literary), 12288 tokens per corpus", ctx=None, tool="Knurlogic 0.1.3 (mlx 0.32.3)", unit_note="KL in millinats/token", L=[159,160,161]),
  table_header_L=169, shared_table="TheDrainFlorist ladder",
  variants=[dict(name="VQ-3.2bpw", row_label="this model", size=68.7, unit="GiB", basis="text_weights", m={"kl_mnats:prose":"305.6","kl_mnats:code":"81.8","kl_mnats:literary":"54.6","mean_kl_mnats":"147.3","top1_pct":"86.7%","ppl":"2.300"}, L=173)],
  ladder=[
    dict(name="affine q3 (ours)", size=75, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"1312.4","top1_pct":"49.5%","ppl":"14.054"}, L=171),
    dict(name="VQ-2.1bpw", size=44.5, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"652.4","top1_pct":"72.3%","ppl":"3.559"}, L=172),
    dict(name="affine q4 (ours)", size=96, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"298.2","top1_pct":"77.8%","ppl":"3.912"}, L=174),
    dict(name="VQ-4.4bpw", size=93.6, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"47.7","top1_pct":"92.2%","ppl":"2.323"}, L=175),
    dict(name="affine q5 (ours)", size=116, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"97.5","top1_pct":"89.0%","ppl":"2.549"}, L=176),
    dict(name="VQ-5.5bpw", size=111.6, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"30.9","top1_pct":"93.7%","ppl":"2.244"}, L=177),
    dict(name="affine q6 (ours)", size=137, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"39.9","top1_pct":"93.4%","ppl":"2.276"}, L=178),
    dict(name="affine q8 (ours)", size=178, unit="GiB", basis="text_weights", m={"mean_kl_mnats":"22.4","top1_pct":"94.8%","ppl":"2.367"}, L=179),
  ],
  baseline={"ppl":"2.31","size":"335 GiB","L":180},
  flags=["KL unit is millinats — not comparable numerically to nat-scale KLD tables without conversion [L159-160]",
         "Sizes are text weights only; VQ builds also carry a 0.84 GiB vision tower and every build ships a 2.1 GiB MTP head [L162-163]",
         "Affine comparators are the publisher's own conversions, not community builds [L31-32, L311]",
         "Headline says 68.7 GiB while the 2026-09-17 changelog gives 69.5 GiB for the same v2 weights [L21, L87]",
         "Prose PPL of this build (2.300) is below the bf16 teacher (2.31); card says to rank by KL, not PPL [L173, L180, L188-192]"])

repo(repo="DJLougen/Qwen3.8-Flash-Next-Mooney", sha="84f867d3236f430af02776d36fc111dbd8daae96", fmt="GGUF (PQ2_0 ternary experts)",
  runtime="buun-llama-cpp / DJLougen cuda.fast fork / prism-llama.cpp fork (stock llama.cpp cannot load)", weights_modified=True,
  method="Mooney recipe: routed experts ternary (PQ2_0, 2.125 bpw) in a rotated basis, then recovered with distillation training; non-experts Q8_0; PLE table Q8_0 read lazily from SSD",
  methodology=dict(reference="BF16 original (own run on vLLM, 8xA100)", dataset="own suite of 536 auto-verified tasks (unit tests, exact answers, schema and tool-call checks), run twice per model; needle retrieval 32k/64k/128k", ctx=None, tool="own harness on vLLM", L=[112,120,169]),
  table_header_L=137,
  variants=[dict(name="Mooney (PQ2_0)", size=92.0, unit="GB", basis="disk", bpw=4.16, mem=39,
     m={"suite_score":"88.71%","suite_run1":"475/536 (88.62%)","suite_run2":"476/536 (88.81%)","retained_pct":"95.5%","needle":"30/30","top1_pct:long_text":"≈72%","kld_top20:long_text":"≈0.27"},
     L=140, L_extra=[20,100,115,120,127])],
  baseline={"suite_score":"92.91%","suite_run1":"500/536 (93.28%)","suite_run2":"496/536 (92.54%)","L":139},
  flags=["Experts recovered with distillation training after ternarization — weights changed beyond rounding [L86]",
         "Quality numbers measured on vLLM 'with the same weight values', not on this GGUF; the GGUF was measured on DGX Spark for speed and memory only [L112]",
         "Own task suite, no third-party benchmarks; card says so [L169]",
         "mem ≈39 GiB is MemAvailable delta incl. KV/context (≈35 GiB weights); 92.0 GB download includes the 54.4 GB PLE table kept on SSD [L99-100, L108]",
         "Long-text KL is truncated top-20 KL, not full-vocab [L127]"])

repo(repo="nazeshinjite/Qwen3.8-Flash-Next-Q8-UDQ4K-DS4", sha="9374dd4e1f8da135e65bb94196d4b3b58b6906bb", fmt="GGUF (ds4)",
  runtime="DwarfStar (ds4) Metal", weights_modified=False,
  method="Repack of antirez Q4: all ffn_down_exps to Q8_0 (from unsloth UD-Q5_K_XL shards), gate/up Q8_0 in layers 0,1,2,45 (unsloth Q8_0 shards), other gate/up swapped to unsloth UD-Q4_K_XL imatrix Q4_K; BF16 n-gram table unchanged",
  methodology=dict(reference="Alibaba API official continuations (top-five logprobs), not BF16", dataset="qwen38-flash-alibaba-100 (100 prompts, 5,696 target tokens, ctx 4096) and qwen38-flash-alibaba-long (12 continuations, 766 target tokens, ctx 32768)", ctx=4096, tool="ds4 gguf-tools/quality-testing/score_official; llama.cpp control scorer score_llama for Unsloth files", L=[82,103,104]),
  table_header_L=91,
  variants=[dict(name="Q8-UDQ4K", size=188.57, unit="GiB", basis="disk", mem=93.19,
     m={"nll":"0.2842","api_top1":"92.96%","topn_recall":"89.49%","pair_order_agree":"92.13%","top5_mae":"0.0917","greedy_first_token":"85/100","nll:long":"0.1207","dnll_vs_base":"−0.0064"},
     L=94, L_extra=[25,97,109])],
  ladder=[dict(name="base antirez Q4 (this box)", m={"nll":"0.2906","api_top1":"92.08%","topn_recall":"88.62%","pair_order_agree":"91.58%","top5_mae":"0.0985","greedy_first_token":"84/100"}, L=93)],
  cross=[
    dict(of="antirez/qwen3.8-flash-next-gguf", variant="Q4", row_label="base antirez Q4", size=69.7, unit="GiB", basis="resident", m={"nll":"0.2906","nll:long":"0.1253","dnll":"+0.0064 (+0.0026..+0.0095)"}, L=111),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-Q5_K_XL", size=96.7, unit="GiB", basis="resident", m={"nll":"0.2831","greedy_first_token":"88/100","nll:long":"0.1209","dnll":"−0.0011 (−0.0032..+0.0010)"}, L=108),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-Q4_K_XL", size=76.9, unit="GiB", basis="resident", m={"nll":"0.2861","greedy_first_token":"86/100","nll:long":"0.1229","dnll":"+0.0019 (+0.0005..+0.0034)"}, L=110),
  ],
  flags=["Reference is Alibaba API top-five logprobs, not BF16; card states a BF16 KLD is not obtainable for this file [L82, L121-123]",
         "Unsloth files scored with a llama.cpp control scorer, this file with ds4 — different runtimes in one table [L103-110]",
         "Headline claims 'Equivalent quality to Unsloth's UD-Q5_K_XL'; table shows Q5 0.0011 NLL better on the short set, CI spanning zero [L19, L108, L114]",
         "Small long set: 12 continuations, 766 target tokens [L82]",
         "Not an original quant: tensors taken from unsloth and antirez files [L69-75]",
         "188.57 GiB file includes 95.37 GiB BF16 n-gram table read from disk; 93.19 GiB resident [L25, L28]"])

repo(repo="Robot-Haus/Qwen3.8-Flash-Next-oQ3.5e-fp16-mtp", sha="612c2b7889c91698564e0d8df23c2859d273902c", fmt="MLX",
  runtime="oMLX 0.7.0rc1", weights_modified=False,
  method="oMLX oQ 3-bit mixed precision, group size 64, FP16; custom imatrix corpus (3,629 samples) extending oMLX stock mix",
  methodology=dict(reference="none (no BF16 or base-model comparison)", dataset="partial samples: MMLU 1000/14042, HellaSwag 300/10042, TruthfulQA 300/817, HumanEval full 164, MBPP 200/500, LiveCodeBench 100/1055, SafetyBench 300/11435; no-thinking", ctx=None, tool="oMLX 0.7.0rc1 (M1 Ultra 128GB)", L=[179,387]),
  table_header_L=169,
  variants=[dict(name="oQ3.5e-fp16", size=None,
     m={"bench:MMLU":"85.4%","bench:HellaSwag":"94.3%","bench:TruthfulQA":"95.3%","bench:HumanEval pass@1":"91.5%","bench:MBPP pass@1":"87.5%","bench:LiveCodeBench pass@1":"48.0%","bench:SafetyBench":"86.7%"},
     L=171, L_extra=[172,173,174,175,176,177])],
  flags=["No reference: absolute scores only, no BF16 or base-model comparison on the same harness [L167-179]",
         "Partial samples (e.g. LiveCodeBench 100 of 1055, HellaSwag 300 of 10042), self-reported, no-thinking [L179, L387]",
         "Card gives no file size; peak-memory figures (~94–112 GB) are speed-run measurements [L153-158]",
         "Name says 3.5e; card's quantization details say 3-bit group size 64 [L436-437]"])

repo(repo="klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP", sha="600715b31bed3833fb4a51d9ea5d684ea2c7f32c", fmt="AutoRound (GPTQ-format)",
  runtime="patched vLLM a5a30471 (PLE SSD patch)", weights_modified=False,
  method="Mixed precision: 2/3-bit routed experts (~2.998 effective bpw), 8-bit main projections, BF16 sensitive paths, 4-bit MTP experts",
  methodology=dict(reference="original weights (precision not stated)", dataset="not stated", ctx=None, tool="not stated", L=[30]),
  variants=[dict(name="AutoRound 3bpw + MTP", size=47.3, unit="GiB", basis="vram", bpw=2.998, m={"kld":"0.1208"}, L=30, L_extra=[25,40])],
  flags=["Single KLD figure with no dataset, context, token count or tool stated [L30]",
         "47.3 GiB is loaded VRAM; the 95.37 GiB BF16 PLE table stays on SSD [L39-40]",
         "2.998 bpw refers to routed experts only [L25]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-oQ4", sha="dde9d8d76ace98086f8cfaac47ff589dd64b576c", fmt="MLX",
  runtime="oMLX / MLX-VLM with qwen4_exp", weights_modified=False,
  method="oQ4 mixed precision: 4-bit affine base, 228 protected modules at 5/8-bit, group size 32; sensitivity from a 4-bit proxy",
  methodology=None,
  variants=[dict(name="oQ4", size=111.69, unit="GB", basis="disk", L=65)],
  flags=["No quality claims (speed-only measurements and generation smoke tests) [L101-115, L141-144]",
         "MTP head not included [L70]"])

repo(repo="kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-imatrix-GGUF", sha="9abf014a574f5217070b4cad245bd93bdf2b9fb0", fmt="GGUF (ROCmFP4)",
  runtime="kingjones30/ROCmFPX fork (or charlie12345/ROCmFPX d3ca537 + bundled patches)", weights_modified=True,
  method="Q4_0_ROCMFP4_FAST (TYPE_101, 4.251 bpw) on all tensors incl. PLE, Q6_K lm head; imatrix from bartowski calibration_datav3 computed on the 4-bit model; source orcarouter/Qwen3.8-Flash-Next-Uncensored BF16",
  methodology=dict(reference="none (no BF16 or base-model PPL; compared to own non-imatrix sibling)", dataset="WikiText-2 raw test (wiki.test.raw)", ctx=512, tool="llama-perplexity (ROCmFPX)", L=[53,226]),
  table_header_L=55,
  variants=[dict(name="ROCmFP4-FAST imatrix", size=87.9, unit="GiB", basis="disk", bpw=4.27, m={"ppl":"5.0337 ± 0.031","ppl_delta_pct":"−5.9%","ppl:4bit_head":"5.1075"}, L=58, L_extra=[60,71])],
  cross=[dict(of="kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF", variant="plain FAST (no imatrix)", size=87.9, unit="GiB", basis="disk", m={"ppl":"5.3465 ± 0.034"}, L=57, L_extra=[60])],
  flags=["Abliterated/uncensored source weights (orcarouter) [L5, L50, L149-151]",
         "No BF16 reference: delta is vs the publisher's non-imatrix sibling only [L55-58]",
         "imatrix computed on the 4-bit model, not BF16 (card states this) [L66-67]",
         "Size 87.9 GiB is the model GGUF; shards listed as 44.70 + 44.67 + 5.06 GB, plus 0.91 GB mmproj [L60, L199-202]"])

repo(repo="tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8", sha="58ccaf6e24f634a12483064d8f8aea51fcb662ca", fmt="MXFP4/FP8",
  runtime="tcclaviger/vllm :dev image only (AMD AI PRO R9700)", weights_modified=False,
  method="Quark RTN MXFP4 experts; FP8 MTP, n-gram table and attention; calibrated FP8 KV-cache scales", methodology=None,
  variants=[dict(name="MXFP4/FP8", size=125.8, unit="GB", basis="disk", L=19, L_extra=[21,22,23,24])],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables",
         "Publisher marks this repo as superseded by the GPTQ-protected model and calls it an initial RTN tech-test [L17, L68]"])

repo(repo="leoncca/Qwen3.8-Flash-Next-AWQ-g32", sha="ff2fb53a7e33f21b0d97967eae09b6c077ad7ff8", fmt="AWQ",
  runtime="private V100-optimized vLLM build (TP4); generic AWQ support insufficient", weights_modified=False,
  method="AWQ-format W4A16 g32 asymmetric, routed experts requantized from BF16 with damped GPTQ on captured activations (public80-v1mix20, 169,614 rows/layer); s1/s2 re-searched; PLE uses official FP8 weights; QSA FP8 KV scales preserved",
  methodology=dict(reference="BF16 official text model, transformers CPU full-vocab forward expanded to FP32", dataset="fixed stratified 66-record text subset, 39,101 scored tokens", ctx=None, metric="coarse-grained KL(quantized || BF16) from top-20 plus tail mass", tool="own collection via served vLLM + transformers CPU reference", L=[123,124,125,130,147]),
  table_header_L=134,
  variants=[dict(name="AWQ g32 (2026-10-17)", row_label="New", size=137.07, unit="GB", basis="disk",
     m={"kld_top20_tail":"0.0207","top1_pct":"97.06%","kld:instructions":"0.0278","top1:instructions":"95.56%","kld:dialogue":"0.0236","top1:dialogue":"96.17%","kld:documents_code":"0.0194","top1:documents_code":"97.49%","kld_bounds":"[0.020465, 0.317644]"},
     L=136, L_extra=[19,137,138,139,156])],
  ladder=[dict(name="previous release of this repo", row_label="Previous", m={"kld_top20_tail":"0.0305","top1_pct":"96.32%","kld_bounds":"[0.030235, 0.327424]"}, L=136, L_extra=[155])],
  flags=["KL is a top-20-plus-tail approximation (a lower bound); card's conservative full-KL bounds span [0.020465, 0.317644] [L147-156]",
         "Comparison includes runtime arithmetic and KV effects (CPU FP32 reference vs GPU FP16 serving), not weight quantization alone [L126-128, L161-162]",
         "Small eval: 66 records / 39,101 tokens [L123, L159]",
         "Earlier GSM8K/HumanEval/MBPP/IFEval results describe previous weights and are not claimed for this release [L174-177]",
         "PLE table is official FP8, not BF16 [L25-26]"])

repo(repo="pipenetwork/Qwen3.8-Flash-Next-MLX-6bit", sha="b4ccbdba8cd3605af41e97fc334b9eaccdb580ac", fmt="MLX",
  runtime="PipeNetwork MLX port (qwen38-flash-next-mlx)", weights_modified=False,
  method="Uniform 6-bit affine: experts and non-experts group 64, n-gram tables group 32; routers/gates/vision BF16",
  methodology=dict(reference="bfloat16 (upstream), identical windows through the same runtime", dataset="wikitext-2 test, 296,815 tokens in 145 windows of 2048; paired per-window ΔNLL bootstrapped (20,000 resamples)", ctx=2048, tool="own MLX runtime", L=[72,73,74,75]),
  table_header_L=77, shared_table="pipenetwork ladder",
  variants=[dict(name="6bit", size=148.0, unit="GB", basis="disk", m={"ppl":"4.4767","dnll_vs_bf16":"+0.0013 [−0.0003, +0.0029]","windows_worse":"81/145"}, L=81, L_extra=[57])],
  ladder=[
    dict(name="8bit", size=192.2, unit="GB", basis="disk", m={"ppl":"4.4749","dnll_vs_bf16":"+0.0009 [−0.0003, +0.0021]","windows_worse":"73/145"}, L=80),
    dict(name="mixed-4_8bit", size=106.2, unit="GB", basis="disk", m={"ppl":"4.5286","dnll_vs_bf16":"+0.0128 [+0.0109, +0.0148]","windows_worse":"128/145"}, L=82),
    dict(name="4bit", size=103.8, unit="GB", basis="disk", m={"ppl":"5.3914","dnll_vs_bf16":"+0.1872 [+0.1778, +0.1968]","windows_worse":"145/145"}, L=83),
  ],
  baseline={"ppl":"4.4708","size":"360.0 GB","L":79},
  flags=["Only wikitext-2 PPL; no held-out or task benchmarks",
         "4-bit ablation builds (not published) are cited in a separate table [L93-101]"])

repo(repo="lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8", sha="a9786bf9d5bdff3ef3adcb062cd644b8b2ff2439", fmt="NVFP4 (+FP8 PLE)",
  runtime="patched SGLang (lovedheart/qwen38-flash-next docker image)", weights_modified=False,
  method="ModelOpt v0.46.0 NVFP4 W4A4 routed experts only; PLE tables from Qwen FP8 revision; rest BF16; calib 128 cnn_dailymail articles @512",
  methodology=dict(reference="BF16 runs on an earlier checkpoint revision", dataset="GSM8K full 1319 (t0.6/top-p 0.95/max 8192); AIME26 30 problems x 8 (t1.0/max 130k)", ctx=None, tool="not stated", L=[223,225,226,228]),
  table_header_L=223,
  variants=[dict(name="NVFP4", size=135, unit="GB", basis="disk", m={"bench:GSM8K":"97.27","bench:AIME26 pass@1":"98.75"}, L=225, L_extra=[226,183])],
  baseline={"bench:GSM8K":"97.12–97.50","L":225},
  baseline2={"bench:AIME26 pass@1":"100","L":226},
  flags=["BF16 baselines from an earlier checkpoint revision; card says treat as indicative [L228-230]",
         "AIME26 figure measured on a previous revision of this checkpoint (PLE tables differ) [L232-233]",
         "Card body describes itself as 'RadixArk Qwen3.8-Flash-Next-NVFP4' and its usage example loads RadixArk/Qwen3.8-Flash-Next-NVFP4; evaluation text and figures match that card [L110, L191, L220-236]",
         "Card notes long agentic generations run longer than BF16 [L235-236]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-oQ6", sha="aabd8cdf4cfdb414d8f4b20fcb0c9a82452fee84", fmt="MLX",
  runtime="oMLX / MLX-VLM with qwen4_exp", weights_modified=False,
  method="oQ6 mixed precision: 6-bit affine base, 228 protected modules at 6/8-bit, group size 32; sensitivity from a 4-bit proxy",
  methodology=None,
  variants=[dict(name="oQ6", size=155.78, unit="GB", basis="disk", L=65)],
  flags=["No quality claims (speed-only measurements and generation smoke tests)",
         "MTP head not included [L70]"])

repo(repo="devan-carlin/Qwen3.8-Flash-Next-W4A16", sha="40b8f18df4d4a32cb6e687a51c78207e5e438522", fmt="compressed-tensors W4A16",
  runtime="patched vLLM (devan-carlin/vllm xpu-qwen4exp) on Intel Arc Pro B70", weights_modified=False,
  method="Experts int4 group-128 (W4A16) via compressed-tensors; lm_head, embeddings, MTP, PLE, vision, gates, hyper-connections, indexer, linear_attn, shared experts kept BF16",
  methodology=None,
  variants=[dict(name="W4A16", size=77, unit="GB", basis="trunk_only", L=30, L_extra=[51])],
  flags=["No quality claims (decode speed only) [L40-43]",
         "~77GB covers the 17 weight shards; the ~96GB PLE table ships separately in the repo [L30, L52, L77-79]",
         "Card reports repetition/incoherent loops above temperature 0.7 and at greedy on this checkpoint [L110]"])

repo(repo="petergilani/Qwen3.8-Flash-Next-MLX-6bit", sha="ce5f824d4ada8d028cfd5a163525c3bccb04fa63", fmt="MLX",
  runtime="mlx-lm 0.31 + bundled ngram_tier.py (n-gram table streamed from paired repo)", weights_modified=False,
  method="Routed experts 6-bit g64, dense/attention linears 8-bit; vision and MTP kept; BF16 n-gram table externalized to petergilani/Qwen3.8-Flash-Next-bf16-ngram",
  methodology=dict(reference="not stated", dataset="not stated", ctx=None, tool="not stated", L=[45,46]),
  variants=[dict(name="6-bit (n-gram from disk)", size=106.52, unit="GB", basis="trunk_only", m={"ppl":"4.4765"}, L=46, L_extra=[18])],
  flags=["Single perplexity figure with no dataset, reference, context or token count stated [L45-46]",
         "106.52 GB excludes the 102.4 GB BF16 n-gram table shipped in a separate repo [L18, L21, L32-33]"])

repo(repo="ashuaria/Qwen3.8-Flash-Next-GGUF-FineTuning", sha="147b9cc10f74e962484b9ffac9409b18557e96fd", fmt="GGUF",
  runtime="llama.cpp / Unsloth Desktop (per card)", weights_modified=False,
  method="not stated (card is Unsloth's banner plus Qwen's base-model card)", methodology=None,
  variants=[dict(name="not listed on card", size=None)],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables",
         "Card text is Unsloth's GGUF banner plus Qwen's base card; it names no files or sizes and describes no fine-tuning despite the repo name [L11-31, L37-104]"])

repo(repo="Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-GGUF", sha="151ad0cb1d5f9b64f53a995a36f9f5d430a85abd", fmt="GGUF (custom qwen4exp schema, ds4)",
  runtime="Baekpica/ds4 dfm branch (planned; validation pending)", weights_modified=False,
  method="MQ-NG5-Q2K-Q4-SPLIT-E4: interior routed gate/up Q2_K (no imatrix), layers 0,1,46,47 Q4_K, down Q4_K + Q4_0 tail, PLE Q5_1, most always-active Q8_0",
  methodology=None, table_header_L=41,
  variants=[dict(name="MQ-NG5-Q2K-Q4-SPLIT-E4", size=98501648608, unit="bytes", basis="disk", L=52, L_extra=[43]),
            dict(name="MQ-NG5-Q4-SPLIT-E5 (projected, not built)", size=93.85, unit="GB", basis="disk", L=44)],
  flags=["No quality claims; card states model quality remains provisional pending validation gates [L31, L88-97]",
         "E5 size is a projection for an unbuilt artifact [L44]",
         "Snapshot README contains a second card (Mixed-Quant SSD-PLE GGUF) concatenated from L128 on"])

repo(repo="Baekpica/Qwen3.8-Flash-Next-GGUF", sha="5dfe6f0a6baa4776b2f5943fc264f6d17ca098b1", fmt="GGUF (custom qwen4exp schema, ds4)",
  runtime="Baekpica/ds4 dfm branch (runtime support under validation)", weights_modified=False,
  method="Lossless BF16 conversion and Q8_0 conversion", methodology=None,
  variants=[dict(name="BF16", size=360011029056, unit="bytes", basis="disk", L=45),
            dict(name="Q8_0", size=192201208384, unit="bytes", basis="disk", L=58)],
  flags=["No quality claims; only structural/byte-level verification [L43-60]",
         "Card says files are conversion artifacts, not a ready-to-run release [L72]"])

repo(repo="intel-ai/Qwen3.8-Flash-Next-MXFP8-CT-AutoRound", sha="8c4d7050d293143b1a539cf079b3994c799e5499", fmt="MXFP8 (compressed-tensors)",
  runtime="vLLM (TP>=2, moe_backend=marlin)", weights_modified=False,
  method="AutoRound 0.16.0 model-free RTN MXFP8 W8A8 g32 on all text-tower Linear incl. 73,728 routed experts; embeddings, lm_head, vision, MTP, PLE, hyper-connections, routers, in_proj_a/b, shared experts BF16",
  methodology=dict(reference="BF16 reference table supplied for the base checkpoint (not a local run, no stderr)", dataset="gsm8k 5-shot chat (1,319), piqa 0-shot (1,838), mmlu 0-shot (14,042), hellaswag 0-shot (10,042)", ctx=8192, settings="seed=42, batch_size=32, enable_thinking=false, max_gen_toks 2048", tool="lm-eval 0.4.13, vllm backend, TP=2 on 2x B300", L=[111,116,117,126,185]),
  table_header_L=161,
  variants=[dict(name="MXFP8", row_label=["Δ", "This checkpoint"], size=240.39, unit="GB", basis="disk", bpw=10.68,
     m={"bench:GSM8K strict":"0.9704","bench:GSM8K strict stderr":"±0.0047","bench:MMLU":"0.8514","bench:MMLU stderr":"±0.0029","bench:PIQA":"0.8210","bench:HellaSwag":"0.6913","bench_avg":"0.8335","delta_avg":"−0.26 pp","delta:MMLU":"−1.37 pp"},
     L=164, L_extra=[52,54,126,130,165])],
  baseline={"bench:GSM8K strict":"0.9674","bench:MMLU":"0.8652","bench:PIQA":"0.8194","bench:HellaSwag":"0.6927","bench_avg":"0.8362","L":163},
  flags=["BF16 row is a supplied reference table, not a matched local run; no BF16 stderr [L37-39, L185-188]",
         "Publisher reports MMLU −1.37 pp (−4.8σ) and says not to describe it as lossless [L39-40, L172-177]",
         "Same four tasks as Intel/Qwen3.8-Flash-Next-W4A16-AutoRound, whose BF16 row reads 0.9673 / 0.8651 vs 0.9674 / 0.8652 here [L163]",
         "Saturated/easy benchmarks; thinking disabled [L118]",
         "Shared experts left BF16 at export despite plan (deviation documented) [L190-197]"])

repo(repo="vkshdev/Qwen3.8-Flash-Next-FP8-180B", sha="1e0c4cfd485dc2463c517eaea571a5e5682c2978", fmt="FP8",
  runtime="Transformers/vLLM/SGLang (per card)", weights_modified=False,
  method="Fine-grained FP8, block size 128 (card text copied from Qwen's official FP8 card)", methodology=None,
  variants=[dict(name="FP8", size=None)],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables",
         "Card is Qwen's official FP8 card text, incl. usage examples loading Qwen/Qwen3.8-Flash-Next-FP8; the only quality statement is the copied 'nearly identical to those of the original model' line [L11-18, L380]"])

repo(repo="risk-magician/Qwen3.8-Flash-Next-AMX_Q8R.amxq", sha="50647bc1f4aaab001949d7e2d47201d177b252cb", fmt="AMX_Q8R (.amxq)",
  runtime="not stated", weights_modified=False, method="not stated on card", methodology=None,
  variants=[dict(name="AMX_Q8R", size=None)],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables",
         "Card is Qwen's base-model card verbatim; it does not describe the AMX_Q8R format, files or sizes [L12-30]"])

repo(repo="scorpoon/Qwen3.8-Flash-Next-exl3-SC", sha="81652966b01ea24b2b5115a0476e95f25280c7f7", fmt="EXL3",
  runtime="ExLlamaV3 >= v1.5.3 / v1.5.4 / v1.6.0 (per build)", weights_modified=False,
  method="ExLlamaV3 convert.py with sc_optimize bit allocation; 8-bit output head; separate 6-bit n-gram table (not uploaded); calibration 217–467 x 2048 rows of agentic / self-sampled traces",
  methodology=dict(reference="Qwen3.8-Flash-Next BF16", dataset="held-out evaluation trace (~45k tokens, disjoint from calibration data)", ctx=None, noise_floor="KLD 0.0027", tool="not stated (ExLlamaV3)", L=[55]),
  table_header_L=57,
  variants=[
    dict(name="3.05bpw h8 SC", size=49.56, unit="GiB", basis="vram", bpw=3.05, m={"mean_kld":"0.01543","ppl":"1.4041","ppl_delta_pct":"+1.41%"}, L=60, L_extra=[31]),
    dict(name="3.15bpw h8 SC", size=50.44, unit="GiB", basis="vram", bpw=3.15, m={"mean_kld":"0.01454","ppl":"1.4016","ppl_delta_pct":"+1.23%"}, L=61, L_extra=[32]),
    dict(name="2.05bpw h8 SC", size=34.25, unit="GiB", basis="vram", bpw=2.05, m={"mean_kld":"0.05761","ppl":"1.4697","ppl_delta_pct":"+6.15%"}, L=62, L_extra=[33]),
  ],
  cross=[
    dict(of="turboderp/Qwen3.8-Flash-Next-exl3", variant="3.05bpw h5", m={"mean_kld":"0.01730","ppl":"1.4071","ppl_delta_pct":"+1.63%"}, L=63),
    dict(of="turboderp/Qwen3.8-Flash-Next-exl3", variant="2.05bpw h4", m={"mean_kld":"0.06840","ppl":"1.4879","ppl_delta_pct":"+7.46%"}, L=64),
  ],
  baseline={"ppl":"1.3846","L":59},
  flags=["Sizes are weight shards only; the 36.4 GiB n-gram table is not uploaded and must come from turboderp's repo [L29-33, L51]",
         "Eval is a single ~45k-token trace of unstated content; tool/context not stated [L55]",
         "Calibration for 3.15bpw is agentic traces and for 2.05bpw in-domain self-sampled traces; eval is described as disjoint from calibration [L39-43, L55]"])

repo(repo="Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2", sha="d6faf17cadf826849848a58a65dc34f961f5b535", fmt="HGN (Halogen)",
  runtime="halogen-flash-server (not loaded/qualified for current files)", weights_modified=True,
  method="bf16-to-hgn native conversion (profile v2 and ht43) of UkisAI Swift 1.5 fine-tune BF16 source", methodology=None,
  variants=[dict(name="v2 (swift15)", size=66687678432, unit="bytes", basis="trunk_only", L=40),
            dict(name="HT43 (swift15)", size=57618253792, unit="bytes", basis="trunk_only", L=41)],
  flags=["No quality claims for current files; card states they have not been loaded or inference-qualified and historical results do not apply [L34, L55, L69]",
         "Source is UkisAI's Swift 1.5 fine-tune, not the base model [L32, L77]",
         "Sizes are main checkpoint only; n-gram and vision files (~52 GB) not uploaded [L45]"])

repo(repo="wemoh/Qwen3.8-Flash-Next-strixw", sha="b64f3a69070c68721cbbf5ffa701ecee164b0991", fmt="strixw",
  runtime="strixite only (AMD Strix Halo gfx1151)", weights_modified=False,
  method="Asymmetric integer group quant: routed experts and GDN input 4-bit g128, other dense projections 8-bit g64, MTP input proj 4-bit g64; n-gram table int8 per-row",
  methodology=dict(reference="FP32 outputs of the unquantized model", dataset="242 positions of three reference prompts; terminal-bench-mini core19", ctx=None, tool="strixite", L=[93,94,95]),
  variants=[dict(name="strixw (U-gdn_in-g128)", size=65.9, unit="GiB", basis="trunk_only", bpw=4.25,
     m={"mean_kld":"0.028","top1":"226 of 242","agent_tasks":"19 of 19"}, L=94, L_extra=[52,84,93,95])],
  flags=["Tiny eval: 242 token positions from three prompts [L93-94]",
         "~116 GiB download includes the 48.3 GiB int8 n-gram table read from SSD; 65.9 GiB is main weights [L48, L52-54]",
         "4.25 bits refers to routed weights only [L84]"])

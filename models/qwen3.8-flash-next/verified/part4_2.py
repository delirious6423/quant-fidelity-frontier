# Queue 2 — extracted from verbatim README snapshots at pinned commits (snapshots/index.json, captured 2026-10-17).
# Each variant/claim cites README line numbers (L) at that sha.
# size basis: disk = full download unless noted; vram/resident/ram_in_use = memory figures; trunk_only = excludes n-gram table.
R = []
def repo(**k): R.append(k); return k

repo(repo="cygnal/Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4-GGUF", sha="b7a350f3ba4c29962cefe9e69af225f142cda9bc", fmt="GGUF",
  runtime="llama-flash-next fork of llama.cpp (qwen4exp branch) only", weights_modified=True,
  method="Heretic v1.3.0+custom abliteration of BF16 -> Q8_0 intermediate -> IQ4_XS experts, Q8_0 dense, Q6_K output, BF16 token_embd; NGQ4: per-layer n-gram table IQ4_NL",
  methodology=dict(reference="none (no BF16 run; compared with other GGUF builds incl. a non-abliterated IQ4_XS NGQ4 build)",
    dataset="HumanEval+ (EvalPlus), 164 problems", ctx=None, settings="enable_thinking false, temperature 0.0, max_tokens 1024", n_runs="not stated",
    tool="EvalPlus via llama-server", L=[79,81]),
  table_header_L=83,
  variants=[dict(name="Heretic2 IQ4_XS NGQ4", size=92, unit="GB", basis="disk",
    m={"bench:HumanEval+ base pass@1":"84.1%","bench:HumanEval+ plus pass@1":"79.3%","abliteration_kl_vs_original":"0.0818"}, L=85, L_extra=[33,125])],
  ladder=[
    dict(name="Q6_K_v7_v7", size=129, unit="GB", m={"bench:HumanEval+ base pass@1":"88.4%","bench:HumanEval+ plus pass@1":"82.9%"}, L=86, L_extra=[93]),
    dict(name="Q6_K", size=129, unit="GB", m={"bench:HumanEval+ base pass@1":"88.4%","bench:HumanEval+ plus pass@1":"81.7%"}, L=87, L_extra=[93]),
    dict(name="Base flash-next IQ4_XS NGQ4 (not abliterated)", size=None, m={"bench:HumanEval+ base pass@1":"82.3%","bench:HumanEval+ plus pass@1":"78.0%"}, L=88),
    dict(name="Uncensored IQ4_XS NGQ4", size=None, m={"bench:HumanEval+ base pass@1":"81.7%","bench:HumanEval+ plus pass@1":"78.7%"}, L=90),
    dict(name="Heretic2 IQ4_XS run in Halogen (BYO, no overlay)", size=None, m={"bench:HumanEval+ base pass@1":"65.2%","bench:HumanEval+ plus pass@1":"61.0%"}, L=91),
  ],
  cross=[dict(of="peonist-ai/halogen-qwen3.8-flash-next", variant="Halogen Official w4b (native + overlay)", size=None,
    m={"bench:HumanEval+ base pass@1":"82.3%","bench:HumanEval+ plus pass@1":"78.0%"}, L=89, L_extra=[97])],
  flags=["Abliterated (Heretic) weights: not a faithful quant of Qwen's model [L27, L119-121]",
         "No BF16 reference; only comparisons among quantized builds [L83-91]",
         "Single code benchmark (164 problems, thinking off, 1024-token cap), no run count or error bars; card itself calls the 79.3% vs 78.0% gap 'within noise' [L81, L128]",
         "NGQ4 claimed to have 'no measurable accuracy impact through 32K context' with no numbers given [L38]",
         "Q6_K builds' origin not stated; 129 GB size given only in prose [L86-87, L93]",
         "Abliteration KL 0.0818 is Heretic's divergence from the original model, not quantization error [L123-126]"])

repo(repo="CMSManhattan/JiRackDeltaNet_180b", sha="82d1140b21d0ae84cb294e9a2d25e8325d377b0e", fmt="BF16 safetensors (JiRack port); GGUF planned",
  runtime="PyTorch (JiRack code); llama.cpp/Ollama planned", weights_modified=False,
  method="Hand-rolled JiRack port with BitLinear layers at lambda=0 (passthrough); ternary QAT not started; quantized GGUFs not published",
  methodology=None, table_header_L=125,
  variants=[dict(name=n, size=None) for n in ["BF16 (JiRack export)", "Q8_0 (planned)", "Q4_K_M (planned)", "Q3_K_M (planned)", "Q2_K (planned)", "TQ2_0 (planned, after QAT)"]],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables [L234, L276]",
         "No quantized files published: QAT 'not started', GGUF/TQ2_0 builds 'not yet published' [L67-68, L123]",
         "Card states weights are numerically identical to the base model until QAT [L70]",
         "Planned GGUF sizes (~360/~190/~110/~88/~75/~65–95 GB) are estimates from parameter count, not files [L123-132]",
         "Ollama run commands are listed for q4/q3 builds while the card says Ollama builds are planned after QAT [L100, L157-158]"])

repo(repo="dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4", sha="15ef113c7090844f97f355f400169791dc76c2e3", fmt="NVFP4",
  runtime="SGLang (2x DGX Spark TP2, modelopt_fp4)", weights_modified=True,
  method="Refusal-removal weight edit ('CRACK') on an NVFP4 build; routed experts NVFP4 W4A4, attention/shared experts/PLE/vision/MTP higher precision",
  methodology=dict(reference="base model in NVFP4 (not BF16)", dataset="MMLU-logit, 40 Q/subject (2,280 questions), greedy; hard multi-step question set (college math/physics/chemistry, formal logic; size not stated) at xhigh, greedy",
    ctx=None, tool="not stated (same harness for both)", L=[55,130,132]),
  table_header_L=132,
  variants=[dict(name="CYBERSECURITY NVFP4", size=None,
    m={"bench:MMLU":"81.27%","bench:MMLU delta":"-0.84 pp","bench:hard multi-step accuracy":"98.3%","reasoning_trace_median":"422 tok (0.97×)"}, L=134, L_extra=[59,60])],
  baseline={"bench:MMLU":"82.11%","bench:hard multi-step accuracy":"96.7%","reasoning_trace_median":"434 tok","L":134,"L_extra":[59,60]},
  flags=["Abliterated / refusal-removed weights: not a faithful quant [L29, L43-51]",
         "Reference is the base model in NVFP4, so the deltas measure the refusal removal, not quantization loss [L132]",
         "MMLU subsampled to 40 questions per subject; hard-question set size not stated [L55, L130]",
         "Per-subject MMLU swings up to -22.5 pp (machine_learning) against a -0.84 pp overall [L140, L190]",
         "No absolute size given on card"])

_pipe_method = dict(reference="bfloat16 (upstream), same runtime and windows", dataset="wikitext-2 (test), 296,815 tokens in 145 windows of 2048",
    ctx=2048, n_chunks=145, tool="bundled mlx qwen4_exp.py runtime; paired per-window ΔNLL, bootstrap 20,000 resamples", L=[72,73,74,75])
_pipe_rows = {
  "8bit": dict(name="8bit", size=192.2, unit="GB", basis="disk", m={"ppl":"4.4749","dnll_vs_bf16":"+0.0009 [−0.0003, +0.0021]","windows_worse":"73/145"}, L=80),
  "6bit": dict(name="6bit (pipenetwork/Qwen3.8-Flash-Next-MLX-6bit)", size=148.0, unit="GB", m={"ppl":"4.4767","dnll_vs_bf16":"+0.0013 [−0.0003, +0.0029]","windows_worse":"81/145"}, L=81),
  "mixed": dict(name="mixed-4_8bit", size=106.2, unit="GB", basis="disk", m={"ppl":"4.5286","dnll_vs_bf16":"+0.0128 [+0.0109, +0.0148]","windows_worse":"128/145","ppl_delta_pct":"+1.3%"}, L=82, L_extra=[101]),
  "4bit": dict(name="4bit", size=103.8, unit="GB", basis="disk", m={"ppl":"5.3914","dnll_vs_bf16":"+0.1872 [+0.1778, +0.1968]","windows_worse":"145/145","ppl_delta_pct":"+20.6%"}, L=83, L_extra=[97]),
}
_pipe_ablations = [
  dict(name="4-bit + hyper-connection read gates at 8-bit (unpublished ablation)", size=None, m={"ppl":"4.9744","ppl_delta_pct":"+11.3%"}, L=98),
  dict(name="4-bit + attention/DeltaNet/shared experts/PLE projections at 8-bit (unpublished ablation)", size=None, m={"ppl":"4.8969","ppl_delta_pct":"+9.5%"}, L=99),
  dict(name="4-bit + embed_tokens and lm_head at 8-bit (unpublished ablation)", size=None, m={"ppl":"5.2843","ppl_delta_pct":"+18.2%"}, L=100),
]
_pipe_flags = ["Uses wikitext-2 PPL; agentionai's AP card reports wikitext is memorised by this architecture's n-gram table",
               "MTP head not included in the build [L24]",
               "Runs only with the repo's bundled runtime (open, unmerged mlx-lm PR plus three fixes) [L29-30, L42-49]"]

def _pipe(rid, sha, own, method, extra_flags):
    lad = [dict(v) for k, v in _pipe_rows.items() if k != own] + ([dict(a) for a in _pipe_ablations] if own != "mixed" else [dict(a) for a in _pipe_ablations])
    for e in lad: e.pop("basis", None)
    return repo(repo=rid, sha=sha, fmt="MLX", runtime="mlx-lm + bundled qwen4_exp.py (trust_remote_code)", weights_modified=False,
      method=method, methodology=dict(_pipe_method), table_header_L=77, variants=[dict(_pipe_rows[own], L_extra=list(_pipe_rows[own].get("L_extra", [])) + [57])],
      ladder=lad, baseline={"ppl":"4.4708","size_GB":"360.0","L":79}, flags=_pipe_flags + extra_flags)

_pipe("pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit", "b2c422f3c643e36f04227a64d61796b44a4b1029", "mixed",
      "MLX affine: routed experts 4-bit g64, n-gram tables 4-bit g32, attention/DeltaNet/hyper-connections/shared experts/embeddings/lm_head 8-bit g64, routers etc. bf16",
      ["95% CI excludes zero: +0.0128 nats/token, worse than bf16 on 128 of 145 windows [L82]"])
_pipe("pipenetwork/Qwen3.8-Flash-Next-MLX-8bit", "f1c23d4870dd35c6c940945a92d8d2bc564bf43a", "8bit",
      "MLX affine 8-bit: experts and dense g64, n-gram tables g32, routers etc. bf16",
      ["95% CI straddles zero; card calls 8-bit statistically indistinguishable from bf16 on this corpus [L80, L89]"])
_pipe("pipenetwork/Qwen3.8-Flash-Next-MLX-4bit", "aa7c790e804bbf9d491ddb109c3d61bc4a555f7c", "4bit",
      "MLX affine 4-bit everywhere (experts/dense g64, n-gram tables g32), routers etc. bf16",
      ["Publisher calls this uniform 4-bit build 'dominated' and published for completeness: +20.6% PPL, worse on 145/145 windows [L83, L89]"])

repo(repo="Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata", sha="9ac833ee7936d8cf6921d7bcd5434f98a754a399", fmt="NVFP4+Q8_0 (Strata pack)",
  runtime="Strata NVFP4 fork only", weights_modified=False,
  method="Per-layer mix of the publisher's NVFP4 GPTQ experts and Q8_0 experts (byte copies), chosen by a KL model fitted to 9 measured packs under a 110 GiB RAM budget; dense GGUF Q8_0/BF16, Qwen FP8 PLE",
  methodology=dict(reference="publisher's own Q8_0 Strata build (Q8_0 experts and attention, BF16 router, Qwen FP8 PLE), not BF16",
    dataset="5,153 tokens of code, docs, English and Ukrainian chats and agent transcripts; plus thinking text written by the GPTQ model", ctx=None,
    tool="Strata NVFP4 0.1.42-nvfp4.1 (experts on CPU, route tail skip off)", L=[51,52,53,54,55,115,116]),
  table_header_L=38,
  variants=[dict(name="NVFP4 + Q8_0 mix", size=171.6, unit="GB", basis="disk", mem=105, mem_unit="GiB", mem_basis="ram_in_use",
    m={"top1_pct":"95.0%","top1:thinking":"97.4%","mean_kl":"0.0215","kl:thinking":"0.0058","ppl_vs_ref":"100.0%"}, L=41, L_extra=[120,130])],
  ladder=[
    dict(name="Q8_0 Strata (reference; Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata)", size=106, unit="GiB", basis="ram_in_use", m={"top1_pct":"100%"}, L=40),
    dict(name="NVFP4 GPTQ Strata (Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata)", size=69, unit="GiB", basis="ram_in_use",
         m={"top1_pct":"93.6%","top1:thinking":"96.5%","mean_kl":"0.0379","kl:thinking":"0.0124","ppl_vs_ref":"100.1%"}, L=43, L_extra=[121]),
  ],
  cross=[
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-Q4_K_XL", size=77, unit="GiB", basis="ram_in_use", m={"top1_pct":"94.1%"}, L=42),
    dict(of="nvidia/Qwen3.8-Flash-Next-NVFP4", variant="NVFP4 (converted to Strata by publisher)", size=69, unit="GiB", basis="ram_in_use", m={"top1_pct":"92.9%"}, L=44),
    dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-Q3_K_XL", size=58, unit="GiB", basis="ram_in_use", m={"top1_pct":"90.7%"}, L=45),
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="IQ3_S", size=52, unit="GiB", basis="ram_in_use", m={"top1_pct":"90.0%"}, L=46),
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="IQ3_XXS", size=45, unit="GiB", basis="ram_in_use", m={"top1_pct":"87.3%"}, L=47),
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="IQ2_XS", size=38, unit="GiB", basis="ram_in_use", m={"top1_pct":"85.0%"}, L=48),
    dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="Q2_0", size=37, unit="GiB", basis="ram_in_use", m={"top1_pct":"84.1%"}, L=49),
  ],
  flags=["Reference is the publisher's own Q8_0 build, not BF16 [L40, L51-55]",
         "Small eval: 5,153 tokens [L51]",
         "Thinking-text column uses text written by the GPTQ model, which the card notes favours GPTQ [L115-116]",
         "Comparison table sizes are RAM in use (engine + server); disk download is 171.6 GB (159.8 GiB) [L59, L130]",
         "Other publishers' files were re-run inside Strata with route tail skip off, not in their own runtimes [L52-53, L70-71]",
         "Runs only on the Strata NVFP4 fork [L98-99]"])

_sushi_xmethod = dict(reference="bf16 checkpoint", dataset="16 prompts x 512 tokens, scored to the first EOS (7186 positions)", ctx=None,
    settings="KV cache kv8 (Sushi-2bpw: bf16 KV); n-gram table not counted in sizes", tool="sushi")
_sushi_cross = lambda o: [
    dict(variant="oMLX oQ5e (repo not named on card)", size=83.97, unit="GiB", basis="vram", m={"kld":"0.0625","top1_pct":"92.40%"}, L=o),
    dict(of="ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit", variant="mixed-4-8bit", size=70.13, unit="GiB", basis="vram", m={"kld":"0.0818","top1_pct":"91.39%"}, L=o+1),
    dict(variant="oMLX oQ4e (repo not named on card)", size=69.21, unit="GiB", basis="vram", m={"kld":"0.1370","top1_pct":"88.87%"}, L=o+4),
    dict(of="ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw", variant="iQ-MLX 3.3bpw", size=50.60, unit="GiB", basis="vram", m={"kld":"0.1987","top1_pct":"86.28%"}, L=o+5),
    dict(variant="Vontra 4-bit g32 (TensorFold; repo not named on card)", size=75.60, unit="GiB", basis="vram", m={"kld":"0.2074","top1_pct":"85.01%"}, L=o+6),
]

repo(repo="beamster/Qwen3.8-Flash-Next-Sushi-2.6bpw", sha="0658b1533ff2467ac38e194313015d05e292cd5e", fmt="EXL3 (sushi pack)",
  runtime="sushi v1.0.4+ on Apple Silicon only", weights_modified=False,
  method="Routed experts EXL3 2.6 bpw (publisher's own converter); dense 8-bit affine; n-gram table 4-bit affine g32 on SSD",
  methodology=dict(_sushi_xmethod, L=[43,44,45]), table_header_L=47,
  variants=[dict(name="Sushi-2.6bpw (4-bit n-gram table)", size=74, unit="GiB", basis="disk", mem=43.95, mem_unit="GiB", mem_basis="vram",
    m={"kld":"0.1355","top1_pct":"89.08%","kld:bf16_ngram_table":"0.1303","bench:LLMProbe Hard 50":"88.0%"}, L=53, L_extra=[32,64,72])],
  ladder=[
    dict(name="Sushi-4bpw (bf16 n-gram table)", size=63.68, unit="GiB", basis="vram", m={"kld":"0.0592","top1_pct":"92.89%"}, L=49),
    dict(name="Sushi-3bpw (4-bit n-gram table)", size=49.33, unit="GiB", basis="vram", m={"kld":"0.1036","top1_pct":"90.31%","kld:bf16_ngram_table":"0.1006"}, L=52, L_extra=[83]),
    dict(name="Sushi-2bpw (4-bit n-gram table, bf16 KV)", size=34.97, unit="GiB", basis="vram", m={"kld":"0.2080","top1_pct":"85.94%"}, L=57),
  ],
  cross=_sushi_cross(50),
  flags=["Sizes are weights in GPU memory; the n-gram table (29.8 GiB here) stays on SSD and is excluded [L32, L44-45]",
         "Each pack scored with the n-gram table it ships (4-bit, 4/5-bit oMLX, bf16), so KLD rows mix table precisions [L62-63]",
         "Small eval: 16 prompts, 7186 scored positions [L43]",
         "Other publishers' packs were run in sushi; oMLX n-gram tables repacked into sushi's file [L64-65]",
         "LLMProbe Hard 50 is 50 questions, single run, no reference score [L71-72]"])

repo(repo="beamster/Qwen3.8-Flash-Next-Sushi-4bpw", sha="f25c0c4acdc44289a9cc6dab7f0eb788d1125ded", fmt="EXL3 (sushi pack)",
  runtime="sushi v1.0.0+ on Apple Silicon only", weights_modified=False,
  method="Routed experts EXL3 (publisher's own converter); dense 8-bit affine; n-gram table bf16 as released, read from SSD",
  methodology=dict(_sushi_xmethod, settings="KV cache kv8; Sushi-3bpw and Sushi-4bpw on a later sushi build than other rows; Sushi-2bpw on v1.0.4 with bf16 KV; n-gram table not counted", L=[46,47,48]),
  table_header_L=50,
  variants=[dict(name="Sushi-4bpw (bf16 n-gram table)", size=159, unit="GiB", basis="disk", mem=63.68, mem_unit="GiB", mem_basis="vram",
    m={"kld":"0.0592","top1_pct":"92.89%","kld:4bit_ngram_table":"0.0654","bench:LLMProbe Hard 50":"90.0%"}, L=52, L_extra=[33,83,72])],
  ladder=[
    dict(name="Sushi-3bpw (4-bit n-gram table)", size=49.33, unit="GiB", basis="vram", m={"kld":"0.1036","top1_pct":"90.31%","kld:bf16_ngram_table":"0.1006"}, L=55, L_extra=[84]),
    dict(name="Sushi-2.6bpw (4-bit n-gram table)", size=43.95, unit="GiB", basis="vram", m={"kld":"0.1355","top1_pct":"89.08%"}, L=56),
    dict(name="Sushi-2bpw (4-bit n-gram table, bf16 KV)", size=34.97, unit="GiB", basis="vram", m={"kld":"0.2080","top1_pct":"85.94%"}, L=60),
  ],
  cross=_sushi_cross(53),
  flags=["Sizes are weights in GPU memory; the 95.4 GiB bf16 n-gram table stays on SSD and is excluded [L33, L47-48]",
         "Rows were scored on different sushi builds and KV settings [L47]",
         "Headline row uses the bf16 n-gram table while most compared packs use 4-bit tables; with the 4-bit table Sushi-4bpw reads 0.0654 [L64, L83]",
         "Small eval: 16 prompts, 7186 scored positions [L46]",
         "chatnerdie's card re-measures this pack at KLD 0.006137 on a different prompt set and settings and says the numbers are not comparable [chatnerdie L61, L77]"])

repo(repo="FenomAI/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", sha="904afcb39146e3788f6861ff408c2cbe5a7106bc", fmt="GGUF",
  runtime="llama.cpp / Ollama / LM Studio", weights_modified=False,
  method="GSQ (Gumbel-Softmax scalar quant) + RCO per-tensor type allocation; n-gram shard fixed IQ4_NL",
  methodology=dict(reference="BF16 base model", dataset="5 zero-shot tasks (arc_easy, arc_challenge, hellaswag, winogrande, piqa) + AIME25, GPQA-Diamond, LiveCodeBench v6", ctx=None, tool="not stated", L=[119]),
  table_header_L=121,
  variants=[
    dict(name="Q2_0", size=66.4, unit="GB", basis="disk", bpw=2.40, mem=37.6, mem_unit="GB", mem_basis="resident", m={"zs_avg":"78.00","recovery_pct":"101.4%","bench:AIME25":"96.67","bench:GPQA-Diamond":"89.39","bench:LiveCodeBench v6":"81.14","task_avg":"89.07"}, L=124, L_extra=[75]),
    dict(name="IQ2_XS", size=68.0, unit="GB", basis="disk", bpw=2.50, mem=39.2, mem_unit="GB", mem_basis="resident", m={"zs_avg":"77.16","recovery_pct":"100.3%","bench:AIME25":"96.67","bench:GPQA-Diamond":"87.37","bench:LiveCodeBench v6":"83.43","task_avg":"89.16"}, L=125, L_extra=[76]),
    dict(name="IQ3_XXS", size=75.8, unit="GB", basis="disk", bpw=3.00, mem=47.0, mem_unit="GB", mem_basis="resident", m={"zs_avg":"77.23","recovery_pct":"100.4%","bench:AIME25":"100.00","bench:GPQA-Diamond":"91.41","bench:LiveCodeBench v6":"86.29","task_avg":"92.57"}, L=126, L_extra=[77]),
    dict(name="IQ3_S", size=83.6, unit="GB", basis="disk", bpw=3.50, mem=54.8, mem_unit="GB", mem_basis="resident", m={"bench:AIME25":"100.00","bench:GPQA-Diamond":"92.93","bench:LiveCodeBench v6":"86.86","task_avg":"93.26"}, L=127, L_extra=[78]),
  ],
  baseline={"zs_avg":"76.94","bench:AIME25":"100.00","bench:GPQA-Diamond":"91.92","bench:LiveCodeBench v6":"87.43","task_avg":"93.12","L":123},
  flags=["Results table and prose match ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF's card value for value; the card credits DASLab for the methods and compute and does not say who ran these evaluations or whether the files are ISTA-DASLab's [L63, L119-133, L226]",
         "Usage commands still contain the placeholder '<this-repo>' [L150, L164, L175]",
         "YAML license is apache-2.0 while the card says the weights inherit the base model's license [L6, L232]",
         "Card itself says >100% recovery is within ~1 SE and means parity [L129]",
         "Own BF16 baseline (LCB v6 87.43, GPQA 91.92) differs from Qwen's official 91.9 / 91.7",
         "mem = shard 1 (must be resident); 28.8 GB n-gram shard can stay on disk [L89-94]"])

repo(repo="mensaprodigy/Qwen3.8-Flash-Next-REAP-384-mlx-mtp", sha="ae3a5046145ad63460911b6947d4291244eb933f", fmt="MLX (oMLX oQ4e)",
  runtime="oMLX main (needs PR #3659 + MTP stacking fix)", weights_modified=True,
  method="sh0wie REAP-384 BF16 target (512 -> 384 experts per layer) + original 512-expert native MTP, converted with oMLX oQ4e (4-bit g64 primary, 392 overrides; PLE 4-bit g32)",
  methodology=None, table_header_L=129,
  variants=[dict(name="REAP-384 oQ4e + native MTP", size=None)],
  cross=[
    dict(of="sh0wie (REAP lineage; card [2] links sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit)", variant="Full Flash-Next Q4 (512 experts)", measured=False, size=None, m={"bench:HumanEval pass@1":"93.9%"}, L=131),
    dict(of="sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit", variant="REAP-384 Q4", measured=False, size=None, m={"bench:HumanEval pass@1":"92.1%"}, L=132),
    dict(of="sh0wie (REAP lineage)", variant="REAP-288 Q4", measured=False, size=None, m={"bench:HumanEval pass@1":"91.5%"}, L=133),
    dict(of="sh0wie (REAP lineage)", variant="REAP-256 Q4", measured=False, size=None, m={"bench:HumanEval pass@1":"88.4%"}, L=134),
  ],
  flags=["Pruned (REAP 512 -> 384 experts per layer): not a faithful quant [L19, L38]",
         "No quality measurement of this checkpoint; the HumanEval table is sh0wie's reported lineage figures, explicitly 'not direct measurements of this release' [L127, L165]",
         "Lineage figures are single-run HumanEval (164 problems) with no confidence intervals, per the card [L136]",
         "Throughput numbers are service counters, not a controlled benchmark [L140-146]",
         "No size stated on card"])

repo(repo="evsinlb/Qwen3.8-Flash-Next-oQ4e-fp16-mtp", sha="d2ec554e4697c19337b0775f13a35fbf59a83d8c", fmt="MLX (oMLX oQ4e)",
  runtime="oMLX 0.6.4 / 0.7.0.dev2", weights_modified=False,
  method="oQ4e: 4-bit affine g64 with 5-bit boosts on sensitive tensors; PLE 5.00 bits/param; imatrix 1024 x 512 (oqe_code_multilingual); fp16 non-quant tensors",
  methodology=dict(reference="none", dataset="3-scenario multi-turn agentic tool-loop battery (9 tool-call turns) + generation spot-checks", ctx=None, tool="not stated", L=[91,93]),
  table_header_L=33,
  variants=[dict(name="oQ4e-fp16-mtp", size=99.82, unit="GiB", basis="disk", mem=70.00, mem_unit="GiB", mem_basis="resident",
    m={"agentic_battery":"3/3"}, L=37, L_extra=[36,91])],
  flags=["Only quality evidence is a 3-scenario agentic smoke battery and spot-checks; no reference comparison or standard benchmark [L91-93]",
         "70.00 GiB is always-resident weights; 29.82 GiB PLE table can be mmapped [L35-37]",
         "Peak-memory figures in the throughput table are GB while sizes are GiB [L58-62, L37]"])

repo(repo="tls20/Qwen3.8-Flash-Next-oQ5e-mtp", sha="6bffcfdd9b9bcf83bc63a92b8cd1bcf430f632e2", fmt="MLX (oMLX oQ5e)",
  runtime="oMLX", weights_modified=False, method="oQ (oMLX v0.6.4) 5-bit, group 64, oQe enabled, MTP preserved",
  methodology=dict(reference="none stated", dataset="MBPP, HumanEval, MMLU, JMMLU, TruthfulQA, LiveCodeBench (subsets, versions, settings not stated)", ctx=None, tool="not stated", L=[79]),
  table_header_L=81,
  variants=[dict(name="oQ5e-mtp", size=None,
    m={"bench:MBPP":"88.6%","bench:HumanEval":"94.5%","bench:MMLU":"92%","bench:JMMLU":"90.6%","bench:TruthfulQA":"95.3%","bench:LiveCodeBench":"59%"}, L=83, L_extra=[84,85,86,87,88])],
  flags=["No methodology, sample sizes, settings or baseline for any score [L79-88]",
         "LiveCodeBench 59% with version unstated; Qwen's card reports 91.9 on LiveCodeBench v6, so the two are not comparable as given [L88]",
         "No disk size stated; only peak memory during throughput tests (84.7–99.6 GB) [L42-62]"])

repo(repo="lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000", sha="0e63a89e8ce4b0ee1d909d4cbabf6898e43a8add", fmt="NVFP4+FP8",
  runtime="patched SGLang (lovedheart fork; stock SGLang emits garbled output)", weights_modified=True,
  method="AIMER expert pruning 512 -> 448 per layer (calibration-free), NVFP4 routed experts + FP8 E4M3 128x128 blockwise weight-only attention/GDN projections",
  methodology=dict(reference="BF16 runs on an earlier checkpoint revision (as stated in the copied RadixArk evaluation section)", dataset="GSM8K (1319), AIME26 (30x8)", ctx=None, tool="not stated", L=[193,198]),
  table_header_L=193,
  variants=[dict(name="NVFP4-FP8 pruned (448 experts)", size=None, m={"bench:GSM8K":"97.27","bench:AIME26 pass@1":"98.75"}, L=195, L_extra=[196])],
  baseline={"bench:GSM8K":"97.12–97.50","L":195},
  baseline2={"bench:AIME26 pass@1":"100","L":196},
  flags=["Pruned (AIMER 512 -> 448 experts per layer): not a faithful quant [L21-32]",
         "Body below L91 is RadixArk's NVFP4 card text: it names 'RadixArk Qwen3.8-Flash-Next-NVFP4', 512 routed experts, BF16 non-expert tensors, a 135 GB size and a RadixArk model path, which contradict this repo's pruned 448-expert, FP8-attention description [L65-69, L94, L120-122, L161-167, L175]",
         "GSM8K/AIME26 numbers and footnotes are identical to RadixArk/Qwen3.8-Flash-Next-NVFP4's card; the card does not show they were measured on this pruned checkpoint [L193-203]",
         "No size stated for this checkpoint (135 GB is in the copied RadixArk text) [L167]"])

repo(repo="tapsin/TAPSIN-Qwen3.8-4B-Distill-GGUF", sha="994c5893423c8a97171d685b0c17e8950cb1b368", fmt="GGUF",
  runtime="llama.cpp (recent, Qwen3.5 support) / Ollama / LM Studio", weights_modified=True,
  method="GGUF quants of a 4B distilled model (Qwen3.5-4B architecture, trained on ~45,000 teacher traces)", methodology=None, table_header_L=30,
  variants=[dict(name=n, size=s, unit="GB", basis="disk", L=l) for n, s, l in [("Q4_K_M",2.783,32),("Q5_K_M",3.161,33),("Q6_K",3.563,34),("Q8_0",4.611,35),("BF16",8.666,36)]],
  flags=["Not a quant of Qwen3.8-Flash-Next: a distillation into the Qwen3.5-4B architecture despite base_model: Qwen/Qwen3.8-Flash-Next [L7, L14, L68]",
         "Distilled/fine-tuned weights [L14, L68]",
         "Headline table compares the distilled source model with Qwen3.5-4B base, not quants against a reference; no quant-level quality numbers [L18-23]",
         "Card says the GGUFs are quantizations of tapsin/TAPSIN-Qwen3.8-4B-Distill-GGUF, i.e. this same repo [L14]"])

repo(repo="FalconZypp/Qwen3.8-Flash-Next-AWQ-INT4", sha="2b1c4869435ca9e75992305c4497ca25b9e07b7f", fmt="AWQ",
  runtime="transformers / vLLM / SGLang (per copied card)", weights_modified=False, method="AWQ INT4 (method otherwise not described)",
  methodology=None, variants=[dict(name="AWQ-INT4", size=188.29, unit="GB", basis="disk", L=35)],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables [L112-134]",
         "Banner image and 'STEM and Agentic' calibration link point to cyankiwi's buckets/dataset, not this publisher [L11, L23]",
         "Size given only as 'Model Size 188.29 GB' with no basis [L35]"])

repo(repo="stgallenquants/Qwen3.8-Flash-Next-GGUF", sha="34cf4406a95ed9060381c4181115104f35da3239", fmt="GGUF",
  runtime="llama.cpp (per copied unsloth header)", weights_modified=False, method="not stated",
  methodology=None, variants=[dict(name="GGUF (files not listed on card)", size=None)],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables [L89-104]",
         "Header is unsloth's README text and links ('Unsloth Dynamic 3.0 achieves superior accuracy & outperforms other leading quants', unsloth tag) with no file list, method or sizes [L9-33]"])

repo(repo="maclocal/Qwen3.8-Flash-Next-AFM-MLX-4bit", sha="2ac1a930c82051d2cf50ccf2642ae852ff7a89f5", fmt="MLX (AFM)",
  runtime="unreleased AFM (maclocal-api) build", weights_modified=False,
  method="afm-mapped-4: affine 4-bit g64 main + MTP projections, 8-bit lm_head, n-gram table 4-bit g32, vision BF16",
  methodology=None, variants=[dict(name="AFM MLX 4-bit", size=None)],
  flags=["No quality claims for the quant; card reproduces Qwen's base-model tables [L117, L181-203]",
         "Local validation is decode speed on one prompt with 'coherent numbered sequence' as the only output check [L88-100]",
         "Run commands reference scouzi1966/Qwen3.8-Flash-Next-AFM-MLX-4bit, not this repo id [L64, L73]",
         "Card warns the artifact targets an unreleased AFM version [L16-23]",
         "No size stated"])

_potato_lang = [("bash","97.1","98.8"),("c","100.0","99.6"),("cpp","93.3","92.5"),("csharp","97.1","99.6"),("dart","97.5","96.7"),("flutter (widget tests)","95.8","94.6"),
  ("go","97.9","99.6"),("java","96.2","100.0"),("javascript","94.2","96.7"),("kotlin","97.9","96.2"),("php","92.5","87.1"),("python","95.4","92.9"),
  ("ruby","89.6","94.6"),("rust","97.5","95.4"),("swift","85.8","94.2"),("typescript","92.9","95.0")]
_potato_dev = [("LiveCodeBench (144)","125","118",118),("16 programming languages (640 tasks)","616","621",119),("8 human languages (160)","157","159",120),
  ("tool-call suite (18)","17","18",121),("HumanEval+ / MBPP+ (264)","229","238",122),("BFCL v3 (200)","148","164",123)]
_potato_m = {f"lang:{n}": v for n, f_, v in _potato_lang}
_potato_m.update({f"dev:{n}": v for n, v, _, _ in _potato_dev})
_potato_base = {f"lang:{n}": f_ for n, f_, v in _potato_lang}

repo(repo="Slomin/Potato-CODER-48GB", sha="a96007bb288ea04dc3da2c9b4ed2f8a7dd2b48bd", fmt="GGUF",
  runtime="llama.cpp potato branch (stock llama.cpp loads without MTP)", weights_modified=True,
  method="Routed experts pruned 512 -> 256 per layer by routing statistics on a 16-language coding/agentic corpus; IQ4_XS experts with imatrix from the unpruned model; 610 non-expert tensors Q6_K/Q8_0",
  methodology=dict(reference="full 512-expert model in the publisher's own IQ4_XS (not BF16)",
    dataset="20 sealed + 40 development tasks x 4 samples per language family (16 families); pre-registered gate: one-sided 95% upper bound of (full - v1.0) < 15 points",
    ctx=None, settings="temp 1.0, top_p 0.95, top_k 20, reasoning_effort medium, 16,384-token thinking budget", tool="llama-server (potato branch)", L=[83,84,86,87,212]),
  table_header_L=89,
  variants=[dict(name="Potato-CODER-48GB IQ4_XS (v1.0, withdrawn)", size=60.99, unit="GiB", basis="disk", m=_potato_m,
    L=91, L_extra=list(range(92,107)) + [d[3] for d in _potato_dev] + [78])],
  baseline=dict(_potato_base, L=91, L_extra=list(range(92,107))),
  cross=[dict(of="nvidia/Qwen3.8-Flash-Next-NVFP4", variant="NVFP4 (512 experts, run by publisher on DGX Spark under vLLM)", size=None,
    m={f"dev:{n}": b for n, _, b, _ in _potato_dev}, L=118, L_extra=[d[3] for d in _potato_dev][1:])],
  flags=["Model files withdrawn 11 Oct 2026 after the publisher found degradation incl. repetitive reasoning loops [L36-40]",
         "Pruned (half the routed experts removed): not a faithful quant [L70-71]",
         "Reference is the publisher's own unpruned IQ4_XS, not BF16 [L84]",
         "NVIDIA comparison uses different engines, quantizations, machines, thinking budgets and parallelism, one run per benchmark; card labels it descriptive [L129-130]",
         "Prose says 'on competitive coding the pruned model solved more than the full one' while the comparison is against NVIDIA's NVFP4 under a different setup [L125, L129-130]",
         "Size is the main file only (60.99 GiB); MTP head 1.3 GiB and projector 0.6 GiB separate [L78]"])

repo(repo="GBP-DE/Qwen3.8-Flash-Next-oQ5e-WLSRefit-MTP", sha="559775fbfecd6c02676aca0e2c0c947897d89e3b", fmt="MLX (oMLX oQ5e)",
  runtime="oMLX 0.7.1.dev1", weights_modified=False, method="oQ5e with oMLX Weighted Least-Squares refit, imatrix-guided, BF16 non-quantized weights, MTP preserved",
  methodology=None, variants=[dict(name="oQ5e WLSRefit MTP", size=None)],
  flags=["No measured quality claims; card says practical quality 'has not yet been independently benchmarked' [L47-50]",
         "Only qualitative comparison is of a different oQ4e build on an unpublished 60-question challenge, and the card notes the runtime also changed [L39-54]",
         "'11–15% lower KL divergence' is attributed to the oMLX developer, not measured on this model [L27-29]",
         "No size stated"])

repo(repo="lesj0610/Qwen3.8-Flash-Next-NVFP4", sha="e8045c65a50adedb6a465f29311ee562c51a0266", fmt="NVFP4",
  runtime="not verified (ModelOpt HF export)", weights_modified=False,
  method="NVIDIA ModelOpt general NVFP4 experts-only recipe (constant input amax 2688, FP8 KV cast) + 4 disabled quantizers (PLE, conv1d, hyper-connections, QSA indexer); everything but routed experts BF16",
  methodology=dict(reference="original BF16 checkpoint weights (layer-0 experts 0/7/255/511)", dataset="n/a (weight-space reconstruction, no inference)", ctx=None, tool="own unpack/rescale comparison", L=[160,161]),
  table_header_L=163,
  variants=[dict(name="NVFP4", size=186.4, unit="GB", basis="disk", m={"weight_cos":"0.9954","weight_rel_frob_err":"0.094"}, L=170, L_extra=[30])],
  flags=["Only weight-reconstruction fidelity (cosine/relative error on 4 sampled experts of layer 0); card states no inference was run and task quality is unmeasured [L163-170, L210-214]",
         "Card reproduces Qwen's base-model tables below L245 [L245, L315]"])

repo(repo="Manav1011/Qwen3.8-Flash-Next-NVFP4", sha="f86fa7242b476a3aef2861f6f1ef4b8f82606ebb", fmt="NVFP4",
  runtime="SGLang (qwen4_exp)", weights_modified=False, method="NVFP4 MoE experts (ModelOpt v0.46.0); calib 128 cnn_dailymail articles @512; PLE from Qwen FP8 revision",
  methodology=dict(reference="BF16 runs on an earlier checkpoint revision", dataset="GSM8K (1319), AIME26 (30x8)", ctx=None, tool="not stated", L=[118,123]),
  table_header_L=118,
  variants=[dict(name="NVFP4", size=135, unit="GB", basis="disk", m={"bench:GSM8K":"97.27","bench:AIME26 pass@1":"98.75"}, L=120, L_extra=[121,92])],
  baseline={"bench:GSM8K":"97.12–97.50","L":120},
  baseline2={"bench:AIME26 pass@1":"100","L":121},
  flags=["Card is RadixArk's NVFP4 card text: it names 'RadixArk Qwen3.8-Flash-Next-NVFP4', a RadixArk release date and the --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 [L19, L40, L100]",
         "Evaluation numbers and footnotes are identical to RadixArk/Qwen3.8-Flash-Next-NVFP4's card; the 'this exact checkpoint' attribution is RadixArk's wording [L116-128]",
         "BF16 baselines from an earlier checkpoint revision; AIME26 measured on a previous revision [L123-128]"])

repo(repo="spectator2026/Qwen3.8-Flash-Next-W6-HQQ-g32", sha="05848f88f4a5bac85fc738d4327acac3c976a0ad", fmt="compressed-tensors int6 (W6A16)",
  runtime="patched vLLM (pinned nightly + 7 patches + custom 6-bit Marlin kernel), A100", weights_modified=False,
  method="Routed experts int6 symmetric g32: SmoothQuant fold (α=0.65), MSE-clip init, HQQ proximal scale refit; everything else BF16",
  methodology=dict(reference="BF16 model", dataset="wikitext; agentic multi-turn conversations (token counts not stated)", ctx=None,
    settings="weights-only (BF16 KV) and as served (fp8 KV)", tool="vLLM (earlier preview build; wikitext re-measured on pinned nightly)", L=[106,113]),
  table_header_L=108,
  variants=[dict(name="W6 HQQ g32", size=224, unit="GB", basis="disk",
    m={"kld:wikitext fp8KV":"0.0228","kld:wikitext bf16KV":"0.0210","kld:agentic fp8KV":"0.0749","kld:agentic bf16KV":"0.0672","kld:wikitext pinned image":"0.0223"},
    L=110, L_extra=[111,113,25])],
  flags=["Eval set sizes, context length and chunking not stated [L106-113]",
         "Main figures measured on an earlier vLLM preview build; only wikitext re-measured on the pinned image (0.0223) [L113]",
         "Uses wikitext, which agentionai's AP card reports is memorised by this architecture's n-gram table",
         "Size given as ≈224 GB (209 GiB) [L25]"])

repo(repo="chatnerdie/Qwen3.8-Flash-Next-EXL3-Sushi", sha="5329132501fa2c3983fa3593a78afe4903e84996", fmt="EXL3 (sushi pack)",
  runtime="sushi on Apple Silicon only", weights_modified=False,
  method="Routed experts with open-source ExLlamaV3 v1.6.0 convert.py (-cb mcg, -sh scaled Hessian), calib 250 x 2,048 tokens generated by turboderp's 6.05 bpw EXL3; non-expert files copied from beamster Sushi-4bpw",
  methodology=dict(reference="BF16 checkpoint", dataset="24 prompts (8 coding, 8 prose, 8 JSON extraction/tool calls); BF16 greedy 128-token replies, scored to first end-of-turn (2,516 positions)",
    ctx=None, settings="sushi 1.2.3, f16 KV, bf16 n-gram table; YaRN x4 as shipped and native RoPE; paired 95% bootstrap ratios", tool="sushi kld compare", L=[57]),
  table_header_L=59,
  variants=[
    dict(name="4bpw-sh", size=63.7, unit="GiB", basis="vram",
         m={"kld:yarn":"0.005780","top1:yarn":"98.5%","kld:native":"0.003525","top1:native":"98.4%","kld_ratio_vs_sushi4bpw:yarn":"0.94 (0.88–1.00)","kld_ratio_vs_sushi4bpw:native":"0.94 (0.72–1.15)","tool_calling":"24 / 28"},
         L=62, L_extra=[70,96,24]),
    dict(name="5bpw-sh", size=77.8, unit="GiB", basis="vram",
         m={"kld:yarn":"0.004843","top1:yarn":"98.4%","kld:native":"0.002124","top1:native":"99.0%","kld_ratio_vs_sushi4bpw:yarn":"0.79 (0.72–0.86)","kld_ratio_vs_sushi4bpw:native":"0.57 (0.46–0.68)","tool_calling":"28 / 28"},
         L=63, L_extra=[71,96,25]),
  ],
  ladder=[dict(name="earlier 4-bit -sh build (different calibration data)", size=None, m={"bench:MMLU-Pro 1,400 q, YaRN":"68.64%","bench:MMLU-Pro 1,400 q, native RoPE":"68.79%"}, L=83)],
  cross=[
    dict(of="beamster/Qwen3.8-Flash-Next-Sushi-4bpw", variant="Sushi-4bpw", size=63.7, unit="GiB", basis="vram", m={"kld:yarn":"0.006137","top1:yarn":"98.2%","kld:native":"0.003740","top1:native":"98.4%","tool_calling":"28 / 28"}, L=61, L_extra=[96]),
    dict(of="SiliconSpecies/Qwen3.8-Flash-Next-EXL3", variant="5bpw-hq (rev ccd64f0)", size=78.3, unit="GiB", basis="vram", m={"kld:yarn":"0.004729","top1:yarn":"98.3%","kld:native":"0.002070","top1:native":"98.7%"}, L=64),
  ],
  flags=["Small eval: 24 prompts, 2,516 positions; card itself says not to over-read absolute values [L57, L116]",
         "As-shipped KLD includes the YaRN x4 RoPE change (about 40% of 4-bit and 56% of 5-bit KLD per card) [L81]",
         "Card says its numbers are not comparable with the 0.0592 on the Sushi-4bpw card (different prompts, reply length, KV, sushi build) [L77]",
         "Sizes are weights in GPU memory; the 95.4 GiB bf16 n-gram table is not included and must be downloaded from Sushi-4bpw [L29]",
         "Non-expert files are beamster Sushi-4bpw's, byte-identical [L18, L111]",
         "4bpw-sh misses one tool-calling prompt (4 greedy runs) with thinking off [L96, L103]",
         "MMLU-Pro figures are from an earlier build with different calibration data [L83]"])

repo(repo="Tylogi/Qwen3.8-Flash-Next-EW-MFQ", sha="3cdfba425c44177019d3fdac94fbfa5d17379505", fmt="MFQ (.mfq)",
  runtime="MFQ only (CUDA / Metal)", weights_modified=False,
  method="Expert-wise mixed precision: per-expert Gate/Up/Down format and bit-width (NINT scalar or NVQ vector) allocated under a global budget with NatScale contrast-calibrated sensitivity",
  methodology=dict(reference="not stated ('the reference model')", dataset="WikiText-2", ctx=None, tool="not stated", L=[74]),
  table_header_L=78,
  variants=[dict(name=n, size=s, unit="GB", basis="disk", size_note="excludes vision encoder and MTP head (card L97)", m={"kld":k,"top1_pct":t}, L=l, L_extra=[97])
    for n, s, k, t, l in [("V1-S",72.536,"0.22157","83.77%",80),("V1-M",74.529,"0.19689","84.72%",81),("S2-L",78.859,"0.14630","86.54%",82),
                          ("V3-XXS",81.952,"0.12436","87.62%",83),("S3-L",89.976,"0.08698","89.55%",84),("V4-XS",93.672,"0.07524","90.36%",85)]]
    + [dict(name="S4-L", size=None), dict(name="S5-L", size=None)],
  cross=[dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant=v, size=s, unit="GB", basis="disk", m={"kld":k,"top1_pct":t}, L=l)
         for v, s, k, t, l in [("UD-IQ1_S",72.5,"0.39607","77.33%",86),("UD-IQ1_M",74.5,"0.31474","79.69%",87),("UD-Q2_K_XL",78.9,"0.22461","82.72%",88),
                               ("UD-IQ3_XXS",82.0,"0.16512","85.41%",89),("UD-Q3_K_XL",90.0,"0.10650","88.31%",90),("UD-IQ4_XS",93.7,"0.08363","89.55%",91)]]
      + [dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant=v, size=s, unit="GB", basis="disk", m={"kld":k,"top1_pct":t}, L=l)
         for v, s, k, t, l in [("Q2_0",66.424,"0.40660","77.27%",92),("IQ2_XS",68.026,"0.33374","79.06%",93),("IQ3_XXS",75.840,"0.22134","83.00%",94),("IQ3_S",83.618,"0.13717","86.90%",95)]],
  flags=["Reference model, context, chunk count and tool not stated; only 'WikiText-2' [L74]",
         "Unsloth rows equal Unsloth's published docs values (rounded) and link to Unsloth's analysis; the card does not say they were re-measured under the same setup as the MFQ and GSQ-RCO rows [L86-91]",
         "Uses wikitext, which agentionai's AP card reports is memorised by this architecture's n-gram table",
         "Tiers S4-L and S5-L have no quality numbers [L67-68, L78-85]",
         "Weight sizes exclude vision encoder and MTP head; V1-M download is 72.44 GiB with 43.36 GiB resident [L97, L153]"])

# Re-extracted from verbatim README snapshots at pinned commits (snapshots/index.json -> cache/cards/, captured 2026-10-17).
# Queue: cache/autoq_2.json (previously regex-triaged only). Every card read in full.
# Each variant/claim cites README line numbers (L) at that sha.
# size basis: disk = full download unless noted; vram/resident/ram_in_use = memory figures; trunk_only = weights excluding n-gram table.
R = []
def repo(**k): R.append(k); return k

repo(repo="ArtomYuan/Qwen3.8-Flash-Next-ROCmFPX", sha="e194a61cd0ee1609ef86c8e5391b9d0bc9dc08b4", fmt="GGUF (ROCmFPX extension types)",
  runtime="ArtomYuan/llama.cpp-rocm fork (upstream llama.cpp cannot load)", weights_modified=False,
  method="ROCmFPX tiers (GGML file types 103/104/105/110/111/114/115) requantized locally from official weights; per-tier hardening of embeddings/attention",
  methodology=None,
  table_header_L=74,
  variants=[
    dict(name="Q4_0_ROCMFP4_FAST", size=87.79, unit="GiB", basis="disk", bpw=4.26, L=76),
    dict(name="Q4_0_ROCMFP4_FAST_COHERENT", size=113.30, unit="GiB", basis="disk", bpw=5.50, L=77),
    dict(name="Q4_0_ROCMFP4_STRIX", size=113.33, unit="GiB", basis="disk", bpw=5.50, L=78),
    dict(name="Q6_0_ROCMFPX", size=135.68, unit="GiB", basis="disk", bpw=6.59, L=79),
    dict(name="Q6_0_ROCMFPX_AGENT", size=158.01, unit="GiB", basis="disk", bpw=7.67, L=80),
    dict(name="Q8_0_ROCMFPX", size=170.15, unit="GiB", basis="disk", bpw=8.26, L=81),
    dict(name="Q8_0_ROCMFPX_AGENT", size=173.58, unit="GiB", basis="disk", bpw=8.43, L=82),
  ],
  flags=["No quality claims for the quant (card read in full): benchmarks are llama-bench throughput only (pp256/pp2048/tg256) [L84-114]; tier labels such as 'balanced quality' / 'maximum quality' are descriptive, with no measured quality numbers [L40-46, L74-82]",
         "Byte sizes also given per file (e.g. 94,268,794,528 B for FAST) [L61-67]; mmproj 0.85 GiB separate [L68]"])

repo(repo="BenMonkey11/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF", sha="85a485c27716b0d8c26c2c0408ca5da6141a6854", fmt="GGUF",
  runtime="llama.cpp / Strata", weights_modified=True,
  method="Abliterated (refusal removal) v2 of Qwen3.8-Flash-Next; GGUFs described as coming from unsloth/Qwen3.8-Flash-Next-GGUF and ukisai Swift series",
  methodology=None,
  variants=[dict(name="UD-Q4_K_XL", size=None)],
  flags=["No quality claims for the quant (card read in full): card describes abliteration, Strata/llama.cpp usage and usage warnings only [L30-76]",
         "Weights modified: abliterated / uncensored [L30-34]",
         "Card text is byte-identical to huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF, including its title naming huihui-ai [L27]",
         "No sizes stated on card"])

repo(repo="DevQuasar/Qwen.Qwen3.8-Flash-Next-GGUF", sha="ef7dd837a8bac5e560bfcf3e0f213e4ed1284f07", fmt="GGUF",
  runtime="not stated", weights_modified=False, method="not stated", methodology=None,
  variants=[dict(name="GGUF (variants not listed)", size=None)],
  flags=["No quality claims for the quant (card read in full): card is a logo, a slogan and a link to the base model [L7-13]",
         "No variants or sizes stated on card"])

repo(repo="GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP", sha="3f3eb3a76124b9d9ad9bb934186f942f2e40e05f", fmt="MLX (oMLX oQ)",
  runtime="oMLX", weights_modified=False,
  method="oQ4e with oMLX 0.7.1.dev1 Weighted Least-Squares Refit, imatrix-guided; BF16 non-quantized weights; MTP preserved",
  methodology=None,
  variants=[dict(name="oQ4e WLSRefit", size=None)],
  flags=["No quality claims for the quant (card read in full): only qualitative observations from a GPT-6-reviewed 60-question 'Silbergrat Reasoning Challenge' and a trading-research prompt, which the card itself calls 'qualitative, not standardized benchmarks' [L35-62]",
         "'approximately 11–15% lower KL divergence' is attributed to the oMLX developer's quantizer evaluations, not measured on this build (quoted, not measured) [L28-29]",
         "Card notes the runtime version also changed, so improvements cannot be attributed to WLS Refit alone [L61-62]",
         "No size stated on card"])

repo(repo="IHaveNoClueAndIMustPost/Qwen3.8-Flash-Next-Simple-Q3_K", sha="7ee3c667e10fb06749d73ce5cf3983d9cc3f4ca6", fmt="GGUF",
  runtime="llama.cpp", weights_modified=False,
  method="Simple mix on Q8_0 base: signal path Q8_0, gate/up Q3_K (edge blocks IQ4_NL), down IQ4_NL, n-gram table Q8_0",
  methodology=dict(reference="full BF16", dataset="WikiText-2", ctx=512, tool="not stated", L=[20]),
  variants=[dict(name="Simple-Q3_K", size=60.9, unit="GiB", basis="trunk_only", bpw=5.42,
    m={"ppl":"4.6587 ± 0.028","ppl_delta_pct":"+1.49%"}, L=20, L_extra=[9])],
  baseline={"ppl":"4.5903","L":20},
  flags=["Total size stated as 114,245 MiB (core 60.9 GiB + 50.7 GiB n-gram table on disk); recorded size is the core excluding the table [L9]",
         "Single PPL figure; perplexity tool and chunk count not stated [L19-20]",
         "MTP not included in main files; separate standalone Q4_0 MTP GGUF added [L11-12]"])

repo(repo="Jundot/Qwen3.8-Flash-Next-oQ2.5e-mtp", sha="ebfb10474baa3c1c78d38453ce0d5228febd32e9", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ2.5e (oMLX v0.7.1, Improved oQe): sensitivity-based bit allocation + imatrix (1024x512) + weighted LSQ scale/bias refit; n-gram table 4-bit g32 outside budget",
  methodology=None,
  variants=[dict(name="oQ2.5e", size=82.898, unit="GB", basis="disk", bpw=3.684, L=53, L_extra=[43,26])],
  flags=["No quality claims for the quant (card read in full): only bit-distribution and storage tables [L28-53]",
         "3.161 bpw excluding the n-gram table, 3.684 for the full checkpoint; total 77.205 GiB [L26, L43, L53]"])

repo(repo="Jundot/Qwen3.8-Flash-Next-oQ3e-mtp", sha="b73babc4459da74834c7f9ef714f142f750d50fa", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ3e (oMLX v0.7.1, Improved oQe): sensitivity-based bit allocation + imatrix (1024x512) + weighted LSQ scale/bias refit; n-gram table 4-bit g32 outside budget",
  methodology=None,
  variants=[dict(name="oQ3e", size=91.201, unit="GB", basis="disk", bpw=4.053, L=55, L_extra=[45,26])],
  flags=["No quality claims for the quant (card read in full): only bit-distribution and storage tables [L28-55]",
         "3.677 bpw excluding the n-gram table, 4.053 for the full checkpoint; total 84.937 GiB [L26, L45, L55]"])

repo(repo="Kandandan/Qwen3.8-Flash-Next-MLX-4bit-MTP", sha="40e15b0057a1748b4a1f43fd6ff92fa70445ee47", fmt="MLX",
  runtime="mlxturbo (flash_spec) / mlx-lm (text only, no MTP)", weights_modified=False,
  method="MLX 4-bit affine group 64; bf16 MTP sidecar (5.2GB); n-gram (PLE) table handled as a separate sidecar, not in the main weights",
  methodology=None,
  variants=[dict(name="4bit g64 + MTP", size=77, unit="GB", basis="disk", L=33)],
  flags=["No quality claims for the quant (card read in full): only speculative-decoding acceptance rates and speedups (M3 Max, 160 tokens x 10 tasks) [L48-60]",
         "Total stated as approx. 77GB ('約 77GB'); card says the n-gram table is not included in the main body [L33-35]"])

repo(repo="Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-fp16-DWQ-MTP-Vision-MLX", sha="46a83dc51b5b78e90549ff33bd9a92f1026df1e2", fmt="MLX (oMLX)",
  runtime="oMLX", weights_modified=True,
  method="REAP expert pruning 512->320 + oQ3e (3-bit g64 routed experts, M4Q init, 5/6/8-bit trunk, 4-bit g32 n-gram) + end-to-end KL-DWQ (v7->v9) against bf16 teacher top-20 logprobs; 16-bit tensors recast BF16->FP16, vision FP32",
  methodology=dict(reference="bf16 Qwen3.8-Flash-Next teacher, top-20 logprobs incl. reasoning tokens",
    dataset="held-out calibration-distribution records: 41-record table (same records every round); independent gate 17 held-out caches, 456,602 tokens; real-image vision 120 held-out records",
    ctx=None, tool="custom end-to-end KL-DWQ tooling for qwen4_exp (not named further)", L=[164,185,196,198,203]),
  table_header_L=187,
  variants=[dict(name="REAP320 oQ3e fp16 DWQ v9", size=71.7, unit="GB", basis="disk", bpw=4.24, mem=37, mem_unit="GiB", mem_basis="resident",
    m={"ce_gap:Overall":"0.0901","kl20:Overall":"0.1035","kl20_delta_v8_v9:Overall":"−1.6%",
       "ce_gap:Reasoning":"0.0596","kl20:Reasoning":"0.1077","ce_gap:Code":"0.0437","kl20:Code":"0.0595",
       "ce_gap:Russian chat":"0.1561","kl20:Russian chat":"0.1136","ce_gap:Knowledge QA":"0.0950","kl20:Knowledge QA":"0.1483",
       "ce_gap:Agent":"0.2235","kl20:Agent":"0.1854",
       "ce_gap:independent gate":"0.0968","kl20:independent gate":"0.0833",
       "ce_gap:real-image vision":"0.2533","kl20:real-image vision":"0.2245",
       "ce_gap_reduction_vs_pruned_init":"14.9%","kl20_reduction_vs_pruned_init":"11.2%","pruning_gap_recovered_pct":"33.6%",
       "fp16_exact_values_pct":"99.99973%","fp16_overflow_values":"0","fp16_flushed_to_zero":"148"},
    L=194, L_extra=[189,190,191,192,193,196,197,198,199,200,201,56,57,161,162,163,90,91,92],
    flags=["inherited family evaluation"])],
  ladder=[
    dict(name="v7 (earlier DWQ round)", m={"ce_gap:Overall":"0.0934","kl20:Overall":"0.1065"}, L=194),
    dict(name="v8 (earlier DWQ round)", m={"ce_gap:Overall":"0.0921","kl20:Overall":"0.1052"}, L=194),
    dict(name="BF16-typed sibling (Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX)", size=70.8, unit="GB", basis="disk", L=212),
  ],
  flags=["Card states this FP16 repo shares weights, quantization, DWQ training and 'same evaluation numbers' with its BF16 sibling; teacher-gap tables describe both repos [L62-64]",
         "Metrics are fidelity to the bf16 teacher (CE-gap | KL20 over top-20 logprobs) on held-out calibration distributions, not task accuracy; no task or vision-benchmark score claimed [L203-204]",
         "Pruned: 192 of 512 experts per layer removed (REAP-320); card calls pruning lossy [L114-117, L255-256]",
         "FP16 recast checked in value domain: 99.99973% of 5,787,423,569 FP16-typed values exact, 0 overflow, 148 flushed to zero [L87-94]",
         "~20% M1/M2 prefill gain is quoted from oMLX documentation; card says the speedup itself was not measured on this pack [L66, L103-106]",
         "33.6% recovery is of a 0.047-nat gap to an unpruned 512-expert 3-bit reference [L200-201]"])

repo(repo="MagneticLab/Qwen3.8-Flash-Next-NVFP4", sha="e30ead75503c0b734af6ec0614ef22089eee6bf3", fmt="NVFP4 (ModelOpt) + NVFP4 PLE sidecar",
  runtime="vLLM (vllm/vllm-openai:qwen38-flash-next image) with primitive-ai PLE overlay files", weights_modified=False,
  method="ModelOpt NVFP4 on routed experts only; 51.2B PLE n-gram table served from a quantized (4-bit) sidecar mmapped from disk, quantized from the BF16 release",
  methodology=None,
  variants=[dict(name="NVFP4 (full download)", size=200, unit="GB", basis="disk", mem=74.34, mem_unit="GiB", mem_basis="resident",
    m={"ple_sidecar_worst_shard_rel_err":"0.0896"}, L=61, L_extra=[232,108])],
  flags=["No output-quality claims for the quant (card read in full): only a weight-level PLE sidecar figure (worst_shard_rel_err 0.0896 vs 0.1493 for an unnamed 'reference build' quantized from FP8) [L231-233]; needle-in-haystack passes at 8k/25k/99k/185k [L178-179]; throughput tables [L167-199]",
         "Full download 200 GB incl. a 99 GB duplicate PLE table never read at runtime; lean download 101 GB; checkpoint dir 186 GB -> 84 GB trimmed [L34-36, L76, L94]",
         "Weights in memory 74.34 GiB (identical trimmed vs full) [L108]; elsewhere 'weights 75.9' at rest [L181]",
         "vLLM overlay files are primitive-ai's work, bundled [L83-88]"])

repo(repo="RESMP-DEV/Qwen3.8-Flash-Next-IQ2_XS-Q2_0-Standard-iMatrix-GGUF", sha="694d57e65b887b0620cd32fdebb23f2f0a6f3003", fmt="GGUF",
  runtime="Strata engine 0.1.41 (llama.cpp generation not validated)", weights_modified=False,
  method="Research static requantization: routed gate/up -> IQ2_XS, down -> Q2_0, from mradermacher's IQ4_XS GGUF (not BF16); private 256x2,048-token standard iMatrix",
  methodology=None,
  variants=[dict(name="IQ2_XS / Q2_0 standard-iMatrix", size=66649935712, unit="bytes", basis="disk", bpw=3.01, L=61, L_extra=[62,142])],
  flags=["No quality claims for the quant (card read in full): card states no perplexity, task-quality, long-generation or broad throughput evaluation has been run; checks prove integrity and execution only [L31-32, L133-136]",
         "Compounded requantization from an already-quantized IQ4_XS GGUF (mradermacher), not from BF16 [L36-39, L102-106]",
         "Calibration corpus and importance matrix not published [L43-45]",
         "Reported quantized size 63,551.82 MiB / 3.01 bpw; files 39.969 GB + 26.681 GB [L62, L74-75]"])

repo(repo="Sawfwair/Qwen3.8-Flash-Next-MLX-4bit", sha="6cc9bbc0fae9ce26b7670b3ed1e26d557c154506", fmt="MLX",
  runtime="Qwen4Exp-aware MLX runtime (not named)", weights_modified=False,
  method="MLX affine Q4 group 64 for language/MTP/vision matrices; n-gram table Q4 group 32; routers, norms, biases, convs dense",
  methodology=None,
  variants=[dict(name="MLX 4-bit", size=97.55, unit="GiB", basis="disk", L=22)],
  flags=["No quality claims for the quant (card read in full): only conversion inventory and a statement that packing/layout are validated by a bundled JSON [L20-34]"])

repo(repo="SuperPUPS/Qwen3.8-Flash-Next-INT8x17-AttnINT4", sha="8d7e6f73b95df50fc15e6bcf10e657725fd011c5", fmt="compressed-tensors (INT8/INT4 pack-quantized)",
  runtime="vllm-qfn patched vLLM snapshot (feature/qsa-int8-kv, commit e12683d245)", weights_modified=False,
  method="Community W4A16 checkpoint (INT4 g128 experts) with 17 of 48 expert layers re-quantized to INT8 g128 from the official FP8 checkpoint (msegrid scale refinement); attention INT4 g128",
  methodology=dict(reference="official FP8 checkpoint (weight-level rel-Frobenius / cosine); live gates on serving stack with no reference model",
    dataset="1536 expert-matrix pairs per layer (weight-level); 3 golden probes, 10-task battery, 8x determinism, NIAH @128K, 30-min soak (live gates)",
    ctx=None, tool="quantization/ scripts (rank_sensitive_layers.py, verify_checkpoint.py); vllm-qfn serving", L=[53,54,55,81,93]),
  table_header_L=93,
  variants=[dict(name="INT8x17-AttnINT4", size=188.5, unit="GiB", basis="disk",
    m={"int8_roundtrip_cos_vs_fp8":"0.999962","golden_probes":"3/3","bench:10-task battery":"10/10","determinism_8x":"7/8","niah_128k":"HIT","soak_errors":"0 errors"},
    L=33, L_extra=[81,95,96,97,98,99,219])],
  ladder=[dict(name="INT8x25-AttnBF16 (SuperPUPS sibling)", size=199.4, unit="GiB", basis="disk", L=33)],
  flags=["Quality evidence is weight-level (INT8 roundtrip cosine 0.999962 vs FP8 reference; parent INT4 0.9925) plus pass/fail live gates; no perplexity/KLD or benchmark scores vs BF16 [L81, L85-99]",
         "Per-layer INT4-error ranking table is for the parent W4A16 experts, used to choose INT8 layers [L62-72]",
         "Base W4A16 checkpoint is a community build, not named on the card [L52, L226]",
         "Sibling INT8x25 is listed as 'all passed' quality gates with no numbers [L38]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP", sha="45032a193a5d4126ba8534822cbaf8fb30b3e368", fmt="MLX",
  runtime="oMLX 0.6.3rc3 / MLX-VLM 0.6.3", weights_modified=False,
  method="Uniform 6-bit affine group 32 from official BF16; multimodal modules and router gates BF16; native MTP preserved",
  methodology=None,
  variants=[dict(name="MLX 6-bit g32", size=158.08, unit="GB", basis="disk", L=78)],
  flags=["No quality claims for the quant (card read in full): only MTP on/off throughput, draft acceptance and pass/fail parity/coherence checks [L114-129, L172-182]",
         "'Exact-output parity' refers to MTP-off vs MTP-on decoding of this checkpoint, not to BF16 [L126, L179]",
         "Weight size also given as 147.224 GiB [L78]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-oQ6-MTP", sha="d6209ba59bd0b53d853b0bbfa2cd1d20f1bebecf", fmt="MLX (oQ)",
  runtime="oMLX / MLX-VLM with qwen4_exp native MTP", weights_modified=False,
  method="oQ6 mixed precision: 6-bit affine base g32, 232 protected/overridden modules at 6/8-bit; sensitivity from a local 4-bit proxy; all tensors rebuilt from BF16",
  methodology=None,
  variants=[dict(name="oQ6", size=158.07, unit="GB", basis="disk", L=76)],
  flags=["No quality claims for the quant (card read in full): only short-run speeds, MTP acceptance and coherence checks [L104-123, L147-154]",
         "Weight size also given as 147.21 GiB [L76]"])

repo(repo="V1ta11y1/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF", sha="09de5fa7214177db1273cd119b824f9bd967ac78", fmt="GGUF",
  runtime="llama.cpp / Strata", weights_modified=True,
  method="Abliterated (refusal removal) v2 of Qwen3.8-Flash-Next; GGUFs described as coming from unsloth/Qwen3.8-Flash-Next-GGUF and ukisai Swift series",
  methodology=None,
  variants=[dict(name="UD-Q4_K_XL", size=None)],
  flags=["No quality claims for the quant (card read in full): card describes abliteration, Strata/llama.cpp usage and usage warnings only [L30-76]",
         "Weights modified: abliterated / uncensored [L30-34]",
         "Card text is byte-identical to huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF, including its title naming huihui-ai [L27]",
         "No sizes stated on card"])

repo(repo="Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality", sha="56b2d9ffcf04b6fb33617af8f9d6523cfbafa573", fmt="MLX (MTPLX)",
  runtime="MTPLX >= 2.12.0", weights_modified=False,
  method="8-bit g64 experts, attention and MTP head; BF16 structural weights; 4-bit n-gram table sidecar streamed from SSD",
  methodology=None,
  variants=[dict(name="Optimized Quality (8-bit)", size=170, unit="GB", basis="disk", mem=128.5, mem_unit="GiB", mem_basis="resident", L=68, L_extra=[69])],
  flags=["No quality claims for the quant (card read in full): described as 'highest-fidelity' / 'higher-fidelity' without measured numbers; speed on 256/512 GB Macs not measured [L43, L47, L75-76]",
         "Download 170 GB includes the 32 GB n-gram table [L68]"])

repo(repo="agnosticeng/Qwen3.8-Flash-Next-4bit", sha="97a3bd1a847e60bebdf46ecf85a6ebdbf8cc72f0", fmt="MLX-style (lisa runtime)",
  runtime="lisa", weights_modified=False,
  method="Affine 4-bit (mx.quantize) group 32 of the text tower; MTP head and merged 4-bit gs32 n-gram table included",
  methodology=None,
  variants=[dict(name="4-bit gs32", size=None)],
  flags=["No quality claims for the quant (card read in full): only quantization format, contents and conversion notes [L23-49]",
         "No total size stated (trunk in 17 shards of ~4.9 GB each) [L38]"])

repo(repo="arnomatic/Qwen38-AWQ-g32-ple8", sha="d40f5e9f8285b444c9ef79dea0c73801e0e110b6", fmt="AWQ (moe_wna16) + W8A16 PLE",
  runtime="vLLM with AMD Qwen3.8-Flash-Next support + 4 patch files (2x Strix Halo, TP2+EP)", weights_modified=False,
  method="Experts AWQ g32 asymmetric from leoncca/Qwen3.8-Flash-Next-AWQ-g32; PLE n-gram table baked to W8A16 g32 from the official FP8 checkpoint",
  methodology=dict(reference="reference W8A16 PLE table (aixiaoma/Qwen3.8-Flash-Next-W4A16 lineage)", dataset="PLE table weights (weight-level comparison)", ctx=None, tool="container/bake_awq_ple8.py", L=[30,31,105]),
  variants=[dict(name="AWQ-g32 + PLE8", size=131, unit="GB", basis="disk", mem=68, mem_unit="GiB", mem_basis="resident",
    m={"ple_corr_vs_ref_w8a16":"0.9996","ple_max_numeric_diff":"2.29e-04"}, L=30, L_extra=[31,37,38])],
  flags=["Only fidelity figures are weight-level for the PLE table (correlation 0.9996, max diff 2.29e-04 vs a reference W8A16 table); no output-level quality numbers [L30-31]",
         "Asymmetric experts described as fixing word-form degradation seen with a symmetric RTN/g128 build ('reasoning ~3x shorter'), stated qualitatively without numbers [L25-27]",
         "Expert quantization is leoncca's AWQ-g32, not re-done by this publisher [L23-24]",
         "Memory ~68 GiB weights per rank (two ranks) [L38]"])

repo(repo="batiai/Qwen3.8-Flash-Next-GGUF", sha="1d0ec19ce9b05014c6236c2813ce77aef6fbc42a", fmt="GGUF",
  runtime="llama.cpp mainline (PR #27742)", weights_modified=False,
  method="Quantized from official BF16 with imatrix (4.90M chars: 20.13% Hangul, code, English wikitext; --chunks 200); PLE falls back to 32-block types (Q6_K rebuilt with Q5_1 PLE)",
  methodology=dict(reference="none (pass/fail functional checks, no BF16 comparison)", dataset="Korean generation, trivial arithmetic, tool calls, coding, long-form Korean; external validation adds two arithmetic expressions",
    ctx=8192, tool="llama-server -ngl 0 -c 8192 -b 512 -ub 512 -np 1", L=[128,129,83,84]),
  table_header_L=70,
  variants=[
    dict(name="IQ3_XXS", size=79.2, unit="GiB", basis="disk", m={"korean_hangul_pct":"70.0%"}, L=72, L_extra=[229,230]),
    dict(name="IQ4_XS", size=90.8, unit="GiB", basis="disk", mem=94.28, mem_unit="GiB", mem_basis="ram_in_use",
      m={"functional_checks":"7/7","korean_hangul_pct":"70.4%","korean_longform_hangul_pct":"73.1%"}, L=73, L_extra=[126,133,139,90]),
    dict(name="Q4_K_M", size=111.0, unit="GiB", basis="disk", L=74),
    dict(name="Q6_K (Q5_1 PLE)", size=141.2, unit="GiB", basis="disk", m={"korean_hangul_pct":"70.5%"}, L=75, L_extra=[220]),
  ],
  flags=["Quality evidence is pass/fail functional checks and Hangul-share percentages; no perplexity, KLD or benchmark comparison with BF16 [L126-139]",
         "External validation on the IQ4_XS files: two deterministic arithmetic prompts answered wrong (11,323 vs 11,289; 100 vs 200) [L107-114]",
         "Length-constrained thinking test, 5 runs per quant: IQ4_XS 1/3, Q6_K 0/3 with constraint, both 2/2 without; card attributes it to the prompt, not quantization [L173-186]",
         "Q6_K carries a Q5_1 PLE (38.40 GB), not the default Q8_0, due to the 50 GB per-file limit [L212-218]",
         "Peak RSS 94.28 GiB for IQ4_XS on an external M2 Ultra at 8K context [L90]",
         "Card's 'What BatiAI did differently' table says IQ3_XXS failed, while the PLE section says Korean checks passed at every tier including IQ3_XXS [L229-231, L306]"])

repo(repo="butterf1ying/Qwen3.8-Flash-Next-Uncensored-REAP288-MTP-MLX-4bit", sha="7fef629a4c196b8c7994aa411a405bf31a9915cd", fmt="MLX (mlx-serve)",
  runtime="mlx-serve 26.8.11", weights_modified=True,
  method="REAP-288 expert pruning (512->288, from sh0wie) + MLX affine 4-bit g64 (from Sawfwair conversion); n-gram table 4-bit g32 mmapped; native MTP; assembled for reduced refusals",
  methodology=None,
  variants=[dict(name="Uncensored REAP288 MLX 4-bit", size=71.67, unit="GiB", basis="disk", L=76, L_extra=[43])],
  flags=["No quality claims for the quant (card read in full): smoke test only (loads, generates, MTP active; peak ~40.3 GB on M1 Max); no benchmark or speed claim [L81-90, L175]",
         "Weights modified: REAP-pruned to 288 experts and labelled Uncensored; refusal reduction not formally benchmarked [L35, L53]",
         "N-gram table approx. 29.8 GiB, memory-mapped [L72]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ2-mtp", sha="120564da3df7b6b5d8d9003b02973b39790d2447", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ2 (oMLX v0.7.0) mixed precision, default 2-bit g64; sensitivity map measured on Jundot oQ4e proxy (128x256, code_multilingual)",
  methodology=None,
  variants=[dict(name="oQ2", size=64, unit="GB", basis="disk", bpw=2.85, L=27)],
  flags=["No quality claims for the quant (card read in full): only quantization details and bit-width shares [L20-47]",
         "Sensitivity measured on a quantized proxy (Jundot/Qwen3.8-Flash-Next-oQ4e-mtp), not BF16 [L47]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ2e-mtp", sha="7680df8ed9f67f77dfe0a39e056fb7b08c68ff58", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ2e (oMLX v0.7.0) imatrix-weighted mixed precision, default 2-bit g64; imatrix oqe_code_multilingual 1,024x512 streamed from bf16; sensitivity from Jundot oQ4e proxy",
  methodology=None,
  variants=[dict(name="oQ2e", size=64, unit="GB", basis="disk", bpw=2.85, L=31)],
  flags=["No quality claims for the quant (card read in full): only quantization details, bit-width shares and imatrix expert coverage (99.97%) [L24-58]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ4-mtp", sha="12e38a27b37aec52d886fb8eab1fbe7c1516f015", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ4 (oMLX v0.7.0) mixed precision, default 4-bit g64; sensitivity from Jundot oQ4e proxy",
  methodology=None,
  variants=[dict(name="oQ4", size=106, unit="GB", basis="disk", bpw=4.72, L=27)],
  flags=["No quality claims for the quant (card read in full): only quantization details and bit-width shares [L20-45]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ6e-mtp", sha="fb34e33945802280896f898f2f0b097a13a9f16f", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ6e (oMLX v0.7.0) imatrix-weighted mixed precision, default 6-bit g64; imatrix oqe_code_multilingual 1,024x512 from bf16; sensitivity from Jundot oQ4e proxy",
  methodology=None,
  variants=[dict(name="oQ6e", size=150, unit="GB", basis="disk", bpw=6.69, L=31)],
  flags=["No quality claims for the quant (card read in full): only quantization details, bit-width shares and imatrix expert coverage [L24-54]"])

repo(repo="ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw", sha="a0e89db090a3e4b2e8ea2118fe9d0324d3854aca", fmt="MLX (mlx-serve)",
  runtime="mlx-serve", weights_modified=False,
  method="iQ-MLX: every routed expert 4-bit g64 quantized with an importance matrix from bf16 (1.35 M tokens: agent transcripts, code incl. SWE-bench Lite, prose, math); rest 8-bit g64; n-gram 4-bit g32 mmapped",
  methodology=dict(reference="bf16 model's own logits", dataset="290 held-out positions (agent, code, math, prose); KLD = KL(bf16 || pack) over bf16's top-1024 tokens",
    ctx=None, tool="tests/qwen38_flash_next_score.py (mlx-serve repo)", L=[53,54,55,56,138,139]),
  table_header_L=59,
  variants=[dict(name="iQ-MLX-4.7bpw", size=75.3, unit="GB", basis="resident", bpw=4.68,
    m={"top1:all":"88.6","top1:agent":"89.0","top1:code":"96.0","top1:math":"90.0","top1:prose":"81.2","kld_median":"0.0052","mean_kld":"0.126"},
    L=62, L_extra=[68,69,37])],
  ladder=[dict(name="mixed-4-8bit (no calibration; ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit)", size=75.3, unit="GB", basis="resident",
    m={"top1:all":"87.9","top1:agent":"89.0","top1:code":"88.0","top1:math":"87.5","top1:prose":"87.0","kld_median":"0.0066","mean_kld":"0.158"}, L=61)],
  flags=["Small eval: 290 held-out positions; card notes the KLD mean is dominated by a few flat prose positions and prefers the median [L54-57]",
         "KLD computed over bf16's top-1024 tokens, not the full vocabulary [L55-56]",
         "Prose top-1 is lower than the uncalibrated pack (81.2 vs 87.0) while code/math are higher [L61-62]",
         "Download 107 GB incl. 32 GB n-gram table; 75 GB resident [L37, L91]",
         "The sibling iQ-MLX-3.3bpw card lists mixed-4-8bit at 73 GB / top-1 89.1%, this card at 75.3 GB / 87.9 — the two cards differ (different held-out sets are possible: 1,200 vs 290 positions)"])

repo(repo="drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only", sha="b3b60a4ce9d5922e8395887a1e5a58734b51b522", fmt="not stated",
  runtime="not stated", weights_modified=True, method="not stated (repo name indicates an abliterated derivative)", methodology=None,
  variants=[dict(name="(no variants listed)", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot is empty",
         "weights_modified set from the repo name ('ablit'); the card itself gives no description"])

repo(repo="gabmiranda/Qwen3.8-Flash-Next-oQ2.5e-mtp", sha="e372f69f7cc317102db5ab321eb94e9b70aedf09", fmt="MLX",
  runtime="not stated (oQ tag)", weights_modified=False, method="oQ2.5e (from title/tags only)", methodology=None,
  variants=[dict(name="oQ2.5e", size=None)],
  flags=["No quality claims for the quant (card read in full): card is YAML front matter and a title only [L1-11]",
         "No size stated on card"])

repo(repo="gabmiranda/Qwen3.8-Flash-Next-oQ4e-mtp", sha="e3ce05a5a391653149da99d6ab7b16a9f0665c1a", fmt="MLX",
  runtime="not stated (oQ tag)", weights_modified=False, method="oQ4e (from title/tags only)", methodology=None,
  variants=[dict(name="oQ4e", size=None)],
  flags=["No quality claims for the quant (card read in full): card is YAML front matter and a title only [L1-11]",
         "No size stated on card"])

repo(repo="glyd/Qwen3.8-Flash-Next-hummingbird", sha="a101fa6e2ee04e829a9f6fcb9614464127e14176", fmt="Glyd proprietary format",
  runtime="Glyd engine (BSL 1.1, compiled) only", weights_modified=False,
  method="Usage-based expert compression: most-routed experts keep most detail; precise ~2.7 bpw / fast ~2.1 bpw experts; non-expert weights (attention, embeddings, n-gram) in model/",
  methodology=dict(reference="none (absolute scores, no BF16 run on the card)", dataset="Inspect AI GSM8K and MMLU-Pro as shipped, 150 questions each; one-prompt game task (Star Drift) x6",
    ctx=None, n_samples=150, tool="Inspect AI; greedy, thinking off, 3,072-token cap", L=[85,93]),
  table_header_L=87,
  variants=[
    dict(name="precise", size=82, unit="GB", basis="disk", m={"bench:GSM8K":"148 / 150","bench:MMLU-Pro":"124 / 150","bench:Star Drift playable":"3 of 6"}, L=89, L_extra=[90,91,59,61]),
    dict(name="fast", size=68, unit="GB", basis="disk", m={"bench:GSM8K":"148 / 150","bench:MMLU-Pro":"118 / 150","bench:Star Drift playable":"1 of 6"}, L=89, L_extra=[90,91,59,63]),
  ],
  flags=["No BF16 baseline on the card; scores are absolute on 150-question subsets [L85-91]",
         "Sizes are model/ (37 GB) plus one expert folder (45 GB precise / 31 GB fast) [L55-59]",
         "bpw figures are approximate ('about 2.7' / 'about 2.1') averages over experts only [L61-63]",
         "Runs only on Glyd's own engine; not loadable in Transformers, vLLM or llama.cpp [L29-31]",
         "Text only; vision encoder not supported [L95]"])

repo(repo="halt95/Qwen3.8-Flash-Next-W4A16-Merlin", sha="cec53e9f90e5995b81691604ec0b32191debf797", fmt="W4A16 (INT4 AutoRound experts) + INT8 GDN + FP8 PLE",
  runtime="patched vLLM from halt95/qwen38-flash-next-3090s (v2.5.2)", weights_modified=False,
  method="Routed experts INT4 g128 from Intel AutoRound release; MTP experts INT4; GDN projections INT8 per channel; PLE FP8 (RadixArk); attention/norms/embeddings/vision BF16; FP8 KV with calibrated static scales",
  methodology=dict(reference="BF16 model", dataset="GSM8K (200), structured output (80), 262K needle, divergence check — details not stated on the card", ctx=None, tool="not stated (linked bench cards)", L=[44,45]),
  variants=[dict(name="W4A16-Merlin", size=115, unit="GiB", basis="disk", m={"bench:GSM8K":"198/200","bench:structured output":"80/80"}, L=45, L_extra=[56])],
  flags=["Quality summary is one line: 'gated against the BF16 model (divergence, 262K needle, GSM8K 198/200, structured output 80/80)'; no BF16 scores or divergence values on the card [L44-45]",
         "Expert quantization is Intel's AutoRound release; PLE FP8 table from RadixArk [L117, L120]",
         "Tested on one setup (4x RTX 3090) with the companion engine; stock vLLM not tested [L126, L130]"])

repo(repo="huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF", sha="43b4c1a8871afcc27e37bf996a431528b3d957e5", fmt="GGUF",
  runtime="llama.cpp / Strata", weights_modified=True,
  method="Abliterated (refusal removal) v2, ablation emphasis adjusted, no experts ablated; GGUFs described as coming from unsloth/Qwen3.8-Flash-Next-GGUF and ukisai Swift series",
  methodology=None,
  variants=[dict(name="UD-Q4_K_XL", size=None)],
  flags=["No quality claims for the quant (card read in full): card describes abliteration, Strata/llama.cpp usage and usage warnings only [L30-76]",
         "Weights modified: abliterated / uncensored [L30-34]",
         "Card says the GGUFs come from unsloth/Qwen3.8-Flash-Next-GGUF; how abliteration relates to those files is not explained [L40, L56]",
         "No sizes stated on card"])

repo(repo="infosave/Qwen3.8-Flash-Next-cmf", sha="d043fa6d77f72c0e359c3de1c41ac5ae62e70d3e", fmt="CMF (cortiq)",
  runtime="cortiq (Rust, Vulkan) >= 0.8.15", weights_modified=False,
  method="Quantized from bf16: q2tp experts 2-bit gate/up + 4-bit down, q4tp experts 4-bit; attention, embeddings and output head 8-bit; text only",
  methodology=dict(reference="none (no bf16 perplexity given)", dataset="1,536 tokens of Rust source", ctx=None, tool="not stated (cortiq)", L=[68,69]),
  table_header_L=59,
  variants=[
    dict(name="q2tp", size=76.95, unit="GB", basis="disk", m={"ppl:Rust code":"4.763"}, L=61),
    dict(name="q4tp", size=97.12, unit="GB", basis="disk", m={"ppl:Rust code":"4.111"}, L=62),
  ],
  flags=["Single small perplexity eval (1,536 tokens of Rust source) with no BF16 reference value [L69]",
         "Vision encoder not included (text only) [L32, L198]",
         "MTP files listed separately (1.08 GB / 1.50 GB) [L63-64]"])

repo(repo="intel-ai/Qwen3.8-Flash-Next-MXFP4-BF16SharedExperts-CT-AutoRound", sha="a0ed54ef4582de8627e2aa900f03348bf657d75c", fmt="compressed-tensors mixed-precision (MXFP4/MXFP8)",
  runtime="vLLM >= 0.29.0 (evaluated on 0.29.1rc1.dev528), TP=1", weights_modified=False,
  method="AutoRound 0.16.0 --model_free (RTN, no calibration): routed experts MXFP4 W4A4 g32, attention and GDN MXFP8 W8A8 g32; shared experts, indexer, PLE, router, MTP, embeddings, lm_head, vision BF16",
  methodology=dict(reference="Qwen/Qwen3.8-Flash-Next bf16 baseline row in the same table (whether re-run under this harness not stated)",
    dataset="GSM8K (chat template, few-shot multiturn), MMLU, PIQA, HellaSwag",
    ctx=8192, tool="lm-eval 0.4.13 + vLLM 0.29.1rc1.dev528, TP=1, bf16 KV, seed=42, enable_thinking=false", L=[85,86,87]),
  table_header_L=89,
  variants=[dict(name="MXFP4 + BF16 shared experts", row_label="This checkpoint", size=180.0, unit="GB", basis="disk",
    m={"bench:GSM8K strict":"0.9666","bench:GSM8K flexible":"0.9659","bench:MMLU":"0.8643","bench:PIQA acc":"0.8145","bench:PIQA acc_norm":"0.8275","bench:HellaSwag acc":"0.6832","bench:HellaSwag acc_norm":"0.8694"},
    L=92, L_extra=[47])],
  ladder=[dict(name="Sibling variant (shared_expert = MXFP8)",
    m={"bench:GSM8K strict":"0.9621","bench:GSM8K flexible":"0.9621","bench:MMLU":"0.8626","bench:PIQA acc":"0.8226","bench:PIQA acc_norm":"0.8270","bench:HellaSwag acc":"0.6832","bench:HellaSwag acc_norm":"0.8707"}, L=93)],
  cross=[dict(of="nvidia/Qwen3.8-Flash-Next-NVFP4", variant="NVFP4 (ModelOpt)", size=132.7, measured=False, unit="GB", basis="disk", L=113, flags=["quoted, not measured"])],
  baseline={"bench:GSM8K strict":"0.9674","bench:MMLU":"0.8652","bench:PIQA acc":"0.8194","bench:HellaSwag acc":"0.6927","L":91},
  flags=["Baseline row gives only strict/acc values ('—' for flexible/acc_norm), so acc_norm figures have no BF16 comparison [L91]",
         "Card states deltas vs the sibling variant are <= 0.6 sigma; standard errors themselves not printed [L95-97]",
         "Size 180.0 GB dominated by the 102 GB BF16 PLE table [L39, L47-48]",
         "NVIDIA NVFP4 entry is a size/design comparison only (132.7 GB), no quality numbers measured for it [L110-118]",
         "TP>1 produces degenerate outputs on current vLLM's MXFP4 MoE path (card attributes to upstream) [L72-75]"])

repo(repo="jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4", sha="f24d2b68ff2814f24455ae86717be276619b5664", fmt="NVFP4 (ModelOpt 0.46.0)",
  runtime="Pennyroyal/SGLang", weights_modified=True,
  method="ModelOpt NVFP4 W4A4 g16 on selected Linear modules of OrcaRouter's uncensored checkpoint; attention/GDN, routers, shared experts, PLE (FP8 E4M3), vision, embeddings, lm_head, MTP excluded",
  methodology=None,
  variants=[dict(name="ModelOpt NVFP4", size=135.3, unit="GB", basis="disk", L=43)],
  flags=["No quality claims for the quant (card read in full): only daily-use runtime evidence; card explicitly makes no benchmark claim [L45]",
         "Weights modified: source is OrcaRouter/Qwen3.8-Flash-Next-Uncensored [L29, L37]",
         "Mentions community reports of it outperforming the stock model on agentic tasks, with no numbers [L31]",
         "Size also given as 126 GiB [L43]; source license metadata conflicts noted by the publisher [L53]"])

repo(repo="kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF", sha="ce1bddb037c4980f292ff5039b8090b8bd396f04", fmt="GGUF (ROCmFP4)",
  runtime="kingjones30/ROCmFPX fork (qwen4exp + ROCmFP4) with bundled patches", weights_modified=True,
  method="Q4_0_ROCMFP4_FAST: attention, experts, token embeddings and PLE all ROCmFP4 (TYPE_101), output head Q6_K; quantized from orcarouter's uncensored BF16",
  methodology=dict(reference="publisher's own aligned build of the same tier", dataset="harmful (24), harmless (12), quality (8) prompts; prompts and completions not published",
    ctx=None, tool="same harness, greedy (not named)", L=[207,216]),
  table_header_L=209,
  variants=[dict(name="Q4_0_ROCMFP4_FAST (Uncensored)", row_label="this build", size=87.94, unit="GiB", basis="disk", bpw=4.27, mem=63.3, mem_unit="GiB", mem_basis="vram",
    m={"quality_set":"6/8","harmless_ok":"11","harmful_comply":"22"}, L=213, L_extra=[211,212,130,181])],
  ladder=[dict(name="aligned build, same tier (publisher's own)", m={"quality_set":"6/8","harmless_ok":"10","harmful_comply":"0"}, L=213, L_extra=[211,212])],
  flags=["Quality evidence is an 8-prompt quality split (6/8, same two failures as the aligned build); prompts not published [L207-216]",
         "Weights modified: abliterated by orcarouter; this repo only converts and quantizes [L96-97, L123-125]",
         "Comparison is against the publisher's aligned ROCmFP4 build, not BF16 [L207]",
         "GPU memory 63.3 GiB resident at full offload [L181]"])

repo(repo="manuuuel/mica-qwen3.8-flash-next-q4", sha="354af7ad9d757c0b8aba916926f1f3c80ff5cf4b", fmt="GGUF (Mica)",
  runtime="Mica (Apple silicon, C)", weights_modified=False,
  method="Importance-calibrated recipe: experts gate/up Q4_K, down MXFP4; dense/head/shared expert/n-gram key-value Q8_0; token embedding and n-gram table BF16 bit for bit; 252 calibration texts x 512 tokens",
  methodology=dict(reference="authors' unquantised weights run in the authors' code (single precision on CPU for the KL table)",
    dataset="Mica's 100 quality texts x 512 tokens (25 WikiText-2 test, 25 CPython stdlib v3.14.7, 25 No Robots test in chat template, 25 Wikipedia in nine languages), 51,100 predicted positions; KL over the full 248,320-token vocabulary",
    ctx=512, n_tokens=51100, tool="Mica make quality (GPU) and tools/quality/quality.py (authors' code)", L=[87,88,89,90,91,93,94,95,105,106,107,108]),
  table_header_L=110,
  variants=[dict(name="mica q4", size=177280263200, unit="bytes", basis="disk",
    m={"mean_kld:All":"0.0399","kld_p99:All":"0.585","top1_pct:All (authors' code)":"94.00%","ppl_delta_pct:All (authors' code)":"+0.14%",
       "mean_kld:English prose":"0.0542","mean_kld:Code":"0.0080","mean_kld:Chat":"0.0567","mean_kld:Other languages":"0.0407",
       "nats_per_token:All (Mica)":"1.5510","ppl_delta_pct:All (Mica)":"+0.12%","top1_pct:All (Mica)":"93.98%"},
    L=116, L_extra=[112,113,114,115,103,27])],
  baseline={"nats_per_token:All":"1.5498","L":103},
  flags=["Two measurement paths on the same 100 texts: Mica's GPU engine (nats/token, perplexity change, same top token) and the authors' code with this file's weights (adds KL mean and 99th percentile) [L93-116]",
         "Code text shows slightly lower perplexity than the original (-0.11% / -0.08%) [L100, L113]",
         "Calibration texts are stated to come from documents disjoint from the quality texts [L75-76]",
         "Vision tower shipped as a separate BF16 GGUF (897,881,920 bytes); importance statistics file published [L28-29]"])

repo(repo="mbehr90/Qwen3.8-Flash-Next-nvfp4", sha="d28eef6b7a1d3e19f7c581a47e48e96e2dbd321d", fmt="NVFP4A16 (compressed-tensors)",
  runtime="vLLM with Qwen4ExpForConditionalGeneration plugin (Marlin on H100)", weights_modified=False,
  method="NVFP4A16 weight-only (fp4 weights, fp8 group-16 scales, bf16 activations) on the 48x512 routed experts, quantized shard by shard without calibration",
  methodology=dict(reference="bf16 source, same table", dataset="GSM8K, 200 questions, via chat endpoint with final number parsed", ctx=None, n_samples=200, tool="own chat-endpoint script (lm-eval strict-match not used); 4x H100", L=[24,26,32,33,34,35]),
  table_header_L=26,
  variants=[dict(name="NVFP4", size=170.2, unit="GiB", basis="disk", m={"bench:GSM8K (200 q, chat)":"0.960","mean_answer_chars":"1087"}, L=30, L_extra=[20])],
  ladder=[dict(name="FP8 (mbehr90/Qwen3.8-Flash-Next-fp8)", size=220.8, unit="GiB", basis="disk", m={"bench:GSM8K (200 q, chat)":"0.965","mean_answer_chars":"1146"}, L=29)],
  baseline={"bench:GSM8K (200 q, chat)":"0.950","mean_answer_chars":"1183","L":28},
  flags=["Single benchmark, n=200; card states standard error ~1.5 pp and all three statistically indistinguishable [L32-33]",
         "Quantized model scores above the bf16 source (0.960 vs 0.950), within the stated error [L28-30]",
         "No calibration (weight-only scheme) [L22]"])

repo(repo="mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit", sha="52a21f4a18963bffa64f5a356e10459f2513a900", fmt="MLX (mlx-optiq)",
  runtime="mlx-optiq >= 0.5.14 with SSD expert streaming", weights_modified=False,
  method="2-bit routed experts; 4-bit attention, DeltaNet, shared expert, embeddings, LM head and n-gram table; 8-bit hyper-connections; bf16 router",
  methodology=None,
  variants=[dict(name="OptiQ-2bit", size=81, unit="GB", basis="disk", mem=4.6, mem_unit="GB", mem_basis="resident", L=45, L_extra=[46])],
  flags=["No quality claims for the quant (card read in full): card states no Capability Score is published; only a Flappy Bird demo [L31-35, L49]",
         "Peak memory ~4.6 GB is with routed experts and n-gram table streamed from SSD; decode ~3.8 tok/s [L46-47]",
         "MTP head not included; vision untested [L51]"])

repo(repo="mxxm-t/Qwen3.8-Flash-Next-MTP-GGUF", sha="c7c423f1d0e192130b97e4f600941e4dfdb8660a", fmt="GGUF (MTP draft head only)",
  runtime="mxxm-t/mx-llama.cpp fork", weights_modified=False,
  method="NextN/MTP draft head extracted from mtp.* tensors via convert_hf_to_gguf.py --mtp; Q8_0 and BF16",
  methodology=dict(reference="BF16 head file (draft-token identity)", dataset="prompts tested (not listed), temperature 0", ctx=None, tool="llama-server --spec-type draft-mtp", L=[40,41]),
  table_header_L=35,
  variants=[
    dict(name="qwen4exp-mtp-head-Q8_0", size=4.14, unit="GB", basis="disk", m={"draft_acceptance_pct":"75.1%"}, L=37, L_extra=[41]),
    dict(name="qwen4exp-mtp-head-BF16", size=7.77, unit="GB", basis="disk", m={"draft_acceptance_pct":"75.1%"}, L=38, L_extra=[41]),
  ],
  flags=["Not a full model: draft head only (~2.6B params plus embeddings) [L28-31]",
         "Only fidelity claim: Q8_0 and BF16 heads give identical draft tokens at temperature 0 (181 drafted, 136 accepted) [L40-41]"])

repo(repo="nitinpanj/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF", sha="967e14129dc685801fd0fefe6f22d0f8f945717b", fmt="GGUF",
  runtime="qwen4exp-enabled llama.cpp fork (e.g. unslothai/llama.cpp)", weights_modified=False,
  method="Q4_0 weights with Q8_0 output/embedding tensors, imatrix (calibration-v6, 926 entries / 582 chunks)",
  methodology=None,
  variants=[dict(name="Q4_0-Q8out v3", size=96, unit="GiB", basis="disk", L=40, L_extra=[48,49,50])],
  flags=["No quality claims for the quant (card read in full): card says it was 'built and benchmarked locally' but gives no numbers [L61-62]"])

repo(repo="orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF", sha="d27c0a15447dd92ae7c90919352c6810496945b1", fmt="GGUF",
  runtime="not stated", weights_modified=True, method="not stated (repo name indicates an uncensored derivative)", methodology=None,
  variants=[dict(name="(no variants listed)", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot is empty",
         "weights_modified set from the repo name ('Uncensored'); the card itself gives no description"])

repo(repo="paperniuk/Qwen3.8-Flash-Next-MTP-block", sha="d98c24aabe795ada199e6e84b4fb41e1a07b317c", fmt="GGUF (MTP block only)",
  runtime="paperniuk/ds4 fork (m1-flash-next)", weights_modified=False,
  method="32 blk.48.* MTP tensors copied byte for byte from ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2 (Q4_K gate/up, MXFP4 down, Q8_0 dense, F16 mixers)",
  methodology=None,
  variants=[dict(name="MTP-block", size=1.52, unit="GB", basis="disk", L=24)],
  flags=["No quality claims for the quant (card read in full): only speculative-decoding speeds and draft acceptance; output identical to plain decoding [L52-64]",
         "Not a full model: MTP block only, no requantization [L16-17, L31-35]"])

repo(repo="r0b0tlab/Qwen3.8-Flash-Next-NVFP4-W4A16-sm121", sha="8b73dc60cfa9b224100daaf3f44e9fe7825781dc", fmt="NVFP4 W4A16 (ModelOpt)",
  runtime="SGLang (source 7d2e8fcb), 2x GB10 TP=2", weights_modified=False,
  method="Self-quantized ModelOpt NVFP4 W4A16, BF16 activations, BF16 mtp.* NEXTN tensors; FP8 E4M3 KV at runtime",
  methodology=dict(reference="none (absolute pass counts, no BF16 run on the card)", dataset="Q200-v2: GSM8K 80, HumanEval 40, IFEval 40, hard reasoning 20, BFCL v4 multi_turn_base structural-hard20 subset",
    ctx=262144, n_samples=200, tool="strict full-context AR profile, native thinking low effort; bfcl-eval 2025.12.17 for BFCL", L=[44,45,59,70,73,74,75]),
  table_header_L=61,
  variants=[dict(name="NVFP4 W4A16", size=None,
    m={"bench:Q200-v2 overall":"188/200 (94.0%)","bench:GSM8K (80)":"80","bench:HumanEval (40)":"39","bench:IFEval (40)":"36","bench:Hard reasoning (20)":"20","bench:BFCL v4 structural-hard20":"13"},
    L=68, L_extra=[63,64,65,66,67,70])],
  flags=["Own composite suite (Q200-v2) with no BF16 baseline on the card [L59-71]",
         "BFCL figure is a 20-case subset, not a full-category score [L73-76, L138]",
         "Quality measured on the strict AR profile; public run uses a different NEXTN profile [L44-47]",
         "No checkpoint size stated (131 shards) [L24]"])

repo(repo="scottlowry/Qwen3.8-Flash-Next-oQ4e-mtp", sha="998d2d58c7e9fcad9a42ed2c30e06028c4c7d3aa", fmt="MLX",
  runtime="oMLX", weights_modified=False, method="oQ (oMLX v0.7.1.dev1) mixed precision, 4 bits, group 64", methodology=None,
  variants=[dict(name="oQ4e", size=None)],
  flags=["No quality claims for the quant (card read in full): only a four-line quantization-details list [L15-20]", "No size stated on card"])

repo(repo="scottlowry/Qwen3.8-Flash-Next-oQ8e-fp16-mtp", sha="6e1875bf3f5d4d20fb5564834fdbad1d1e2525f6", fmt="MLX",
  runtime="oMLX", weights_modified=False, method="oQ (oMLX v0.7.1.dev1) mixed precision, 8 bits, group 64, fp16", methodology=None,
  variants=[dict(name="oQ8e-fp16", size=None)],
  flags=["No quality claims for the quant (card read in full): only a four-line quantization-details list [L15-20]", "No size stated on card"])

repo(repo="sleepyeldrazi/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ5_0", sha="cb186e74baa7486de61ada9ae2380882e0f7c595", fmt="GGUF",
  runtime="stock llama.cpp (trunk); sleepyeldrazi/llama.cpp-strix feat/strix-prod for MTP", weights_modified=False,
  method="ISTA-DASLab GSQ-RCO IQ3_XXS pack with 194 hyper-connection tensors requantized BF16 -> Q5_0; all other tensors byte-identical; MTP draft sidecars (full and 65,280-token head)",
  methodology=None,
  variants=[dict(name="IQ3_XXS-HCQ5_0", size=46.2, unit="GiB", basis="trunk_only", m={"greedy_identical_prefix_tokens":"47"}, L=68, L_extra=[57,69])],
  cross=[dict(of="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", variant="IQ3_XXS (base pack)", measured=False, m={"ppl:wikitext":"3.9044"}, L=62, flags=["quoted, not measured"])],
  flags=["Card states perplexity was not measured for this build; quality gates are a 128-token greedy comparison (identical for first 47 tokens), a 33K needle and a 1K coherence check [L56-62]",
         "Base-pack wikitext PPL 3.9044 is quoted from GSQHalo.cpp, not measured by this publisher [L61-62]",
         "Trunk shard 46.2 GiB plus 54.4 GiB PLE shard (Q8_0, byte-identical to base pack) [L68-69]",
         "Not an original full quant: derived from ISTA-DASLab's IQ3_XXS pack [L27-29]"])

repo(repo="tarruda/Qwen3.8-Flash-Next-GGUF", sha="0eb873e932f0a995bb7ae6b79666a25f717a9531", fmt="GGUF",
  runtime="llama.cpp", weights_modified=False,
  method="imatrix (tag) GGUF recipes on Q8_0 default with expert/PLE overrides",
  methodology=None,
  table_header_L=24,
  variants=[
    dict(name="Q5_K", size=None, bpw=7.14, L=26),
    dict(name="IQ4_NL", size=None, bpw=4.62, L=27),
  ],
  flags=["No quality claims for the quant (card read in full): only recipe table and a launch script [L22-50]",
         "Sizes are given in MiB only (150694.87 MiB / 97480.69 MiB), so not recorded in GB/GiB [L26-27]"])

repo(repo="trustos/Qwen3.8-Flash-Next-GSQ-RCO-Q2-MTPLX", sha="09d7bd17aea10cd6b4a53afa9a0d9c8bdcccaacd", fmt="MLX (MTPLX)",
  runtime="MTPLX 2.12.2 (+ bundled mtplx-q2-stage3 lane)", weights_modified=False,
  method="Routed experts repacked bit for bit from ISTA-DASLab GSQ-RCO Q2_0 GGUF as MLX affine 2-bit g64 (scales stored bf16); all other tensors from the MTPLX Optimized Speed pack",
  methodology=None,
  variants=[dict(name="GSQ-RCO Q2 MTPLX", size=77.3, unit="GB", basis="disk", mem=45, mem_unit="GiB", mem_basis="resident", L=51, L_extra=[52])],
  flags=["No quality claims for the quant (card read in full): Quality section says 'Not evaluated'; ISTA's evaluation covers a different dense-layer pairing [L147-155]",
         "Only signals: MTP acceptance 79% vs 80% (code) and 43% vs 53% (prose) against the 4-bit pack; coherent answers [L153-155]",
         "bf16 scale storage rounds each scale by at most about 0.2% [L133]"])

repo(repo="vcruz305/Qwen3.8-Flash-Next-EXL3-SAGE-4.15bpw", sha="d5d792a98197dce5bfd019704e6212913a496feb", fmt="EXL3",
  runtime="vcruz305/exllamav3 fork @ f6e42ec (TabbyAPI)", weights_modified=False,
  method="SAGE mixed-K EXL3 allocation for 4.14999 bits/param decoder target from Qwen/Qwen3.8-Flash-Next-FP8; head/ngram/vision 6 bits, MTP 4 bits",
  methodology=None,
  variants=[dict(name="EXL3 SAGE 4.15bpw", size=109.04, unit="GB", basis="disk", bpw=4.14999, L=44, L_extra=[40])],
  flags=["No quality claims for the quant (card read in full): card states quality gates have not run and the pack carries no quality claim [L25, L52-54, L72]",
         "Quantized from the FP8 release, not BF16 [L29, L35]"])

repo(repo="xautonomics/Qwen3.8-Flash-Next-UD-IQ4_XS.LAYR.GGUF", sha="9ee54f348b56af9268b2db10346c539df89b3273", fmt="GGUF (per-layer library)",
  runtime="dreamcatcher (xAutonomics fork of ik_llama.cpp)", weights_modified=False,
  method="unsloth UD-IQ4_XS (rev 38bb39ee97) sliced into one GGUF per transformer block plus shared parts; nothing re-quantized",
  methodology=None,
  variants=[dict(name="UD-IQ4_XS (unsloth, sliced)", size=87.8, unit="GiB", basis="disk", L=25)],
  flags=["No quality claims for the quant (card read in full): support status covers coherence, throughput and token-identity to the monolithic file, not quantization quality [L31-46]",
         "Not an original quant: tensors are byte-for-byte unsloth's UD-IQ4_XS [L18, L86]"])

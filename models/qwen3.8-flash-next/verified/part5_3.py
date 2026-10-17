# Re-extracted from verbatim README snapshots at pinned commits (snapshots/index.json -> cache/cards/, captured 2026-10-17).
# Queue: cache/autoq_3.json (previously regex-triaged only). Every card read in full; each variant/claim cites README line numbers (L) at that sha.
# size basis: disk = full download unless noted; vram/resident/ram_in_use = memory figures; trunk_only = weights excluding n-gram table.
R = []
def repo(**k): R.append(k); return k

repo(repo="AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP", sha="4fdb4a602ac8ed53afcad1874861ee0ecdd636e1", fmt="MLX",
  runtime="MLX-LM (oMLX / MTPLX branches for MTP)", weights_modified=False,
  method="AXQuant 1.9.0 mixed-precision affine (4/8-bit, group 32/64, BF16 protected tensors) from BF16; allocation from architecture priors, no calibration",
  methodology=None,
  variants=[dict(name="AXQ MXFP4 budget class (incl. MTP)", size=132.26, unit="GB", basis="disk", bpw=5.8769, L=48, L_extra=[46])],
  flags=["No quality claims for the quant (card read in full): card states quality versus BF16 or uniform baselines is 'Not published; no quality-retention claim' [L166, L27-29]",
         "Measured main-model BPW 5.7281 vs planned 6.3924; 5.8769 including MTP [L44-46]",
         "Runtime branch checks are described as development compatibility checks, not quality [L131-134]"])

repo(repo="BlivionIaG/Qwen3.8-Flash-Next-EXL3-3bpw-3inst", sha="ceaf21e9f6ec1a112fa74af6af6abbb257cb895d", fmt="EXL3",
  runtime="exllamav3 / opengfx1030 vllm-rdna fork", weights_modified=False,
  method="EXL3 trellis, 3inst codebook, LDLQ calibration on 32 dummy sequences; experts K=3, dense group K=6 (turboderp mul1 3.05bpw split shifted up one K)",
  methodology=None,
  variants=[dict(name="EXL3 3inst (experts K=3, dense K=6)", size=None)],
  flags=["No quality claims for the quant (card read in full): no size, KLD, PPL or benchmark figures; card only says outputs should be representative 'modulo expected quantization loss' [L20]"])

repo(repo="DhruvalLabs/Qwen3.8-Flash-Next-GGUF", sha="c257284cfa058996040d474a080cec19ba2fe326", fmt="GGUF",
  runtime="llama.cpp / Ollama / LM Studio", weights_modified=False, method="Q4_K_M (quant-kit), 3 shards",
  methodology=None,
  variants=[dict(name="Q4_K_M", size=111, unit="GB", basis="disk", L=33, L_extra=[43,44,45])],
  flags=["No quality claims for the quant (card read in full): 'Quality' column is a star rating (4 stars) with no measurement behind it [L41-45]",
         "Shard sizes 45.0 + 45.0 + 21.0 GB [L43-45]"])

repo(repo="GBP-DE/Qwen3.8-Flash-Next-oQ5e-mtp", sha="387311fe0412f2ff36f00fc05067a35598bd553a", fmt="MLX",
  runtime="oMLX 0.6.4 (SSD N-gram Offload, Lightning MTP)", weights_modified=False,
  method="oQ5e enhanced/imatrix from BF16; group size 64; sensitivity map from Jundot/Qwen3.8-Flash-Next-oQ4e-mtp",
  methodology=dict(reference="none (head-to-head vs Jundot oQ4e only; no BF16 reference)",
    dataset="five practical local prompts (8-person constraint puzzle, five-trigger state machine, Python debugging, four-tank state machine, 10-variable Boolean/SAT); one run each as presented",
    ctx=None, tool="oMLX 0.6.4 on MacBook Pro M5 Max 128 GB, SSD N-gram Offload ON, Lightning MTP ON, TurboQuant KV/SpecPrefill/DFlash off",
    L=[73,74,75,76,77,78,79,80,104,106,256]),
  variants=[dict(name="oQ5e", size=125.7, unit="GB", basis="disk", mem=102, mem_unit="GB", mem_basis="ram_in_use",
    m={"test:8-person constraint puzzle (thinking off)":"Exact solution",
       "test:five-trigger state machine (thinking on)":"Two synthetic post-hoc verification values (checksum and signature) were calculated incorrectly",
       "test:Python debugging (thinking on)":"coding failure despite a correct analysis and correct expected results",
       "test:four-tank state machine":"Exact",
       "test:Boolean/SAT":"Exact"},
    L=136, L_extra=[60,87,171,211,236,250])],
  cross=[dict(of="Jundot/Qwen3.8-Flash-Next-oQ4e-mtp", variant="oQ4e", mem=93, mem_unit="GB", mem_basis="ram_in_use",
    m={"test:8-person constraint puzzle (thinking off)":"Failed",
       "test:five-trigger state machine (thinking on)":"did not complete the required answer",
       "test:Python debugging (thinking on)":"never produced the required final answer",
       "test:four-tank state machine":"Exact",
       "test:Boolean/SAT":"Exact"},
    L=144, L_extra=[86,180,220,236,250])],
  flags=["Quality evidence is pass/fail outcomes on five self-made prompts; card itself calls them 'practical local tests rather than a standardized academic evaluation suite' [L256]",
         "Memory figures are total system memory including macOS, oMLX and other apps (e.g. Chrome), not model-only [L91]",
         "No BF16 or KLD/PPL reference",
         "Card states this earlier oQ5e is superseded by WLSRefit versions for new downloads [L27-38]"])

repo(repo="Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF", sha="c104e657903116b7ad852cdd31b9775ad4c7e5d9", fmt="GGUF (MTP draft head)",
  runtime="llama.cpp b11433+ (draft-mtp) / Xyntetik Runner 1.1.1", weights_modified=False,
  method="MTP (NextN) head extracted from BF16 checkpoint; matrices Q8_0, norms F32, router/hyper-connection BF16",
  methodology=None,
  variants=[dict(name="MTP-Q8_0 (draft head)", size=4165605056, unit="bytes", basis="disk", L=28)],
  flags=["No quality claims for the quant (card read in full): this is a draft head; card reports only decode speed and draft acceptance (69-82%) [L35-47, L92-94]",
         "Card notes drafted llama.cpp run is not byte-identical to plain run [L50-51]",
         "trunk-with-head/ set repackages ISTA-DASLab's IQ3_S trunk with tensor bytes untouched [L96-106]"])

repo(repo="Jundot/Qwen3.8-Flash-Next-oQ2.7e-mtp", sha="0e751a5005421ee87256efbb846674c7961c0982", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ2.7e: sensitivity-based bit allocation + imatrix (1024x512) + weighted LSQ scale/bias refit; n-gram table 4-bit outside budget",
  methodology=None,
  variants=[dict(name="oQ2.7e", size=85.205, unit="GB", basis="disk", bpw=3.787, L=50, L_extra=[23,40])],
  flags=["No quality claims for the quant (card read in full): only bit distribution and storage breakdown [L29-50]",
         "3.305 bpw excluding the n-gram table, 3.787 for the complete checkpoint [L23]"])

repo(repo="Jundot/Qwen3.8-Flash-Next-oQ4e-fp16-mtp", sha="3ed70e9799280af861b83443c2098a74be96d210", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ4e (imatrix 1024x512 + weighted LSQ refit); scales/biases and unquantized tensors FP16, vision FP32",
  methodology=None,
  variants=[dict(name="oQ4e-fp16", size=107.461, unit="GB", basis="disk", bpw=4.776, L=54, L_extra=[25,44])],
  flags=["No quality claims for the quant (card read in full): only bit distribution and storage breakdown [L31-54]"])

repo(repo="Kashapps/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF", sha="1aa1c0999a6adf4cc8d961aecb742557eadad9c2", fmt="GGUF (ROCmFP4, fork-only)",
  runtime="kingjones30/ROCmFPX fork (qwen4exp + ROCmFP4)", weights_modified=True,
  method="Q4_0_ROCMFP4_FAST recipe from orcarouter/Qwen3.8-Flash-Next-Uncensored BF16 (abliterated); all ROCmFP4 except Q6_K output head",
  methodology=dict(reference="publisher's aligned (non-abliterated) build of the same tier, not BF16",
    dataset="own prompt splits: 24 harmful, 12 harmless, 8 quality; prompts and completions not published",
    ctx=None, tool="same harness, greedy", L=[144,153]),
  variants=[dict(name="Q4_0_ROCMFP4_FAST (uncensored)", row_label="this build", size=87.94, unit="GiB", basis="disk", bpw=4.27, mem=63.3, mem_unit="GiB", mem_basis="vram",
    m={"refusal:Harmful (24)":"22 comply","Harmless (12)":"11 ok","quality:Quality (8)":"6/8"}, L=72, L_extra=[118,148,149,150])],
  ladder=[dict(name="aligned build (same tier)", m={"refusal:Harmful (24)":"0 comply","Harmless (12)":"10 ok","quality:Quality (8)":"6/8"}, L=148, L_extra=[149,150])],
  flags=["Weights are abliterated (orcarouter), not a faithful quant of Qwen's model [L65-67]",
         "Quality evidence is an 8-item set with prompts not published [L150, L153]",
         "Comparison is abliterated quant vs aligned quant of same tier, not vs BF16 [L144]",
         "Card's patch download URL and field-report links point to kingjones777 repositories [L53, L103]; card text and measurement tables match kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-GGUF"])

repo(repo="Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-fp16-DWQ-MTP-Vision-MTPLX", sha="0bb3ca69790f8ec5caffac9c069ab3ee3eaa263e", fmt="MLX (MTPLX layout)",
  runtime="MTPLX >= 2.11.2", weights_modified=True,
  method="REAP expert pruning 512->320 + oQ3e (3-bit routed experts, mixed trunk) + end-to-end KL-DWQ; MTPLX repack with self-distilled 8-bit MTP sidecar; 16-bit tensors recast BF16->FP16, vision FP32",
  methodology=dict(reference="BF16 sibling repo's tensors (value-domain FP16 conversion check only)",
    dataset="all 5,786,885,968 FP16-typed values in the pack", ctx=None, tool="not stated", L=[86,92]),
  variants=[dict(name="REAP320 oQ3e fp16 DWQ MTPLX", size=None, bpw=4.28, mem=39.7, mem_unit="GiB", mem_basis="resident",
    m={"fp16_exact_values_pct":"99.99973%","fp16_overflow_values":"0","fp16_flushed_to_zero":"598"}, L=92, L_extra=[91,93,121,189])],
  flags=["No task-quality claims for the quant: card states 'no task or vision-benchmark score is claimed yet' [L224]; teacher-gap tables are referenced on the MLX sibling card, not reproduced here [L117-118]",
         "Recorded figures measure FP16 recast fidelity vs the BF16 sibling, not quantization error vs Qwen's BF16 [L84-95]",
         "Pruned: REAP keeps 320 of 512 experts [L117]",
         "Speed figures measured on the BF16 sibling pack on M4 Pro; no M1/M2 timing claimed [L164-166]"])

repo(repo="MonumentalSystems/Qwen3.8-Flash-Next-control-vectors", sha="455a5a15b7f977eedf48db280936bbe613d9fdd1", fmt="GGUF control vectors",
  runtime="llama.cpp-compatible control-vector loaders / Atlas", weights_modified=False,
  method="Contrastive mean-difference control vectors (verbosity, comment-density) derived from NVFP4-weight activations; not a weight quantization",
  methodology=None,
  variants=[dict(name="control vectors (verbosity, comment-density)", size=None)],
  flags=["No quality claims for the quant (card read in full): repo contains activation-steering vectors, not model weights; measured tables report steering effects (token-length deltas, comment ratios), not quantization fidelity [L99-110, L141-155]"])

repo(repo="RedHatAI/Qwen3.8-Flash-Next-NVFP4", sha="c8f2fb1b9869f686b214782036123b10ff96d14a", fmt="NVFP4 (compressed-tensors)",
  runtime="vLLM", weights_modified=False,
  method="LLM Compressor NVFP4 W4A4 on MoE experts only, rest BF16; calibration 1024 samples of open-perfectblend",
  methodology=dict(reference="Qwen/Qwen3.8-Flash-Next (unquantized), same table", dataset="GPQA Diamond, MMLU-Pro, AIME 2026, LiveCodeBench, Terminal-Bench 2.1, SWE-bench Verified, SWE-bench Pro, DeepSWE 1.1",
    ctx=None, n_samples="not stated", tool="Inspect with vLLM server, single seed", L=[56,57,138]),
  table_header_L=64,
  variants=[dict(name="NVFP4 (RedHatAI)", size=None,
    m={"bench:GPQA Diamond":"92.9","bench:MMLU-Pro":"88.2","bench:AIME 2026":"100","bench:LiveCodeBench":"95.0","bench:Terminal-Bench 2.1":"86.1",
       "bench:SWE-bench Verified":"79.4","bench:SWE-bench Pro":"63.6","bench:DeepSWE 1.1":"62.3",
       "recovery:MMLU-Pro":"99.6","recovery:SWE-bench Verified":"98.5","recovery:DeepSWE 1.1":"94.3","recovery_avg":"99.1"},
    L=77, L_extra=[85,93,101,108,115,122,129,161,187,199,205])],
  baseline={"bench:GPQA Diamond":"90.4","bench:MMLU-Pro":"88.5","bench:AIME 2026":"96.7","bench:LiveCodeBench":"95.0","bench:Terminal-Bench 2.1":"85.5",
            "bench:SWE-bench Verified":"80.6","bench:SWE-bench Pro":"62.9","bench:DeepSWE 1.1":"66.0","L":76,"L_extra":[84,92,100,107,114,121,128]},
  cross=[
    dict(of="Inferact/Qwen3.8-Flash-Next-NVFP4", variant="NVFP4",
      m={"bench:GPQA Diamond":"91.4","bench:MMLU-Pro":"88.0","bench:AIME 2026":"100","bench:LiveCodeBench":"94.0","bench:Terminal-Bench 2.1":"84.5",
         "bench:SWE-bench Verified":"79.8","bench:SWE-bench Pro":"60.9","bench:DeepSWE 1.1":"58.1","recovery:DeepSWE 1.1":"88.0","recovery_avg":"97.6"},
      L=78, L_extra=[86,94,102,109,116,123,130,200,206]),
    dict(of="nvidia/Qwen3.8-Flash-Next-NVFP4", variant="NVFP4",
      m={"bench:GPQA Diamond":"92.4","bench:MMLU-Pro":"88.0","bench:AIME 2026":"100","bench:LiveCodeBench":"95.0","bench:Terminal-Bench 2.1":"86.4",
         "bench:SWE-bench Verified":"79.8","bench:SWE-bench Pro":"62.8","bench:DeepSWE 1.1":"56.7","recovery:DeepSWE 1.1":"85.9","recovery_avg":"98.0"},
      L=79, L_extra=[87,95,103,110,117,124,131,201,207]),
  ],
  flags=["Single-seed evaluations [L57]",
         "Recovery is capped at 100%, so above-baseline scores (e.g. GPQA 92.9 vs 90.4, AIME 100 vs 96.7) count as 100 [L138]",
         "Headline 'highest accuracy recovery' 99.1% vs 97.6% / 98.0% [L56]; the gap is largest on DeepSWE 1.1 (94.3 vs 88.0 / 85.9) [L199-201]",
         "Card does not state sample counts, context length or harness settings per benchmark",
         "No checkpoint size stated"])

repo(repo="Sawfwair/Qwen3.8-Flash-Next-MLX-Activation-3bit", sha="c699bd611366cbc441377275bd1b7a6d2e18e1df", fmt="MLX",
  runtime="mere.run / mlx-vlm (Qwen4Exp-aware)", weights_modified=False,
  method="Routed-expert banks fresh MLX affine Q3/group-64 from BF16, scales/biases refit on image+text expert-input second moments; other matrices Q4, n-gram Q4/g32",
  methodology=dict(reference="publisher's pinned published Q4 checkpoint (mere.run), not BF16",
    dataset="61 cases scored by exact expected output + 16-case sealed holdout (incl. 8 image/OCR cases)", ctx=None,
    tool="mere.run on 128 GiB Apple Silicon Mac, MTP disabled", L=[63,64,65,67,69,70]),
  variants=[dict(name="Activation-weighted 3-bit", size=83.49, unit="GiB", basis="disk", mem=62065713472, mem_unit="bytes", mem_basis="resident",
    m={"suite_exact_pass":"58 of 61","holdout_identical_to_Q4":"all 16"}, L=67, L_extra=[22,69,71])],
  ladder=[dict(name="published Q4 checkpoint", mem=77180432496, mem_unit="bytes", mem_basis="resident", L=72)],
  flags=["No BF16 comparison; card states the bounded suites 'don't establish general model quality or BF16 parity' [L75-76]",
         "Reference is the publisher's own Q4 checkpoint; the three misses matched Q4 output [L68]",
         "Memory is holdout peak footprint (19.58% below Q4) [L71-72]"])

repo(repo="SuperPUPS/Qwen3.8-Flash-Next-INT8x25-AttnBF16", sha="6c077c300961f92738c14379ae8157efe3285f00", fmt="compressed-tensors (INT8/INT4 experts, BF16 attention)",
  runtime="patched vLLM (vllm-qfn snapshot)", weights_modified=False,
  method="Community W4A16 base; 25 of 48 expert layers re-quantized to INT8 g128 from official FP8 checkpoint (msegrid scales); attention byte-copied BF16 from FP8 checkpoint",
  methodology=dict(reference="official FP8 checkpoint (proxy) for per-layer INT4 error and INT8 roundtrip cosine",
    dataset="1536 expert-matrix pairs per layer (weight-space); live gates: 3 golden probes, 10-task battery (temp=0), NIAH @128K (112,917 prompt tokens)",
    ctx=None, tool="vllm-qfn on 2x CMP 170HX, TP2+EP, MTP k=4", L=[63,64,65,66,95,96,110,111,113]),
  variants=[dict(name="INT8x25-AttnBF16", row_label="This checkpoint", size=212, unit="GB", basis="disk",
    m={"int8_roundtrip_cos":"0.99995","int8_roundtrip_cos_worst":"0.99990","golden_probes":"3/3","battery_10task":"10/10","niah_128k":"HIT"},
    L=96, L_extra=[36,110,111,113])],
  ladder=[dict(name="INT8x17-AttnINT4", size=202, unit="GB", basis="disk", L=36, L_extra=[120])],
  flags=["Per-layer rel-F/cos ranking table measures INT4 error of the W4A16 base layers used for selection, not this file's INT8 layers [L72-90]",
         "No output-level comparison vs BF16/FP8 (no KLD, PPL or benchmark scores); quality evidence is weight-roundtrip cosine and pass/fail gates [L96, L106-114]",
         "'the highest-quality W4-class release we could measure' is stated without a comparison table [L25-26]",
         "Wave-1 per-layer metrics were not retained [L69-70]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-8bit-MTP", sha="a98d2ff3811f1042bf092b82b65a549442f83ca3", fmt="MLX",
  runtime="oMLX 0.6.3rc3 / MLX-VLM 0.6.3", weights_modified=False,
  method="Uniform 8-bit affine, group 32, from BF16; multimodal and router gates BF16; native MTP kept",
  methodology=None,
  variants=[dict(name="MLX 8-bit g32", size=202.951, unit="GB", basis="disk", L=78)],
  flags=["No quality claims for the quant (card read in full): only MTP speed/acceptance and pass/fail checks (MTP on/off exact-output parity, coherence) [L120-126, L178-182]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-oQ8-MTP", sha="eb863705ad752e20f99eb74f71d1c1be4686216c", fmt="MLX",
  runtime="oMLX / MLX-VLM with qwen4_exp MTP", weights_modified=False,
  method="oQ8 near-uniform 8-bit affine, base group 32, from BF16; native MTP kept",
  methodology=None,
  variants=[dict(name="MLX oQ8", size=202.75, unit="GB", basis="disk", L=76)],
  flags=["No quality claims for the quant (card read in full): only MTP speed and deterministic smoke gates (hello / Paris / arithmetic) [L110-116]"])

repo(repo="Vitruvi0/qwen3.8-flash-next-nvfp4-fp8hybrid", sha="9a463b467a445f300d4492845e7614d310ca1441", fmt="NVFP4/FP8 (per repo name)",
  runtime="not stated", weights_modified=False, method="not stated (card empty)",
  methodology=None,
  variants=[dict(name="nvfp4-fp8hybrid", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes)"])

repo(repo="Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed", sha="34ae535f2c817737a9c73c364fd6f20629f9a8f7", fmt="MLX (MTPLX layout)",
  runtime="MTPLX", weights_modified=False,
  method="4-bit g64 experts and dense matrices; QSA projections 8-bit; GDN conv/state, norms, indexer, MTP head 16-bit; n-gram table as SSD-streamed sidecar",
  methodology=None,
  variants=[dict(name="Optimized Speed (4-bit, 8-bit attention)", size=115.1, unit="GB", basis="disk", mem=87, mem_unit="GiB", mem_basis="resident", L=122, L_extra=[124])],
  flags=["No quality claims for the quant (card read in full): 'Higher quality' vs Bare Speed sibling is asserted without measurements [L45, L111-113]; only speed and speculative-sampling exactness figures [L68-107]"])

repo(repo="airagrp/Qwen3.8-Flash-Next-mixed-6-8-ple8-mlx-serve", sha="b909fe3ab65866660e2a7cb4df30c4d78cb07dd8", fmt="MLX (mlx-serve pack)",
  runtime="mlx-serve 26.9.2 / MLX 0.32.2", weights_modified=False,
  method="Routed experts 6-bit g64; attention/GDN/HC/indexer/shared experts/lm_head 8-bit g64; embed 4-bit; n-gram table 8-bit g32 in ngram_table.bin; MTP experts 6-bit",
  methodology=None,
  variants=[dict(name="mixed 6/8, PLE 8-bit", size=105.2, unit="GB", basis="trunk_only", mem=96, mem_unit="GB", mem_basis="resident", L=114, L_extra=[32])],
  flags=["No quality claims for the quant (card read in full): '8-bit MTP-expert variant ... no measurable quality ... difference' and '--kv-quant 8 ... no quality loss' are stated without numbers [L87-88, L93]",
         "Trunk 105.2 GB plus 57.6 GB ngram_table.bin [L114]"])

repo(repo="ashbash/Qwen3.8-Flash-Next-MTP-Drafter-GGUF", sha="c82053e8db1eff3f5fcc219c3146fc2ed34258d6", fmt="GGUF (MTP draft head)",
  runtime="gmlx", weights_modified=False, method="MTP head tensors extracted from BF16 and re-encoded (Q8_0; Q4_K_L mixed)",
  methodology=None,
  variants=[dict(name="mtp-drafter-Q8_0", size=2.6, unit="GiB", basis="disk", L=32),
            dict(name="mtp-drafter-Q4_K_L", size=1.8, unit="GiB", basis="disk", L=33)],
  flags=["No quality claims for the quant (card read in full): draft heads only; no acceptance or quality figures"])

repo(repo="beginning-ai/Qwen3.8-Flash-Next-NVFP4", sha="9c8cb346a367fa083b552cf84ebdcdc334cbafc7", fmt="NVFP4 + FP8 (ModelOpt mixed)",
  runtime="patched SGLang (beginning-ai fork, commit 93540334d2)", weights_modified=False,
  method="ModelOpt 0.47.0: routed experts NVFP4 W4A4 g16, shared experts W4A16 NVFP4, attention FP8 128x128 blocks, PLE FP8 copied from Qwen FP8 release; activation scales max(own cnn_dailymail calib, RadixArk per-expert)",
  methodology=dict(reference="Qwen's reported BF16 DeepSWE score (113 tasks, high effort), quoted",
    dataset="DeepSWE v1.1, fixed 30-task subset of 113, mini-swe-agent harness, reasoning effort medium, one attempt per task", ctx=524288,
    tool="patched SGLang, FP8 KV, MTP 3 steps / 4 draft tokens, 1x RTX PRO 6000", L=[82,86,87,88]),
  variants=[dict(name="NVFP4 mixed (beginning-ai)", size=120, unit="GiB", basis="disk", mem=78, mem_unit="GB", mem_basis="vram",
    m={"bench:DeepSWE v1.1 (30-task subset)":"19/30 = 63.3%","partial:new-feature tests passed":"95.7%","partial:existing tests passed":"97.7%","conversation_replay_wellformed":"120 of 120"},
    L=92, L_extra=[23,24,101,102,144])],
  baseline={"bench:DeepSWE v1.1 (113 tasks, high effort, Qwen reported)":"58.7%","L":93},
  flags=["Baseline DeepSWE 58.7% is Qwen's reported score: quoted, not measured; different task count, effort and attempts [L93]",
         "Card itself says the two scores are not directly comparable; 95% range about ±17 points [L95-99]",
         "'A 4-bit head changed the top token at about 20% of positions' refers to an unshipped variant (head kept BF16) [L68-70]"])

repo(repo="cafonez/Qwen3.8-Flash-Next-HC-Q8", sha="986da0f41c04a303ad98002608b3ccad73760ddf", fmt="GGUF (ROCmFPX types)",
  runtime="ROCmFPX HIP llama-server", weights_modified=False,
  method="Q4_0_ROCMI4 network, Q3_0_ROCMFPX n-gram table; 200 hyperconnection up/down matrices restored from BF16 as Q8_0",
  methodology=None,
  variants=[dict(name="HC-Q8", size=92347778272, unit="bytes", basis="disk", bpw=4.11, L=21, L_extra=[22])],
  flags=["No quality claims for the quant (card read in full): only layout, run instructions and a decode speed at 262k [L103-105]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ2.5-mtp", sha="9588892b77d6010d7d00c0178704eeef4e25594e", fmt="MLX",
  runtime="oMLX v0.7.0", weights_modified=False, method="oQ2.5 mixed precision; sensitivity map from Jundot oQ4e-mtp (128x256)",
  methodology=None,
  variants=[dict(name="oQ2.5", size=71, unit="GB", basis="disk", bpw=3.15, L=27)],
  flags=["No quality claims for the quant (card read in full): only bit-width shares and size [L27-42]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ3-mtp", sha="cf5fd1a2f6484b3f5c033bf21df891bf5747a380", fmt="MLX",
  runtime="oMLX v0.7.0", weights_modified=False, method="oQ3 mixed precision; sensitivity map from Jundot oQ4e-mtp (128x256)",
  methodology=None,
  variants=[dict(name="oQ3", size=84, unit="GB", basis="disk", bpw=3.75, L=27)],
  flags=["No quality claims for the quant (card read in full): only bit-width shares and size [L27-42]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp", sha="0b047d1542f6778de69baa70c9663c9c99838832", fmt="MLX",
  runtime="oMLX v0.7.0", weights_modified=False, method="oQ4e imatrix (oqe_code_multilingual, 1,024x512) streamed from BF16; sensitivity map from Jundot oQ4e-mtp",
  methodology=None,
  variants=[dict(name="oQ4e", size=106, unit="GB", basis="disk", bpw=4.72, L=31)],
  flags=["No quality claims for the quant (card read in full): only bit-width shares, imatrix coverage (99.97% experts activated) and size [L31-52]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ8-mtp", sha="24a21ad6e8b328e765146b055446fa2e1ecb69b8", fmt="MLX",
  runtime="oMLX v0.7.0", weights_modified=False, method="oQ8; n-gram table 8-bit g32",
  methodology=None,
  variants=[dict(name="oQ8", size=195, unit="GB", basis="disk", bpw=8.66, L=26)],
  flags=["No quality claims for the quant (card read in full): only size [L26]"])

repo(repo="dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4", sha="be794b990578ef3031eccf9f28e675a289a09ee9", fmt="NVFP4 (ModelOpt FP4)",
  runtime="SGLang (2x DGX Spark TP2)", weights_modified=True,
  method="Abliteration (direct weight-level refusal removal) + NVFP4 W4A4 routed experts; attention, shared experts, PLE, vision, MTP higher precision",
  methodology=dict(reference="base model in identical harness (base precision not stated)",
    dataset="MMLU 40 Q/subject x 57 subjects (2,280 Q); HarmBench-320 real-harm subset (240 behaviors), greedy temp 0",
    ctx=None, tool="not stated", L=[47,48,69]),
  variants=[dict(name="ABLITERATED NVFP4", size=None,
    m={"bench:MMLU (2,280 Q)":"81.93 %","harmbench_real_harm_compliance":"100 %"}, L=71, L_extra=[52,134])],
  baseline={"bench:MMLU (2,280 Q)":"82.11 %","L":71,"L_extra":[134]},
  flags=["Abliterated: weights modified, not a faithful quant [L39-43]",
         "MMLU subset of 40 questions per subject; per-subject deltas range from -12 to +15 points [L78-133]",
         "Base-model run precision/quant not stated for the MMLU baseline [L69]",
         "Table formatting: Overall row is appended to the 'world religions' row [L134]",
         "No checkpoint size stated"])

repo(repo="dzannotti/Qwen3.8-Flash-Next-MTP-GGUF", sha="0b2551d191548168d3254ddea4ab943a5ef4f809", fmt="GGUF (MTP draft head)",
  runtime="patched llama.cpp (PR 27742 + draft-head patch) / LaurentZuijdwijk fork", weights_modified=False,
  method="MTP head exported from BF16 via convert_hf_to_gguf --mtp, quantized Q4_K_M with Q4_K output/token embeddings",
  methodology=None,
  variants=[dict(name="MTP-Q4_K_M", size=2.5, unit="GB", basis="disk", L=35),
            dict(name="MTP-BF16", size=7.8, unit="GB", basis="disk", L=36)],
  flags=["No quality claims for the quant (card read in full): draft head only; 'a Q8_0 head measured worse' refers to draft agreement, and tables report speed/acceptance [L35, L91-97]"])

repo(repo="gabmiranda/Qwen3.8-Flash-Next-oQ2.7e-mtp", sha="79f0ee184d8902255fffd9a319997442c2ba42b1", fmt="MLX",
  runtime="not stated", weights_modified=False, method="oQ2.7e (per title/tags)",
  methodology=None,
  variants=[dict(name="oQ2.7e", size=None)],
  flags=["No quality claims for the quant (card read in full): card is front matter plus a title only [L1-11]"])

repo(repo="gabmiranda/Qwen3.8-Flash-Next-oQ5e-mtp", sha="cf2679f5dac679e7d2b25280c343dfff98a68883", fmt="MLX",
  runtime="not stated", weights_modified=False, method="oQ5e (per title/tags)",
  methodology=None,
  variants=[dict(name="oQ5e", size=None)],
  flags=["No quality claims for the quant (card read in full): card is front matter plus a title only [L1-11]"])

repo(repo="groxaxo/Qwen3.8-Flash-Next-Q2_0-H64-S4-GGUF", sha="5eea3d52aecc2ffc0b5599212e71062dc55fa805", fmt="GGUF (Hadamard-rotated, patched runtime)",
  runtime="patched llama.cpp (runtime/ in repo)", weights_modified=False,
  method="Routed experts Q2_0 ternary g64 FP16 scales after Walsh-Hadamard blocks of 64 (seed 4); head/attention/shared experts Q8_0; n-gram Q4_0; no MTP or vision",
  methodology=dict(reference="none for prompts; weight reconstruction compared to direct Q2 on 26 weight samples",
    dataset="27-prompt diagnostic set (thinking off) + retry of 6 misses with thinking on", ctx=65536,
    tool="3x RTX 3090, per-layer token table on CPU, MTP disabled", L=[36,38,42]),
  variants=[dict(name="Q2_0 H64-S4", size=68.9, unit="GB", basis="disk",
    m={"smoke_prompts_thinking_off":"21 of 27","retry_thinking_on":"all six passed","recon_error_vs_direct_Q2":"2.91% lower"}, L=38, L_extra=[23,39,42])],
  flags=["Card calls the prompts 'a useful smoke check, not a broad model benchmark' [L40]",
         "Reconstruction-error figure is from the 26 samples used to choose the setting; card says it does not prove a general accuracy gain [L42]",
         "ctx 65,536 is the configured allocation, not a checked quality window [L36]",
         "No BF16 or KLD/PPL reference"])

repo(repo="him0413/Qwen3.8-Flash-Next-Uncensored-Q4_K_M-MTP", sha="54fc5130b3f6b4555ba9d70a27fd74803b7b1ca3", fmt="GGUF",
  runtime="llama.cpp with qwen4exp support", weights_modified=True,
  method="Q4_K_M of orcarouter/Qwen3.8-Flash-Next-Uncensored (abliterated) with Q4_K_M MTP head merged as blk.48",
  methodology=None,
  variants=[dict(name="Q4_K_M + MTP (uncensored)", size=113.7, unit="GiB", basis="disk", L=43)],
  flags=["No quality claims for the quant (card read in full): only model details and usage [L37-67]",
         "Abliterated source weights [L40]"])

repo(repo="igorls/Qwen3.8-Flash-Next-mixed-NInfer", sha="ef61b3c856f91e58a58e41a5a6269b6c8648d266", fmt="NInfer container (NVFP4/FP8 + INT4 PLE)",
  runtime="NInfer workstation fork", weights_modified=False,
  method="Format conversion and splice of primitive-ai mixed NVFP4/FP8 compute weights with primitive-ai INT4 g16 PLE table; MTP BF16 on disk (v2) / NVFP4 (v3)",
  methodology=None,
  variants=[dict(name="mixed NInfer v2", size=113.30, unit="GB", basis="disk", L=57),
            dict(name="mixed NInfer v3", size=109681106432, unit="bytes", basis="disk", L=221)],
  flags=["No quality claims for the quant (card read in full): card states it 'adds no benchmark campaign or quality score' [L207-208]",
         "'8 of the 10 prompts match the bf16-KV run exactly' compares KV-cache dtype, not quantization, and is labelled a smoke test [L289]",
         "Weights are primitive-ai's quantizations, repackaged [L36-39]"])

repo(repo="intel-ai/Qwen3.8-Flash-Next-MXFP4-BF16Attn-CT-AutoRound", sha="2855a0097727c34007796deda1985ce5ca9ef0a1", fmt="compressed-tensors (MXFP4)",
  runtime="vLLM >= 0.29.0 (TP=1)", weights_modified=False,
  method="AutoRound 0.16.0 model-free RTN: routed experts MXFP4 W4A4 g32 UE8M0, dynamic activations; everything else BF16 incl. attention and PLE",
  methodology=dict(reference="Qwen/Qwen3.8-Flash-Next bf16 baseline (same table)", dataset="GSM8K (chat template, few-shot multiturn), MMLU, PIQA, HellaSwag",
    ctx=8192, tool="lm-eval 0.4.13 + vLLM 0.29.1rc1.dev528, TP=1, bf16 KV, seed=42, enable_thinking=false", L=[82,83,84]),
  table_header_L=86,
  variants=[dict(name="MXFP4 experts, BF16 attention", size=182.6, unit="GB", basis="disk",
    m={"bench:GSM8K strict":"0.9644","bench:GSM8K flexible":"0.9636","bench:MMLU":"0.8641","bench:PIQA acc":"0.8248","bench:PIQA acc_norm":"0.8308",
       "bench:HellaSwag acc":"0.6856","bench:HellaSwag acc_norm":"0.8726"}, L=89, L_extra=[25])],
  baseline={"bench:GSM8K strict":"0.9674","bench:MMLU":"0.8652","bench:PIQA acc":"0.8194","bench:HellaSwag acc":"0.6927","L":88},
  cross=[dict(of="nvidia/Qwen3.8-Flash-Next-NVFP4", variant="NVFP4", size=132.7, measured=False, unit="GB", basis="disk", L=106, L_extra=[114], flags=["quoted, not measured"])],
  flags=["Baseline row lacks GSM8K flexible, PIQA acc_norm and HellaSwag acc_norm ('—') [L88]",
         "Card does not state whether the bf16 baseline was run in the same harness",
         "Saturated/easy benchmarks only, thinking disabled, max_model_len 8192 [L82-84]",
         "NVIDIA comparison is size/format only, no quality numbers [L103-114]"])

repo(repo="jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4", sha="f3f020e2545cbeea957a49b16cac2f86e61c3dd8", fmt="GGUF (custom CIRU IU4 format)",
  runtime="CIRU runtime (custom llama.cpp/ggml fork) on Strix Halo", weights_modified=False,
  method="Routed experts 4.5175-bpw compact indexed affine (4-bit codes + scale/offset indices); rest mixed higher precision; PLE table SSD-paged",
  methodology=dict(reference="BF16 full-vocabulary reference", dataset="2,048 next-token positions (data source not stated); task checks: 15-case hard tool set, HermesAgent-20 (two passes), thinking off",
    ctx=None, tool="each stack's own runtime; Ryzen AI Max+ 395 / Radeon 8060S", L=[178,179,180,181,182]),
  table_header_L=184,
  variants=[dict(name="CIRU v5.0", size=72112333312, unit="bytes", basis="disk", mem=87.20, mem_unit="GiB", mem_basis="resident",
    m={"kl":"0.050030","top1_pct":"94.482%","tail_ppl":"2.04505","hard_tools_per100":"80","hermes_pass1_per100":"93","hermes_pass2_per100":"96"},
    L=187, L_extra=[201,218])],
  ladder=[
    dict(name="CIRU v4.4.1 (retained)", size=79397818720, unit="bytes", basis="disk", mem=90.60, mem_unit="GiB", mem_basis="resident",
      m={"kl":"0.056117","top1_pct":"93.750%","tail_ppl":"2.05516","hard_tools_per100":"67","hermes_pass1_per100":"89","hermes_pass2_per100":"88"},
      L=186, L_extra=[200,217]),
    dict(name="v4 (historical, earlier runtime/settings)",
      m={"bench:HumanEval":"160/164","bench:HumanEval+":"155/164","bench:ARC-Challenge":"1,143/1,172","bench:ToolEval Standard":"115/138","bench:ToolEval Hard":"23/30"},
      L=276, L_extra=[277,278]),
  ],
  cross=[
    dict(of="HaloBox (stack; repo not named on card)", variant="HaloBox", mem=93.40, mem_unit="GiB", mem_basis="resident",
      m={"kl":"0.029182","top1_pct":"96.143%","tail_ppl":"2.03642","hard_tools_per100":"63","hermes_pass1_per100":"90","hermes_pass2_per100":"96"}, L=188, L_extra=[202]),
    dict(of="Gufo (stack; repo not named on card)", variant="Gufo", mem=88.54, mem_unit="GiB", mem_basis="resident",
      m={"kl":"0.027966","top1_pct":"96.436%","tail_ppl":"2.03502","hard_tools_per100":"73","hermes_pass1_per100":"95","hermes_pass2_per100":"95"}, L=189, L_extra=[203]),
    dict(of="agentionai/Qwen3.8-Flash-Next-AP-GGUF", variant="AP-Q5_K_XL", mem=99.64, mem_unit="GiB", mem_basis="resident",
      m={"kl":"0.029135","top1_pct":"96.436%","tail_ppl":"2.02636","hard_tools_per100":"73","hermes_pass1_per100":"91","hermes_pass2_per100":"91"}, L=190, L_extra=[152,204]),
    dict(of="Halogen (stack; repo not named on card)", variant="Halogen", mem=34.47, mem_unit="GiB", mem_basis="resident",
      m={"kl":"0.085905","top1_pct":"92.773%","tail_ppl":"2.07968","hard_tools_per100":"70","hermes_pass1_per100":"93","hermes_pass2_per100":"92"}, L=191, L_extra=[205]),
  ],
  flags=["Small objective panel: 2,048 positions; card says it does not establish long-context factual accuracy [L178, L196]",
         "Rows compare complete weight+runtime stacks, not quants in isolation [L181-182]",
         "Full quality panel and task suite were not rerun on the final v5.5.0 release build [L171-174]",
         "In the card's own table v5.0 KL (0.050030) is higher than HaloBox, Gufo and AgentionAI AP [L186-191]",
         "Halogen memory (34.47 GiB) is not comparable; card notes 78.16 GiB process PSS [L191, L193]",
         "HumanEval 0-9 used for speed, not pass@1 [L207-208]",
         "Historical v4 benchmark results come from earlier runtime/settings and are not comparable [L275-282]"])

repo(repo="kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF", sha="72805b661c4af3ef63f0ed11e6dafcfdaba33982", fmt="GGUF (MTP draft heads)",
  runtime="kingjones30/ROCmFPX fork + mtp-graph patch", weights_modified=False,
  method="Extracted MTP block converted and quantised (BF16, ROCmFP4 FAST/STRIX, Q4_K_M, Q6_K, Q8_0)",
  methodology=None,
  variants=[dict(name="mtp BF16", size=7770801344, unit="bytes", basis="disk", L=92),
            dict(name="mtp Q4_0_ROCMFP4_FAST", size=2083139232, unit="bytes", basis="disk", L=93),
            dict(name="mtp Q4_0_ROCMFP4_STRIX", size=2266977952, unit="bytes", basis="disk", L=94),
            dict(name="mtp Q4_K_M", size=2786245792, unit="bytes", basis="disk", L=95),
            dict(name="mtp Q6_K", size=3404798752, unit="bytes", basis="disk", L=96),
            dict(name="mtp Q8_0", size=4135934112, unit="bytes", basis="disk", L=97)],
  flags=["No quality claims for the quant (card read in full): draft heads; only speed and acceptance figures [L39-48]",
         "Card contains both a section saying the heads now work (+27.7%) and an older section saying 'These files do not load' [L23-31, L60-72]"])

repo(repo="kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-GGUF", sha="a4d9d7bfcbac90fda3342587a3e13d1660d57a20", fmt="GGUF (ROCmFP4, fork-only)",
  runtime="kingjones30/ROCmFPX fork (qwen4exp + ROCmFP4)", weights_modified=True,
  method="Q4_0_ROCMFP4_STRIX_LEAN recipe from orcarouter/Qwen3.8-Flash-Next-Uncensored BF16 (abliterated); PLE Q5_1, token_embd Q5_K, output Q6_K",
  methodology=dict(reference="publisher's aligned (non-abliterated) build of the same tier, not BF16",
    dataset="own prompt splits: 24 harmful, 12 harmless, 8 quality; prompts and completions not published",
    ctx=None, tool="same harness, greedy", L=[206,215]),
  variants=[dict(name="Q4_0_ROCMFP4_STRIX_LEAN (uncensored)", row_label="this build", size=98.49, unit="GiB", basis="disk", bpw=4.78, mem=63.3, mem_unit="GiB", mem_basis="vram",
    m={"refusal:Harmful (24)":"22 comply","Harmless (12)":"11 ok","quality:Quality (8)":"6/8"}, L=129, L_extra=[180,210,211,212])],
  ladder=[dict(name="aligned build (same tier)", m={"refusal:Harmful (24)":"0 comply","Harmless (12)":"10 ok","quality:Quality (8)":"6/8"}, L=210, L_extra=[211,212])],
  flags=["Weights are abliterated (orcarouter), not a faithful quant of Qwen's model [L123-125]",
         "Quality evidence is an 8-item set with prompts not published [L212, L215]",
         "Refusal/quality table and long-context table are identical to the card of Kashapps/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-GGUF (a different tier) [L132-136 there, L194-198 here]",
         "MTP speed figures were measured on other builds (FAST imatrix / STRIX_LEAN-imatrix weights) [L56-60, L76-78]"])

repo(repo="matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i", sha="d32978ccdde1729caec0034571b03a14779698e9", fmt="ModelSlim INT4 W4 g128 (Ascend 310P)",
  runtime="OpenSensor vLLM + vLLM Ascend forks with custom 310P operators", weights_modified=False,
  method="ModelSlim asymmetric per-group min/max RTN INT4 g128 on 73,728 routed-expert projections from BF16; rest FP16/FP32 (incl. PLE)",
  methodology=dict(reference="quality gate: none stated; GPQA: matched llama.cpp UD-IQ4_XS GGUF on one RTX 6000 Pro (not BF16)",
    dataset="fixed 228-question zero-shot gate; GPQA Diamond 198 Q, AISBench zero-shot CoT prompts (dataset version b1ed2c), temp 0, seed 1024, top-k 20, top-p 1, 8,192-token output ceiling",
    ctx=None, tool="native W4A8 service, 2x Atlas 300I Duo, TP4/EP4, MTP2", L=[94,95,176,177,178,179,180,181,182]),
  table_header_L=185,
  variants=[dict(name="W4 G128 (native W4A8 path)", size=181637185528, unit="bytes", basis="disk",
    m={"quality_gate_228q_zero_shot":"206/228","bench:GPQA Diamond (AISBench Answer: X)":"140/198 (70.71%)",
       "gpqa_final_response_correct":"114/198 (57.58%)","gpqa_hit_output_limit":"81/198"},
    L=94, L_extra=[95,66,187,189,190])],
  cross=[dict(of="UD-IQ4_XS GGUF on llama.cpp (publisher not named on card)", variant="UD-IQ4_XS",
    m={"bench:GPQA Diamond (AISBench Answer: X)":"140/198 (70.71%)","gpqa_final_response_correct":"120/198 (60.61%)","gpqa_hit_output_limit":"74/198"},
    L=187, L_extra=[189,190])],
  ladder=[dict(name="opt-in dynamic-W8A8 PLE projection (runtime option, one-token shape)", m={"rel_l2_err":"0.58%","cos":"0.999983"}, L=152, L_extra=[153])],
  flags=["Card calls the GPQA run a serving-stack comparison, not an isolated quantization test; chat templates and formats differ [L182-183]",
         "Benchmark extractor credits the last 'Answer: X' in unfinished reasoning: 26 Ascend / 20 RTX cases credited without a final answer [L193-197]",
         "81/198 Ascend responses hit the 8,192-token output limit [L190]",
         "228-question gate is a fixed sampled gate, not official full-dataset MMLU [L95-96, L426-427]",
         "GPQA Ascend cases were resumed across server changes [L208-210]",
         "No BF16 reference; quality gate not rerun after later runtime changes [L128-129, L259-260]"])

repo(repo="mduca771/Qwen3.8-Flash-Next-NG8-GGUF", sha="c516e96b708f6c6dbdee78b05b79ae34906a89b1", fmt="GGUF",
  runtime="llama.cpp > b10660", weights_modified=False,
  method="unsloth UD-IQ3_XXS / UD-Q3_K_XL backbone tensors unchanged, per_layer_token_embd replaced with the tensor from the official Q8_0 build",
  methodology=None,
  variants=[dict(name="UD-IQ3_XXS-NG8", size=107.56, unit="GB", basis="disk", L=50),
            dict(name="UD-Q3_K_XL-NG8", size=115.59, unit="GB", basis="disk", L=51)],
  flags=["No quality claims for the quant (card read in full): states PLE compression has 'direct and perceptible' quality impact and NG8 gives near-Q8_0 embedding quality, without measurements [L25, L33]; validation is tensor/byte integrity plus smoke test [L39-44]"])

repo(repo="mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp", sha="e171af86f499f1855b0fb71d781105e8dd609610", fmt="MLX",
  runtime="oMLX 0.6.4", weights_modified=False, method="oQ6e (imatrix), group 64, MTP preserved",
  methodology=None,
  variants=[dict(name="oQ6e", size=None)],
  flags=["No quality claims for the quant (card read in full): benchmark tables are speed only (write/read tok/s, tokens written) vs oQ4e and oQ8e [L28-62]"])

repo(repo="myllmbox/Qwen3.8-Flash-Next-hibrid47", sha="7b83fa0d4e556c238e49b2be346d655963b2e978", fmt="NVFP4 mixed (vLLM)",
  runtime="vLLM with myllmbox PLE patch, 2x DGX Spark TP=2", weights_modified=False,
  method="Routed experts NVFP4 W4A4 from Inferact; GDN projections NVFP4 W4A16; attention/shared/lm_head bf16; PLE n-gram table NVFP4 (28.6 GiB)",
  methodology=dict(reference="none (comparison vs own hibrid46 with int3 table)", dataset="32-scene 'boss-animals' render gauntlet, thinking on; scoring criteria not stated",
    ctx=None, tool="vLLM, 2x DGX Spark GB10, MTP K=4, temperature 1.0", L=[28,30,31,45]),
  variants=[dict(name="hibrid47 (NVFP4 PLE table)", size=None,
    m={"render_gauntlet_good":"26 good","render_gauntlet_partial":"3 partial","render_gauntlet_broken":"3 broken"}, L=30, L_extra=[18])],
  ladder=[dict(name="hibrid46 (int3 PLE table)", m={"render_gauntlet":"about half/half"}, L=31)],
  flags=["Custom render test; pass/partial/broken criteria not stated [L30]",
         "Ladder score is qualitative ('about half/half') [L31]",
         "No BF16 reference; no total checkpoint size stated (PLE table 28.6 GiB) [L43]",
         "Routed-expert tensors derived from Inferact/Qwen3.8-Flash-Next-NVFP4 [L38, L102]"])

repo(repo="nmatteo3294/Qwen3.8-Flash-Next-Q3_PLE-MTP-GGUF", sha="665025060afabe597e0c5ab6770478bd81f509f6", fmt="GGUF (private Q3_PLE type, patched llama.cpp)",
  runtime="patched llama.cpp (cafe-llama.cpp + 36 patches)", weights_modified=False,
  method="AtomicChat AD-4.27bpw Q4_K_M-M64 target with only per_layer_token_embd re-quantized Q5_1 -> private Q3_PLE; detached learned MTP sidecar",
  methodology=None,
  variants=[dict(name="AD-4.27bpw-Q3_PLE-M64 target", size=78525318176, unit="bytes", basis="disk", L=36),
            dict(name="MTP sidecar DOWNQ4-FC-HC-OUTQ4", size=2202883264, unit="bytes", basis="disk", L=38)],
  flags=["No quality claims for the quant (card read in full): 'exact output'/parity checks compare MTP vs target-only decoding, not quantization quality [L105-111, L138-141]",
         "Negative results reported: code parity failed at 59,750 tokens; prose fixture returned 90 words for 170-210 requested [L145-150]",
         "Other 1,223 tensors byte-identical to AtomicChat source [L72-73]"])

repo(repo="orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX", sha="fb8c6ba14deff7998bc28348d247e0cefcb48be4", fmt="MLX (per repo name)",
  runtime="not stated", weights_modified=True, method="not stated (card empty)",
  methodology=None,
  variants=[dict(name="Uncensored MLX", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes)",
         "weights_modified set from the repo name ('Uncensored') and orcarouter's abliterated BF16 release referenced by other cards; card itself states nothing"])

repo(repo="peasantsmith/Qwen3.8-Flash-Next-PS-IQ2_XXS-GGUF", sha="c8122ffc62ae9befdf8b6017119e186bcf002a20", fmt="GGUF",
  runtime="llama.cpp with qwen4exp", weights_modified=False,
  method="IQ2_XXS expert gate/up, Q4_0 expert down, PLE IQ4_NL, attention Q4_K, output Q8_0; unsloth imatrix",
  methodology=None,
  variants=[dict(name="PS-IQ2_XXS", size=75.2, unit="GB", basis="disk", bpw=3.40, L=19, L_extra=[62])],
  flags=["No quality claims for the quant (card read in full): only single-prompt pass/fail checks (tool call, needle, smoke) [L45-53]; perplexity explicitly not measured [L54-56]"])

repo(repo="robertbschuette/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF", sha="24b9683263b0086f033de57ce6ff271538114447", fmt="GGUF",
  runtime="llama.cpp / Strata", weights_modified=True,
  method="Abliterated (huihui-ai method); GGUF series stated to come from unsloth and ukisai Swift-1.5 GSQ-RCO repos",
  methodology=None,
  variants=[dict(name="abliterated GGUF series (unsloth / Swift)", size=None)],
  flags=["No quality claims for the quant (card read in full): no sizes, KLD, PPL or benchmarks",
         "Card heading names huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF [L27]; GGUFs described as coming from unsloth and ukisai repos [L35, L52, L59]",
         "Card suggests disabling Strata's SHA256 verification for the unsloth series [L39-45]"])

repo(repo="scottlowry/Qwen3.8-Flash-Next-oQ5e-fp16-mtp", sha="f876fb266903db70b4f753c167eeeaa5be24bda6", fmt="MLX",
  runtime="oMLX v0.7.1.dev1", weights_modified=False, method="oQ5e, group 64",
  methodology=None,
  variants=[dict(name="oQ5e-fp16", size=None)],
  flags=["No quality claims for the quant (card read in full): only quantization details [L13-20]"])

repo(repo="scottlowry/Qwen3.8-Flash-Next-oQ8e-mtp", sha="4bf7525a860ada67c4d495dacccc5cca15092076", fmt="MLX",
  runtime="oMLX v0.7.1.dev1", weights_modified=False, method="oQ8e, group 64",
  methodology=None,
  variants=[dict(name="oQ8e", size=None)],
  flags=["No quality claims for the quant (card read in full): only quantization details [L13-20]"])

repo(repo="spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF", sha="cd72aeb4f07abb0e672dd07ce92fdcd7f40cd2bc", fmt="GGUF",
  runtime="llama.cpp (master)", weights_modified=True,
  method="Static GGUF quants (Q2_K..Q8_0) of cyberneurova/cyberneurova-Qwen3.8-Flash-Next",
  methodology=None,
  variants=[dict(name="Q2_K", size=80.4, unit="GB", basis="disk", L=38),
            dict(name="Q3_K_S", size=88.8, unit="GB", basis="disk", L=39),
            dict(name="Q3_K_M", size=94.2, unit="GB", basis="disk", L=40),
            dict(name="Q3_K_L", size=96.6, unit="GB", basis="disk", L=41),
            dict(name="Q4_K_S", size=111.8, unit="GB", basis="disk", L=42),
            dict(name="Q4_K_M", size=119.2, unit="GB", basis="disk", L=43),
            dict(name="Q5_K_S", size=127.8, unit="GB", basis="disk", L=44),
            dict(name="Q5_K_M", size=134.1, unit="GB", basis="disk", L=45),
            dict(name="Q6_K", size=167.6, unit="GB", basis="disk", L=46),
            dict(name="Q8_0", size=188.2, unit="GB", basis="disk", L=47)],
  flags=["No quality claims for the quant (card read in full): 'lower quality' / 'very good quality' notes are generic labels without measurements [L40-47]; CaliperBench scores are linked, not shown [L31-32]",
         "Source is cyberneurova's derivative model, not the Qwen base; nature of its modification not stated on this card [L17, L29]",
         "Front matter license 'gemma' [L4]"])

repo(repo="thoddnn/Qwen3.8-Flash-Next-4bit", sha="b070255b2c42e71b9fa564beb667f79aedd474d8", fmt="MLX",
  runtime="mlx-vlm main (post-#2032)", weights_modified=False, method="4-bit MLX, group size 32 (mlx-vlm d1bd74ed)",
  methodology=None,
  variants=[dict(name="MLX 4-bit g32", size=None)],
  flags=["No quality claims for the quant (card read in full): verification covers norm-gain conversion correctness and coherent generation only [L37-45]"])

repo(repo="trymirai/Qwen3.8-Flash-Next-S-experimental", sha="af20c488b259d33acf1d53e2609410efaddbfd9e", fmt="not stated",
  runtime="not stated", weights_modified=False, method="not stated (card empty)",
  methodology=None,
  variants=[dict(name="S-experimental", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes)"])

repo(repo="vcruz305/Qwen3.8-Flash-Next-GGUF", sha="0c50dde2735c6a30caa2a8f9160310bef3ff2a68", fmt="GGUF",
  runtime="qwen4exp-capable llama.cpp (PR #27742 class)", weights_modified=False,
  method="K-quant ladder Q2_K..Q6_K from BF16 with MTP in-main; shared BF16 PLE; AtomicChat-compatible imatrix; tok emb Q8_0, output Q6_K",
  methodology=None,
  variants=[dict(name="Q2_K", size=49.5, unit="GiB", unit_L=34, basis="trunk_only", L=38),
            dict(name="Q3_K_M", size=62.5, unit="GiB", unit_L=34, basis="trunk_only", L=39),
            dict(name="Q4_K_M", size=80.2, unit="GiB", unit_L=34, basis="trunk_only", L=40),
            dict(name="Q5_K_M", size=91.3, unit="GiB", unit_L=34, basis="trunk_only", L=41),
            dict(name="Q6_K", size=107.8, unit="GiB", unit_L=34, basis="trunk_only", L=42)],
  flags=["No quality claims for the quant (card read in full): only approximate backbone sizes; shared BF16 PLE ~95.4 GiB on top [L34-43]"])

repo(repo="xpashkin/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw", sha="fb2a8be9d8053c7a6ecb12eab7f7e0120859a61d", fmt="MLX (mlx-serve pack)",
  runtime="mlx-serve 26.10.2 / MLX 0.32.3", weights_modified=False,
  method="Imatrix-weighted affine (unsloth imatrix remapped): experts 6-bit g64, non-expert 8-bit, embed 4-bit, n-gram 8-bit g32, MTP 6-bit plain",
  methodology=None,
  variants=[dict(name="iQ-MLX 6.6 bpw", size=153, unit="GiB", basis="disk", L=36)],
  flags=["No quality claims for the quant (card read in full): only imatrix correlation checks and allocation error-mass analysis, no output-level quality numbers [L173-177, L193-197]; performance table is speed only [L113-122]",
         "Total stated as ≈153 GiB = index 105.2 GB + ngram 57.6 GB (mixed units) [L36]"])

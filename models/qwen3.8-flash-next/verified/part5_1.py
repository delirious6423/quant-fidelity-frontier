# Re-extracted from verbatim README snapshots at pinned commits (snapshots/index.json -> cache/cards/, captured 2026-10-17).
# Queue: cache/autoq_1.json (previously regex-triaged only). Every card read in full.
# Each variant/claim cites README line numbers (L) at that sha.
# size basis: disk = full download unless noted; vram/resident/ram_in_use = memory figures; trunk_only = weights excluding n-gram table.
R = []
def repo(**k): R.append(k); return k

repo(repo="Ar3ntiz/qwen3.8_flash_next_side8", sha="ba49b297058050c03a6037e210995589a64e4911", fmt="SUC1",
  runtime="SUC C++ / Metal runtime (suc1-generate, suc1-server) or SUC Studio; Apple Silicon only", weights_modified=False,
  method="SUC (Semantic Universal Compression): routed experts 2-D vector codebook per tensor (VQ, G=2, K=256) with one-sided randomized Hadamard rotation and 0.5% activation-selected INT8 column residuals; attention/linear attention/shared expert/PLE k-v/LM head affine 8-bit g64; token embedding BF16; n-gram tables row-wise affine 4-bit; text-only (no vision, no MTP)",
  methodology=dict(reference="the original model", dataset="16 × 512 held-out positions (KL, top-1); WikiText-2 test (PPL); answer-level 120 graded questions (20 per subset), greedy", ctx=512, tool="not stated (SUC runtime implied); single seed, Mac Studio M1 Ultra", L=[45,59,112]),
  table_header_L=47,
  variants=[dict(name="SUC1 side8", size=96.32, unit="GB", basis="disk", bpw=4.35, mem=73.82, mem_unit="GB", mem_basis="resident",
    m={"mean_kld":"0.064","top1":"0.913","ppl":"4.704","bench:Total":"97/120","bench:GSM8K":"17","bench:MATH":"15","bench:MMLU-Pro":"6","bench:ARC-C":"20","bench:BBH":"19","bench:Long context":"20"},
    L=49, L_extra=[27,28,55,63])],
  baseline={"ppl":"4.695","L":49},
  cross=[
    dict(of="GGUF Q4 (ds4) — publisher/repo not identified on card", variant="GGUF Q4 (ds4)", size=177.28, unit="GB", basis="disk",
      m={"mean_kld":"0.061","bench:Total":"99/120","bench:GSM8K":"18","bench:MATH":"16","bench:MMLU-Pro":"7","bench:ARC-C":"19","bench:BBH":"19","bench:Long context":"20"}, L=56, L_extra=[64]),
    dict(of="MLX affine 4-bit — publisher/repo not identified on card", variant="MLX affine 4-bit", m={"mean_kld":"0.202"}, L=57),
  ],
  flags=["KL eval over 16 × 512 held-out positions only; answer-level suite is 120 questions (20 per subset); card states single-seed measurements on one machine [L45, L59, L112]",
         "Card's own comparison shows GGUF Q4 (ds4) with lower KL (0.061 vs 0.064) and higher answer total (99/120 vs 97/120) [L55-56, L63-64]",
         "Comparison formats are not identified by repo; MLX affine 4-bit has no size [L56-57]",
         "4.35 bits per parameter stated to include every overhead; 73.82 GB fully resident at run time [L27-28]",
         "Core suite of 30 graded smoke-test questions: 30/30 [L66]"])

repo(repo="Beinsezii/Qwen3.8-Flash-Next-GGUF-HALO", sha="e1c7270252f58ad852f1cf3333b7a8ed60caf6f7", fmt="GGUF",
  runtime="llama.cpp (Strix Halo 128GiB target)", weights_modified=False,
  method="Custom tensor-type recipe (in tensor_types.txt, not on card) optimized for quality/speed on Strix Halo; n-gram table expected to be mmapped from NVMe",
  methodology=None,
  variants=[dict(name="HALO GGUF", size=None)],
  flags=["No quality claims for the quant (card read in full): 11-line card with only the recipe pointer and a note that it is larger than resident memory [L7-11]",
         "No size stated on card; no MTP yet [L7]"])

repo(repo="Delgon/Qwen3.8-Flash-Next-PLE-GGUF", sha="5de86c89b5b78fc2711744475b4da50ddd9acb72", fmt="GGUF (PLE sidecar tables only)",
  runtime="llama.cpp build with PLE sidecar support (--ple); not mainline, LM Studio or Ollama", weights_modified=False,
  method="Standalone per_layer_token_embd (n-gram PLE) table in three precisions: BF16 byte copy of unsloth's BF16 GGUF table; Q8_0 converted from BF16; IQ4_NL byte copy of unsloth UD-Q4_K_XL table. Not a runnable model.",
  methodology=dict(reference="same model (main weights UD-Q4_K_XL) with the BF16 table", dataset="held-out corpus written after the model's training cutoff; wikitext-2 as memorized-text control; 150 chunks", ctx=512, tool="llama-perplexity (KL divergence, top-1); Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, 128 GB)", L=[65,66,67,68]),
  table_header_L=70,
  variants=[
    dict(name="PLE-Q8_0", size=54.4, unit="GB", basis="disk",
      m={"ppl_ratio:held-out":"1.0019 +- 0.0013","mean_kld:held-out":"0.0162 +- 0.0004","kld99:held-out":"0.217","top1_pct:held-out":"96.20 +- 0.10 %",
         "ppl_ratio:wikitext-2":"0.9987 +- 0.0010","mean_kld:wikitext-2":"0.0122 +- 0.0002","kld99:wikitext-2":"0.150","top1_pct:wikitext-2":"96.44 +- 0.10 %",
         "row_rel_rms_vs_bf16":"0.0053"}, L=72, L_extra=[74,38]),
    dict(name="PLE-IQ4_NL", size=28.8, unit="GB", basis="disk",
      m={"ppl_ratio:held-out":"1.0022 +- 0.0016","mean_kld:held-out":"0.0255 +- 0.0005","kld99:held-out":"0.352","top1_pct:held-out":"95.02 +- 0.11 %",
         "ppl_ratio:wikitext-2":"1.0044 +- 0.0014","mean_kld:wikitext-2":"0.0239 +- 0.0004","kld99:wikitext-2":"0.289","top1_pct:wikitext-2":"94.71 +- 0.11 %",
         "row_rel_rms_vs_bf16":"0.076"}, L=73, L_extra=[75,39]),
    dict(name="PLE-BF16 (reference)", size=102.4, unit="GB", basis="disk", L=37),
  ],
  flags=["Files are only the n-gram PLE table, not a runnable model; quality figures measure swapping this table into unsloth UD-Q4_K_XL, relative to the same model with the BF16 table [L21-23, L65-67]",
         "IQ4_NL table is a byte copy of unsloth's UD-Q4_K_XL table; BF16 is a byte copy of unsloth's BF16 GGUF table [L37, L39]",
         "Q8_0 table stated identical to AesSedai's Q8_0 table on 20,000 sampled rows [L38]",
         "Card reports no measurable speed or MTP-acceptance difference between tables [L85-94]"])

repo(repo="EigenLabs/Qwen3.8-Flash-Next-MLX-4bit-mtp", sha="e555958420096f695d449111693a6647b698479b", fmt="MLX",
  runtime="runtimes with native qwen4_exp support (Darkbloom d-inference / mlx-swift-lm draft PRs)", weights_modified=False,
  method="MLX affine, predominantly 4-bit/group-64 with selected 5-, 6- and 8-bit modules; output projection 8-bit; vision BF16; embedded MTP; n-gram tables retained",
  methodology=None,
  variants=[dict(name="MLX 4-bit mtp", size=106294664646, unit="bytes", basis="disk", L=36)],
  flags=["No quality claims for the quant (card read in full): only compatibility pass counts (59 standard cases, 24 tool-history cases) and integrity checks; card states these do not establish BF16 equivalence [L102-107, L115]",
         "Safetensors 106,294,664,646 bytes (~98.99 GiB) across 21 shards [L36]",
         "Card lists known remaining failures (adversarial literal-string copying, some reasoning/tool cases, brief-video recognition) [L109-111]"])

repo(repo="Jundot/Qwen3.8-Flash-Next-oQ2.5e-fp16-mtp", sha="9eed92cf57591e476a60e7f6f03139de251f4472", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ2.5e (oMLX v0.7.1): sensitivity-based bit allocation + imatrix (1024x512) + weighted LSQ scale/bias refit; FP16 scales/unquantized tensors, vision FP32; n-gram table 4-bit outside budget",
  methodology=None,
  variants=[dict(name="oQ2.5e-fp16", size=83.796, unit="GB", basis="disk", bpw=3.724, L=53, L_extra=[25,43])],
  flags=["No quality claims for the quant (card read in full): only bit distribution and storage tables [L31-53]",
         "3.217 bpw excluding n-gram table, 3.724 for full checkpoint; total 78.041 GiB [L25, L43, L53]"])

repo(repo="Jundot/Qwen3.8-Flash-Next-oQ3e-fp16-mtp", sha="14f743ec606eab80ab3f24f4fbcc81f97086b189", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ3e (oMLX v0.7.1): sensitivity-based bit allocation + imatrix (1024x512) + weighted LSQ scale/bias refit; FP16 scales/unquantized tensors, vision FP32; n-gram table 4-bit outside budget",
  methodology=None,
  variants=[dict(name="oQ3e-fp16", size=92.099, unit="GB", basis="disk", bpw=4.093, L=55, L_extra=[25,45])],
  flags=["No quality claims for the quant (card read in full): only bit distribution and storage tables [L31-55]",
         "3.733 bpw excluding n-gram table, 4.093 for full checkpoint; total 85.774 GiB [L25, L45, L55]"])

repo(repo="Guile/Qwen3.8-Flash-Next-GGUF", sha="75862fae26e8e9bffc198ca8e0d187004cbceccb", fmt="GGUF",
  runtime="llama.cpp (b10665+)", weights_modified=False,
  method="llama.cpp imatrix quants (release b10665); calibration-v6 corpus rendered through the chat template (583 chunks of 512); *_L variants use Q8_0 embed/output",
  methodology=None,
  variants=[
    dict(name="bf16", size=354.03, unit="GB", basis="disk", L=44),
    dict(name="Q8_0", size=188.26, unit="GB", basis="disk", L=45),
    dict(name="Q6_K", size=168.11, unit="GB", basis="disk", L=46),
    dict(name="Q5_K_L", size=151.06, unit="GB", basis="disk", L=47),
    dict(name="Q4_K_L", size=139.27, unit="GB", basis="disk", L=48),
    dict(name="Q5_K_M", size=134.67, unit="GB", basis="disk", L=49),
    dict(name="Q5_K_S", size=128.10, unit="GB", basis="disk", L=50),
    dict(name="Q4_K_M", size=119.60, unit="GB", basis="disk", L=51),
    dict(name="Q3_K_XL", size=119.35, unit="GB", basis="disk", L=52),
    dict(name="Q4_K_S", size=113.14, unit="GB", basis="disk", L=53),
    dict(name="Q4_1", size=111.26, unit="GB", basis="disk", L=54),
    dict(name="Q2_K_L", size=107.15, unit="GB", basis="disk", L=55),
    dict(name="Q4_0", size=100.60, unit="GB", basis="disk", L=56),
    dict(name="IQ4_NL", size=100.29, unit="GB", basis="disk", L=57),
    dict(name="IQ4_XS", size=97.68, unit="GB", basis="disk", L=58),
    dict(name="IQ3_M", size=93.23, unit="GB", basis="disk", L=59),
    dict(name="Q3_K_L", size=93.19, unit="GB", basis="disk", L=60),
    dict(name="Q3_K_M", size=91.96, unit="GB", basis="disk", L=61),
    dict(name="IQ3_XS", size=91.95, unit="GB", basis="disk", L=62),
    dict(name="Q3_K_S", size=89.44, unit="GB", basis="disk", L=63),
    dict(name="IQ3_XXS", size=88.02, unit="GB", basis="disk", L=64),
    dict(name="Q2_K", size=80.93, unit="GB", basis="disk", L=65),
    dict(name="IQ2_M", size=80.37, unit="GB", basis="disk", L=66),
    dict(name="IQ2_S", size=77.83, unit="GB", basis="disk", L=67),
    dict(name="IQ2_XS", size=77.74, unit="GB", basis="disk", L=68),
    dict(name="IQ2_XXS", size=75.19, unit="GB", basis="disk", L=69),
    dict(name="IQ1_M", size=72.01, unit="GB", basis="disk", L=70),
    dict(name="IQ1_S", size=70.10, unit="GB", basis="disk", L=71),
  ],
  flags=["No quality claims for the quant (card read in full): only file sizes and qualitative per-file descriptions ('Good quality', 'Very low quality but surprisingly usable') with no measured numbers [L42-71]",
         "Card text is bartowski's card: front matter 'quantized_by: bartowski' and all download links/commands point to bartowski/Qwen3.8-Flash-Next-GGUF, not this repo [L2, L38, L44-71, L76]",
         "Speculative decoding listed as 'no' [L20]"])

repo(repo="Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX", sha="2f7b82f63cf39a935382d25a8e2e54e62b147e45", fmt="MLX (MTPLX layout)",
  runtime="MTPLX >= 2.11.2 only (use the MLX repo for oMLX)", weights_modified=True,
  method="Byte-exact MTPLX repack of Litwein/...-REAP320-oQ3e-DWQ-MTP-Vision-MLX (REAP 512->320 experts, oQ3e 3-bit routed experts + mixed trunk, end-to-end KL-DWQ v9) plus a new self-distilled 8-bit MTP sidecar; n-gram table separate 4-bit g32 file streamed from SSD",
  methodology=None,
  variants=[dict(name="REAP320-oQ3e-DWQ MTPLX", size=37, unit="GiB", basis="trunk_only", bpw=4.28, L=62, L_extra=[64,65])],
  flags=["No quality claims for the quant (card read in full): only speed/MTP A/B and memory figures; card says no task or vision-benchmark score is claimed and refers to the MLX repo for teacher-gap tables [L59, L91-95, L104-115, L152-153]",
         "Weights modified: REAP expert pruning 512->320 plus DWQ calibration [L58, L75]",
         "4.28 bpw over 133.7B params overall; 3.84 bpw / ≈37 GiB over the 82.5B weights kept in RAM (table streamed) [L62-65]"])

repo(repo="MESHIVEAI/Qwen3.8-Flash-Next-NVFP4-W4A16", sha="d58bb7aa3eb1d22462db83016904551b51a94321", fmt="NVFP4 (ModelOpt, W4A16)",
  runtime="SGLang / vLLM >= 0.29.0 (serving not validated end to end per card)", weights_modified=False,
  method="ModelOpt 0.46.0 NVFP4_EXPERTS_ONLY_CFG with expert input quantizers disabled (W4A16), group 16, data-free (algorithm: max); routed experts only, everything else incl. PLE tables BF16",
  methodology=None,
  variants=[dict(name="NVFP4-W4A16", size=169, unit="GiB", basis="disk", L=34)],
  flags=["No quality claims for the quant (card read in full): card has no evaluation and states serving has not been validated end to end [L68-70]",
         "Data-free calibration (algorithm: max) [L33]",
         "PLE n-gram tables (97.7 GiB BF16) copied unquantized; ~60 GiB on-GPU weights with PLE offloaded [L48-51, L65-66]"])

repo(repo="Quat3rnion/Signal-3.8-Flash-Next-HGN-v2", sha="d93045e754cc983e009e15a4bb019ae9006ad8fc", fmt="HGN (Halogen)",
  runtime="Halogen 0.18.5 (Peonist baseline runtime)", weights_modified=True,
  method="bf16-to-hgn 0.1.0 profile v2 conversion from agentionai/Signal-3.8-Flash-Next-GGUF Q8_0 (not original BF16); MTP/vision auxiliaries from original Qwen tensors; baseline-plus-delta publication over peonist-ai/halogen-qwen3.8-flash-next",
  methodology=None,
  variants=[dict(name="HGN v2 (assembled 14-file package)", size=118812702598, unit="bytes", basis="disk", L=90, L_extra=[96,97,98])],
  flags=["No quality claims for the quant (card read in full): section titled 'performance, not quality'; card states no IFEval or BFCL evaluation completed and no claim about accuracy, retained quality or BF16 equivalence [L154, L188-190]",
         "weights_modified: language-model source is agentionai/Signal-3.8-Flash-Next-GGUF (a Signal derivative, not Qwen's base weights); card says plain, non-abliterated [L20, L32-34]",
         "Converted from a Q8_0 GGUF, so it is a requantization of an already-quantized file [L51-53]",
         "Only throughput figures (65.217879 tok/s weighted decode) from one run [L163-165, L183]"])

repo(repo="SargeDev/qwen3.8-flash-next-lil-rascal-uncensored", sha="e20e53c9f923c932199a37e0496519c169f744eb", fmt="NVFP4/MXFP8",
  runtime="vLLM (single-Spark recipe, TP=1, MTP 3, fp8 KV)", weights_modified=True,
  method="Abliterated ('homebrew workflow') directly on local-inference-lab/Qwen3.8-Flash-Next-NVFP4 (NVFP4/MXFP8 QAD checkpoint)",
  methodology=None,
  variants=[dict(name="NVFP4/MXFP8 uncensored", size=None)],
  flags=["No quality claims for the quant (card read in full): card is provenance, intended/prohibited use and disclaimers only [L23-85]",
         "Weights modified: abliterated / uncensored [L9-10, L23]",
         "Quantization is local-inference-lab's NVFP4 checkpoint, not re-done by this publisher [L25]",
         "No size stated on card"])

repo(repo="SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Sushi", sha="950fcc19cf085ea81435f1b755680ae0dc5f8e7d", fmt="EXL3 (Sushi pack)",
  runtime="Sushi (Apple Silicon); -hq speeds need a dense hyper-connection read engine patch", weights_modified=False,
  method="Routed experts re-encoded in EXL3 trellis format by publisher's own converter at 5 or 6 bits; 8-bit trunk/embeddings/head/MTP from beamster/Qwen3.8-Flash-Next-Sushi-4bpw; -hq keeps hyper-connection mixes bf16; n-gram table is the base model's own (95 GiB) assembled by script",
  methodology=dict(reference="bf16 teacher (whole model)", dataset="wikitext 16 x 512 tokens; 16 real reasoning traces; 4 x 32,768-token documents", ctx="512 (wikitext) / 32,768 (32K set)", tool="not stated; 95% bootstrap intervals", L=[28,38,39]),
  table_header_L=30,
  variants=[
    dict(name="5bpw", size=78, unit="GiB", basis="trunk_only", m={"kld:wikitext":"0.0357","kld:reasoning":"0.0105","kld:32K":"0.0163","top1_pct":"92.9%"}, L=34, L_extra=[24]),
    dict(name="5bpw-hq", size=78, unit="GiB", basis="trunk_only", m={"kld:wikitext":"0.0325","kld:reasoning":"0.0095","kld:32K":"0.0153","top1_pct":"93.1%"}, L=35, L_extra=[25]),
    dict(name="6bpw-hq", size=92, unit="GiB", basis="trunk_only", m={"kld:wikitext":"0.0284","kld:reasoning":"0.0099","kld:32K":"0.0124","top1_pct":"93.5%"}, L=36, L_extra=[26]),
  ],
  cross=[
    dict(of="beamster/Qwen3.8-Flash-Next-Sushi-4bpw", variant="Sushi-4bpw", m={"kld:wikitext":"0.0550","kld:reasoning":"0.0180","kld:32K":"0.0248","top1_pct":"91.0%"}, L=33),
    dict(of="oMLX OptSpeed 4-bit affine 'port' (repo not identified on card)", variant="port (oMLX OptSpeed, 4-bit affine)", m={"kld:wikitext":"0.1416","kld:reasoning":"0.0528","kld:32K":"0.0841","top1_pct":"86.1%"}, L=32),
  ],
  flags=["Small eval sets: wikitext 16 x 512 tokens, 16 reasoning traces, 4 x 32K documents [L38]",
         "Sizes (~78 / ~92 GiB) exclude the 95 GiB n-gram table, which is downloaded separately from Qwen's shards; 6bpw-hq ~99 GB resident [L24-26, L54-55, L65]",
         "Headline 'several times closer to the bf16 model than 4-bit affine packs at similar memory' rests on the 'port' row, whose source repo is not named [L17-18, L32]",
         "Note: 6bpw-hq reasoning KLD (0.0099) is higher than 5bpw-hq (0.0095) on the card's own table [L35-36]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP", sha="2b170fa6309d5d1ee380b35636075fac7945f286", fmt="MLX",
  runtime="oMLX 0.6.3rc3 / MLX-VLM 0.6.3 (qwen4_exp native-MTP support required)", weights_modified=False,
  method="Uniform 4-bit MLX affine, group size 32, from official BF16 (incl. n-gram tables and MTP); multimodal modules and MoE router gates BF16",
  methodology=None,
  variants=[dict(name="MLX 4-bit g32 MTP", size=113.209, unit="GB", basis="disk", L=78)],
  flags=["No quality claims for the quant (card read in full): only MTP speed/acceptance and pass/fail parity/coherence checks with no scores [L117-126, L175-182]",
         "Weight size 113.209 GB / 105.434 GiB [L78]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-oQ4-MTP", sha="069090c38f60e654c08ac6cb8cd83fce1affbdbc", fmt="MLX",
  runtime="oMLX / MLX-VLM with qwen4_exp native-MTP support", weights_modified=False,
  method="oQ4 mixed precision: 4-bit affine g32 base with 232 protected modules at 5/8-bit, sensitivity from a 'validated local proxy'; rebuilt from official BF16 incl. MTP",
  methodology=None,
  variants=[dict(name="oQ4 MTP", size=113.33, unit="GB", basis="disk", L=77)],
  flags=["No quality claims for the quant (card read in full): only short-prompt speed tests (27-71 output tokens) and qualitative coherence notes [L111-119, L149-153]",
         "Card links an unresolved long-generation repetition report involving oQ4-MTP [L180]",
         "Weight size 113.33 GB, 105.54 GiB [L77]"])

repo(repo="UltimateIntent/Qwen3.8-Flash-Next-Abliterated-s1.5-GGUF", sha="dc11a1635f49fbd1e95dd92794a33e6ed7dc7a43", fmt="GGUF",
  runtime="llama.cpp / LM Studio", weights_modified=True,
  method="Abliterated derivative: strength 1.5 refusal-direction projection on 146 tensors (vision, MTP head, LM head, Gated Residual controllers unchanged); monolithic GGUFs Q8_0..IQ4_XS",
  methodology=None,
  variants=[
    dict(name="Q8_0", size=175.29, unit="GiB", basis="disk", L=39),
    dict(name="Q6_K", size=156.13, unit="GiB", basis="disk", L=40),
    dict(name="Q5_K_M", size=124.89, unit="GiB", basis="disk", L=41),
    dict(name="Q4_K_M", size=110.96, unit="GiB", basis="disk", L=42),
    dict(name="IQ4_XS", size=91.66, unit="GiB", basis="disk", L=43),
  ],
  flags=["No quality claims for the quant (card read in full): only a qualitative statement that evaluation showed 'no noticeable performance / capability loss', with no numbers, method or reference [L76-77]",
         "Weights modified: abliterated (refusal-direction projection, strength 1.5) [L31-33]",
         "Sizes labelled 'Approximate size' [L37]"])

repo(repo="Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed", sha="4285a4a04041511bf6581f543fdbfd154c93a608", fmt="MLX (MTPLX pack)",
  runtime="MTPLX 2.11.3", weights_modified=False,
  method="Flat 4-bit g64 for every MoE expert and dense matrix; GDN conv/recurrent params, norms, QSA indexer, MTP head 16-bit; n-gram table separate sidecar streamed from SSD; vision preserved",
  methodology=None,
  variants=[dict(name="Bare Speed (flat 4-bit)", size=106.3, unit="GB", basis="disk", L=100, L_extra=[101,102])],
  flags=["No quality claims for the quant (card read in full): only speed tables and a sampling-exactness statement for MTP vs plain decoding (not quantization quality) [L62-70, L80-83]",
         "Card describes the Optimized Speed sibling as 'higher quality' without figures [L50-52, L126]",
         "First speed row (125.8 tok/s) is measured on the Optimized Speed pack, not this repo [L64]",
         "Download 106.3 GB includes 32 GB n-gram table; ~74 GB resident; peak 78 GiB serving [L100-102]"])

repo(repo="agentionai/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST-GGUF", sha="5a2cf56c3e0f8bc8d395d53bec649ec8e358a993", fmt="GGUF (ROCmFP4, MTP draft head only)",
  runtime="llama.cpp fork LaurentZuijdwijk/llama.cpp vulkan/qwen4exp-rocmfpx", weights_modified=False,
  method="Qwen's MTP head quantized to ROCmFP4-FAST to match agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-GGUF target; draft head only, not a model",
  methodology=None,
  variants=[dict(name="MTP-ROCmFP4-FAST draft head", size=2.28, unit="GiB", basis="disk", L=23)],
  flags=["No quality claims for the quant (card read in full): only speculative-decoding throughput and draft acceptance on Radeon 8060S, 250 tokens [L56-71]",
         "Draft head only; used with the separate ROCmFP4-FAST target repo [L27-29]",
         "Card reports a Q8_0 draft measured worse on throughput and acceptance (30.3 t/s, 0.587) [L64-65]"])

repo(repo="argyelan/Qwen3.8-Flash-Next-GGUF", sha="22e87259f31184ad03ed4080c67ec53012c27635", fmt="GGUF",
  runtime="llama.cpp with PR #27742", weights_modified=False,
  method="convert_hf_to_gguf from official FP8 checkpoint to Q8_0 (--fp8-as-q8), then llama-quantize --allow-requantize to IQ4_XS without imatrix; 51B n-gram table requantized to Q5_0 with custom chunked streaming requantizer",
  methodology=dict(reference="none (absolute scores, no BF16/FP8 comparison)", dataset="HumanEval (164); GSM8K (200 problems)", ctx=None, tool="llama.cpp server, 'greedy-ish defaults'; two-box DGX Spark + RTX 5090 over 10 GbE RPC", L=[38,39,40,56,57]),
  variants=[dict(name="IQ4XS-NGQ5", size=98, unit="GB", basis="disk", m={"bench:HumanEval":"97.0 %","bench:HumanEval (passed/total)":"159/164","bench:GSM8K":"96.0 %","bench:GSM8K (correct/total)":"192/200"}, L=56, L_extra=[25])],
  flags=["Absolute benchmark scores only; no BF16 or FP8 baseline measured on the card [L56-57]",
         "GSM8K on a 200-problem subset; decoding described only as 'greedy-ish defaults' [L56-57]",
         "Requantized from the official FP8 checkpoint (not BF16), no imatrix [L51]",
         "Evals were run on a two-box RPC setup (DGX Spark + RTX 5090) [L38-40]",
         "Card lists Q8_0 master (176 GB) as not yet uploaded [L26]"])

repo(repo="axiomofmind/Qwen3.8-Flash-Next-W4A16-NVFP4", sha="41ef3b02fa1ef61b6f3c10165f9e6955de03d2a2", fmt="NVFP4 (ModelOpt, W4A16)",
  runtime="runtime with ModelOpt W4A16 NVFP4 support (not named)", weights_modified=False,
  method="ModelOpt W4A16 NVFP4: routed experts NVFP4 g16, BF16 activations; attention, vision, shared experts, routers, PLE, embeddings, head, MTP at source precision",
  methodology=None,
  variants=[dict(name="W4A16 NVFP4", size=None)],
  flags=["No quality claims for the quant (card read in full): 29-line card with format description only [L15-29]",
         "No size stated on card"])

repo(repo="brianbellx/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF-copy", sha="0dec8ab02819304704202e3bb78ce5b4159c8bdf", fmt="GGUF",
  runtime="llama.cpp", weights_modified=True,
  method="Copy of huihui-ai's abliterated GGUF card: abliteration via remove-refusals-with-transformers; card states GGUFs come from unsloth/Qwen3.8-Flash-Next-GGUF",
  methodology=None,
  variants=[dict(name="UD-Q4_K_XL (per example command)", size=None)],
  flags=["No quality claims for the quant (card read in full): card is abliteration description, usage warnings and donation info only [L18-54]",
         "Weights modified: abliterated / uncensored [L21-22]",
         "Card heading names huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF and repo name ends in '-copy'; card text matches the huihui-ai card [L18]",
         "Card states 'GGUFs come from unsloth/Qwen3.8-Flash-Next-GGUF' without explaining how that relates to the abliteration [L25]",
         "No sizes stated on card"])

repo(repo="huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF", sha="06a318de4011598a142ee54bcfec47c6921c958e", fmt="GGUF",
  runtime="llama.cpp / Strata", weights_modified=True,
  method="Abliteration via remove-refusals-with-transformers; card states GGUF series come from unsloth/Qwen3.8-Flash-Next-GGUF and (Swift series) from ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
  methodology=None,
  variants=[dict(name="unsloth series (e.g. UD-Q4_K_XL)", size=None), dict(name="Swift series (from ukisai GSQ-RCO)", size=None)],
  flags=["No quality claims for the quant (card read in full): card is abliteration description, update notes, usage warnings and donation info only [L27-88]",
         "Weights modified: abliterated / uncensored [L30-31]",
         "Card states GGUFs come from unsloth and ukisai repos; both update sections say 'This is just a test/validation' [L35, L47, L52, L56, L59]",
         "Card instructs disabling Strata's SHA256 verification for the unsloth series [L39-45]",
         "No sizes stated on card"])

repo(repo="codelight-ai/Qwen3.8-Flash-Next-NVFP4-PLE-INT3", sha="59ecf4e3e99cd8df95b2316032e89386df4015bf", fmt="NVFP4 experts + INT3 PLE (compressed-tensors style)",
  runtime="vLLM PR #53896 + repo's vllm-qwen4-exp-fixes.patch (not stock vLLM); GB10 / DGX Spark", weights_modified=False,
  method="From RadixArk/Qwen3.8-Flash-Next-NVFP4: routed experts NVFP4 copied byte-identical; PLE n-gram tables re-quantized from Qwen BF16 to INT3 asymmetric per-row with per-row MSE-optimal clip search (3.20 bpw, no calibration); MTP head removed",
  methodology=dict(reference="BF16 PLE weights (weight-space rel-err ||Wq-W||_F/||W||_F and cosine)", dataset="PLE table tensors (weight space; no end-to-end eval)", ctx=None, tool="not stated", L=[97,106,107]),
  table_header_L=99,
  variants=[dict(name="NVFP4 + INT3 PLE", size=99.3, unit="GB", basis="disk", mem=91.9, mem_unit="GiB", mem_basis="resident",
    m={"ple_rel_err":"0.1854","ple_cos":"0.982685","ple_size_gb":"20.5 GB"}, L=64, L_extra=[65,104])],
  ladder=[
    dict(name="INT6 asym per-row (PLE scheme measured, not shipped)", m={"ple_rel_err":"0.0241","ple_cos":"0.999710"}, L=102),
    dict(name="INT4 asym per-row (PLE scheme measured, not shipped)", m={"ple_rel_err":"0.0958","ple_cos":"0.995406"}, L=103),
  ],
  cross=[dict(of="RadixArk/Qwen3.8-Flash-Next-NVFP4", variant="FP8 e4m3 per-tensor PLE", row_label="RadixArk NVFP4", size=135.2, unit="GB", basis="disk", m={"ple_rel_err":"0.0265","ple_cos":"0.999654"}, L=101, L_extra=[64])],
  flags=["Only weight-space error figures for the PLE table; card states end-to-end quality has not been measured and INT3 is ~7x the error of the FP8 it replaces [L106-107]",
         "Functional checks are a few short prompts and 4/4 tool-calling cases; long-form/agentic quality stated as unmeasured [L23-31]",
         "MTP head removed: speculative decoding not available [L86-89]",
         "Routed experts are RadixArk's NVFP4 tensors, byte-identical [L60, L81]",
         "99.3 GB on disk; 91.9 GiB resident [L64-65]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ2.7e-mtp", sha="77f5e6313c723169224bf333e4cec8b329bed7f8", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQe (oMLX v0.7.0) imatrix-weighted mixed precision, oQ level 2.7, default 2-bit g64; imatrix oqe_code_multilingual 1,024x512 streamed from bf16; sensitivity map measured on Jundot oQ4e-mtp proxy",
  methodology=None,
  variants=[dict(name="oQ2.7e", size=74, unit="GB", basis="disk", bpw=3.29, L=31)],
  flags=["No quality claims for the quant (card read in full): only bit-share table and calibration details [L38-57]",
         "Layer sensitivity measured on a quantized proxy (Jundot oQ4e-mtp), not bf16 [L57]",
         "Size stated as '~3.29 bits per weight, 74 GB' [L31]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ3e-mtp", sha="24e392e8fdbce51b7bfbad0bbf0d167bed505d01", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQe (oMLX v0.7.0) imatrix-weighted mixed precision, oQ level 3, default 3-bit g64; imatrix oqe_code_multilingual 1,024x512 streamed from bf16; sensitivity map measured on Jundot oQ4e-mtp proxy",
  methodology=None,
  variants=[dict(name="oQ3e", size=84, unit="GB", basis="disk", bpw=3.75, L=31)],
  flags=["No quality claims for the quant (card read in full): only bit-share table and calibration details [L38-57]",
         "Layer sensitivity measured on a quantized proxy (Jundot oQ4e-mtp), not bf16 [L57]",
         "Size stated as '~3.75 bits per weight, 84 GB' [L31]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ6-mtp", sha="fe8b2e38688bbbee1a7d1648b87905f5c64ac811", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ (oMLX v0.7.0) mixed precision, oQ level 6, default 6-bit g64 (99.1% 6-bit, 0.9% 8-bit); sensitivity map measured on Jundot oQ4e-mtp proxy",
  methodology=None,
  variants=[dict(name="oQ6", size=150, unit="GB", basis="disk", bpw=6.69, L=27)],
  flags=["No quality claims for the quant (card read in full): only bit-share table [L34-43]",
         "Layer sensitivity measured on a quantized proxy (Jundot oQ4e-mtp), not bf16 [L43]",
         "Size stated as '~6.69 bits per weight, 150 GB' [L27]"])

repo(repo="ddalcu/Qwen3.8-Flash-Next-MLX-Serve-8bit", sha="f61a712f5f40f6fe0e886c4a3fc36c71faee3f9e", fmt="MLX (mlx-serve pack)",
  runtime="mlx-serve", weights_modified=False,
  method="Every quantized tensor 8-bit g64 (n-gram table 8-bit g32 in a separate mmapped ngram_table.bin, 57.6 GB); routers/gates/norms/convs/SSM state bf16; vision dense bf16; MTP same policy as trunk",
  methodology=None,
  variants=[dict(name="8bit", size=195, unit="GB", basis="disk", mem=137, mem_unit="GB", mem_basis="resident", L=32, L_extra=[25])],
  ladder=[
    dict(name="iQ-MLX-3.3bpw", size=86, unit="GB", basis="disk", mem=52, L=29),
    dict(name="iQ-MLX-4.7bpw", size=107, unit="GB", basis="disk", mem=75, L=30),
    dict(name="mixed-4-8bit", size=107, unit="GB", basis="disk", mem=75, L=31),
  ],
  flags=["No quality claims for the quant (card read in full): 'closest-to-bf16 pack in the family' and iQ-MLX-4.7bpw 'closer to bf16' are stated without any figures [L22, L31]",
         "Download 195 GB includes the 57.6 GB n-gram table, which is mmapped and not resident; 137 GB resident [L32-34, L62]",
         "Engine validated against HF transformers on a tiny random model only, before the full conversion [L109-111]"])

repo(repo="drluoto/Qwen3.8-Flash-Next-MTP-GGUF", sha="ac875a98457a8effe8fb83de8f3701421f507a24", fmt="GGUF (MTP draft head only)",
  runtime="drluoto/llama.cpp strix-halo-vulkan branch (detached draft head support, PR #27836 under review)", weights_modified=False,
  method="Qwen MTP draft-head tensors (+ shared embeddings, lm_head) converted from the official checkpoint; FR-Spec 65k trimmed-vocab variants; Q5_K default requantized from Q8_0",
  methodology=None,
  variants=[
    dict(name="Q5_K-frspec-65k", size=2.70, unit="GB", basis="disk", L=42),
    dict(name="Q8_0-frspec-65k", size=3.64, unit="GB", basis="disk", L=43),
    dict(name="Q8_0", size=4.14, unit="GB", basis="disk", L=44),
    dict(name="Q4_K_M", size=2.79, unit="GB", basis="disk", L=45),
    dict(name="Q5_K-frspec-65k-bf16path", size=2.70, unit="GB", basis="disk", L=46),
    dict(name="bf16", size=7.78, unit="GB", basis="disk", L=47),
  ],
  flags=["No quality claims for the quant (card read in full): draft head only; all figures are whole-stack decode throughput and draft acceptance, not quantization quality [L76-98, L118-124]",
         "Measured against the publisher's own AgenticRequant Q5K trunk on Strix Halo (Vulkan) [L51-53, L81-82]",
         "Card states Q5_K greedy output is not identical to Q8_0-head output and reports qualitative checks only ('same answers on arithmetic at 24k context, tool calls...') [L104-107]"])

repo(repo="fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl", sha="0f94842e3682b49acace4daeb4cf2e65a6f94f83", fmt="lily qwen4_exp-affine-v1 (safetensors, engine-specific)",
  runtime="lily-qwen3.8-flash-next (Metal, Apple Silicon) only", weights_modified=False,
  method="Per-module affine: routed experts + GDN projections 4-bit g64; attention q/k/v/o, shared expert, LM head, embeddings 8-bit g64; draft head 4-bit; n-gram tables 4-bit g32; vision bf16 (converter --q4-xl)",
  methodology=dict(reference="sibling 4-bit checkpoint fabiogreter/Qwen3.8-Flash-Next-lily-q4 (behavioural comparison, no bf16 reference)", dataset="one captured opencode turn with 56K tokens of context, 30 samples per checkpoint, thinking controls on", ctx=56000, tool="lily-qwen3.8-flash-next server", L=[86,87,88,89,90,91]),
  variants=[dict(name="lily-q4-xl", size=99.6, unit="GiB", basis="disk", m={"tool_call_continuation:opencode 56K turn":"23 of 30"}, L=44, L_extra=[87,88])],
  ladder=[dict(name="fabiogreter/Qwen3.8-Flash-Next-lily-q4 (4-bit sibling)", m={"tool_call_continuation:opencode 56K turn":"18 of 30"}, L=88)],
  flags=["Only quality-related figure is a single-prompt behavioural count (samples that went on to a tool call: 23/30 vs 18/30 for the 4-bit sibling); card itself says 'one prompt, not a task-level evaluation' [L86-91]",
         "No bf16 reference or KLD/PPL; card notes quantized output is not bit-identical to bf16 [L154-155]",
         "Size 99.6 GiB across 55 safetensors files incl. draft head and vision; ~75 GB GPU-resident, 32 GB n-gram tables via page cache [L44, L97-99]"])

repo(repo="gabmiranda/Qwen3.8-Flash-Next-oQ3e-mtp", sha="d1453d36c2efe749dffb0db01df30729e202ea79", fmt="MLX",
  runtime="not stated (oQ / MLX per tags)", weights_modified=False, method="oQ3e per repo name and tags; card states nothing else",
  methodology=None,
  variants=[dict(name="oQ3e-mtp", size=None)],
  flags=["No quality claims for the quant (card read in full): 11-line card with front matter and a title only [L1-11]",
         "No size stated on card"])

repo(repo="garnermccloud/Qwen3.8-Flash-Next-MLX-SSD-Stream", sha="9a624d9cdae5e4be1ad00845a9304210fde18562", fmt="MLX (mlx-serve pack)",
  runtime="garnermccloud/mlx-serve (fork of ddalcu/mlx-serve)", weights_modified=False,
  method="Routed experts (incl. MTP) Q4 affine g64; draft-only vocabulary head Q6 g64; core/embeddings/verification head/vision BF16; n-gram table is RadixArk's FP8 E4M3 table streamed from SSD",
  methodology=None,
  variants=[dict(name="SSD Stream (Q4 experts, BF16 core)", size=127, unit="GB", basis="disk", mem=75.3, mem_unit="GB", mem_basis="resident", L=58, L_extra=[56])],
  flags=["No quality claims for the quant (card read in full): 'Quality-first' heading but only speed figures, one long-context retrieval pass and 39 pass/fail end-to-end checks with no scores [L21-26, L66-75]",
         "n-gram table taken from RadixArk/Qwen3.8-Flash-Next-NVFP4 (FP8 E4M3) [L39, L109]",
         "Approximately 127 GB download; ~75.3 GB allocated before request caches; 51.2 GB table streamed [L56-58]"])

repo(repo="haihengh/Qwen3.8-Flash-Next-125B-finch-4bit", sha="2464910db4eed30232e9bb15c049231d3a0a6567", fmt="FinchMoE .finch (custom streaming layout)",
  runtime="FinchMoE (Swift/Metal) only", weights_modified=False,
  method="4-bit affine g64 (BF16 scale/bias) for routed/shared experts, attention, embedding; GDN and router 8-bit; produced by FinchMoE installer; PLE shards 95 GB",
  methodology=None,
  variants=[dict(name="finch 4-bit", size=162, unit="GB", basis="disk", L=57, L_extra=[50,51,52])],
  flags=["No quality claims for the quant (card read in full): only an oracle argmax-agreement correctness check; card states no downstream benchmark has been run and none is claimed [L164-169]",
         "Total 162 GB: 63 GB expert pool + 95 GB PLE shards + 3.6 GB other weights [L50-57]",
         "Speed figures from a 16 GB Mac mini streaming from external SSD (2.76–2.91 tok/s decode) [L117-125]"])

repo(repo="inferencerlabs/Qwen3.8-Flash-Next-MLX-Q9", sha="627f6d688b430f2ff14ecd24b3979ce736ccb181", fmt="MLX",
  runtime="Inferencer app (M3 Ultra); quantized with a modified MLX", weights_modified=False,
  method="Q9 (9 bpw) quantization with a modified version of MLX; recipe not described",
  methodology=dict(reference="Base model ('predicting base tokens')", dataset="not stated", ctx=None, tool="not stated (metric definitions only: perplexity of base tokens, token accuracy, missed divergence)", L=[43,44,45]),
  variants=[dict(name="Q9", size=None, mem=189.6, mem_unit="GiB", mem_basis="ram_in_use",
    m={"ppl":"1.20312","token_accuracy_pct":"97.80%","missed_divergence_pct":"9.60%"}, L=37, L_extra=[17])],
  ladder=[
    dict(name="Q3.5", m={"ppl":"168.0","token_accuracy_pct":"43.45%","missed_divergence_pct":"72.57%"}, L=32),
    dict(name="Q4.5", m={"ppl":"1.33593","token_accuracy_pct":"91.65%","missed_divergence_pct":"17.28%"}, L=33),
    dict(name="Q5.5", m={"ppl":"1.23437","token_accuracy_pct":"95.05%","missed_divergence_pct":"17.28%"}, L=34),
    dict(name="Q6.5", m={"ppl":"1.21875","token_accuracy_pct":"96.65%","missed_divergence_pct":"12.03%"}, L=35),
    dict(name="Q8.5", m={"ppl":"1.21875","token_accuracy_pct":"97.65%","missed_divergence_pct":"9.92%"}, L=36),
  ],
  baseline={"ppl":"1.20312","token_accuracy_pct":"100%","missed_divergence_pct":"0.000%","L":38},
  flags=["Dataset, sample count, context length and tool for the table are not stated; only metric definitions are given [L43-45]",
         "Table headline 'Q9-bit quant typically achieves near lossless accuracy' — card does not say whether the table was measured on this model [L25-38]",
         "Ladder rows (Q3.5–Q8.5) are not linked to specific repos; Q4.5 and Q5.5 show identical Missed Divergence 17.28% [L33-34]",
         "Only memory figure is ~189.6 GiB during multimodal inference on M3 Ultra; no file size stated [L17]"])

repo(repo="intel-ai/Qwen3.8-Flash-Next-MXFP4-BF16Attn-PLEMXFP8-CT-AutoRound", sha="b17da51a760d41e93f6fb8cd278cd900482db1f1", fmt="compressed-tensors (MXFP4 experts + MXFP8 PLE)",
  runtime="vLLM 0.29.1rc1.dev528 + mandatory PLE MXFP8 patch (patches/)", weights_modified=False,
  method="Intel AutoRound 0.16.0 --model_free: routed experts MXFP4 W4A4 g32 UE8M0 dynamic activations; PLE n-gram table converted offline to MXFP8 (E4M3 + UE8M0 g32); attention, GDN, shared experts, routers, MTP, embeddings, lm_head, vision BF16",
  methodology=dict(reference="Qwen/Qwen3.8-Flash-Next bf16 (own run)", dataset="GSM8K, MMLU, PIQA, HellaSwag (lm-eval tasks)", ctx=8192, tool="lm-eval 0.4.13 + vLLM 0.29.1rc1.dev528 with patch, TP=1, bf16 KV, seed=42, enable_thinking=false", L=[95,96,97]),
  table_header_L=99,
  variants=[dict(name="MXFP4 + PLE MXFP8 (BF16Attn)", row_label="This checkpoint", size=133.0, unit="GB", basis="disk",
    m={"bench:GSM8K (strict / flexible)":"0.9636 / 0.9636","bench:MMLU":"0.8650","bench:PIQA (acc / acc_norm)":"0.8150 / 0.8319","bench:HellaSwag (acc / acc_norm)":"0.6834 / 0.8726","ple_snr_db":"53.6 dB"},
    L=103, L_extra=[26,28])],
  ladder=[dict(name="Sibling -BF16Attn (PLE table bf16)", m={"bench:GSM8K (strict / flexible)":"0.9644 / 0.9636","bench:MMLU":"0.8641","bench:PIQA (acc / acc_norm)":"0.8248 / 0.8308","bench:HellaSwag (acc / acc_norm)":"0.6856 / 0.8726"}, L=102)],
  cross=[dict(of="nvidia/Qwen3.8-Flash-Next-NVFP4", variant="NVFP4", size=132.7, measured=False, unit="GB", basis="disk", L=26, L_extra=[125], flags=["quoted, not measured"])],
  baseline={"bench:GSM8K":"0.9674","bench:MMLU":"0.8652","bench:PIQA":"0.8194","bench:HellaSwag":"0.6927","L":101},
  flags=["Four lm-eval tasks at max_model_len 8192 with thinking disabled; card says long-context (128k) behaviour of the MXFP8 PLE table is not validated [L95-97, L130-132]",
         "Card's 'no measurable accuracy cost' claim compares against its own bf16-table sibling; deltas within ~1σ per card [L105-106]",
         "Requires a vLLM patch; card states unpatched vLLM silently misreads the table and degenerates [L30-33, L129]",
         "PLE SNR 53.6 dB is a weight-level fidelity figure [L28]",
         "NVIDIA NVFP4 size 132.7 GB quoted for comparison, not a measured quality comparison [L26, L125]"])

repo(repo="jockevaupptaget/Qwen3.8-Flash-Next-MTP-GGUF", sha="69da733459b737b79273c0a322340de9c9c08fa2", fmt="GGUF (MTP draft head only)",
  runtime="patched llama.cpp (apepojken/llama.cpp qwen4exp-spec-mtp)", weights_modified=False,
  method="Qwen's native 4B MTP head converted to standalone Q8_0 sidecar GGUF from official weights (convert_hf_to_gguf --mtp --outtype q8_0)",
  methodology=None,
  variants=[dict(name="mtp Q8_0", size=4.1, unit="GB", basis="disk", L=14)],
  flags=["No quality claims for the quant (card read in full): draft head only; figures are decode speedup and draft acceptance with a UD-Q3_K_XL target [L24-25]",
         "Not a runnable model on its own [L11-17]"])

repo(repo="madlabs-ai/Qwen3.8-Flash-Next-NVFP4-MTP-BF16", sha="85bee1eb093af8a99a9f9d041f9bc1de2251a782", fmt="NVFP4 (ModelOpt)",
  runtime="SGLang (pennyroyal build), --quantization modelopt_fp4, single RTX PRO 6000", weights_modified=False,
  method="Transplant: nvidia/Qwen3.8-Flash-Next-NVFP4 body (byte-identical pure-body shards) with FP8 MTP tensors dropped and RadixArk's 31 BF16 MTP tensors grafted; config ple_embedding_dtype set; no re-quantization",
  methodology=None,
  variants=[dict(name="NVFP4 body + BF16 MTP", size=None)],
  flags=["No quality claims for the quant (card read in full): 'gets NVIDIA's quantization quality' is stated without figures; only decode speed and a tool-call malformation count tied to an SGLang strictness setting [L65-66, L85-96]",
         "Not an original quant: NVIDIA NVFP4 body + RadixArk BF16 MTP head [L27-31, L45-47]",
         "No size stated; VRAM delta vs RadixArk export stated as 0.00 GiB [L60]"])

repo(repo="kingjones777/Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-GGUF", sha="bf7fbc6658acb8aa87144fbc70983ca18e86020a", fmt="GGUF (ROCmFP4)",
  runtime="kingjones30/ROCmFPX fork (qwen4exp + ROCmFP4) with bundled patches; gfx1151 / ROCm 7.2.4", weights_modified=False,
  method="STRIX_LEAN tier: ROCmFP4 experts/shared expert (TYPE_101), attention half TYPE_100 / half TYPE_101, PLE table Q5_1, token_embd Q5_K, output Q6_K; quantized from publisher's own BF16 conversion",
  methodology=None,
  variants=[dict(name="Q4_0-ROCmFP4-STRIX_LEAN", size=98.49, unit="GiB", basis="disk", bpw=4.78, L=134, L_extra=[205])],
  ladder=[
    dict(name="FAST tier", size=87.94, unit="GiB", basis="disk", L=204),
    dict(name="STRIX tier", size=113.47, unit="GiB", basis="disk", L=206),
  ],
  flags=["No quality claims for the quant (card read in full): card explicitly states 'Quality is the one thing I have not measured, so I won't claim it' [L297-298]",
         "All measurements are speed, memory (GTT) and MTP acceptance on Strix Halo; MTP numbers measured on the Uncensored STRIX_LEAN-imatrix weights, not this repo's file [L54-56, L74-76]",
         "4.78 bpw, 98.49 GiB; PLE table 35.76 GiB of that; ~63–75 GiB resident [L134, L143, L332]"])

repo(repo="mbehr90/Qwen3.8-Flash-Next-fp8", sha="e70592b7a40d1b85ac879088ba2093a6504fb6ec", fmt="FP8 (compressed-tensors, FP8_DYNAMIC)",
  runtime="vLLM with Qwen4ExpForConditionalGeneration (out-of-tree plugin; not vLLM 0.26.0)", weights_modified=False,
  method="FP8 per-output-channel weights, dynamic per-token activations, routed experts only (quantized shard by shard without building the model, no calibration); PLE, embeddings, attention/GDN, shared experts, gates, vision, MTP, lm_head bf16",
  methodology=dict(reference="bf16 source (own run)", dataset="GSM8K, 200 questions, via chat endpoint with final number parsed from response", ctx=8192, tool="vLLM on 4x H100 80 GB (serving command max-model-len 8192); custom parsing, not lm-eval strict-match", L=[26,32,33,34,35,60]),
  table_header_L=26,
  variants=[dict(name="FP8", size=220.8, unit="GiB", basis="disk", m={"bench:GSM8K (200 q, chat)":"0.965","mean_answer_chars":"1146"}, L=29, L_extra=[20])],
  ladder=[dict(name="NVFP4 (mbehr90/Qwen3.8-Flash-Next-nvfp4)", size=170.2, unit="GiB", basis="disk", m={"bench:GSM8K (200 q, chat)":"0.960","mean_answer_chars":"1087"}, L=30)],
  baseline={"bench:GSM8K (200 q, chat)":"0.950","mean_answer_chars":"1183","size":"335.3 GiB","L":28},
  flags=["Single benchmark, n=200; card states SE ~1.5 pp and all three statistically indistinguishable [L26-33]",
         "GSM8K answers parsed from chat responses by the publisher, not lm-eval strict-match [L33-35]",
         "Size 220.8 GiB vs 335.3 GiB bf16 source [L20]",
         "NVFP4 row is the publisher's own sibling build (repo name not given on the card) [L30]"])

repo(repo="mlx-community/Qwen3.8-Flash-Next-4bit-mtp", sha="825df623302aeebc434f44ca9122827b04f9a0f1", fmt="MLX",
  runtime="MLX runtimes with qwen4_exp MTP support (not named)", weights_modified=False,
  method="mlx-community/Qwen3.8-Flash-Next-4bit shards unchanged (4-bit affine g32; routers/shared-expert gates 8-bit g64) plus grafted MTP head shard (routed experts and hyper-connection mixes 4-bit g32, rest bf16)",
  methodology=None,
  variants=[dict(name="4bit + MTP shard", size=None), dict(name="MTP head shard", size=1.61, unit="GiB", basis="disk", L=23)],
  flags=["No quality claims for the quant (card read in full): 27-line card describing contents only [L13-27]",
         "Main 22 shards are mlx-community/Qwen3.8-Flash-Next-4bit unchanged; no total size stated [L21]"])

repo(repo="nirecom/Qwen3.8-Flash-Next-mixed-3_8bit", sha="a158876aa37c101e8bc39eb7f93031f50ddfc86b", fmt="MLX",
  runtime="mlx_vlm / LM Studio MLX runtime", weights_modified=False,
  method="Mixed precision adapted from pipenetwork's layout: routers/MoE gates and QSA indexer unquantized; PLE table 4-bit g32; routed experts 3-bit g64; everything else 8-bit g64",
  methodology=dict(reference="none (absolute IFEval scores; external comparison to unsloth UD-Q3_K_XL)", dataset="IFEval, full 541 prompts, 0-shot", ctx=None, tool="lm-evaluation-harness 0.4.13 local-chat-completions, greedy; max_gen_toks 1280 (thinking off) / 8192 (thinking on, --thinking-budget 4096); M5 Max 128 GB", L=[77,78,79,80,106,107,129]),
  table_header_L=87,
  variants=[dict(name="mixed-3_8bit", size=85, unit="GB", basis="disk", bpw=4.094,
    m={"bench:IFEval prompt_strict (thinking off)":"0.8447","bench:IFEval prompt_loose (thinking off)":"0.8743","bench:IFEval inst_strict (thinking off)":"0.8969","bench:IFEval inst_loose (thinking off)":"0.9185",
       "bench:IFEval prompt_strict (thinking on)":"0.9279","bench:IFEval prompt_loose (thinking on)":"0.9464","bench:IFEval inst_strict (thinking on)":"0.9484","bench:IFEval inst_loose (thinking on)":"0.9628"},
    L=89, L_extra=[91,28])],
  cross=[dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-Q3_K_XL",
    m={"bench:IFEval prompt_strict (thinking off)":"0.8521","bench:IFEval prompt_loose (thinking off)":"0.8872","bench:IFEval inst_strict (thinking off)":"0.9017","bench:IFEval inst_loose (thinking off)":"0.9269",
       "bench:IFEval prompt_strict (thinking on)":"0.9261","bench:IFEval prompt_loose (thinking on)":"0.9427","bench:IFEval inst_strict (thinking on)":"0.9472","bench:IFEval inst_loose (thinking on)":"0.9604"},
    L=90, L_extra=[92])],
  flags=["IFEval only (instruction-following); card states knowledge, reasoning, coding, multilingual, safety and vision not measured [L71-75, L140-145]",
         "No BF16 baseline; comparison point is unsloth UD-Q3_K_XL run through the same harness [L84-85]",
         "Thinking-off: unsloth UD-Q3_K_XL slightly ahead on all four metrics; thinking-on: this build slightly ahead [L89-92, L96-97]",
         "Thinking-on rows used different reasoning caps per runtime (MLX --thinking-budget 4096, llama.cpp --reasoning-budget 4096; GGUF had 2 empty answers of 541) [L106-108]",
         "85 GB, 4.094 bpw [L27-28]"])

repo(repo="p4ik/Qwen3.8-Flash-Next-MLX-BEAM-4bit", sha="29e259b674945ea3a75a334c6b8fc02867adf10b", fmt="MLX (B.E.A.M. layout)",
  runtime="mlx-lm with --trust-remote-code (text only)", weights_modified=False,
  method="Repackage of pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit (4-bit routed experts, 4-bit g32 n-gram tables, 8-bit everything else); vision tower moved to extras/, hardened chat template; no re-quantization",
  methodology=None,
  variants=[dict(name="BEAM-4bit (pipenetwork mixed-4_8bit repack)", size=106, unit="GB", basis="disk", L=23)],
  flags=["No quality claims for the quant (card read in full): card states quality was not measured by the publisher and refers to the source build's card [L21, L52, L81]",
         "Not an original quant: weights bit-identical to pipenetwork mixed-4_8bit [L23, L42]",
         "Only a smoke test under the publisher's own expert-streaming loader (two short prompts and a 3.7k prompt) [L80]"])

repo(repo="quimmedes/Qwen3.8-Flash-Next-MTP-GGUF", sha="e56ac7b14f211f715e997121f26fdae09fb22e34", fmt="GGUF (MTP draft head only)",
  runtime="cafe-llama.cpp fork (quimmedes)", weights_modified=False,
  method="MTP draft heads re-quantized from BF16 with llama-quantize (cafe-llama.cpp build c2f5366e3) to Q2_0 / Q3_0 / Q4_0",
  methodology=None,
  variants=[
    dict(name="mtp Q2_0", size=1.10, unit="GB", basis="disk", L=67),
    dict(name="mtp Q3_0", size=1.50, unit="GB", basis="disk", L=68),
    dict(name="mtp Q4_0", size=2.20, unit="GB", basis="disk", L=69),
  ],
  flags=["No quality claims for the quant (card read in full): draft heads only; only an approximate acceptance-rate range (~50-70%) is mentioned [L78]",
         "Card states earlier Q4_K_M/Q6_K/Q8_0/BF16 files came from a broken conversion and were removed [L71-73]",
         "Sizes marked approximate (~) [L67-69]"])

repo(repo="scottlowry/Qwen3.8-Flash-Next-oQ4e-fp16-mtp", sha="aa8cb7ccca2b019d903863ee10b538d96327fcb5", fmt="MLX",
  runtime="oMLX (v0.7.1.dev1 used for quantization)", weights_modified=False,
  method="oQ (oMLX v0.7.1.dev1) mixed precision, bits 4, group size 64",
  methodology=None,
  variants=[dict(name="oQ4e-fp16", size=None)],
  flags=["No quality claims for the quant (card read in full): 20-line card with quantization parameters only [L11-20]",
         "No size stated on card"])

repo(repo="scottlowry/Qwen3.8-Flash-Next-oQ6e-mtp", sha="9cb47a71c2215c9dfaafc2e443958e60cd17be2d", fmt="MLX",
  runtime="oMLX (v0.7.1.dev1 used for quantization)", weights_modified=False,
  method="oQ (oMLX v0.7.1.dev1) mixed precision, bits 6, group size 64",
  methodology=None,
  variants=[dict(name="oQ6e", size=None)],
  flags=["No quality claims for the quant (card read in full): 20-line card with quantization parameters only [L11-20]",
         "No size stated on card"])

repo(repo="sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit", sha="c9bb01bda259a48a89be82db5dcd6aa62cceb4c5", fmt="MLX",
  runtime="mlx-vlm (git main after 2026-08-27); NVMe n-gram mode needs publisher's unreleased patch", weights_modified=True,
  method="REAP expert pruning 512 -> 384 per layer (saliency calibrated on the quantized weights over ~686K tokens of agentic-coding traffic) applied to Sawfwair/Qwen3.8-Flash-Next-MLX-4bit (affine 4-bit g64, n-gram table g32); RMSNorm re-centering and n-gram tensor renames",
  methodology=dict(reference="stock 512-expert MLX 4-bit conversion (same harness), not BF16", dataset="HumanEval, 164 problems, unit-test verified, one run per build", ctx=None, tool="not stated; same harness and machine for all builds", L=[98,99,134,135]),
  table_header_L=18,
  variants=[dict(name="REAP-384 4-bit", size=80, unit="GB", basis="disk", mem=51, mem_unit="GB", mem_basis="resident", m={"bench:HumanEval pass@1":"92.1%"}, L=21, L_extra=[93])],
  cross=[dict(of="Sawfwair/Qwen3.8-Flash-Next-MLX-4bit", variant="Base Q4 / stock conversion (512 experts)", size=98, unit="GB", basis="disk", mem=97, m={"bench:HumanEval pass@1":"93.9%"}, L=20, L_extra=[92])],
  ladder=[
    dict(name="REAP-320", size=72, unit="GB", basis="disk", m={"bench:HumanEval pass@1":"90.9%"}, L=94),
    dict(name="REAP-288 (sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit)", size=68, unit="GB", basis="disk", m={"bench:HumanEval pass@1":"91.5%"}, L=95),
    dict(name="REAP-256", size=65, unit="GB", basis="disk", m={"bench:HumanEval pass@1":"88.4%"}, L=96),
  ],
  flags=["Weights modified: REAP expert pruning (384 of 512 experts kept) [L26-27, L117]",
         "Single-run HumanEval (164 problems), no confidence intervals; card says 1-2 point differences between neighbours are within noise [L98-99, L134-135]",
         "Reference is the stock 4-bit conversion (identified in provenance as Sawfwair/Qwen3.8-Flash-Next-MLX-4bit), not BF16 [L20, L115]",
         "Calibration is one team's agentic-coding traffic; card warns non-code domains may degrade more [L118, L131-133]",
         "51 GB resident requires the publisher's NVMe n-gram patch; ~80 GB resident when loaded fully [L23-24, L107-110]",
         "Card mentions an eleven-point ladder but shows five rows; 320 scores below 288 on the card's own table [L87, L92-96]"])

repo(repo="talhasarac/Qwen3.8-Flash-Next-REAP128-86B-Unity-OpenCode-GGUF", sha="d396d76ff3516db6a80a079819f1c8e03beea31b", fmt="GGUF",
  runtime="llama.cpp", weights_modified=True,
  method="REAP expert pruning 512 -> 128 routed experts per layer (activation-weighted saliency from 50 English Unity/C#/OpenCode prompts, 5,120 observer tokens) applied to unsloth UD-IQ4_XS GGUF; no training",
  methodology=None,
  variants=[dict(name="REAP128 UD-IQ4_XS", size=48853994112, unit="bytes", basis="disk", L=58)],
  flags=["No quality claims for the quant (card read in full): card states no broad benchmark suite has been run; validation notes are anecdotal and report clear general-quality degradation [L42, L143-147]",
         "Weights modified: REAP expert pruning to 128 of 512 experts per layer [L25, L50]",
         "Built from unsloth's UD-IQ4_XS GGUF, not re-quantized [L26-28, L57]",
         "Calibration set is small and domain-biased (Unity, C#, tool calls) [L39-41, L78-82]",
         "48,853,994,112 bytes (45.50 GiB) [L58]"])

repo(repo="tiyuvta/Qwen3.8-Flash-Next-NVFP4", sha="715444446b0df4fc91f6c901de45658704908a0c", fmt="NVFP4 (ModelOpt)",
  runtime="tiyuvta serving engine (bring-up, single request; not wired into server)", weights_modified=False,
  method="ModelOpt 0.46.0 NVFP4 (W4A4, FP8-e4m3 scales, group 16) on routed experts; calibration 256 x 512 tokens abisee/cnn_dailymail 3.0.0 seed 17; MTP block BF16 grafted byte-exact; everything else BF16",
  methodology=dict(reference="BF16 parent, same box, same prompts", dataset="4 fixed probe prompts, last-token logits", ctx=None, tool="not stated", L=[40,42]),
  table_header_L=44,
  variants=[dict(name="NVFP4 + BF16 MTP", size=None,
    m={"kl:merge two sorted linked lists":"0.0272","top20_overlap:merge two sorted linked lists":"18/20","argmax_same:merge two sorted linked lists":"yes",
       "kl:TCP vs UDP in two sentences":"0.0035","top20_overlap:TCP vs UDP in two sentences":"16/20",
       "kl:def fib(n) continuation":"0.0124","top20_overlap:def fib(n) continuation":"19/20",
       "kl:capital of Australia":"0.0355","top20_overlap:capital of Australia":"18/20","argmax_same:capital of Australia":"no"},
    L=46, L_extra=[47,48,49])],
  flags=["Quality evidence is 4 probe prompts, last-token logits only; argmax differs from BF16 on 1 of 4 ('capital of Australia') [L42-49]",
         "Card states the artifact is a single-request bring-up and not ready for multi-session serving [L68-76]",
         "No size stated on card"])

repo(repo="wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16", sha="0939125b929543a783ce700c90e36dd1a575c00c", fmt="compressed-tensors AWQ INT4 (W4A16)",
  runtime="vLLM (wtdcode/vllm-backport fork suggested for older GPUs)", weights_modified=False,
  method="AWQ weight-only INT4 symmetric g128 (pack-quantized) on routed experts; embeddings, lm_head, GDN, QSA + indexer, hyper-connections, router, shared expert, PLE incl. n-gram table, vision, MTP BF16",
  methodology=None,
  variants=[dict(name="AWQ-W4A16", size=None)],
  flags=["No quality claims for the quant (card read in full): card lists only what is quantized [L12-27]",
         "No size or calibration details stated on card"])

repo(repo="vaultai/Qwen3.8-Flash-Next-MLX-oQ4-MTP", sha="6b96f4c428c8c140aa5bbd7b7812748cc7f23b99", fmt="MLX",
  runtime="oMLX / MLX-VLM with qwen4_exp native-MTP support", weights_modified=False,
  method="oQ4 mixed precision: 4-bit affine g32 base with 232 protected modules at 5/8-bit, sensitivity from a 'validated local proxy'; rebuilt from official BF16 incl. MTP",
  methodology=None,
  variants=[dict(name="oQ4 MTP", size=113.33, unit="GB", basis="disk", L=72)],
  flags=["No quality claims for the quant (card read in full): only short-prompt speed tests and qualitative coherence notes [L106-114, L144-148]",
         "Card names the repository as Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP and credits Vontra; text and figures match TensorFold/Qwen3.8-Flash-Next-MLX-oQ4-MTP's card [L63, L92, L160]",
         "Card links an unresolved long-generation repetition report involving oQ4-MTP [L175]",
         "Weight size 113.33 GB, 105.54 GiB [L72]"])

repo(repo="zankich/Qwen3.8-Flash-Next-W4A16-Merlin-INT4PLE-FP8KV", sha="ba8fff329b872f4a5a68f6bc05129014737dc922", fmt="compressed-tensors W4A16 (GPTQ) + INT4 PLE + FP8 KV scales",
  runtime="zankich/vllm fork (int4 PLE plugin + qwen4_exp), TP=4 + expert parallel", weights_modified=False,
  method="Repack of halt95/Qwen3.8-Flash-Next-W4A16-Merlin (W4A16 GPTQ body, MTP bf16 unchanged) with n-gram PLE table converted FP8 -> INT4 g32 symmetric and calibrated per-layer static FP8 E4M3 KV scales (12 QSA layers, margin 1.20)",
  methodology=dict(reference="none for the 700-question run (absolute score); FP8-KV subset compared against same model at default KV dtype", dataset="MMLU-Pro 700 questions (5-shot, chat template); 112-question deterministic subset for FP8 KV", ctx=None, tool="lm-eval", L=[98,99,100,101,102,103]),
  variants=[dict(name="W4A16-Merlin + INT4 PLE + FP8 KV", size=95, unit="GB", basis="disk",
    m={"bench:MMLU-Pro (700 q, 5-shot)":"0.8486 ± 0.0133","bench:MMLU-Pro (112 q subset, FP8 KV)":"0.875 ± 0.031"}, L=98, L_extra=[99,101,82])],
  ladder=[dict(name="same checkpoint, default KV dtype (112 q subset)", measured=True, m={"bench:MMLU-Pro (112 q subset, default KV)":"0.8839"}, L=102)],
  cross=[dict(of="omlx 4-bit build of the base model (repo not identified on card)", variant="omlx 4-bit", measured=False, m={"bench:MMLU-Pro (300 q subset)":"0.843"}, L=99, L_extra=[100], flags=["quoted, not measured"])],
  flags=["MMLU-Pro on a 700-question sample; FP8-KV comparison on a 112-question subset (±0.031) [L98-103]",
         "No BF16 baseline; the omlx 4-bit figure is quoted from a published measurement on a different 300-question subset [L99-100]",
         "W4A16 body and MTP head are halt95's checkpoint unchanged; this repo adds INT4 PLE repack and FP8 KV scales [L19-25]",
         "95 GB total; INT4 PLE table 26.8 GB pinned in host RAM [L29-30, L82]"])

repo(repo="KaedeTai/Qwen3.8-Flash-Next-Uncensored-oQ4e-mtp-zhTW", sha="c4d04ba66d68b9b674730441ea2fbb01882547fc", fmt="MLX (per repo name)",
  runtime="not stated", weights_modified=True, method="not stated (card empty)",
  methodology=None,
  variants=[dict(name="Uncensored oQ4e-mtp zhTW", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes)",
         "weights_modified set from the repo name ('Uncensored'); card itself states nothing"])

repo(repo="msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47", sha="bb05ede97502dd0df53425acb3d3010e7d712f39", fmt="not stated",
  runtime="not stated", weights_modified=True, method="not stated (card empty)",
  methodology=None,
  variants=[dict(name="abliterated-cyber-GLP-47", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes)",
         "weights_modified set from the repo name ('abliterated'); card itself states nothing"])

repo(repo="orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8", sha="a9208847b98133d336315f062c156ba1f41c7588", fmt="FP8 (per repo name)",
  runtime="not stated", weights_modified=True, method="not stated (card empty)",
  methodology=None,
  variants=[dict(name="Uncensored FP8", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes)",
         "weights_modified set from the repo name ('Uncensored'); card itself states nothing"])

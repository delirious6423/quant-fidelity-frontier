# Re-extracted from verbatim README snapshots at pinned commits (snapshots/index.json -> cache/cards/, captured 2026-10-17).
# Queue: cache/autoq_4.json (previously regex-triaged only). Every card read in full.
# Each variant/claim cites README line numbers (L) at that sha.
# size basis: disk = full download unless noted; vram/resident/ram_in_use = memory figures; trunk_only = weights excluding n-gram table.
R = []
def repo(**k): R.append(k); return k

repo(repo="AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP8-MTP", sha="48b9aa5f002f29a9f8ab7cfe5c7185e342595b1b", fmt="MLX (AXQuant)",
  runtime="MLX-LM (text/backbone); oMLX/MTPLX MTP not established", weights_modified=False,
  method="AXQuant 1.9.0 mixed precision from BF16: 8bit affine g32 (97.95%) + bf16 (2.05%); MTP head BF16; allocation from architecture priors, no calibration",
  methodology=None,
  variants=[dict(name="AXQ-MXFP8", size=189.24, unit="GB", basis="disk", bpw=8.4092, L=62, L_extra=[60])],
  flags=["No quality claims for the quant (card read in full): card states quality versus BF16 is not published and no quality-retention claim is made [L42-43, L164]",
         "Measured main-model BPW 8.2977, total incl. MTP 8.4092; safetensors weight size 189.21 GB [L59-61]",
         "Card says MTPLX MXFP8 export is blocked and native MTP generation not established for this pack [L25-29]"])

repo(repo="BlivionIaG/Qwen3.8-Flash-Next-EXL3-3bpw-3inst-uncalibrated", sha="5c7f0eb2f184916342a4219378432a62905d27df", fmt="EXL3",
  runtime="opengfx1030/vllm-rdna fork (test target)", weights_modified=False,
  method="EXL3 trellis, 3inst codebook, input Hadamard only, Viterbi encode without LDLQ (uncalibrated); experts K=3, dense group K=6, index_qk_proj bf16",
  methodology=None,
  variants=[dict(name="EXL3 3inst ~3bpw (uncalibrated)", size=None)],
  flags=["No quality claims for the quant (card read in full): card labels it uncalibrated WIP, 'not a quality release', outputs not representative [L20]",
         "No size stated on card"])

repo(repo="Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS", sha="f48e9785cc255af99406940a4c560caac125a68c", fmt="GGUF (bundle)",
  runtime="R9V dual-RDNA4 profile (llama.cpp-based)", weights_modified=False,
  method="Repackaged bundle: unsloth UD-IQ4_XS target shards + ggml-org Q8_0 vision projector + R9V-assembled MTP checkpoint (BF16 dense, FP8 experts); nothing requantized by R9V",
  methodology=None,
  variants=[dict(name="UD-IQ4_XS (unsloth, rebundled)", size=None)],
  flags=["No quality claims for the quant (card read in full): only provenance and a link to an external performance qualification report [L51-56]",
         "Not an original quant: target GGUF is unsloth UD-IQ4_XS [L20-21]",
         "PLE payload stated as 26.82 GiB inside target shard 2 [L36]"])

repo(repo="GCSA-AiLab/Qwen3.8-Flash-Next-FP8-Abliterated-GGUF", sha="881b6c5027fc5a1b2ca94aaf63f77187b710e2d8", fmt="GGUF",
  runtime="llama.cpp", weights_modified=True,
  method="Abliterated (abliterix, refusal-behavior changes) derivative, GGUF Q4_KM / Q6_K / Q8; details in linked docs not on card",
  methodology=None,
  variants=[dict(name="Q4_KM", size=None), dict(name="Q6_K", size=None), dict(name="Q8", size=None)],
  flags=["No quality claims for the quant (card read in full): card is only links to per-language docs and download folders [L36-42]",
         "Weights modified: abliterated / refusal behavior altered [L44]",
         "No sizes stated on card"])

repo(repo="Jon-Nielsen/Qwen3.8-Flash-Next-AWQ-INT4-FP8PLE", sha="c2aa55f204e110787115f02c8bf459e3b8de9519", fmt="compressed-tensors AWQ INT4 + FP8 PLE",
  runtime="wtdcode/vllm-backport fork (with ple_layer.py overlay)", weights_modified=False,
  method="Repack of cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4 (routed experts AWQ INT4 asym g32) with the n-gram PLE table converted BF16 -> FP8 e4m3, single global bf16 scale",
  methodology=dict(reference="BF16 PLE shard (weight-level comparison, serving/verify_fp8_ple.py)", dataset="not stated (weight tensors, shard_0)", ctx=None, tool="serving/verify_fp8_ple.py", L=[55,56]),
  variants=[dict(name="AWQ-INT4 + FP8 PLE", size=127.72, unit="GiB", basis="disk",
    m={"ple_shard0_snr_db":"31.5 dB","ple_max_rel_err":"0.195"}, L=18, L_extra=[56])],
  flags=["Only fidelity figures are weight-level for the PLE FP8 conversion (shard_0 SNR, max relative error); no output-level quality numbers for the model [L55-56]",
         "Expert quantization is cyankiwi's AWQ-INT4, not re-done by this publisher [L26]",
         "PLE pinned-RAM footprint ~95 GiB -> ~48 GiB with CPU offload [L16]"])

repo(repo="Jundot/Qwen3.8-Flash-Next-oQ3.5e-fp16-mtp", sha="42ce563fa93537d3e847328d6209d00acd7cea8a", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ3.5e (oMLX v0.7.1): sensitivity-based bit allocation + imatrix (1024x512) + weighted LSQ scale/bias refit; FP16 scales/unquantized tensors, vision FP32; n-gram table 4-bit outside budget",
  methodology=None,
  variants=[dict(name="oQ3.5e-fp16", size=97.132, unit="GB", basis="disk", bpw=4.317, L=55, L_extra=[25,45])],
  flags=["No quality claims for the quant (card read in full): only bit distribution and storage tables [L31-55]",
         "4.045 bpw excluding n-gram table, 4.317 for full checkpoint; total 90.461 GiB [L25, L45]"])

repo(repo="Jundot/Qwen3.8-Flash-Next-oQ6e-mtp", sha="816d1ebafce205afabe91f300698cec0829b6de4", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ6e (oMLX v0.7.1): sensitivity-based bit allocation + imatrix (1024x512) + weighted LSQ scale/bias refit",
  methodology=None,
  variants=[dict(name="oQ6e", size=150.549, unit="GB", basis="disk", bpw=6.691, L=48, L_extra=[23])],
  flags=["No quality claims for the quant (card read in full): only bit distribution and storage tables [L29-48]",
         "Total 140.210 GiB [L48]"])

repo(repo="LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58", sha="a6692a8b67991512ccf42d27cc179ebda2eafac5", fmt="MLX (mlx-serve mixed)",
  runtime="mlx-serve >= 26.9.1 (measured on 26.9.6 / MLX 0.32.2)", weights_modified=True,
  method="Quantized from orcarouter's uncensored BF16 weights; imatrix calibration on publisher's own AI session logs + official general corpus; per-layer greedy bit allocation: routed experts 3-bit g64 x70 groups + 4-bit g64 x26 (layers 0-45), 4-bit g64 layers 46-47 + MTP experts; attention/GDN/shared expert 6-bit g64; lm_head 8-bit; n-gram table 4-bit g32 separate mmap file",
  methodology=dict(reference="BF16 original (top-1 agreement)", dataset="600 held-out scoring positions, scoring prefixes 65~2,297 tokens; business acceptance 24 questions x 7 packs (tool orchestration / code / long docs, 8 each)", ctx=None, tool="mlx-serve pipeline held-out window scoring", L=[46,86,89]),
  table_header_L=86,
  variants=[dict(name="C58", size=92, unit="GB", basis="disk", mem=61, mem_unit="GB", mem_basis="resident",
    m={"top1_pct":"85.40%","acceptance:tool_orchestration":"3/8"}, L=87, L_extra=[62,69,90])],
  ladder=[dict(name="Ap73 (same-budget all-4-bit pack)", m={"top1_pct":"85.76%","top1_delta_pts":"+0.36"}, L=87)],
  cross=[dict(of="unnamed community 4-bit pack (not identified on card)", variant="4-bit", m={"top1_pct":"87.82%","top1_delta_pts":"−2.42"}, L=87)],
  flags=["Weights modified: built from orcarouter's uncensored BF16; card says no further behavior changes by this publisher [L45, L109]",
         "Top-1 agreement over only 600 positions with short prefixes (65~2,297 tokens); card says long-context quality not validated and top-1 can decouple from real task ability [L86, L108]",
         "Comparison 4-bit community pack is not named [L87]",
         "Card notes tool-orchestration acceptance 3/8 as a shared weakness of the packs tested [L90]",
         "Calibration corpus is the publisher's own AI agent session logs [L47]",
         "Size: ~92GB total; 59.83GB weights+vision plus 32,000,153,976-byte n-gram table; resident ~61GB after load [L62, L69]"])

repo(repo="Lucebox/Qwen3.8-Flash-Next-GGUF", sha="f048355c3482919e06cd90f3bcdf28c7c1502486", fmt="GGUF",
  runtime="Lucebox engine (luce_server)", weights_modified=False,
  method="Byte-identical mirror of unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL + MTP Q8_0 drafter at revision 766911a6",
  methodology=None,
  variants=[dict(name="UD-Q4_K_XL (unsloth mirror)", size=None), dict(name="MTP shared Q8_0", size=2786568256, unit="bytes", basis="disk", L=28)],
  flags=["No quality claims for the quant (card read in full): only Lucebox decode/prefill speed figures [L32-39]",
         "Not an original quant: byte-identical copies of unsloth files [L15]",
         "UD-Q4_K_XL shard sizes listed per file in bytes, no total stated [L24-27]"])

repo(repo="Nielk38/Qwen3.8-Flash-Next-GGUF-SPLIT", sha="b816449a52b8baa9036f0cc368e3995a7ce076e0", fmt="GGUF",
  runtime="llama.cpp (PR #27742)", weights_modified=False,
  method="Lossless re-sharding of unsloth UD-Q2_K_XL (llama-gguf-split merge + split 4900M); no requantization",
  methodology=None,
  variants=[dict(name="UD-Q2_K_XL (unsloth, re-split)", size=78869130144, unit="bytes", basis="disk", L=27)],
  flags=["No quality claims for the quant (card read in full): only a 2+2 smoke test and tensor checksum match [L96-98]",
         "Not an original quant: same tensors as unsloth UD-Q2_K_XL [L19-21]"])

repo(repo="RockTalk/Qwen3.8-Flash-Next-MLX-4bit", sha="0afead4d1507d2184a4a51d164bbd75a91b3c17c", fmt="MLX",
  runtime="mlx_qwen4exp (Rocktalk-Holdings/mlx-qwen4exp)", weights_modified=False,
  method="4-bit affine g64; embed_tokens/lm_head/router/shared_expert_gate 8-bit g64; MTP weights same policy; n-gram table shipped as float16",
  methodology=None,
  variants=[dict(name="MLX 4-bit g64", size=71, unit="GB", basis="trunk_only", L=172, L_extra=[177])],
  flags=["No quality claims for the quant (card read in full): only throughput, MTP acceptance and qualitative vision/tool tests [L66-73, L107-134]",
         "Card says running without the optional ~95 GB n-gram table 'slightly' reduces output quality, with no figure [L179-180]",
         "Size ~71 GB transformer weights excludes ~95 GB float16 n-gram table and ~1.4 GB MTP [L172-177]",
         "gmlx throughput section measures unsloth UD-Q4_K_XL GGUF, not this repo's MLX weights [L85-91]"])

repo(repo="Sawfwair/Qwen3.8-Flash-Next-MLX-Activation-3bit-Native-PLE", sha="1cee9301c745836e0abb8933e89cf27a38b98125", fmt="MLX",
  runtime="mere.run (> v0.45.0) / mlx-vlm", weights_modified=False,
  method="Routed experts fresh MLX affine Q3 g64 from BF16 with scales/biases refit against image+text expert-input second moments; other core/MTP/vision Q4; n-gram table Q4 g32",
  methodology=dict(reference="publisher's pinned published Q4 checkpoint (no-regression comparison), not BF16", dataset="61 cases by exact expected output + sealed 16-case holdout (8 image/OCR)", ctx=None, tool="mere.run on 128 GiB Apple Silicon Mac, MTP disabled", L=[73,74,75,77,79]),
  variants=[dict(name="Activation-weighted 3-bit Native PLE", size=83.49, unit="GiB", basis="disk", mem=62065713472, mem_unit="bytes", mem_basis="resident",
    m={"cases_exact":"58 of 61","holdout_identical_to_Q4":"all 16"}, L=29, L_extra=[77,79,81])],
  ladder=[dict(name="published Q4 checkpoint", mem=77180432496, m={"holdout_peak_memory_bytes":"77,180,432,496","mem_reduction_pct":"19.58%"}, L=82)],
  flags=["Reference is the publisher's own Q4 checkpoint, not BF16; card itself says the bounded suites don't establish general quality or BF16 parity [L73-74, L85-86]",
         "Small evals: 61 cases + 16-case holdout [L77, L79-80]",
         "Memory figures are holdout peak footprint [L81-82]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-2bit-MTP", sha="28b109043ec5576f8ad5a3192a66467915c76736", fmt="MLX",
  runtime="oMLX 0.6.3rc3 / MLX 0.32.0 / MLX-VLM 0.6.3", weights_modified=False,
  method="Non-sensitivity affine 2-bit g32 for routed+shared experts and n-gram embedding paths; attention/GDN/routing/vision/embeddings/head/MTP kept BF16",
  methodology=None,
  variants=[dict(name="MLX 2-bit g32 + BF16 exclusions", size=80.070, unit="GB", basis="disk", L=78)],
  flags=["No quality claims for the quant (card read in full): only throughput, MTP acceptance and pass/fail smoke gates (instruction, factual, arithmetic, coherence) with no scores [L111-120, L153]",
         "Weight size 80.070 GB / 74.571 GiB [L78]",
         "Card says visual quality not benchmarked [L166]"])

repo(repo="TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP", sha="0a1bf6faa0a2f16a314da889e8e6c80226455dbf", fmt="MLX",
  runtime="oMLX (app version not recorded) with qwen4_exp native-MTP support", weights_modified=False,
  method="Sensitivity-aware oQ2 mixed precision: 2-bit affine g32 base with 746 overrides (297 @3-bit, 60 @4, 36 @5, 129 @6, 224 @8)",
  methodology=None,
  variants=[dict(name="oQ2", size=70.782, unit="GB", basis="disk", L=80)],
  flags=["No quality claims for the quant (card read in full): only throughput, MTP acceptance and pass/fail smoke checks with no scores [L113-122, L151-156]",
         "Weight size 70.782 GB / 65.921 GiB [L80]"])

repo(repo="ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit", sha="f1280695600cc302914640c7ddb9eb07a4d5aef1", fmt="MLX",
  runtime="mlx-vlm 0.6.17 (MTP drafter needs >= 0.7.0)", weights_modified=False,
  method="mlx_vlm convert -q --q-bits 4 --q-group-size 32 from official bf16 (revision de4b8e4); n-gram table included in quantization",
  methodology=None,
  variants=[dict(name="MLX 4-bit g32", size=104, unit="GiB", basis="disk", bpw=5.029, L=18)],
  flags=["No quality claims for the quant (card read in full): only conversion command and usage [L14-52]"])

repo(repo="Wladastic/Qwen3.8-Flash-Next-Ombro-v13b-GGUF", sha="4b20669b42c68b677f22e613ce01997a5baa579b", fmt="GGUF",
  runtime="llama.cpp", weights_modified=False,
  method="Mixed-precision per-tensor override recipe (successor of Ombro V9): most routed experts Q4_K/Q4_K/Q5_1, selected layers Q6_K/Q6_K/Q8_0; attention/SSM/hyper-connection/PLE projections BF16; shared experts and embed/output Q8_0; PLE table IQ4_NL",
  methodology=None,
  variants=[dict(name="Ombro V13b", size=118.9, unit="GB", basis="disk", bpw=5.38, L=27, L_extra=[59])],
  flags=["No quality claims for the quant (card read in full): only size/BPW and per-tensor type comparison against UD-Q4_K_XL, Ombro V9 and UD-Q5_K_XL [L23-59]",
         "Comparison table lists sizes/BPW only for other builds (UD-Q4_K_XL 111.3 GB, V9 114.5 GB, UD-Q5_K_XL 158.3 GB), no quality figures [L25-28]"])

repo(repo="Zalypamoy91/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF", sha="8894dadc80e132d25db4e4ca67cbe90380e1adc2", fmt="GGUF",
  runtime="llama.cpp", weights_modified=True,
  method="Card describes an abliterated (refusal-removed) version; states GGUFs come from unsloth/Qwen3.8-Flash-Next-GGUF",
  methodology=None,
  variants=[dict(name="UD-Q4_K_XL", size=None)],
  flags=["No quality claims for the quant (card read in full): only abliteration description, usage command and warnings [L21-45]",
         "Card title names huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF while the repo is Zalypamoy91's [L18]",
         "Card says the GGUFs come from unsloth/Qwen3.8-Flash-Next-GGUF while also describing the model as abliterated [L21, L25]",
         "Weights modified: abliterated / uncensored [L21]"])

repo(repo="amd/Qwen3.8-Flash-Next-EXTENSOR", sha="c55e64bd301b932917f46a72a6a8c2ee23ba0560", fmt="EXTENSOR GGUF (not llama.cpp compatible)",
  runtime="EXTENSOR 3.0.0", weights_modified=False,
  method="ROCmFP4 experts; F16/F32 residents; BF16 n-gram table; text model only",
  methodology=None,
  variants=[dict(name="ROCmFP4 EXTENSOR", size=176688558080, unit="bytes", basis="disk", L=26)],
  flags=["No quality claims for the quant (card read in full): only file, size, hash and quantization summary [L23-30]",
         "Vision and MTP weights not included [L21]"])

repo(repo="auggie246/Swift-1.5-Qwen3.8-Flash-Next-W4A16-FP8PLE", sha="aecff0758ff8078916d2a0e516f5034c8e77a983", fmt="AutoRound W4A16 (auto_gptq) + FP8 PLE",
  runtime="patched vLLM (DominikBucko/qwen38-flash-next-2x3090)", weights_modified=True,
  method="Repack of ukisai Swift 1.5 (post-trained) AutoRound W4A16 INT4 sym g128 backbone; PLE table replaced by RadixArk FP8 E4M3FN table; MTP replaced by INT4 g32 draft built from base MTP; no backbone requantization",
  methodology=None,
  variants=[dict(name="Swift 1.5 W4A16 + FP8 PLE + MTP3", size=112.497, unit="GiB", basis="disk", L=57)],
  flags=["No quality claims for the quant (card read in full): card states there are no speed or quality measurements yet [L33-35]",
         "Weights modified: Swift 1.5 is a post-trained (fine-tuned) backbone [L39]",
         "Card refers to the upstream Swift card for evaluation results [L83-84]",
         "86–91% draft acceptance cited is measured on the base target, not this checkpoint [L72-73]",
         "Size is tensor payload 112.497 GiB [L57]"])

repo(repo="boxrunner/Qwen3.8-Flash-Next-IQ3_S-GGUF", sha="ffd258d863133f98e0216ba55fc163b682663196", fmt="GGUF",
  runtime="llama.cpp >= 73a43d1f / Boxrunner", weights_modified=False,
  method="boxrunner map C with routed-expert gate/up requantized to IQ3_S from own Q8_0 conversion; down IQ4_NL, n-gram IQ4_NL, rest Q5_K/Q6_K/Q8_0; AtomicChat imatrix",
  methodology=None,
  variants=[dict(name="IQ3_S-expert (map C + IQ3_S gate/up)", bpw=4.081, size=45950320320, unit="bytes", basis="disk", L=32, L_extra=[18,33])],
  flags=["No quality claims for the quant (card read in full): card states 'No quality benchmark is reported here' [L87]",
         "Model shards 45,950,320,320 + 44,312,166,688 bytes; experts 53.3 GiB [L25, L32-33]",
         "Card says it does not fit llama.cpp GPU memory on a 64 GB Mac [L51-53]"])

repo(repo="carsenk/Qwen3.8-Flash-Next-ROGUE-NVFP4", sha="dbd1690be71afd4a213f6790f33a62a5388003a7", fmt="NVFP4 (ModelOpt)",
  runtime="vLLM source + included runtime/ compatibility modules, b12x 1.2.6", weights_modified=True,
  method="ROGUE residual-stream refusal-direction intervention (32 tensors in 8 layers) then ModelOpt 0.46.0 activation-calibrated W4A4 NVFP4 g16; PLE table packed NVFP4; native MTP BF16; text only",
  methodology=None,
  variants=[dict(name="ROGUE NVFP4", size=111858152472, unit="bytes", basis="disk", L=50)],
  flags=["No quality claims for the quant (card read in full): only coherent-generation and refusal smoke tests (pass/fail) and throughput [L66-68]",
         "Weights modified: refusal-direction intervention on 32 tensors [L26-33]",
         "Size is indexed tensor bytes [L50]",
         "Runtime-tested only at 4,096 tokens [L107]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ2.5e-mtp", sha="c9fd3c3e919c7d99b6b74b5f9d636fe84e030231", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQe (oMLX v0.7.0) imatrix-weighted mixed precision, default 2-bit g64; imatrix oqe_code_multilingual 1,024x512; sensitivity map measured on Jundot oQ4e-mtp (proxy)",
  methodology=None,
  variants=[dict(name="oQ2.5e", size=71, unit="GB", basis="disk", bpw=3.15, L=31)],
  flags=["No quality claims for the quant (card read in full): only quantization details, bit-width shares / imatrix and sensitivity notes [L38-57]",
         "Size and bpw stated as approximate ('~3.15 bits per weight, 71 GB') [L31]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ3.5-mtp", sha="c6d197df4e6126fd709aa30d3d1e7f7ee2dcf4e8", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ (oMLX v0.7.0) mixed precision, default 3-bit g64; sensitivity map measured on Jundot oQ4e-mtp (proxy)",
  methodology=None,
  variants=[dict(name="oQ3.5", size=90, unit="GB", basis="disk", bpw=3.99, L=27)],
  flags=["No quality claims for the quant (card read in full): only quantization details, bit-width shares / imatrix and sensitivity notes [L34-46]",
         "Size and bpw stated as approximate ('~3.99 bits per weight, 90 GB') [L27]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ5-mtp", sha="da3149c350418910edd4f653ed9e12f8b372a438", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQ (oMLX v0.7.0) mixed precision, default 5-bit g64; sensitivity map measured on Jundot oQ4e-mtp (proxy)",
  methodology=None,
  variants=[dict(name="oQ5", size=129, unit="GB", basis="disk", bpw=5.71, L=27)],
  flags=["No quality claims for the quant (card read in full): only quantization details, bit-width shares / imatrix and sensitivity notes [L34-44]",
         "Size and bpw stated as approximate ('~5.71 bits per weight, 129 GB') [L27]"])

repo(repo="d9beuD/Qwen3.8-Flash-Next-oQ8e-mtp", sha="3db9c562d88b4b19ecc544070091afc2fd8e9cf2", fmt="MLX",
  runtime="oMLX", weights_modified=False,
  method="oQe (oMLX v0.7.0) imatrix-weighted, 8-bit g64, n-gram table 8-bit g32; imatrix oqe_code_multilingual 1,024x512; sensitivity map measured on Jundot oQ4e-mtp (proxy)",
  methodology=None,
  variants=[dict(name="oQ8e", size=195, unit="GB", basis="disk", bpw=8.66, L=30)],
  flags=["No quality claims for the quant (card read in full): only quantization details, bit-width shares / imatrix and sensitivity notes [L37-46]",
         "Size and bpw stated as approximate ('~8.66 bits per weight, 195 GB') [L30]"])

repo(repo="densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE", sha="705a2ce7f529f61bcd94723bcde66866e0b63286", fmt="AutoRound W4A16 (GPTQ packing) + FP8 PLE",
  runtime="patched vLLM overlay (DominikBucko/qwen38-flash-next-2x3090)", weights_modified=True,
  method="Reassembly: trohrbaugh heretic-2 AutoRound W4A16 INT4 sym g128 target unchanged + RadixArk FP8 E4M3FN PLE table + INT4 g32 MTP draft; no target tensor requantized",
  methodology=None,
  variants=[dict(name="heretic-2 W4A16 + FP8 PLE + MTP3", size=116.183, unit="GiB", basis="disk", L=101, L_extra=[100])],
  flags=["No quality claims for the quant (card read in full): only throughput figures, measured on the upstream albucino checkpoint, not this one [L72-75, L133-134]",
         "Card states quality/benchmarks for this heretic-2 assembly have not been re-measured [L195-196]",
         "Weights modified: heretic-2 is a modified (heretic) derivative of the base model [L34]",
         "Size is target tensor payload 124,750,778,874 bytes (116.183 GiB), plus 3.855 GiB MTP draft [L100-103]"])

repo(repo="edougawa/Qwen3.8-Flash-Next-NVFP4-PLE-MMap-MTP-Spark", sha="c3e61234a968cca449116b44d068997e661e8131", fmt="NVFP4 (ModelOpt W4A16_NVFP4)",
  runtime="upstream vLLM 7c5dc571 + qwen38-nvfp4-mmap-ple plugin (DGX Spark, TP=1)", weights_modified=False,
  method="ModelOpt W4A16_NVFP4 block 16 for GEMMs (experts, shared expert, QSA, GDN projections, PLE k/v proj, MTP linears, lm_head); BF16 for vision/embeddings/norms/routers; PLE table packed NVFP4 outside index, mmap",
  methodology=dict(reference="BF16 source tensors (weight-level cosine)", dataset="PLE: 10000 random rows; body: all quantized tensors (per validation/body-nvfp4-validation.json)", ctx=None, tool="publisher validators (validation/*.json)", L=[84,88]),
  table_header_L=80,
  variants=[dict(name="NVFP4 body + mmap NVFP4 PLE", size=69.9, unit="GiB", basis="trunk_only",
    m={"ple_row_cosine":"cosine 0.995540","body_cosine_mean":"cosine mean 0.995620","body_cosine_min":"min 0.995146"}, L=88, L_extra=[84,47])],
  flags=["Only fidelity figures are weight-level cosine similarities (PLE rows, body tensors); no output-level quality (KLD/PPL/benchmarks) [L84, L88]",
         "Card states nothing has been booted on GB10; 32K instance test not run, long-context and memory envelope pending [L92-101]",
         "Size: 69.9 GiB model shards plus 26.822 GiB external PLE table [L47-48]"])

repo(repo="gabmiranda/Qwen3.8-Flash-Next-oQ2e-mtp", sha="9a235a251ffdb90a48c466382e09ea446d096534", fmt="MLX",
  runtime="oMLX", weights_modified=False, method="oQ2e (per name/tags only)",
  methodology=None, variants=[dict(name="oQ2e", size=None)],
  flags=["No quality claims for the quant (card read in full): card is only front matter and a title [L1-11]"])

repo(repo="gabmiranda/Qwen3.8-Flash-Next-oQ6e-mtp", sha="0d4741fe4be65f8ce6cf3fd3a2febf6de8dc0148", fmt="MLX",
  runtime="oMLX v0.7.0", weights_modified=False, method="oQ (oMLX v0.7.0) mixed precision, 6 bits, group size 64",
  methodology=None, variants=[dict(name="oQ6e", size=None)],
  flags=["No quality claims for the quant (card read in full): only quantization details (bits, group size) [L13-20]", "No size stated"])

repo(repo="groxaxo/Qwen3.8-Flash-Next-exl3-3.50bpw_hq_h6_ng6", sha="5ebeaa02729de4195d54e63984d6ebd787e49c00", fmt="EXL3",
  runtime="ExLlamaV3 1.5.0 / TabbyAPI (tensor_parallel unsupported)", weights_modified=False,
  method="EXL3 3.50 bpw HQ (mul1 codebook, out_scales always), head 6, n-gram 6, MTP 4, vision 6; calibration 250x2048; cut of turboderp 4.05bpw_h6_ng6 recipe",
  methodology=None,
  variants=[dict(name="3.50bpw_hq_h6_ng6", size=93, unit="GB", basis="disk", bpw=3.52, L=40, L_extra=[41])],
  flags=["No quality claims for the quant (card read in full): only load ceilings, throughput and MTP acceptance; card states these are not a quality score vs BF16 [L86-169]",
         "Size 93 GB includes 37 GB n-gram table [L40]"])

repo(repo="hstolte/Qwen3.8-Flash-Next-HALOLOOM-INT4-INT8-AWQ", sha="dadedd9623bb6ae2336825f6a3a02ba618d54bda", fmt="Quark mixed INT4/INT8 (AWQ)",
  runtime="HawgAuto/vllm fork, HaloLoom gfx1151 image", weights_modified=False,
  method="AMD Quark mixed INT4/INT8 routed experts; two builds: BF16 core or INT8 core; BF16 PLE table on SSD; MTP draft",
  methodology=None,
  variants=[dict(name="Bf16Core-int4-int8", size=188.84, unit="GB", basis="disk", mem=76.85, mem_unit="GiB", mem_basis="resident", L=60, L_extra=[82]),
            dict(name="int8Core-int4-int8", size=184.64, unit="GB", basis="disk", mem=73.05, mem_unit="GiB", mem_basis="resident", L=60, L_extra=[82])],
  flags=["No quality claims for the quant (card read in full): card states independent model-quality evaluation is not claimed [L215]",
         "Only serving checks (1M-token needle recalled on int8 ordinary) and throughput [L137, L217-222]",
         "Memory figures are model-load allocations at 4,096 ctx, not peak [L82-85]"])

repo(repo="immiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8", sha="4c1ffc278db2c11437ed0bb9050f834714e063db", fmt="Orinfer safetensors (E8P)",
  runtime="Orinfer (Jetson AGX Orin 64GB)", weights_modified=False,
  method="Routed experts near-2-bit E8P (8-weight vectors in 16-bit codes) with signed Hadamard rotation, INT8 compute; dense/shared/head W8; vision FP16; weight-only, seed 20261002",
  methodology=None,
  variants=[dict(name="E8P Q2A8", size=51.3, unit="GB", basis="disk", mem=34.52, mem_unit="GiB", mem_basis="vram", L=37, L_extra=[38])],
  flags=["No quality claims for the quant (card read in full): card states a full LLM quality benchmark remains to be added; only prefill/decode speed [L40, L58-80]",
         "Download size approximate (~51.3 GB); GPU read-only weights ~34.52 GiB [L37-38]"])

repo(repo="intel-ai/Qwen3.8-Flash-Next-MXFP4-BF16Attn-MXFP8MTP-CT-AutoRound", sha="781020389a524aba5019605faf4e58e4635bcdab", fmt="compressed-tensors (MXFP4/MXFP8)",
  runtime="vLLM >= 0.29.0 (evaluated on 0.29.1rc1.dev528 nightly), TP=1, no patches", weights_modified=False,
  method="AutoRound 0.16.0 --model_free (RTN, no calibration): main routed experts MXFP4 W4A4 g32 UE8M0 dynamic act; MTP routed experts MXFP8 W8A8 g32; attention, GDN, shared experts, PLE table, embeddings, head, vision BF16",
  methodology=dict(reference="bf16 Qwen/Qwen3.8-Flash-Next (same table; card does not state whether re-run or copied)", dataset="GSM8K (chat template + few-shot multiturn), MMLU, PIQA, HellaSwag; MTP check GSM8K limit-100",
    ctx=8192, tool="lm-eval 0.4.13 + vLLM 0.29.1rc1.dev528, TP=1, bf16 KV, seed=42, max_num_seqs=64, enable_thinking=false", L=[78,79,80,71]),
  table_header_L=82,
  variants=[dict(name="MXFP4 experts + MXFP8 MTP experts, BF16 attn", row_label="This checkpoint", size=180.1, unit="GB", basis="disk",
    m={"bench:GSM8K strict":"0.9651","bench:GSM8K flexible":"0.9644","bench:MMLU":"0.8655","bench:PIQA acc":"0.8243","bench:PIQA acc_norm":"0.8330","bench:HellaSwag acc":"0.6845","bench:HellaSwag acc_norm":"0.8708"},
    L=86, L_extra=[26])],
  baseline={"source":"bf16 baseline","bench:GSM8K strict":"0.9674","bench:MMLU":"0.8652","bench:PIQA acc":"0.8194","bench:HellaSwag acc":"0.6927","L":84},
  ladder=[dict(name="Sibling -BF16Attn-1007 (MTP fully bf16)", m={"bench:GSM8K strict":"0.9644","bench:GSM8K flexible":"0.9636","bench:MMLU":"0.8641","bench:PIQA acc":"0.8248","bench:PIQA acc_norm":"0.8308","bench:HellaSwag acc":"0.6856","bench:HellaSwag acc_norm":"0.8726"}, L=85),
          dict(name="This checkpoint with MTP enabled (num_speculative_tokens 1)", m={"bench:GSM8K limit-100":"0.98"}, L=71)],
  flags=["Saturated/easy benchmarks only (GSM8K, MMLU, PIQA, HellaSwag), thinking disabled, max_model_len 8192 [L78-80, L82-86]",
         "BF16 baseline row has only one metric per benchmark (flexible / acc_norm shown as —) [L84]",
         "Card states all deltas vs bf16-MTP sibling are <= ~0.3 σ; MTP weights are not loaded without speculative decoding [L88-90]",
         "MTP check is GSM8K limit-100 only; acceptance/throughput not validated [L71-73]",
         "Only size quoted for nvidia NVFP4 (132.7 GB), no quality comparison [L105-106]",
         "Long-context (128k) not validated; validated B300, ctx <= 8k [L115-116]"])

repo(repo="jedisct1/Qwen3.8-Flash-Next-oQ4e-100K-MTP", sha="a7b39c732acd1f2f1dd06d45b95d087edee9a29d", fmt="MLX",
  runtime="oMLX 0.6.3", weights_modified=False,
  method="oQ4e affine g64 with strict oQe imatrix (937 modules, 1,024x512); PLE table 4-bit g32; uncalibrated experts in base layers 0-1 kept Q8; lm_head, hyper-connection mixers, MTP fusion projections BF16",
  methodology=dict(reference="none for task tests (no BF16 run); Q8 vs Q4 reconstruction error measured on official BF16 rows", dataset="Swival tool-calling suite: 50 positive cases + 10 adversarial guards; long-context retrieval markers at ~100K tokens", ctx=100000, tool="Swival via oMLX OpenAI-compatible API", L=[41,42,143,146,147,122,123]),
  variants=[dict(name="oQ4e (MTP disabled)", size=108816636376, unit="bytes", basis="disk",
    m={"tool_calls:positive":"all 50 positive cases passed","tool_calls:adversarial_guards":"all 10 adversarial guards"}, L=49, L_extra=[146,147]),
    dict(name="oQ4e (MTP enabled)", m={"tool_calls:positive":"all 50 positive cases still passed","tool_calls:adversarial_guards":"9 of 10 guards","mtp_accepted":"10,631 of 12,152"}, L=148, L_extra=[149,151])],
  flags=["Quality evidence is a functional tool-calling suite (50 positive + 10 guards) and needle-style retrieval, not KLD/PPL/standard benchmarks; no BF16 comparison [L146-151, L122-124]",
         "Q8 'reduced normalized reconstruction error by 164.46 times compared with Q4' applies only to the affected uncalibrated expert rows [L41-42]",
         "With MTP enabled one speculative run changed an exact tool argument (9/10 guards) [L148-150]",
         "Recommended 100,000-token context on 128 GiB Mac; peak memory figures 87.84–99.96 GiB [L101-120]"])

repo(repo="kingjones777/Qwen3.8-Flash-Next-ROCmFP4-FAST-GGUF", sha="7565b1e84aaac1243166a3e7a2f45e2878a71f2a", fmt="GGUF (ROCmFP4)",
  runtime="kingjones30/ROCmFPX fork (qwen4exp + ROCmFP4); stock llama.cpp cannot load", weights_modified=False,
  method="Q4_0_ROCMFP4_FAST (file_type 103): experts, shared expert, attention, token embeddings and PLE table all ROCmFP4 (TYPE_101, 4.251 bpw); output head Q6_K; from own BF16 conversion",
  methodology=None,
  variants=[dict(name="ROCmFP4 FAST", size=87.94, unit="GiB", basis="disk", bpw=4.27, mem=63.3, mem_unit="GiB", mem_basis="vram", L=134, L_extra=[216])],
  flags=["No quality claims for the quant (card read in full): card states 'Quality is the one thing I have not measured, so I won't claim it' [L297-298]",
         "Only throughput, MTP acceptance (measured on a different Uncensored build) and memory residency figures [L54-56, L74-83, L256-263]",
         "Memory figure is GTT resident on unified-memory Strix Halo at 8K ctx (63.3–74.9 GiB across depths) [L216, L263]"])

repo(repo="lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448", sha="8d565c90add22b39cf3632943f859d7a9a411b48", fmt="NVFP4 (ModelOpt)",
  runtime="vLLM >= 0.30.0 / TRT-LLM / ModelOpt NVFP4 paths (num_experts=448)", weights_modified=True,
  method="REAP expert pruning (keep 448 of 512 routed experts per layer) of nvidia/Qwen3.8-Flash-Next-NVFP4; kept weights bit-identical; calibration 16.49M tokens of the publisher's production agentic traffic",
  methodology=dict(reference="unpruned expert set (REAP saliency mass only; no output-level comparison)", dataset="16.49M tokens of production traffic (681 LLM calls, 626 unique images)", ctx=None, tool="REAP reference-semantics tooling", L=[39,40,41]),
  table_header_L=24,
  variants=[dict(name="NVFP4-REAP-k448", size=115.6, unit="GiB", basis="disk", m={"reap_mass_retained_pct":"92.65% retained"}, L=28, L_extra=[29,47])],
  baseline={"size_gib_unpruned_nvidia_nvfp4":"123.6 GiB","L":28},
  flags=["Weights modified: pruned (64 of 512 routed experts removed per layer) [L18-19]",
         "Only fidelity figure is retained REAP saliency mass (92.65%; mean 7.35% removed per layer, worst 9.11%), a calibration proxy — no output-level quality numbers [L29, L47]",
         "Card recommends QA diff vs unpruned baseline before production use [L64-65]",
         "Pruned at 48% calibration coverage (16.49M of 34.2M tokens) [L62]"])

repo(repo="matteiuspi/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i", sha="27e5cdcbbcd147fed3fa5a44a90bd41cce8d5371", fmt="Ascend ModelSlim W8A8_DYNAMIC (ascend_v1)",
  runtime="OpenSensor vLLM + vLLM Ascend forks (custom dev runtime), 4x Ascend 310P3 TP4", weights_modified=False,
  method="ModelSlim W8A8_DYNAMIC: routed-expert weights INT8 per-output-channel (FP32 scales/offsets), dynamic per-token INT8 activations in grouped matmul; embeddings, PLE, attention, GDN, shared experts, vision, lm_head FP16",
  methodology=dict(reference="BF16 reference (conversion-time NLL/PPL); RTX 6000 PRO llama.cpp UD-IQ4_XS run (GPQA)",
    dataset="NLL/PPL: 28,245 next-token predictions on a paired text calibration set; GPQA Diamond first five items (AISBench 19018e9c, dataset b1ed2c, temp 0, seed 1024, 8,192-token output ceiling); seven-item semantic/tool smoke check",
    ctx=None, tool="ModelSlim QDQ reference (NLL/PPL); AISBench (GPQA)", L=[315,316,331,332,333,334,254]),
  table_header_L=318,
  variants=[dict(name="W8A8 Dynamic", size=240.06, unit="GB", basis="disk",
    m={"mean_nll":"2.5046","ppl":"12.2393","bench:GPQA Diamond first-5 correct":"5 / 5","bench:GPQA Diamond first-5 acc":"100%","smoke:semantic_tool_7":"6/7"},
    L=321, L_extra=[204,338,339,254])],
  baseline={"source":"BF16 reference","mean_nll":"2.5476","ppl":"12.7768","L":320},
  cross=[dict(of="unsloth/Qwen3.8-Flash-Next-GGUF", variant="UD-IQ4_XS (RTX 6000 PRO llama.cpp)", m={"bench:GPQA Diamond first-5 correct":"5 / 5","bench:GPQA Diamond first-5 acc":"100%","bench:GPQA Diamond (198)":"138/198, 69.70%"}, L=338, L_extra=[339,360])],
  ladder=[dict(name="Companion W4 G128 checkpoint (matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i)", measured=False, m={"bench:GPQA Diamond (198)":"140/198 (70.71%)","bench:GPQA Diamond contains correct answer":"114/198"}, L=97, L_extra=[98,99], flags=["quoted, not measured"])],
  flags=["PPL/NLL is a W8A8 QDQ (simulated) reference on the calibration set, 28,245 predictions; card itself says the lower quantized PPL is not evidence of improvement and held-out evaluation is required [L315-325]",
         "GPQA result for this checkpoint is only the first five (non-random) items; card calls it a pipeline smoke check, too small to estimate quality; full 198-item W8 replay pending [L331-348, L361]",
         "Smoke check matched control at 6/7; outputs not bit-exact [L254-257]",
         "198-item GPQA figures are for the companion W4 checkpoint and the RTX UD-IQ4_XS reference, not this W8A8 checkpoint [L95-103, L360]",
         "RTX IQ4_XS reference is given as 138/198 (69.70%) in the frozen-setting table but the W4 section says W4's 140/198 was 'equal to a matched RTX 6000 Pro IQ4_XS run' [L97-98, L360]",
         "Export ~240.06 GB (223.57 GiB) because PLE table and non-expert state remain FP16 [L204-205, L431-432]"])

repo(repo="meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest", sha="13b21e10a6c83397f6cb6fd637ba48beb975cc62", fmt="NVFP4 (ModelOpt, per HF tags)",
  runtime="SGLang (per HF tags)", weights_modified=False, method=None,
  methodology=None, variants=[dict(name="NVFP4", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes); format/runtime from HF tags only"])

repo(repo="mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp", sha="2a9de025436ea977720e4e3d6f369b185838e791", fmt="MLX",
  runtime="oMLX 0.6.4", weights_modified=False,
  method="oQe (imatrix) via oMLX v0.6.4, 8 bits, group size 64, preserve_mtp (76 mtp.* tensors)",
  methodology=None, variants=[dict(name="oQ8e", size=None)],
  flags=["No quality claims for the quant (card read in full): only speed benchmark vs Jundot oQ4e and an oQ6e pack (time to completion, tok/s) [L28-64]",
         "No size stated"])

repo(repo="myllmbox/Qwen3.8-Flash-Next-hibrid47-uncensored", sha="e2edbc409cb477308861f173a4fbfd8c7ece43cf", fmt="compressed-tensors NVFP4/FP8 (per HF tags)",
  runtime="vLLM (per HF tags)", weights_modified=True, method=None,
  methodology=None, variants=[dict(name="hibrid47", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes); format from HF tags only",
         "Weights modified per HF tags 'abliterated', 'uncensored' and repo name"])

repo(repo="nopmobiel/Qwen3.8-Flash-Next-MLX-4bit", sha="8577fa9a910c8af835254cf19b5330ebe60fbe0a", fmt="MLX",
  runtime="mlx-vlm with bundled qwen4_exp module; LM Studio with compatibility patch", weights_modified=False,
  method="Affine 4-bit g64 language weights; PLE n-gram table 4-bit g32; MoE routing gates 8-bit g64; vision BF16; no MTP",
  methodology=None,
  variants=[dict(name="MLX 4-bit g64", size=None, mem=104.07, mem_unit="GB", mem_basis="resident", L=89)],
  flags=["No quality claims for the quant (card read in full): only throughput and peak memory; functional chat/image checks without scores [L76-91]",
         "Memory figure is peak unified memory on 128 GB M5 Max [L89]",
         "No disk size stated"])

repo(repo="orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4", sha="cd56f6c148c6287cf26fa6618223ad046b9af27c", fmt="compressed-tensors NVFP4/FP8 (per HF tags)",
  runtime="vLLM (per HF tags)", weights_modified=True, method=None,
  methodology=None, variants=[dict(name="Uncensored NVFP4", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes); format from HF tags only",
         "Weights modified per HF tags 'abliterated', 'uncensored' and repo name"])

repo(repo="peonist-ai/halogen-qwen3.8-flash-next", sha="43898d38c1deae87449619ab0ad0ffe993856e0f", fmt="halogen .hgn (custom)",
  runtime="halogen-flash-server (closed source, gfx1151)", weights_modified=False,
  method="Own quantization from BF16: v2 checkpoint 4-bit experts/trunk, 6-bit mixing layers, 8-bit MTP dense projections, n-gram table FP8 separate file; older w4b Q4C-P + activation-aware quality sidecar (o_proj 8-bit); ht43 smaller opt-in",
  methodology=None,
  variants=[dict(name="v2 (default, 0.15+)", size=62.1, unit="GiB", basis="trunk_only", bpw=4.16, L=158, L_extra=[50]),
            dict(name="ht43 (opt-in)", size=53.7, unit="GiB", basis="trunk_only", L=52),
            dict(name="w4b (0.14)", size=115.55, unit="GiB", basis="disk", L=168, L_extra=[55])],
  flags=["No numeric quality claims for the quant (card read in full): only qualitative statements — 8-bit o_proj 'measures as a statistical tie with full BF16 rows', sidecar holds 'most of the quality' — with no figures [L174-180]",
         "MTP acceptance 51% to 59% on prose is a speed figure, not quality [L185-186]",
         "Retrieval battery note: indexer budget 4096 recovered two planted facts at 16k missed by default; not a quant comparison [L138-140]",
         "Sizes: v2 62.1 GiB + n-gram table 47.7 GiB FP8 (~110 GiB download) [L50-51, L61-62]"])

repo(repo="scottlowry/Qwen3.8-Flash-Next-oQ3.5e-fp16-mtp", sha="d203e0af94012be6f24ed95d288a604eefd4d076", fmt="MLX",
  runtime="oMLX v0.7.1.dev1", weights_modified=False, method="oQ mixed precision (oMLX v0.7.1.dev1), 3 bits, group size 64",
  methodology=None, variants=[dict(name="oQ3.5e-fp16", size=None)],
  flags=["No quality claims for the quant (card read in full): only quantization details [L13-20]", "No size stated"])

repo(repo="scottlowry/Qwen3.8-Flash-Next-oQ5e-mtp", sha="38b03cec265d73b31b6397d62534116749872388", fmt="MLX",
  runtime="oMLX v0.7.1.dev1", weights_modified=False, method="oQ mixed precision (oMLX v0.7.1.dev1), 5 bits, group size 64",
  methodology=None, variants=[dict(name="oQ5e", size=None)],
  flags=["No quality claims for the quant (card read in full): only quantization details [L13-20]", "No size stated"])

repo(repo="sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit", sha="668f31bcc56bf9400e64c9463445eee47597c2d9", fmt="MLX",
  runtime="mlx-vlm (qwen4_exp MTP support) / omlx >= 0.6.4", weights_modified=True,
  method="REAP expert pruning 512 -> 288 per layer (top-10 routing kept) of Sawfwair/Qwen3.8-Flash-Next-MLX-4bit (affine 4-bit g64, n-gram g32), calibrated on-device over ~686K tokens of agentic-coding traffic; RMSNorm re-centered, n-gram tensors renamed",
  methodology=dict(reference="stock 512-expert 4-bit conversion (same harness), not BF16", dataset="HumanEval 164 problems, unit-test verified, one run per build; rare-name sampling 10 seeded generations", ctx=None, tool="not stated (same harness, same machine)", L=[118,119,102,103]),
  table_header_L=110,
  variants=[dict(name="REAP-288 4-bit", size=68, unit="GB", basis="disk", mem=68, mem_unit="GB", mem_basis="resident",
    m={"bench:HumanEval pass@1":"91.5%","rare_name_sampling":"9 of 10"}, L=115, L_extra=[21,104])],
  baseline={"source":"Base Q4 (512 experts, stock conversion)","bench:HumanEval pass@1":"93.9%","size_gb":"98 GB","L":112,"L_extra":[20]},
  ladder=[dict(name="REAP-384", size=80, unit="GB", basis="disk", m={"bench:HumanEval pass@1":"92.1%"}, L=113),
          dict(name="REAP-320", size=72, unit="GB", basis="disk", m={"bench:HumanEval pass@1":"90.9%"}, L=114),
          dict(name="REAP-256", size=65, unit="GB", basis="disk", m={"bench:HumanEval pass@1":"88.4%","rare_name_sampling":"1 of 10"}, L=116, L_extra=[103]),
          dict(name="REAP-288 with top-6 routing", m={"bench:HumanEval pass@1":"84.8%"}, L=121),
          dict(name="REAP-288 with top-4 routing", m={"bench:HumanEval pass@1":"63.4%"}, L=121)],
  flags=["Weights modified: pruned (224 of 512 routed experts removed per layer) [L23]",
         "Baseline is the publisher's 512-expert 4-bit conversion, not BF16 [L20, L112]",
         "Single-run HumanEval (164 problems), no confidence intervals; card itself says 1–2 point differences are within noise — 288 (91.5%) scores above 320 (90.9%) [L118-119, L210-211, L114-115]",
         "KL divergence against the stock model mentioned for the ladder but no values given [L101-102]",
         "Calibration from one team's agentic-coding traffic; card warns non-code domains may degrade more [L207-209]",
         "Only text quality evaluated; vision untested after pruning [L37-38, L215]",
         "Resident ~68 GB, or ~39 GB with n-gram table streamed [L21, L29-30]"])

repo(repo="starkweatherdigital/qwen3.8-flash-next-nvfp4", sha="1b304e5f99de0faaf43c3a959f2b4000294bf65c", fmt="NVFP4 (ModelOpt)",
  runtime="vLLM with loader patches (VLLM_PLE_NVFP4=1), GB10 image", weights_modified=False,
  method="NVFP4 quantization incl. 51.2B-param n-gram (PLE) table at 4-bit",
  methodology=None,
  variants=[dict(name="NVFP4", size=109, unit="GB", basis="disk", L=18)],
  flags=["No quality claims for the quant (card read in full): only throughput and MTP acceptance; card points to GitHub for verification/benchmarks [L27-29]"])

repo(repo="vumpt/Qwen3.8-Flash-Next-GGUF", sha="83a79fc5ba3ba3719f576e03d0a1f38de0774a32", fmt="GGUF",
  runtime="llama.cpp PR #27742 branch", weights_modified=False,
  method="Q4_K_M recipe with per-tensor fallback to Q5_0/Q8_0 where ncols % 256 != 0; token embedding/output Q6_K; PLE table Q5_0",
  methodology=None,
  variants=[dict(name="Q4_K_M", size=120, unit="GB", basis="disk", L=35)],
  flags=["No quality claims for the quant (card read in full): only layout, file and usage [L21-47]",
         "Size stated as approximate (~120 GB) [L35]"])

repo(repo="thousandml/Qwen3.8-Flash-Next-CQ-4bit", sha="a4492d6b5e5f75006cf5f431fb59140baa3d1a3e", fmt="CQ 4-bit (custom)",
  runtime="vLLM + CQ plugin/custom CUDA kernels (thousandml Docker image)", weights_modified=False,
  method="Coded Quantization (CQ) 4-bit, calibration-free",
  methodology=None, variants=[dict(name="CQ 4-bit", size=None)],
  flags=["No quality claims for the quant (card read in full): only qualitative statements that CQ is 'designed to preserve' behavior / generalize, with no figures [L27, L35, L176]",
         "No size stated"])

repo(repo="trymirai/Qwen3.8-Flash-Next-S-small-experimental", sha="c26c47d61befcb17cff8436c37936b16e71634ad", fmt="safetensors (2-bit, per HF tags)",
  runtime="vLLM (per HF tags)", weights_modified=False, method=None,
  methodology=None, variants=[dict(name="S-small-experimental", size=None)],
  flags=["No quality claims for the quant (card read in full): README snapshot at this sha is empty (0 bytes); format from HF tags only"])

repo(repo="xpashkin/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-7.1bpw", sha="f8e29279d22dcf21dc1c8b1fd45c68a6661b710e", fmt="MLX (mlx-serve)",
  runtime="patched mlx-serve 26.10.2 (mlx_qwen4_dense_attn.patch) / MLX 0.32.3", weights_modified=False,
  method="Imatrix-weighted affine gs64 experts (85 groups 6x64 + 11 gate_up groups 8x64, error-mass driven) using unsloth's imatrix; MTP 6-bit; all non-expert 2-D weights bf16; embed 8-bit; n-gram table 8-bit gs32",
  methodology=None,
  variants=[dict(name="iQ-MLX 7.1bpw", size=172.8, unit="GB", basis="disk", bpw=7.1, mem=115, mem_unit="GB", mem_basis="resident", L=62, L_extra=[18,64])],
  flags=["No quality claims for the quant (card read in full): only throughput/memory; long-context fidelity rationale for bf16 attention is stated without measurements [L73-77, L131-151]",
         "Size 172.8 GB (160.9 GiB) total on disk incl. 57.6 GB n-gram table; resident ≈115 GB [L62, L64]",
         "Imatrix taken from unsloth's GGUF repo (829k-token corpus) [L79-80]"])

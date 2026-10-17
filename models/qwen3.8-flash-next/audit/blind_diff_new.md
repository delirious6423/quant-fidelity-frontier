## 30cmtonyfaker/Qwen3.8-Flash-Next-REAP-288-MLX-Serve-4bit-MTP
  agreed: 0
  auditor only: 30[L146 ngram_table.bin file size (dis]
  ours only: 
## 6block/Qwen3.8-Flash-Next-GGUF
  agreed: 18
  auditor only: 14[L52 PPL improvement vs uniform tie], 15.73[L39 bpw], 329.7[L39 size]
  ours only: 
## AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF
  agreed: 6
  auditor only: 12.83[L70 shard size], 26.8[L190 PLE table size], 31[L167 VRAM used (32 GB card, n-cpu-m], 44.86[L70 shard size], 57.7[L121 size], 79.6[L72 routing mass retained worst la], 83.8[L121 size]
  ours only: 
## AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF
  agreed: 19
  auditor only: 1.2[L59 HumanEval delta vs Q3], 1.78[L126 file size], 2[L59 VRAM margin (--n-cpu-moe 3)], 3.2[L129 VRAM overhead of MTP head], 22[L262 VRAM (expert-cache 192, 98k)], 26.8[L175 extra RAM with --lazy-mode off], 30.3[L55 VRAM (--n-cpu-moe 3)], 31.8[L56 VRAM (fully resident)], 61.5[L63 size on disk (decimal)], 68.96[L63 size on disk (decimal)], 75.3[L25 retained routing mass (withdra], 78[L25 retained routing mass], 82.7[L25 retained routing mass (withdra], 83.8[L16 size on disk], 86.2[L96 VRAM (--n-cpu-moe 8, no spec)], 93.8[L131 VRAM (--n-cpu-moe 10, 96k)], 95.9[L97 VRAM (--n-cpu-moe 8, draft-mtp], 98.7[L259 VRAM (--n-cpu-moe 0, 98k)], 156[L52 HumanEval-164 pass@1 count], 158[L38 HumanEval-164 pass@1 count], 164[L52 HumanEval-164 pass@1 count], 840[L127 VRAM saved vs shared-Q8_0], 28104[L96 VRAM (--n-cpu-moe 8, no spec)], 29552[L256 VRAM (--n-cpu-moe 4, 8k)], 30108[L257 VRAM (--n-cpu-moe 3, 8k)], 30578[L131 VRAM (--n-cpu-moe 10, 96k)], 30730[L255 VRAM (--n-cpu-moe 4, 98k)], 31276[L97 VRAM (--n-cpu-moe 8, draft-mtp], 31658[L258 VRAM (--n-cpu-moe 0, 8k)], 32168[L259 VRAM (--n-cpu-moe 0, 98k)]
  ours only: 
## AxionML/Qwen3.8-Flash-Next-NVFP4
  agreed: 6
  auditor only: 360[L42 checkpoint size]
  ours only: 
## Baekpica/Qwen3.8-Flash-Next-GGUF
  agreed: 2
  auditor only: 3.6e+11[L47 source payload bytes]
  ours only: 
## Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-GGUF
  agreed: 2
  auditor only: 4.1[L86 DGX Spark headroom (projected)], 4.36[L75 size delta vs E5], 8.43[L86 DGX Spark headroom (projected)], 87.4[L81 model artifact memory (256K pr], 91.7368[L52 artifact size], 91.74[L81 model artifact memory (256K pr], 98.5016[L43 artifact size], 4.67927e+09[L75 size delta vs E5]
  ours only: 
## CMSManhattan/JiRackDeltaNet_180b
  agreed: 0
  auditor only: 0[L65 max logit difference], 65[L132 estimated file size], 75[L131 estimated file size], 88[L130 estimated file size], 95[L132 estimated file size], 96[L132 estimated RAM], 100[L64 next-token match vs reference], 110[L129 estimated file size], 128[L132 estimated RAM], 190[L128 estimated file size], 200[L128 estimated RAM], 256[L128 estimated RAM], 360[L127 estimated file size], 384[L127 estimated RAM]
  ours only: 
## Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF
  agreed: 12
  auditor only: 3.6[L73 n-gram table size], 28[L18 size reduction vs full quant], 66[L62 VRAM fully offloaded], 70[L70 CPU-only RAM needed], 72[L73 transformer on GPU], 77[L62 VRAM fully offloaded], 80[L70 CPU-only RAM needed]
  ours only: 
## DJLougen/Qwen3.8-Flash-Next-Mooney
  agreed: 18
  auditor only: 0.0002[L165 eval2048 dCE vs unquantized PL], 0.0006[L163 A/A CE delta], 0.001[L163 eval2048 CE delta (8-bit non-e], 0.0024[L163 eval2048 CE delta (8-bit non-e], 0.003[L127 KL contribution of 8-bit non-e], 0.004[L163 A/A KL], 0.005[L127 KL floor (BF16 vs BF16)], 0.0055[L165 KL added by PLE (long ctx)], 0.006[L163 truncated top-20 KL (8-bit non], 0.01[L127 KL floor (BF16 vs BF16)], 0.4[L261 truncated KL], 0.9[L102 mmproj file size], 0.972[L163 top-1 agreement (8-bit non-exp], 0.977[L163 A/A top-1 agreement], 1.6[L235 shard 1 size], 2.125[L96 bpw (routed experts)], 2.39[L98 bpw (transformer total)], 4[L260 score gap vs BF16 (points)], 5[L261 needle retrieval 256k], 5.5[L97 size (other transformer weight], 8.5[L99 bpw (PLE table)], 8.9[L97 bpw (other transformer weights], 11.1[L238 shard 4 size], 25[L237 shard 3 size], 32.1[L96 size (routed experts)], 35[L198 weights in memory], 37.6[L98 size (transformer total)], 38.6[L198 MemAvailable delta at load], 54.4[L236 shard 2 size (PLE)], 57.8[L191 memory at load -c 262144 (no M], 60.8[L191 memory at load -c 262144 (with], 96[L127 top-1 agreement floor (BF16 vs], 104[L108 file size], 114[L187 file size], 335[L108 memory needed]
  ours only: 
## DevelopingDad/Qwen3.8-Flash-Next-Uncensored-NVFP4-DGX-Spark
  agreed: 1
  auditor only: 11.08[L101 allocated KV memory (before dr], 13[L110 synthetic diagnostic strict pa], 13.89[L101 allocated KV memory (after dra], 14[L110 diagnostic fidelity full marks], 17[L110 diagnostic useful completion f], 18[L110 diagnostic format full marks], 20[L110 diagnostic format full marks], 170.9[L43 checkpoint size on disk]
  ours only: 
## Dyluhn/qwen3.8-flash-next-exl3
  agreed: 11
  auditor only: 0.022[L47 KL delta (agent slice) vs base], 4.05[L19 bpw (routed experts)], 94.06[L49 top-1 agreement (tp2 re-score)], 112.3[L30 file size]
  ours only: 0.0041, 0.0102, 95
## EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4
  agreed: 20
  auditor only: 0[L167 fp8 KV top-1 change], 0.0003[L169 fp8 KV delta KL free/locked do], 0.0004[L169 fp8 KV delta KL free/locked >=], 0.0005[L166 fp8 KV delta KL free/locked >=], 0.0006[L168 fp8 KV delta KL free/locked do], 0.0007[L169 fp8 KV delta KL free/locked >=], 0.0008[L157 long-doc NLL nats/token 4-8K], 0.0009[L169 fp8 KV delta KL free/locked do], 0.0011[L168 fp8 KV delta KL free/locked >=], 0.0012[L166 fp8 KV delta KL free/locked >=], 0.0013[L166 fp8 KV delta KL free/locked do], 0.0017[L168 fp8 KV delta KL free/locked do], 0.0021[L166 fp8 KV delta KL free/locked 4K], 0.0022[L167 fp8 KV delta KL free/locked do], 0.0023[L166 fp8 KV delta KL free/locked 4K], 0.0026[L169 fp8 KV delta KL free/locked 4K], 0.0027[L157 long-doc NLL nats/token doc-le], 0.0028[L158 long-doc NLL nats/token 2-4K], 0.003[L158 long-doc NLL nats/token >=8K], 0.0034[L156 long-doc NLL nats/token 4-8K], 0.0035[L158 long-doc NLL nats/token <2K], 0.0041[L158 long-doc NLL nats/token 4-8K], 0.005[L157 long-doc NLL nats/token 2-4K], 0.0051[L157 long-doc NLL nats/token <2K], 0.0053[L168 fp8 KV delta KL free/locked 4K], 0.0061[L156 long-doc NLL nats/token >=8K], 0.0069[L167 fp8 KV delta KL free/locked 4K], 0.0071[L156 long-doc NLL nats/token doc-le], 0.008[L108 KL between two bf16 implementa], 0.0082[L155 long-doc NLL nats/token <2K], 0.0083[L155 long-doc NLL nats/token >=8K], 0.0108[L155 long-doc NLL nats/token 2-4K], 0.011[L155 long-doc NLL nats/token 4-8K], 0.0115[L141 long-v1 doc-level KL t+a locke], 0.0116[L156 long-doc NLL nats/token 2-4K], 0.0122[L156 long-doc NLL nats/token <2K], 0.0149[L140 long-v1 doc-level KL t+a locke], 0.0225[L141 long-v1 doc-level KL t+a free], 0.0266[L142 long-v1 doc-level KL t+a locke], 0.0267[L140 long-v1 doc-level KL t+a free]
  ours only: 
## EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4
  agreed: 37
  auditor only: 0[L45 HumanEval pass@1 difference vs], 0.0004[L117 NLL long-v1 4-8K], 0.0006[L117 NLL long-v1 >=8K], 0.0016[L117 NLL long-v1 2-4K], 0.0028[L117 NLL long-v1 document level], 0.0046[L117 NLL long-v1 2-4K], 0.0059[L117 NLL long-v1 <2K], 0.007[L117 NLL long-v1 >=8K], 0.0072[L117 NLL long-v1 2-4K], 0.0082[L117 NLL long-v1 4-8K], 0.0083[L116 NLL long-v1 >=8K], 0.0088[L117 NLL long-v1 <2K], 0.0097[L116 NLL long-v1 document level], 0.0108[L116 NLL long-v1 2-4K], 0.011[L116 NLL long-v1 4-8K], 0.0134[L117 NLL long-v1 4-8K], 0.018[L127 KL t+a free/locked/top-1 docum], 0.0266[L128 KL t+a free/locked/top-1 docum], 0.0307[L127 KL t+a free/locked/top-1 docum], 0.0395[L128 KL t+a free/locked/top-1 docum], 0.0408[L126 KL t+a free/locked/top-1 docum], 0.0489[L127 KL free/locked [top-1] <2K], 0.0561[L126 KL t+a free/locked/top-1 docum], 0.0756[L128 KL free/locked [top-1] <2K], 0.0955[L127 KL free/locked [top-1] <2K], 0.1[L44 GSM8K difference vs 16-bit], 0.1043[L126 KL free/locked [top-1] <2K], 0.1282[L127 KL free/locked [top-1] >=8K], 0.1313[L128 KL free/locked [top-1] <2K], 0.1554[L126 KL free/locked [top-1] <2K], 0.1573[L127 KL free/locked [top-1] 2-4K], 0.16[L81 NLL long docs (document level)], 0.1711[L127 KL free/locked [top-1] 4-8K], 0.1979[L128 KL free/locked [top-1] >=8K], 0.215[L126 KL free/locked [top-1] >=8K], 0.23[L69 PPL unseen ctx512 change], 0.2618[L128 KL free/locked [top-1] 2-4K], 0.2794[L128 KL free/locked [top-1] 4-8K], 0.2931[L126 KL free/locked [top-1] 2-4K], 0.2984[L126 KL free/locked [top-1] 4-8K]
  ours only: 
## FenomAI/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
  agreed: 36
  auditor only: 0.51[L131 GPQA-D gap vs BF16], 0.57[L83 LCB v6 gap vs BF16], 0.91[L96 file size], 1.14[L131 LCB v6 gap vs BF16], 1.52[L83 GPQA-Diamond gap vs IQ3_S], 1.6[L111 size difference], 1.89[L52 bpw (effective over original p], 4.5[L71 bpw (n-gram table)], 4.7[L131 compression vs BF16], 5.3[L131 compression vs BF16], 7.8[L83 size smaller than IQ3_S], 16[L123 bpw], 28.8[L156 resident footprint removed by ], 58.4[L52 total size], 95.7[L131 task avg recovery], 99.4[L131 task avg recovery], 354[L131 size]
  ours only: 
## GBP-DE/Qwen3.8-Flash-Next-oQ5e-WLSRefit-MTP
  agreed: 0
  auditor only: 11[L28 KL divergence reduction], 15[L28 KL divergence reduction]
  ours only: 
## HamboneLabs-AI/Qwen3.8-Flash-Next-uint3-g64
  agreed: 14
  auditor only: 0.002591[L84 max scale relative error], 0.0152[L166 decisive disagreement], 0.20966[L81 relative Frobenius error media], 0.21234[L81 relative Frobenius error media], 0.22889[L81 relative Frobenius error media], 0.29639[L81 relative Frobenius error media], 0.5[L74 bf16 scale overhead vs fp8], 0.539[L174 agreement p<0.50], 0.8[L175 agreement 0.50-0.90], 0.955[L176 agreement 0.90-0.99], 0.95758[L82 cosine similarity median / min], 0.97809[L82 cosine similarity median / min], 0.992[L177 agreement p>=0.99], 1[L144 GSM8K delta], 2.5[L145 MATH-500 delta], 3.458[L440 effective bpw], 4.18[L149 pooled harm one-sided 95% uppe], 4.88[L146 HumanEval delta], 16.64[L440 relative Frobenius error], 17.82[L122 steady-state VRAM delta vs 4.5], 20.01[L440 relative Frobenius error], 32.67[L416 relative error at 3.000 bpw], 52.04[L121 model load memory], 63[L158 long-context replay], 68[L222 VRAM/unified memory floor stea], 69.55[L121 model load memory], 72.46[L122 steady-state VRAM], 85.3[L122 steady-state VRAM], 91.58[L122 steady-state VRAM], 95.4[L255 PLE table size], 99.2[L180 agreement where reference conf], 147[L227 total download], 1.024e+11[L255 PLE table size bytes]
  ours only: 
## INCModel3/Qwen3.8-Flash-Next-MXFP4-Mixed-CT-RTN-AutoRound
  agreed: 12
  auditor only: 0.007[L68 FP8 KV delta hellaswag acc_nor], 0.51[L72 piqa at TP=2 (broken)], 102[L73 n-gram PLE table size (pinned ]
  ours only: 
## ISTA-DASLab/Qwen3.8-Flash-Next-P48NVFP4-MoESQ
  agreed: 32
  auditor only: 1.1[L78 LCB v6 presence penalty effect], 1.33[L131 LCB v6 65k without presence pe], 2[L37 bpw (kept values)], 4[L144 routed experts size], 4.5[L38 GPU-resident compression], 4.9[L146 MTP module size], 5.8[L36 routed experts compression], 8[L144 routed experts size], 10[L145 non-expert dense weights size], 16[L148 n-gram tables size], 27[L62 weights per GPU (2x B200)], 38.7[L144 routed experts size], 43.09[L131 LCB v6 65k without presence pe], 59.8[L152 routed experts size if dense], 64[L41 share of checkpoint that is n-], 68.6[L136 SWE-bench Verified (484 subset], 70.6[L98 SWE-bench Verified 170-instanc], 74.7[L98 SWE-bench Verified 170-instanc], 81.8[L136 SWE-bench Verified (484 subset], 95.4[L148 n-gram tables size], 225[L144 routed experts size], 239.9[L147 GPU-resident total], 335.3[L149 checkpoint total size]
  ours only: 
## JANGQ-AI/Qwen3.8-Flash-Next-JANGH4
  agreed: 16
  auditor only: 0[L85 reasoning: median KL], 0.0001[L85 reasoning: median KL], 0.0006[L118 median KL vs bf16 MTP], 0.0007[L92 agentic: median KL], 0.0008[L89 cybersecurity: median KL], 0.0011[L91 terminal: median KL], 0.0013[L88 networking: median KL], 0.0022[L90 creative: median KL], 0.0023[L87 long context: median KL], 0.0029[L88 networking: median KL], 0.003[L89 cybersecurity: median KL], 0.0034[L91 terminal: median KL], 0.0035[L86 science: median KL], 0.006[L73 reasoning traces: KL p90 / p95], 0.0098[L90 creative: median KL], 0.013[L73 reasoning traces: KL p90 / p95], 0.015[L73 reasoning traces: mean KL (top], 0.034[L74 reasoning traces: KL p90 / p95], 0.067[L74 reasoning traces: KL p90 / p95], 0.09[L75 tool conversations: mean KL (t], 0.112[L71 general: KL p90 / p95 / p99], 0.113[L72 general: KL p90 / p95 / p99], 0.186[L75 tool conversations: KL p90 / p], 0.29[L73 reasoning traces: KL p90 / p95], 0.314[L72 general: KL p90 / p95 / p99], 0.432[L75 tool conversations: KL p90 / p], 0.568[L76 tool conversations: KL p90 / p], 0.776[L71 general: KL p90 / p95 / p99], 0.867[L74 reasoning traces: KL p90 / p95], 1.275[L76 tool conversations: KL p90 / p], 1.579[L75 tool conversations: KL p90 / p], 1.593[L72 general: KL p90 / p95 / p99], 3[L99 answer points flipped vs bf16], 3.522[L76 tool conversations: KL p90 / p], 9[L98 tool-call points flipped vs bf], 16[L98 tool-call points flipped vs bf], 53.6[L42 n-gram table on SSD], 55.6[L107 medium: exact-answer accuracy], 58.3[L107 medium: exact-answer accuracy], 66.7[L108 xhigh: exact-answer accuracy]
  ours only: 
## JRamirez-UAB/Qwen3.8-Flash-Next-EXL3-2.25bpw-48GB
  agreed: 18
  auditor only: 0.0193[L288 KL plugin 0.3.1 vs 0.3.0], 0.02[L288 KL between two 0.3.0 runs], 0.048[L119 KL vs BF16 (public / sysadmin ], 0.053[L119 KL vs BF16 (public / sysadmin ], 6[L197 host RAM (TabbyAPI)], 8[L83 needle 8 facts], 15[L83 vision], 19[L122 hard sysadmin tasks], 20[L122 hard sysadmin tasks], 24[L122 hard sysadmin tasks], 26[L62 n-gram table file size], 26.2[L66 EXL3 n-gram table size], 29[L120 tool calls (xhigh · no reasoni], 39[L121 agent tasks (20x2)], 40[L121 agent tasks (20x2)], 40.5[L67 download size without n-gram t], 51[L180 PLE table pinned in RAM], 54[L71 PLE file download size], 57[L83 tools opencode], 66.7[L66 download size], 90[L180 host RAM needed (vLLM)], 18780[L83 reasoning xhigh correct · toke], 19264[L82 reasoning xhigh correct · toke], 45125[L88 VRAM at peak], 48417[L87 VRAM at peak], 48471[L230 VRAM at peak (many users)], 241437[L83 needle 8 facts]
  ours only: 
## Jakevin/Qwen3.8-Flash-Next-ternary-mlx
  agreed: 5
  auditor only: 1.85[L25 bits per weight (ternary units], 6[L32 points lower than ISTA 2-bit], 13[L32 correct answers fewer than IST], 20[L65 RSS + Metal active memory]
  ours only: 
## Kymidnight/Qwen3.8-Flash-Next-ROCmFP4-FAST-GGUF
  agreed: 6
  auditor only: 1.3[L56 n-gram table per-head tensor s], 1.5[L118 extra size of Q8_0 draft], 2.27[L97 MTP head size], 2.3[L121 added VRAM], 3.5[L66 bpw (n-gram PLE table)], 4.25[L65 bpw (experts/attention)], 6.56[L67 bpw (token_embd/output)], 20.9[L56 n-gram table total size], 85[L121 VRAM], 88.6[L93 VRAM fully resident]
  ours only: 
## Lottolabs/Qwen3.8-Flash-Next-Coder-P150
  agreed: 12
  auditor only: 0.18[L86 perplexity change code], 0.37[L86 perplexity change chat], 22[L81 placement size], 24[L62 DRAM-resident experts size], 44[L89 GSM8K first 50], 708[L40 per-layer pool file size]
  ours only: 
## Manav1011/Qwen3.8-Flash-Next-NVFP4
  agreed: 6
  auditor only: 1[L121 AIME26 pass@1], 2.7[L92 compression vs BF16], 98.86[L120 GSM8K stop rate], 99.17[L121 AIME26 stop rate], 118.4[L137 unchanged tensors size], 240[L121 AIME26], 360[L92 checkpoint size]
  ours only: 
## Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata
  agreed: 27
  auditor only: 0.77[L144 MTP size], 1.18[L143 token embedding size], 1.43[L139 dense.bin size], 5.58[L141 dense gguf size], 12[L104 KL removed vs GPTQ], 18[L104 extra size], 43[L29 KL reduction vs NVFP4 GPTQ], 47.68[L142 PLE fp8 gguf size], 63.3[L81 experts size], 103.1[L81 experts size], 103.13[L137 experts.bin size], 110[L111 RAM budget], 119.5[L81 experts size], 159.8[L130 disk size]
  ours only: 
## Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata
  agreed: 21
  auditor only: 0.035[L97 KL shift from route tail skip], 0.3[L77 expert error relative to NVFP4], 0.77[L113 MTP size], 1.18[L112 token embd size], 1.43[L108 dense.bin size], 5.58[L110 dense gguf size], 8.5[L76 bpw (experts)], 47.68[L111 PLE fp8 gguf size], 68[L62 RAM needed to keep experts res], 88[L61 RAM needed to keep experts res], 119.5[L91 experts size], 119.53[L106 experts.bin size], 176.2[L99 disk size], 360[L153 download size]
  ours only: 
## Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound
  agreed: 11
  auditor only: 0.14[L166 average log-likelihood change ], 13[L35 experts in host RAM per card], 18.6[L35 VRAM weights per GPU], 18.8[L35 VRAM weights per GPU], 20.44[L35 VRAM weights per GPU], 21.05[L35 VRAM weights per GPU], 21.18[L35 VRAM weights per GPU], 21.2[L35 VRAM weights per GPU], 21.71[L35 VRAM weights per GPU], 58[L178 routed experts size], 73.3[L78 IFBench strict prompt level], 76.2[L78 IFBench strict instruction lev], 83.4[L78 IFBench loose instruction leve], 84.2[L79 GPQA biology], 86.9[L80 LCB hard], 87.1[L79 GPQA chemistry], 92[L178 routed experts share of body], 94.9[L80 LCB medium], 95.3[L79 GPQA physics], 95.37[L125 PLE table file], 100[L80 LCB easy]
  ours only: 16
## Myric/Qwen3.8-Flash-Next-APEX-GGUF
  agreed: 8
  auditor only: 0.02495[L148 perplexity wikitext-2 (non-can], 0.9[L32 file size], 4.2801[L148 perplexity wikitext-2 (non-can], 28.8[L209 PLE table size at Q4_0], 77[L205 total size range], 112[L205 total size range], 360[L221 source safetensors size], 9.16556e+10[L159 file size (bytes)]
  ours only: 
## OsaurusAI/Allosaurus-v0.1-125B-A6B-JANGH2
  agreed: 27
  auditor only: 0.0002[L80 office/finance/math: median KL], 0.0005[L99 systems long prompts: median K], 0.0006[L95 Chinese: median KL], 0.0013[L94 academic multiple choice: medi], 0.0014[L98 coding long prompts: median KL], 0.0016[L92 science: median KL], 0.0019[L88 cybersecurity: median KL], 0.0021[L89 coding: median KL], 0.0034[L91 general chat: median KL], 0.0035[L90 systems: median KL], 0.0037[L97 agentic long prompts: median K], 0.0041[L96 cybersecurity long prompts: me], 0.0056[L93 agentic: median KL], 0.007[L80 office/finance/math: mean KL], 0.008[L74 reasoning traces: mean KL], 0.011[L80 office/finance/math: KL p90 / ], 0.0134[L95 Chinese: median KL], 0.0154[L94 academic multiple choice: medi], 0.017[L74 reasoning traces: KL p90 / p95], 0.0209[L100 long documents: median KL], 0.023[L80 office/finance/math: KL p90 / ], 0.0246[L88 cybersecurity: median KL], 0.0274[L89 coding: median KL], 0.0331[L90 systems: median KL], 0.0341[L91 general chat: median KL], 0.0371[L96 cybersecurity long prompts: me], 0.044[L95 Chinese: median KL], 0.0441[L92 science: median KL], 0.0488[L97 agentic long prompts: median K], 0.054[L71 general: mean KL], 0.0619[L88 cybersecurity: median KL], 0.0621[L94 academic multiple choice: medi], 0.0656[L93 agentic: median KL], 0.0795[L91 general chat: median KL], 0.092[L71 general: KL p90 / p95 / p99], 0.094[L80 office/finance/math: KL p90 / ], 0.1033[L90 systems: median KL], 0.1044[L98 coding long prompts: median KL], 0.1169[L93 agentic: median KL], 0.125[L77 tool conversations: mean KL]
  ours only: 
## PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2-GGUF
  agreed: 10
  auditor only: 4.1[L317 file size], 4.14[L165 file size], 16.7[L149 quality bench score: Real worl], 26.8[L323 PLE table size], 66.7[L146 quality bench score: Math], 67.9[L321 memory at full window], 69.5[L321 memory at full window], 74.1[L143 quality bench score: Code], 78.3[L190 model size on disk], 83.3[L148 quality bench score: Creative], 90.9[L145 quality bench score: Instructi], 360[L44 size], 4.13743e+09[L271 file size (bytes)], 7.82718e+10[L163 file size (bytes)]
  ours only: 
## Quat3rnion/halogen-qwen3.8-flash-next-v2-abliterated
  agreed: 10
  auditor only: 0.027[L375 per-token NLL advantage over r], 0.84[L296 vision tower size], 2.4[L281 calibration size], 3.181[L147 WikiText-2 PPL], 3.229[L148 WikiText-2 PPL], 13.6[L183 size reduction vs v2], 22[L283 size over stock v2], 47.7[L294 n-gram table size], 51.2[L307 n-gram table size], 53.66[L278 file size], 62.1[L400 resident weights], 62.11[L190 referenced weights size], 64.8[L304 server container memory (262K ], 66.7[L283 file size], 68.3[L400 resident weights], 88.7[L307 file size], 137[L307 total disk], 8.97916e+08[L74 planned vision file size], 5.12002e+10[L72 planned n-gram file size], 5.12002e+10[L72 n-gram file size], 5.76183e+10[L186 file size]
  ours only: 
## Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2
  agreed: 2
  auditor only: 3.181[L55 WikiText-2 perplexity], 52[L45 n-gram + vision size], 5.76183e+10[L69 file size (bytes)], 6.81426e+10[L55 file size (bytes)], 1.09716e+11[L63 payload bytes (complete output]
  ours only: 
## Robot-Haus/Qwen3.8-Flash-Next-MLX-oQ3-MTP
  agreed: 4
  auditor only: 3[L70 base bits], 64.6[L126 peak memory pp1024 (M1 Ultra)], 67.25[L127 peak memory pp4096], 67.39[L128 peak memory pp8192], 67.68[L129 peak memory pp16384], 68.26[L130 peak memory pp32768], 69.42[L131 peak memory pp65536], 70[L139 peak memory all contexts], 86.152[L76 weight size]
  ours only: 
## Robot-Haus/Qwen3.8-Flash-Next-oQ3.5e-fp16-mtp
  agreed: 7
  auditor only: 3[L436 bits], 62.93[L268 peak memory pp1024/tg128], 64.49[L246 peak memory pp1024/tg128], 69.68[L269 peak memory pp4096/tg128], 71.68[L247 peak memory pp4096/tg128], 72.49[L270 peak memory pp8192/tg128], 73.03[L271 peak memory pp16384/tg128], 74.71[L248 peak memory pp8192/tg128], 75.38[L249 peak memory pp16384/tg128], 78.35[L272 peak memory pp32768/tg128], 80.13[L273 peak memory pp65536/tg128], 80.87[L250 peak memory pp32768/tg128], 82.81[L251 peak memory pp65536/tg128], 85.21[L274 peak memory pp131072/tg128], 86.85[L224 peak memory pp1024/tg128], 87.63[L252 peak memory pp131072/tg128], 89.4[L225 peak memory pp4096/tg128], 89.49[L226 peak memory pp8192/tg128], 89.78[L227 peak memory pp16384/tg128], 90.35[L228 peak memory pp32768/tg128], 91.49[L229 peak memory pp65536/tg128], 93.84[L230 peak memory pp131072/tg128], 94.1[L158 peak memory (summary)], 94.14[L275 peak memory pp200000/tg128], 95.59[L209 peak memory pp200000/tg128], 95.6[L157 peak memory (summary)], 98[L155 peak memory (summary)], 98.09[L231 peak memory pp200000/tg128], 112.5[L156 peak memory (summary)], 112.54[L253 peak memory pp200000/tg128], 150[L174 HumanEval (pass@1) correct], 175[L175 MBPP (pass@1) correct], 260[L177 SafetyBench correct], 283[L172 HellaSwag correct], 286[L173 TruthfulQA correct], 854[L171 MMLU correct]
  ours only: 
## Robot-Haus/Qwen3.8-Flash-Next-oQ4e-fp16-mtp
  agreed: 2
  auditor only: 5[L28 PLE table bpw], 29.82[L39 PLE table size], 70[L103 always-resident size], 73.7[L64 peak memory 1,024 depth], 81.1[L65 peak memory 4,096 depth], 84.1[L66 peak memory 8,192 depth]
  ours only: 
## ScottzillaSystems/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4
  agreed: 5
  auditor only: 47.5[L196 MMLU moral_scenarios], 50[L190 MMLU machine_learning], 52.5[L159 MMLU college_chemistry], 55[L208 MMLU virology], 57.5[L159 MMLU college_chemistry], 60[L204 MMLU public_relations], 62.5[L200 MMLU professional_accounting], 65[L208 MMLU virology], 67.5[L204 MMLU public_relations], 70[L205 MMLU security_studies], 72.5[L201 MMLU professional_law], 75[L205 MMLU security_studies], 77.5[L185 MMLU human_aging], 80[L168 MMLU elementary_mathematics], 82.5[L197 MMLU nutrition], 85[L199 MMLU prehistory], 87.5[L209 MMLU world_religions], 90[L206 MMLU sociology], 91.2[L74 HarmBench real-harm TRUE_COMPL], 92.5[L207 MMLU us_foreign_policy], 95[L209 MMLU world_religions], 97.5[L206 MMLU sociology], 100[L198 MMLU philosophy], 219[L74 HarmBench real-harm TRUE_COMPL], 222[L73 HarmBench real-harm TRUE_COMPL], 240[L74 HarmBench real-harm TRUE_COMPL]
  ours only: 
## Slomin/Potato-CODER-48GB
  agreed: 32
  auditor only: 0.6[L78 vision projector size], 0.8[L231 non-expert tensor upgrade budg], 1.3[L78 MTP head size], 5.4[L108 PHP gap vs full model], 7[L125 LiveCodeBench advantage vs NVF], 12.5[L108 PHP gap upper bound], 27[L55 PLE lookup table RAM], 40[L55 system RAM], 64[L60 total files size], 640[L126 16 languages vs NVFP4], 23706[L54 VRAM peak both cards], 23754[L54 VRAM peak both cards]
  ours only: 
## SpeederX/Qwen3.8-Flash-Next-UD-Q2_K_XL-EM-GGUF
  agreed: 5
  auditor only: 73.46[L130 file size]
  ours only: 
## TensorFold/Qwen3.8-Flash-Next-MLX-4bit
  agreed: 1
  auditor only: 103.91[L64 weight size]
  ours only: 
## TensorFold/Qwen3.8-Flash-Next-MLX-oQ2
  agreed: 1
  auditor only: 63.02[L69 weight size]
  ours only: 
## TensorFold/Qwen3.8-Flash-Next-MLX-oQ4
  agreed: 1
  auditor only: 104.02[L65 weight size]
  ours only: 
## TensorFold/Qwen3.8-Flash-Next-MLX-oQ6
  agreed: 1
  auditor only: 145.08[L65 weight size]
  ours only: 
## TheDrainFlorist/Qwen3.8-Flash-Next-VQ-3.2bpw
  agreed: 41
  auditor only: 0[L140 KL old vs new runtime logits], 0.77[L252 MTP acceptance], 0.82[L305 MTP acceptance (exo)], 0.84[L163 vision tower size], 0.88[L305 MTP acceptance (exo)], 1.6[L98 KL change v1->v2 literary], 1.6354[L102 perplexity code (pre-fix)], 1.6407[L102 perplexity code (pre-fix)], 2.1[L245 MTP head file size], 2.2[L249 MTP extra resident], 3.1[L96 KL change v1->v2 prose], 3.2[L97 KL change v1->v2 code], 4.8[L179 literary KL (mnat/token)], 5.0297[L102 perplexity prose (pre-fix)], 5.0721[L102 perplexity prose (pre-fix)], 6.6612[L102 perplexity literary (pre-fix)], 6.6798[L102 perplexity literary (pre-fix)], 7.3[L177 literary KL (mnat/token)], 8.7[L178 literary KL (mnat/token)], 12.1[L179 code KL (mnat/token)], 13.5[L175 literary KL (mnat/token)], 16.5[L177 code KL (mnat/token)], 25.7[L176 literary KL (mnat/token)], 26.9[L178 code KL (mnat/token)], 27[L87 size change vs v1], 28[L185 size vs affine q4], 29.5[L175 code KL (mnat/token)], 33[L97 KL mnat/token (pre-fix runtime], 34.1[L97 KL mnat/token (pre-fix runtime], 46[L185 KL vs affine q4 literary], 50.4[L179 prose KL (mnat/token)], 51[L185 KL vs affine q4 prose], 52[L185 KL vs affine q4 code], 53.3[L153 peak memory RSS (2048 prefill)], 68.8[L177 prose KL (mnat/token)], 69.5[L87 size (v2 weights)], 76.8[L206 peak memory (speed-pair)], 83.7[L206 peak memory], 84.1[L178 prose KL (mnat/token)], 85[L186 KL vs affine q3 prose]
  ours only: 
## TheDrainFlorist/Qwen3.8-Flash-Next-VQ-5.5bpw
  agreed: 41
  auditor only: 0[L166 mean KL (millinats/token)], 0.84[L149 vision tower size], 1.6[L173 size ratio vs this model], 2.1[L236 MTP head size], 2.2[L240 MTP extra resident], 3[L275 size cost], 4[L275 size cost], 4.8[L165 literary KL (millinats/token)], 8.7[L164 literary KL (millinats/token)], 12.1[L165 code KL (millinats/token)], 13.5[L161 literary KL (millinats/token)], 15[L275 KL gain], 16[L172 literary KL vs affine q6], 18[L172 prose KL vs affine q6], 24[L275 KL gain], 25[L172 size larger than this model], 25.7[L162 literary KL (millinats/token)], 26.9[L164 code KL (millinats/token)], 29.5[L161 code KL (millinats/token)], 37[L173 code KL vs this model], 38[L172 code KL vs affine q6], 50.4[L165 prose KL (millinats/token)], 51[L173 literary KL vs this model], 54.6[L159 literary KL (millinats/token)], 61[L171 prose KL vs affine q5], 61.3[L132 peak RSS (load + 2048 prefill ], 72[L171 literary KL vs affine q5], 81[L171 code KL vs affine q5], 81.8[L159 code KL (millinats/token)], 84.1[L164 prose KL (millinats/token)], 88.4[L162 code KL (millinats/token)], 100[L166 top-1 (prose)], 101.1[L160 literary KL (millinats/token)], 112[L135 worst-case memory], 123[L213 disk size], 169.3[L160 code KL (millinats/token)], 178.4[L162 prose KL (millinats/token)], 276.8[L158 code KL (millinats/token)], 305.6[L159 prose KL (millinats/token)], 624.1[L160 prose KL (millinats/token)]
  ours only: 
## TracNetwork/mayhem-catalog-Qwen-Qwen3-8-Flash-Next-NVFP4
  agreed: 7
  auditor only: 1[L121 AIME26 pass@1], 2.7[L92 compression vs BF16], 98.86[L120 GSM8K stop rate], 99.17[L121 AIME26 stop rate], 118.4[L137 unchanged tensors size], 360[L92 checkpoint size]
  ours only: 
## Tylogi/Qwen3.8-Flash-Next-EW-MFQ
  agreed: 47
  auditor only: 5.03[L76 top-1 gain vs UD IQ1_M], 29.54[L153 PLE table size], 37.4[L76 KLD reduction vs UD IQ1_M], 41.04[L116 resident RAM usage (CUDA)], 42.9[L117 resident RAM usage (CUDA)], 43.36[L153 resident weights], 46.93[L118 resident RAM usage (CUDA)], 47.41[L106 resident RAM usage (Apple)], 49.81[L119 resident RAM usage (CUDA)], 50.33[L107 resident RAM usage (Apple)], 57.28[L120 resident RAM usage (CUDA)], 57.95[L108 resident RAM usage (Apple)], 60.72[L121 resident RAM usage (CUDA)], 61.48[L109 resident RAM usage (Apple)], 72.44[L153 download size], 78.25[L110 resident RAM usage (Apple)]
  ours only: 
## Vemloir/Qwen3.8-Flash-Next-sc-exl3
  agreed: 14
  auditor only: 0.00209[L60 KL Math proofs], 0.00263[L70 KL Portuguese], 0.00264[L74 KL French], 0.00292[L74 KL French], 0.00301[L60 KL Math proofs], 0.00309[L73 KL Italian], 0.00351[L76 KL German], 0.00359[L71 KL Korean], 0.0036[L70 KL Portuguese], 0.00361[L73 KL Italian], 0.0038[L76 KL German], 0.00406[L78 KL Japanese], 0.0041[L78 KL Japanese], 0.00446[L77 KL Spanish], 0.00451[L77 KL Spanish], 0.0047[L71 KL Korean], 0.00541[L75 KL Chinese], 0.00591[L75 KL Chinese], 0.00613[L63 KL Chat], 0.00619[L65 KL Multi-turn instruction foll], 0.00621[L65 KL Multi-turn instruction foll], 0.00701[L63 KL Chat], 0.00761[L69 KL Russian], 0.00964[L69 KL Russian], 0.01011[L62 KL Knowledge-base support agen], 0.01035[L72 KL Hindi], 0.01061[L64 KL Agentic tool calling], 0.01104[L61 KL SWE], 0.01145[L64 KL Agentic tool calling], 0.01191[L62 KL Knowledge-base support agen], 0.01269[L72 KL Hindi], 0.01343[L61 KL SWE], 0.4[L65 KL difference vs turboderp Mul], 1.07[L49 median per-token KL Nemotron], 1.16[L49 median per-token KL multilingu], 1.2[L78 KL difference vs turboderp Jap], 3[L49 median per-token KL multilingu], 4[L49 median per-token KL multilingu], 4.0498[L24 bpw], 7.3[L64 KL difference vs turboderp Age]
  ours only: 6.2, 10.1, 18.6, 24.6
## VnimanieAI/Qwen3.8-Flash-Next-W4A16
  agreed: 18
  auditor only: 3.4[L177 extra weights per GPU vs this ], 4[L185 GPQA hard-20 correct], 16[L184 GPQA hard-20 truncations], 20[L184 GPQA hard-20 truncations], 23[L36 weights per GPU at TP4], 37.9[L215 GPQA truncation rate @32k], 102[L27 PLE tables size (host RAM)], 110[L67 host RAM needed for PLE], 335[L200 size], 470[L47 shared expert BF16 size]
  ours only: 0.025, 0.68, 0.7, 0.837, 0.854, 3.8, 8, 27, 3391
## WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090
  agreed: 15
  auditor only: 0.0158[L134 ΔNLL chat text (Strata prompts], 0.84[L184 vision tower pinned host RAM], 3.5[L164 startup GPU memory], 29.1[L97 n-gram volume size], 33[L163 startup pinned host RAM], 40[L94 free RAM needed], 70[L97 disk], 96.5[L133 Strata self top-1 agreement], 99.2[L133 Strata self top-1 agreement], 9.01214e+09[L71 model part file size bytes], 2.88001e+10[L235 file size bytes], 2.91273e+10[L72 n-gram volume size bytes], 3.2e+10[L70 model file size bytes], 3.9226e+10[L235 file size bytes]
  ours only: 
## YTan2000/Qwen3.8-Flash-Next-TQ3_4S
  agreed: 18
  auditor only: 2.8[L70 shard 5 size], 4[L42 bpw], 27.8[L68 shard 3 size], 27.9[L69 shard 4 size], 50.7[L67 shard 2 size (PLE)], 74[L92 resident memory at 262k ctx], 511[L66 shard 1 size]
  ours only: 
## airagrp/Qwen3.8-Flash-Next-mixed-iQ5-Q8-ple16-mlx-serve
  agreed: 13
  auditor only: 0.8[L21 vision tower size], 0.9[L94 vision tower size], 1[L43 top-1 gain from bf16 table], 2.5[L126 top-1 change], 3.5[L39 resident memory saved by Q8 he], 16[L63 checkpoint size], 91.9[L21 trunk shards size], 95.4[L21 ngram_table.bin size], 99.6[L131 resident memory], 102.4[L72 ngram_table.bin size], 202[L81 disk size], 360[L162 checkpoint size]
  ours only: 
## aixiaoma/Qwen3.8-Flash-Next-W4A16
  agreed: 7
  auditor only: 66[L26 GPU-resident weights], 102[L26 PLE tables host RAM], 110[L59 free host RAM for PLE], 335[L120 size], 470[L39 shared expert BF16 size]
  ours only: 8
## aniolekx/Qwen3.8-Flash-Next-E9-MLX-8bit
  agreed: 13
  auditor only: 0.17589[L105 NLL], 0.17602[L105 NLL (ENGINE)], 5[L106 top-1 agreement ENGINE vs mlx-], 6[L106 top-1 agreement ENGINE vs mlx-], 16[L119 xhigh suite verifiable correct], 21[L122 xhigh suite verifiable correct], 52.9[L86 peak memory saved], 99.57[L105 argmax agreement vs official t], 181.6[L51 size on disk], 194.9[L161 weights memory], 195[L170 resident model], 252.6[L162 peak memory 262k prefill], 316[L106 top-1 agreement ENGINE vs mlx-], 345[L106 top-1 agreement ENGINE vs mlx-], 4040[L106 top-1 agreement ENGINE vs mlx-], 4096[L106 top-1 agreement ENGINE vs mlx-], 1.94946e+11[L51 tensor data bytes]
  ours only: 0.00068, 0.00171, 0.00345, 0.00591, 0.66, 16.5, 95
## beamster/Qwen3.8-Flash-Next-Sushi-2.6bpw
  agreed: 31
  auditor only: 0.0654[L82 KLD with 4-bit table], 0.136[L61 KLD (rounded)], 0.199[L61 KLD (rounded)], 2.6[L26 bpw (experts)], 5.4[L61 extra memory vs 2.6bpw], 6.7[L61 GPU memory less than iQ-MLX 3.], 24[L61 KLD reduction vs 2.6bpw], 25[L60 GPU memory less than oQ4e], 29.8[L82 size on disk], 43.9[L32 weights size], 66[L77 disk saved vs bf16 table], 95.4[L83 size on disk]
  ours only: 
## beamster/Qwen3.8-Flash-Next-Sushi-2bpw
  agreed: 31
  auditor only: 0.199[L60 KLD vs bf16], 2[L26 bpw (routed experts)], 9[L60 extra memory vs Sushi-2bpw], 15.6[L60 memory less than iQ-MLX 3.3bpw], 29.8[L80 size on disk], 35[L60 KLD lower than Sushi-2bpw], 42[L69 LLMProbe Hard 50 correct], 65[L32 disk], 66[L75 disk saving vs bf16 table], 95.4[L81 size on disk]
  ours only: 
## beamster/Qwen3.8-Flash-Next-Sushi-3bpw
  agreed: 30
  auditor only: 0.059[L59 KLD (rounded)], 0.063[L59 KLD (rounded)], 0.104[L60 KLD (rounded)], 0.199[L60 KLD (rounded)], 20[L60 GPU memory less than oQ5e], 29.8[L80 size on disk], 49.3[L32 weights size], 66[L75 disk saved vs bf16 table], 79[L32 disk size], 95.4[L81 size on disk]
  ours only: 
## beamster/Qwen3.8-Flash-Next-Sushi-4bpw
  agreed: 31
  auditor only: 0.059[L62 KLD vs bf16], 0.063[L62 KLD vs bf16], 0.104[L63 KLD vs bf16], 0.199[L63 KLD vs bf16], 20[L63 memory less than oQ5e], 29.8[L83 size on disk], 45[L72 LLMProbe Hard 50 correct], 63.7[L33 weights size], 66[L78 disk saving vs bf16 table], 95.4[L84 size on disk]
  ours only: 
## bitlamas/Qwen3.8-Flash-Next-GGUF
  agreed: 20
  auditor only: 0.00583[L123 down squared error vs BF16], 0.00689[L123 down squared error vs BF16], 0.008[L172 KLD cost of int8 prompt matmul], 0.012[L131 KLD cost q8_0 KV], 0.019[L133 KLD cost q4_0 K / q8_0 V], 0.026[L132 KLD cost q4_0 KV], 0.027[L145 KLD shift from int8 coopmat (1], 0.03[L74 KLD (int8 coopmat off)], 0.0301[L153 KLD (b11681 coopmat off)], 0.047[L145 KLD shift from int8 coopmat (5], 1.5[L135 q4_0 KV memory saved at 262k], 1.91[L73 file size], 2.2[L71 memory freed by -ub 256], 2.79[L73 file size], 2.9117[L55 perplexity (earlier all-GPU ru], 5.5[L54 other tensors size (Q8_0)], 8.9[L44 size saved vs Q5_1 downs], 15.4[L123 down squared error change], 26.5[L181 down weighted error reduction], 28.8[L72 n-gram table size], 78.2[L71 GPU memory (b11160-mix shared ], 78.3[L71 GPU memory (mainline b11681)], 78.6[L71 GPU memory (mainline b11681)], 80.5[L104 fetched size], 94.62[L153 top token agreement (b11681 co], 95.43[L91 total size (bytes)], 360[L102 size], 1.09472e+07[L86 shard 1 bytes], 1.04627e+10[L89 shard 4 bytes], 4.45527e+10[L88 shard 3 bytes], 4.74479e+10[L87 shard 2 bytes], 1.02474e+11[L91 total size (bytes)]
  ours only: 
## btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit
  agreed: 20
  auditor only: 0.39[L103 Python code 2k KL p99], 1.23[L102 wikitext-2 4k KL p99], 8[L77 MTP size], 20[L80 per-GPU at TP4], 83.9[L102 wikitext-2 4k top-5 overlap], 86.3[L103 Python code 2k top-5 overlap], 100[L125 host RAM for table], 102[L78 n-gram table size], 250[L77 body size], 360[L77 size]
  ours only: 
## cc1244578151/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
  agreed: 8
  auditor only: 0.51[L131 GPQA-D gap vs BF16], 0.57[L83 LCB v6 gap vs BF16], 0.91[L96 file size], 1.14[L131 LCB v6 gap vs BF16], 1.52[L83 GPQA-Diamond gap vs IQ3_S], 1.6[L111 size difference], 1.89[L52 bpw (effective over original p], 4.5[L71 bpw (n-gram table)], 4.7[L131 compression vs BF16], 5.3[L131 compression vs BF16], 7.8[L83 size smaller than IQ3_S], 16[L123 bpw], 28.8[L156 resident footprint removed by ], 37.6[L91 resident size (shard 1)], 39.2[L92 resident size (shard 1)], 47[L93 resident size (shard 1)], 54.8[L94 resident size (shard 1)], 58.4[L52 total size], 76.94[L123 zero-shot avg], 77.16[L125 zero-shot avg], 77.23[L126 zero-shot avg], 78[L124 zero-shot avg], 81.14[L133 LCB v6], 83.43[L133 LCB v6], 86.29[L126 LCB v6], 86.86[L127 LCB v6], 87.37[L133 GPQA-Diamond], 87.43[L123 LCB v6], 89.07[L124 task avg], 89.16[L125 task avg], 89.39[L133 GPQA-Diamond], 91.41[L126 GPQA-Diamond], 91.92[L123 GPQA-Diamond], 92.57[L131 task average], 92.93[L127 GPQA-Diamond], 93.12[L131 task average], 93.26[L127 task avg], 95.7[L131 task avg recovery], 96.67[L131 AIME25], 99.4[L131 task avg recovery]
  ours only: 
## cc1244578151/myqwen3.8flashgsqrco
  agreed: 32
  auditor only: 0.51[L131 GPQA-D gap vs BF16], 0.57[L83 LCB v6 gap vs base], 0.91[L96 size], 1.14[L131 LCB v6 gap vs BF16], 1.52[L83 GPQA-Diamond gap vs IQ3_S], 1.6[L111 size difference vs IQ2_XS], 1.89[L52 effective bpw], 4.5[L71 bpw], 4.7[L131 size reduction], 5.3[L131 size reduction], 7.8[L83 size saving vs IQ3_S], 16[L123 bpw], 28.8[L156 resident footprint removed by ], 37.6[L91 resident size (shard 1)], 39.2[L92 resident size (shard 1)], 47[L93 resident size (shard 1)], 54.8[L94 resident size (shard 1)], 58.4[L52 total size], 95.7[L131 task avg retained], 99.4[L131 task avg recovery], 354[L131 size]
  ours only: 
## chatnerdie/Qwen3.8-Flash-Next-EXL3-Sushi
  agreed: 31
  auditor only: 0.0592[L77 KLD (Sushi card, different pro], 0.51[L75 KLD ratio vs 4bpw-sh (native)], 0.6[L75 KLD ratio vs 4bpw-sh (native)], 0.69[L75 KLD ratio vs 4bpw-sh (native)], 0.78[L75 KLD ratio vs 4bpw-sh (YaRN)], 0.84[L75 KLD ratio vs 4bpw-sh (YaRN)], 0.9[L75 KLD ratio vs SiliconSpecies 5b], 0.92[L75 KLD ratio vs SiliconSpecies 5b], 1.02[L75 KLD ratio vs SiliconSpecies 5b], 1.03[L75 KLD ratio vs SiliconSpecies 5b], 1.11[L75 KLD ratio vs SiliconSpecies 5b], 1.16[L75 KLD ratio vs SiliconSpecies 5b], 3[L25 bpw (experts)], 4[L97 needles 400K], 5[L25 bpw (experts)], 13[L70 KLD ratio vs Sushi-4bpw (nativ], 14[L27 extra memory vs 4-bit], 14.1[L53 extra memory vs Sushi-4bpw], 18[L70 KLD ratio vs Sushi-4bpw (YaRN)], 20[L71 KLD ratio vs Sushi-4bpw (nativ], 21[L71 KLD ratio vs Sushi-4bpw (YaRN)], 29.8[L51 file size], 95.4[L41 file size], 95.9[L101 peak memory], 98.9[L101 peak memory], 110[L101 peak memory]
  ours only: 
## cygnal/Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4-GGUF
  agreed: 13
  auditor only: 2.4[L103 size], 27[L38 PLE table size], 54[L38 PLE table size], 116[L38 model size], 866[L34 file size]
  ours only: 
## dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4
  agreed: 5
  auditor only: 47.5[L196 MMLU moral_scenarios], 50[L190 MMLU machine_learning], 52.5[L159 MMLU college_chemistry], 55[L208 MMLU virology], 57.5[L159 MMLU college_chemistry], 60[L204 MMLU public_relations], 62.5[L200 MMLU professional_accounting], 65[L208 MMLU virology], 67.5[L204 MMLU public_relations], 70[L205 MMLU security_studies], 72.5[L201 MMLU professional_law], 75[L205 MMLU security_studies], 77.5[L185 MMLU human_aging], 80[L168 MMLU elementary_mathematics], 82.5[L197 MMLU nutrition], 85[L199 MMLU prehistory], 87.5[L209 MMLU world_religions], 90[L206 MMLU sociology], 92.5[L207 MMLU us_foreign_policy], 95[L209 MMLU world_religions], 97.5[L206 MMLU sociology], 100[L198 MMLU philosophy]
  ours only: 0.97, 422, 434
## devan-carlin/Qwen3.8-Flash-Next-W4A16
  agreed: 1
  auditor only: 19[L51 VRAM per GPU], 96[L77 PLE table size]
  ours only: 
## donught/Qwen3.8-Flash-Next-NVFP4-dense
  agreed: 6
  auditor only: 76[L38 GPU-resident], 102[L30 PLE table size], 360[L66 size]
  ours only: 378.9, 450.9
## drluoto/Qwen3.8-Flash-Next-AgenticRequant-Q5K-GGUF
  agreed: 7
  auditor only: 0.01[L141 shard 1 size], 0.25[L174 router F32 read per token], 0.7[L186 Q6_K tensors size], 0.9[L185 IQ4_XS tensors size], 1.4[L103 size difference], 2.1[L191 Q5_K bytes], 4.49[L85 bytes read per token], 5.6[L183 Q8_0 tensors size], 5.84[L84 bytes read per token], 23[L86 bytes/token change], 33.9[L182 IQ3_S tensors size], 42.9[L143 shard 3 size], 49.1[L181 IQ4_NL tensors size], 49.4[L142 shard 2 size]
  ours only: 
## edougawa/Qwen3.8-Flash-Next-W4A16-PLE4-MTP-Spark
  agreed: 2
  auditor only: 0.016[L86 stored size ple_support], 0.113[L89 stored size shared_experts], 0.117[L88 stored size routers], 0.305[L83 stored size lm_head], 0.324[L80 stored size full_attention], 0.748[L81 stored size hyper_connection], 0.836[L91 stored size vision], 1.004[L82 stored size linear_attention], 1.184[L90 stored size token_embedding], 1.27[L84 stored size mtp], 4.1[L85 bpw ple_embedding], 4.125[L89 bpw shared_experts], 4.133[L82 bpw linear_attention], 4.151[L86 bpw ple_support], 4.183[L84 bpw mtp], 4.503[L80 bpw full_attention], 10.035[L81 bpw hyper_connection], 16[L91 bpw vision], 24.438[L85 stored size ple_embedding], 58.008[L87 stored size routed_experts], 88.39[L123 resident weight artifact], 9.49129e+10[L40 safetensors size (bytes)]
  ours only: 
## eemin/halogen-qwen3.8-flash-next-copy
  agreed: 2
  auditor only: 0.09[L133 MTP dense projections size], 0.84[L50 vision file size], 2.31[L49 speed overlay size], 5[L58 perplexity worse without overl], 9[L58 perplexity worse without overl], 47.7[L120 n-gram table FP8 size], 68[L61 resident memory], 106[L127 o_proj tensors size], 118[L61 download size together], 173[L112 size], 335[L112 size]
  ours only: 
## evsinlb/Qwen3.8-Flash-Next-oQ4e-fp16-mtp
  agreed: 3
  auditor only: 5[L24 PLE bits/param], 29.82[L35 PLE table size], 73.7[L60 peak memory pp1024], 81.1[L61 peak memory pp4096], 84.1[L62 peak memory pp8192]
  ours only: 
## fuutott/Qwen3.8-Flash-Next-MLX-t5-imatrix-q3down-ple8
  agreed: 58
  auditor only: 0.012[L114 KLD difference vs all-affine Q], 0.12[L115 imatrix KLD gain over weight-o], 1.875[L29 gate/up bpw], 3.297[L102 perplexity wikitext-2], 36.8[L65 resident weights on disk], 39.45[L86 MLX allocator peak at 8K], 41.4[L81 physical peak 32K], 41.5[L79 physical peak 8K], 42.3[L82 physical peak 64K], 43.4[L83 physical peak 98K (rejected)], 43.5[L84 physical peak 129K], 53.6[L191 PLE table size], 90.4[L63 all shards size], 360[L28 size], 3.94596e+10[L65 resident weights on disk bytes], 5.76003e+10[L64 PLE shards size bytes]
  ours only: 
## gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis
  agreed: 26
  auditor only: 0.099[L85 KLD long window code], 0.128[L85 KLD long window prose], 0.97[L44 file size (bytes)], 1.2[L76 perplexity (BF16 in parens) py], 1.21[L76 perplexity (BF16 in parens) py], 1.53[L77 perplexity (BF16 in parens) ma], 1.8[L73 perplexity (BF16 in parens) ch], 2.5[L56 bpw (gate/up mean)], 3.44[L74 perplexity (BF16 in parens) co], 3.48[L74 perplexity (BF16 in parens) co], 3.58[L83 perplexity (BF16 in parens) mm], 3.74[L83 perplexity (BF16 in parens) mm], 3.93[L80 perplexity (BF16 in parens) de], 4.41[L78 perplexity (BF16 in parens) en], 4.55[L78 perplexity (BF16 in parens) en], 4.73[L79 perplexity (BF16 in parens) it], 4.78[L79 perplexity (BF16 in parens) it], 7.44[L81 perplexity (BF16 in parens) ja], 7.85[L81 perplexity (BF16 in parens) ja], 9.23[L75 perplexity (BF16 in parens) pr], 9.31[L75 perplexity (BF16 in parens) pr], 15.42[L82 perplexity (BF16 in parens) zh], 15.77[L82 perplexity (BF16 in parens) zh], 28.8[L58 n-gram table size], 38[L97 pinned experts host RAM], 46[L97 host RAM needed], 48[L97 host RAM needed], 66.83[L42 file size (bytes)], 72.8[L103 files on disk], 102[L88 served argmax match vs quantiz], 1.03791e+09[L44 file size (bytes)]
  ours only: 
## impulsai/CHImp-Alpha-v1.0
  agreed: 15
  auditor only: 0.05[L264 KL threshold for expert format], 0.6[L246 shard 2 size], 1.8[L248 MTP helper size], 2.3[L205 VRAM saving without helper], 3.8[L175 LM head etc], 4.25[L264 expert input bpw (IQ4_XS)], 4.5[L264 expert output bpw], 6.6[L265 other tensors bpw], 8.5[L265 other tensors bpw], 21.1[L176 expert output weights], 24.9[L174 GPU weights], 26.8[L247 shard 3 (n-gram) size], 31[L234 VRAM at 64k with helper], 31.8[L235 VRAM tuned 96k], 39.8[L177 expert input weights], 40.5[L182 RAM in constant use], 57.9[L112 BFCL irrelevance], 62.5[L115 BFCL live_parallel], 64.8[L245 shard 1 size], 65.5[L110 BFCL parallel], 67[L111 BFCL parallel_multiple], 67.3[L183 total mapped], 73.9[L81 BFCL v4 single-turn plain mean], 75[L116 BFCL live_parallel_multiple], 75.6[L114 BFCL live_multiple], 76[L113 BFCL live_simple], 81.2[L118 BFCL live_relevance], 175.3[L66 download size]
  ours only: 0, 8
## intel-ai/Qwen3.8-Flash-Next-MXFP4-Mixed-CT-AutoRound
  agreed: 12
  auditor only: 0.26[L74 MMLU delta], 0.33[L74 PIQA delta], 0.53[L74 GSM8K delta], 0.95[L77 HellaSwag delta], 88[L100 extra memory without PLE offlo], 99.6[L33 recovery], 102[L55 PLE table size (BF16)], 360[L43 size on disk]
  ours only: 
## intel-ai/Qwen3.8-Flash-Next-MXFP8-CT-AutoRound
  agreed: 14
  auditor only: 0.12[L195 shared expert BF16 size overhe], 0.14[L165 hellaswag delta], 0.16[L165 piqa delta], 0.3[L165 gsm8k delta], 0.7936[L135 MMLU Humanities], 0.8313[L129 piqa acc_norm], 0.857[L135 MMLU STEM], 0.8761[L135 MMLU Other], 0.8771[L132 hellaswag acc_norm], 0.9093[L135 MMLU Social Sciences], 0.9712[L127 gsm8k flexible], 1.5[L53 compression ratio], 10.51[L54 effective bits/param (weights ], 16[L54 bits/param], 23.8[L231 n-gram table per rank TP=4], 47.7[L231 n-gram table per rank TP=2], 95.37[L230 n-gram table VRAM total], 102.4[L105 n-gram table size], 111.49[L232 weights per rank TP=2], 180[L106 size], 223[L152 weights total (TP=2)], 223.88[L52 size on disk], 224[L266 output size], 335.28[L52 size on disk], 360[L52 size on disk]
  ours only: 0.0029, 0.0047
## jcbtc/Qwen3.8-Flash-CIRU-STRIX-Orca
  agreed: 11
  auditor only: 2.0516[L205 tail PPL], 2.0595[L204 tail PPL], 6.785[L107 disk saving vs original Orca G], 6.88[L109 GPU-visible memory saving], 10[L289 HumanEval 0–9], 20[L287 HumanEval 0–19], 67.16[L107 file size], 83.2[L205 label accuracy], 83.5[L205 label accuracy], 92.5[L288 HermesAgent 20 mean scores], 95.25[L290 HermesAgent 20 mean], 98.5[L288 HermesAgent 20 mean scores], 100[L290 HermesAgent 20 mean], 120[L268 required files total], 4.13589e+09[L219 file size bytes]
  ours only: 
## jlkivey/Qwen3.8-Flash-Next-MTP-PR27836-GGUF
  agreed: 1
  auditor only: 2.6[L46 added disk on graft], 3.5[L67 resident memory cost], 7.7[L84 streamed size], 360[L84 checkpoint size]
  ours only: 
## jomcgi-org/Qwen3.8-Flash-Next-NVFP4-oominf
  agreed: 17
  auditor only: 10.1[L175 dense.bin size], 51.2[L174 tables.bin size], 73.1[L173 experts.bin size], 134[L74 model weights size]
  ours only: 
## junafinity/Qwen-3.8-Flash-Next-Uncensored-MLX-MXFP4
  agreed: 1
  auditor only: 0.085[L163 abliteration KL], 1.3[L152 WikiText PPL delta], 4.4708[L152 WikiText PPL (PipeNetwork)], 4.5286[L152 WikiText PPL], 20.6[L152 WikiText PPL delta], 76.2[L198 MMLU-Pro CoT], 77.8[L198 MMLU-Pro CoT (orcarouter eval)], 81.6[L200 CMMLU], 81.8[L200 CMMLU (orcarouter eval)], 87.7[L197 MMLU (0-shot letter)], 90[L197 MMLU (0-shot letter, orcaroute], 92[L199 GSM8K CoT (orcarouter eval)], 93.3[L199 GSM8K CoT], 93.63[L109 weights size], 103[L112 practical working set], 110[L112 practical working set], 163[L70 size]
  ours only: 
## khuasar/Qwen3.8-Flash-Next-REAP-288-Q4g32-Q8-MTPLX
  agreed: 5
  auditor only: 0.836[L56 vision weight payload], 1.583[L55 MTP head payload], 29.802[L57 SSD n-gram table payload], 44.39[L54 main-model weight payload], 49.399[L60 peak MLX allocation 65,536-tok], 49.982[L61 peak MLX allocation MTP task s], 53.45[L63 peak synthetic cache-copy diag]
  ours only: 
## kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-FAST-imatrix-GGUF
  agreed: 8
  auditor only: 0.91[L202 mmproj size], 1.5[L164 PPL cost of 4-bit head], 4.251[L155 bpw (ROCmFP4 tensors)], 5.06[L201 shard 3 size], 44.67[L200 shard 2 size], 44.7[L199 shard 1 size], 63.3[L189 GTT memory]
  ours only: 
## kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-imatrix-GGUF
  agreed: 8
  auditor only: 0.9[L60 PPL change vs FAST imatrix], 0.91[L188 size], 5.9[L67 isolated imatrix PPL effect], 16.1[L187 shard 3 size], 44.7[L186 shard 2 size], 44.9[L185 shard 1 size], 63.3[L178 GTT memory]
  ours only: 
## klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP
  agreed: 3
  auditor only: 16[L39 PLE table size (SSD)], 95.37[L39 PLE table size (SSD)]
  ours only: 
## kstoyanov99/qwen38-flash-next-nvfp4
  agreed: 7
  auditor only: 1[L139 AIME26 pass@1], 2.7[L99 size reduction], 8[L139 AIME26 majority@8], 51[L125 PLE n-gram table size], 118.4[L155 unchanged tensors size], 360[L99 size]
  ours only: 
## lackonendes/PAW-125B-FLASH-NEXT-X3.1
  agreed: 27
  auditor only: 0.97[L156 MTP head size], 2.5[L23 bpw (cold experts)], 3[L23 bpw (experts average)], 3.5[L23 bpw (hot experts)], 3.9[L59 KV+state at 262k], 4.6[L57 dense weights on GPU], 6.4[L65 hot experts on GPU], 13.5[L58 routed experts on GPU], 23.5[L69 GPU memory per card (2 cards)], 23.6[L60 GPU total at 256k], 25[L71 RAM (2 cards)], 38.8[L61 routed experts in CPU RAM], 42[L65 routed experts total size], 43[L62 server resident], 47.54[L154 weights shard size], 47.7[L56 n-gram table on disk], 48.28[L155 n-gram shard size], 48.3[L28 n-gram shard size], 95.8[L18 total size], 96.8[L63 disk total both files], 97[L148 disk size]
  ours only: 16
## leoncca/Qwen3.8-Flash-Next-AWQ-g32
  agreed: 15
  auditor only: 0.0273[L139 KL Documents/code], 0.0372[L138 KL Dialogue], 0.0427[L137 KL Instructions], 4.88[L201 MTP draft payload], 32[L30 KL decrease], 32.04[L145 overall KL decrease], 93.78[L137 top-1 Instructions], 94.71[L138 top-1 Dialogue], 97.11[L139 top-1 Documents/code], 127.66[L19 size]
  ours only: 
## lesj0610/Qwen3.8-Flash-Next-NVFP4
  agreed: 3
  auditor only: 0.095[L168 expert 511 down_proj cos/rel e], 0.9953[L166 expert 7 down_proj cos/rel err], 0.99532[L167 expert 255 down_proj cos/rel e], 0.99539[L168 expert 511 down_proj cos/rel e], 0.99543[L166 expert 7 up_proj cos/rel err v], 0.99544[L167 expert 255 up_proj cos/rel err], 0.99545[L168 expert 511 gate_proj cos/rel e], 0.99546[L168 expert 511 up_proj cos/rel err], 0.99551[L165 expert 0 down_proj cos/rel err], 102.4[L230 n-gram table size], 360[L29 size]
  ours only: 
## lhoestq/Qwen3.8-Flash-Next-NVFP4-Michel
  agreed: 18
  auditor only: 2.7[L188 size reduction vs BF16], 63[L188 disk size reduction]
  ours only: 8
## lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8
  agreed: 6
  auditor only: 1[L226 AIME26 pass@1], 2.7[L183 compression vs BF16], 98.86[L225 GSM8K stop rate], 99.17[L226 AIME26 stop rate], 118.4[L242 unchanged tensors size], 240[L226 AIME26], 360[L183 checkpoint size]
  ours only: 
## lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000
  agreed: 5
  auditor only: 1[L196 AIME26 pass@1], 2.7[L167 size reduction], 8[L196 AIME26 majority@8], 118.4[L212 unchanged tensors size], 135[L167 checkpoint size], 240[L196 AIME26], 360[L167 size]
  ours only: 
## manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64
  agreed: 5
  auditor only: 0.84[L49 vision tower size], 3.3[L36 bpw], 4[L201 delayed literal recall], 6.2[L224 PPL improvement vs pre-rebuild], 17.88[L378 n-gram table size], 17.9[L31 n-gram table size], 19.2[L116 planner n-gram table], 21.86[L115 planner resident], 33.97[L114 planner experts], 34[L32 GPU memory with --ngram-offloa], 34.13[L379 GPU memory with offload], 34.4[L302 streamed-experts peak], 36.2[L374 vision peak with offload], 36.64[L123 projected weights at 16K (offl], 40.18[L128 projected peak at 16K (48GB cl], 42.19[L84 projected peak at 32K (offload], 47.1[L55 wired peak 16K with offload], 50.6[L37 size], 52.01[L380 GPU memory without offload], 54.1[L374 vision peak without offload], 55.84[L113 planner weights (exact)], 58.1[L51 resident peak with image], 58.28[L153 peak 4,110 tok], 58.67[L158 projected at 16K], 59.43[L158 measured at 15,700], 59.48[L159 projected at 32K], 60.93[L159 measured at 29,782], 63[L32 size], 63.4[L55 wired peak 16K without offload], 124[L184 size if n-gram table unquantiz]
  ours only: 5, 7, 13, 15
## meldlabs/Qwen3.8-Flash-Next-Meld-Turbo-Quality-GGUF
  agreed: 10
  auditor only: 2.625[L33 bpw (routed experts)]
  ours only: 0, 8
## mensaprodigy/Qwen3.8-Flash-Next-REAP-288-oQ4e-BF16-MTP-PLE
  agreed: 8
  auditor only: 4[L29 base bits], 43.3[L52 peak memory 1k], 46[L53 peak memory 4k], 46.1[L54 peak memory 8k], 1054[L42 LiveCodeBench (100 sample)]
  ours only: 
## mensaprodigy/Qwen3.8-Flash-Next-REAP-384-oQ4e-BF16-MTP-PLE
  agreed: 5
  auditor only: 4[L30 base bits], 58.7[L68 peak memory 1k], 61.5[L69 peak memory 4k], 61.6[L70 peak memory 8k]
  ours only: 
## mensaprodigy/Qwen3.8-Flash-Next-REAP-384-oQ5e-BF16-MTP-PLE
  agreed: 9
  auditor only: 5[L29 bits (primary)], 70[L99 peak memory 1k], 72.3[L65 peak memory 4k], 72.8[L100 peak memory 4k], 72.9[L101 peak memory 8k], 73.2[L111 peak memory 16k], 100[L85 LiveCodeBench Thinking Yes]
  ours only: 
## mk2878d/Qwen3.8-Flash-Next-GGUF
  agreed: 19
  auditor only: 0.0005[L182 mean KLD change], 0.0155[L136 PPL over imatrix calibration s], 0.23[L204 everything else bpw contributi], 1.03[L203 ffn_down_exps bpw contribution], 1.24[L202 ffn_gate/up_exps bpw contribut], 1.74[L201 n-gram table bpw contribution], 4.6812[L136 PPL over imatrix calibration s], 6.8[L189 top-1 gain vs 3.84bpw], 17.6[L87 size smaller than 5.00bpw], 38.4[L83 on SSD], 39[L60 n-gram table size], 39.1[L82 on SSD], 54.4[L84 on SSD], 63[L189 mean KLD cut vs 3.84bpw recipe], 85[L43 file size], 37265[L52 host model memory (MiB)], 43720[L51 Metal model memory (MiB)]
  ours only: 
## munekazu/Qwen3.8-Flash-Next-exl3-CMP170HX-64GB
  agreed: 9
  auditor only: 3[L75 bits (experts layers 10-37)], 4[L74 bits (experts layers 0-9, 38-4], 6[L72 bits (attention/shared/lm_head], 39[L121 n-gram table size]
  ours only: 
## myllmbox/Qwen3.8-Flash-Next-hibrid48
  agreed: 6
  auditor only: 0.33[L24 lm_head size (NVFP4)], 1.18[L21 lm_head size (bf16)], 4[L40 lm_head bits], 9[L40 lm_head mean |Δw| relative], 16[L40 lm_head bits], 28.6[L44 PLE n-gram table NVFP4 size], 39[L38 routed experts share of checkp], 50[L96 memory per box (2 Sparks)]
  ours only: 
## nazeshinjite/Qwen3.8-Flash-Next-Q8-UDQ4K-DS4
  agreed: 35
  auditor only: 0.0004[L60 short-set NLL change], 0.0008[L64 short-set NLL vs reference], 0.0029[L65 long set NLL change], 0.03[L118 KLD (Unsloth published)], 0.047[L117 KLD (Unsloth published)], 0.2903[L95 avg NLL], 0.78[L41 per-layer cost of Q8_0 gate/up], 2.34[L78 extra size layers 0/1/45], 3.5[L114 size less than UD-Q5], 16[L113 resident size delta vs this fi], 20.34[L34 down experts size], 20.5[L40 size added by Q8_0 downs], 21.53[L34 gate/up experts size each], 69.74[L34 resident size], 92.26[L117 top-1 vs BF16 (Unsloth publish], 93.2[L109 resident], 93.68[L118 top-1 vs BF16 (Unsloth publish], 94.12[L118 top-1 vs BF16 (Unsloth publish], 95.37[L28 BF16 n-gram table size], 96.87[L99 long set top-1], 97.39[L99 long set top-1], 106.6[L149 total memory 131k], 109.68[L150 total memory 196k], 112.8[L151 total memory 262k], 165.11[L34 file size], 588[L26 mmproj size], 616[L170 file size]
  ours only: 
## nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M
  agreed: 13
  auditor only: 2.4[L383 bpw], 46[L453 host pinned memory for cold ex], 104.73[L549 payload], 105[L469 container size], 611[L440 vision linears F16 VRAM]
  ours only: 
## nitinpanj/qwen38-flash-next-v3
  agreed: 4
  auditor only: 1.69[L166 disk size delta vs unspliced], 1.9[L124 draft head size (vs unshared 2]
  ours only: 0.751, 0.817
## nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4
  agreed: 11
  auditor only: 0.84[L133 vision size], 4.86[L131 MTP size], 9.15[L132 attention/shared/norms/head si], 51.8[L31 size ratio vs BF16], 63.28[L129 routed experts size], 95.43[L130 PLE size], 173.6[L31 size], 225[L129 routed experts size], 335.28[L134 total size], 335.3[L31 size]
  ours only: 
## perkel/Qwen3.8-Flash-Next-MC
  agreed: 14
  auditor only: 0.019[L85 KL math+code at 32K], 0.02[L70 KL held-out math], 0.038[L70 KL held-out code], 0.04[L70 KL held-out reasoning], 0.052[L85 KL math+code at 32K], 0.066[L71 KL held-out wiki], 0.07[L70 KL held-out math], 0.086[L70 KL held-out code], 0.098[L70 KL held-out reasoning], 0.143[L71 KL held-out chat], 0.163[L71 KL held-out wiki], 0.31[L71 KL held-out chat], 0.56[L46 size], 0.99[L47 size], 1.4[L45 size], 2[L30 routed experts bits], 3[L31 routed experts bits], 3.05[L31 bpw], 5.1[L43 size], 18[L56 min experts in VRAM], 26.3[L44 size], 28.6[L52 routed experts size], 30.7[L41 model file size], 42.6[L52 routed experts size], 45.8[L42 model file size]
  ours only: 
## petergilani/Qwen3.8-Flash-Next-MLX-6bit
  agreed: 2
  auditor only: 102.4[L33 n-gram table size], 105.63[L92 active memory after eval], 106[L32 weights size], 1.024e+11[L90 table bytes]
  ours only: 
## pipenetwork/Qwen3.8-Flash-Next-MLX-4bit
  agreed: 33
  auditor only: 2.4[L89 size over 4bit], 4[L21 bits], 102[L68 n-gram tables size if bf16]
  ours only: 
## pipenetwork/Qwen3.8-Flash-Next-MLX-6bit
  agreed: 25
  auditor only: 1.3[L101 PPL vs bfloat16], 2.4[L89 size delta vs 4bit], 4.8969[L99 perplexity], 4.9744[L98 perplexity], 5.2843[L100 perplexity], 9.5[L99 PPL vs bfloat16], 11.3[L98 PPL vs bfloat16], 18.2[L100 PPL vs bfloat16], 20.6[L97 PPL vs bfloat16], 102[L68 size if bf16]
  ours only: 
## pipenetwork/Qwen3.8-Flash-Next-MLX-8bit
  agreed: 33
  auditor only: 2.4[L89 size over 4bit], 8[L21 bits], 102[L68 n-gram tables size if bf16]
  ours only: 
## pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit
  agreed: 33
  auditor only: 2.4[L89 size delta vs 4bit], 102[L68 size if bf16]
  ours only: 
## pugant/Qwen3.8-Flash-Next-ROCMFP4_STRIX_LEAN-GGUF
  agreed: 18
  auditor only: 0.006[L130 PPL Italian holdout (CPU)], 0.244[L277 F32 size], 0.317[L332 FP2 family relative RMSE], 0.359[L332 FP2 family relative RMSE], 0.407[L276 Q6_K / Q5_K size], 0.486[L276 Q6_K / Q5_K size], 0.6[L125 PPL Italian delta], 0.949[L333 PLE cosine with source], 1[L236 holdout improvement vs v1], 1.095[L130 PPL Italian holdout (fork runt], 1.0952[L165 PPL memorized holdout (MTP por], 1.156[L130 PPL Italian holdout (CPU)], 2.07[L236 PPL holdout], 2.17[L324 PPL Dante 8k / EN 8k], 2.28[L330 EN PPL multiple vs pre-09-02 L], 3.27[L236 PPL English], 3.345[L275 Q4_0_ROCMFP4 size], 3.85[L148 size], 4.443[L274 Q3_0_ROCMFPX size], 4.9[L126 PPL English delta], 7.29[L28 PPL EN], 16.1[L53 shard sizes], 18.3[L236 holdout improvement vs v1], 31.173[L273 Q2_0_ROCMFPX size], 32[L324 PPL Dante 8k / EN 8k], 35.76[L340 PLE Q5_1 disk-resident size], 35.763[L272 Q5_1 PLE size], 37.9[L28 PPL Dante 8k], 39.56[L266 shard 2 size], 41.91[L266 shard 1 size], 44.7[L53 shard sizes], 44.9[L53 shard sizes], 53.12[L217 size on disk], 55[L327 host peak ctx 32k], 56[L217 resident], 64[L144 resident at 8k ctx], 87.2[L200 size sharded], 93.7[L200 size merged], 98.5[L67 file size], 354[L314 size]
  ours only: 
## qtum/Qwen3.8-Flash-Next-GGUF
  agreed: 19
  auditor only: 14[L52 PPL improvement vs uniform], 15.73[L39 bpw]
  ours only: 
## r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw
  agreed: 14
  auditor only: 3[L27 n-gram table bpw], 4[L27 MTP head bpw], 18.5[L36 n-gram table size], 20.4[L85 VRAM in use mean], 20.5[L85 VRAM in use max], 20.7[L46 VRAM peak full-context], 21.3[L47 VRAM loaded short-context], 22[L73 CPU expert tail mcs 320], 22.4[L47 VRAM peak short-context], 26.5[L74 CPU expert tail mcs 384], 41.5[L35 quantized weights size], 64.6[L37 repo total], 335.3[L39 size]
  ours only: 
## rapid-mlx/Qwen3.8-Flash-Next-4bit
  agreed: 11
  auditor only: 6.6[L134 extra active memory with MTP], 45[L156 correctness battery], 99[L92 weights size], 102.8[L114 active memory], 103[L46 active memory serving], 103.8[L114 active memory], 148.1[L115 peak allocator memory], 335[L215 size]
  ours only: 
## scorpoon/Qwen3.8-Flash-Next-exl3-SC
  agreed: 22
  auditor only: 0.0027[L55 KLD noise floor], 0.0576[L66 mean KLD (rounded)], 2.09[L33 bpw (reported, measured)], 3.09[L31 bpw (reported, measured)], 3.19[L32 bpw (reported, measured)], 5[L33 weight shards size (VRAM)], 7[L32 weight shards size (VRAM)], 8[L33 bpw (reported, measured)], 16[L66 KLD reduction vs turboderp 2.0], 36.4[L51 n-gram table size], 57[L49 steady-state VRAM across 2 car], 58[L49 steady-state VRAM across 2 car]
  ours only: 
## scouzi1966/Qwen3.8-Flash-Next-AFM-MLX-4bit
  agreed: 0
  auditor only: 4[L36 main projections bits], 64[L36 main projections bits]
  ours only: 
## showkieo/Qwen3.8-Flash-Next-Sushi-3bpw
  agreed: 23
  auditor only: 0.063[L123 KLD (rounded)], 0.105[L124 KLD (rounded)], 0.199[L124 KLD (rounded)], 0.7[L55 top-1 drop with 4-bit KV], 0.7805[L78 NLL of generated text], 0.7819[L79 NLL of generated text], 0.84[L93 GPU memory: vision tower], 0.98[L92 GPU memory: MTP head], 1.27[L92 GPU memory: MTP head], 4.06[L95 GPU memory: KV cache, 256k tok], 5.6[L125 GPU memory less than affine q3], 8.12[L96 GPU memory: KV cache, 512k tok], 9[L55 KLD increase with 4-bit KV], 16.25[L97 GPU memory: KV cache, 1M token], 20[L123 GPU memory less than oQ5e], 27[L124 KLD reduction vs MLX affine q3], 29.8[L138 size on disk], 47.51[L91 GPU memory: model weights], 49.3[L32 weights size], 53.4[L98 GPU memory: total at 256k / 51], 57.5[L98 GPU memory: total at 256k / 51], 61.58[L91 GPU memory: model weights], 65.6[L98 GPU memory: total at 256k / 51], 66[L133 disk saved vs bf16 table], 67.7[L98 GPU memory: total at 256k / 51], 71.8[L98 GPU memory: total at 256k / 51], 79[L32 disk size], 79.9[L98 GPU memory: total at 256k / 51], 95.4[L139 size on disk]
  ours only: 
## sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF
  agreed: 122
  auditor only: 0[L281 MKQA delta vs FLE-IQ3_M], 0.0031[L244 internal text E KLD delta], 0.0079[L244 internal text F KLD delta vs U], 0.029[L248 KLD delta vs FLE-IQ3_M], 0.036[L248 KLD delta vs FLE-IQ3_M], 0.056[L248 KLD delta vs FLE-IQ3_M], 0.0905[L259 KLD GPU0 vs GPU1], 0.0975[L259 KLD GPU1], 0.132[L293 KLD E / W / K], 0.135[L293 KLD E / W / K], 0.2[L280 MKQA delta vs FLE-IQ3_M], 0.261[L293 KLD E / W / K], 0.5[L280 MKQA delta vs FLE-IQ3_M], 0.8[L281 MKQA delta vs FLE-IQ3_M], 1.53[L83 size], 1.8[L281 MKQA delta vs FLE-IQ3_M], 2.2[L280 MKQA delta vs FLE-IQ3_M], 2.5[L282 MKQA delta vs FLE-IQ3_M], 2.59[L194 experts bpw], 2.9[L246 size saving vs UD-IQ1_M], 3.08[L193 experts bpw], 3.2[L281 MKQA delta vs FLE-IQ3_M], 3.62[L195 experts bpw], 4.05[L195 total bpw], 4.2[L280 MKQA delta vs FLE-IQ3_M], 4.31[L194 embeddings/output bpw], 4.5[L281 MKQA delta vs FLE-IQ3_M], 4.51[L194 PLE table bpw], 4.53[L193 embeddings/output bpw], 4.8[L241 size saving vs UD-IQ4_XS], 4.81[L31 size saving vs FLE-IQ3_M], 5[L280 MKQA delta vs FLE-IQ3_M], 5.8[L282 MKQA delta vs FLE-IQ3_M], 6.5[L281 MKQA delta vs FLE-IQ3_M], 6.61[L194 attention/shared/norms bpw], 6.9[L247 size saving vs FLE-IQ3_M], 7.32[L193 attention/shared/norms bpw], 7.54[L195 embeddings/output bpw], 7.8[L282 MKQA delta vs FLE-IQ3_M], 8.5[L283 MKQA delta vs FLE-IQ3_M]
  ours only: 
## sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF
  agreed: 58
  auditor only: 0[L109 MKQA delta vs FLE-IQ3_M], 0.029[L91 KLD increase vs FLE-IQ3_M E/W/], 0.036[L91 KLD increase vs FLE-IQ3_M E/W/], 0.2[L108 MKQA delta vs FLE-IQ3_M], 0.5[L108 MKQA delta vs FLE-IQ3_M], 0.8[L109 MKQA delta vs FLE-IQ3_M], 1.6[L92 size delta vs this], 1.8[L109 MKQA delta vs FLE-IQ3_M], 2.2[L108 MKQA delta vs FLE-IQ3_M], 2.5[L115 size smaller than REAP448], 3.2[L109 MKQA delta vs FLE-IQ3_M], 3.62[L69 bpw routed experts], 4.02[L70 bpw total], 4.2[L108 MKQA delta vs FLE-IQ3_M], 4.5[L109 MKQA delta vs FLE-IQ3_M], 5[L108 MKQA delta vs FLE-IQ3_M], 5.8[L110 MKQA delta vs FLE-IQ3_M], 6.5[L109 MKQA delta vs FLE-IQ3_M], 6.9[L91 size reduction vs FLE-IQ3_M], 7.54[L70 bpw embeddings/output], 7.8[L110 MKQA delta vs FLE-IQ3_M], 8.5[L111 MKQA delta vs FLE-IQ3_M], 9[L70 bpw attention/shared/norms], 9.05[L71 bpw attention/shared/norms], 9.2[L110 MKQA delta vs FLE-IQ3_M], 11.8[L110 MKQA delta vs FLE-IQ3_M], 12[L111 MKQA delta vs FLE-IQ3_M], 13.2[L111 MKQA delta vs FLE-IQ3_M], 15.8[L111 MKQA delta vs FLE-IQ3_M], 16[L110 MKQA delta vs FLE-IQ3_M], 21.8[L111 MKQA delta vs FLE-IQ3_M], 26.8[L147 n-gram table mmapped], 37.09[L49 shard 2 size], 44.91[L48 shard 1 size], 46.1[L146 pinned RAM], 48[L146 system RAM taken], 52.2[L147 pinned RAM], 54.3[L147 system RAM taken], 79.5[L89 size (GB)], 88.9[L85 size (GB)]
  ours only: 
## sorenmat/Qwen3.8-Flash-Next-NVFP4
  agreed: 18
  auditor only: 2.7[L186 size reduction vs BF16], 63[L186 disk size reduction]
  ours only: 8
## spectator2026/Qwen3.8-Flash-Next-W6-HQQ-g32
  agreed: 6
  auditor only: 6[L58 bits], 95[L92 n-gram table host RAM], 128[L92 host shared memory per TP2 eng], 209[L25 size (GiB)], 360[L25 size]
  ours only: 
## spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF
  agreed: 37
  auditor only: 55.8[L53 Q8_0 trunk size]
  ours only: 
## tapsin/TAPSIN-Qwen3.8-4B-Distill-GGUF
  agreed: 5
  auditor only: 0.065[L23 gsm8k_cot delta], 0.199[L22 mmlu CoT delta], 0.354[L22 mmlu CoT], 0.553[L22 mmlu CoT], 0.785[L23 gsm8k_cot], 0.85[L23 gsm8k_cot]
  ours only: 
## tawei-amd/Qwen3.8-Flash-Next-strixflash-GGUF
  agreed: 0
  auditor only: 4.4[L62 weights read per decoded token], 3.75532e+08[L32 shard 1 bytes], 2.78657e+09[L31 MTP head bytes], 4.10366e+09[L37 shard 6 bytes], 1.96086e+10[L35 shard 4 bytes], 1.97197e+10[L36 shard 5 bytes], 1.97904e+10[L34 shard 3 bytes], 2.88001e+10[L33 shard 2 bytes]
  ours only: 
## tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8
  agreed: 1
  auditor only: 2.7[L24 attention FP8 size], 51.3[L23 n-gram FP8 size], 64.2[L22 experts MXFP4 size]
  ours only: 
## tls20/Qwen3.8-Flash-Next-oQ5e-mtp
  agreed: 6
  auditor only: 5[L26 bits], 84.7[L42 peak memory (no MTP) pp 1024 /], 84.9[L43 peak memory (no MTP) pp 4096 /], 85.7[L44 peak memory (no MTP) pp 8192 /], 86.6[L55 peak memory (MTP) pp 1024 / tg], 86.7[L56 peak memory (MTP) pp 4096 / tg], 87.4[L46 peak memory (no MTP) pp 32768 ], 87.8[L57 peak memory (MTP) pp 8192 / tg], 88.8[L47 peak memory (no MTP) pp 65536 ], 89[L58 peak memory (MTP) pp 16384 / t], 89.9[L59 peak memory (MTP) pp 32768 / t], 91.4[L60 peak memory (MTP) pp 65536 / t], 96.3[L61 peak memory (MTP) pp 131072 / ], 96.5[L49 peak memory (no MTP) pp 200000], 99.6[L62 peak memory (MTP) pp 200000 / ]
  ours only: 
## txgsync/Qwen3.8-Flash-Next-Dynamic-oQ5e-BF16-PLE-mtp
  agreed: 12
  auditor only: 0[L92 KL from BF16], 0.0632[L96 perplexity lower than Mac prox], 0.0885[L96 top-1 lower than Mac proxy], 0.1039[L105 perplexity lower than Mac prox], 0.5945[L96 KL lower than Mac proxy], 3.1521[L105 KL lower than Mac proxy], 5.008[L72 quantized tensors average bits], 5.21[L77 BF16 MTP size], 22.2[L128 llm-bench coding], 73.8[L131 llm-bench coding], 75[L42 llm-bench coding], 75.9[L130 llm-bench coding], 79[L129 llm-bench coding], 82.15[L130 llm-bench role-play], 85[L131 llm-bench role-play], 90.55[L129 llm-bench role-play], 96.3[L128 llm-bench role-play], 100[L92 top-1 agreement], 102.4[L76 BF16 PLE size]
  ours only: 
## txgsync/Qwen3.8-Flash-Next-MXFP4-MLX
  agreed: 1
  auditor only: 6.75[L84 cache at 256K], 92.2624[L39 payload size], 92.27[L115 active memory after load], 95.4[L50 size], 96.94[L117 peak memory multimodal], 103[L86 expected working set], 107[L86 expected working set], 3.6e+11[L104 source weight payload bytes]
  ours only: 
## wemoh/Qwen3.8-Flash-Next-strixw
  agreed: 6
  auditor only: 0.66[L116 n-gram median row error], 0.84[L118 vision memory], 13[L104 KV cache at 512k], 48.3[L105 n-gram table size], 66[L103 resident weights], 116[L48 download size], 360[L48 size]
  ours only: 
## whm0627/Qwen3.8-Flash-Next-177B-A3B-fits64GB-PLElast-GGUF
  agreed: 9
  auditor only: 1.9[L71 file size], 27[L95 embedding table size], 42[L96 VRAM use], 44[L60 VRAM/wired use], 48[L60 VRAM/wired use]
  ours only: 
## xv0y5ncu/Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4
  agreed: 6
  auditor only: 0.84[L193 vision tower size], 3[L250 decoder bpw], 4[L249 n-gram table bpw], 4.81[L222 implied wikitext-2 PPL (extrap], 4.86[L193 MTP head size], 12.15[L121 weights on GPU (offload 36)], 15.33[L254 HC mixers SQNR], 15.91[L255 model mean SQNR], 23.39[L249 n-gram table SQNR], 24.31[L120 weights on GPU (offload 24)], 24.4[L249 n-gram table size], 49.53[L241 resident with n-gram offloaded], 71.83[L193 text decoder size], 95.4[L248 n-gram table size], 335.3[L210 size]
  ours only: 
## yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz
  agreed: 12
  auditor only: 1.3[L94 tool calling delta], 10[L23 needle recall 64K/128K], 14[L53 free memory at 131k ctx], 65[L94 MMLU-Pro preset vs plain], 65.5[L94 MMLU-Pro preset vs plain], 89[L94 IFEval preset vs plain], 91[L94 IFEval preset vs plain], 96.7[L94 GSM8K preset vs plain], 98[L94 GSM8K preset vs plain], 98.8[L94 top-1 vs plain model (harmless], 113[L53 memory needed]
  ours only: 
## zerodigest/Qwen3.8-Flash-Next-REAP-320-YMQ-GGUF
  agreed: 1
  auditor only: 2[L164 size], 478[L154 size]
  ours only: 
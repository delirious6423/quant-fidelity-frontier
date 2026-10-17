## AtomicChat/Qwen3.8-Flash-Next-GGUF
  agreed: 18
  auditor only: 0.0005[L184 mean KLD change], 0.0155[L138 PPL over calibration set], 0.23[L206 share / bpw contribution], 1[L191 top-1 change], 1.03[L205 share / bpw contribution], 1.24[L204 share / bpw contribution], 1.74[L203 share / bpw contribution], 4.6812[L138 PPL over calibration set], 6.8[L191 top-1 change], 16[L88 size difference], 24[L205 share / bpw contribution], 29[L204 share / bpw contribution], 38.4[L84 on SSD (n-gram shard)], 39[L61 n-gram table size], 39.1[L83 on SSD (n-gram shard)], 41[L203 share / bpw contribution], 54.4[L85 on SSD (n-gram shard)], 63[L191 mean KLD reduction (claimed sa], 85[L44 total file size], 37265[L53 host model memory (MiB)], 43720[L52 Metal model memory (llama.cpp ]
  ours only: 
## FINAL-Bench/POCKET-Qwen3.8-Flash-Next-GGUF
  agreed: 8
  auditor only: 3[L84 folder size], 3.7[L61 VRAM used (with -ot exps=CPU)], 4.1[L59 VRAM used (with -ot exps=CPU)], 16[L84 size reduction vs Q4_K_M]
  ours only: 
## ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
  agreed: 9
  auditor only: 3.5[L46 retained weight bpw], 4.5[L109 bpw], 28.8[L109 size (not resident)], 354[L110 size / resident]
  ours only: 
## ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
  agreed: 32
  auditor only: 0.51[L131 GPQA-D gap to BF16], 0.57[L83 LCB v6 gap], 0.91[L96 size], 1.14[L131 LCB v6 gap to BF16], 1.52[L83 GPQA-Diamond gap], 1.6[L111 size difference], 1.89[L52 effective bpw], 4.5[L71 bpw], 4.7[L131 compression vs BF16], 5.3[L131 compression], 7.8[L83 size saving], 16[L123 bpw], 28.8[L156 n-gram resident saving with mm], 37.6[L91 must be resident (shard 1)], 39.2[L92 must be resident (shard 1)], 47[L93 must be resident (shard 1)], 54.8[L94 must be resident (shard 1)], 58.4[L52 total size], 95.7[L131 task avg retained], 99.4[L131 task avg recovery], 354[L131 size]
  ours only: 
## Jundot/Qwen3.8-Flash-Next-oQ4e-mtp
  agreed: 6
  auditor only: 0.002[L47 tensor storage], 0.193[L46 tensor storage], 0.231[L49 tensor storage], 0.476[L48 tensor storage], 0.6[L45 tensor storage], 0.836[L61 vision encoder storage], 0.898[L61 vision encoder storage], 0.961[L52 tensor storage], 1.381[L62 MTP head storage], 1.39[L50 tensor storage], 1.483[L62 MTP head storage], 4.5[L44 effective bpw], 5[L43 effective bpw], 5.25[L45 effective bpw], 5.5[L46 effective bpw], 6.25[L47 effective bpw], 6.5[L48 effective bpw], 8.25[L49 effective bpw], 8.5[L50 effective bpw], 10.2[L35 KL change, Novel], 11.6[L34 KL change, Code], 16[L52 effective bpw], 29.802[L43 tensor storage], 32[L60 n-gram embedding storage], 65.339[L44 tensor storage], 66.974[L59 language backbone storage], 71.913[L59 language backbone storage], 98.994[L63 total storage]
  ours only: 
## Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata
  agreed: 22
  auditor only: 6e-05[L151 energy-weighted rel. MSE], 0.0023[L150 energy-weighted rel. MSE], 0.01684[L147 energy-weighted rel. MSE], 0.28[L153 ratio min/median/max], 0.53[L153 ratio min/median/max], 0.63[L153 ratio min/median/max], 0.73[L154 ratio range], 0.76[L154 ratio range], 0.77[L97 mtp size], 1.18[L96 token-embd-bf16.gguf size], 1.43[L92 pack/dense.bin + index size], 2.66[L110 RMS error vs BF16 table (524K ], 4.5[L75 bits per weight (experts)], 5.58[L94 dense gguf size], 33.3[L149 of RTN], 44.8[L149 of NVIDIA's], 47.68[L95 ple-fp8.gguf size (Qwen FP8 PL], 48.9[L151 summed error], 63.28[L90 pack/experts.bin size], 63.3[L75 experts size], 74.4[L148 of RTN], 360[L277 checkpoint download size], 1930.52[L150 summed error], 4716.95[L149 summed expert error vs BF16], 10534.2[L148 summed expert error vs BF16], 14161.9[L147 summed expert error vs BF16]
  ours only: 
## RadixArk/Qwen3.8-Flash-Next-NVFP4
  agreed: 6
  auditor only: 1[L121 AIME26 pass@1], 2.7[L92 compression ratio], 8[L121 AIME26 majority@8], 98.86[L120 GSM8K stop rate], 99.17[L121 AIME26 stop rate], 118.4[L137 unchanged-content bytes], 240[L121 AIME26 30x8], 360[L47 source checkpoint size]
  ours only: 
## TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw
  agreed: 4
  auditor only: 0[L181 mean KL (millinats/token) (Knu], 0.84[L164 vision tower size], 1.7[L88 down_proj padding saving], 1.7128[L103 PPL code], 1.7243[L103 PPL code], 2.1[L244 size], 2.2[L248 extra resident when enabled], 2.244[L178 ppl (prose) (Knurlogic 0.1.3, ], 2.276[L179 ppl (prose) (Knurlogic 0.1.3, ], 2.3[L174 ppl (prose) (Knurlogic 0.1.3, ], 2.31[L167 prose perplexity], 2.323[L176 ppl (prose) (Knurlogic 0.1.3, ], 2.367[L180 ppl (prose) (Knurlogic 0.1.3, ], 2.549[L177 ppl (prose) (Knurlogic 0.1.3, ], 3.912[L175 ppl (prose) (Knurlogic 0.1.3, ], 4.8[L180 literary KL (millinats/token) ], 5.7485[L103 PPL prose], 5.8255[L103 PPL prose], 6.3[L97 KL prose change], 6.4[L99 KL literary change], 7.3[L178 literary KL (millinats/token) ], 7.6574[L103 PPL literary], 7.824[L103 PPL literary], 8.5[L98 KL code change], 8.7[L179 literary KL (millinats/token) ], 12.1[L180 code KL (millinats/token) (Knu], 13.5[L176 literary KL (millinats/token) ], 14.054[L172 ppl (prose) (Knurlogic 0.1.3, ], 16.5[L178 code KL (millinats/token) (Knu], 22.4[L180 mean KL (millinats/token) (Knu], 25.7[L177 literary KL (millinats/token) ], 26.9[L179 code KL (millinats/token) (Knu], 29.5[L176 code KL (millinats/token) (Knu], 30.9[L178 mean KL (millinats/token) (Knu], 39.9[L179 mean KL (millinats/token) (Knu], 45.3[L88 size], 45.8[L154 peak RSS (load+2048 prefill+12], 47.7[L176 mean KL (millinats/token) (Knu], 49.5[L172 top-1 (prose) (Knurlogic 0.1.3], 50.4[L180 prose KL (millinats/token) (Kn]
  ours only: 
## TheDrainFlorist/Qwen3.8-Flash-Next-VQ-4.4bpw
  agreed: 34
  auditor only: 0[L189 mean KL (millinats/token) (Knu], 0.33[L87 size delta of promotion], 0.7[L98 KL literary change], 0.84[L172 vision tower size], 1.5[L97 KL code change], 1.6292[L102 PPL code], 1.6315[L102 PPL code], 2[L194 size diff], 2.1[L258 size], 2.2[L262 extra resident when enabled], 3.559[L181 ppl (prose) (Knurlogic 0.1.3, ], 4.8[L188 literary KL (millinats/token) ], 5.0583[L102 PPL prose], 5.2045[L102 PPL prose], 6.5232[L102 PPL literary], 6.5297[L102 PPL literary], 6.7[L195 KL diff prose / code / literar], 7.3[L186 literary KL (millinats/token) ], 8.7[L187 literary KL (millinats/token) ], 12.1[L188 code KL (millinats/token) (Knu], 12.9[L97 KL code (millinats/token)], 13.1[L97 KL code (millinats/token)], 13.5[L184 literary KL (millinats/token) ], 15.7[L195 KL diff prose / code / literar], 16.5[L96 KL prose change], 17.2[L195 KL diff prose / code / literar], 19.8[L194 KL diff prose / code / literar], 23[L195 size diff], 25.7[L185 literary KL (millinats/token) ], 26.9[L187 code KL (millinats/token) (Knu], 27.3[L194 KL diff prose / code / literar], 29.5[L184 code KL (millinats/token) (Knu], 36[L154 sustained RSS short-context ch], 42.6[L98 KL literary (millinats/token)], 42.9[L98 KL literary (millinats/token)], 44[L195 KL diff prose / code / literar], 44.5[L181 text-weight size], 47.3[L194 KL diff prose / code / literar], 48[L195 KL diff prose / code / literar], 50.4[L188 prose KL (millinats/token) (Kn]
  ours only: 
## agentionai/Qwen3.8-Flash-Next-AP-GGUF
  agreed: 52
  auditor only: 1[L95 top-1 inflation on wikitext], 1.9[L80 VRAM saving / top-1 cost], 2.8[L81 top-1 lead vs nearest 2-bit al], 6[L80 VRAM saving / top-1 cost], 6.9[L95 top-1 inflation on wikitext], 8.6[L95 top-1 inflation on wikitext], 9[L85 extra n-gram-table size], 9.4[L79 VRAM saving], 9.5[L74 VRAM saving], 26.82[L53 n-gram table host RAM (GiB)], 35.76[L47 n-gram table host RAM (GiB)], 76.86[L71 VRAM (GiB)], 91.82[L99 top-1 wikitext]
  ours only: 
## agentionai/Qwen3.8-Flash-Next-Gyro-GGUF
  agreed: 30
  auditor only: 3.75[L128 extra memory], 4[L26 MMLU-Pro relative to AP-Q4_K_X], 20[L31 memory saving], 24[L29 memory saving], 25[L31 memory saving], 27.6[L21 GPU memory (32 GB card tier)], 28.8[L33 GPU memory at 64k context], 30[L71 LiveCodeBench v6 @128k], 35[L26 memory], 50[L21 GPU memory (64 GB tier)], 50.2[L28 GPU memory], 52.3[L28 GPU memory], 64[L33 GPU memory at 64k context], 67[L71 LiveCodeBench v6 @64k], 68.2[L28 GPU memory], 78.1[L69 MMLU-Pro (70q)], 80[L69 MMLU-Pro (70q)], 81.4[L69 MMLU-Pro (70q)], 83[L71 LiveCodeBench v6 @128k], 90[L70 LiveCodeBench v6 @64k], 98[L26 MMLU-Pro relative to AP-Q4_K_X], 108[L139 download size]
  ours only: 
## agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF
  agreed: 13
  auditor only: 2.28[L128 size], 2.5[L23 PPL vs unquantized], 3.2[L107 KV cache at full 262144 ctx q8], 20[L41 size advantage], 28.8[L215 n-gram table size in host RAM], 30[L41 size advantage], 87[L108 weights (in 96 GiB carve-out)]
  ours only: 
## antirez/qwen3.8-flash-next-gguf
  agreed: 4
  auditor only: 95.37[L25 BF16 n-gram table size (in eac]
  ours only: 
## apetersson/Qwen3.8-Flash-Next-GGUF
  agreed: 2
  auditor only: 0[L134 attention+indexer KV per 262,1], 0.5[L134 MTP weights/compute], 0.62036[L98 driver reservation], 0.84[L60 file size (GiB)], 1.08[L134 vision weights/reserved comput], 2.6[L61 file size (GiB)], 2.67[L134 MTP weights/compute], 4.38[L134 attention+indexer KV per 262,1], 5.09[L94 minimum free VRAM], 7.74[L96 minimum free VRAM], 8[L134 attention+indexer KV per 262,1], 8.25[L134 attention+indexer KV per 262,1], 16[L134 attention+indexer KV per 262,1], 20.84[L95 minimum free VRAM], 74.76[L95 peak VRAM 262,144 ctx F16 KV v], 87.85[L96 peak VRAM 1,048,576 ctx Q8_0 K], 89.885[L98 peak VRAM breakdown allocated], 90.51[L94 peak VRAM 262,144 ctx F16 KV v], 95.37[L59 file size (GiB)], 95.5[L153 peak system wired memory], 110.33[L155 peak system wired memory], 187.76[L119 host cgroup memory], 188[L119 host cgroup memory], 235.47[L65 total active artifacts on disk], 262144[L134 MTP weights/compute]
  ours only: 
## cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4
  agreed: 0
  auditor only: 188.29[L35 model size]
  ours only: 
## ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw
  agreed: 8
  auditor only: 0.9[L124 vision tower size], 3.14[L111 bpw over routed experts], 16[L90 checkpoint size], 21[L59 resident saving vs mixed-4-8bi], 32[L99 ngram_table.bin size], 52.2[L75 active memory after load], 55.6[L75 peak memory during 36k prefill], 75[L41 resident], 86[L39 download], 99[L58 probability mass retained vs m], 107[L41 download], 137[L42 resident], 195[L42 download], 360[L90 checkpoint size]
  ours only: 
## ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit
  agreed: 3
  auditor only: 0.0052[L23 KLD median vs bf16], 0.9[L112 vision tower size], 16[L75 checkpoint size], 32[L84 ngram_table.bin size], 52[L46 resident], 86[L46 download], 88.6[L23 top-1 agreement vs bf16], 107[L48 download], 137[L49 resident], 195[L49 download], 360[L75 checkpoint size]
  ours only: 
## doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw
  agreed: 2
  auditor only: 0.56[L180 size], 0.621[L152 next-token loss @8K windows], 0.625[L152 next-token loss 73K corpus @8K], 0.9[L179 size], 1.5[L171 format size], 1.578[L153 next-token loss @16K windows], 1.579[L153 next-token loss @16K windows], 2[L119 per-node shard], 4[L119 per-node shard], 6[L178 n-gram table size], 9[L171 format size], 39[L178 n-gram table size], 57[L119 per-node shard], 68[L177 weights size], 71[L119 per-node shard], 360[L35 size], 8.979e+08[L45 size]
  ours only: 
## ggml-org/Qwen3.8-Flash-Next-GGUF
  agreed: 0
  auditor only: 
  ours only: 83.13, 102.05, 162.62
## ilintar/qwen3.8-flash-next-gguf-strix-halo
  agreed: 1
  auditor only: 2.6[L25 file size], 93[L64 resident in managed memory (ap]
  ours only: 
## ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2
  agreed: 12
  auditor only: 4.4[L107 target NLL increase], 5.16[L20 size saving], 11[L20 size saving], 29.8[L48 disk size], 32[L48 disk size], 41.73[L18 file size], 71.53[L26 total disk size], 76.81[L26 total disk size]
  ours only: 
## ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4
  agreed: 11
  auditor only: 29.8[L106 disk (demand-paged)], 32[L106 disk (demand-paged)]
  ours only: 
## ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF
  agreed: 35
  auditor only: 0.056[L37 Code mean KLD], 0.058[L37 English mean KLD], 0.078[L38 Code mean KLD], 0.085[L38 English mean KLD], 0.125[L39 Code mean KLD], 0.133[L39 English mean KLD], 2.21[L70 download], 2.4[L39 GPU-resident model tensors], 4.5[L38 GPU-resident model tensors], 5.7[L25 size reduction], 6.6[L72 download], 7.5[L45 KV cache + compute buffer at 2], 7.7[L25 size reduction], 20.3[L98 PLE table on disk (--defer-ple], 25[L98 PLE table on disk (--defer-ple], 28.19[L65 download], 36.31[L68 download], 38.07[L62 download], 42[L25 mean KLD reduction code], 44[L25 mean KLD reduction English], 44.33[L64 download], 44.41[L67 download], 44.68[L61 download], 47.2[L38 routed experts RAM or VRAM], 52.6[L39 routed experts RAM or VRAM], 55[L25 mean KLD reduction code], 56[L25 mean KLD reduction English], 312[L71 download]
  ours only: 
## ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF_v2
  agreed: 17
  auditor only: 36.31[L39 download size], 44.41[L38 download size]
  ours only: 
## lmstudio-community/Qwen3.8-Flash-Next-GGUF
  agreed: 0
  auditor only: 
  ours only: 119.15, 167.64, 188.22
## mlx-community/Qwen3.8-Flash-Next-4bit
  agreed: 0
  auditor only: 
  ours only: 111.55
## mradermacher/Qwen3.8-Flash-Next-i1-GGUF
  agreed: 14
  auditor only: 0.7[L44 size/GB]
  ours only: 
## mudler/Qwen3.8-Flash-Next-APEX-GGUF
  agreed: 13
  auditor only: 0.00126[L110 KL per GB], 0.0324[L110 KL divergence cost], 0.71[L110 PPL rise], 1.6[L94 bpw of 45% of model], 4.5[L94 compression vs BF16], 6[L110 top-1 predictions changed], 25.6[L110 size saved], 28.8[L45 n-gram shard size], 29.1[L41 system RAM required], 51.5[L68 forced floor size], 354[L94 original size]
  ours only: 
## nvidia/Qwen3.8-Flash-Next-NVFP4
  agreed: 18
  auditor only: 2.7[L186 size vs BF16], 63[L186 disk size reduction]
  ours only: 8, 132.73
## pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF
  agreed: 5
  auditor only: 0[L285 peak GPU 0 memory], 0.3[L316 PPL gap vs layer split], 0.9[L177 size], 8[L44 mtp tensors streamed size], 27[L238 PLE table in host RAM], 32.3[L280 idle GPU memory per GPU], 32.4[L285 peak GPU 0 memory], 80[L30 file size]
  ours only: 
## pfeifferj/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
  agreed: 15
  auditor only: 0.004775[L85 McNemar p], 0.91[L70 size], 1.72[L19 PPL increase vs base], 2.85[L85 MMLU-Pro paired difference], 3.065[L67 file bpw], 4.551[L69 file bpw], 16.002[L68 file bpw], 16.173[L70 file bpw], 29.49[L69 size], 103.68[L68 size], 1108[L83 MMLU-Pro correct], 1165[L82 MMLU-Pro correct], 2000[L83 MMLU-Pro correct]
  ours only: 
## primitive-ai/Qwen3.8-Flash-Next-NVFP4
  agreed: 29
  auditor only: 1.5[L259 MTP draft weights size], 5.2[L259 MTP draft weights size], 7.57[L68 KV cache at 262,144 ctx], 16[L163 knowledge], 28.8[L158 table sizes], 32[L158 table sizes], 49[L158 table sizes], 73.3[L383 toolace], 78.5[L291 tool-calling runs (pooled)], 79[L240 tool calling], 79.5[L267 tool-calling], 82.2[L382 xlam], 86[L372 math_500], 86.7[L377 commonsenseqa], 88.7[L375 hellaswag], 88.8[L38 VRAM at serve], 89[L371 mmlu_pro], 91.7[L187 MTP strict score none/2/3], 93.3[L190 MTP strict score n=3], 94[L378 boolq], 95[L68 n-gram table size loaded], 95.3[L376 winogrande], 95.4[L77 disk store written], 96[L379 openbookqa], 96.7[L381 bfcl_v4], 97[L373 gsm8k], 98.7[L374 arc_challenge], 99.5[L349 finished], 100[L384 glaive], 172[L111 checkpoint streamed at boot], 360[L38 size], 664[L285 out/answer], 88828[L300 VRAM at serve]
  ours only: 
## primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8
  agreed: 24
  auditor only: 1.42[L219 draft weights], 1.5[L296 MTP draft size], 2.52[L218 draft weights], 2.7[L334 size difference], 3[L248 tool-calling (n=3)], 5.03[L217 draft weights], 5.2[L296 MTP draft size], 16[L169 knowledge], 28.8[L164 table sizes], 32[L164 table sizes], 49[L164 table sizes], 57.3[L218 draft acceptance], 59.2[L217 draft acceptance], 73.3[L405 toolace], 77.7[L170 tool-calling n=3], 78.2[L170 tool-calling n=3], 78.5[L126 tool-calling suite], 78.7[L170 tool-calling n=3], 79.2[L248 tool-calling (n=3)], 82.2[L404 xlam], 86[L394 math_500], 86.7[L399 commonsenseqa], 88.7[L397 hellaswag], 89[L393 mmlu_pro], 91.7[L227 60-item scores], 92.9[L169 knowledge], 93.3[L227 60-item scores], 94[L400 boolq], 95[L71 n-gram table loaded], 95.3[L398 winogrande], 95.4[L82 disk store written], 96[L401 openbookqa], 96.7[L403 bfcl_v4], 97[L395 gsm8k], 98.7[L396 arc_challenge], 100[L406 glaive], 172[L117 checkpoint streamed at boot], 686[L322 out/answer]
  ours only: 83.8
## tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ
  agreed: 9
  auditor only: 0.0218[L121 mean per-row rel. L2 error vs ], 0.0267[L121 mean per-row rel. L2 error vs ], 0.055[L134 mean KL vs BF16 (end-of-prefil], 0.064[L134 mean KL / top-1], 2.7[L23 size], 18[L121 error / size delta], 19[L121 error / size delta], 41.6[L121 size], 51.3[L121 size], 64.2[L21 size], 82[L71 system RAM at runtime], 93.3[L134 mean KL / top-1], 95.2[L134 top-1 vs BF16]
  ours only: 
## turboderp/Qwen3.8-Flash-Next-exl3
  agreed: 1
  auditor only: 2.05[L19 bpw], 3.05[L20 bpw], 5.05[L22 bpw], 6.05[L23 bpw]
  ours only: 0.0031, 0.004, 0.0067, 0.0084, 0.01, 0.0165, 0.0177, 0.0241, 0.0349, 0.0533, 0.0684, 0.0804, 1.3825, 1.38423, 1.3848, 1.39, 1.3926, 1.397, 1.4017, 1.4075, 1.4199, 1.4342, 1.4619, 1.4879, 1.5031, 2.09, 2.85, 3.09, 3.12, 3.31, 4.09, 4.78, 5.09, 5.18, 6.09, 62.82, 85.14, 107.46, 123.25, 139.04
## unsloth/Qwen3.8-Flash-Next-GGUF
  agreed: 40
  auditor only: 1[L90 top-1 recovery / disk saving c], 1.35[L327 savings], 1.91[L328 shared MTP size], 2.54[L326 savings], 2.79[L328 general MTP size], 4.14[L327 general MTP size], 5.23[L326 shared MTP size], 7.77[L326 general MTP size], 16[L11 size], 75[L37 hardware requirement (total me], 76[L333 MTP hardware requirement (tota], 79[L37 hardware requirement (total me], 80[L333 MTP hardware requirement (tota], 91[L333 MTP hardware requirement (tota], 96[L37 hardware requirement (total me], 97[L333 MTP hardware requirement (tota], 114[L37 hardware requirement (total me], 115[L333 MTP hardware requirement (tota], 163[L37 hardware requirement (total me], 164[L333 MTP hardware requirement (tota], 200[L333 MTP hardware requirement (tota], 355[L333 MTP hardware requirement], 880[L328 savings]
  ours only: 
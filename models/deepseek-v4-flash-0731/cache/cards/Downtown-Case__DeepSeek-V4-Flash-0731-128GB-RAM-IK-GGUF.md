---
license: mit
language:
- en
- zh
library_name: gguf
pipeline_tag: text-generation
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: Downtown-Case
tags:
- imatrix
- gguf
- quantized
- moe
- deepseek
- 3-bit
- conversational
- ik_llama.cpp
metrics:
  - perplexity
  - kl_divergence
---

# DeepSeek V4 Flash

DSV4-0731, specifically sized for hybrid inference with 128GB CPU RAM, with an ik_llama.cpp quant type mix.

With a measured KL divergence of 0.079, the "V1" GGUF has the least *measured* quantization loss I can find among similarly sized GGUFs. It adopts findings/fixes from AtomicChat's mainline DS4 GGUFs, see their page for details: https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF

**Requires** ik_llama.cpp: https://github.com/ikawrakow/ik_llama.cpp

Or croco.cpp: https://github.com/Nexesenex/croco.cpp

***

## V1 GGUF:

121.8GB on disk. 115.27GB allocated in CPU RAM, with all the experts offloaded.

`llama-sweep-bench` on a Ryzen 7800, 128GB DDR5, single underclocked RTX 3090:

```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |     72 |      0 |    8.143 |   502.99 |    5.080 |    14.17 |
|  4096 |     72 |   4096 |    8.113 |   504.86 |    5.115 |    14.08 |
|  4096 |     72 |   8192 |    8.268 |   495.41 |    5.111 |    14.09 |
|  4096 |     72 |  12288 |    8.381 |   488.72 |    5.137 |    14.02 |
|  4096 |     72 |  16384 |    8.627 |   474.82 |    5.184 |    13.89 |
|  4096 |     72 |  20480 |    9.171 |   446.62 |    5.361 |    13.43 |
|  4096 |     72 |  24576 |    9.299 |   440.49 |    5.344 |    13.47 |
|  4096 |     72 |  28672 |    9.505 |   430.93 |    5.373 |    13.40 |
|  4096 |     72 |  32768 |    9.617 |   425.93 |    5.421 |    13.28 |
|  4096 |     72 |  36864 |    9.755 |   419.89 |    5.431 |    13.26 |
|  4096 |     72 |  40960 |    9.853 |   415.70 |    5.434 |    13.25 |
|  4096 |     72 |  45056 |   10.180 |   402.34 |    5.456 |    13.20 |
|  4096 |     72 |  49152 |   10.584 |   387.01 |    5.450 |    13.21 |
|  4096 |     72 |  53248 |   10.715 |   382.27 |    5.483 |    13.13 |
|  4096 |     72 |  57344 |   10.816 |   378.70 |    5.499 |    13.09 |
|  4096 |     72 |  61440 |   10.957 |   373.82 |    5.555 |    12.96 |
|  4096 |     72 |  65536 |   11.112 |   368.59 |    5.554 |    12.96 |
|  4096 |     72 |  69632 |   11.422 |   358.62 |    5.621 |    12.81 |
|  4096 |     72 |  73728 |   11.507 |   355.95 |    5.587 |    12.89 |
|  4096 |     72 |  77824 |   11.648 |   351.65 |    5.630 |    12.79 |
|  4096 |     72 |  81920 |   11.840 |   345.95 |    5.631 |    12.79 |
|  4096 |     72 |  86016 |   11.987 |   341.71 |    5.689 |    12.66 |
|  4096 |     72 |  90112 |   12.041 |   340.18 |    5.675 |    12.69 |
|  4096 |     72 |  94208 |   12.458 |   328.77 |    5.710 |    12.61 |
|  4096 |     72 |  98304 |   12.527 |   326.96 |    5.727 |    12.57 |
|  4096 |     72 | 102400 |   12.736 |   321.60 |    5.785 |    12.45 |
|  4096 |     72 | 106496 |   12.840 |   319.00 |    5.739 |    12.55 |
|  4096 |     72 | 110592 |   12.957 |   316.13 |    5.811 |    12.39 |
|  4096 |     72 | 114688 |   13.396 |   305.77 |    5.782 |    12.45 |
|  4096 |     72 | 118784 |   13.503 |   303.35 |    5.850 |    12.31 |
|  4096 |     72 | 122880 |   13.702 |   298.93 |    5.807 |    12.40 |
```

Prompt processing drops to ~340 t/s at bs 2048.

***

Example command to run:

`build/bin/llama-server -mqkv -thp --batch_size 4096 --ubatch_size 4096 --ctx-size 131072 --host 127.0.0.1 --ui-mcp-proxy --port 5000 -ngl 99 -ngld 99 -cram 0 --n-cpu-moe 44 --parallel 1 --threads 8 --threads-batch 8 --threads-http 8 --no-mmap --mlock --webui none --jinja --chat-template DeepSeek-V4 --model path_to_model`

- `-thp` needs transparent huge pages support from your OS, but does notably boost performance.

- I disable `cram` caching, the web ui and such to save CPU RAM.

- There is no chat template in the GGUF, so be sure to use a jinja.

- For DSV4, I like adding `--top-n-sigma 1.0` to sampling.

- Adjust context/slots to free vram.

- If this takes too much CPU ram for your system, decrease `n-cpu-moe` to offload 1 or 2 MoE layers to GPU RAM.


***

<details>
<summary>Quantization Details (click):</summary>

```
custom="
#/usr/bin/env bash

### Attention (GPU)

blk\.(0|16|22|24|28|34|36|38|4[0-2])\.attn_q_a\.weight=q8_0
blk\..*\.attn_q_a\.weight=iq6_k


### Make sure these aren't requantized (to satisfy my OCD)

blk\..*\.ffn_gate_tid2eid\.weight=i32
blk\..*\.attn_kv_a_norm\.weight=f32
blk\..*\.attn_norm\.weight=f32
blk\..*\.attn_q_a_norm\.weight=f32
blk\..*\.attn_sinks\.weight=f32
blk\..*\.ffn_norm\.weight=f32
blk\..*\.hc_attn_base\.weight=f32
blk\..*\.hc_attn_fn\.weight=f32
blk\..*\.hc_attn_scale\.weight=f32
blk\..*\.hc_ffn_base\.weight=f32
blk\..*\.hc_ffn_fn\.weight=f32
blk\..*\.hc_ffn_scale\.weight=f32
blk\..*\.attn_kv_a_norm\.weight=f32
blk\..*\.attn_kv_a_norm\.weight=f32
blk\..*\.attn_kv_a_norm\.weight=f32
blk\..*\.attn_compressor_ape\.weight=f32
blk\..*\.attn_compressor_norm\.weight=f32
blk\..*\.indexer_compressor_ape\.weight=f32
blk\..*\.indexer_compressor_norm\.weight=f32
blk\..*\.indexer\.proj\.weight=f32

output_hc_base\.weight=f32
output_hc_fn\.weight=f32
output_hc_scale\.weight=f32
output_norm\.weight=f32

### Dense Experts (GPU)

blk\.(0|20|21|22|3[6-9]|4[0-2])\.ffn_down_shexp\.weight=q8_0
blk\.(36|37|38|39|4[0-2])\.ffn_gate_shexp\.weight=q8_0
blk\.(36|37|38|39|4[0-2])\.ffn_up_shexp\.weight=q8_0

blk\..*\.ffn_down_shexp\.weight=iq6_k
blk\..*\.ffn_gate_shexp\.weight=iq6_k
blk\..*\.ffn_up_shexp\.weight=iq6_k
blk\..*\.ffn_gate_inp\.weight=bf16

### MoE Experts Layers (CPU)

# Gate/Up

blk\.(0|29|3[0-5])\.ffn_(gate|up)_exps\.weight=iq3_k_r4
blk\.(36|37|38|39|4[0-2])\.ffn_(gate|up)_exps\.weight=mxfp4
blk\..*\.ffn_(gate|up)_exps\.weight=iq3_ks

# Down
blk\.([(1-9]|1[0-9]|21|25])\.ffn_down_exps\.weight=iq3_k_r4
blk\..*\.ffn_down_exps\.weight=mxfp4

# Non-Repeating Layers
token_embd\.weight=iq5_k
output\.weight=iq6_k
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

./build/bin/llama-quantize --allow-requantize \
     --custom-q "$custom" \
     --imatrix /home/alpha/Models/iMatrix/ds4-0731-v2-ik-imatrix.dat \
     /home/alpha/Models/GGUF/Deepseek_V4/DeepSeek-V4-Flash-0731-Atomic/AD-BF16/DeepSeek-V4-Flash-0731-AD-BF16-00001-of-00004.gguf \
     /home/alpha/Models/GGUF/Deepseek_V4/DeepSeek-V4-Flash-0731-mixed-v3-128G_r4.gguf \
     Q8_0 \
     16
```

- Uses AtomicChat's BF16 as a base: https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF

- Uses ubergarm's iMatrix corpus extended with tool call examples, Deepseek-formatted chats, multilingual code/text, English stories of various formats/genres, and (per AtomicChat's findings) a full sweep of the DSV4 tokenizer's vocabulary, at a chunk/batch size of 8192.

- Mostly IQ3_KS and IQ3_K_R4 MoE FFNs, ik_llama.cpp-only quantization types that are fast on CPU.

- Gradually transitions to unquantized MXFP4 for the last layers, following the iMatrix "sensitivity" data.
 
- IQ6_K output, IQ5_K token embeddings.

- A few extra attention layers at Q8_0 instead of Q6_K.

- ...Plus a lot of trial-and-error.

</details>

***

<details>
<summary>KLD/Perplexity</summary>

```
====== Perplexity statistics ======
Mean PPL(Q)                   :   4.474183 ±   0.028540
Mean PPL(base)                :   4.341717 ±   0.027501
Cor(ln(PPL(Q)), ln(PPL(base))):  97.25%
Mean ln(PPL(Q)/PPL(base))     :   0.030054 ±   0.001490
Mean PPL(Q)/PPL(base)         :   1.030510 ±   0.001536
Mean PPL(Q)-PPL(base)         :   0.132467 ±   0.006648

====== KL divergence statistics ======
Mean    KLD:   0.078715 ±   0.000679
Maximum KLD:  13.481635
99.9%   KLD:   3.230898
99.0%   KLD:   1.149215
95.0%   KLD:   0.336894
90.0%   KLD:   0.166125
Median  KLD:   0.014272
10.0%   KLD:   0.000010
 5.0%   KLD:   0.000000
 1.0%   KLD:  -0.000001
 0.1%   KLD:  -0.000004
Minimum KLD:  -0.000010

====== Token probability statistics ======
Mean    Δp: -0.774 ± 0.026 %
Maximum Δp: 98.126%
99.9%   Δp: 63.917%
99.0%   Δp: 26.286%
95.0%   Δp:  8.732%
90.0%   Δp:  4.349%
75.0%   Δp:  0.458%
Median  Δp: -0.001%
25.0%   Δp: -1.052%
10.0%   Δp: -6.160%
 5.0%   Δp: -12.496%
 1.0%   Δp: -40.097%
 0.1%   Δp: -81.561%
Minimum Δp: -99.867%
RMS Δp    :  9.696 ± 0.068 %
Same top p: 90.885 ± 0.076 %
```

</details>

***

## KL Divergence:

"KLD" is a measure of how much a (quantized) model's tokens differ from the base model. See llama.cpp's notes on KLD testing: https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md

Lower KLD = closer to the unquantized model.

KLD scores of ~128GB models:

- ddh0's IQ3S (127.817 GiB 3.861 BPW): **0.090**

- ddh0's Q3_K (127.817 GiB 3.861 BPW) : **0.087** *(better than the IQ3?)*

- Unsloth's Q3_K_XL (119.397 GiB 3.607 BPW): **0.111**

- AtomicChat's IQ3_S (121.774 GiB 3.679 BPW): **0.098**

- V1 Mix (121.801 GiB 3.680 BPW): **0.079**

<details>

<summary>KLD Notes(click):</summary>

- KLD measurements are not comparable across repos. It also varies with hardware. See AtomicChat's findings.

- My KLD testing is done on standard Wikitext, but at a batch size of 8192 instead of the default 512.

- Command used to generate logits: `systemd-run --user --scope -p MemorySwapMax=0 -p MemoryZSwapMax=0./build/bin/llama-perplexity -m /.../DeepSeek-V4-Flash-0731-Atomic/AD-BF16/DeepSeek-V4-Flash-0731-AD-BF16-00001-of-00004.gguf -f wikitext-2-raw/wiki.test.raw  --kl-divergence-base DeepSeek-V4-Flash-0731-bf16-logits-8192.dat -c 8192 -b 8192 -ub 8192 --threads 8 --prefetch-experts -ngl 2`

- KLD command: `./build/bin/llama-perplexity -m model.gguf -c 8192 --batch-size 8192 --ubatch-size 8192 --threads 8 -ngl 999 --n-cpu-moe 43 --no-mmap --mlock -thp -mqkv --kl-divergence-base /.../DeepSeek-V4-Flash-0731-bf16-logits-8192.dat --kl-divergence 2>&1 | tee model.log`


</details>

***

## TODO:

- Test/quantize speculative decoding with `llama-spec-bench`.

- KLD test more GGUFs.

- Test "real-world" impact of Q8_0 cache quantization at long context

- MXFP4_R8 variant for faster(?) TG at the cost of PP.

- V2 formula, slightly smaller for more free CPU RAM.

- Test merging DSV4-0731 with its base model.

- Make other sizes, upon request.

***

## Notes, for *Science*:

- Q8_0 K cache quantization shows virtually no KLD change. This is interesting, as older models (like GLM 4.6) showed measurable KLD changes with KV cache quantization.

- ddh0's testing confirms that the Q3_K_XL has a better KLD than the IQ3_S, at the exact same bpw, with everything else being equal. *Why?* IQ3_S should be a higher quality quantization; do the FP4 MoEs "prefer" certain quantization types?

- It appears that some combination of AtomicChat's "lossless" BF16 GGUF, a higher iMatrix batch size, and a "vocab sweep" in the iMatrix data appreciably improves KLD, all other things being equal.

- Ubergarm posts great general info about ik_llama.cpp quantization. I recommend checking their repositories out: https://huggingface.co/ubergarm/

- Underclocking my 3090 and 7800 has little effect on speed, at least outside of parallel generation. I highly recommended it, and posted an example script to underclock/undervolt a 3090.

- I build ik_llama.cpp with AVX512 force-enabled for my Ryzen 7800. Not sure what effect this has on DSV4, but I posted my build flags.

- DSV4-0731 appears to be "bottom heavy," with the last few layers *particularly* sensitive to quantization. See: the wild iMatrix metrics.

<details>
<summary>Per-Layer BF16 iMatrix Data (click):</summary>
  
```
 Layer	       Tensor	          Σ(Act²)	  Min	            Max	           μ	   σ	 % Active	N	   Entropy	E (norm)	ZD	  CosSim
0.01.394.559 I =========================================================================================================================================================================
0.01.394.574 I    40	attn_compressor_gate	    220.46	  0.0040	   177.8002	  0.05	  2.78	  100.00%	  4096	    2.5334	 21.11%	      0.07%	  0.0866
0.01.394.578 I    34	attn_compressor_gate	    152.95	  0.0033	   102.6580	  0.04	  1.67	  100.00%	  4096	    2.7774	 23.15%	      0.05%	  0.1072
0.01.394.580 I    16	attn_compressor_gate	    102.98	  0.0038	    51.3552	  0.03	  0.92	  100.00%	  4096	    4.2088	 35.07%	      0.05%	  0.1092
0.01.394.582 I    22	attn_compressor_gate	     99.95	  0.0062	    33.5665	  0.02	  0.59	  100.00%	  4096	    6.6583	 55.49%	      0.15%	  0.1192
0.01.394.584 I    38	attn_compressor_gate	     89.28	  0.0043	    33.0348	  0.02	  0.57	  100.00%	  4096	    6.4538	 53.78%	      0.10%	  0.1333
0.01.394.586 I    24	attn_compressor_gate	     73.88	  0.0054	    20.3442	  0.02	  0.37	  100.00%	  4096	    7.3008	 60.84%	      0.15%	  0.1450
0.01.394.588 I    28	attn_compressor_gate	     69.13	  0.0022	    51.1201	  0.02	  0.80	  100.00%	  4096	    3.4863	 29.05%	      0.05%	  0.1772
0.01.394.589 I    20	attn_compressor_gate	     68.64	  0.0028	    25.5560	  0.02	  0.55	  100.00%	  4096	    4.7777	 39.81%	      0.05%	  0.1845
0.01.394.591 I    32	attn_compressor_gate	     52.57	  0.0024	    24.4735	  0.01	  0.43	  100.00%	  4096	    4.8843	 40.70%	      0.05%	  0.1101
0.01.394.592 I    30	attn_compressor_gate	     51.31	  0.0025	    13.7820	  0.01	  0.31	  100.00%	  4096	    5.9912	 49.93%	      0.07%	  0.1196
0.01.394.594 I    14	attn_compressor_gate	     49.35	  0.0032	    22.9597	  0.01	  0.37	  100.00%	  4096	    6.2063	 51.72%	      0.05%	  0.0995
0.01.394.595 I    12	attn_compressor_gate	     43.61	  0.0025	    15.9644	  0.01	  0.27	  100.00%	  4096	    6.3918	 53.26%	      0.10%	  0.1075
0.01.394.597 I    26	attn_compressor_gate	     42.05	  0.0022	    17.4225	  0.01	  0.33	  100.00%	  4096	    5.3200	 44.33%	      0.05%	  0.1784
0.01.394.598 I    18	attn_compressor_gate	     40.06	  0.0013	    15.5157	  0.01	  0.30	  100.00%	  4096	    4.5824	 38.19%	      0.07%	  0.4172
0.01.394.600 I    23	attn_compressor_gate	     34.69	  0.0062	     0.0478	  0.01	  0.00	  100.00%	  4096	   11.9715	 99.76%	      8.57%	  0.0627
0.01.394.602 I    36	attn_compressor_gate	     33.18	  0.0021	    17.9069	  0.01	  0.28	  100.00%	  4096	    6.4848	 54.04%	      0.02%	  0.0775
0.01.394.603 I     8	attn_compressor_gate	     28.93	  0.0024	     9.8737	  0.01	  0.16	  100.00%	  4096	    7.7816	 64.85%	      0.05%	  0.2032
0.01.394.605 I    10	attn_compressor_gate	     28.22	  0.0026	     6.9477	  0.01	  0.13	  100.00%	  4096	    8.4242	 70.20%	      0.07%	  0.1561
0.01.394.606 I    37	attn_compressor_gate	     22.57	  0.0035	     0.0323	  0.01	  0.00	  100.00%	  4096	   11.9569	 99.64%	     10.18%	  0.0516
0.01.394.608 I    17	attn_compressor_gate	     21.21	  0.0033	     0.5185	  0.01	  0.01	  100.00%	  4096	   11.8232	 98.53%	      0.17%	  0.7418
0.01.394.610 I    39	attn_compressor_gate	     20.31	  0.0032	     0.0339	  0.00	  0.00	  100.00%	  4096	   11.9553	 99.63%	     10.33%	  0.1030
0.01.394.612 I    41	attn_compressor_gate	     16.61	  0.0023	     0.0227	  0.00	  0.00	  100.00%	  4096	   11.9400	 99.50%	     11.69%	  0.0304
0.01.394.614 I     4	attn_compressor_gate	     16.30	  0.0014	     3.8507	  0.00	  0.07	  100.00%	  4096	    8.4898	 70.75%	      0.05%	  0.2090
0.01.394.616 I     6	attn_compressor_gate	     15.27	  0.0012	     6.2976	  0.00	  0.10	  100.00%	  4096	    7.5735	 63.11%	      0.05%	  0.2231
0.01.394.617 I    35	attn_compressor_gate	     14.15	  0.0023	     0.0309	  0.00	  0.00	  100.00%	  4096	   11.9544	 99.62%	      9.84%	  0.1415
0.01.394.619 I    42	attn_compressor_gate	     13.63	  0.0011	     0.5170	  0.00	  0.01	  100.00%	  4096	   11.5541	 96.28%	      0.15%	  0.3448
0.01.394.620 I     2	attn_compressor_gate	     11.12	  0.0012	     0.7982	  0.00	  0.01	  100.00%	  4096	   11.2380	 93.65%	      0.10%	  0.0000
0.01.394.622 I    15	attn_compressor_gate	      9.99	  0.0016	     0.1894	  0.00	  0.00	  100.00%	  4096	   11.8296	 98.58%	      0.32%	  0.7024
0.01.394.624 I    21	attn_compressor_gate	      8.85	  0.0014	     0.0163	  0.00	  0.00	  100.00%	  4096	   11.9458	 99.55%	      7.96%	  0.0821
0.01.394.625 I     3	attn_compressor_gate	      8.04	  0.0009	     0.0289	  0.00	  0.00	  100.00%	  4096	   11.9316	 99.43%	      6.79%	  0.2356
0.01.394.627 I    33	attn_compressor_gate	      7.95	  0.0011	     0.0167	  0.00	  0.00	  100.00%	  4096	   11.9395	 99.50%	      8.11%	  0.1514
0.01.394.628 I    27	attn_compressor_gate	      7.77	  0.0013	     0.0218	  0.00	  0.00	  100.00%	  4096	   11.9554	 99.63%	      7.50%	  0.0726
0.01.394.630 I    29	attn_compressor_gate	      7.30	  0.0011	     0.0222	  0.00	  0.00	  100.00%	  4096	   11.9394	 99.50%	      7.23%	  0.1927
0.01.394.631 I    31	attn_compressor_gate	      6.88	  0.0011	     0.0133	  0.00	  0.00	  100.00%	  4096	   11.9600	 99.67%	      8.37%	  0.1022
0.01.394.632 I    19	attn_compressor_gate	      5.41	  0.0008	     0.0172	  0.00	  0.00	  100.00%	  4096	   11.9392	 99.49%	      6.59%	  0.0859
0.01.394.634 I    13	attn_compressor_gate	      4.84	  0.0008	     0.0523	  0.00	  0.00	  100.00%	  4096	   11.9165	 99.30%	      1.49%	  0.1068
0.01.394.635 I     9	attn_compressor_gate	      4.31	  0.0006	     0.0394	  0.00	  0.00	  100.00%	  4096	   11.8949	 99.12%	      1.76%	  0.1423
0.01.394.637 I    11	attn_compressor_gate	      4.06	  0.0006	     0.0456	  0.00	  0.00	  100.00%	  4096	   11.9104	 99.25%	      1.88%	  0.0699
0.01.394.638 I     7	attn_compressor_gate	      3.64	  0.0005	     0.0231	  0.00	  0.00	  100.00%	  4096	   11.9319	 99.43%	      3.76%	  0.1596
0.01.394.639 I     5	attn_compressor_gate	      3.55	  0.0004	     0.0109	  0.00	  0.00	  100.00%	  4096	   11.9422	 99.52%	      7.71%	  0.0666
0.01.394.641 I    25	attn_compressor_gate	      2.26	  0.0003	     0.0059	  0.00	  0.00	  100.00%	  4096	   11.9538	 99.62%	      7.40%	  0.1029
0.01.394.642 I    40	attn_compressor_kv  	    220.46	  0.0040	   177.8002	  0.05	  2.78	  100.00%	  4096	    2.5334	 21.11%	      0.07%	  0.0866
0.01.394.644 I    34	attn_compressor_kv  	    152.95	  0.0033	   102.6580	  0.04	  1.67	  100.00%	  4096	    2.7774	 23.15%	      0.05%	  0.1072
0.01.394.645 I    16	attn_compressor_kv  	    102.98	  0.0038	    51.3552	  0.03	  0.92	  100.00%	  4096	    4.2088	 35.07%	      0.05%	  0.1092
0.01.394.647 I    22	attn_compressor_kv  	     99.95	  0.0062	    33.5665	  0.02	  0.59	  100.00%	  4096	    6.6583	 55.49%	      0.15%	  0.1192
0.01.394.648 I    38	attn_compressor_kv  	     89.28	  0.0043	    33.0348	  0.02	  0.57	  100.00%	  4096	    6.4538	 53.78%	      0.10%	  0.1333
0.01.394.650 I    24	attn_compressor_kv  	     73.88	  0.0054	    20.3442	  0.02	  0.37	  100.00%	  4096	    7.3008	 60.84%	      0.15%	  0.1450
0.01.394.651 I    28	attn_compressor_kv  	     69.13	  0.0022	    51.1201	  0.02	  0.80	  100.00%	  4096	    3.4863	 29.05%	      0.05%	  0.1772
0.01.394.653 I    20	attn_compressor_kv  	     68.64	  0.0028	    25.5560	  0.02	  0.55	  100.00%	  4096	    4.7777	 39.81%	      0.05%	  0.1845
0.01.394.654 I    32	attn_compressor_kv  	     52.57	  0.0024	    24.4735	  0.01	  0.43	  100.00%	  4096	    4.8843	 40.70%	      0.05%	  0.1101
0.01.394.656 I    30	attn_compressor_kv  	     51.31	  0.0025	    13.7820	  0.01	  0.31	  100.00%	  4096	    5.9912	 49.93%	      0.07%	  0.1196
0.01.394.657 I    14	attn_compressor_kv  	     49.35	  0.0032	    22.9597	  0.01	  0.37	  100.00%	  4096	    6.2063	 51.72%	      0.05%	  0.0995
0.01.394.658 I    12	attn_compressor_kv  	     43.61	  0.0025	    15.9644	  0.01	  0.27	  100.00%	  4096	    6.3918	 53.26%	      0.10%	  0.1075
0.01.394.660 I    26	attn_compressor_kv  	     42.05	  0.0022	    17.4225	  0.01	  0.33	  100.00%	  4096	    5.3200	 44.33%	      0.05%	  0.1784
0.01.394.661 I    18	attn_compressor_kv  	     40.06	  0.0013	    15.5157	  0.01	  0.30	  100.00%	  4096	    4.5824	 38.19%	      0.07%	  0.4172
0.01.394.663 I    23	attn_compressor_kv  	     34.69	  0.0062	     0.0478	  0.01	  0.00	  100.00%	  4096	   11.9715	 99.76%	      8.57%	  0.0627
0.01.394.664 I    36	attn_compressor_kv  	     33.18	  0.0021	    17.9069	  0.01	  0.28	  100.00%	  4096	    6.4848	 54.04%	      0.02%	  0.0775
0.01.394.666 I     8	attn_compressor_kv  	     28.93	  0.0024	     9.8737	  0.01	  0.16	  100.00%	  4096	    7.7816	 64.85%	      0.05%	  0.2032
0.01.394.667 I    10	attn_compressor_kv  	     28.22	  0.0026	     6.9477	  0.01	  0.13	  100.00%	  4096	    8.4242	 70.20%	      0.07%	  0.1561
0.01.394.669 I    37	attn_compressor_kv  	     22.57	  0.0035	     0.0323	  0.01	  0.00	  100.00%	  4096	   11.9569	 99.64%	     10.18%	  0.0516
0.01.394.670 I    17	attn_compressor_kv  	     21.21	  0.0033	     0.5185	  0.01	  0.01	  100.00%	  4096	   11.8232	 98.53%	      0.17%	  0.7418
0.01.394.671 I    39	attn_compressor_kv  	     20.31	  0.0032	     0.0339	  0.00	  0.00	  100.00%	  4096	   11.9553	 99.63%	     10.33%	  0.1030
0.01.394.673 I    41	attn_compressor_kv  	     16.61	  0.0023	     0.0227	  0.00	  0.00	  100.00%	  4096	   11.9400	 99.50%	     11.69%	  0.0304
0.01.394.674 I     4	attn_compressor_kv  	     16.30	  0.0014	     3.8507	  0.00	  0.07	  100.00%	  4096	    8.4898	 70.75%	      0.05%	  0.2090
0.01.394.676 I     6	attn_compressor_kv  	     15.27	  0.0012	     6.2976	  0.00	  0.10	  100.00%	  4096	    7.5735	 63.11%	      0.05%	  0.2231
0.01.394.677 I    35	attn_compressor_kv  	     14.15	  0.0023	     0.0309	  0.00	  0.00	  100.00%	  4096	   11.9544	 99.62%	      9.84%	  0.1415
0.01.394.679 I    42	attn_compressor_kv  	     13.63	  0.0011	     0.5170	  0.00	  0.01	  100.00%	  4096	   11.5541	 96.28%	      0.15%	  0.3448
0.01.394.680 I     2	attn_compressor_kv  	     11.12	  0.0012	     0.7982	  0.00	  0.01	  100.00%	  4096	   11.2380	 93.65%	      0.10%	  0.0000
0.01.394.681 I    15	attn_compressor_kv  	      9.99	  0.0016	     0.1894	  0.00	  0.00	  100.00%	  4096	   11.8296	 98.58%	      0.32%	  0.7024
0.01.394.683 I    21	attn_compressor_kv  	      8.85	  0.0014	     0.0163	  0.00	  0.00	  100.00%	  4096	   11.9458	 99.55%	      7.96%	  0.0821
0.01.394.684 I     3	attn_compressor_kv  	      8.04	  0.0009	     0.0289	  0.00	  0.00	  100.00%	  4096	   11.9316	 99.43%	      6.79%	  0.2356
0.01.394.686 I    33	attn_compressor_kv  	      7.95	  0.0011	     0.0167	  0.00	  0.00	  100.00%	  4096	   11.9395	 99.50%	      8.11%	  0.1514
0.01.394.687 I    27	attn_compressor_kv  	      7.77	  0.0013	     0.0218	  0.00	  0.00	  100.00%	  4096	   11.9554	 99.63%	      7.50%	  0.0726
0.01.394.688 I    29	attn_compressor_kv  	      7.30	  0.0011	     0.0222	  0.00	  0.00	  100.00%	  4096	   11.9394	 99.50%	      7.23%	  0.1927
0.01.394.690 I    31	attn_compressor_kv  	      6.88	  0.0011	     0.0133	  0.00	  0.00	  100.00%	  4096	   11.9600	 99.67%	      8.37%	  0.1022
0.01.394.691 I    19	attn_compressor_kv  	      5.41	  0.0008	     0.0172	  0.00	  0.00	  100.00%	  4096	   11.9392	 99.49%	      6.59%	  0.0859
0.01.394.692 I    13	attn_compressor_kv  	      4.84	  0.0008	     0.0523	  0.00	  0.00	  100.00%	  4096	   11.9165	 99.30%	      1.49%	  0.1068
0.01.394.694 I     9	attn_compressor_kv  	      4.31	  0.0006	     0.0394	  0.00	  0.00	  100.00%	  4096	   11.8949	 99.12%	      1.76%	  0.1423
0.01.394.695 I    11	attn_compressor_kv  	      4.06	  0.0006	     0.0456	  0.00	  0.00	  100.00%	  4096	   11.9104	 99.25%	      1.88%	  0.0699
0.01.394.696 I     7	attn_compressor_kv  	      3.64	  0.0005	     0.0231	  0.00	  0.00	  100.00%	  4096	   11.9319	 99.43%	      3.76%	  0.1596
0.01.394.697 I     5	attn_compressor_kv  	      3.55	  0.0004	     0.0109	  0.00	  0.00	  100.00%	  4096	   11.9422	 99.52%	      7.71%	  0.0666
0.01.394.699 I    25	attn_compressor_kv  	      2.26	  0.0003	     0.0059	  0.00	  0.00	  100.00%	  4096	   11.9538	 99.62%	      7.40%	  0.1029
0.01.394.700 I    40	attn_kv             	    220.46	  0.0040	   177.8002	  0.05	  2.78	  100.00%	  4096	    2.5334	 21.11%	      0.07%	  0.0866
0.01.394.702 I    34	attn_kv             	    152.95	  0.0033	   102.6580	  0.04	  1.67	  100.00%	  4096	    2.7774	 23.15%	      0.05%	  0.1072
0.01.394.703 I    16	attn_kv             	    102.98	  0.0038	    51.3552	  0.03	  0.92	  100.00%	  4096	    4.2088	 35.07%	      0.05%	  0.1092
0.01.394.705 I    22	attn_kv             	     99.95	  0.0062	    33.5665	  0.02	  0.59	  100.00%	  4096	    6.6583	 55.49%	      0.15%	  0.1192
0.01.394.706 I    38	attn_kv             	     89.28	  0.0043	    33.0348	  0.02	  0.57	  100.00%	  4096	    6.4538	 53.78%	      0.10%	  0.1333
0.01.394.707 I    24	attn_kv             	     73.88	  0.0054	    20.3442	  0.02	  0.37	  100.00%	  4096	    7.3008	 60.84%	      0.15%	  0.1450
0.01.394.709 I    28	attn_kv             	     69.13	  0.0022	    51.1201	  0.02	  0.80	  100.00%	  4096	    3.4863	 29.05%	      0.05%	  0.1772
0.01.394.710 I    20	attn_kv             	     68.64	  0.0028	    25.5560	  0.02	  0.55	  100.00%	  4096	    4.7777	 39.81%	      0.05%	  0.1845
0.01.394.711 I    32	attn_kv             	     52.57	  0.0024	    24.4735	  0.01	  0.43	  100.00%	  4096	    4.8843	 40.70%	      0.05%	  0.1101
0.01.394.713 I    30	attn_kv             	     51.31	  0.0025	    13.7820	  0.01	  0.31	  100.00%	  4096	    5.9912	 49.93%	      0.07%	  0.1196
0.01.394.714 I    14	attn_kv             	     49.35	  0.0032	    22.9597	  0.01	  0.37	  100.00%	  4096	    6.2063	 51.72%	      0.05%	  0.0995
0.01.394.715 I    12	attn_kv             	     43.61	  0.0025	    15.9644	  0.01	  0.27	  100.00%	  4096	    6.3918	 53.26%	      0.10%	  0.1075
0.01.394.717 I    26	attn_kv             	     42.05	  0.0022	    17.4225	  0.01	  0.33	  100.00%	  4096	    5.3200	 44.33%	      0.05%	  0.1784
0.01.394.718 I    18	attn_kv             	     40.06	  0.0013	    15.5157	  0.01	  0.30	  100.00%	  4096	    4.5824	 38.19%	      0.07%	  0.4172
0.01.394.719 I    23	attn_kv             	     34.69	  0.0062	     0.0478	  0.01	  0.00	  100.00%	  4096	   11.9715	 99.76%	      8.57%	  0.0627
0.01.394.721 I    36	attn_kv             	     33.18	  0.0021	    17.9069	  0.01	  0.28	  100.00%	  4096	    6.4848	 54.04%	      0.02%	  0.0775
0.01.394.722 I     8	attn_kv             	     28.93	  0.0024	     9.8737	  0.01	  0.16	  100.00%	  4096	    7.7816	 64.85%	      0.05%	  0.2032
0.01.394.726 I    10	attn_kv             	     28.22	  0.0026	     6.9477	  0.01	  0.13	  100.00%	  4096	    8.4242	 70.20%	      0.07%	  0.1561
0.01.394.727 I    37	attn_kv             	     22.57	  0.0035	     0.0323	  0.01	  0.00	  100.00%	  4096	   11.9569	 99.64%	     10.18%	  0.0516
0.01.394.729 I    17	attn_kv             	     21.21	  0.0033	     0.5185	  0.01	  0.01	  100.00%	  4096	   11.8232	 98.53%	      0.17%	  0.7418
0.01.394.731 I    39	attn_kv             	     20.31	  0.0032	     0.0339	  0.00	  0.00	  100.00%	  4096	   11.9553	 99.63%	     10.33%	  0.1030
0.01.394.732 I    41	attn_kv             	     16.61	  0.0023	     0.0227	  0.00	  0.00	  100.00%	  4096	   11.9400	 99.50%	     11.69%	  0.0304
0.01.394.734 I     4	attn_kv             	     16.30	  0.0014	     3.8507	  0.00	  0.07	  100.00%	  4096	    8.4898	 70.75%	      0.05%	  0.2090
0.01.394.735 I     6	attn_kv             	     15.27	  0.0012	     6.2976	  0.00	  0.10	  100.00%	  4096	    7.5735	 63.11%	      0.05%	  0.2231
0.01.394.736 I    35	attn_kv             	     14.15	  0.0023	     0.0309	  0.00	  0.00	  100.00%	  4096	   11.9544	 99.62%	      9.84%	  0.1415
0.01.394.738 I    42	attn_kv             	     13.63	  0.0011	     0.5170	  0.00	  0.01	  100.00%	  4096	   11.5541	 96.28%	      0.15%	  0.3448
0.01.394.739 I     2	attn_kv             	     11.12	  0.0012	     0.7982	  0.00	  0.01	  100.00%	  4096	   11.2380	 93.65%	      0.10%	  0.2404
0.01.394.740 I    15	attn_kv             	      9.99	  0.0016	     0.1894	  0.00	  0.00	  100.00%	  4096	   11.8296	 98.58%	      0.32%	  0.7024
0.01.394.742 I    21	attn_kv             	      8.85	  0.0014	     0.0163	  0.00	  0.00	  100.00%	  4096	   11.9458	 99.55%	      7.96%	  0.0821
0.01.394.743 I     3	attn_kv             	      8.04	  0.0009	     0.0289	  0.00	  0.00	  100.00%	  4096	   11.9316	 99.43%	      6.79%	  0.2356
0.01.394.744 I    33	attn_kv             	      7.95	  0.0011	     0.0167	  0.00	  0.00	  100.00%	  4096	   11.9395	 99.50%	      8.11%	  0.1514
0.01.394.746 I    27	attn_kv             	      7.77	  0.0013	     0.0218	  0.00	  0.00	  100.00%	  4096	   11.9554	 99.63%	      7.50%	  0.0726
0.01.394.747 I    29	attn_kv             	      7.30	  0.0011	     0.0222	  0.00	  0.00	  100.00%	  4096	   11.9394	 99.50%	      7.23%	  0.1927
0.01.394.748 I    31	attn_kv             	      6.88	  0.0011	     0.0133	  0.00	  0.00	  100.00%	  4096	   11.9600	 99.67%	      8.37%	  0.1022
0.01.394.749 I     1	attn_kv             	      6.15	  0.0008	     0.0074	  0.00	  0.00	  100.00%	  4096	   11.9415	 99.51%	     11.82%	  0.8979
0.01.394.751 I    19	attn_kv             	      5.41	  0.0008	     0.0172	  0.00	  0.00	  100.00%	  4096	   11.9392	 99.49%	      6.59%	  0.0859
0.01.394.752 I    13	attn_kv             	      4.84	  0.0008	     0.0523	  0.00	  0.00	  100.00%	  4096	   11.9165	 99.30%	      1.49%	  0.1068
0.01.394.753 I     9	attn_kv             	      4.31	  0.0006	     0.0394	  0.00	  0.00	  100.00%	  4096	   11.8949	 99.12%	      1.76%	  0.1423
0.01.394.754 I    11	attn_kv             	      4.06	  0.0006	     0.0456	  0.00	  0.00	  100.00%	  4096	   11.9104	 99.25%	      1.88%	  0.0699
0.01.394.756 I     7	attn_kv             	      3.64	  0.0005	     0.0231	  0.00	  0.00	  100.00%	  4096	   11.9319	 99.43%	      3.76%	  0.1596
0.01.394.757 I     5	attn_kv             	      3.55	  0.0004	     0.0109	  0.00	  0.00	  100.00%	  4096	   11.9422	 99.52%	      7.71%	  0.0666
0.01.394.758 I     0	attn_kv             	      3.51	  0.0004	     0.0035	  0.00	  0.00	  100.00%	  4096	   11.8854	 99.04%	     13.13%	  0.0000
0.01.394.760 I    25	attn_kv             	      2.26	  0.0003	     0.0059	  0.00	  0.00	  100.00%	  4096	   11.9538	 99.62%	      7.40%	  0.1029
0.01.394.761 I    23	attn_output_b       	   3581.84	  0.3038	     0.8507	  0.44	  0.06	  100.00%	  8192	   12.9878	 99.91%	     17.40%	  0.9119
0.01.394.763 I    20	attn_output_b       	   2880.15	  0.1662	     2.4122	  0.35	  0.14	  100.00%	  8192	   12.9004	 99.23%	     11.71%	  0.8488
0.01.394.764 I    40	attn_output_b       	   2351.54	  0.0638	     2.5421	  0.29	  0.29	  100.00%	  8192	   12.4811	 96.01%	     13.71%	  0.5861
0.01.394.766 I    42	attn_output_b       	   2176.09	  0.0531	     2.0464	  0.27	  0.17	  100.00%	  8192	   12.7526	 98.10%	     11.85%	  0.7182
0.01.394.767 I    14	attn_output_b       	   2101.89	  0.1391	     0.8400	  0.26	  0.06	  100.00%	  8192	   12.9676	 99.75%	     11.80%	  0.9361
0.01.394.769 I    36	attn_output_b       	   2065.02	  0.1069	     1.2942	  0.25	  0.10	  100.00%	  8192	   12.8996	 99.23%	     12.73%	  0.8034
0.01.394.770 I    16	attn_output_b       	   2056.66	  0.1338	     0.7591	  0.25	  0.07	  100.00%	  8192	   12.9483	 99.60%	     14.45%	  0.9449
0.01.394.772 I    39	attn_output_b       	   2051.65	  0.0782	     1.0098	  0.25	  0.14	  100.00%	  8192	   12.8106	 98.54%	     16.17%	  0.6377
0.01.394.773 I    21	attn_output_b       	   1911.73	  0.1061	     0.6612	  0.23	  0.08	  100.00%	  8192	   12.9123	 99.33%	     20.17%	  0.8317
0.01.394.774 I    22	attn_output_b       	   1772.11	  0.0518	     1.0062	  0.22	  0.09	  100.00%	  8192	   12.8798	 99.08%	     14.93%	  0.8832
0.01.394.776 I     1	attn_output_b       	   1711.11	  0.0951	     0.9123	  0.21	  0.09	  100.00%	  8192	   12.8886	 99.14%	     14.59%	  0.8411
0.01.394.777 I    35	attn_output_b       	   1632.48	  0.0773	     0.7368	  0.20	  0.11	  100.00%	  8192	   12.8238	 98.64%	     13.48%	  0.8271
0.01.394.779 I    41	attn_output_b       	   1617.08	  0.0953	     0.8022	  0.20	  0.09	  100.00%	  8192	   12.8775	 99.06%	     13.90%	  0.5568
0.01.394.780 I    33	attn_output_b       	   1606.15	  0.0947	     0.9392	  0.20	  0.07	  100.00%	  8192	   12.9131	 99.33%	     14.20%	  0.9104
0.01.394.782 I    26	attn_output_b       	   1576.15	  0.0855	     1.1007	  0.19	  0.12	  100.00%	  8192	   12.7958	 98.43%	     12.02%	  0.7834
0.01.394.783 I     0	attn_output_b       	   1483.62	  0.0584	     1.1093	  0.18	  0.09	  100.00%	  8192	   12.8431	 98.79%	     12.89%	  0.0000
0.01.394.785 I    34	attn_output_b       	   1478.00	  0.0728	     1.0362	  0.18	  0.08	  100.00%	  8192	   12.8743	 99.03%	     16.44%	  0.8709
0.01.394.786 I    24	attn_output_b       	   1411.71	  0.0237	     2.0718	  0.17	  0.21	  100.00%	  8192	   12.2212	 94.01%	     15.53%	  0.6513
0.01.394.788 I    27	attn_output_b       	   1342.26	  0.0419	     0.9930	  0.16	  0.11	  100.00%	  8192	   12.7114	 97.78%	     16.60%	  0.6959
0.01.394.789 I    32	attn_output_b       	   1290.32	  0.0541	     0.8315	  0.16	  0.08	  100.00%	  8192	   12.8404	 98.77%	     15.28%	  0.7974
0.01.394.790 I     3	attn_output_b       	   1270.99	  0.0674	     0.8279	  0.16	  0.07	  100.00%	  8192	   12.8862	 99.12%	     11.85%	  0.8742
0.01.394.792 I    19	attn_output_b       	   1226.04	  0.0714	     0.5792	  0.15	  0.05	  100.00%	  8192	   12.9153	 99.35%	     16.15%	  0.8335
0.01.394.793 I    38	attn_output_b       	   1157.85	  0.0414	     1.2268	  0.14	  0.11	  100.00%	  8192	   12.6977	 97.67%	     12.81%	  0.6677
0.01.394.795 I    18	attn_output_b       	   1144.72	  0.0675	     0.6266	  0.14	  0.06	  100.00%	  8192	   12.8861	 99.12%	     15.54%	  0.8908
0.01.394.796 I     9	attn_output_b       	   1088.81	  0.0699	     0.5699	  0.13	  0.04	  100.00%	  8192	   12.9530	 99.64%	     11.57%	  0.8957
0.01.394.798 I    15	attn_output_b       	   1086.43	  0.0685	     0.3864	  0.13	  0.03	  100.00%	  8192	   12.9545	 99.65%	     14.14%	  0.9411
0.01.394.799 I    17	attn_output_b       	    923.78	  0.0654	     0.3838	  0.11	  0.03	  100.00%	  8192	   12.9484	 99.60%	     16.52%	  0.9240
0.01.394.801 I    12	attn_output_b       	    918.84	  0.0671	     0.4816	  0.11	  0.04	  100.00%	  8192	   12.9212	 99.39%	     13.99%	  0.9527
0.01.394.802 I    10	attn_output_b       	    906.04	  0.0473	     0.4872	  0.11	  0.04	  100.00%	  8192	   12.9292	 99.46%	     14.86%	  0.9286
0.01.394.804 I    13	attn_output_b       	    871.54	  0.0596	     0.3918	  0.11	  0.03	  100.00%	  8192	   12.9341	 99.49%	     15.39%	  0.8984
0.01.394.805 I     2	attn_output_b       	    835.11	  0.0357	     0.4255	  0.10	  0.04	  100.00%	  8192	   12.8992	 99.22%	     12.93%	  0.8601
0.01.394.807 I    28	attn_output_b       	    814.70	  0.0388	     0.3836	  0.10	  0.03	  100.00%	  8192	   12.9200	 99.38%	     12.84%	  0.8139
0.01.394.808 I    37	attn_output_b       	    808.60	  0.0414	     0.3082	  0.10	  0.04	  100.00%	  8192	   12.9025	 99.25%	     17.53%	  0.8841
0.01.394.809 I     7	attn_output_b       	    792.06	  0.0542	     0.2858	  0.10	  0.02	  100.00%	  8192	   12.9740	 99.80%	     12.21%	  0.9388
0.01.394.811 I    11	attn_output_b       	    765.32	  0.0530	     0.2424	  0.09	  0.02	  100.00%	  8192	   12.9607	 99.70%	     17.19%	  0.9173
0.01.394.812 I     5	attn_output_b       	    695.47	  0.0372	     0.2798	  0.08	  0.03	  100.00%	  8192	   12.9420	 99.55%	     12.90%	  0.8779
0.01.394.814 I    25	attn_output_b       	    685.94	  0.0263	     0.4113	  0.08	  0.04	  100.00%	  8192	   12.8593	 98.92%	     13.53%	  0.6016
0.01.394.815 I    30	attn_output_b       	    674.04	  0.0339	     0.4769	  0.08	  0.04	  100.00%	  8192	   12.8376	 98.75%	     13.05%	  0.8927
0.01.394.817 I    29	attn_output_b       	    663.74	  0.0373	     0.2639	  0.08	  0.02	  100.00%	  8192	   12.9355	 99.50%	     16.21%	  0.9112
0.01.394.818 I     8	attn_output_b       	    611.16	  0.0289	     0.2706	  0.07	  0.03	  100.00%	  8192	   12.9087	 99.30%	     13.66%	  0.9258
0.01.394.819 I    31	attn_output_b       	    578.22	  0.0411	     0.3068	  0.07	  0.04	  100.00%	  8192	   12.8442	 98.80%	     12.59%	  0.7300
0.01.394.821 I     4	attn_output_b       	    322.41	  0.0140	     0.1865	  0.04	  0.02	  100.00%	  8192	   12.8846	 99.11%	     12.68%	  0.8445
0.01.394.822 I     6	attn_output_b       	    207.78	  0.0102	     0.0892	  0.03	  0.01	  100.00%	  8192	   12.9350	 99.50%	     12.62%	  0.9104
0.01.394.824 I    40	attn_q_a            	    220.46	  0.0040	   177.8002	  0.05	  2.78	  100.00%	  4096	    2.5334	 21.11%	      0.07%	  0.0866
0.01.394.825 I    34	attn_q_a            	    152.95	  0.0033	   102.6580	  0.04	  1.67	  100.00%	  4096	    2.7774	 23.15%	      0.05%	  0.1072
0.01.394.826 I    16	attn_q_a            	    102.98	  0.0038	    51.3552	  0.03	  0.92	  100.00%	  4096	    4.2088	 35.07%	      0.05%	  0.1092
0.01.394.828 I    22	attn_q_a            	     99.95	  0.0062	    33.5665	  0.02	  0.59	  100.00%	  4096	    6.6583	 55.49%	      0.15%	  0.1192
0.01.394.829 I    38	attn_q_a            	     89.28	  0.0043	    33.0348	  0.02	  0.57	  100.00%	  4096	    6.4538	 53.78%	      0.10%	  0.1333
0.01.394.831 I    24	attn_q_a            	     73.88	  0.0054	    20.3442	  0.02	  0.37	  100.00%	  4096	    7.3008	 60.84%	      0.15%	  0.1450
0.01.394.832 I    28	attn_q_a            	     69.13	  0.0022	    51.1201	  0.02	  0.80	  100.00%	  4096	    3.4863	 29.05%	      0.05%	  0.1772
0.01.394.833 I    20	attn_q_a            	     68.64	  0.0028	    25.5560	  0.02	  0.55	  100.00%	  4096	    4.7777	 39.81%	      0.05%	  0.1845
0.01.394.835 I    32	attn_q_a            	     52.57	  0.0024	    24.4735	  0.01	  0.43	  100.00%	  4096	    4.8843	 40.70%	      0.05%	  0.1101
0.01.394.836 I    30	attn_q_a            	     51.31	  0.0025	    13.7820	  0.01	  0.31	  100.00%	  4096	    5.9912	 49.93%	      0.07%	  0.1196
0.01.394.837 I    14	attn_q_a            	     49.35	  0.0032	    22.9597	  0.01	  0.37	  100.00%	  4096	    6.2063	 51.72%	      0.05%	  0.0995
0.01.394.839 I    12	attn_q_a            	     43.61	  0.0025	    15.9644	  0.01	  0.27	  100.00%	  4096	    6.3918	 53.26%	      0.10%	  0.1075
0.01.394.840 I    26	attn_q_a            	     42.05	  0.0022	    17.4225	  0.01	  0.33	  100.00%	  4096	    5.3200	 44.33%	      0.05%	  0.1784
0.01.394.841 I    18	attn_q_a            	     40.06	  0.0013	    15.5157	  0.01	  0.30	  100.00%	  4096	    4.5824	 38.19%	      0.07%	  0.4172
0.01.394.843 I    23	attn_q_a            	     34.69	  0.0062	     0.0478	  0.01	  0.00	  100.00%	  4096	   11.9715	 99.76%	      8.57%	  0.0627
0.01.394.844 I    36	attn_q_a            	     33.18	  0.0021	    17.9069	  0.01	  0.28	  100.00%	  4096	    6.4848	 54.04%	      0.02%	  0.0775
0.01.394.845 I     8	attn_q_a            	     28.93	  0.0024	     9.8737	  0.01	  0.16	  100.00%	  4096	    7.7816	 64.85%	      0.05%	  0.2032
0.01.394.846 I    10	attn_q_a            	     28.22	  0.0026	     6.9477	  0.01	  0.13	  100.00%	  4096	    8.4242	 70.20%	      0.07%	  0.1561
0.01.394.848 I    37	attn_q_a            	     22.57	  0.0035	     0.0323	  0.01	  0.00	  100.00%	  4096	   11.9569	 99.64%	     10.18%	  0.0516
0.01.394.849 I    17	attn_q_a            	     21.21	  0.0033	     0.5185	  0.01	  0.01	  100.00%	  4096	   11.8232	 98.53%	      0.17%	  0.7418
0.01.394.850 I    39	attn_q_a            	     20.31	  0.0032	     0.0339	  0.00	  0.00	  100.00%	  4096	   11.9553	 99.63%	     10.33%	  0.1030
0.01.394.852 I    41	attn_q_a            	     16.61	  0.0023	     0.0227	  0.00	  0.00	  100.00%	  4096	   11.9400	 99.50%	     11.69%	  0.0304
0.01.394.853 I     4	attn_q_a            	     16.30	  0.0014	     3.8507	  0.00	  0.07	  100.00%	  4096	    8.4898	 70.75%	      0.05%	  0.2090
0.01.394.854 I     6	attn_q_a            	     15.27	  0.0012	     6.2976	  0.00	  0.10	  100.00%	  4096	    7.5735	 63.11%	      0.05%	  0.2231
0.01.394.856 I    35	attn_q_a            	     14.15	  0.0023	     0.0309	  0.00	  0.00	  100.00%	  4096	   11.9544	 99.62%	      9.84%	  0.1415
0.01.394.857 I    42	attn_q_a            	     13.63	  0.0011	     0.5170	  0.00	  0.01	  100.00%	  4096	   11.5541	 96.28%	      0.15%	  0.3448
0.01.394.858 I     2	attn_q_a            	     11.12	  0.0012	     0.7982	  0.00	  0.01	  100.00%	  4096	   11.2380	 93.65%	      0.10%	  0.2404
0.01.394.860 I    15	attn_q_a            	      9.99	  0.0016	     0.1894	  0.00	  0.00	  100.00%	  4096	   11.8296	 98.58%	      0.32%	  0.7024
0.01.394.861 I    21	attn_q_a            	      8.85	  0.0014	     0.0163	  0.00	  0.00	  100.00%	  4096	   11.9458	 99.55%	      7.96%	  0.0821
0.01.394.862 I     3	attn_q_a            	      8.04	  0.0009	     0.0289	  0.00	  0.00	  100.00%	  4096	   11.9316	 99.43%	      6.79%	  0.2356
0.01.394.863 I    33	attn_q_a            	      7.95	  0.0011	     0.0167	  0.00	  0.00	  100.00%	  4096	   11.9395	 99.50%	      8.11%	  0.1514
0.01.394.865 I    27	attn_q_a            	      7.77	  0.0013	     0.0218	  0.00	  0.00	  100.00%	  4096	   11.9554	 99.63%	      7.50%	  0.0726
0.01.394.866 I    29	attn_q_a            	      7.30	  0.0011	     0.0222	  0.00	  0.00	  100.00%	  4096	   11.9394	 99.50%	      7.23%	  0.1927
0.01.394.867 I    31	attn_q_a            	      6.88	  0.0011	     0.0133	  0.00	  0.00	  100.00%	  4096	   11.9600	 99.67%	      8.37%	  0.1022
0.01.394.869 I     1	attn_q_a            	      6.15	  0.0008	     0.0074	  0.00	  0.00	  100.00%	  4096	   11.9415	 99.51%	     11.82%	  0.8979
0.01.394.870 I    19	attn_q_a            	      5.41	  0.0008	     0.0172	  0.00	  0.00	  100.00%	  4096	   11.9392	 99.49%	      6.59%	  0.0859
0.01.394.871 I    13	attn_q_a            	      4.84	  0.0008	     0.0523	  0.00	  0.00	  100.00%	  4096	   11.9165	 99.30%	      1.49%	  0.1068
0.01.394.872 I     9	attn_q_a            	      4.31	  0.0006	     0.0394	  0.00	  0.00	  100.00%	  4096	   11.8949	 99.12%	      1.76%	  0.1423
0.01.394.874 I    11	attn_q_a            	      4.06	  0.0006	     0.0456	  0.00	  0.00	  100.00%	  4096	   11.9104	 99.25%	      1.88%	  0.0699
0.01.394.875 I     7	attn_q_a            	      3.64	  0.0005	     0.0231	  0.00	  0.00	  100.00%	  4096	   11.9319	 99.43%	      3.76%	  0.1596
0.01.394.876 I     5	attn_q_a            	      3.55	  0.0004	     0.0109	  0.00	  0.00	  100.00%	  4096	   11.9422	 99.52%	      7.71%	  0.0666
0.01.394.878 I     0	attn_q_a            	      3.51	  0.0004	     0.0035	  0.00	  0.00	  100.00%	  4096	   11.8854	 99.04%	     13.13%	  0.0000
0.01.394.879 I    25	attn_q_a            	      2.26	  0.0003	     0.0059	  0.00	  0.00	  100.00%	  4096	   11.9538	 99.62%	      7.40%	  0.1029
0.01.394.880 I    23	attn_q_b            	     16.58	  0.0049	     0.1550	  0.02	  0.02	  100.00%	  1024	    9.5458	 95.46%	      8.79%	  0.6091
0.01.394.882 I    20	attn_q_b            	      5.08	  0.0021	     0.0320	  0.00	  0.00	  100.00%	  1024	    9.7892	 97.89%	     10.25%	  0.0000
0.01.394.883 I    20	attn_q_b            	      5.08	  0.0021	     0.0320	  0.00	  0.00	  100.00%	  1024	    9.7892	 97.89%	     10.25%	  0.6222
0.01.394.884 I    40	attn_q_b            	      4.34	  0.0015	     0.0374	  0.00	  0.00	  100.00%	  1024	    9.6359	 96.36%	      9.18%	  0.0000
0.01.394.885 I    40	attn_q_b            	      4.34	  0.0015	     0.0374	  0.00	  0.00	  100.00%	  1024	    9.6359	 96.36%	      9.18%	  0.5013
0.01.394.887 I    22	attn_q_b            	      4.14	  0.0018	     0.0215	  0.00	  0.00	  100.00%	  1024	    9.7974	 97.97%	     11.33%	  0.6243
0.01.394.888 I    22	attn_q_b            	      4.14	  0.0018	     0.0215	  0.00	  0.00	  100.00%	  1024	    9.7974	 97.97%	     11.33%	  0.0000
0.01.394.889 I    21	attn_q_b            	      4.04	  0.0014	     0.0397	  0.00	  0.00	  100.00%	  1024	    9.6255	 96.26%	      8.40%	  0.6015
0.01.394.891 I    16	attn_q_b            	      3.25	  0.0014	     0.0184	  0.00	  0.00	  100.00%	  1024	    9.8074	 98.07%	     10.84%	  0.7010
0.01.394.892 I    16	attn_q_b            	      3.25	  0.0014	     0.0184	  0.00	  0.00	  100.00%	  1024	    9.8074	 98.07%	     10.84%	  0.0000
0.01.394.893 I    38	attn_q_b            	      3.24	  0.0011	     0.0231	  0.00	  0.00	  100.00%	  1024	    9.6794	 96.79%	     11.13%	  0.0000
0.01.394.894 I    38	attn_q_b            	      3.24	  0.0011	     0.0231	  0.00	  0.00	  100.00%	  1024	    9.6794	 96.79%	     11.13%	  0.6144
0.01.394.896 I    19	attn_q_b            	      3.18	  0.0011	     0.0311	  0.00	  0.00	  100.00%	  1024	    9.6328	 96.33%	      9.28%	  0.6031
0.01.394.897 I    24	attn_q_b            	      3.15	  0.0013	     0.0214	  0.00	  0.00	  100.00%	  1024	    9.7584	 97.58%	     11.13%	  0.0000
0.01.394.898 I    24	attn_q_b            	      3.15	  0.0013	     0.0214	  0.00	  0.00	  100.00%	  1024	    9.7584	 97.58%	     11.13%	  0.5628
0.01.394.900 I    35	attn_q_b            	      3.11	  0.0011	     0.0416	  0.00	  0.00	  100.00%	  1024	    9.5161	 95.16%	      6.54%	  0.4544
0.01.394.901 I    26	attn_q_b            	      2.96	  0.0012	     0.0221	  0.00	  0.00	  100.00%	  1024	    9.7940	 97.94%	     10.45%	  0.0000
0.01.394.902 I    26	attn_q_b            	      2.96	  0.0012	     0.0221	  0.00	  0.00	  100.00%	  1024	    9.7940	 97.94%	     10.45%	  0.6728
0.01.394.903 I    31	attn_q_b            	      2.80	  0.0011	     0.0340	  0.00	  0.00	  100.00%	  1024	    9.7371	 97.37%	      8.30%	  0.6913
0.01.394.905 I    14	attn_q_b            	      2.73	  0.0013	     0.0140	  0.00	  0.00	  100.00%	  1024	    9.8522	 98.52%	     10.16%	  0.7219
0.01.394.906 I    14	attn_q_b            	      2.73	  0.0013	     0.0140	  0.00	  0.00	  100.00%	  1024	    9.8522	 98.52%	     10.16%	  0.0000
0.01.394.907 I    30	attn_q_b            	      2.62	  0.0012	     0.0125	  0.00	  0.00	  100.00%	  1024	    9.8370	 98.37%	     11.23%	  0.0000
0.01.394.909 I    30	attn_q_b            	      2.62	  0.0012	     0.0125	  0.00	  0.00	  100.00%	  1024	    9.8370	 98.37%	     11.23%	  0.6523
0.01.394.910 I    29	attn_q_b            	      2.54	  0.0009	     0.0191	  0.00	  0.00	  100.00%	  1024	    9.6269	 96.27%	      9.18%	  0.6260
0.01.394.911 I    33	attn_q_b            	      2.52	  0.0009	     0.0274	  0.00	  0.00	  100.00%	  1024	    9.5857	 95.86%	     10.55%	  0.5832
0.01.394.912 I    32	attn_q_b            	      2.49	  0.0010	     0.0137	  0.00	  0.00	  100.00%	  1024	    9.7638	 97.64%	      9.96%	  0.0000
0.01.394.914 I    32	attn_q_b            	      2.49	  0.0010	     0.0137	  0.00	  0.00	  100.00%	  1024	    9.7638	 97.64%	      9.96%	  0.6332
0.01.394.915 I    36	attn_q_b            	      2.40	  0.0008	     0.0260	  0.00	  0.00	  100.00%	  1024	    9.6270	 96.27%	     10.35%	  0.4712
0.01.394.916 I    36	attn_q_b            	      2.40	  0.0008	     0.0260	  0.00	  0.00	  100.00%	  1024	    9.6270	 96.27%	     10.35%	  0.0000
0.01.394.918 I    34	attn_q_b            	      2.37	  0.0008	     0.0437	  0.00	  0.00	  100.00%	  1024	    9.5402	 95.40%	      6.45%	  0.0000
0.01.394.919 I    34	attn_q_b            	      2.37	  0.0008	     0.0437	  0.00	  0.00	  100.00%	  1024	    9.5402	 95.40%	      6.45%	  0.4768
0.01.394.920 I    39	attn_q_b            	      2.37	  0.0008	     0.0233	  0.00	  0.00	  100.00%	  1024	    9.5292	 95.29%	      8.50%	  0.5152
0.01.394.921 I    15	attn_q_b            	      2.34	  0.0009	     0.0157	  0.00	  0.00	  100.00%	  1024	    9.7519	 97.52%	     11.43%	  0.7248
0.01.394.923 I    41	attn_q_b            	      2.32	  0.0007	     0.0176	  0.00	  0.00	  100.00%	  1024	    9.6110	 96.11%	     11.33%	  0.5424
0.01.394.924 I    17	attn_q_b            	      2.31	  0.0009	     0.0148	  0.00	  0.00	  100.00%	  1024	    9.7424	 97.42%	     11.04%	  0.6940
0.01.394.925 I    18	attn_q_b            	      2.30	  0.0010	     0.0151	  0.00	  0.00	  100.00%	  1024	    9.7582	 97.58%	      9.47%	  0.0000
0.01.394.927 I    18	attn_q_b            	      2.30	  0.0010	     0.0151	  0.00	  0.00	  100.00%	  1024	    9.7582	 97.58%	      9.47%	  0.6738
0.01.394.928 I    37	attn_q_b            	      2.22	  0.0008	     0.0178	  0.00	  0.00	  100.00%	  1024	    9.6747	 96.75%	     12.21%	  0.6034
0.01.394.929 I    27	attn_q_b            	      2.14	  0.0008	     0.0220	  0.00	  0.00	  100.00%	  1024	    9.6712	 96.71%	      8.98%	  0.6416
0.01.394.931 I    42	attn_q_b            	      2.02	  0.0002	     0.0135	  0.00	  0.00	  100.00%	  1024	    9.7011	 97.01%	     11.04%	  0.5898
0.01.394.932 I    42	attn_q_b            	      2.02	  0.0002	     0.0135	  0.00	  0.00	  100.00%	  1024	    9.7011	 97.01%	     11.04%	  0.0000
0.01.394.933 I    13	attn_q_b            	      1.77	  0.0007	     0.0113	  0.00	  0.00	  100.00%	  1024	    9.7616	 97.62%	     10.55%	  0.7507
0.01.394.934 I    28	attn_q_b            	      1.71	  0.0007	     0.0100	  0.00	  0.00	  100.00%	  1024	    9.7708	 97.71%	     11.72%	  0.6398
0.01.394.936 I    28	attn_q_b            	      1.71	  0.0007	     0.0100	  0.00	  0.00	  100.00%	  1024	    9.7708	 97.71%	     11.72%	  0.0000
0.01.394.937 I     1	attn_q_b            	      1.56	  0.0004	     0.0093	  0.00	  0.00	  100.00%	  1024	    9.6883	 96.88%	     12.60%	  0.6609
0.01.394.939 I    12	attn_q_b            	      1.47	  0.0007	     0.0057	  0.00	  0.00	  100.00%	  1024	    9.8694	 98.69%	     13.38%	  0.7833
0.01.394.940 I    12	attn_q_b            	      1.47	  0.0007	     0.0057	  0.00	  0.00	  100.00%	  1024	    9.8694	 98.69%	     13.38%	  0.0000
0.01.394.941 I     0	attn_q_b            	      1.43	  0.0003	     0.0073	  0.00	  0.00	  100.00%	  1024	    9.7399	 97.40%	     13.18%	  0.0000
0.01.394.942 I    25	attn_q_b            	      1.39	  0.0005	     0.0097	  0.00	  0.00	  100.00%	  1024	    9.7270	 97.27%	      9.67%	  0.6385
0.01.394.944 I    11	attn_q_b            	      1.36	  0.0005	     0.0105	  0.00	  0.00	  100.00%	  1024	    9.8025	 98.02%	     11.13%	  0.7683
0.01.394.945 I     8	attn_q_b            	      1.33	  0.0006	     0.0072	  0.00	  0.00	  100.00%	  1024	    9.8270	 98.27%	     10.94%	  0.0000
0.01.394.946 I     8	attn_q_b            	      1.33	  0.0006	     0.0072	  0.00	  0.00	  100.00%	  1024	    9.8270	 98.27%	     10.94%	  0.7499
0.01.394.947 I    10	attn_q_b            	      1.25	  0.0007	     0.0058	  0.00	  0.00	  100.00%	  1024	    9.8879	 98.88%	     11.43%	  0.0000
0.01.394.948 I    10	attn_q_b            	      1.25	  0.0007	     0.0058	  0.00	  0.00	  100.00%	  1024	    9.8879	 98.88%	     11.43%	  0.7095
0.01.394.950 I     9	attn_q_b            	      1.14	  0.0004	     0.0107	  0.00	  0.00	  100.00%	  1024	    9.6961	 96.96%	     10.16%	  0.6890
0.01.394.951 I     7	attn_q_b            	      1.13	  0.0005	     0.0052	  0.00	  0.00	  100.00%	  1024	    9.8205	 98.21%	     11.13%	  0.7377
0.01.394.952 I     5	attn_q_b            	      0.92	  0.0003	     0.0055	  0.00	  0.00	  100.00%	  1024	    9.7593	 97.59%	     11.72%	  0.7236
0.01.394.954 I     2	attn_q_b            	      0.92	  0.0004	     0.0045	  0.00	  0.00	  100.00%	  1024	    9.8724	 98.72%	     11.91%	  0.0000
0.01.394.955 I     2	attn_q_b            	      0.92	  0.0004	     0.0045	  0.00	  0.00	  100.00%	  1024	    9.8724	 98.72%	     11.91%	  0.7142
0.01.394.956 I     4	attn_q_b            	      0.87	  0.0003	     0.0041	  0.00	  0.00	  100.00%	  1024	    9.8107	 98.11%	     12.79%	  0.0000
0.01.394.958 I     4	attn_q_b            	      0.87	  0.0003	     0.0041	  0.00	  0.00	  100.00%	  1024	    9.8107	 98.11%	     12.79%	  0.7286
0.01.394.959 I     3	attn_q_b            	      0.84	  0.0003	     0.0049	  0.00	  0.00	  100.00%	  1024	    9.7836	 97.84%	     12.21%	  0.7632
0.01.394.960 I     6	attn_q_b            	      0.81	  0.0003	     0.0046	  0.00	  0.00	  100.00%	  1024	    9.7891	 97.89%	     10.06%	  0.6989
0.01.394.961 I     6	attn_q_b            	      0.81	  0.0003	     0.0046	  0.00	  0.00	  100.00%	  1024	    9.7891	 97.89%	     10.06%	  0.0000
0.01.394.963 I     0	blk.0.attn_output_a.weight (reshaped)	   3775.40	  0.0039	    25.1050	  0.12	  0.72	  100.00%	 32768	   12.4928	 83.29%	      1.31%	  0.0000
0.01.394.965 I     1	blk.1.attn_output_a.weight (reshaped)	   5707.69	  0.0114	    84.9634	  0.17	  1.99	  100.00%	 32768	   11.1066	 74.04%	      0.41%	  0.0170
0.01.394.966 I    10	blk.10.attn_output_a.weight (reshaped)	   4979.40	  0.0032	    11.9158	  0.15	  0.21	  100.00%	 32768	   14.5157	 96.77%	      4.21%	  0.2900
0.01.394.968 I    11	blk.11.attn_output_a.weight (reshaped)	   3321.86	  0.0163	     8.3151	  0.10	  0.27	  100.00%	 32768	   13.9372	 92.91%	      2.24%	  0.4810
0.01.394.969 I    12	blk.12.attn_output_a.weight (reshaped)	   4830.80	  0.0003	     8.5601	  0.15	  0.18	  100.00%	 32768	   14.5678	 97.12%	      4.28%	  0.4059
0.01.394.971 I    13	blk.13.attn_output_a.weight (reshaped)	   4565.09	  0.0196	    14.7889	  0.14	  0.36	  100.00%	 32768	   14.1977	 94.65%	      1.84%	  0.3103
0.01.394.972 I    14	blk.14.attn_output_a.weight (reshaped)	  12913.71	  0.0025	     8.4395	  0.39	  0.27	  100.00%	 32768	   14.7785	 98.52%	      7.18%	  0.2786
0.01.394.974 I    15	blk.15.attn_output_a.weight (reshaped)	   5223.95	  0.0074	     8.3440	  0.16	  0.30	  100.00%	 32768	   14.2805	 95.20%	      2.99%	  0.3969
0.01.394.975 I    16	blk.16.attn_output_a.weight (reshaped)	  13286.50	  0.0115	    12.6803	  0.41	  0.38	  100.00%	 32768	   14.7260	 98.17%	      2.92%	  0.3531
0.01.394.977 I    17	blk.17.attn_output_a.weight (reshaped)	   4392.06	  0.0001	    11.9341	  0.13	  0.27	  100.00%	 32768	   14.2544	 95.03%	      3.17%	  0.4404
0.01.394.978 I    18	blk.18.attn_output_a.weight (reshaped)	   7412.01	  0.0063	    25.7256	  0.23	  0.47	  100.00%	 32768	   14.2658	 95.11%	      2.08%	  0.2497
0.01.394.980 I    19	blk.19.attn_output_a.weight (reshaped)	   4964.39	  0.0073	    11.1614	  0.15	  0.32	  100.00%	 32768	   14.2715	 95.14%	      2.40%	  0.2544
0.01.394.981 I     2	blk.2.attn_output_a.weight (reshaped)	   3129.39	  0.0002	    13.1809	  0.10	  0.38	  100.00%	 32768	   13.1184	 87.46%	      2.45%	  0.0245
0.01.394.983 I    20	blk.20.attn_output_a.weight (reshaped)	  14582.57	  0.0027	     6.2166	  0.45	  0.25	  100.00%	 32768	   14.8030	 98.69%	     11.55%	  0.4338
0.01.394.984 I    21	blk.21.attn_output_a.weight (reshaped)	   8177.57	  0.0343	    27.1820	  0.25	  0.83	  100.00%	 32768	   13.9215	 92.81%	      1.49%	  0.2549
0.01.394.986 I    22	blk.22.attn_output_a.weight (reshaped)	  13451.09	  0.0088	    13.8423	  0.41	  0.60	  100.00%	 32768	   14.1774	 94.52%	      9.89%	  0.1919
0.01.394.987 I    23	blk.23.attn_output_a.weight (reshaped)	   6976.67	  0.0137	     8.4695	  0.21	  0.28	  100.00%	 32768	   14.5947	 97.30%	      2.27%	  0.3505
0.01.394.989 I    24	blk.24.attn_output_a.weight (reshaped)	   7175.15	  0.0003	    11.4520	  0.22	  0.52	  100.00%	 32768	   13.6473	 90.98%	      4.57%	  0.2240
0.01.394.990 I    25	blk.25.attn_output_a.weight (reshaped)	   2000.44	  0.0025	     9.2283	  0.06	  0.23	  100.00%	 32768	   13.2929	 88.62%	      2.17%	  0.1344
0.01.394.992 I    26	blk.26.attn_output_a.weight (reshaped)	   7071.32	  0.0000	    11.0091	  0.22	  0.32	  100.00%	 32768	   14.3471	 95.65%	      5.57%	  0.2128
0.01.394.993 I    27	blk.27.attn_output_a.weight (reshaped)	   2718.96	  0.0012	     6.9599	  0.08	  0.23	  100.00%	 32768	   13.6327	 90.88%	      2.91%	  0.1957
0.01.394.994 I    28	blk.28.attn_output_a.weight (reshaped)	   4520.24	  0.0001	    12.0188	  0.14	  0.36	  100.00%	 32768	   13.9190	 92.79%	      2.63%	  0.1536
0.01.394.996 I    29	blk.29.attn_output_a.weight (reshaped)	   3037.35	  0.0028	    12.2150	  0.09	  0.39	  100.00%	 32768	   13.1351	 87.57%	      1.64%	  0.1472
0.01.394.997 I     3	blk.3.attn_output_a.weight (reshaped)	   4507.79	  0.0013	    35.9606	  0.14	  0.99	  100.00%	 32768	   11.9675	 79.78%	      0.90%	  0.0867
0.01.394.999 I    30	blk.30.attn_output_a.weight (reshaped)	   5129.52	  0.0011	    21.8684	  0.16	  0.56	  100.00%	 32768	   13.4694	 89.80%	      2.32%	  0.0705
0.01.395.000 I    31	blk.31.attn_output_a.weight (reshaped)	   2303.49	  0.0019	    12.5896	  0.07	  0.30	  100.00%	 32768	   13.2993	 88.66%	      1.59%	  0.0678
0.01.395.002 I    32	blk.32.attn_output_a.weight (reshaped)	   5428.05	  0.0001	    11.7948	  0.17	  0.19	  100.00%	 32768	   14.5741	 97.16%	      6.77%	  0.1480
0.01.395.003 I    33	blk.33.attn_output_a.weight (reshaped)	   5739.78	  0.0159	    23.9675	  0.18	  0.72	  100.00%	 32768	   13.5033	 90.02%	      1.57%	  0.2070
0.01.395.005 I    34	blk.34.attn_output_a.weight (reshaped)	   7393.06	  0.0042	     6.5633	  0.23	  0.20	  100.00%	 32768	   14.7081	 98.05%	      6.85%	  0.1973
0.01.395.006 I    35	blk.35.attn_output_a.weight (reshaped)	   8027.18	  0.0411	    28.5640	  0.24	  0.76	  100.00%	 32768	   14.0351	 93.57%	      1.55%	  0.2343
0.01.395.008 I    36	blk.36.attn_output_a.weight (reshaped)	   5442.13	  0.0001	    12.0553	  0.17	  0.40	  100.00%	 32768	   14.0251	 93.50%	      2.59%	  0.1199
0.01.395.009 I    37	blk.37.attn_output_a.weight (reshaped)	   4037.17	  0.0154	    29.1565	  0.12	  0.75	  100.00%	 32768	   12.7493	 85.00%	      1.26%	  0.0722
0.01.395.011 I    38	blk.38.attn_output_a.weight (reshaped)	   6420.35	  0.0049	    21.8031	  0.20	  0.47	  100.00%	 32768	   14.1117	 94.08%	      2.28%	  0.0572
0.01.395.012 I    39	blk.39.attn_output_a.weight (reshaped)	   7157.78	  0.0191	    24.1994	  0.22	  0.57	  100.00%	 32768	   14.0681	 93.79%	      1.88%	  0.1960
0.01.395.014 I     4	blk.4.attn_output_a.weight (reshaped)	   2039.11	  0.0001	    17.3404	  0.06	  0.40	  100.00%	 32768	   12.3479	 82.32%	      1.32%	  0.0201
0.01.395.015 I    40	blk.40.attn_output_a.weight (reshaped)	   8467.82	  0.0002	    16.2467	  0.26	  0.28	  100.00%	 32768	   14.5969	 97.31%	      6.07%	  0.4279
0.01.395.016 I    41	blk.41.attn_output_a.weight (reshaped)	   6199.72	  0.0166	    21.7655	  0.19	  0.55	  100.00%	 32768	   13.9915	 93.28%	      1.61%	  0.4096
0.01.395.018 I    42	blk.42.attn_output_a.weight (reshaped)	   4594.31	  0.0002	    44.0105	  0.14	  0.91	  100.00%	 32768	   12.3018	 82.01%	      1.24%	  0.0454
0.01.395.019 I     5	blk.5.attn_output_a.weight (reshaped)	   2548.75	  0.0022	    17.0989	  0.08	  0.53	  100.00%	 32768	   12.5638	 83.76%	      0.99%	  0.0274
0.01.395.021 I     6	blk.6.attn_output_a.weight (reshaped)	   1487.96	  0.0002	     3.7224	  0.05	  0.11	  100.00%	 32768	   13.6390	 90.93%	      4.74%	  0.0955
0.01.395.022 I     7	blk.7.attn_output_a.weight (reshaped)	   2794.66	  0.0144	     8.5085	  0.09	  0.25	  100.00%	 32768	   14.0215	 93.48%	      2.03%	  0.2172
0.01.395.024 I     8	blk.8.attn_output_a.weight (reshaped)	   2704.77	  0.0001	     6.8156	  0.08	  0.21	  100.00%	 32768	   13.9526	 93.02%	      3.00%	  0.1747
0.01.395.025 I     9	blk.9.attn_output_a.weight (reshaped)	   3812.86	  0.0142	    10.4809	  0.12	  0.35	  100.00%	 32768	   13.7818	 91.88%	      2.49%	  0.1992
0.01.395.027 I    42	ffn_down_exps       	2188310.00	  0.0034	  2307.5815	  4.17	 42.45	  100.00%	524288	   13.8542	 72.92%	      0.95%	  0.0075
0.01.395.029 I    41	ffn_down_exps       	1342491.50	  0.0014	  1482.0859	  2.56	 30.71	  100.00%	524288	   13.2723	 69.85%	      0.84%	  0.0061
0.01.395.030 I    39	ffn_down_exps       	 326504.12	  0.0048	   237.1173	  0.62	  1.95	  100.00%	524288	   17.6960	 93.14%	      3.59%	  0.0121
0.01.395.032 I    40	ffn_down_exps       	 312423.81	  0.0026	   411.8223	  0.60	  5.31	  100.00%	524288	   15.6911	 82.58%	      0.84%	  0.0248
0.01.395.033 I    37	ffn_down_exps       	 222362.38	  0.0007	   489.9371	  0.42	  6.33	  100.00%	524288	   14.1953	 74.71%	      0.24%	  0.0012
0.01.395.035 I    36	ffn_down_exps       	 156658.64	  0.0230	  3400.4160	  0.30	  5.36	  100.00%	524288	   16.2856	 85.71%	      0.23%	  0.0081
0.01.395.036 I    38	ffn_down_exps       	 120999.50	  0.0140	   894.2704	  0.23	  1.63	  100.00%	524288	   17.5129	 92.17%	      0.95%	  0.0046
0.01.395.038 I    20	ffn_down_exps       	 118100.43	  0.0088	   302.3193	  0.23	  1.26	  100.00%	524288	   17.8586	 93.99%	      0.80%	  0.0530
0.01.395.039 I    35	ffn_down_exps       	 106088.88	  0.0172	    89.1712	  0.20	  0.60	  100.00%	524288	   17.9852	 94.66%	      2.55%	  0.1036
0.01.395.041 I    34	ffn_down_exps       	 105549.59	  0.0193	    57.0928	  0.20	  0.74	  100.00%	524288	   17.7523	 93.43%	      1.45%	  0.0199
0.01.395.042 I    22	ffn_down_exps       	  96935.09	  0.0094	   124.3763	  0.18	  0.48	  100.00%	524288	   18.1703	 95.63%	      2.99%	  0.0478
0.01.395.044 I    32	ffn_down_exps       	  86532.07	  0.0165	   152.3934	  0.17	  1.09	  100.00%	524288	   17.4677	 91.94%	      0.60%	  0.0596
0.01.395.045 I    30	ffn_down_exps       	  83201.19	  0.0204	   380.0406	  0.16	  1.08	  100.00%	524288	   17.3665	 91.40%	      0.47%	  0.0130
0.01.395.047 I    29	ffn_down_exps       	  77579.06	  0.0181	   310.9636	  0.15	  1.10	  100.00%	524288	   17.1501	 90.26%	      0.46%	  0.0343
0.01.395.048 I    26	ffn_down_exps       	  75697.02	  0.0131	   323.1925	  0.14	  0.63	  100.00%	524288	   18.0774	 95.14%	      1.35%	  0.0090
0.01.395.050 I    28	ffn_down_exps       	  70282.80	  0.0140	    95.2578	  0.13	  0.30	  100.00%	524288	   18.3332	 96.49%	      3.29%	  0.0554
0.01.395.051 I    33	ffn_down_exps       	  69956.70	  0.0164	   523.0419	  0.13	  0.87	  100.00%	524288	   18.1195	 95.37%	      0.65%	  0.0158
0.01.395.053 I    31	ffn_down_exps       	  69351.28	  0.0149	    96.7603	  0.13	  0.44	  100.00%	524288	   18.0518	 95.01%	      2.42%	  0.0188
0.01.395.054 I    18	ffn_down_exps       	  67680.16	  0.0069	   215.6972	  0.13	  0.50	  100.00%	524288	   18.2169	 95.88%	      1.26%	  0.0091
0.01.395.056 I    21	ffn_down_exps       	  62098.25	  0.0068	    83.9246	  0.12	  0.29	  100.00%	524288	   18.3104	 96.37%	      2.87%	  0.0494
0.01.395.057 I    24	ffn_down_exps       	  56159.25	  0.0044	    60.2592	  0.11	  0.31	  100.00%	524288	   18.1794	 95.68%	      2.64%	  0.0146
0.01.395.059 I    27	ffn_down_exps       	  47317.23	  0.0120	    49.1935	  0.09	  0.32	  100.00%	524288	   18.2059	 95.82%	      1.53%	  0.0212
0.01.395.060 I    25	ffn_down_exps       	  45894.51	  0.0141	   317.9896	  0.09	  0.57	  100.00%	524288	   18.2478	 96.04%	      0.57%	  0.0186
0.01.395.061 I    23	ffn_down_exps       	  43613.12	  0.0078	   557.6200	  0.08	  1.00	  100.00%	524288	   17.8259	 93.82%	      0.36%	  0.0184
0.01.395.063 I    17	ffn_down_exps       	  38313.35	  0.0057	   123.1870	  0.07	  0.23	  100.00%	524288	   18.4736	 97.23%	      1.24%	  0.0284
0.01.395.064 I    19	ffn_down_exps       	  34140.08	  0.0026	    25.3453	  0.07	  0.15	  100.00%	524288	   18.0750	 95.13%	      3.60%	  0.0337
0.01.395.066 I    16	ffn_down_exps       	  26969.89	  0.0052	    43.9049	  0.05	  0.13	  100.00%	524288	   18.4342	 97.02%	      1.78%	  0.0919
0.01.395.067 I     7	ffn_down_exps       	  21039.93	  0.0040	    16.9352	  0.04	  0.09	  100.00%	524288	   18.2099	 95.84%	      4.02%	  0.0307
0.01.395.069 I     8	ffn_down_exps       	  19691.43	  0.0030	    60.6809	  0.04	  0.21	  100.00%	524288	   17.9929	 94.70%	      1.01%	  0.0338
0.01.395.070 I     6	ffn_down_exps       	  19079.35	  0.0020	    80.1530	  0.04	  0.19	  100.00%	524288	   17.9975	 94.72%	      1.39%	  0.0262
0.01.395.072 I    15	ffn_down_exps       	  18530.90	  0.0043	    16.3784	  0.04	  0.05	  100.00%	524288	   18.5393	 97.58%	      5.38%	  0.1252
0.01.395.073 I    14	ffn_down_exps       	  18457.52	  0.0038	    32.4825	  0.04	  0.07	  100.00%	524288	   18.4495	 97.10%	      2.92%	  0.1021
0.01.395.075 I    12	ffn_down_exps       	  17647.14	  0.0032	    25.4635	  0.03	  0.08	  100.00%	524288	   18.3605	 96.63%	      2.86%	  0.0544
0.01.395.076 I    13	ffn_down_exps       	  17050.14	  0.0043	    85.4791	  0.03	  0.14	  100.00%	524288	   18.3883	 96.78%	      0.90%	  0.0714
0.01.395.077 I     0	ffn_down_exps       	  14612.26	  0.0017	   111.8095	  0.03	  0.39	  100.00%	524288	   16.1380	 84.94%	      0.57%	  0.0000
0.01.395.079 I    11	ffn_down_exps       	  13912.36	  0.0033	   120.1434	  0.03	  0.24	  100.00%	524288	   18.2498	 96.05%	      0.27%	  0.0460
0.01.395.080 I     9	ffn_down_exps       	  12179.78	  0.0025	   186.1999	  0.02	  0.44	  100.00%	524288	   17.7973	 93.67%	      0.08%	  0.0278
0.01.395.082 I    10	ffn_down_exps       	  11200.07	  0.0018	    40.7784	  0.02	  0.09	  100.00%	524288	   18.2278	 95.94%	      1.16%	  0.0437
0.01.395.083 I     5	ffn_down_exps       	  10742.23	  0.0018	    34.6680	  0.02	  0.09	  100.00%	524288	   18.1245	 95.39%	      1.51%	  0.0310
0.01.395.085 I     3	ffn_down_exps       	   8788.01	  0.0007	    22.1987	  0.02	  0.06	  100.00%	524288	   17.9272	 94.35%	      2.39%	  0.0278
0.01.395.086 I     2	ffn_down_exps       	   8732.57	  0.0031	    64.2533	  0.02	  0.12	  100.00%	524288	   17.8960	 94.19%	      0.64%	  0.0337
0.01.395.088 I     4	ffn_down_exps       	   6289.40	  0.0008	    22.9802	  0.01	  0.05	  100.00%	524288	   17.9751	 94.61%	      2.15%	  0.0645
0.01.395.089 I     1	ffn_down_exps       	   6086.41	  0.0026	     6.2940	  0.01	  0.03	  100.00%	524288	   18.4388	 97.05%	      1.50%	  0.0152
0.01.395.091 I    42	ffn_down_shexp      	  34612.52	  0.0160	  1341.1948	 16.90	 93.21	  100.00%	  2048	    7.1877	 65.34%	      2.54%	  0.0310
0.01.395.092 I    41	ffn_down_shexp      	   2724.08	  0.0114	   212.8199	  1.33	 10.03	  100.00%	  2048	    7.2744	 66.13%	      1.51%	  0.0351
0.01.395.094 I    40	ffn_down_shexp      	    827.90	  0.0074	    43.3332	  0.40	  1.79	  100.00%	  2048	    9.0304	 82.09%	      1.86%	  0.0707
0.01.395.095 I    39	ffn_down_shexp      	    786.74	  0.0171	    37.1856	  0.38	  1.40	  100.00%	  2048	    9.5218	 86.56%	      1.51%	  0.0738
0.01.395.097 I    38	ffn_down_shexp      	    350.24	  0.0090	    18.1399	  0.17	  0.58	  100.00%	  2048	    9.6738	 87.94%	      2.39%	  0.0668
0.01.395.098 I    37	ffn_down_shexp      	    331.55	  0.0115	    19.0700	  0.16	  0.69	  100.00%	  2048	    9.0949	 82.68%	      1.95%	  0.0813
0.01.395.099 I    36	ffn_down_shexp      	    316.18	  0.0118	     7.7782	  0.15	  0.35	  100.00%	  2048	    9.9761	 90.69%	      3.37%	  0.2196
0.01.395.101 I    20	ffn_down_shexp      	    234.34	  0.0115	     8.4255	  0.11	  0.33	  100.00%	  2048	    9.4939	 86.31%	      4.00%	  0.1512
0.01.395.102 I     0	ffn_down_shexp      	    230.03	  0.0010	     3.6309	  0.11	  0.17	  100.00%	  2048	   10.2337	 93.03%	      5.08%	  0.0000
0.01.395.103 I    22	ffn_down_shexp      	    229.66	  0.0093	     6.3586	  0.11	  0.26	  100.00%	  2048	    9.7870	 88.97%	      5.27%	  0.1002
0.01.395.105 I    34	ffn_down_shexp      	    211.43	  0.0078	     7.5232	  0.10	  0.34	  100.00%	  2048	    9.3473	 84.98%	      3.52%	  0.0922
0.01.395.106 I    35	ffn_down_shexp      	    181.20	  0.0122	     2.7099	  0.09	  0.13	  100.00%	  2048	   10.3347	 93.95%	      5.66%	  0.1978
0.01.395.107 I    21	ffn_down_shexp      	    171.94	  0.0055	    18.8485	  0.08	  0.62	  100.00%	  2048	    8.0438	 73.13%	      1.07%	  0.0879
0.01.395.109 I    31	ffn_down_shexp      	    171.21	  0.0092	     3.8894	  0.08	  0.20	  100.00%	  2048	    9.8624	 89.66%	      3.81%	  0.1497
0.01.395.110 I    33	ffn_down_shexp      	    146.38	  0.0069	     5.2085	  0.07	  0.22	  100.00%	  2048	    9.5951	 87.23%	      2.44%	  0.1711
0.01.395.111 I    27	ffn_down_shexp      	    142.28	  0.0062	    10.0730	  0.07	  0.35	  100.00%	  2048	    9.0720	 82.47%	      1.51%	  0.0197
0.01.395.113 I    32	ffn_down_shexp      	    140.41	  0.0069	     2.9945	  0.07	  0.13	  100.00%	  2048	   10.1864	 92.60%	      3.81%	  0.2066
0.01.395.114 I    30	ffn_down_shexp      	    139.77	  0.0095	     5.1566	  0.07	  0.19	  100.00%	  2048	    9.9563	 90.51%	      2.29%	  0.1502
0.01.395.115 I    26	ffn_down_shexp      	    128.48	  0.0074	    20.1095	  0.06	  0.50	  100.00%	  2048	    8.5832	 78.03%	      1.03%	  0.1130
0.01.395.117 I    18	ffn_down_shexp      	    116.00	  0.0071	     4.6894	  0.06	  0.16	  100.00%	  2048	    9.6093	 87.36%	      4.00%	  0.1583
0.01.395.118 I    19	ffn_down_shexp      	    106.18	  0.0030	     3.0635	  0.05	  0.11	  100.00%	  2048	   10.0625	 91.48%	      5.42%	  0.1697
0.01.395.120 I    24	ffn_down_shexp      	    100.60	  0.0060	    18.3339	  0.05	  0.41	  100.00%	  2048	    9.0142	 81.95%	      0.44%	  0.0532
0.01.395.121 I    29	ffn_down_shexp      	     98.30	  0.0090	     3.2956	  0.05	  0.12	  100.00%	  2048	   10.0478	 91.34%	      2.59%	  0.2410
0.01.395.122 I    28	ffn_down_shexp      	     94.77	  0.0095	     1.9621	  0.05	  0.07	  100.00%	  2048	   10.4385	 94.90%	      4.25%	  0.1522
0.01.395.124 I    23	ffn_down_shexp      	     92.10	  0.0063	     5.0679	  0.04	  0.13	  100.00%	  2048	   10.1048	 91.86%	      2.64%	  0.1845
0.01.395.125 I    25	ffn_down_shexp      	     70.68	  0.0068	     1.6708	  0.03	  0.07	  100.00%	  2048	   10.2358	 93.05%	      3.61%	  0.0562
0.01.395.126 I    17	ffn_down_shexp      	     64.33	  0.0054	     2.6332	  0.03	  0.09	  100.00%	  2048	    9.6997	 88.18%	      3.76%	  0.1993
0.01.395.128 I     2	ffn_down_shexp      	     60.10	  0.0001	     0.9880	  0.03	  0.05	  100.00%	  2048	    9.7403	 88.55%	      9.77%	  0.3717
0.01.395.129 I    16	ffn_down_shexp      	     44.27	  0.0042	     1.2203	  0.02	  0.05	  100.00%	  2048	   10.0215	 91.10%	      4.64%	  0.1345
0.01.395.130 I    15	ffn_down_shexp      	     40.14	  0.0028	     2.0710	  0.02	  0.08	  100.00%	  2048	    9.5433	 86.76%	      1.76%	  0.0919
0.01.395.132 I     7	ffn_down_shexp      	     38.57	  0.0016	     0.9163	  0.02	  0.03	  100.00%	  2048	   10.3202	 93.82%	      5.37%	  0.3248
0.01.395.133 I    13	ffn_down_shexp      	     34.90	  0.0026	     3.3175	  0.02	  0.09	  100.00%	  2048	    9.2855	 84.41%	      1.12%	  0.0927
0.01.395.134 I     8	ffn_down_shexp      	     34.03	  0.0009	     0.6567	  0.02	  0.03	  100.00%	  2048	   10.3375	 93.98%	      5.76%	  0.3255
0.01.395.136 I    14	ffn_down_shexp      	     32.17	  0.0028	     1.2743	  0.02	  0.04	  100.00%	  2048	    9.9871	 90.79%	      3.22%	  0.0695
0.01.395.137 I    12	ffn_down_shexp      	     30.67	  0.0026	     0.7244	  0.01	  0.03	  100.00%	  2048	   10.1094	 91.90%	      3.56%	  0.2364
0.01.395.138 I    11	ffn_down_shexp      	     25.76	  0.0019	     0.4718	  0.01	  0.02	  100.00%	  2048	   10.1934	 92.67%	      4.39%	  0.3184
0.01.395.139 I     6	ffn_down_shexp      	     25.16	  0.0013	     0.6841	  0.01	  0.02	  100.00%	  2048	   10.3666	 94.24%	      4.83%	  0.2502
0.01.395.141 I     9	ffn_down_shexp      	     23.50	  0.0011	     0.5647	  0.01	  0.03	  100.00%	  2048	   10.0643	 91.49%	      3.37%	  0.2357
0.01.395.142 I    10	ffn_down_shexp      	     21.54	  0.0016	     0.2399	  0.01	  0.01	  100.00%	  2048	   10.3685	 94.26%	      6.20%	  0.2763
0.01.395.143 I     1	ffn_down_shexp      	     21.02	  0.0002	     0.2970	  0.01	  0.01	  100.00%	  2048	   10.3180	 93.80%	      8.35%	  0.3348
0.01.395.145 I     4	ffn_down_shexp      	     15.74	  0.0011	     0.1577	  0.01	  0.01	  100.00%	  2048	   10.4121	 94.66%	      6.69%	  0.4026
0.01.395.146 I     3	ffn_down_shexp      	     15.25	  0.0019	     0.1763	  0.01	  0.01	  100.00%	  2048	   10.5377	 95.80%	      4.10%	  0.3213
0.01.395.148 I     5	ffn_down_shexp      	     14.10	  0.0007	     0.5160	  0.01	  0.01	  100.00%	  2048	   10.2768	 93.43%	      3.76%	  0.2988
0.01.395.149 I    39	ffn_gate_exps       	 564302.81	  0.1316	    15.6649	  0.54	  0.19	  100.00%	1048576	   19.9326	 99.66%	     10.32%	  0.4499
0.01.395.151 I    40	ffn_gate_exps       	 429271.81	  0.0665	    17.6149	  0.41	  0.16	  100.00%	1048576	   19.9181	 99.59%	     10.92%	  0.4318
0.01.395.152 I    36	ffn_gate_exps       	 383794.44	  0.1214	     3.8468	  0.37	  0.10	  100.00%	1048576	   19.9549	 99.77%	     10.81%	  0.4734
0.01.395.154 I    38	ffn_gate_exps       	 374274.75	  0.1055	     5.1450	  0.36	  0.11	  100.00%	1048576	   19.9439	 99.72%	     10.32%	  0.4459
0.01.395.156 I    41	ffn_gate_exps       	 355878.41	  0.0645	     9.3773	  0.34	  0.11	  100.00%	1048576	   19.9343	 99.67%	     11.33%	  0.4562
0.01.395.158 I    37	ffn_gate_exps       	 350681.38	  0.0677	     4.2945	  0.33	  0.11	  100.00%	1048576	   19.9368	 99.68%	     10.21%	  0.4729
0.01.395.160 I    34	ffn_gate_exps       	 349419.84	  0.1110	     4.0321	  0.33	  0.09	  100.00%	1048576	   19.9566	 99.78%	     10.56%	  0.4333
0.01.395.161 I    35	ffn_gate_exps       	 320619.75	  0.0981	     4.6523	  0.31	  0.08	  100.00%	1048576	   19.9586	 99.79%	      9.91%	  0.4773
0.01.395.163 I    42	ffn_gate_exps       	 316240.47	  0.0196	    13.4769	  0.30	  0.14	  100.00%	1048576	   19.8863	 99.43%	     10.12%	  0.4658
0.01.395.165 I    32	ffn_gate_exps       	 302187.94	  0.0889	     5.1394	  0.29	  0.08	  100.00%	1048576	   19.9570	 99.79%	     10.03%	  0.4991
0.01.395.167 I    33	ffn_gate_exps       	 296675.22	  0.0927	     7.1363	  0.28	  0.09	  100.00%	1048576	   19.9527	 99.76%	      8.75%	  0.4660
0.01.395.169 I    30	ffn_gate_exps       	 285929.78	  0.1188	     3.7843	  0.27	  0.07	  100.00%	1048576	   19.9648	 99.82%	     10.66%	  0.5025
0.01.395.171 I    31	ffn_gate_exps       	 278590.72	  0.0779	     4.9173	  0.27	  0.07	  100.00%	1048576	   19.9577	 99.79%	      9.64%	  0.5192
0.01.395.172 I    28	ffn_gate_exps       	 272039.31	  0.0981	     6.7494	  0.26	  0.07	  100.00%	1048576	   19.9649	 99.82%	      9.30%	  0.4305
0.01.395.174 I    29	ffn_gate_exps       	 262653.12	  0.1071	     3.3004	  0.25	  0.06	  100.00%	1048576	   19.9662	 99.83%	     10.34%	  0.3835
0.01.395.176 I    26	ffn_gate_exps       	 247435.94	  0.1134	     3.8431	  0.24	  0.05	  100.00%	1048576	   19.9676	 99.84%	     11.29%	  0.4614
0.01.395.177 I    27	ffn_gate_exps       	 239436.41	  0.0902	     2.7469	  0.23	  0.06	  100.00%	1048576	   19.9645	 99.82%	     10.56%	  0.4097
0.01.395.179 I    20	ffn_gate_exps       	 223345.44	  0.0741	     1.8885	  0.21	  0.07	  100.00%	1048576	   19.9440	 99.72%	     10.94%	  0.4723
0.01.395.180 I    22	ffn_gate_exps       	 214484.31	  0.0699	     2.0111	  0.20	  0.06	  100.00%	1048576	   19.9541	 99.77%	     10.38%	  0.4723
0.01.395.181 I    25	ffn_gate_exps       	 211179.98	  0.1016	     3.3329	  0.20	  0.04	  100.00%	1048576	   19.9722	 99.86%	     10.16%	  0.5228
0.01.395.183 I    21	ffn_gate_exps       	 186570.50	  0.0609	     1.7274	  0.18	  0.05	  100.00%	1048576	   19.9512	 99.76%	     10.58%	  0.5012
0.01.395.184 I    24	ffn_gate_exps       	 184062.50	  0.0842	     1.7379	  0.18	  0.04	  100.00%	1048576	   19.9698	 99.85%	     10.75%	  0.5585
0.01.395.186 I    18	ffn_gate_exps       	 175111.98	  0.0609	     1.6294	  0.17	  0.05	  100.00%	1048576	   19.9495	 99.75%	     10.96%	  0.5132
0.01.395.187 I    23	ffn_gate_exps       	 172542.22	  0.0686	     1.9863	  0.16	  0.04	  100.00%	1048576	   19.9665	 99.83%	      9.60%	  0.5395
0.01.395.189 I    17	ffn_gate_exps       	 142821.53	  0.0660	     1.6086	  0.14	  0.03	  100.00%	1048576	   19.9617	 99.81%	     10.47%	  0.5255
0.01.395.190 I    19	ffn_gate_exps       	 131582.78	  0.0318	     1.2363	  0.13	  0.04	  100.00%	1048576	   19.9497	 99.75%	     11.25%	  0.4994
0.01.395.191 I     7	ffn_gate_exps       	 124106.82	  0.0529	     0.7467	  0.12	  0.03	  100.00%	1048576	   19.9598	 99.80%	     11.54%	  0.3699
0.01.395.193 I     8	ffn_gate_exps       	 117147.20	  0.0494	     0.6752	  0.11	  0.03	  100.00%	1048576	   19.9609	 99.80%	     11.82%	  0.4538
0.01.395.194 I    16	ffn_gate_exps       	 116568.26	  0.0500	     0.9518	  0.11	  0.03	  100.00%	1048576	   19.9619	 99.81%	     10.99%	  0.4664
0.01.395.196 I    15	ffn_gate_exps       	 106448.66	  0.0494	     2.1372	  0.10	  0.03	  100.00%	1048576	   19.9607	 99.80%	     10.08%	  0.4695
0.01.395.197 I    14	ffn_gate_exps       	 104297.16	  0.0458	     0.7072	  0.10	  0.02	  100.00%	1048576	   19.9622	 99.81%	     11.19%	  0.4700
0.01.395.199 I     6	ffn_gate_exps       	 103518.63	  0.0266	     0.7293	  0.10	  0.03	  100.00%	1048576	   19.9526	 99.76%	     10.88%	  0.3884
0.01.395.200 I    12	ffn_gate_exps       	 100729.63	  0.0253	     0.6453	  0.10	  0.02	  100.00%	1048576	   19.9631	 99.82%	     11.68%	  0.4684
0.01.395.202 I     2	ffn_gate_exps       	 100703.21	  0.0367	     0.6439	  0.10	  0.02	  100.00%	1048576	   19.9596	 99.80%	     11.21%	  0.7654
0.01.395.203 I    13	ffn_gate_exps       	 100609.74	  0.0460	     0.6313	  0.10	  0.02	  100.00%	1048576	   19.9641	 99.82%	     11.42%	  0.4525
0.01.395.205 I     9	ffn_gate_exps       	  89197.70	  0.0393	     0.8781	  0.09	  0.02	  100.00%	1048576	   19.9639	 99.82%	     11.09%	  0.4528
0.01.395.206 I    11	ffn_gate_exps       	  87554.77	  0.0421	     0.7821	  0.08	  0.02	  100.00%	1048576	   19.9668	 99.83%	     10.77%	  0.5018
0.01.395.207 I    10	ffn_gate_exps       	  78064.84	  0.0297	     0.8073	  0.07	  0.02	  100.00%	1048576	   19.9555	 99.78%	     11.22%	  0.6000
0.01.395.209 I     5	ffn_gate_exps       	  76448.54	  0.0182	     0.4443	  0.07	  0.02	  100.00%	1048576	   19.9686	 99.84%	     11.62%	  0.4219
0.01.395.210 I     3	ffn_gate_exps       	  70009.18	  0.0134	     0.6840	  0.07	  0.02	  100.00%	1048576	   19.9488	 99.74%	     11.07%	  0.6184
0.01.395.212 I     1	ffn_gate_exps       	  65489.70	  0.0297	     0.3855	  0.06	  0.01	  100.00%	1048576	   19.9762	 99.88%	     11.47%	  0.5548
0.01.395.213 I     4	ffn_gate_exps       	  60911.07	  0.0118	     0.6279	  0.06	  0.02	  100.00%	1048576	   19.9354	 99.68%	     11.42%	  0.4593
0.01.395.214 I     0	ffn_gate_exps       	  54487.66	  0.0179	     5.7763	  0.05	  0.06	  100.00%	1048576	   19.8257	 99.13%	      1.62%	  0.0000
0.01.395.216 I    39	ffn_gate_inp        	   2201.18	  0.3737	     2.5695	  0.54	  0.15	  100.00%	  4096	   11.9517	 99.60%	     10.86%	  0.9894
0.01.395.217 I    40	ffn_gate_inp        	   1674.19	  0.2684	     2.0984	  0.41	  0.13	  100.00%	  4096	   11.9423	 99.52%	     12.38%	  0.9839
0.01.395.219 I    36	ffn_gate_inp        	   1497.15	  0.2629	     1.2691	  0.37	  0.07	  100.00%	  4096	   11.9734	 99.78%	     11.91%	  0.9910
0.01.395.220 I    38	ffn_gate_inp        	   1460.05	  0.2514	     1.4725	  0.36	  0.09	  100.00%	  4096	   11.9641	 99.70%	     11.04%	  0.9917
0.01.395.222 I    41	ffn_gate_inp        	   1388.13	  0.1858	     1.4241	  0.34	  0.09	  100.00%	  4096	   11.9555	 99.63%	     12.21%	  0.9809
0.01.395.223 I    37	ffn_gate_inp        	   1368.03	  0.2311	     1.6520	  0.33	  0.08	  100.00%	  4096	   11.9613	 99.68%	     11.47%	  0.9908
0.01.395.225 I    34	ffn_gate_inp        	   1363.18	  0.2445	     1.6408	  0.33	  0.07	  100.00%	  4096	   11.9751	 99.79%	     11.94%	  0.9920
0.01.395.226 I    35	ffn_gate_inp        	   1250.79	  0.2286	     1.4944	  0.31	  0.06	  100.00%	  4096	   11.9762	 99.80%	     10.96%	  0.9930
0.01.395.227 I    42	ffn_gate_inp        	   1230.70	  0.1566	     1.9333	  0.30	  0.11	  100.00%	  4096	   11.9195	 99.33%	     10.62%	  0.9463
0.01.395.229 I    32	ffn_gate_inp        	   1178.87	  0.2104	     1.4438	  0.29	  0.06	  100.00%	  4096	   11.9767	 99.81%	     10.94%	  0.9947
0.01.395.230 I    33	ffn_gate_inp        	   1157.67	  0.2051	     2.2589	  0.28	  0.07	  100.00%	  4096	   11.9725	 99.77%	      8.84%	  0.9961
0.01.395.232 I    30	ffn_gate_inp        	   1115.81	  0.2130	     0.9576	  0.27	  0.05	  100.00%	  4096	   11.9815	 99.85%	     11.08%	  0.9983
0.01.395.233 I    31	ffn_gate_inp        	   1086.89	  0.2003	     1.4724	  0.27	  0.06	  100.00%	  4096	   11.9750	 99.79%	     10.06%	  0.9951
0.01.395.234 I    28	ffn_gate_inp        	   1061.43	  0.1990	     1.1912	  0.26	  0.04	  100.00%	  4096	   11.9835	 99.86%	     10.50%	  0.9953
0.01.395.236 I    29	ffn_gate_inp        	   1024.82	  0.1917	     0.9705	  0.25	  0.04	  100.00%	  4096	   11.9820	 99.85%	     11.13%	  0.9971
0.01.395.237 I    26	ffn_gate_inp        	    965.37	  0.1770	     0.5396	  0.24	  0.04	  100.00%	  4096	   11.9840	 99.87%	     12.52%	  0.9930
0.01.395.239 I    27	ffn_gate_inp        	    934.23	  0.1748	     0.6325	  0.23	  0.04	  100.00%	  4096	   11.9811	 99.84%	     11.77%	  0.9959
0.01.395.240 I    20	ffn_gate_inp        	    870.99	  0.1508	     0.9396	  0.21	  0.05	  100.00%	  4096	   11.9675	 99.73%	     11.11%	  0.9760
0.01.395.241 I    22	ffn_gate_inp        	    835.82	  0.1461	     0.8002	  0.20	  0.04	  100.00%	  4096	   11.9721	 99.77%	     10.25%	  0.9930
0.01.395.243 I    25	ffn_gate_inp        	    823.88	  0.1590	     0.7985	  0.20	  0.03	  100.00%	  4096	   11.9855	 99.88%	     10.94%	  0.9973
0.01.395.244 I    21	ffn_gate_inp        	    727.50	  0.1236	     0.8955	  0.18	  0.04	  100.00%	  4096	   11.9685	 99.74%	     10.40%	  0.9846
0.01.395.246 I    24	ffn_gate_inp        	    718.04	  0.1331	     0.5412	  0.18	  0.03	  100.00%	  4096	   11.9847	 99.87%	     11.40%	  0.9963
0.01.395.247 I    18	ffn_gate_inp        	    682.98	  0.1168	     0.6745	  0.17	  0.04	  100.00%	  4096	   11.9694	 99.75%	     10.77%	  0.9876
0.01.395.248 I    23	ffn_gate_inp        	    672.96	  0.1242	     0.6819	  0.16	  0.03	  100.00%	  4096	   11.9815	 99.85%	     10.23%	  0.9913
0.01.395.250 I    17	ffn_gate_inp        	    557.13	  0.1029	     0.6520	  0.14	  0.03	  100.00%	  4096	   11.9773	 99.81%	     10.18%	  0.9920
0.01.395.251 I    19	ffn_gate_inp        	    512.79	  0.0887	     0.4220	  0.13	  0.03	  100.00%	  4096	   11.9721	 99.77%	     11.99%	  0.9779
0.01.395.253 I     7	ffn_gate_inp        	    484.26	  0.0851	     0.3598	  0.12	  0.02	  100.00%	  4096	   11.9749	 99.79%	     11.77%	  0.9894
0.01.395.254 I     8	ffn_gate_inp        	    457.19	  0.0818	     0.3399	  0.11	  0.02	  100.00%	  4096	   11.9759	 99.80%	     12.72%	  0.9893
0.01.395.255 I    16	ffn_gate_inp        	    454.66	  0.0825	     0.3976	  0.11	  0.02	  100.00%	  4096	   11.9792	 99.83%	     11.28%	  0.9900
0.01.395.257 I    15	ffn_gate_inp        	    415.26	  0.0754	     0.6388	  0.10	  0.02	  100.00%	  4096	   11.9755	 99.80%	      9.59%	  0.9877
0.01.395.258 I    14	ffn_gate_inp        	    406.91	  0.0746	     0.3592	  0.10	  0.02	  100.00%	  4096	   11.9784	 99.82%	     11.52%	  0.9920
0.01.395.259 I     6	ffn_gate_inp        	    403.81	  0.0711	     0.3994	  0.10	  0.02	  100.00%	  4096	   11.9696	 99.75%	     11.06%	  0.9876
0.01.395.261 I     2	ffn_gate_inp        	    392.69	  0.0676	     0.3027	  0.10	  0.02	  100.00%	  4096	   11.9735	 99.78%	     10.99%	  0.9815
0.01.395.262 I    12	ffn_gate_inp        	    392.67	  0.0504	     0.3115	  0.10	  0.02	  100.00%	  4096	   11.9774	 99.81%	     11.52%	  0.9904
0.01.395.264 I    13	ffn_gate_inp        	    392.34	  0.0699	     0.3270	  0.10	  0.02	  100.00%	  4096	   11.9782	 99.82%	     11.38%	  0.9926
0.01.395.265 I     9	ffn_gate_inp        	    348.20	  0.0633	     0.4913	  0.09	  0.02	  100.00%	  4096	   11.9783	 99.82%	     11.18%	  0.9901
0.01.395.266 I    11	ffn_gate_inp        	    341.50	  0.0611	     0.4904	  0.08	  0.02	  100.00%	  4096	   11.9803	 99.84%	     10.40%	  0.9908
0.01.395.268 I    10	ffn_gate_inp        	    304.42	  0.0523	     0.2139	  0.07	  0.02	  100.00%	  4096	   11.9738	 99.78%	     11.47%	  0.9907
0.01.395.269 I     5	ffn_gate_inp        	    298.24	  0.0544	     0.2072	  0.07	  0.01	  100.00%	  4096	   11.9823	 99.85%	     12.16%	  0.9818
0.01.395.271 I     3	ffn_gate_inp        	    273.21	  0.0468	     0.2274	  0.07	  0.01	  100.00%	  4096	   11.9706	 99.76%	     11.16%	  0.9831
0.01.395.272 I     1	ffn_gate_inp        	    255.36	  0.0483	     0.1469	  0.06	  0.01	  100.00%	  4096	   11.9893	 99.91%	     12.01%	  0.6957
0.01.395.273 I     4	ffn_gate_inp        	    237.58	  0.0398	     0.2100	  0.06	  0.01	  100.00%	  4096	   11.9595	 99.66%	     11.65%	  0.9813
0.01.395.275 I     0	ffn_gate_inp        	    212.36	  0.0313	     2.6984	  0.05	  0.05	  100.00%	  4096	   11.8510	 98.76%	      1.39%	  0.0000
0.01.395.276 I    39	ffn_gate_shexp      	   2201.18	  0.3737	     2.5695	  0.54	  0.15	  100.00%	  4096	   11.9517	 99.60%	     10.86%	  0.9894
0.01.395.277 I    40	ffn_gate_shexp      	   1674.19	  0.2684	     2.0984	  0.41	  0.13	  100.00%	  4096	   11.9423	 99.52%	     12.38%	  0.9839
0.01.395.279 I    36	ffn_gate_shexp      	   1497.15	  0.2629	     1.2691	  0.37	  0.07	  100.00%	  4096	   11.9734	 99.78%	     11.91%	  0.9910
0.01.395.280 I    38	ffn_gate_shexp      	   1460.05	  0.2514	     1.4725	  0.36	  0.09	  100.00%	  4096	   11.9641	 99.70%	     11.04%	  0.9917
0.01.395.281 I    41	ffn_gate_shexp      	   1388.13	  0.1858	     1.4241	  0.34	  0.09	  100.00%	  4096	   11.9555	 99.63%	     12.21%	  0.9809
0.01.395.283 I    37	ffn_gate_shexp      	   1368.03	  0.2311	     1.6520	  0.33	  0.08	  100.00%	  4096	   11.9613	 99.68%	     11.47%	  0.9908
0.01.395.284 I    34	ffn_gate_shexp      	   1363.18	  0.2445	     1.6408	  0.33	  0.07	  100.00%	  4096	   11.9751	 99.79%	     11.94%	  0.9920
0.01.395.286 I    35	ffn_gate_shexp      	   1250.79	  0.2286	     1.4944	  0.31	  0.06	  100.00%	  4096	   11.9762	 99.80%	     10.96%	  0.9930
0.01.395.287 I    42	ffn_gate_shexp      	   1230.70	  0.1566	     1.9333	  0.30	  0.11	  100.00%	  4096	   11.9195	 99.33%	     10.62%	  0.9463
0.01.395.288 I    32	ffn_gate_shexp      	   1178.87	  0.2104	     1.4438	  0.29	  0.06	  100.00%	  4096	   11.9767	 99.81%	     10.94%	  0.9947
0.01.395.289 I    33	ffn_gate_shexp      	   1157.67	  0.2051	     2.2589	  0.28	  0.07	  100.00%	  4096	   11.9725	 99.77%	      8.84%	  0.9961
0.01.395.291 I    30	ffn_gate_shexp      	   1115.81	  0.2130	     0.9576	  0.27	  0.05	  100.00%	  4096	   11.9815	 99.85%	     11.08%	  0.9983
0.01.395.292 I    31	ffn_gate_shexp      	   1086.89	  0.2003	     1.4724	  0.27	  0.06	  100.00%	  4096	   11.9750	 99.79%	     10.06%	  0.9951
0.01.395.293 I    28	ffn_gate_shexp      	   1061.43	  0.1990	     1.1912	  0.26	  0.04	  100.00%	  4096	   11.9835	 99.86%	     10.50%	  0.9953
0.01.395.295 I    29	ffn_gate_shexp      	   1024.82	  0.1917	     0.9705	  0.25	  0.04	  100.00%	  4096	   11.9820	 99.85%	     11.13%	  0.9971
0.01.395.296 I    26	ffn_gate_shexp      	    965.37	  0.1770	     0.5396	  0.24	  0.04	  100.00%	  4096	   11.9840	 99.87%	     12.52%	  0.9930
0.01.395.297 I    27	ffn_gate_shexp      	    934.23	  0.1748	     0.6325	  0.23	  0.04	  100.00%	  4096	   11.9811	 99.84%	     11.77%	  0.9959
0.01.395.298 I    20	ffn_gate_shexp      	    870.99	  0.1508	     0.9396	  0.21	  0.05	  100.00%	  4096	   11.9675	 99.73%	     11.11%	  0.9760
0.01.395.300 I    22	ffn_gate_shexp      	    835.82	  0.1461	     0.8002	  0.20	  0.04	  100.00%	  4096	   11.9721	 99.77%	     10.25%	  0.9930
0.01.395.301 I    25	ffn_gate_shexp      	    823.88	  0.1590	     0.7985	  0.20	  0.03	  100.00%	  4096	   11.9855	 99.88%	     10.94%	  0.9973
0.01.395.303 I    21	ffn_gate_shexp      	    727.50	  0.1236	     0.8955	  0.18	  0.04	  100.00%	  4096	   11.9685	 99.74%	     10.40%	  0.9846
0.01.395.304 I    24	ffn_gate_shexp      	    718.04	  0.1331	     0.5412	  0.18	  0.03	  100.00%	  4096	   11.9847	 99.87%	     11.40%	  0.9963
0.01.395.305 I    18	ffn_gate_shexp      	    682.98	  0.1168	     0.6745	  0.17	  0.04	  100.00%	  4096	   11.9694	 99.75%	     10.77%	  0.9876
0.01.395.306 I    23	ffn_gate_shexp      	    672.96	  0.1242	     0.6819	  0.16	  0.03	  100.00%	  4096	   11.9815	 99.85%	     10.23%	  0.9913
0.01.395.308 I    17	ffn_gate_shexp      	    557.13	  0.1029	     0.6520	  0.14	  0.03	  100.00%	  4096	   11.9773	 99.81%	     10.18%	  0.9920
0.01.395.309 I    19	ffn_gate_shexp      	    512.79	  0.0887	     0.4220	  0.13	  0.03	  100.00%	  4096	   11.9721	 99.77%	     11.99%	  0.9779
0.01.395.310 I     7	ffn_gate_shexp      	    484.26	  0.0851	     0.3598	  0.12	  0.02	  100.00%	  4096	   11.9749	 99.79%	     11.77%	  0.9894
0.01.395.312 I     8	ffn_gate_shexp      	    457.19	  0.0818	     0.3399	  0.11	  0.02	  100.00%	  4096	   11.9759	 99.80%	     12.72%	  0.9893
0.01.395.313 I    16	ffn_gate_shexp      	    454.66	  0.0825	     0.3976	  0.11	  0.02	  100.00%	  4096	   11.9792	 99.83%	     11.28%	  0.9900
0.01.395.314 I    15	ffn_gate_shexp      	    415.26	  0.0754	     0.6388	  0.10	  0.02	  100.00%	  4096	   11.9755	 99.80%	      9.59%	  0.9877
0.01.395.316 I    14	ffn_gate_shexp      	    406.91	  0.0746	     0.3592	  0.10	  0.02	  100.00%	  4096	   11.9784	 99.82%	     11.52%	  0.9920
0.01.395.317 I     6	ffn_gate_shexp      	    403.81	  0.0711	     0.3994	  0.10	  0.02	  100.00%	  4096	   11.9696	 99.75%	     11.06%	  0.9876
0.01.395.318 I     2	ffn_gate_shexp      	    392.69	  0.0676	     0.3027	  0.10	  0.02	  100.00%	  4096	   11.9735	 99.78%	     10.99%	  0.9815
0.01.395.319 I    12	ffn_gate_shexp      	    392.67	  0.0504	     0.3115	  0.10	  0.02	  100.00%	  4096	   11.9774	 99.81%	     11.52%	  0.9904
0.01.395.321 I    13	ffn_gate_shexp      	    392.34	  0.0699	     0.3270	  0.10	  0.02	  100.00%	  4096	   11.9782	 99.82%	     11.38%	  0.9926
0.01.395.322 I     9	ffn_gate_shexp      	    348.20	  0.0633	     0.4913	  0.09	  0.02	  100.00%	  4096	   11.9783	 99.82%	     11.18%	  0.9901
0.01.395.323 I    11	ffn_gate_shexp      	    341.50	  0.0611	     0.4904	  0.08	  0.02	  100.00%	  4096	   11.9803	 99.84%	     10.40%	  0.9908
0.01.395.325 I    10	ffn_gate_shexp      	    304.42	  0.0523	     0.2139	  0.07	  0.02	  100.00%	  4096	   11.9738	 99.78%	     11.47%	  0.9907
0.01.395.326 I     5	ffn_gate_shexp      	    298.24	  0.0544	     0.2072	  0.07	  0.01	  100.00%	  4096	   11.9823	 99.85%	     12.16%	  0.9818
0.01.395.327 I     3	ffn_gate_shexp      	    273.21	  0.0468	     0.2274	  0.07	  0.01	  100.00%	  4096	   11.9706	 99.76%	     11.16%	  0.9831
0.01.395.328 I     1	ffn_gate_shexp      	    255.36	  0.0483	     0.1469	  0.06	  0.01	  100.00%	  4096	   11.9893	 99.91%	     12.01%	  0.6957
0.01.395.330 I     4	ffn_gate_shexp      	    237.58	  0.0398	     0.2100	  0.06	  0.01	  100.00%	  4096	   11.9595	 99.66%	     11.65%	  0.9813
0.01.395.331 I     0	ffn_gate_shexp      	    212.36	  0.0313	     2.6984	  0.05	  0.05	  100.00%	  4096	   11.8510	 98.76%	      1.39%	  0.0000
0.01.395.332 I    39	ffn_up_exps         	 564302.81	  0.1316	    15.6649	  0.54	  0.19	  100.00%	1048576	   19.9326	 99.66%	     10.32%	  0.4499
0.01.395.334 I    40	ffn_up_exps         	 429271.81	  0.0665	    17.6149	  0.41	  0.16	  100.00%	1048576	   19.9181	 99.59%	     10.92%	  0.4318
0.01.395.335 I    36	ffn_up_exps         	 383794.44	  0.1214	     3.8468	  0.37	  0.10	  100.00%	1048576	   19.9549	 99.77%	     10.81%	  0.4734
0.01.395.337 I    38	ffn_up_exps         	 374274.75	  0.1055	     5.1450	  0.36	  0.11	  100.00%	1048576	   19.9439	 99.72%	     10.32%	  0.4459
0.01.395.338 I    41	ffn_up_exps         	 355878.41	  0.0645	     9.3773	  0.34	  0.11	  100.00%	1048576	   19.9343	 99.67%	     11.33%	  0.4562
0.01.395.339 I    37	ffn_up_exps         	 350681.38	  0.0677	     4.2945	  0.33	  0.11	  100.00%	1048576	   19.9368	 99.68%	     10.21%	  0.4729
0.01.395.341 I    34	ffn_up_exps         	 349419.84	  0.1110	     4.0321	  0.33	  0.09	  100.00%	1048576	   19.9566	 99.78%	     10.56%	  0.4333
0.01.395.342 I    35	ffn_up_exps         	 320619.75	  0.0981	     4.6523	  0.31	  0.08	  100.00%	1048576	   19.9586	 99.79%	      9.91%	  0.4773
0.01.395.344 I    42	ffn_up_exps         	 316240.47	  0.0196	    13.4769	  0.30	  0.14	  100.00%	1048576	   19.8863	 99.43%	     10.12%	  0.4658
0.01.395.345 I    32	ffn_up_exps         	 302187.94	  0.0889	     5.1394	  0.29	  0.08	  100.00%	1048576	   19.9570	 99.79%	     10.03%	  0.4991
0.01.395.346 I    33	ffn_up_exps         	 296675.22	  0.0927	     7.1363	  0.28	  0.09	  100.00%	1048576	   19.9527	 99.76%	      8.75%	  0.4660
0.01.395.348 I    30	ffn_up_exps         	 285929.78	  0.1188	     3.7843	  0.27	  0.07	  100.00%	1048576	   19.9648	 99.82%	     10.66%	  0.5025
0.01.395.349 I    31	ffn_up_exps         	 278590.72	  0.0779	     4.9173	  0.27	  0.07	  100.00%	1048576	   19.9577	 99.79%	      9.64%	  0.5192
0.01.395.350 I    28	ffn_up_exps         	 272039.31	  0.0981	     6.7494	  0.26	  0.07	  100.00%	1048576	   19.9649	 99.82%	      9.30%	  0.4305
0.01.395.352 I    29	ffn_up_exps         	 262653.12	  0.1071	     3.3004	  0.25	  0.06	  100.00%	1048576	   19.9662	 99.83%	     10.34%	  0.3835
0.01.395.353 I    26	ffn_up_exps         	 247435.94	  0.1134	     3.8431	  0.24	  0.05	  100.00%	1048576	   19.9676	 99.84%	     11.29%	  0.4614
0.01.395.355 I    27	ffn_up_exps         	 239436.41	  0.0902	     2.7469	  0.23	  0.06	  100.00%	1048576	   19.9645	 99.82%	     10.56%	  0.4097
0.01.395.356 I    20	ffn_up_exps         	 223345.44	  0.0741	     1.8885	  0.21	  0.07	  100.00%	1048576	   19.9440	 99.72%	     10.94%	  0.4723
0.01.395.357 I    22	ffn_up_exps         	 214484.31	  0.0699	     2.0111	  0.20	  0.06	  100.00%	1048576	   19.9541	 99.77%	     10.38%	  0.4723
0.01.395.359 I    25	ffn_up_exps         	 211179.98	  0.1016	     3.3329	  0.20	  0.04	  100.00%	1048576	   19.9722	 99.86%	     10.16%	  0.5228
0.01.395.360 I    21	ffn_up_exps         	 186570.50	  0.0609	     1.7274	  0.18	  0.05	  100.00%	1048576	   19.9512	 99.76%	     10.58%	  0.5012
0.01.395.361 I    24	ffn_up_exps         	 184062.50	  0.0842	     1.7379	  0.18	  0.04	  100.00%	1048576	   19.9698	 99.85%	     10.75%	  0.5585
0.01.395.363 I    18	ffn_up_exps         	 175111.98	  0.0609	     1.6294	  0.17	  0.05	  100.00%	1048576	   19.9495	 99.75%	     10.96%	  0.5132
0.01.395.364 I    23	ffn_up_exps         	 172542.22	  0.0686	     1.9863	  0.16	  0.04	  100.00%	1048576	   19.9665	 99.83%	      9.60%	  0.5395
0.01.395.365 I    17	ffn_up_exps         	 142821.53	  0.0660	     1.6086	  0.14	  0.03	  100.00%	1048576	   19.9617	 99.81%	     10.47%	  0.5255
0.01.395.367 I    19	ffn_up_exps         	 131582.78	  0.0318	     1.2363	  0.13	  0.04	  100.00%	1048576	   19.9497	 99.75%	     11.25%	  0.4994
0.01.395.368 I     7	ffn_up_exps         	 124106.82	  0.0529	     0.7467	  0.12	  0.03	  100.00%	1048576	   19.9598	 99.80%	     11.54%	  0.3699
0.01.395.370 I     8	ffn_up_exps         	 117147.20	  0.0494	     0.6752	  0.11	  0.03	  100.00%	1048576	   19.9609	 99.80%	     11.82%	  0.4538
0.01.395.371 I    16	ffn_up_exps         	 116568.26	  0.0500	     0.9518	  0.11	  0.03	  100.00%	1048576	   19.9619	 99.81%	     10.99%	  0.4664
0.01.395.372 I    15	ffn_up_exps         	 106448.66	  0.0494	     2.1372	  0.10	  0.03	  100.00%	1048576	   19.9607	 99.80%	     10.08%	  0.4695
0.01.395.374 I    14	ffn_up_exps         	 104297.16	  0.0458	     0.7072	  0.10	  0.02	  100.00%	1048576	   19.9622	 99.81%	     11.19%	  0.4700
0.01.395.375 I     6	ffn_up_exps         	 103518.63	  0.0266	     0.7293	  0.10	  0.03	  100.00%	1048576	   19.9526	 99.76%	     10.88%	  0.3884
0.01.395.376 I    12	ffn_up_exps         	 100729.63	  0.0253	     0.6453	  0.10	  0.02	  100.00%	1048576	   19.9631	 99.82%	     11.68%	  0.4684
0.01.395.378 I     2	ffn_up_exps         	 100703.21	  0.0367	     0.6439	  0.10	  0.02	  100.00%	1048576	   19.9596	 99.80%	     11.21%	  0.7654
0.01.395.379 I    13	ffn_up_exps         	 100609.74	  0.0460	     0.6313	  0.10	  0.02	  100.00%	1048576	   19.9641	 99.82%	     11.42%	  0.4525
0.01.395.380 I     9	ffn_up_exps         	  89197.70	  0.0393	     0.8781	  0.09	  0.02	  100.00%	1048576	   19.9639	 99.82%	     11.09%	  0.4528
0.01.395.382 I    11	ffn_up_exps         	  87554.77	  0.0421	     0.7821	  0.08	  0.02	  100.00%	1048576	   19.9668	 99.83%	     10.77%	  0.5018
0.01.395.383 I    10	ffn_up_exps         	  78064.84	  0.0297	     0.8073	  0.07	  0.02	  100.00%	1048576	   19.9555	 99.78%	     11.22%	  0.6000
0.01.395.385 I     5	ffn_up_exps         	  76448.54	  0.0182	     0.4443	  0.07	  0.02	  100.00%	1048576	   19.9686	 99.84%	     11.62%	  0.4219
0.01.395.386 I     3	ffn_up_exps         	  70009.18	  0.0134	     0.6840	  0.07	  0.02	  100.00%	1048576	   19.9488	 99.74%	     11.07%	  0.6184
0.01.395.387 I     1	ffn_up_exps         	  65489.70	  0.0297	     0.3855	  0.06	  0.01	  100.00%	1048576	   19.9762	 99.88%	     11.47%	  0.5548
0.01.395.389 I     4	ffn_up_exps         	  60911.07	  0.0118	     0.6279	  0.06	  0.02	  100.00%	1048576	   19.9354	 99.68%	     11.42%	  0.4593
0.01.395.390 I     0	ffn_up_exps         	  54487.66	  0.0179	     5.7763	  0.05	  0.06	  100.00%	1048576	   19.8257	 99.13%	      1.62%	  0.0000
0.01.395.391 I    39	ffn_up_shexp        	   2201.18	  0.3737	     2.5695	  0.54	  0.15	  100.00%	  4096	   11.9517	 99.60%	     10.86%	  0.9894
0.01.395.393 I    40	ffn_up_shexp        	   1674.19	  0.2684	     2.0984	  0.41	  0.13	  100.00%	  4096	   11.9423	 99.52%	     12.38%	  0.9839
0.01.395.394 I    36	ffn_up_shexp        	   1497.15	  0.2629	     1.2691	  0.37	  0.07	  100.00%	  4096	   11.9734	 99.78%	     11.91%	  0.9910
0.01.395.395 I    38	ffn_up_shexp        	   1460.05	  0.2514	     1.4725	  0.36	  0.09	  100.00%	  4096	   11.9641	 99.70%	     11.04%	  0.9917
0.01.395.397 I    41	ffn_up_shexp        	   1388.13	  0.1858	     1.4241	  0.34	  0.09	  100.00%	  4096	   11.9555	 99.63%	     12.21%	  0.9809
0.01.395.398 I    37	ffn_up_shexp        	   1368.03	  0.2311	     1.6520	  0.33	  0.08	  100.00%	  4096	   11.9613	 99.68%	     11.47%	  0.9908
0.01.395.399 I    34	ffn_up_shexp        	   1363.18	  0.2445	     1.6408	  0.33	  0.07	  100.00%	  4096	   11.9751	 99.79%	     11.94%	  0.9920
0.01.395.401 I    35	ffn_up_shexp        	   1250.79	  0.2286	     1.4944	  0.31	  0.06	  100.00%	  4096	   11.9762	 99.80%	     10.96%	  0.9930
0.01.395.402 I    42	ffn_up_shexp        	   1230.70	  0.1566	     1.9333	  0.30	  0.11	  100.00%	  4096	   11.9195	 99.33%	     10.62%	  0.9463
0.01.395.403 I    32	ffn_up_shexp        	   1178.87	  0.2104	     1.4438	  0.29	  0.06	  100.00%	  4096	   11.9767	 99.81%	     10.94%	  0.9947
0.01.395.404 I    33	ffn_up_shexp        	   1157.67	  0.2051	     2.2589	  0.28	  0.07	  100.00%	  4096	   11.9725	 99.77%	      8.84%	  0.9961
0.01.395.406 I    30	ffn_up_shexp        	   1115.81	  0.2130	     0.9576	  0.27	  0.05	  100.00%	  4096	   11.9815	 99.85%	     11.08%	  0.9983
0.01.395.407 I    31	ffn_up_shexp        	   1086.89	  0.2003	     1.4724	  0.27	  0.06	  100.00%	  4096	   11.9750	 99.79%	     10.06%	  0.9951
0.01.395.408 I    28	ffn_up_shexp        	   1061.43	  0.1990	     1.1912	  0.26	  0.04	  100.00%	  4096	   11.9835	 99.86%	     10.50%	  0.9953
0.01.395.409 I    29	ffn_up_shexp        	   1024.82	  0.1917	     0.9705	  0.25	  0.04	  100.00%	  4096	   11.9820	 99.85%	     11.13%	  0.9971
0.01.395.411 I    26	ffn_up_shexp        	    965.37	  0.1770	     0.5396	  0.24	  0.04	  100.00%	  4096	   11.9840	 99.87%	     12.52%	  0.9930
0.01.395.412 I    27	ffn_up_shexp        	    934.23	  0.1748	     0.6325	  0.23	  0.04	  100.00%	  4096	   11.9811	 99.84%	     11.77%	  0.9959
0.01.395.413 I    20	ffn_up_shexp        	    870.99	  0.1508	     0.9396	  0.21	  0.05	  100.00%	  4096	   11.9675	 99.73%	     11.11%	  0.9760
0.01.395.414 I    22	ffn_up_shexp        	    835.82	  0.1461	     0.8002	  0.20	  0.04	  100.00%	  4096	   11.9721	 99.77%	     10.25%	  0.9930
0.01.395.416 I    25	ffn_up_shexp        	    823.88	  0.1590	     0.7985	  0.20	  0.03	  100.00%	  4096	   11.9855	 99.88%	     10.94%	  0.9973
0.01.395.417 I    21	ffn_up_shexp        	    727.50	  0.1236	     0.8955	  0.18	  0.04	  100.00%	  4096	   11.9685	 99.74%	     10.40%	  0.9846
0.01.395.418 I    24	ffn_up_shexp        	    718.04	  0.1331	     0.5412	  0.18	  0.03	  100.00%	  4096	   11.9847	 99.87%	     11.40%	  0.9963
0.01.395.420 I    18	ffn_up_shexp        	    682.98	  0.1168	     0.6745	  0.17	  0.04	  100.00%	  4096	   11.9694	 99.75%	     10.77%	  0.9876
0.01.395.421 I    23	ffn_up_shexp        	    672.96	  0.1242	     0.6819	  0.16	  0.03	  100.00%	  4096	   11.9815	 99.85%	     10.23%	  0.9913
0.01.395.422 I    17	ffn_up_shexp        	    557.13	  0.1029	     0.6520	  0.14	  0.03	  100.00%	  4096	   11.9773	 99.81%	     10.18%	  0.9920
0.01.395.424 I    19	ffn_up_shexp        	    512.79	  0.0887	     0.4220	  0.13	  0.03	  100.00%	  4096	   11.9721	 99.77%	     11.99%	  0.9779
0.01.395.425 I     7	ffn_up_shexp        	    484.26	  0.0851	     0.3598	  0.12	  0.02	  100.00%	  4096	   11.9749	 99.79%	     11.77%	  0.9894
0.01.395.426 I     8	ffn_up_shexp        	    457.19	  0.0818	     0.3399	  0.11	  0.02	  100.00%	  4096	   11.9759	 99.80%	     12.72%	  0.9893
0.01.395.428 I    16	ffn_up_shexp        	    454.66	  0.0825	     0.3976	  0.11	  0.02	  100.00%	  4096	   11.9792	 99.83%	     11.28%	  0.9900
0.01.395.429 I    15	ffn_up_shexp        	    415.26	  0.0754	     0.6388	  0.10	  0.02	  100.00%	  4096	   11.9755	 99.80%	      9.59%	  0.9877
0.01.395.430 I    14	ffn_up_shexp        	    406.91	  0.0746	     0.3592	  0.10	  0.02	  100.00%	  4096	   11.9784	 99.82%	     11.52%	  0.9920
0.01.395.431 I     6	ffn_up_shexp        	    403.81	  0.0711	     0.3994	  0.10	  0.02	  100.00%	  4096	   11.9696	 99.75%	     11.06%	  0.9876
0.01.395.433 I     2	ffn_up_shexp        	    392.69	  0.0676	     0.3027	  0.10	  0.02	  100.00%	  4096	   11.9735	 99.78%	     10.99%	  0.9815
0.01.395.434 I    12	ffn_up_shexp        	    392.67	  0.0504	     0.3115	  0.10	  0.02	  100.00%	  4096	   11.9774	 99.81%	     11.52%	  0.9904
0.01.395.435 I    13	ffn_up_shexp        	    392.34	  0.0699	     0.3270	  0.10	  0.02	  100.00%	  4096	   11.9782	 99.82%	     11.38%	  0.9926
0.01.395.437 I     9	ffn_up_shexp        	    348.20	  0.0633	     0.4913	  0.09	  0.02	  100.00%	  4096	   11.9783	 99.82%	     11.18%	  0.9901
0.01.395.438 I    11	ffn_up_shexp        	    341.50	  0.0611	     0.4904	  0.08	  0.02	  100.00%	  4096	   11.9803	 99.84%	     10.40%	  0.9908
0.01.396.565 I    10	ffn_up_shexp        	    304.42	  0.0523	     0.2139	  0.07	  0.02	  100.00%	  4096	   11.9738	 99.78%	     11.47%	  0.9907
0.01.396.571 I     5	ffn_up_shexp        	    298.24	  0.0544	     0.2072	  0.07	  0.01	  100.00%	  4096	   11.9823	 99.85%	     12.16%	  0.9818
0.01.396.573 I     3	ffn_up_shexp        	    273.21	  0.0468	     0.2274	  0.07	  0.01	  100.00%	  4096	   11.9706	 99.76%	     11.16%	  0.9831
0.01.396.575 I     1	ffn_up_shexp        	    255.36	  0.0483	     0.1469	  0.06	  0.01	  100.00%	  4096	   11.9893	 99.91%	     12.01%	  0.6957
0.01.396.576 I     4	ffn_up_shexp        	    237.58	  0.0398	     0.2100	  0.06	  0.01	  100.00%	  4096	   11.9595	 99.66%	     11.65%	  0.9813
0.01.396.578 I     0	ffn_up_shexp        	    212.36	  0.0313	     2.6984	  0.05	  0.05	  100.00%	  4096	   11.8510	 98.76%	      1.39%	  0.0000
0.01.396.580 I    37	hc_attn_fn          	  16363.59	  0.0544	    54.7747	  1.00	  1.28	  100.00%	 16384	   13.2558	 94.68%	     11.82%	  0.7985
0.01.396.582 I    39	hc_attn_fn          	  16363.50	  0.0715	    69.1367	  1.00	  1.27	  100.00%	 16384	   13.3153	 95.11%	     10.52%	  0.9704
0.01.396.584 I    38	hc_attn_fn          	  16363.45	  0.0659	    46.0165	  1.00	  1.27	  100.00%	 16384	   13.2760	 94.83%	     11.94%	  0.9906
0.01.396.585 I    26	hc_attn_fn          	  16363.38	  0.0391	   126.1928	  1.00	  1.74	  100.00%	 16384	   13.1744	 94.10%	      5.23%	  0.8613
0.01.396.587 I    21	hc_attn_fn          	  16363.25	  0.0504	    80.9469	  1.00	  1.85	  100.00%	 16384	   12.3829	 88.45%	     22.20%	  0.8395
0.01.396.588 I    32	hc_attn_fn          	  16363.24	  0.0450	    98.5872	  1.00	  1.33	  100.00%	 16384	   13.3645	 95.46%	      7.05%	  0.9929
0.01.396.590 I    27	hc_attn_fn          	  16363.23	  0.0406	   111.9977	  1.00	  1.53	  100.00%	 16384	   13.3468	 95.33%	      4.77%	  0.9946
0.01.396.591 I    31	hc_attn_fn          	  16363.20	  0.0399	    75.7520	  1.00	  1.25	  100.00%	 16384	   13.3547	 95.39%	      8.45%	  0.9717
0.01.396.593 I    20	hc_attn_fn          	  16363.16	  0.0220	   168.7064	  1.00	  2.45	  100.00%	 16384	   12.3712	 88.37%	     10.16%	  0.9055
0.01.396.595 I    34	hc_attn_fn          	  16363.16	  0.0499	   142.0065	  1.00	  1.57	  100.00%	 16384	   13.3404	 95.29%	      5.38%	  0.9324
0.01.396.596 I    19	hc_attn_fn          	  16363.15	  0.0216	    87.0788	  1.00	  1.85	  100.00%	 16384	   12.3356	 88.11%	     24.26%	  0.7183
0.01.396.598 I    22	hc_attn_fn          	  16363.10	  0.0518	   129.8769	  1.00	  2.20	  100.00%	 16384	   12.4220	 88.73%	     13.81%	  0.9146
0.01.396.599 I    14	hc_attn_fn          	  16363.09	  0.1251	   132.0050	  1.00	  1.65	  100.00%	 16384	   13.3488	 95.35%	      4.41%	  0.4924
0.01.396.600 I    30	hc_attn_fn          	  16363.05	  0.0399	    61.8656	  1.00	  1.31	  100.00%	 16384	   13.3664	 95.47%	      6.37%	  0.9879
0.01.396.602 I    25	hc_attn_fn          	  16362.89	  0.0430	    60.9690	  1.00	  1.24	  100.00%	 16384	   13.3026	 95.02%	      8.87%	  0.8936
0.01.396.603 I    40	hc_attn_fn          	  16362.82	  0.1353	    60.4133	  1.00	  1.12	  100.00%	 16384	   13.4675	 96.20%	     10.15%	  0.9945
0.01.396.605 I    35	hc_attn_fn          	  16362.78	  0.0533	   124.7359	  1.00	  1.38	  100.00%	 16384	   13.4371	 95.98%	      4.97%	  0.9937
0.01.396.606 I    16	hc_attn_fn          	  16362.70	  0.0674	   178.8101	  1.00	  2.19	  100.00%	 16384	   13.0201	 93.00%	      3.70%	  0.7535
0.01.396.608 I    33	hc_attn_fn          	  16362.66	  0.0490	    84.0758	  1.00	  1.19	  100.00%	 16384	   13.4367	 95.98%	      7.06%	  0.9962
0.01.396.609 I    36	hc_attn_fn          	  16362.66	  0.0675	   102.9326	  1.00	  1.22	  100.00%	 16384	   13.4832	 96.31%	      6.12%	  0.9952
0.01.396.611 I    29	hc_attn_fn          	  16362.65	  0.0387	    67.5904	  1.00	  1.30	  100.00%	 16384	   13.3857	 95.61%	      5.74%	  0.9378
0.01.396.612 I    23	hc_attn_fn          	  16362.64	  0.0493	    94.9439	  1.00	  1.86	  100.00%	 16384	   12.6616	 90.44%	     17.10%	  0.8746
0.01.396.613 I    41	hc_attn_fn          	  16362.63	  0.1522	    47.1840	  1.00	  0.97	  100.00%	 16384	   13.5488	 96.78%	     10.76%	  0.9846
0.01.396.615 I    28	hc_attn_fn          	  16362.57	  0.0389	    92.7073	  1.00	  1.36	  100.00%	 16384	   13.3964	 95.69%	      4.43%	  0.9628
0.01.396.616 I    15	hc_attn_fn          	  16362.53	  0.0853	    60.3145	  1.00	  1.30	  100.00%	 16384	   13.1741	 94.10%	     22.88%	  0.8835
0.01.396.618 I    17	hc_attn_fn          	  16362.49	  0.0447	    64.4162	  1.00	  1.47	  100.00%	 16384	   12.9131	 92.24%	     23.79%	  0.8219
0.01.396.619 I    24	hc_attn_fn          	  16362.35	  0.0559	    89.8142	  1.00	  1.69	  100.00%	 16384	   12.9263	 92.33%	     16.41%	  0.9966
0.01.396.621 I    42	hc_attn_fn          	  16362.34	  0.1933	    38.7918	  1.00	  0.89	  100.00%	 16384	   13.6287	 97.35%	     10.58%	  0.9685
0.01.396.622 I    13	hc_attn_fn          	  16362.33	  0.1513	    58.0500	  1.00	  1.20	  100.00%	 16384	   13.3379	 95.27%	     23.21%	  0.8754
0.01.396.624 I    18	hc_attn_fn          	  16362.33	  0.0303	   188.3410	  1.00	  2.44	  100.00%	 16384	   12.5657	 89.75%	      7.07%	  0.7666
0.01.396.625 I    12	hc_attn_fn          	  16361.46	  0.2248	    86.6606	  1.00	  1.28	  100.00%	 16384	   13.6233	 97.31%	      3.51%	  0.8995
0.01.396.627 I     1	hc_attn_fn          	  16361.14	  0.1232	    18.8806	  1.00	  1.47	  100.00%	 16384	   12.8287	 91.63%	     22.11%	  0.5245
0.01.396.628 I     9	hc_attn_fn          	  16360.87	  0.1033	    41.1735	  1.00	  0.95	  100.00%	 16384	   13.4373	 95.98%	     14.91%	  0.9172
0.01.396.630 I     5	hc_attn_fn          	  16360.73	  0.1452	    24.0969	  1.00	  1.15	  100.00%	 16384	   13.2480	 94.63%	     19.24%	  0.9925
0.01.396.631 I     2	hc_attn_fn          	  16360.50	  0.1199	    19.5846	  1.00	  1.42	  100.00%	 16384	   12.9181	 92.27%	     21.75%	  0.9992
0.01.396.633 I     6	hc_attn_fn          	  16360.49	  0.1409	    18.5569	  1.00	  1.08	  100.00%	 16384	   13.3574	 95.41%	     18.81%	  0.9899
0.01.396.634 I     3	hc_attn_fn          	  16360.40	  0.1304	    17.6758	  1.00	  1.37	  100.00%	 16384	   12.9949	 92.82%	     21.55%	  0.9985
0.01.396.636 I     4	hc_attn_fn          	  16360.39	  0.1592	    14.4615	  1.00	  1.24	  100.00%	 16384	   13.1544	 93.96%	     20.48%	  0.9891
0.01.396.637 I     7	hc_attn_fn          	  16360.31	  0.1333	    45.9761	  1.00	  1.01	  100.00%	 16384	   13.4518	 96.08%	     15.55%	  0.9556
0.01.396.638 I    11	hc_attn_fn          	  16360.30	  0.2220	   106.6885	  1.00	  1.08	  100.00%	 16384	   13.7467	 98.19%	      2.61%	  0.9424
0.01.396.640 I     8	hc_attn_fn          	  16360.22	  0.1222	    50.0073	  1.00	  0.98	  100.00%	 16384	   13.5054	 96.47%	      9.33%	  0.9519
0.01.396.641 I    10	hc_attn_fn          	  16360.17	  0.2248	   111.6190	  1.00	  1.10	  100.00%	 16384	   13.7705	 98.36%	      2.07%	  0.6718
0.01.396.643 I     0	hc_attn_fn          	  16344.64	  0.4832	     4.8273	  1.00	  0.47	  100.00%	 16384	   13.8715	 99.08%	     13.18%	  0.0000
0.01.396.644 I    42	hc_ffn_fn           	  16363.92	  0.1637	   365.9755	  1.00	  3.40	  100.00%	 16384	   13.2499	 94.64%	      1.03%	  0.5446
0.01.396.646 I    36	hc_ffn_fn           	  16363.76	  0.0492	    56.5910	  1.00	  1.35	  100.00%	 16384	   13.1893	 94.21%	     12.20%	  0.7773
0.01.396.647 I    38	hc_ffn_fn           	  16363.70	  0.0633	    72.3041	  1.00	  1.34	  100.00%	 16384	   13.2545	 94.68%	     10.99%	  0.9695
0.01.396.649 I    37	hc_ffn_fn           	  16363.60	  0.0583	    43.7595	  1.00	  1.26	  100.00%	 16384	   13.2556	 94.68%	     12.57%	  0.9953
0.01.396.650 I    25	hc_ffn_fn           	  16363.47	  0.0364	   129.1548	  1.00	  1.78	  100.00%	 16384	   13.1319	 93.80%	      5.23%	  0.8610
0.01.396.652 I    26	hc_ffn_fn           	  16363.46	  0.0388	   119.2187	  1.00	  1.63	  100.00%	 16384	   13.2673	 94.77%	      5.02%	  0.9981
0.01.396.653 I    19	hc_ffn_fn           	  16363.40	  0.0182	   179.6631	  1.00	  2.59	  100.00%	 16384	   12.1773	 86.98%	      9.95%	  0.7693
0.01.396.655 I    33	hc_ffn_fn           	  16363.34	  0.0423	   145.1136	  1.00	  1.61	  100.00%	 16384	   13.3019	 95.01%	      5.67%	  0.9382
0.01.396.656 I    21	hc_ffn_fn           	  16363.27	  0.0506	    80.0188	  1.00	  1.83	  100.00%	 16384	   12.4407	 88.86%	     22.11%	  0.8635
0.01.396.658 I    31	hc_ffn_fn           	  16363.27	  0.0397	    74.7489	  1.00	  1.24	  100.00%	 16384	   13.3603	 95.43%	      8.21%	  0.9985
0.01.396.659 I    24	hc_ffn_fn           	  16363.20	  0.0414	    63.3099	  1.00	  1.30	  100.00%	 16384	   13.2390	 94.56%	      9.17%	  0.8959
0.01.396.661 I    30	hc_ffn_fn           	  16363.19	  0.0358	    79.7872	  1.00	  1.31	  100.00%	 16384	   13.3195	 95.14%	      8.17%	  0.9778
0.01.396.662 I    39	hc_ffn_fn           	  16363.13	  0.0836	    65.2300	  1.00	  1.19	  100.00%	 16384	   13.3880	 95.63%	      9.99%	  0.9937
0.01.396.663 I    20	hc_ffn_fn           	  16363.08	  0.0739	   143.3494	  1.00	  2.16	  100.00%	 16384	   12.5054	 89.32%	     14.43%	  0.9916
0.01.396.665 I    29	hc_ffn_fn           	  16363.06	  0.0352	    62.9305	  1.00	  1.33	  100.00%	 16384	   13.3427	 95.31%	      6.54%	  0.9896
0.01.396.666 I    18	hc_ffn_fn           	  16363.04	  0.0265	    49.6249	  1.00	  1.65	  100.00%	 16384	   12.5183	 89.42%	     24.80%	  0.9596
0.01.396.668 I    15	hc_ffn_fn           	  16363.03	  0.0848	   126.6352	  1.00	  1.52	  100.00%	 16384	   13.2035	 94.31%	     11.60%	  0.9425
0.01.396.669 I    34	hc_ffn_fn           	  16362.88	  0.0496	   132.0341	  1.00	  1.46	  100.00%	 16384	   13.3996	 95.71%	      5.10%	  0.9966
0.01.396.671 I    40	hc_ffn_fn           	  16362.87	  0.1389	    57.1339	  1.00	  1.06	  100.00%	 16384	   13.5087	 96.49%	     10.13%	  0.9894
0.01.396.672 I    14	hc_ffn_fn           	  16362.85	  0.1086	    75.5316	  1.00	  1.31	  100.00%	 16384	   13.2975	 94.98%	     16.99%	  0.4887
0.01.396.673 I    22	hc_ffn_fn           	  16362.78	  0.0553	   109.8390	  1.00	  1.90	  100.00%	 16384	   12.7338	 90.96%	     13.58%	  0.9414
0.01.396.675 I    32	hc_ffn_fn           	  16362.77	  0.0447	    90.8367	  1.00	  1.24	  100.00%	 16384	   13.4108	 95.79%	      6.92%	  0.9931
0.01.396.676 I    35	hc_ffn_fn           	  16362.74	  0.0539	   114.4799	  1.00	  1.31	  100.00%	 16384	   13.4491	 96.06%	      5.57%	  0.9967
0.01.396.678 I    28	hc_ffn_fn           	  16362.73	  0.0357	    70.7796	  1.00	  1.36	  100.00%	 16384	   13.3403	 95.29%	      5.88%	  0.9370
0.01.396.679 I    41	hc_ffn_fn           	  16362.67	  0.1564	    40.6944	  1.00	  0.95	  100.00%	 16384	   13.5605	 96.86%	     11.35%	  0.9842
0.01.396.681 I    16	hc_ffn_fn           	  16362.58	  0.0594	    75.4775	  1.00	  1.43	  100.00%	 16384	   13.0656	 93.33%	     20.92%	  0.9567
0.01.396.682 I    17	hc_ffn_fn           	  16362.52	  0.0405	    87.3139	  1.00	  1.56	  100.00%	 16384	   12.8579	 91.84%	     23.60%	  0.9793
0.01.396.683 I    23	hc_ffn_fn           	  16362.51	  0.0509	    91.2406	  1.00	  1.74	  100.00%	 16384	   12.8756	 91.97%	     16.09%	  0.8904
0.01.396.685 I    27	hc_ffn_fn           	  16362.43	  0.0339	    92.5911	  1.00	  1.38	  100.00%	 16384	   13.3763	 95.55%	      4.57%	  0.9621
0.01.396.686 I    12	hc_ffn_fn           	  16362.37	  0.2012	    75.7538	  1.00	  1.14	  100.00%	 16384	   13.5515	 96.80%	     11.90%	  0.9599
0.01.396.688 I    13	hc_ffn_fn           	  16361.88	  0.1374	   132.4517	  1.00	  1.57	  100.00%	 16384	   13.2121	 94.37%	     15.30%	  0.9515
0.01.396.690 I     1	hc_ffn_fn           	  16360.87	  0.1205	    18.4227	  1.00	  1.42	  100.00%	 16384	   12.9165	 92.26%	     21.82%	  0.4957
0.01.396.691 I    10	hc_ffn_fn           	  16360.69	  0.2590	    84.5304	  1.00	  0.91	  100.00%	 16384	   13.8128	 98.66%	      2.45%	  0.9627
0.01.396.692 I    11	hc_ffn_fn           	  16360.62	  0.2716	    67.9250	  1.00	  0.97	  100.00%	 16384	   13.7704	 98.36%	      2.91%	  0.9569
0.01.396.694 I     7	hc_ffn_fn           	  16360.59	  0.1314	    43.7607	  1.00	  0.97	  100.00%	 16384	   13.4906	 96.36%	     13.83%	  0.9707
0.01.396.695 I     3	hc_ffn_fn           	  16360.54	  0.1249	    15.8568	  1.00	  1.24	  100.00%	 16384	   13.1541	 93.96%	     20.21%	  0.9894
0.01.396.697 I     6	hc_ffn_fn           	  16360.53	  0.1455	    22.7893	  1.00	  1.03	  100.00%	 16384	   13.4072	 95.77%	     18.03%	  0.9858
0.01.396.698 I     4	hc_ffn_fn           	  16360.53	  0.1426	    20.7141	  1.00	  1.18	  100.00%	 16384	   13.2088	 94.35%	     19.81%	  0.9952
0.01.396.700 I     2	hc_ffn_fn           	  16360.46	  0.1275	    19.3221	  1.00	  1.38	  100.00%	 16384	   12.9879	 92.77%	     21.47%	  0.9994
0.01.396.701 I     5	hc_ffn_fn           	  16360.39	  0.1456	    21.0676	  1.00	  1.15	  100.00%	 16384	   13.2672	 94.77%	     19.52%	  0.9968
0.01.396.703 I     8	hc_ffn_fn           	  16359.91	  0.1168	    43.6758	  1.00	  0.91	  100.00%	 16384	   13.5133	 96.52%	      9.78%	  0.9529
0.01.396.704 I     9	hc_ffn_fn           	  16359.71	  0.2572	    67.3000	  1.00	  0.83	  100.00%	 16384	   13.7801	 98.43%	      5.17%	  0.8296
0.01.396.706 I     0	hc_ffn_fn           	  16346.61	  0.4164	    10.3538	  1.00	  0.46	  100.00%	 16384	   13.8713	 99.08%	     12.71%	  0.0000
0.01.396.708 I    40	indexer_compressor_gate	    220.46	  0.0040	   177.8002	  0.05	  2.78	  100.00%	  4096	    2.5334	 21.11%	      0.07%	  0.0000
0.01.396.710 I    34	indexer_compressor_gate	    152.95	  0.0033	   102.6580	  0.04	  1.67	  100.00%	  4096	    2.7774	 23.15%	      0.05%	  0.0000
0.01.396.711 I    16	indexer_compressor_gate	    102.98	  0.0038	    51.3552	  0.03	  0.92	  100.00%	  4096	    4.2088	 35.07%	      0.05%	  0.0000
0.01.396.713 I    22	indexer_compressor_gate	     99.95	  0.0062	    33.5665	  0.02	  0.59	  100.00%	  4096	    6.6583	 55.49%	      0.15%	  0.0000
0.01.396.714 I    38	indexer_compressor_gate	     89.28	  0.0043	    33.0348	  0.02	  0.57	  100.00%	  4096	    6.4538	 53.78%	      0.10%	  0.0000
0.01.396.715 I    24	indexer_compressor_gate	     73.88	  0.0054	    20.3442	  0.02	  0.37	  100.00%	  4096	    7.3008	 60.84%	      0.15%	  0.0000
0.01.396.717 I    28	indexer_compressor_gate	     69.13	  0.0022	    51.1201	  0.02	  0.80	  100.00%	  4096	    3.4863	 29.05%	      0.05%	  0.0000
0.01.396.718 I    20	indexer_compressor_gate	     68.64	  0.0028	    25.5560	  0.02	  0.55	  100.00%	  4096	    4.7777	 39.81%	      0.05%	  0.0000
0.01.396.719 I    32	indexer_compressor_gate	     52.57	  0.0024	    24.4735	  0.01	  0.43	  100.00%	  4096	    4.8843	 40.70%	      0.05%	  0.0000
0.01.396.721 I    30	indexer_compressor_gate	     51.31	  0.0025	    13.7820	  0.01	  0.31	  100.00%	  4096	    5.9912	 49.93%	      0.07%	  0.0000
0.01.396.722 I    14	indexer_compressor_gate	     49.35	  0.0032	    22.9597	  0.01	  0.37	  100.00%	  4096	    6.2063	 51.72%	      0.05%	  0.0000
0.01.396.723 I    12	indexer_compressor_gate	     43.61	  0.0025	    15.9644	  0.01	  0.27	  100.00%	  4096	    6.3918	 53.26%	      0.10%	  0.0000
0.01.396.725 I    26	indexer_compressor_gate	     42.05	  0.0022	    17.4225	  0.01	  0.33	  100.00%	  4096	    5.3200	 44.33%	      0.05%	  0.0000
0.01.396.726 I    18	indexer_compressor_gate	     40.06	  0.0013	    15.5157	  0.01	  0.30	  100.00%	  4096	    4.5824	 38.19%	      0.07%	  0.0000
0.01.396.728 I    36	indexer_compressor_gate	     33.18	  0.0021	    17.9069	  0.01	  0.28	  100.00%	  4096	    6.4848	 54.04%	      0.02%	  0.0000
0.01.396.729 I     8	indexer_compressor_gate	     28.93	  0.0024	     9.8737	  0.01	  0.16	  100.00%	  4096	    7.7816	 64.85%	      0.05%	  0.0000
0.01.396.730 I    10	indexer_compressor_gate	     28.22	  0.0026	     6.9477	  0.01	  0.13	  100.00%	  4096	    8.4242	 70.20%	      0.07%	  0.0000
0.01.396.732 I     4	indexer_compressor_gate	     16.30	  0.0014	     3.8507	  0.00	  0.07	  100.00%	  4096	    8.4898	 70.75%	      0.05%	  0.0000
0.01.396.733 I     6	indexer_compressor_gate	     15.27	  0.0012	     6.2976	  0.00	  0.10	  100.00%	  4096	    7.5735	 63.11%	      0.05%	  0.0000
0.01.396.735 I    42	indexer_compressor_gate	     13.63	  0.0011	     0.5170	  0.00	  0.01	  100.00%	  4096	   11.5541	 96.28%	      0.15%	  0.0000
0.01.396.736 I     2	indexer_compressor_gate	     11.12	  0.0012	     0.7982	  0.00	  0.01	  100.00%	  4096	   11.2380	 93.65%	      0.10%	  0.0000
0.01.396.738 I    40	indexer_compressor_kv	    220.46	  0.0040	   177.8002	  0.05	  2.78	  100.00%	  4096	    2.5334	 21.11%	      0.07%	  0.0000
0.01.396.739 I    34	indexer_compressor_kv	    152.95	  0.0033	   102.6580	  0.04	  1.67	  100.00%	  4096	    2.7774	 23.15%	      0.05%	  0.0000
0.01.396.740 I    16	indexer_compressor_kv	    102.98	  0.0038	    51.3552	  0.03	  0.92	  100.00%	  4096	    4.2088	 35.07%	      0.05%	  0.0000
0.01.396.742 I    22	indexer_compressor_kv	     99.95	  0.0062	    33.5665	  0.02	  0.59	  100.00%	  4096	    6.6583	 55.49%	      0.15%	  0.0000
0.01.396.743 I    38	indexer_compressor_kv	     89.28	  0.0043	    33.0348	  0.02	  0.57	  100.00%	  4096	    6.4538	 53.78%	      0.10%	  0.0000
0.01.396.744 I    24	indexer_compressor_kv	     73.88	  0.0054	    20.3442	  0.02	  0.37	  100.00%	  4096	    7.3008	 60.84%	      0.15%	  0.0000
0.01.396.746 I    28	indexer_compressor_kv	     69.13	  0.0022	    51.1201	  0.02	  0.80	  100.00%	  4096	    3.4863	 29.05%	      0.05%	  0.0000
0.01.396.747 I    20	indexer_compressor_kv	     68.64	  0.0028	    25.5560	  0.02	  0.55	  100.00%	  4096	    4.7777	 39.81%	      0.05%	  0.0000
0.01.396.748 I    32	indexer_compressor_kv	     52.57	  0.0024	    24.4735	  0.01	  0.43	  100.00%	  4096	    4.8843	 40.70%	      0.05%	  0.0000
0.01.396.749 I    30	indexer_compressor_kv	     51.31	  0.0025	    13.7820	  0.01	  0.31	  100.00%	  4096	    5.9912	 49.93%	      0.07%	  0.0000
0.01.396.751 I    14	indexer_compressor_kv	     49.35	  0.0032	    22.9597	  0.01	  0.37	  100.00%	  4096	    6.2063	 51.72%	      0.05%	  0.0000
0.01.396.752 I    12	indexer_compressor_kv	     43.61	  0.0025	    15.9644	  0.01	  0.27	  100.00%	  4096	    6.3918	 53.26%	      0.10%	  0.0000
0.01.396.753 I    26	indexer_compressor_kv	     42.05	  0.0022	    17.4225	  0.01	  0.33	  100.00%	  4096	    5.3200	 44.33%	      0.05%	  0.0000
0.01.396.754 I    18	indexer_compressor_kv	     40.06	  0.0013	    15.5157	  0.01	  0.30	  100.00%	  4096	    4.5824	 38.19%	      0.07%	  0.0000
0.01.396.756 I    36	indexer_compressor_kv	     33.18	  0.0021	    17.9069	  0.01	  0.28	  100.00%	  4096	    6.4848	 54.04%	      0.02%	  0.0000
0.01.396.757 I     8	indexer_compressor_kv	     28.93	  0.0024	     9.8737	  0.01	  0.16	  100.00%	  4096	    7.7816	 64.85%	      0.05%	  0.0000
0.01.396.758 I    10	indexer_compressor_kv	     28.22	  0.0026	     6.9477	  0.01	  0.13	  100.00%	  4096	    8.4242	 70.20%	      0.07%	  0.0000
0.01.396.759 I     4	indexer_compressor_kv	     16.30	  0.0014	     3.8507	  0.00	  0.07	  100.00%	  4096	    8.4898	 70.75%	      0.05%	  0.0000
0.01.396.761 I     6	indexer_compressor_kv	     15.27	  0.0012	     6.2976	  0.00	  0.10	  100.00%	  4096	    7.5735	 63.11%	      0.05%	  0.0000
0.01.396.762 I    42	indexer_compressor_kv	     13.63	  0.0011	     0.5170	  0.00	  0.01	  100.00%	  4096	   11.5541	 96.28%	      0.15%	  0.0000
0.01.396.763 I     2	indexer_compressor_kv	     11.12	  0.0012	     0.7982	  0.00	  0.01	  100.00%	  4096	   11.2380	 93.65%	      0.10%	  0.0000
0.01.396.765 I    40	proj                	    220.46	  0.0040	   177.8002	  0.05	  2.78	  100.00%	  4096	    2.5334	 21.11%	      0.07%	  0.0000
0.01.396.766 I    34	proj                	    152.95	  0.0033	   102.6580	  0.04	  1.67	  100.00%	  4096	    2.7774	 23.15%	      0.05%	  0.0000
0.01.396.767 I    16	proj                	    102.98	  0.0038	    51.3552	  0.03	  0.92	  100.00%	  4096	    4.2088	 35.07%	      0.05%	  0.0000
0.01.396.768 I    22	proj                	     99.95	  0.0062	    33.5665	  0.02	  0.59	  100.00%	  4096	    6.6583	 55.49%	      0.15%	  0.0000
0.01.396.770 I    38	proj                	     89.28	  0.0043	    33.0348	  0.02	  0.57	  100.00%	  4096	    6.4538	 53.78%	      0.10%	  0.0000
0.01.396.771 I    24	proj                	     73.88	  0.0054	    20.3442	  0.02	  0.37	  100.00%	  4096	    7.3008	 60.84%	      0.15%	  0.0000
0.01.396.772 I    28	proj                	     69.13	  0.0022	    51.1201	  0.02	  0.80	  100.00%	  4096	    3.4863	 29.05%	      0.05%	  0.0000
0.01.396.774 I    20	proj                	     68.64	  0.0028	    25.5560	  0.02	  0.55	  100.00%	  4096	    4.7777	 39.81%	      0.05%	  0.0000
0.01.396.775 I    32	proj                	     52.57	  0.0024	    24.4735	  0.01	  0.43	  100.00%	  4096	    4.8843	 40.70%	      0.05%	  0.0000
0.01.396.776 I    30	proj                	     51.31	  0.0025	    13.7820	  0.01	  0.31	  100.00%	  4096	    5.9912	 49.93%	      0.07%	  0.0000
0.01.396.777 I    14	proj                	     49.35	  0.0032	    22.9597	  0.01	  0.37	  100.00%	  4096	    6.2063	 51.72%	      0.05%	  0.0000
0.01.396.779 I    12	proj                	     43.61	  0.0025	    15.9644	  0.01	  0.27	  100.00%	  4096	    6.3918	 53.26%	      0.10%	  0.0000
0.01.396.780 I    26	proj                	     42.05	  0.0022	    17.4225	  0.01	  0.33	  100.00%	  4096	    5.3200	 44.33%	      0.05%	  0.0000
0.01.396.781 I    18	proj                	     40.06	  0.0013	    15.5157	  0.01	  0.30	  100.00%	  4096	    4.5824	 38.19%	      0.07%	  0.0000
0.01.396.782 I    36	proj                	     33.18	  0.0021	    17.9069	  0.01	  0.28	  100.00%	  4096	    6.4848	 54.04%	      0.02%	  0.0000
0.01.396.783 I     8	proj                	     28.93	  0.0024	     9.8737	  0.01	  0.16	  100.00%	  4096	    7.7816	 64.85%	      0.05%	  0.0000
0.01.396.785 I    10	proj                	     28.22	  0.0026	     6.9477	  0.01	  0.13	  100.00%	  4096	    8.4242	 70.20%	      0.07%	  0.0000
0.01.396.786 I     4	proj                	     16.30	  0.0014	     3.8507	  0.00	  0.07	  100.00%	  4096	    8.4898	 70.75%	      0.05%	  0.0000
0.01.396.787 I     6	proj                	     15.27	  0.0012	     6.2976	  0.00	  0.10	  100.00%	  4096	    7.5735	 63.11%	      0.05%	  0.0000
0.01.396.788 I    42	proj                	     13.63	  0.0011	     0.5170	  0.00	  0.01	  100.00%	  4096	   11.5541	 96.28%	      0.15%	  0.0000
0.01.396.790 I     2	proj                	     11.12	  0.0012	     0.7982	  0.00	  0.01	  100.00%	  4096	   11.2380	 93.65%	      0.10%	  0.0000
0.01.396.790 I 
Computing weighted average statistics per layer (43 layers)
0.01.396.791 I 
  Layer	     μΣ(Act²)	      μZD	μCosSim
0.01.396.791 I ================================================
0.01.396.791 I     0	      45096.41	    1.6279%	0.0000
0.01.396.792 I     1	      51963.16	    9.5524%	0.4461
0.01.396.792 I     2	      78978.60	    9.0957%	0.6127
0.01.396.793 I     3	      55786.58	    9.3676%	0.5029
0.01.396.793 I     4	      48042.13	    9.5116%	0.3847
0.01.396.794 I     5	      61089.36	    9.6201%	0.3506
0.01.396.794 I     6	      83091.53	    8.9731%	0.3242
0.01.396.794 I     7	      99725.29	    9.9709%	0.3133
0.01.396.795 I     8	      93655.02	    9.4964%	0.3762
0.01.396.795 I     9	      71187.19	    8.7951%	0.3749
0.01.396.796 I    10	      62147.18	    8.9972%	0.4894
0.01.396.796 I    11	      70249.90	    8.5143%	0.4201
0.01.396.797 I    12	      80725.38	    9.7364%	0.3933
0.01.396.797 I    13	      80908.72	    9.3175%	0.3846
0.01.396.797 I    14	      83711.07	    9.4338%	0.3966
0.01.396.798 I    15	      85692.59	    9.1254%	0.4127
0.01.396.798 I    16	      94736.20	    9.0383%	0.3973
0.01.396.799 I    17	     117459.11	    8.7189%	0.4378
0.01.396.799 I    18	     147257.50	    8.9463%	0.4178
0.01.396.800 I    19	     108021.04	    9.7298%	0.4103
0.01.396.800 I    20	     193910.73	    8.9083%	0.3965
0.01.396.801 I    21	     155726.83	    9.1333%	0.4160
0.01.396.801 I    22	     183062.80	    8.9025%	0.3921
0.01.396.801 I    23	     141374.66	    7.8313%	0.4412
0.01.396.802 I    24	     151901.70	    9.0455%	0.4517
0.01.396.802 I    25	     171459.92	    8.1764%	0.4250
0.01.396.803 I    26	     204140.98	    9.1310%	0.3767
0.01.396.803 I    27	     193475.47	    8.6614%	0.3406
0.01.396.803 I    28	     221902.45	    7.9349%	0.3616
0.01.396.804 I    29	     217150.89	    8.2836%	0.3237
0.01.396.804 I    30	     235011.75	    8.4597%	0.4078
0.01.396.805 I    31	     227817.17	    8.1360%	0.4232
0.01.396.805 I    32	     248097.45	    8.0582%	0.4149
0.01.396.805 I    33	     241886.28	    7.0889%	0.3840
0.01.396.806 I    34	     287909.72	    8.6132%	0.3573
0.01.396.806 I    35	     267275.97	    8.3465%	0.4099
0.01.396.807 I    36	     323973.88	    8.5563%	0.3836
0.01.396.807 I    37	     312700.62	    8.2295%	0.3833
0.01.396.808 I    38	     309875.03	    8.3421%	0.3621
0.01.396.808 I    39	     497055.81	    8.9362%	0.3698
0.01.396.808 I    40	     388621.50	    8.8169%	0.3591
0.01.396.809 I    41	     532088.56	    9.2011%	0.3753
0.01.396.809 I    42	     661004.19	    8.1027%	0.3762

```

</details>



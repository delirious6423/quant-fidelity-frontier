---
base_model:
- Qwen/Qwen3.8-Flash-Next
---

A simpler quantization mix trying to balance speed, size and performance on older hardware. Full Q8 n-gram pipeline.

**Size**<br>
114,245 MiB · 5.42 bpw · 6 shards (core 60.9 GiB + 50.7 GiB table on disk). <br>
Approx 65-70GiB total of RAM and/or VRAM needed for 131K context and Q8/F16 cache with n-gram offloading enabled. <br>
MTP not included, use whichever standalone version fits best. Though currently <code>--ngram-mod</code> drafting  and variants seems to work better in most cases.<br>
<b>Addendum</b>: Added a standalone MTP GGUF in simple Q4_0 format. With F16 k/v cache it seems to give a noticeable, but not overwhelming, boost. However prompt processing takes a bit of a hit so depending if your weakest link is PP or TG it might be better to just stick to basic n-gram draft.

**Recipe**<br> (base Q8_0): signal path → Q8_0 · gate/up → Q3_K (edge blk 0–3 /
40–47 → IQ4_NL) · down → IQ4_NL · n-gram table → Q8_0.



**PPL**<br>
**4.6587** ± 0.028 (WikiText-2, 512 ctx) = +1.49% vs the **4.5903** full BF16.
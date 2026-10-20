---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
---

~~This is currently a STATIC quant, because the imatrix tool seems to be broken with Gemma 4 (>100 ppl). I will update with an imatrix once I can verify correctness.~~

I made a custom imatrix dataset by slapping together random columns from some popular datasets on huggingface and formatting using the official jinja template. Comapred to the unstructured bartowski dataset, PPL went from multiple thousands to single digits, so I think it should be good now. **Just in case, I mirrored the old static quant to** https://huggingface.co/Beinsezii/gemma-4-26B-A4B-it-GGUF-6.52BPW-static

6.52 bpw, a mixture of Q6_K, Q5_K, and Q8_0

Fits ~75k F16 CTX + MMPROJ on a 24GiB GPU, or ~150k CTX without vision. Measured on a DE, not headless.
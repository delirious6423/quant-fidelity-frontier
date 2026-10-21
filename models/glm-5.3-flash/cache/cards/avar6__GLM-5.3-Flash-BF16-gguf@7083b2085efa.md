---
base_model:
- zai-org/GLM-5.3-Flash
---
experimental ggufs with vision support by timkronos. Requires this llamacpp fork https://github.com/timkhronos/llama.cpp/tree/GLM5.3-Flash. Link to the current pr for it https://github.com/ggml-org/llama.cpp/pull/27773

also for clarity, these are quantitized from the bf16 safetensors as it says in the title. But the default repo is fp8. I've linked the default repo as the base since its where people looking for llamacpp compatible ggufs for glm 5.3 flash will look

Current Quant Recipes:
- IQ2_XXS-2.32bpw ~ 92gb
- IQ2_S-2.83bpw ~ 114gb
- Q2_K-Q3_K-3.20bpw ~ 128gb
- Q3_S-3.60bpw ~ 144gb
- Q3_XL-3.86bpw ~ 155gb

blk.45→Q8_0, ssm→F32/Q8_0, indexer→F32 on all quants

Updates:
- 08/27/2026 2pm: the mmproj has been updated, should see improvements in vision
- 08/28/2026 8am: uploaded all new quants. New set of mixed precision i and k quants using Aes Sedai's gguf recipe. The mtp tensor is kept at Q8 for performance. MTP currently working in timkronos's mtp fork
- 08/28/2026 6pm: oops all new quants needed
- 08/29/2026 3pm: ok fast moving few days. Quants going back up today. PR looks close to merging. I will keep my quants up until Aes Sedai makes his which may only be a few days. I will keep any odd quants he does not make
- 08/30/2026: Aes Sedai has uploaded his glm flash quants so I recommend checking out his repo: https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF
- I may upload more small sized mixed quants but at this point I need to save space on my hf account and will delete the q5 (same as Aes's anyway)
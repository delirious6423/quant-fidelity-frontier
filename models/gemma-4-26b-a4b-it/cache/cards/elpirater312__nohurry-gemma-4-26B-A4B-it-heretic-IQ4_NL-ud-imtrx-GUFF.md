---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
- nohurry/gemma-4-26B-A4B-it-heretic-GUFF
---
# ⚠️ UPDATE APRIL 09: V2 RELEASE (CRITICAL FIXES)
**The previous version was built on outdated llama.cpp kernels. This V2 release incorporates the latest architectural fixes for Gemma-4, including the `<unused24>` buffer fix and Logit Soft-Capping.**

# THE ULTIMATE IQ4_NL (V2) - APRIL 2026 FIXES

The previous quant is out of date and MISSING critical architectural fixes. This V2 release is the first to incorporate the **Unsloth Dynamic 2.0** re-computed imatrix and the latest `llama.cpp` Gemma-4 patches.

### What's New in V2? (The Unsloth Special)
1.  **CRITICAL FIX:** `<unused24>` token buffer overlap fixed (No more random logic crashes).
2.  **KV-Cache Support:** Full attention rotation for heterogeneous iSWA (Better long-context stability).
3.  **Vocab Fix:** Byte token handling added to BPE de-tokenizer (Fixes weird character output).
4.  **BOS Logic:** `add_bos` now defaults to `True` for native Gemma-4 behavior.
5.  **Logit Soft-Capping:** Correctly reads `final_logit_softcapping` (CRITICAL for preventing NaN/Infinity loops).
6.  **Newline Logic:** Custom newline split logic for superior formatting.

###  Why this Quant?
- **Format:** `IQ4_NL` (Non-Linear) - Higher logic-per-bit than Q4_K_M or Q4_0.
- **Brain:** High-res `Q8_0` Token Embeddings and Output Head (Prevents vocabulary degradation).
- **Speed:** Specially optimized for CPU/Laptop P-Cores (~7 t/s on 12th Gen i3 Dual DDR4 Channel at 3200MHz, CPU ONLY inference!).
- **Matrix:** Built with the latest **April 07 re-computed Unsloth imatrix**.

All credits to Merijn Hendriks (nohurry) for the heretic weights and Unsloth for the imatrix file.
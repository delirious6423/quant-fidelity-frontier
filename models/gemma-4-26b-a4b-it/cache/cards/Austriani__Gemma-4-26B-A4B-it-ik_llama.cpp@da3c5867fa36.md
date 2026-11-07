---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
library_name: ik_llama.cpp
language:
- en
tags:
  - gemma-4
  - gemma-4-26b-a4b
  - google-deepmind
  - moe
  - mixture-of-experts
  - 26b
  - a4b
  - ik-llama.cpp
  - iqk
  - trellis-quant
  - maxoptima
  - gguf
  - quantized
  - imatrix
  - v3-calibration
  - text-generation
  - austriani
---

<style>
.obsidian-wrap {
  max-width: 960px;
  margin: 0 auto;
  background-color: #050505;
  color: #a3a3a3;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  line-height: 1.7;
  padding: 60px 40px;
  border: 1px solid #1a1a1a;
}

.obsidian-header {
  border-bottom: 2px solid #1a1a1a;
  padding-bottom: 40px;
  margin-bottom: 40px;
}

.obsidian-kicker {
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.75rem;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: #10b981;
  margin-bottom: 20px;
  display: block;
}

.obsidian-title {
  font-family: 'Playfair Display', 'Times New Roman', Times, serif;
  font-size: 3.5rem;
  font-weight: 400;
  line-height: 1.1;
  color: #f5f5f5;
  margin: 0 0 16px;
  letter-spacing: -0.5px;
}

.obsidian-subtitle {
  font-size: 1.1rem;
  color: #737373;
  font-weight: 300;
  letter-spacing: 1px;
}

.obsidian-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  margin-bottom: 50px;
  border-top: 1px solid #1a1a1a;
  border-bottom: 1px solid #1a1a1a;
}

.obsidian-stat {
  flex: 1;
  min-width: 150px;
  padding: 30px 20px;
  border-right: 1px solid #1a1a1a;
}

.obsidian-stat:last-child {
  border-right: none;
}

.obsidian-stat-val {
  font-family: 'Playfair Display', 'Times New Roman', Times, serif;
  font-size: 2.5rem;
  color: #f5f5f5;
  line-height: 1;
  margin-bottom: 8px;
}

.obsidian-stat-val.accent {
  color: #10b981;
}

.obsidian-stat-lbl {
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: #525252;
}

.obsidian-h2 {
  font-family: 'Playfair Display', 'Times New Roman', Times, serif;
  font-size: 1.8rem;
  font-weight: 400;
  color: #f5f5f5;
  margin: 50px 0 20px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.obsidian-h2::after {
  content: '';
  flex: 1;
  height: 1px;
  background: #1a1a1a;
}

.obsidian-text {
  font-size: 1rem;
  color: #a3a3a3;
  margin-bottom: 20px;
}

.obsidian-text strong {
  color: #f5f5f5;
  font-weight: 600;
}

.obsidian-text a {
  color: #10b981;
  text-decoration: none;
  border-bottom: 1px solid #10b981;
}

.obsidian-code {
  font-family: 'Courier New', Courier, monospace;
  background: #111111;
  padding: 4px 8px;
  color: #10b981;
  border: 1px solid #262626;
  font-size: 0.85em;
}

.obsidian-callout {
  border-left: 2px solid #10b981;
  padding: 10px 0 10px 20px;
  margin: 30px 0;
  font-size: 1rem;
  color: #d4d4d4;
}

details.obsidian-acc {
  border-bottom: 1px solid #1a1a1a;
}

details.obsidian-acc summary {
  padding: 20px 0;
  font-size: 1rem;
  font-weight: 600;
  color: #f5f5f5;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  list-style: none;
  user-select: none;
  text-transform: uppercase;
  letter-spacing: 1px;
}

details.obsidian-acc summary::-webkit-details-marker {
  display: none;
}

details.obsidian-acc summary:hover {
  color: #10b981;
}

details.obsidian-acc summary::after {
  content: '[ + ]';
  font-family: 'Courier New', Courier, monospace;
  color: #525252;
  font-size: 0.9rem;
}

details.obsidian-acc[open] summary::after {
  content: '[ - ]';
  color: #10b981;
}

.obsidian-acc-content {
  padding: 10px 0 30px;
}

.obsidian-acc-grid {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 20px;
  font-size: 0.95rem;
}

.obsidian-acc-quant {
  font-family: 'Courier New', Courier, monospace;
  color: #f5f5f5;
  font-weight: 600;
}

.obsidian-acc-desc {
  color: #a3a3a3;
}

.obsidian-pre {
  background: #0a0a0a;
  border: 1px solid #1a1a1a;
  padding: 20px;
  overflow-x: auto;
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.85rem;
  color: #10b981;
  line-height: 1.6;
  white-space: pre-wrap;
  word-wrap: break-word;
  margin-top: 20px;
}

.obsidian-footer {
  margin-top: 60px;
  padding-top: 20px;
  border-top: 1px solid #1a1a1a;
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.75rem;
  color: #525252;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

@media (max-width: 700px) {
  .obsidian-wrap { padding: 30px 20px; }
  .obsidian-title { font-size: 2.5rem; }
  .obsidian-stat { border-right: none; border-bottom: 1px solid #1a1a1a; min-width: 50%; }
  .obsidian-acc-grid { grid-template-columns: 1fr; gap: 4px; }
}

@media (max-width: 450px) {
  .obsidian-stat { min-width: 100%; }
}
</style>

<div class="obsidian-wrap">

<div class="obsidian-header">
<span class="obsidian-kicker">Austriani / ik_llama.cpp</span>
<h1 class="obsidian-title">Gemma-4-26B-A4B-it</h1>
<div class="obsidian-subtitle">Architectural Quantization Archive</div>
</div>

<div class="obsidian-stats">
<div class="obsidian-stat">
<div class="obsidian-stat-val">26B</div>
<div class="obsidian-stat-lbl">Total Params</div>
</div>
<div class="obsidian-stat">
<div class="obsidian-stat-val accent">4B</div>
<div class="obsidian-stat-lbl">Active Params</div>
</div>
<div class="obsidian-stat">
<div class="obsidian-stat-val">1024</div>
<div class="obsidian-stat-lbl">Imatrix Ctx</div>
</div>
<div class="obsidian-stat">
<div class="obsidian-stat-val accent">500</div>
<div class="obsidian-stat-lbl">Cycles</div>
</div>
</div>

<div class="obsidian-h2">Base Architecture</div>

<div class="obsidian-text">
This repository hosts <strong>ik_llama.cpp-compatible GGUF quantizations</strong> of <a href="https://huggingface.co/google/gemma-4-26B-A4B-it">google/gemma-4-26B-A4B-it</a>. This is Google DeepMind's 26B-parameter Mixture-of-Experts model, activating only <strong>3.8B parameters per token</strong> through 128 fine-grained experts and top-8 routing. It is built for cost-efficient inference, native function calling, and multimodal input, supporting a <strong>256K-token context window</strong>.
</div>

<div class="obsidian-text">
The architecture uses <strong>30 layers</strong> with a hybrid attention scheme: interleaved local sliding-window attention (<strong>1024 tokens</strong>) and global attention. It has <strong>16 attention heads over 8 KV heads</strong>. The model is released under the Apache 2.0 license.
</div>

<div class="obsidian-h2">MaxOptima Custom Quant</div>

<div class="obsidian-text">
This release includes <strong>MaxOptima</strong>, a bespoke, surgical quantization mix designed specifically for the 128-expert MoE architecture of Gemma 4. The goal is to protect the expert routing and critical attention layers while aggressively compressing the bulk of the expert FFNs, producing a file significantly smaller than a standard Q4_K_M.
</div>

<div class="obsidian-callout">
<strong>MaxOptima Strategy:</strong> Embeddings and output layers are locked at Q6_K. Router inputs, norms, scales, and biases are kept at full precision (F32) to preserve routing integrity. Key attention layers are kept at IQ5_K. The bulk of the expert FFNs are compressed to IQ4_KSS and IQ4_NL, with boundary blocks protected at higher precision.
</div>

<details class="obsidian-acc">
<summary>View Full Custom Quant Configuration</summary>
<div class="obsidian-acc-content">
<div class="obsidian-pre">^output\.weight=Q6_K,^token_embd\.weight=Q6_K,ffn_gate_inp\.weight=F32,rope_freqs\.weight=F32,.*_norm\.weight=F32,.*\.scale=F32,.*_scale\.weight=F32,.*\.bias=F32,blk\.(0|1|2|3|4|5|6|11|17|18|23|24|25|26|27|28|29)\.attn_k\.weight=IQ5_K,attn_k\.weight=IQ4_KSS,attn_v\.weight=IQ5_K,blk\.(0|1|2|3|4|5|11|17|23|25|26|27|28|29)\.attn_output\.weight=IQ5_K,attn_output\.weight=IQ4_KS,attn_q\.weight=IQ5_K,blk\.(0|1|2|3|4|25|26|27|28|29)\.ffn_gate_up_exps\.weight=IQ5_K,ffn_gate_up_exps\.weight=IQ4_KSS,blk\.(0|1|2|3|4|25|26|27|28|29)\.ffn_down_exps\.weight=Q5_0,ffn_down_exps\.weight=IQ4_NL,ffn_down\.weight=Q5_0,ffn_(gate|up)\.weight=IQ5_K,.*=IQ4_KS</div>
</div>
</details>

<div class="obsidian-h2">Available Quants</div>

<details class="obsidian-acc">
<summary>MaxOptima // Custom MoE Mix</summary>
<div class="obsidian-acc-content">
<div class="obsidian-acc-grid">
<div class="obsidian-acc-quant">MaxOptima</div>
<div class="obsidian-acc-desc">Custom mix. F32 routing/scales, IQ5_K attention, IQ4_KSS/IQ4_NL expert FFNs. Best quality-to-size ratio for MoE inference.</div>
</div>
</div>
</details>

<details class="obsidian-acc">
<summary>Standard IQK // High & Mid Fidelity</summary>
<div class="obsidian-acc-content">
<div class="obsidian-acc-grid">
<div class="obsidian-acc-quant">IQ5_K</div>
<div class="obsidian-acc-desc">Higher fidelity 5-bit. Best quality-to-size for systems with sufficient memory.</div>
<div class="obsidian-acc-quant">IQ4_K</div>
<div class="obsidian-acc-desc">Standard 4-bit IQK. Uses 256-value super-blocks for scaling.</div>
<div class="obsidian-acc-quant">IQ4_KS</div>
<div class="obsidian-acc-desc">"Small" 4-bit variant with tighter packing. Lower memory footprint.</div>
<div class="obsidian-acc-quant">IQ4_KS_R4</div>
<div class="obsidian-acc-desc">Row-interleaved variant of IQ4_KS, optimized for ik_llama.cpp's matmul kernels.</div>
</div>
</div>
</details>

<details class="obsidian-acc">
<summary>Trellis // Alternative 4-Bit</summary>
<div class="obsidian-acc-content">
<div class="obsidian-acc-grid">
<div class="obsidian-acc-quant">IQ4_KT</div>
<div class="obsidian-acc-desc">Dynamic programming trellis quant. Uses a fundamentally different error-minimization strategy than IQK.</div>
</div>
</div>
</details>

<div class="obsidian-h2">Imatrix Specification</div>

<div class="obsidian-text">
This release utilizes the exact same <strong>V3 Calibration Dataset</strong> used across previous Austriani releases. No datasets were changed. The calibration was executed at <strong>500 cycles</strong> per chunk with a <strong>1024-token context window</strong>, processing approximately <strong>1000 chunks</strong>.
</div>

<div class="obsidian-h2">Compatibility</div>

<div class="obsidian-text">
<strong>These GGUF files are built exclusively for ik_llama.cpp.</strong> IQK, Trellis, and MaxOptima quants will not load in mainline llama.cpp. Please refer to the official base repository for mainline GGUF compatibility.
</div>

<div class="obsidian-footer">
<div>Austriani // Gemma-4-26B-A4B-it // ik_llama.cpp</div>
<div>Base: google/gemma-4-26B-A4B-it</div>
</div>

</div>
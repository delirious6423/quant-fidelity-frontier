---
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
  - gguf
  - ollama
  - llama.cpp
  - imatrix
  - moe
  - reasoning
  - frontier
  - qwen3
  - onyx
pipeline_tag: text-generation
---

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Geist', Roboto, sans-serif; border: 1px solid #333; border-radius: 18px; box-shadow: 0 18px 40px rgba(255, 255, 255, 0.05); overflow: hidden; background: #000000; margin-bottom: 30px;">
  <div style="background: radial-gradient(circle at 15% 15%, rgba(255, 255, 255, 0.08) 0%, rgba(40, 40, 40, 0.4) 28%, transparent 46%), linear-gradient(135deg, #050505 0%, #1a1a1a 58%, #000000 100%); padding: 28px; color: white; border-bottom: 1px solid #222;">
  <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
  <h1 style="margin: 0; font-size: 28px; font-weight: 900; display: flex; align-items: center; gap: 12px; color: #ffffff; border: none;"><a href="https://huggingface.co/spaces/el4/ONYX" display: flex; align-items: center; gap: 12px; style="color: #fff; text-decoration: none;">🪨 ONYX</a></h1>
  <span style="background: #ffffff; color: #000000; font-size: 11px; font-weight: 900; padding: 5px 12px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.7px; box-shadow: 0 0 12px rgba(255, 255, 255, 0.2);">Adaptive Precision Engine</span>
  </div>
  <p style="margin: 10px 0 0 0; font-size: 14px; color: #a0a0a0; font-weight: 600;">This repo contains ONYX Quants of ornith-ai/Ornith-1.5-35B-A3B</p>
  </div>

  <!-- Technical Highlights -->
  <div style="padding: 24px; border-bottom: 1px solid #222;">
    <h3 style="margin: 0 0 16px 0; color: #fff; font-size: 16px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">⚙️ The ONYX Architecture</h3>
    <div style="display: grid; grid-template-columns: 1fr; gap: 12px;">
      <div style="background: #0d0d0d; border: 1px solid #333; border-radius: 12px; padding: 16px;">
        <h4 style="color: #fff; margin: 0 0 6px 0; font-size: 14px; font-weight: 700;">🧠 Dynamic Layer Sensitivity</h4>
        <p style="color: #888; font-size: 12px; margin: 0; line-height: 1.5;">Replaces hardcoded edge boundaries with real activation variance measurements. ONYX autonomously identifies critical layers (like mid-network attention blocks) and protects them dynamically.</p>
      </div>
      <div style="background: #0d0d0d; border: 1px solid #333; border-radius: 12px; padding: 16px;">
        <h4 style="color: #fff; margin: 0 0 6px 0; font-size: 14px; font-weight: 700;">🎯 Router-Weighted Imatrix</h4>
        <p style="color: #888; font-size: 12px; margin: 0; line-height: 1.5;">Captures MoE router probabilities and multiplies them into activation scales. This forces the quantizer to aggressively crush "cold" experts while fiercely protecting "hot" ones within the same tensor block.</p>
      </div>
      <div style="background: #0d0d0d; border: 1px solid #333; border-radius: 12px; padding: 16px;">
        <h4 style="color: #fff; margin: 0 0 6px 0; font-size: 14px; font-weight: 700;">🏗️ Architecture-Agnostic</h4>
        <p style="color: #888; font-size: 12px; margin: 0; line-height: 1.5;">Dynamically reads HuggingFace modules and the generated F16 GGUF to map tensors. No hardcoded regex. Works out-of-the-box on Llama, DeepSeek, and custom hybrid SSM/MoE architectures.</p>
      </div>
    </div>
  </div>

  <!-- Tiers Table -->
  <div style="padding: 24px;">
    <table style="width: 100%; border-collapse: collapse; font-family: inherit; font-size: 13px;">
    <thead>
    <tr style="background: #111111;">
    <th style="padding: 10px; border-bottom: 2px solid #333; text-align: left; color: #ffffff; font-weight: 900;">Tier Name</th>
    <th style="padding: 10px; border-bottom: 2px solid #333; text-align: left; color: #ffffff; font-weight: 900;">Target Quality</th>
    <th style="padding: 10px; border-bottom: 2px solid #333; text-align: left; color: #ffffff; font-weight: 900;">Target Size</th>
    <th style="padding: 10px; border-bottom: 2px solid #333; text-align: left; color: #ffffff; font-weight: 900;">Middle Layer Strategy</th>
    </tr>
    </thead>
    <tbody>
    <tr><td style="padding: 10px; border-bottom: 1px solid #222; font-weight: 800; color: #e0e0e0;">🪨 <code style="color:#ffffff;">quality</code></td><td style="padding: 10px; border-bottom: 1px solid #222; color: #888;">Q8 Match</td><td style="padding: 10px; border-bottom: 1px solid #222; color: #888;">~21 GB</td><td style="padding: 10px; border-bottom: 1px solid #222; color: #888;">IQ4_XS</td></tr>
    <tr style="background: #0a0a0a;"><td style="padding: 10px; border-bottom: 1px solid #222; font-weight: 800; color: #e0e0e0;">⚖️ <code style="color:#ffffff;">balanced</code></td><td style="padding: 10px; border-bottom: 1px solid #222; color: #888;">Q6 Match</td><td style="padding: 10px; border-bottom: 1px solid #222; color: #888;">~24 GB</td><td style="padding: 10px; border-bottom: 1px solid #222; color: #888;">Q5_K</td></tr>
    <tr><td style="padding: 10px; border-bottom: 1px solid #222; font-weight: 800; color: #e0e0e0;">📦 <code style="color:#ffffff;">compact</code></td><td style="padding: 10px; border-bottom: 1px solid #222; color: #888;">Q4 Match</td><td style="padding: 10px; border-bottom: 1px solid #222; color: #888;">~16 GB</td><td style="padding: 10px; border-bottom: 1px solid #222; color: #888;">Q3_K</td></tr>
    <tr style="background: #0a0a0a;"><td style="padding: 10px; font-weight: 800; color: #e0e0e0;">🚀 <code style="color:#ffffff;">mini</code></td><td style="padding: 10px; color: #888;">Q2 Match</td><td style="padding: 10px; color: #888;">~12 GB</td><td style="padding: 10px; color: #888;">IQ2_XXS</td></tr>
    </tbody>
    </table>
  </div>
</div>

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #222; border-radius: 16px; overflow: hidden; background: #000000; box-shadow: 0 10px 24px rgba(255, 255, 255, 0.03); margin-bottom: 30px;">
  <div style="background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 65%, #000000 100%); padding: 18px 20px; color: white; border-bottom: 1px solid #222;">
  <h3 style="margin: 0; color: #ffffff; border: none; font-size: 19px; font-weight: 900;">📚 Credits & Foundations</h3>
  </div>
  <div style="padding: 20px; font-size: 13px; color: #888; line-height: 1.7;">
  <p style="margin: 0 0 12px 0;">👉 <b style="color: #fff;">APEX Quantization Method</b><br>Ettore Di Giacinto & Richard Palethorpe (LocalAI Team). ONYX evolves the layer-wise precision gradients and MoE-aware tensor classification outlined in the APEX technical paper into a fully dynamic, data-driven engine.</p>
  <p style="margin: 0 0 12px 0;">👉 <b style="color: #fff;">Bartowski and Lamim</b><br>For the excellent semantic imatrix calibration dataset that powers ONYX's activation scaling.</p>
  <p style="margin: 0 0 12px 0;">👉 <b style="color: #fff;">llama.cpp</b><br>Georgi Gerganov and contributors for the foundational inference and quantization engine.</p>
  <p style="margin: 0 0 12px 0;">👉 <b style="color: #fff;">HuggingFace Accelerate</b><br>For the <code style="color:#ddd;">init_empty_weights()</code> context manager that makes the 0-RAM "Ghost Model" possible on consumer hardware.</p>
  </div>
</div>

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #444; border-radius: 18px; box-shadow: 0 18px 40px rgba(255, 255, 255, 0.04); overflow: hidden; background: #000000; margin-top: 30px; margin-bottom: 30px;">
  <div style="background: radial-gradient(circle at 15% 15%, rgba(255, 255, 255, 0.1) 0%, rgba(200, 200, 200, 0.05) 28%, transparent 46%), linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 58%, #333333 100%); padding: 20px; color: white;">
    <h3 style="margin: 0; color: #ffffff; border: none; font-size: 18px; font-weight: 900; display: flex; align-items: center; gap: 10px;">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#fff"><path d="M12 1.75L5.75 12.25L12 16L18.25 12.25L12 1.75M5.75 13.5L12 17.25L18.25 13.5L12 22.25L5.75 13.5Z"/></svg>
      Support the Project
    </h3>
  </div>
  <div style="padding: 24px; text-align: center; background: #050505;">
    <p style="margin: 0 0 16px 0; font-size: 14px; color: #ccc; line-height: 1.6;">
      A coffee in Ethereum would be cool! Although I don't drink coffee—I think it tastes like burnt water—but a pink lemonade would be fire! 🔥
    </p>
    <a href="https://etherscan.io/address/0xDEE7fa8C421BD038D32e4441ea1aDe72fE973706" target="_blank" style="display: inline-block; background: #000; border: 1px solid #666; border-radius: 12px; padding: 12px 24px; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #fff; letter-spacing: 0.5px; text-decoration: underline; text-underline-offset: 4px; transition: all 0.3s ease;">
      0xDEE7fa8C421BD038D32e4441ea1aDe72fE973706
    </a>
  </div>
</div>

### recommended sampling parameters:
``` bash
--temp 0.85 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.1 --presence-penalty 0.0
```

---
library_name: transformers
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- qwen3_5_moe
- qwen3_5
- reasoning
- agentic-coding
- mtp
- apex
- quantization
- gguf
- multimodal
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
---
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin-bottom: 24px;">
<div style="background: #f5f5f7; border-radius: 20px; padding: 36px 32px; margin-bottom: 20px; text-align: center; position: relative; overflow: hidden;">
<div style="position: absolute; top: -30px; right: -30px; width: 120px; height: 120px; background: #dbeafe; border-radius: 50%;"></div>
<div style="position: absolute; bottom: -20px; left: 40px; width: 80px; height: 80px; background: #bfdbfe; border-radius: 50%;"></div>
<div style="position: absolute; top: 50%; left: -15px; width: 60px; height: 60px; background: #dbeafe; border-radius: 50%;"></div>
<div style="display: inline-flex; gap: 8px; margin-bottom: 16px; position: relative; z-index: 1;">
<span style="background: #007aff; color: white; font-size: 11px; font-weight: 600; padding: 5px 14px; border-radius: 20px;">APEX</span>
<span style="background: #af52de; color: white; font-size: 11px; font-weight: 600; padding: 5px 14px; border-radius: 20px;">Native MTP</span>
<span style="background: #30d158; color: white; font-size: 11px; font-weight: 600; padding: 5px 14px; border-radius: 20px;">Vision</span>
<span style="background: #ff9500; color: white; font-size: 11px; font-weight: 600; padding: 5px 14px; border-radius: 20px;">MIT</span>
</div>
<h1 style="margin: 0 0 8px 0; font-size: 32px; font-weight: 700; color: #1d1d1f; letter-spacing: -0.5px; border: none; position: relative; z-index: 1;">Ornith-1.5-35B-A3B-MTP-APEX</h1>
<p style="margin: 8px 0 0 0; font-size: 14px; position: relative; z-index: 1;"><span style="color: #86868b;">English</span> | <a href="https://huggingface.co/SC117/Ornith-1.5-35B-A3B-MTP-APEX-GGUF/blob/main/README_zh.md" style="color: #007aff; text-decoration: none;">📖 中文文档</a></p>
<p style="margin: 0; font-size: 15px; color: #86868b; position: relative; z-index: 1;">Self-improving agentic coding model · APEX quantized GGUFs + BF16 + mmproj</p>
</div>
</div>

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; flex-direction: column; gap: 20px; margin-bottom: 30px;">

<div style="border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
<div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px;"><span>🐦</span> About Ornith</div>
<div style="padding: 16px; font-size: 13px; color: #334155; line-height: 1.7;">
<p style="margin: 0 0 12px 0;"><a href="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B" target="_blank" style="color: #c2410c; text-decoration: none; font-weight: 700;">Ornith-1.5-35B-A3B</a> is a self-improving agentic coding model from the <a href="https://ornith.ai/ornith_1_5.html" target="_blank" style="color: #c2410c; text-decoration: none; font-weight: 700;">Ornith Team</a>. It extends Ornith-1.0 by expanding the self-improvement loop from scaffold and rollout optimization to jointly optimizing task generation, scaffold construction, and solution rollouts — continuously generating new training tasks, discovering effective strategies, and improving the policy through reinforcement learning.</p>
<p style="margin: 0 0 12px 0;">Activating only ~3B parameters per token, it significantly outperforms its similar-sized peer Qwen3.6-35B across all coding and agentic benchmarks: Terminal-Bench 2.1 <b>67.8</b>, SWE-bench Verified <b>79</b>, SWE-bench Pro <b>59.6</b>, SWE-bench Multilingual <b>71.4</b>, NL2Repo <b>46.2</b>, MCP-Atlas <b>70.2</b>, ClawEval <b>72.5</b>.</p>
<p style="margin: 0;">This GGUF package includes the <b>mmproj-BF16.gguf</b> vision projector for multimodal (image + text) capabilities with llama.cpp. Unlike Ornith-1.0, the MTP layer is <b>native to the model</b> — no external grafting required. <b>License: MIT.</b></p>
</div>
</div>

<div style="border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
<div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px;"><span>🧠</span> Model Details</div>
<div style="padding: 16px;">
<table style="width: 100%; border-collapse: collapse; font-size: 13px;"><tbody>
<tr><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); font-weight: bold; color: #334155; background: white;">Architecture</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">Qwen3.5 MoE (Mixture of Experts)</td></tr>
<tr><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); font-weight: bold; color: #334155; background: white;">Parameters</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">35B total, 3B active per token</td></tr>
<tr><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); font-weight: bold; color: #334155; background: white;">Experts</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">256 routed experts, 8 active per token</td></tr>
<tr><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); font-weight: bold; color: #334155; background: white;">Layers</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">40 transformer layers + 1 MTP layer</td></tr>
<tr><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); font-weight: bold; color: #334155; background: white;">Context</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">262,144 tokens</td></tr>
<tr><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); font-weight: bold; color: #334155; background: white;">MTP</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">1 native MTP layer (785 tensors)</td></tr>
<tr><td style="padding: 6px 10px; font-weight: bold; color: #334155; background: white;">License</td><td style="padding: 6px 10px; color: #4b5563; background: white;">MIT</td></tr>
</tbody></table>
</div>
</div>

<div style="border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
<div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px;"><span>🚀</span> Usage</div>
<div style="padding: 16px; font-size: 13px; color: #334155; line-height: 1.7;">
<p style="margin: 0 0 8px 0; font-weight: bold; color: #1e293b;">llama.cpp (text only)</p>
<p style="margin: 0; font-family: monospace; background: #f8fafc; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 12px; color: #1e293b; white-space: pre-wrap;">hf download SC117/Ornith-1.5-35B-A3B-MTP-APEX-GGUF --include "*.gguf" --local-dir ./models
./llama-server -m ./models/Ornith-1.5-35B-A3B-MTP-APEX-I-Compact.gguf -ngl 99 -c 131072</p>
<p style="margin: 12px 0 8px 0; font-weight: bold; color: #1e293b;">llama.cpp (vision + text)</p>
<p style="margin: 0; font-family: monospace; background: #f8fafc; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 12px; color: #1e293b; white-space: pre-wrap;">./llama-server -m ./models/Ornith-1.5-35B-A3B-MTP-APEX-I-Compact.gguf --mmproj ./models/mmproj-BF16.gguf -ngl 99 -c 131072</p>
</div>
</div>

<div style="border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
<div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px;"><span>🎛️</span> Recommended Settings</div>
<div style="padding: 16px;">
<table style="width: 100%; border-collapse: collapse; font-size: 13px;"><thead><tr style="background: rgba(255,107,53,0.05);"><th style="padding: 8px 10px; border-bottom: 2px solid #ff6b35; text-align: left; color: #c2410c; font-weight: bold; width: 30%;">Mode</th><th style="padding: 8px 10px; border-bottom: 2px solid #ff6b35; text-align: left; color: #c2410c; font-weight: bold; width: 70%;">Parameters</th></tr></thead><tbody>
<tr><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); font-weight: bold; color: #334155; background: white;">General</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white; font-family: monospace;">temperature=0.6, top_p=0.95, top_k=20</td></tr>
<tr><td style="padding: 6px 10px; font-weight: bold; color: #334155; background: white;">Coding</td><td style="padding: 6px 10px; color: #4b5563; font-family: monospace; background: white;">temperature=0.6, top_p=0.95, top_k=20</td></tr>
</tbody></table>
</div>
</div>

<div style="border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
<div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px;"><span>💡</span> What is APEX?</div>
<div style="padding: 16px;">
<p style="margin: 0 0 12px 0; font-size: 13px; color: #334155; line-height: 1.7;">These GGUF files are quantized using <a href="https://github.com/mudler/apex-quant" target="_blank" style="color: #c2410c; text-decoration: none; font-weight: 700;">APEX</a>, an MoE-aware mixed-precision quantization technique. APEX classifies every tensor by its role — routed expert, shared expert, or attention — and applies a layer-wise precision gradient, giving sensitive edge layers higher precision and compressing redundant middle layers more aggressively.</p>
<p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7; font-weight: 700;">APEX beats Q8_0 perplexity at half the size — and even beats F16.</p>
</div>
</div>

<div style="border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
<div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px;"><span>📦</span> APEX Quantization Tiers</div>
<div style="padding: 16px;">
<table style="width: 100%; border-collapse: collapse; font-size: 13px;"><thead><tr style="background: rgba(255,107,53,0.05);"><th style="padding: 8px 10px; border-bottom: 2px solid #ff6b35; text-align: left; color: #c2410c; font-weight: bold; width: 35%;">File</th><th style="padding: 8px 10px; border-bottom: 2px solid #ff6b35; text-align: left; color: #c2410c; font-weight: bold; width: 15%;">Size</th><th style="padding: 8px 10px; border-bottom: 2px solid #ff6b35; text-align: left; color: #c2410c; font-weight: bold; width: 15%;">Profile</th><th style="padding: 8px 10px; border-bottom: 2px solid #ff6b35; text-align: left; color: #c2410c; font-weight: bold; width: 35%;">Best For</th></tr></thead><tbody>
<tr><td style="padding: 8px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;"><code>*-APEX-I-Quality.gguf</code></td><td style="padding: 8px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;">22.09 GB</td><td style="padding: 8px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;">I-Quality</td><td style="padding: 8px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;">Highest quality, best accuracy</td></tr>
<tr><td style="padding: 8px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;"><code>*-APEX-I-Balanced.gguf</code></td><td style="padding: 8px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;">24.43 GB</td><td style="padding: 8px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;">I-Balanced</td><td style="padding: 8px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;">Best all-rounder, recommended</td></tr>
<tr><td style="padding: 8px 10px; color: #334155;"><code>*-APEX-I-Compact.gguf</code></td><td style="padding: 8px 10px; color: #334155;">16.24 GB</td><td style="padding: 8px 10px; color: #334155;">I-Compact</td><td style="padding: 8px 10px; color: #334155;">Best quality/size ratio</td></tr>
<tr><td style="padding: 8px 10px; color: #334155;"><code>*-APEX-I-Mini.gguf</code></td><td style="padding: 8px 10px; color: #334155;">13.38 GB</td><td style="padding: 8px 10px; color: #334155;">I-Mini</td><td style="padding: 8px 10px; color: #334155;">Most compact, fits in 16GB VRAM</td></tr>
</tbody></table>
</div>
</div>

<div style="border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
<div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px;"><span>❓</span> FAQ: Why is I-Balanced larger than I-Quality?</div>
<div style="padding: 16px; font-size: 13px; color: #334155; line-height: 1.7;">
<p style="margin: 0 0 12px 0;"><b>Short answer: the tiers are bit-allocation strategies, not a size ladder.</b> APEX assigns precision per tensor from measured importance, so file size does not grow monotonically with the tier name.</p>
<p style="margin: 0 0 12px 0;"><b>I-Quality</b> keeps every sensitive tensor high-precision — attention at Q6_K in all 40 blocks, shared experts at Q8_0, edge blocks high as well — but compresses the redundant middle routed experts (blk.10–29) aggressively to <b>IQ4_XS</b>. Routed experts carry most of an MoE's parameters, and the importance analysis shows the middle blocks absorb this compression with virtually no measurable loss. That is where the bytes are saved.</p>
<p style="margin: 0 0 12px 0;"><b>I-Balanced</b> is the conservative, uniform profile: no tensor anywhere below Q5_K. Uniformity simply costs more bytes.</p>
<p style="margin: 0 0 12px 0;">So bigger does not mean better. I-Quality is the smarter bit allocation and remains the highest-quality preset despite being ~2.3 GB smaller; if you need a smaller file, step down to I-Compact or I-Mini instead of picking by file size.</p>
<table style="width: 100%; border-collapse: collapse; font-size: 13px;"><thead><tr style="background: rgba(255,107,53,0.05);"><th style="padding: 8px 10px; border-bottom: 2px solid #ff6b35; text-align: left; color: #c2410c; font-weight: bold;">Tensor group</th><th style="padding: 8px 10px; border-bottom: 2px solid #ff6b35; text-align: left; color: #c2410c; font-weight: bold;">I-Quality</th><th style="padding: 8px 10px; border-bottom: 2px solid #ff6b35; text-align: left; color: #c2410c; font-weight: bold;">I-Balanced</th></tr></thead><tbody>
<tr><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;">Routed experts · edge blocks (0–4, 35–39)</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">Q6_K</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">Q6_K</td></tr>
<tr><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #334155; background: white;">Routed experts · blk.5–9, 31–34</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">Q5_K</td><td style="padding: 6px 10px; box-shadow: 0 1px 0 0 rgba(128,128,128,0.15); color: #4b5563; background: white;">Q5_K</td></tr>
<tr><td style="padding: 6px 10px; color: #334155; background: white;"><b>Routed experts · middle blocks (10–29)</b></td><td style="padding: 6px 10px; color: #334155; background: white;"><b>IQ4_XS</b></td><td style="padding: 6px 10px; color: #334155; background: white;"><b>Q5_K</b></td></tr>
<tr><td style="padding: 6px 10px; color: #334155; background: white;">Shared experts · all blocks</td><td style="padding: 6px 10px; color: #4b5563; background: white;">Q8_0</td><td style="padding: 6px 10px; color: #4b5563; background: white;">Q8_0</td></tr>
<tr><td style="padding: 6px 10px; color: #334155; background: white;">Attention · all 40 blocks</td><td style="padding: 6px 10px; color: #4b5563; background: white;">Q6_K</td><td style="padding: 6px 10px; color: #4b5563; background: white;">Q6_K</td></tr>
</tbody></table>
<p style="margin: 12px 0 0 0; font-size: 12px; color: #64748b; font-style: italic;">The two tiers are identical everywhere except the middle routed-expert band — that single band is the whole size difference.</p>
</div>
</div>

</div>

## Links

- **Original Model**: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B
- **Ornith Blog**: https://ornith.ai/ornith_1_5.html
- **APEX Quantization**: https://github.com/mudler/apex-quant

## Citation

```bibtex
@misc{ornith_1_5,
    title = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
    url = {https://ornith.ai/ornith_1_5.html},
    author = {{Ornith Team}},
    year = {2026}
}
```

<!-- SC117_SUPPORT_START -->
<div align="center">
  <a href="https://ko-fi.com/sc117">
    <img
      src="https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/resolve/main/assets/support-sc117.png"
      alt="Support SC117 on Ko-fi — support is optional and all releases remain freely available."
      width="100%"
    />
  </a>
</div>
<!-- SC117_SUPPORT_END -->

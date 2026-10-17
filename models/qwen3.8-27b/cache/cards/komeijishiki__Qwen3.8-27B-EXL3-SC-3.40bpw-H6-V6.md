---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - qwen3.8
  - exl3
  - quantized
  - multimodal
  - vision
library_name: exllamav3
---

<div style="background:#14212f; border-radius:22px; padding:34px 30px; color:#c8d2dc; line-height:1.75; font-size:15px;">

<div style="background:#0e1a27; border-radius:16px; padding:48px 38px 40px; position:relative; overflow:hidden; margin-bottom:6px;">

<div style="position:absolute; left:-18%; top:-45%; width:95%; height:170%; border:1px solid rgba(242,237,228,0.14); border-radius:50%; transform:rotate(22deg);"></div>
<div style="position:absolute; left:-2%; top:-12%; width:72%; height:130%; border:1px solid rgba(224,86,63,0.30); border-radius:50%; transform:rotate(-16deg);"></div>
<div style="position:absolute; right:11%; top:20%; width:10px; height:10px; background:#d2402e; border-radius:50%;"></div>
<div style="position:absolute; right:14.5%; top:57%; width:6px; height:6px; background:rgba(242,237,228,0.6); border-radius:50%;"></div>

<div style="position:absolute; left:9%; top:14%; width:34px; height:34px;">
<div style="position:absolute; left:50%; top:0; width:1px; height:100%; background:rgba(242,237,228,0.55);"></div>
<div style="position:absolute; top:50%; left:0; width:100%; height:1px; background:rgba(242,237,228,0.55);"></div>
</div>
<div style="position:absolute; right:26%; top:10%; width:22px; height:22px;">
<div style="position:absolute; left:50%; top:0; width:1px; height:100%; background:rgba(224,86,63,0.55);"></div>
<div style="position:absolute; top:50%; left:0; width:100%; height:1px; background:rgba(224,86,63,0.55);"></div>
</div>

<div style="position:absolute; left:8%; top:49%; color:rgba(242,237,228,0.42); font-size:11px; letter-spacing:7px; white-space:nowrap;">5 5 6 2 &nbsp;&nbsp;|&nbsp;&nbsp; 1 1 6 2</div>

<div style="position:relative; z-index:1;">
<div style="font-size:12px; letter-spacing:9px; color:rgba(242,237,228,0.72);">EXLLAMAV3 · FULL SC</div>
<div style="font-size:34px; font-weight:700; letter-spacing:5px; margin:14px 0 8px; color:#f5f0e6;">QWEN3.8-27B</div>
<div style="font-size:14px; letter-spacing:3px; color:rgba(242,237,228,0.88);">TEXT 3.40 BPW&nbsp;&nbsp;/&nbsp;&nbsp;HEAD 6BIT&nbsp;&nbsp;/&nbsp;&nbsp;VISION 6BIT&nbsp;&nbsp;/&nbsp;&nbsp;MTP 4BIT</div>

<div style="margin-top:22px;">
<span style="border:1px solid rgba(242,237,228,0.5); color:#f2ede4; border-radius:999px; padding:5px 16px; font-size:12px; letter-spacing:2px; margin-right:8px; display:inline-block;">EXL3</span>
<span style="border:1px solid rgba(224,86,63,0.85); color:#e8654f; border-radius:999px; padding:5px 16px; font-size:12px; letter-spacing:2px; margin-right:8px; display:inline-block;">SC RECIPE</span>
<span style="border:1px solid rgba(242,237,228,0.5); color:#f2ede4; border-radius:999px; padding:5px 16px; font-size:12px; letter-spacing:2px; margin-right:8px; display:inline-block;">VISION 6BIT</span>
<span style="border:1px solid rgba(242,237,228,0.5); color:#f2ede4; border-radius:999px; padding:5px 16px; font-size:12px; letter-spacing:2px; display:inline-block;">MTP 4BIT</span>
</div>

</div>

</div>

<div style="border-left:3px solid #d2402e; padding-left:14px; margin:34px 0 16px 0; display:flex; align-items:baseline; gap:14px;">
<span style="color:#8b9aa8; font-size:11px; letter-spacing:6px;">01</span>
<span style="color:#f2ede4; font-size:21px; font-weight:700; letter-spacing:2px;">模型简介 <span style="color:#5f7080; font-weight:400; letter-spacing:1px;">ABOUT</span></span>
</div>

<p style="margin:8px 0;">本仓库是 <a style="color:#e8654f;" href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen/Qwen3.8-27B</a> 的 <b style="color:#f2ede4;">ExLlamaV3（EXL3）</b> 量化版本，采用 turboderp 的 <b style="color:#f2ede4;">Full SC（自校准）recipe</b> 流程生成：文本权重 <b style="color:#e8654f;">3.40 bpw</b>、lm_head 6bit、视觉部分（Vision）6bit、MTP（多 token 预测）模块 4bit。在保留 27B 级综合能力的前提下，大幅压缩显存占用，面向消费级显卡的本地部署。英文版见 <a style="color:#e8654f;" href="https://huggingface.co/komeijishiki/Qwen3.8-27B-EXL3-SC-3.40bpw-H6-V6/blob/main/README_en.md">README_en.md</a>。</p>

<div style="border-top:1px solid rgba(242,237,228,0.12); margin:30px 0;"></div>

<div style="border-left:3px solid #d2402e; padding-left:14px; margin:0 0 16px 0; display:flex; align-items:baseline; gap:14px;">
<span style="color:#8b9aa8; font-size:11px; letter-spacing:6px;">02</span>
<span style="color:#f2ede4; font-size:21px; font-weight:700; letter-spacing:2px;">量化规格 <span style="color:#5f7080; font-weight:400; letter-spacing:1px;">QUANTIZATION</span></span>
</div>

<table style="border-collapse:collapse; width:100%; font-size:14px; border:1px solid rgba(242,237,228,0.16);">
<thead>
<tr style="background:#1e3247;">
<th style="padding:10px 14px; text-align:left; letter-spacing:1px; font-size:13px; color:#f2ede4;">组件 · PART</th>
<th style="padding:10px 14px; text-align:left; letter-spacing:1px; font-size:13px; width:120px; color:#f2ede4;">比特数 · BITS</th>
<th style="padding:10px 14px; text-align:left; letter-spacing:1px; font-size:13px; color:#f2ede4;">说明 · NOTE</th>
</tr>
</thead>
<tbody>
<tr style="background:rgba(255,255,255,0.03);"><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;"><b>TEXT (DECODER)</b></td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#e8654f;"><b>3.40 BPW</b></td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">Full SC recipe，<code style="background:#0e1a27; padding:2px 6px; border-radius:6px; color:#dbe4ec;">sc_optimize -b 3.40 -hb 6 -al 2.0 -mink 2</code></td></tr>
<tr><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;"><b>HEAD (LM_HEAD)</b></td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#e8654f;"><b>6 BIT</b></td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">保持输出分布质量</td></tr>
<tr style="background:rgba(255,255,255,0.03);"><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;"><b>VISION TOWER</b></td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#e8654f;"><b>6 BIT</b></td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">视觉编码器</td></tr>
<tr><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;"><b>MTP HEAD</b></td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#e8654f;"><b>4 BIT</b></td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">多 token 预测模块</td></tr>
</tbody>
</table>

<div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:12px 18px; margin:14px 0; border-left:3px solid #d2402e; color:#9fb0bf; font-size:13px;"><b style="color:#f2ede4;">工具链 · TOOLING：</b>turboderp-org/ExLlamaV3 v1.4.4 · SC 数据源：turboderp/Qwen3.8-27B-exl3</div>

<table style="border-collapse:collapse; width:100%; font-size:14px; border:1px solid rgba(242,237,228,0.16);">
<thead>
<tr style="background:#1e3247;">
<th style="padding:10px 14px; text-align:left; letter-spacing:1px; font-size:13px; color:#f2ede4;">项目 · ITEM</th>
<th style="padding:10px 14px; text-align:left; letter-spacing:1px; font-size:13px; color:#f2ede4;">值 · VALUE</th>
</tr>
</thead>
<tbody>
<tr style="background:rgba(255,255,255,0.03);"><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">参数量 · Parameters</td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;">27B</td></tr>
<tr><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">隐藏维度 · Hidden size</td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;">5120</td></tr>
<tr style="background:rgba(255,255,255,0.03);"><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">词表 · Vocabulary</td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;">248,320（Padded）</td></tr>
<tr><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">层数 · Layers</td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;">64 · 16 × (3 × (GatedDeltaNet → FFN) → 1 × (Gated Attention → FFN))</td></tr>
<tr style="background:rgba(255,255,255,0.03);"><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">上下文 · Context</td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;">262,144 原生 · 可扩展至 1,000,000</td></tr>
<tr><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">视觉 · Vision</td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#e8654f;">原生图片 / 视频理解</td></tr>
<tr style="background:rgba(255,255,255,0.03);"><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">MTP</td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#e8654f;">启用</td></tr>
<tr><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16);">许可证 · License</td><td style="padding:10px 14px; border-top:1px solid rgba(242,237,228,0.16); color:#f2ede4;">Apache-2.0</td></tr>
</tbody>
</table>

<div style="border-top:1px solid rgba(242,237,228,0.12); margin:30px 0;"></div>

<div style="border-left:3px solid #d2402e; padding-left:14px; margin:0 0 16px 0; display:flex; align-items:baseline; gap:14px;">
<span style="color:#8b9aa8; font-size:11px; letter-spacing:6px;">03</span>
<span style="color:#f2ede4; font-size:21px; font-weight:700; letter-spacing:2px;">使用方法 <span style="color:#5f7080; font-weight:400; letter-spacing:1px;">USAGE</span></span>
</div>

<div style="margin:10px 0; color:#f2ede4;"><b>方式一：TABBYAPI（推荐）</b></div>
<ul style="padding-left:22px; margin:8px 0;">
<li style="margin:4px 0;">将本仓库下载到模型目录</li>
<li style="margin:4px 0;">TabbyAPI → Load Model → 选择本目录</li>
<li style="margin:4px 0;">首次加载建议 context = <b style="color:#e8654f;">32768</b>，确认文本 / 视觉 / MTP 正常后逐步调高</li>
</ul>

<div style="margin:14px 0 8px 0; color:#f2ede4;"><b>方式二：EXLLAMAV3 直接加载</b></div>
<pre style="background:#0e1a27; border:1px solid rgba(242,237,228,0.15); border-radius:10px; padding:14px 18px; overflow-x:auto; color:#dbe4ec; font-size:13px; line-height:1.6; font-family:Consolas, Monaco, monospace;"><code>from exllamav3 import ExLlamaV3Config, ExLlamaV3, ExLlamaV3Cache
from exllamav3.generator import ExLlamaV3StreamingGenerator

config = ExLlamaV3Config()
config.model_dir = "/path/to/Qwen3.8-27B-EXL3-SC-3.40bpw-H6-V6"
config.max_seq_len = 32768      # 建议从 32K 起步
config.build()

model = ExLlamaV3(config)
cache = ExLlamaV3Cache(model)
generator = ExLlamaV3StreamingGenerator(model, cache, tokenizer)</code></pre>

<div style="border-top:1px solid rgba(242,237,228,0.12); margin:30px 0;"></div>

<div style="border-left:3px solid #d2402e; padding-left:14px; margin:0 0 16px 0; display:flex; align-items:baseline; gap:14px;">
<span style="color:#8b9aa8; font-size:11px; letter-spacing:6px;">04</span>
<span style="color:#f2ede4; font-size:21px; font-weight:700; letter-spacing:2px;">质量说明 <span style="color:#5f7080; font-weight:400; letter-spacing:1px;">QUALITY</span></span>
</div>

<ul style="padding-left:22px; margin:8px 0;">
<li style="margin:4px 0;"><b style="color:#e8654f;">achieved_bpw = 3.40</b>，recipe 预测 KLD ≈ 0.011 ≈ IQ4_XS </li>
<li style="margin:4px 0;">逐层 SQNR 健康区间 <b style="color:#f2ede4;">39 ~ 43 dB</b>；低比特张量（gate/up proj 2~3bit）按噪声归因策略放置</li>
<li style="margin:4px 0;">vision / MTP 模块分别按 6bit / 4bit 独立转换验证</li>
</ul>

<div style="border-top:1px solid rgba(242,237,228,0.12); margin:30px 0;"></div>

<div style="border-left:3px solid #d2402e; padding-left:14px; margin:0 0 16px 0; display:flex; align-items:baseline; gap:14px;">
<span style="color:#8b9aa8; font-size:11px; letter-spacing:6px;">05</span>
<span style="color:#f2ede4; font-size:21px; font-weight:700; letter-spacing:2px;">致谢 <span style="color:#5f7080; font-weight:400; letter-spacing:1px;">CREDITS</span></span>
</div>

<ul style="padding-left:22px; margin:8px 0;">
<li style="margin:4px 0;">原模型：<a style="color:#e8654f;" href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen/Qwen3.8-27B</a>（Apache-2.0）</li>
<li style="margin:4px 0;">量化框架：<a style="color:#e8654f;" href="https://github.com/turboderp-org/ExLlamaV3">turboderp-org/ExLlamaV3</a></li>
<li style="margin:4px 0;">SC 校准数据：<a style="color:#e8654f;" href="https://huggingface.co/turboderp/Qwen3.8-27B-exl3">turboderp/Qwen3.8-27B-exl3</a></li>
</ul>

<div style="text-align:center; color:#5f7080; font-size:12px; letter-spacing:5px; padding:22px 0 4px;">EXLLAMAV3 · FULL SC · 3.40BPW · H6 · V6 · MTP4</div>

</div>

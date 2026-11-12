---
base_model: Qwen/Qwen3.8-Flash-Next
license: other
language:
- zh
- en
tags:
- technical-demonstration
- abliteration
- refusal-direction
- qwen4_exp
- research
library_name: transformers
---

# Abliteration on Qwen3.8-Flash-Next — a technical demonstration by xCloudinfo

> This is a **technical demonstration**, not a product and not a model we distribute for use.
> We built it to answer one engineering question: *can we cleanly abliterate a 125B-class model
> on a brand-new architecture that no community tooling supports yet — and honestly measure the result?*
> The answer is **yes**. The base is a Chinese-origin model (Qwen), so this work is for research and
> capability demonstration only — not a product, and not something we distribute into deployments.

## What was demonstrated

Starting from `Qwen/Qwen3.8-Flash-Next` (the `qwen4_exp` architecture — ~125B active-6B MoE plus a
51B n-gram embedding and a 4B MTP head, **360 GB in bf16**), we performed single-direction
abliteration (Arditi et al., 2024) — orthogonalizing the refusal direction out of the residual
writers, **without any retraining** — and measured the effect on a held-out set.

| Model | Severe-harm held-out refusal | Coherence |
|---|---|---|
| Base Qwen3.8-Flash-Next | **25 / 25** refused | normal |
| After abliteration (this demo) | **3 / 25** refused (88% answered) | intact; 5/5 benign prompts fluent |

Measurement is honest by construction: the held-out prompts are **disjoint** from the 24 prompts used
to extract the direction; evaluation is run with **thinking turned off** (a reasoning model will
otherwise spend its tokens inside `<think>` and leave `content` empty, which a naive detector
miscounts as compliance); the refusal detector covers **Traditional, Simplified and English** phrasings.
The three remaining "refusals" are mostly a rhetorical "I cannot stress this enough:" opener that is
then followed by compliance, so the true figure is lower. We report exactly what we measured — nothing
is quoted that was not run.

## Why this architecture is hard

- **Brand-new architecture.** `qwen4_exp` needs `transformers >= 5.22.2`; we did the whole abliteration at
  the `transformers` level. (llama.cpp does support it — `unsloth/Qwen3.8-Flash-Next-GGUF` ships a full
  quant ladder — so a GGUF build of this abliterated model is possible; see Limitations.)
- **It does not fit one machine.** 360 GB far exceeds a single 121 GB unified-memory node; the forward
  pass runs with CPU-resident weights plus disk offload.
- **A 95 GB single tensor.** The n-gram PLE embedding is one indivisible 95 GB parameter, which forces
  a careful memory layout or the loader silently offloads the whole model to disk.
- **A four-stream hyper-connection residual.** Most layers carry a 10240 = 4 × 2560 residual. This is
  the key finding below.

## The method, and the one insight that made it work

Residual writers orthogonalized: **147** tensors — `self_attn.o_proj` ×12,
`linear_attn.out_proj` ×36 (the Gated-DeltaNet / SSM path, which must be included),
fused `mlp.experts.down_proj` ×48, `mlp.shared_expert.down_proj` ×48, plus 3 in the MTP head.
The refusal direction was taken from layer 27 (f = 0.56, chosen by cross-layer cosine stability),
applied at strength 2.

The insight: **runtime activation hooks do not work on this architecture.** Ablating a layer's output
residual is undone downstream, because the four-stream hyper-connection re-mixes the streams and
reintroduces the direction from the streams that were not touched. **Persistent weight editing** — so the
direction is never written in the first place — is the correct operator. Strength 1 only softens
refusals; strength 2 is the quality-preserving sweet spot (3/25, coherent); strength 2.5 reaches 0/25 but
begins to degrade fluency and surfaces base-model biases, so we did not use it. Pushing refusal to zero
is the easy direction (just raise the strength); keeping it low **while staying coherent** is the hard
part, and that is what the strength-2 point shows.

## Scope and boundary

- This demonstrates **reduced over-refusal** via a published, measurable technique. It was **not** trained
  to stably produce severe-harm operational content, and single-direction orthogonalization does not
  guarantee full removal.
- The base model is **Chinese-origin (Qwen)**; this is a research/demonstration artifact only — not a
  product and not something we distribute into deployments — and it inherits the base model's behaviours and
  biases.
- This repo provides a **GGUF** ladder (BF16 / Q6_K / Q4_K_M / Q3_K_M / Q2_K), converted with
  **unsloth's llama.cpp fork** (`conversion/qwen4exp.py` — the tool that supports `qwen4_exp`; upstream
  llama.cpp does not yet). Run with that llama.cpp fork, Strata, or Unsloth Desktop. imatrix and mmproj
  (vision, unchanged from base) can follow.
- **Which tier to use.** The held-out figure above was measured at the **Q4_K_M tier and higher**, where it
  holds. The low-bit **Q3_K_M / Q2_K** tiers stay coherent on ordinary tasks but can show **weaker
  decensoring** — refusals partially return (observed at Q2_K) — so use **Q4_K_M or above** where the
  measured behaviour matters. We report only what we ran.

## Data handling and network provenance

A deliberate point of integrity: the base model is Chinese-origin, but **no traffic went to Chinese
infrastructure at any point** of this work. Verified on the machine:

- The 360 GB weights were pulled **from Hugging Face's own servers** — `cas-server.xethub.hf.co` and the
  `cdn-lfs-us-1.hf.co` CDN, which resolves to **AWS CloudFront (US)** (`54.192.248.24`); `huggingface.co`
  resolves to AWS US IPv6. No `HF_ENDPOINT` override, i.e. **no Chinese mirror** (no `hf-mirror.com`).
- Python packages came from the **official PyPI / PyTorch** indexes — no Chinese pip mirror configured.
- No connection to ModelScope, Aliyun, any `.cn` host, or any Chinese mirror was made.
- The download was **anonymous (no token)**, so not even "who is downloading" left the network.
- All compute (direction extraction, weight editing, evaluation) and the resulting weights stayed on the
  **internal network**; nothing was uploaded off-site during the work.

Using a Chinese-made model for research is not the same as sending packets to China — and here the latter
simply did not happen.

## Attribution

Base model: `Qwen/Qwen3.8-Flash-Next`. This demonstration honours the base model's license and terms.
Method: Arditi et al., "Refusal in LLMs is mediated by a single direction" (2024).
Demonstration by **xCloudinfo（云碩科技）**.

---

## 繁體中文說明

這是 **xCloudinfo（云碩科技）的技術展現，不是產品、也不是我們拿來散布使用的模型**。做它只為回答一個
工程問題：一顆社群工具目前完全不支援的全新架構、360GB 的 125B 級模型，我們能不能乾淨地做出
abliteration，並且誠實把結果量出來。答案是**能**。底模是中國來源（Qwen），所以這是研究與能力展示用途，
**不是產品、也不散布進專案部署**。

**展示結果**：原版對 25 題嚴重危害 held-out **全部拒答（25/25）**；經單方向 abliteration（不重訓）後
**只剩 3/25 拒答、88% 照答，且回應連貫、良性題 5/5 正常**。量測方法誠實：held-out 題與抽方向的 24 題
不相交、評測時關閉 thinking、拒絕詞偵測含繁簡英；那 3 個被記為拒答者多半是「I cannot stress this
enough:」修辭開頭後接著照答，真實拒答更低。標多少就是實測多少。

**為什麼難**：`qwen4_exp` 是全新架構，要 transformers 5.22 以上才載得起來，我們整套在 transformers 層做；
360GB 單機放不下；95GB 的 n-gram 嵌入是單一不可切張量；多數層是 10240=4×2560 的四流 hyper-connection 殘差。
（llama.cpp 其實支援這架構——`unsloth/Qwen3.8-Flash-Next-GGUF` 有完整量化階梯——所以本模型也能轉 GGUF。）

**關鍵洞察**：runtime hook 在這顆無效——四流重混合會把消掉的方向還原回來；**從寫入者持久移除（weight-edit）
才是正解**。強度 1 只軟化、強度 2 是保品質的甜蜜點、強度 2.5 雖達 0/25 但開始掉品質，所以不採用。把拒答
壓到 0 是最容易的方向，難的是**壓低的同時保住連貫**——這才是強度 2 要展示的。

**邊界**：只展示「降低過度拒絕」這個可量測的技術，未、也不會為了讓最嚴重類別穩定輸出而重訓；中國底模，
研究展示用，繼承底模的行為與偏誤；本 repo 提供 **GGUF** 五階（BF16／Q6_K／Q4_K_M／Q3_K_M／Q2_K，用
**unsloth 的 llama.cpp fork** 轉，`conversion/qwen4exp.py`——支援 qwen4_exp，上游尚未；用該 fork／Strata／Unsloth Desktop 跑），imatrix、mmproj 後續補。

**要用哪一階**：上表的實測數字是在 **Q4_K_M 以上**量的、在那保持；低位階 **Q3_K_M／Q2_K** 一般任務仍連貫，
但去審查會變弱——拒答會部分漏回來（Q2_K 實測到）——要用到實測那個開放度就用 **Q4_K_M 或更高**。只報跑過的。

**資料與網路來源（實機驗證，一個封包都沒往對岸）**：底模是中國做的，但**全程沒有任何流量連到中國**。
360GB 權重是**從 Hugging Face 自己的伺服器**下載（`cas-server.xethub.hf.co`、CDN `cdn-lfs-us-1.hf.co`
解析到 **AWS CloudFront 美國** `54.192.248.24`；`huggingface.co` 解析到 AWS 美國 IPv6）——**無 `HF_ENDPOINT`
鏡像設定、沒有 hf-mirror.com 這類中國鏡像**；Python 套件走**官方 PyPI／PyTorch**、無中國 pip 鏡像；
**完全沒有連過 ModelScope／aliyun／任何 .cn 站點**；下載**匿名、無 token**（連「誰在下載」都沒送出）；
所有運算與產出的權重**全留在內網**，過程中沒有外推。用中國人做的模型做研究，跟「封包送去中國」是兩回事——
後者在這裡沒有發生。

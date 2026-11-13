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

The base is a Chinese-origin model, so a fair reviewer asks where our data went. We want to be precise
about what we claim. The weak claim is "we observed no packet to China" — but observation can never
exhaustively rule out DoH, IPv6, or a covert channel, so on its own it shows *absence of observed*
exfiltration, not *verified* containment. We therefore report two separate things: what we observed and
configured for this run, and a reproducible method that makes containment true **by construction**.

**Observed / configured for this run** (presented as observation, not as exhaustive proof):

- The weights were pulled **from Hugging Face's own servers** (`cas-server.xethub.hf.co`, the
  `cdn-lfs-*.hf.co` CDN) which resolve to **AWS CloudFront (US)**. No `HF_ENDPOINT` override — **no Chinese
  mirror** (`hf-mirror.com`), no ModelScope, no Aliyun, no `.cn` host configured or observed.
- Python packages came from the **official PyPI / PyTorch** indexes — no Chinese pip mirror.
- The download was **anonymous (no token)**; compute and the resulting weights stayed on the **internal
  network**; nothing was uploaded off-site during the work.

**Egress is a single logging chokepoint.** Our on-prem network reaches the internet through one logging
firewall that records every session's destination and **geo-region** and does **no SSL decryption** of
outbound traffic — it authoritatively answers *who / which country* via SNI + destination, without
MITM-ing our own payloads. Retained logs cover the work window and can be produced de-identified on request.

**The by-construction method (the real deliverable).** For sensitive provenance work we run the
download / edit / convert steps inside a network-namespace **egress jail**: a single logged exit with a
**default-deny allowlist** (DNS to one pinned resolver + the specific CDN range needed; everything else
dropped and logged). A self-test from inside the jail proves the ruler has no blind spot — HF and DNS
pass, while a `.cn` host, **DoH to the pinned resolver's 443**, a non-allowlisted HTTPS host, and **IPv6**
are all dropped and logged. DoH and IPv6 are covered for free, and the allowlist *is* the complete set of
reachable destinations: containment is true by construction, not by observation.

**Straight answer.** This model was built before we started using the jail, so I won't pretend that run
was sealed on every path — what we saw on the wire is what's above, nothing more. Worth adding, though:
we've kept the base model running in our own machine room for about two months, behind this same logging
firewall the whole time. Serving a model for inference doesn't call anywhere — it's a local file answering
prompts — and when we look at that box's outbound traffic it's operating-system updates and nothing to
China. The thing that actually lets me stand behind "every egress path is closed" is the jail, and
sensitive work goes through it from here on.

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

**邊界**：這只展示「降低過度拒絕」這一個可量測的技術；我們沒有、也不會為了讓最嚴重的類別穩定照答而重訓模型。
底模是中國做的，僅供研究與能力展示，也承襲了原模型的行為與偏誤。本 repo 提供 **GGUF** 五階（BF16／Q6_K／Q4_K_M／Q3_K_M／Q2_K，用
**unsloth 的 llama.cpp fork** 轉，`conversion/qwen4exp.py`——支援 qwen4_exp，上游尚未；用該 fork／Strata／Unsloth Desktop 跑），imatrix、mmproj 後續補。

**要用哪一階**：上表的實測數字是在 **Q4_K_M 以上**量的、在那保持；低位階 **Q3_K_M／Q2_K** 一般任務仍連貫，
但去審查會變弱——拒答會部分漏回來（Q2_K 實測到）——要用到實測那個開放度就用 **Q4_K_M 或更高**。只報跑過的。

**資料與網路來源**：底模是中國做的，所以合理的審查者會問我們的資料去了哪。我們把話說精準。弱的說法是
「我們沒觀察到任何封包連到中國」——但光靠觀察永遠無法窮舉 DoH、IPv6 或隱蔽通道，所以它只證明「沒**觀察到**
外傳」，不是「已**驗證**沒有外傳」。因此我們分開報兩件事：這次實際觀察到／設定了什麼，以及一套讓「不外傳」
**依建構為真**的可重現方法。

**這次觀察到／設定的**（當作觀察、不當作窮舉證明）：權重是**從 Hugging Face 自己的伺服器**
（`cas-server.xethub.hf.co`、`cdn-lfs-*.hf.co` CDN，解析到 **AWS CloudFront 美國**）下載；**無 `HF_ENDPOINT`
鏡像**、沒有 hf-mirror.com／ModelScope／aliyun／任何 .cn 站點；Python 套件走**官方 PyPI／PyTorch**；下載
**匿名、無 token**；運算與產出的權重**全留在內網**、過程中沒有外推。

**出口是單一記錄受控點**：地端網路經**一台會記錄的防火牆**對外，逐筆記下目的地與**目的國別**，且對出站流量
**不做 SSL 解密**——用 SNI＋目的地權威回答「跟誰／哪一國」，不對自己的 payload 做 MITM。日誌回溯涵蓋工作
窗口，可去識別化提供。

**依建構的方法（這才是真正的產出）**：敏感的來源驗證工作，我們把下載／編輯／轉檔包進一個 network namespace
**出站牢籠**——單一記錄出口＋**預設全擋白名單**（DNS 只到一個釘死的 resolver＋當次需要的 CDN 範圍，其餘一律
drop＋記錄）。從牢籠內的自我驗證證明這把尺沒有盲點：HF 與 DNS 通，而 `.cn` 主機、**對釘死 resolver 443 的
DoH**、非白名單 HTTPS 主機、**IPv6** 全部被擋並記錄。DoH 與 IPv6 因此免費涵蓋，白名單**就是**可達目的地的
全集——「不外傳」依建構為真，不靠觀察。

**說實在的**：這顆是我們開始用牢籠之前做的，所以我不會硬拗那次已經把每條路都堵死——當時在線路上看到的
就是上面那些，如此而已。不過有件事該講：這顆的底模，我們自己在機房裡放著跑了快兩個月，從頭到尾都在同一台
會記錄的防火牆後面。模型拿來推論本來就不會自己對外連線——它就是個在本機回答問題的檔案；真的去翻那台機器的
對外流量，也只有系統更新那類，沒有一筆連到中國。真正能讓我拍胸脯說「所有外傳途徑都堵死」的，是現在這套牢籠，
機敏的東西從今以後都走它。

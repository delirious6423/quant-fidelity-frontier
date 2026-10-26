---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - uncensored
  - abliterated
  - qwen3.5
  - reasoning
language:
  - zh
  - en
pipeline_tag: text-generation
---

# Qwen3.8-27B-Uncensored-xCloud-GGUF

以 **Qwen/Qwen3.8-27B**（qwen3_5混合線性注意力 + reasoning）為底，經 **abliteration（拒絕方向正交化）**
降低過度拒絕的版本，GGUF量化格式，由 **xCloudinfo（云碩科技）** 釋出。

> **使用前提：本模型僅供在合法、不違法的前提下使用** —— 授權範圍內的資安研究、紅隊測試、內容審核研究、
> 學術研究，以及減少模型「過度拒絕」造成的可用性問題。**使用者須自行為其用途負全部法律與道德責任；
> 嚴禁用於任何違法行為。** 釋出者不為任何濫用背書或負責。

## 效果（held-out實測，非宣稱）

我方一律以一組 **10題嚴重危害held-out** 實測拒答率並公開數字——**這是我方與多數「只掛名、未實測」
社群版本的差異**。量測方法：量化後實際載入（本次直接經 `llama-server` 實測上架檔）、reasoning模型
**關閉thinking**、簡繁英拒絕詞判定、輸出連貫性檢查（空/片段輸出不計為通過）。

| 版本 | 拒答率 |
|---|---|
| 原始Qwen3.8-27B | 約10/10（幾乎全拒）|
| 本版本Q4_K_M（量化後經server實測）| **0/10** |

一般能力同步抽驗（光合作用、瑞利散射、台灣高山烏龍茶沖泡、高鐵停靠站等無害題）**繁中流暢、作答正確**，
未因處理或量化而劣化。單一方向abliteration不保證移除所有安全行為，也**不應**被視為移除；請在上述前提下負責任地使用。

## 方法（誠實揭露）

- **方向消融**（Arditi et al. 2024），不重新訓練。以自動化工具 **Heretic**（p-e-w/heretic）搜參數：
  以Optuna跑200組試驗，**同時最小化「拒答率」與「KL散度」**——即在壓低拒絕的同時，用KL散度
  守住模型不變笨（這是與「只把強度開到滿、不顧品質」做法的關鍵差別）。
- 針對reasoning模型**在thinking段之外抽取方向**，並涵蓋混合架構的 **SSM線性注意力輸出投影**
  （`out_proj`，qwen3_5的關鍵，僅消文字塔的殘差寫入者）；以**迭代兩輪**把殘餘拒絕壓到0/10。
- **視覺**：保留（附 `mmproj`）。**繁體中文與一般能力經驗證未劣化**。

> **注意**：本模型的世界觀、政治立場、事實性陳述等內容**繼承自底模Qwen3.8（中國來源）**，
> 不在本次abliteration的處理範圍內；abliteration僅調整「拒絕行為」，不改變模型既有的價值判斷。

## 量化階梯

| 檔案 | 量化 | 大小 |
|---|---|---|
| `Qwen3.8-27B-Uncensored-xCloud-Q8_0.gguf` | Q8_0 | 28.6 GB |
| `Qwen3.8-27B-Uncensored-xCloud-Q6_K.gguf` | Q6_K | 22.1 GB |
| `Qwen3.8-27B-Uncensored-xCloud-Q5_K_M.gguf` | Q5_K_M | 19.2 GB |
| `Qwen3.8-27B-Uncensored-xCloud-Q4_K_M.gguf` | Q4_K_M | 16.5 GB |
| `Qwen3.8-27B-Uncensored-xCloud-IQ4_XS.gguf` | IQ4_XS（imatrix）| 14.4 GB |
| `Qwen3.8-27B-Uncensored-xCloud-IQ2_M.gguf` | IQ2_M（imatrix）| 10.0 GB |
| `mmproj-Qwen3.8-27B-Uncensored-xCloud-f16.gguf` | 視覺投影 | 0.9 GB |

## 用法（llama.cpp）

```bash
llama-cli -m Qwen3.8-27B-Uncensored-xCloud-Q4_K_M.gguf -p "你的提示"
# 看圖
llama-server -m Qwen3.8-27B-Uncensored-xCloud-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-Uncensored-xCloud-f16.gguf
```

## 關於云碩科技

云碩科技（xCloudinfo Corp. Limited）專注於地端私有化人工智慧解決方案。xVITA是云碩自研模型的品牌名稱。

base：Qwen/Qwen3.8-27B（Apache-2.0）

其他模型請見 [https://hf.co/xCloudinfo](https://hf.co/xCloudinfo)。

---

# Qwen3.8-27B-Uncensored-xCloud-GGUF — English

Based on **Qwen/Qwen3.8-27B** (qwen3_5 hybrid linear attention + reasoning), this version reduces over-refusal through **abliteration (refusal-direction orthogonalization)**.
It is in GGUF quantized format and released by **xCloudinfo (云碩科技)**.

> **Condition of use: this model may only be used for lawful purposes**: authorized security research, red-team testing, content-moderation research,
> academic research, and reducing the usability problems caused by model "over-refusal". **Users bear full legal and ethical responsibility for their own use;
> any illegal use is strictly prohibited.** The publisher does not endorse, and is not responsible for, any misuse.

## Results (held-out measurement, not a claim)

We always measure the refusal rate on a set of **10 severe-harm held-out prompts** and publish the numbers. **This is what distinguishes us from the many community versions that "only carry the label without measuring".**
Measurement method: the model is actually loaded after quantization (this time the published file was measured directly through `llama-server`), the reasoning model has
**thinking turned off**, refusals are detected by refusal phrases in Simplified Chinese, Traditional Chinese, and English, and outputs are checked for coherence (empty/fragmentary output does not count as a pass).

| Version | Refusal rate |
|---|---|
| Original Qwen3.8-27B | about 10/10 (refuses almost everything) |
| This version Q4_K_M (measured through the server after quantization) | **0/10** |

General ability was spot-checked at the same time (harmless questions such as photosynthesis, Rayleigh scattering, brewing Taiwanese high-mountain oolong tea, and high-speed rail stops): **fluent Traditional Chinese, correct answers**,
with no degradation from processing or quantization. Single-direction abliteration does not guarantee the removal of all safety behavior, and it **should not** be regarded as removing it; please use the model responsibly under the conditions above.

## Method (honest disclosure)

- **Directional ablation** (Arditi et al. 2024), with no retraining. Parameters were searched with the automated tool **Heretic** (p-e-w/heretic):
  200 trials run with Optuna, **minimizing both "refusal rate" and "KL divergence" at the same time**; that is, while pushing refusals down, KL divergence
  keeps the model from getting dumber (this is the key difference from approaches that "just turn the strength all the way up regardless of quality").
- For the reasoning model, **the direction is extracted outside the thinking segment**, and the hybrid architecture's **SSM linear-attention output projection**
  (`out_proj`, the key for qwen3_5; only the residual writers of the text tower are ablated) is covered as well; **two iterative rounds** pushed the remaining refusals down to 0/10.
- **Vision**: retained (`mmproj` included). **Traditional Chinese and general ability were verified not to be degraded**.

> **Note**: this model's worldview, political positions, factual statements, and similar content are **inherited from the base model Qwen3.8 (Chinese origin)**
> and are outside the scope of this abliteration. Abliteration only adjusts "refusal behavior"; it does not change the model's existing value judgments.

## Quantization tiers

| File | Quantization | Size |
|---|---|---|
| `Qwen3.8-27B-Uncensored-xCloud-Q8_0.gguf` | Q8_0 | 28.6 GB |
| `Qwen3.8-27B-Uncensored-xCloud-Q6_K.gguf` | Q6_K | 22.1 GB |
| `Qwen3.8-27B-Uncensored-xCloud-Q5_K_M.gguf` | Q5_K_M | 19.2 GB |
| `Qwen3.8-27B-Uncensored-xCloud-Q4_K_M.gguf` | Q4_K_M | 16.5 GB |
| `Qwen3.8-27B-Uncensored-xCloud-IQ4_XS.gguf` | IQ4_XS (imatrix) | 14.4 GB |
| `Qwen3.8-27B-Uncensored-xCloud-IQ2_M.gguf` | IQ2_M (imatrix) | 10.0 GB |
| `mmproj-Qwen3.8-27B-Uncensored-xCloud-f16.gguf` | Vision projector | 0.9 GB |

## Usage (llama.cpp)

```bash
llama-cli -m Qwen3.8-27B-Uncensored-xCloud-Q4_K_M.gguf -p "你的提示"
# 看圖
llama-server -m Qwen3.8-27B-Uncensored-xCloud-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-Uncensored-xCloud-f16.gguf
```

## About xCloudinfo

xCloudinfo Corp. Limited (云碩科技) builds on-premises, private AI solutions. xVITA is xCloudinfo's in-house model brand.

base: Qwen/Qwen3.8-27B (Apache-2.0)

More models: [https://hf.co/xCloudinfo](https://hf.co/xCloudinfo)

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

以 **Qwen/Qwen3.8-27B**（qwen3_5 混合線性注意力 + reasoning）為底，經 **abliteration（拒絕方向正交化）**
降低過度拒絕的版本，GGUF 量化格式，由 **xCloudinfo（云碩科技）** 釋出。

> **使用前提：本模型僅供在合法、不違法的前提下使用** —— 授權範圍內的資安研究、紅隊測試、內容審核研究、
> 學術研究，以及減少模型「過度拒絕」造成的可用性問題。**使用者須自行為其用途負全部法律與道德責任；
> 嚴禁用於任何違法行為。** 釋出者不為任何濫用背書或負責。

## 效果（held-out 實測，非宣稱）

我方一律以一組 **10 題嚴重危害 held-out** 實測拒答率並公開數字——**這是我方與多數「只掛名、未實測」
社群版本的差異**。量測方法：量化後實際載入（本次直接經 `llama-server` 實測上架檔）、reasoning 模型
**關閉 thinking**、簡繁英拒絕詞判定、輸出連貫性檢查（空/片段輸出不計為通過）。

| 版本 | 拒答率 |
|---|---|
| 原始 Qwen3.8-27B | 約 10/10（幾乎全拒）|
| 本版本 Q4_K_M（量化後經 server 實測）| **0/10** |

一般能力同步抽驗（光合作用、瑞利散射、台灣高山烏龍茶沖泡、高鐵停靠站等無害題）**繁中流暢、作答正確**，
未因處理或量化而劣化。單一方向 abliteration 不保證移除所有安全行為，也**不應**被視為移除；請在上述前提下負責任地使用。

## 方法（誠實揭露）

- **方向消融**（Arditi et al. 2024），不重新訓練。以自動化工具 **Heretic**（p-e-w/heretic）搜參數：
  以 Optuna 跑 200 組試驗，**同時最小化「拒答率」與「KL 散度」**——即在壓低拒絕的同時，用 KL 散度
  守住模型不變笨（這是與「只把強度開到滿、不顧品質」做法的關鍵差別）。
- 針對 reasoning 模型**在 thinking 段之外抽取方向**，並涵蓋混合架構的 **SSM 線性注意力輸出投影**
  （`out_proj`，qwen3_5 的關鍵，僅消文字塔的殘差寫入者）；以**迭代兩輪**把殘餘拒絕壓到 0/10。
- **視覺**：保留（附 `mmproj`）。**繁體中文與一般能力經驗證未劣化**。

> **注意**：本模型的世界觀、政治立場、事實性陳述等內容**繼承自底模 Qwen3.8（中國來源）**，
> 不在本次 abliteration 的處理範圍內；abliteration 僅調整「拒絕行為」，不改變模型既有的價值判斷。

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

---
xCloudinfo · 云碩科技　|　base：Qwen/Qwen3.8-27B（Apache-2.0）

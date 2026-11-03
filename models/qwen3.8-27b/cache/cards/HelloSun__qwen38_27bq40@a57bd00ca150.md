---
license: apache-2.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- gguf
- llama.cpp
- cpu
- avx512-vnni
- amx
- qwen3.5
- qwen3.8
---

# Qwen3.8-27B Q4_0 / UD-IQ3_XXS：針對此 CPU 的 llama.cpp 優化（patch + 一鍵編譯）

本 repo 提供一個 **llama.cpp CPU patch** 與 **一鍵編譯腳本**，針對 `unsloth/Qwen3.8-27B-GGUF` 的 **`Qwen3.8-27B-Q4_0.gguf`** 與 **`Qwen3.8-27B-UD-IQ3_XXS.gguf`** 在 Intel Xeon（AVX512-VNNI / AMX）上加速。

- **優化目標：只有這台電腦**（Intel Xeon Platinum 8559C，16 threads）。編譯使用 `-DGGML_NATIVE=ON`，產生的 binary 不保證可在其他 CPU 上取得相同效果。
- **範圍**：Q4_0 檔案的所有量化型別都有優化（見下表）。UD-IQ3_XXS 檔案已優化 IQ3_XXS 與 IQ3_S（約佔 48% 的 bytes），其餘型別**尚未**優化（見「UD-IQ3_XXS 說明」）。Q4_0 檔案內容如下：

| 量化型別 | 大小 | 用途（此檔案） |
|---|---|---|
| Q4_0 | 12.44 GiB（352 個 tensor） | 大部分 attention / FFN 權重 |
| Q5_K | 0.97 GiB（48 個） | `ssm_out`（線性注意力輸出） |
| Q6_K | 0.97 GiB（1 個） | `output.weight`（lm_head） |
| Q4_1 | 0.42 GiB（8 個） | 部分 `ffn_down` |
| Q8_0 | 0.05 GiB（1 個） | 少量權重 |
| F32 | 0.10 GiB | norm、ssm 參數等 |

GGUF 檔案本身**未經修改**，所有加速都來自程式碼（`patches/qwen38-q40-cpu-opt.patch`，base commit `de5122d87`）。

## 結果（tokens/s）

同一台機器、同一份 GGUF、同一套編譯旗標（`-DGGML_NATIVE=ON`），只差 patch。數值為三輪交錯量測的中位數（每輪 `-p 512,2048 -n 64 -r 2`）。

| 版本 | prefill pp512 | prefill pp2048 | decode tg64 |
|---|---|---|---|
| **之前**：stock llama.cpp（de5122d87），Q4_0 走 AMX buffer | 35.8 | 36.3 | **7.22** |
| **之後**：套用 patch | **64.1** | **70.2** | 6.74 |
| 倍率（之後 / 之前） | **1.79x** | **1.93x** | 0.93x |

逐輪數值（tokens/s）：

- 之前 pp512：35.83 / 36.97 / 34.92；pp2048：36.99 / 36.32 / 35.36；tg64：7.32 / 7.12 / 7.22
- 之後 pp512：63.68 / 64.10 / 64.76；pp2048：70.84 / 70.18 / 67.99；tg64：6.69 / 6.84 / 6.74

**誠實說明 decode**：decode 在這台機器上**沒有超過 stock**，約慢 7%。這個檔案 decode 每個 token 要讀約 16 GB 權重，在此機器上約 115 GB/s，已接近記憶體頻寬的實際上限（16 threads 約 125 至 175 GB/s）。stock 的 AMX decode 路徑讀取效率同樣很高，因此差異不大。prefill 才是這次優化的主要收益。

另外參考數據（僅 stock 的 `--repack 0`，即完全不用 AMX/repack buffer）：pp512 20.6、tg16 4.2，說明 stock 的純 CPU 路徑並不快，AMX 是 stock 的預設選擇。

## 準確度

wikitext-2 test，16 chunks x 512 tokens，`llama-perplexity`：

| 版本 | PPL |
|---|---|
| stock | 7.1455 ± 0.2776 |
| patched | 7.1390 ± 0.2772 |

差值 -0.0065，在標準誤差內，數值等價。kernel 層級的檢查：Q4_0 tiled GEMM 與 vec_dot 的相對 L2 誤差約 2e-7；Q4_0 AVX512-VNNI vec_dot 與 generic 參考實作的相對誤差低於 1e-6。整數點積是精確的，只有浮點累加順序不同。

## 做了什麼

1. **Q4_0 不再走 AMX / repack buffer（預設）**。`repack.cpp` 與 `amx/amx.cpp` 加入機器相關的開關：Q4_0 預設留在一般 CPU buffer。若要比較原本行為，設定 `GGML_REPACK_Q4_0=1 GGML_AMX_Q4_0=1`。
2. **Q4_0 tiled AVX512-VNNI GEMM（prefill）**。`w = d·(q-8) = d·q + (-8d)`，因此直接重用 Q4_1 的 tiled kernel（`m = -8d`），activation 為 Q8_0。這是 prefill 提升約 1.8 到 1.9 倍的主因。
3. **Q4_0 AVX512-VNNI `vec_dot`（decode）**。一次處理兩個 32 元素 block。`|w|` 作為 unsigned 運算元，`w` 的符號移到 int8 activation 上（`_mm256_sign_epi8`），整數結果精確。prefetch 距離 `GGML_Q40_PF`，預設 320。
   - prefetch 掃描：0 時 tg64 降到 4.67，明顯證明 prefetch 有效；160 到 1024 之間約 6.2 到 6.8，在雜訊範圍內平坦。
4. **Q4_1 / Q5_K / Q6_K**：沿用同一套 patch 中已有的 kernel（Q4_1 的 VNNI vec_dot 與 tiled GEMM、Q5_K 的 VNNI 合併、Q6_K 的 VNNI vec_dot 取代 AMX M=1）。這些 tensor 同樣在此檔案的推論路徑上。
5. patch 內也包含 NVFP4 的 VNNI kernel（來自另一個專案），此檔案**不含 NVFP4 tensor，該部分不會被執行**。


## UD-IQ3_XXS（10.9 GB）

`Qwen3.8-27B-UD-IQ3_XXS.gguf` 是混合量化檔案，共 14 種型別：

| 量化型別 | 大小 | 本 patch 是否優化 |
|---|---|---|
| IQ3_XXS | 2.78 GiB（120 個） | **是**，AVX512-VNNI vec_dot |
| IQ3_S | 2.42 GiB（106 個） | **是**，AVX512-VNNI vec_dot |
| IQ4_XS | 1.34 GiB（45 個） | 否（沿用 stock 路徑） |
| Q4_K | 0.88 GiB（26 個） | 否（沿用 patch 既有 kernel） |
| IQ2_S | 0.78 GiB（35 個） | 否 |
| Q2_K | 0.54 GiB（10 個） | 否 |
| IQ2_XXS | 0.44 GiB（24 個） | 否 |
| Q6_K | 0.32 GiB（6 個） | 是（patch 既有 AVX512-VNNI 路徑） |
| IQ2_XS | 0.27 GiB（12 個） | 否 |
| Q3_K | 0.15 GiB（6 個） | 否 |
| IQ1_S / IQ1_M | 0.18 GiB（11 個） | 否 |
| Q8_0 / Q5_K / F32 | 約 0.07 GiB | 部分 |

### 結果（tokens/s，三輪交錯量測的中位數）

同一台機器、同一份 GGUF、同一套編譯旗標，只差 patch。

| 版本 | prefill pp512 | prefill pp2048 | decode tg64 |
|---|---|---|---|
| 之前：stock llama.cpp（de5122d87） | 62.3 | 64.0 | 4.30 |
| 之後：套用 patch | 61.6 | 62.3 | **4.72** |
| 倍率（之後 / 之前） | 0.99x（雜訊） | 0.97x（雜訊） | **1.10x** |

逐輪數值：stock tg64 為 4.30 / 4.31 / 3.91，patched 為 4.72 / 3.97 / 4.78。decode 的輪與輪波動很大（主機 load average 約 50），因此 1.10x 只能視為「略有提升」，不是精確數字。prefill 在雜訊內沒有變化。

kernel 層級（單執行緒微基準，每個 superblock）：IQ3_XXS 的 AVX2 stock 為 25.4 ns，優化版 21.6 ns（約快 15%）。

### 準確度

wikitext-2 test，16 chunks x 512 tokens：

| 版本 | PPL |
|---|---|
| stock | 7.4058 ± 0.2885 |
| patched | 7.4150 ± 0.2892 |

差值 +0.009，在標準誤差內。

### 數值取捨（請閱讀）

- **-128 的處理**：vpdpbusd 需要 signed 乘 unsigned，activation 的符號要移到 activation 上，`sign_epi8(-128, -1)` 會溢位。因此 kernel 先把 -128 clamp 為 -127，每個受影響元素有 1 個量化單位的誤差。
- stock 的 AVX2 IQ3_XXS 與 IQ3_S 在含 -128 的資料上有同樣的溢位問題，我們的 kernel 誤差更小（對抗測試中 stock 相對誤差約 3.0，優化版約 6e-3）。這不是精確等價；不含 -128 的資料上，優化版與 generic 相對誤差約 1e-6 到 3e-6（精確）。
- 真實 Q8_K activation 很少出現 -128，實測 PPL 差異在誤差內。

### 尚未完成

未優化的型別合計約 23% 的 bytes（IQ2_S、Q2_K、IQ2_XXS、IQ2_XS、Q3_K、IQ1_S、IQ1_M），以及 IQ4_XS。若要全檔案覆蓋，需要為這些型別加入對應的 VNNI kernel。

## 建置與執行（UD-IQ3_XXS）

```bash
QUANT=IQ3_XXS bash build.sh
```

## 算子層級優化（Q4_0，僅針對此 CPU）

所有算子層級的改動都只針對這台 Xeon 8559C 做，沒有考慮其他 CPU 的通用性。

1. **單 row 的扁平 tensor 改為多執行緒**：`GGML_OP_SCALE` 在 `ne = [786432,1,1]` 這種只有 1 row 的 tensor 上，原本按 row 分配，只有 thread 0 在算（decode 每次約 700 µs）。改為 `nr == 1` 時按元素分配到各 thread。
2. **RMS_NORM+MUL 融合核心改為 SIMD**：融合路徑原本是標量迴圈。改為與非融合路徑相同的 `scale` 與 `vec_mul`，運算順序（`x*scale*w` 由左至右）與逐位結果不變。
3. **Fusion 開關實驗**：`GGML_CPU_DISABLE_FUSION=1` 可關閉融合。修正前，融合開啟比關閉慢約 2–3%；修正後，兩者在誤差內無差異，因此不再以此作為理由保留或移除融合。

Profile 顯示 decode 的 matmul 約佔 92%，非 matmul 只剩約 7.7%，所以算子層級能拿到的上限很小。

| 版本（Q4_0，三輪交錯中位數，tokens/s） | pp512 | pp2048 | tg64 |
|---|---|---|---|
| 修正前（上一版 patch） | 65.4 | 69.7 | 6.82 |
| 修正 SCALE 後 | 66.6 | 69.1 | 6.90 |
| 修正 SCALE + RMS 融合 SIMD（fusion 開） | 65.5 | 68.8 | 6.91 |
| 同上（fusion 關） | 66.7 | 67.2 | 6.93 |

誠實的結論：這兩項修正讓 decode 約快 1%（6.82 → 6.91–6.93），prefill 在誤差內沒有可量測的變化。3% 以內的差異在這台共享主機上不可分辨。

PPL（wikitext-2，16x512）：修正前後都是 **7.1390**，逐位一致（兩項修正不改變數值）。

原始資料與 profile：`benchmarks/ops_v3/`。

## MTP 投機解碼（Q4_0）

`unsloth/Qwen3.8-27B-GGUF` 附有獨立的 MTP 權重 `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`（1.37 GB）。本 patch 使用 llama.cpp 既有的 `draft-mtp` 投機解碼，草稿 token 數預設 3。`build.sh` 預設會下載並啟動 MTP（`MTP=0` 可關閉）。

### 為何原本加速有限，以及修正

第一次量測時，MTP 的接受率很高（k=1 為 0.88），但加速只有約 1.2 倍。反推每個 step 的成本：一次 verify 送 k+1 個 token，成本約是單 token decode 的 2.5 倍（k=3）。Profile（`benchmarks/mtp/profile_mtp2_before_cpy_fix.log`）顯示，原因是 **`CPY [786432,1,3]` 佔 verify 時間的 15.9%**。這是 delta-net 的 recurrent state 快照複製（投機解碼用來回滾），只有 3 個 row，因此 `dup_bytes` 只用到 3 個執行緒，且每次 1.1 ms。

修正：當 row 數少於執行緒數時，改為把元素切給各執行緒（`ggml-cpu/ops.cpp` 的 `ggml_compute_forward_dup_bytes`）。修正後 greedy 輸出與接受數與修正前完全相同。

### 數據（同一台機器，llama-server，4 個程式題 × 256 tokens，tokens/s）

Temp 0（greedy）：

| 設定 | 中位數 tok/s | 逐 prompt | 接受率 |
|---|---|---|---|
| 無 MTP（三輪中位數） | 6.67 | 5.77 / 6.79 / 6.67 | — |
| MTP k=3，修正前 | 6.67 | — | 0.70 |
| MTP k=3，修正後（三輪） | **8.37** | 8.37 / 9.43 / 8.10 | 0.70 |

- 修正後 k=3 與無 MTP 的逐輪比值為 1.21x / 1.39x / 1.45x，中位數約 **1.3x**。
- 單輪的 k 掃描（修正後，第一輪）：k=1 6.56（接受率 0.88），k=2 7.86（0.79），k=3 8.37（0.70）。

Temp 1（單輪）：無 MTP 6.38（逐 prompt 4.05–6.73，波動大），k=2 8.07（接受率 0.72），k=3 8.44（0.62）。

**限制**：主機為共享環境，無 MTP 的 temp 0 逐輪數據從 5.8 到 6.8 變動，因此倍率是區間估計。prefill 不受 MTP 影響。

### 草稿長度 k=3 至 k=6（Q4_0，temp 0，三輪交錯）

| k | decode 中位數 tok/s | 逐輪倍率（同輪基線） | 中位倍率 | 接受率 | prefill 中位數 tok/s |
|---|---|---|---|---|---|
| 無 MTP | 5.64（5.08 / 5.64 / 6.35） | 1.00x | 1.00x | — | 67.0 |
| 3 | **9.33**（9.42 / 9.33 / 9.33） | 1.85 / 1.66 / 1.47 | **1.66x** | 0.70 | 63.5 |
| 4 | 7.27（7.40 / 7.27 / 6.50） | 1.46 / 1.29 / 1.02 | 1.29x | 0.60 | 65.6 |
| 5 | 7.25（7.26 / 7.25 / 7.08） | 1.43 / 1.29 / 1.11 | 1.29x | 0.53 | 63.9 |
| 6 | 6.44（6.12 / 6.70 / 6.44） | 1.20 / 1.19 / 1.01 | 1.19x | 0.49 | 56.2 |

- **k=3 最佳**。k 增加後接受率持續下降（0.70 → 0.49），但 verify 的成本仍隨 token 數增加，淨收益反而減少。k=4 以上的加速明顯低於 k=3。
- 逐輪倍率的範圍很大，原因是共享主機上無 MTP 的基線從 5.1 漂到 6.4。以同輪比值較可靠。
- prefill（約 500 token prompt）在 56–68 之間，多數差異屬雜訊。k=6 的中位數較低（56.2），但沒有在每一輪都明顯低於基線，不能下定論。
- temp 0 的輸出與接受率在三輪中完全相同（greedy 且確定性）。

### 使用

```bash
bash build.sh                  # 預設 MTP=1、MTP_K=3
MTP=0 bash build.sh            # 不用 MTP
MTP_K=2 bash build.sh          # 草稿 token 數改為 2
```

手動啟動：

```bash
llama.cpp/build/bin/llama-server -m Qwen3.8-27B-Q4_0.gguf -md mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 -t 16 -fa on --jinja -c 32768
```

原始資料：`benchmarks/mtp/`（各設定的 JSON、`summary.csv`、profile、`spec_bench.py`、`run_spec.sh`、k 掃描腳本 `k_sweep_run*.sh`）。

## 一鍵編譯

```bash
# 需要：git、cmake、gcc/g++（支援 -march=native）、curl、python3
# 機器需要 AVX512-VNNI（腳本會檢查）
git clone https://huggingface.co/HelloSun/qwen38_27bq40
cd qwen38_27bq40
bash build.sh                  # Q4_0（預設）：clone llama.cpp -> checkout de5122d87 -> 套 patch -> 編譯 -> 下載 -> 啟動
QUANT=IQ3_XXS bash build.sh    # UD-IQ3_XXS
```

可覆寫的環境變數：`LLAMA_DIR`、`MODEL_FILE`、`LLAMA_PORT`（預設 20000）、`CONTEXT_SIZE`（預設 32768）、`LLAMA_THREADS`（預設 nproc）、`HF_TOKEN`（下載用，可選）、`NO_SERVE=1`（只編譯與下載）。

啟動後：

```bash
curl http://127.0.0.1:20000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
```

手動 benchmark（與本文相同設定）：

```bash
taskset -c 0-15 llama.cpp/build/bin/llama-bench -m Qwen3.8-27B-Q4_0.gguf -t 16 -fa 1 -b 1024 -ub 1024 -lm mmap+mlock -p 512,2048 -n 64 -r 2 -o csv
```

單次驗證（一鍵腳本的實測）：clone、checkout、套 patch 後的 `git diff` 與發佈的 patch 完全相同；編譯完成的 `llama-server` 可啟動並正常生成文字。

## 量測注意事項

- 主機為共享環境。stock 的 decode 在不同時段可從 5.9 到 7.3 tokens/s 波動，因此只比較同一輪交錯量測的結果，並使用中位數。
- 所有數值都在此機器上量測，未在其他 CPU 上驗證。
- 僅量測 Q4_0 這個檔案，未重新量測其他 GGUF。
- PPL 為短測試（16 x 512 tokens），未做 KLD 或下游任務評估。

## 檔案

- `build.sh`：一鍵編譯、下載、啟動（`QUANT=Q4_0` 或 `IQ3_XXS`；Q4_0 預設含 MTP）
- `patches/qwen38-q40-cpu-opt.patch`：完整 patch（相對於 `patches/BASE_COMMIT`）
- `patches/BASE_COMMIT`：`de5122d879d00d2895c36be7e0cee94105aeeabd`
- `benchmarks/`：Q4_0 原始 CSV、`summary.csv`、PPL、prefetch 掃描；`benchmarks/iq3_xxs/`：IQ3_XXS 原始 CSV、`summary.csv`、PPL、kernel 測試原始碼（`iq3_dot.c`、`iq3s_dot.c`、`iq3_bench.c`）

## 授權與致謝

模型：Qwen3.8（Apache-2.0），GGUF 量化由 Unsloth 提供，未經修改。patch 建立於 llama.cpp（MIT）。

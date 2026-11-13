---
base_model:
- Qwen/Qwen3.8-27B
language:
- vi
- ja
- zh
- en
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- abliterated
- uncensored
- qwen
- qwen3.8
- 27b
- gguf
- llama-cpp
- ollama
- runpod
---

# Qwen 3.8 27B Instruct Abliterated (GGUF Q4_K_M)

Phiên bản GGUF Q4_K_M của mô hình **Qwen3.8-27B** đã được gỡ bỏ hoàn toàn rào cản từ chối (Abliteration) bởi tác giả `huihui-ai`, tối ưu hóa đặc biệt cho tác vụ dịch thuật đàm thoại điện ảnh, phụ đề 18+ Uncensored đa ngữ (Nhật - Việt - Anh).

## 1. Thông Tin Kỹ Thuật
- **Mô hình gốc:** `Qwen/Qwen3.8-27B` (Alibaba Cloud)
- **Phương pháp gỡ kiểm duyệt:** Refusal Direction Abliteration (Layers 18-51 ablated)
- **Định dạng:** GGUF `UD-DW-Q4_K_M` (~15.41 GB)
- **Yêu cầu phần cứng:** Khuyên dùng GPU **24GB VRAM** (RTX 3090, RTX 4090, A10G, L4) hoặc **48GB VRAM** (A40, RTX 6000 Ada).
- **Hỗ trợ ngữ cảnh:** 16k – 32k tokens trên GPU 24GB VRAM.

## 2. Hướng Dẫn Sử Dụng
### Với Ollama:
```bash
ollama run hf.co/minhtandangtt/Qwen3.8-27B-Instruct-abliterated-Q4KM
```

### Với llama.cpp CLI:
```bash
llama-cli --hf-repo minhtandangtt/Qwen3.8-27B-Instruct-abliterated-Q4KM --hf-file qwen3.8-27b-instruct-abliterated-q4_k_m.gguf -p "Dịch sang tiếng Việt:"
```

### Sử dụng trên RunPod Serverless:
Chạy trên template `runpod-workers/worker-ollama:v1.3.0` với GPU 24GB VRAM, đặt `MODEL_NAME` trỏ về `hf.co/minhtandangtt/Qwen3.8-27B-Instruct-abliterated-Q4KM`.

---
license: apache-2.0
base_model:
  - prism-ml/Ternary-Bonsai-2-27B-gguf
  - Qwen/Qwen3.8-27B
language:
  - en
  - th
  - zh
  - multilingual
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - pq2_0
  - ternary
  - speculative-decoding
  - abliterated
  - uncensored
  - lora-merge
  - svd
  - reasoning
  - qwen3.5
  - vision
---

# Ternary Bonsai 2 27B — Uncensored Heretic + Swift 

**PrismML's ternary Bonsai 2 with the Heretic abliteration baked in, plus the [hotdogs/swift_unc_qwen3.8-27B_lora](https://huggingface.co/hotdogs/swift_unc_qwen3.8-27B_lora) merged at 0.7.**

One ~6.7 GB file: the same 27B ternary runtime as the base — **PQ2_0 packing, Hadamard latent, 262,144-token context**. Vision projector included.

> ⚠️ **This file needs PrismML's llama.cpp fork.** PQ2_0 does **not** load in mainline llama.cpp, Ollama, LM Studio or llama.app. See [Requirements](#requirements).

---

## Table of contents

- [What this is](#what-this-is)
- [Files in this repo](#files-in-this-repo)
- [Requirements](#requirements)
- [Quick start — automatic installer](#quick-start--automatic-installer)
- [Manual installation, step by step](#manual-installation-step-by-step)
- [Running the server](#running-the-server)
- [Flag reference](#flag-reference)
- [Reasoning effort and sampling](#reasoning-effort-and-sampling)
- [Hardware and context sizing](#hardware-and-context-sizing)
- [Calling the API](#calling-the-api)
- [Docker](#docker)
- [Vision](#vision)
- [How this build was made](#how-this-build-was-made)
- [What to expect, and what is honest about it](#what-to-expect-and-what-is-honest-about-it)
- [คู่มือภาษาไทย](#คู่มือภาษาไทย)
- [Safety](#safety)
- [Credits](#credits)

---

## What this is

| | |
|---|---|
| Architecture | `qwen35` (Qwen3.8-27B hybrid GDN/attention), 65 blocks (64 ), hidden 5120, 24 heads / 4 KV heads, head dim 256 |
| Parameters | ~27.3B  |
| Quantisation | **PQ2_0** (prism-ml ternary format, ≈ **2.13 bits/weight**) with the Hadamard latent transform |
| Context | **262,144 tokens** (`qwen35.context_length`), rope base 1e7 |
| Sampler defaults baked into the file | `temp 1.0`, `top_k 20`, `top_p 0.95` |
| Chat template | embedded in the GGUF, also shipped as `chat_template.jinja` — supports `reasoning_effort` (`xhigh` default / `medium` / `low`) and vision (`<\|image\|>`) |
| Vision | yes — pair with `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` from this repo |
| Runtime | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), tag `prism-b10743-adfffbe` or newer |

### The two ingredients

```
PrismML Ternary-Bonsai-2-27B (ternary, PQ2_0)
        │
        ▼   BoldingBuilds Heretic abliteration  → heretic base
OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF
        │
        └── + hotdogs/swift_unc_qwen3.8-27B_lora .... scale 0.25
        │
        ▼
        this model
```

Why merge the LoRA instead of shipping a second 55 GB model: the Swift-unc behaviour change is a small, structured edit relative to stock Qwen3.8-27B, so it fits in a rank-16 adapter (204 `lora_A`/`lora_B` pairs, float32, ~122 MB). Merging it into the already-quantised ternary base keeps a single ~6.7 GB file you can hold on one GPU or run on CPU.

The Heretic base already carries the abliteration (0/100 refusals, KL 0.0135 vs the original). The Swift adapter adds **refusal removal** (98/100 → 23/100 on the Heretic eval) *and* UkisAI's **reasoning-efficiency fine-tune** (the `thinking fix` so it closes its reasoning once the answer is written), projected out of the base at scale 0.7.

---

## Files in this repo

| File | Size (bytes) | Use |
|---|---|---|
| `Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc-PQ2_0.gguf` | 7,165,249,888 (6.67 GiB) | **Recommended.** Full ternary language model, Hadamard latent,   head |
| `Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc-PTQ1_0.gguf` | 5,904,623,968 (5.50 GiB) | Smaller pack. Same 866 tensors, different packing for the   head and the two giant 1.27B-parameter matrices |
| `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` | 931,145,856 (0.87 GiB) | Vision projector (BF16) for image input |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 629,246,976 (0.59 GiB) | Vision projector (Q8_0) |
| `chat_template.jinja` | 8,952 | Reference copy of the chat template (the GGUF already embeds it) |
| `install.sh` | — | One-shot installer: checks hardware, builds the fork, downloads the model, writes `start.sh` |

Both GGUF files contain the **same 866 tensors** (851 language + 15  ). They differ only in how the   head and the two giant 1.27B-parameter matrices are stored — same story as the base repo, different packings.

Pick **`-PQ2_0.gguf`** for the standard release. Pick **`-PTQ1_0.gguf`** if you want a smaller file and don't need the full   head quality.

---

## Requirements

| | |
|---|---|
| Runtime | **PrismML's llama.cpp fork only** — [github.com/PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), tag **`prism-b10743-adfffbe`** or newer |
| OS | Linux, macOS (Apple Silicon), Windows via WSL2 |
| GPU | NVIDIA (**CUDA 12.8+** for Blackwell / RTX 50-series, any recent CUDA for older cards), Apple Metal, or CPU |
| Tools to build | `git`, `cmake` ≥ 3.20, `gcc`/`clang` with C++17, `python3` + `python3-venv`, `curl` |
| Disk | ~15 GB free per install (source + build + 6.7 GB model) |
| Python (only for downloading) | `huggingface_hub` (the installer creates its own venv) |

Why the fork: this model uses PrismML's **PQ2_0 ternary packing** and a **Hadamard latent transform**. Mainline llama.cpp has neither. Older PrismML tags fail at startup with:

```
Hadamard-latent table 'token_embd.weight' is read without the inverse transform
```

Tag `prism-b10743-adfffbe` (or newer) has both the Hadamard inverse transform and working `--spec-type draft- `, so no patch is needed.

---

## Quick start — automatic installer

The installer does everything: checks OS/CPU/RAM/GPU, installs missing tools, clones and builds the PrismML fork for your GPU (or CPU), downloads the GGUF, and writes a `start.sh` that auto-tunes context size to your VRAM.

```bash
# 1) save install.sh (see the full script below, or download install.sh from this repo)
# 2) run it
bash install.sh

# 3) start the server
~/bonsai/start.sh
# then open http://127.0.0.1:8080
```

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `INSTALL_DIR` | `~/bonsai` | Where to put source, build, model, venv |
| `FORCE_CPU` | `0` | `1` = build CPU-only even if a GPU is present |
| `HF_TOKEN` | — | Needed only if you pull from a private/gated repo |
| `REPO_TAG` | `prism-b10743-adfffbe` | PrismML fork tag to build |
| `MODEL_FILE` | `Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc-PQ2_0.gguf` | Which GGUF to fetch |

The installer is idempotent — if the build finished or the model is already downloaded, re-running it skips those steps. If a download is interrupted, just run it again to resume.

### `install.sh` (full script)

```bash
#!/usr/bin/env bash
# =============================================================================
#  Ternary Bonsai 2 27B - ตัวติดตั้งอัตโนมัติ (Linux / macOS)
#    1) ตรวจ OS / CPU / RAM / GPU
#    2) ติดตั้งเครื่องมือที่ขาด
#    3) clone + build PrismML llama.cpp (fork ที่รองรับ PQ2_0 และ  )
#    4) ดาวน์โหลดโมเดล
#    5) สร้าง start.sh สำหรับเปิดเซิร์ฟเวอร์
#
#  ใช้งาน:   bash install.sh
#  ปรับแต่ง (ไม่บังคับ):
#    INSTALL_DIR=/path   โฟลเดอร์ติดตั้ง (ค่าเริ่มต้น ~/bonsai)
#    FORCE_CPU=1         บังคับ build แบบ CPU
#    HF_TOKEN=hf_xxx     ถ้า repo โมเดลเป็น private / gated
# =============================================================================
set -Eeuo pipefail

# ---------- ค่าตั้งต้น ----------
REPO_URL="${REPO_URL:-https://github.com/PrismML-Eng/llama.cpp}"
REPO_TAG="${REPO_TAG:-prism-b10743-adfffbe}"   # ต้องเป็น b10743 ขึ้นไป (  + Hadamard)
HF_REPO="${HF_REPO:-hotdogs/Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc}"
MODEL_FILE="${MODEL_FILE:-Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc-PQ2_0.gguf}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/bonsai}"
MIN_FREE_GB=15

SRC_DIR="$INSTALL_DIR/llama.cpp"
MODEL_DIR="$INSTALL_DIR/models"
VENV_DIR="$INSTALL_DIR/venv"
STAMP="$INSTALL_DIR/.build-stamp"

# ---------- ตัวช่วยแสดงผล ----------
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=""; G=""; Y=""; R=""; N=""; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '%s✔ %s%s\n' "$G" "$*" "$N"; }
warn() { printf '%s⚠ %s%s\n' "$Y" "$*" "$N"; }
die()  { printf '%s✘ %s%s\n' "$R" "$*" "$N" >&2; exit 1; }
trap 'die "ติดตั้งไม่สำเร็จ (บรรทัด $LINENO) - ส่งข้อความด้านบนให้ผู้ดูแลได้เลย"' ERR

have() { command -v "$1" >/dev/null 2>&1; }

# =============================================================================
step "1/6 ตรวจสอบเครื่อง"
# =============================================================================
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Linux|Darwin) ;;
  *) die "ระบบ $OS ไม่รองรับ - ถ้าใช้ Windows ให้ติดตั้ง WSL2 (Ubuntu) แล้วรันสคริปต์นี้ใน WSL" ;;
esac

if [ "$OS" = "Darwin" ]; then
  CORES="$(sysctl -n hw.ncpu)"
  MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
else
  CORES="$(nproc)"
  MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
fi
echo "ระบบ: $OS $ARCH | CPU: $CORES core | RAM: ${MEM_GB} GB"

mkdir -p "$INSTALL_DIR"
FREE_GB=$(( $(df -Pk "$INSTALL_DIR" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
echo "พื้นที่ว่างที่ $INSTALL_DIR: ${FREE_GB} GB"
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || die "พื้นที่ดิสก์ไม่พอ (ต้องการอย่างน้อย ${MIN_FREE_GB} GB)"

# ---------- ตรวจ GPU ----------
BACKEND="cpu"
CUDA_ARCH=""
VRAM_MB=0

if [ "$OS" = "Darwin" ]; then
  BACKEND="metal"
  ok "พบ Apple Silicon/Metal"
elif [ "${FORCE_CPU:-0}" = "1" ]; then
  warn "FORCE_CPU=1 → ใช้ CPU เท่านั้น"
elif have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -n1 | tr -d ' ')"
  CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | grep -E '^[0-9]+\.[0-9]+$' || true)"
  if [ -n "$CAPS" ]; then
    CUDA_ARCH="$(echo "$CAPS" | sed 's/\.//' | paste -sd';' -)"
  else
    CUDA_ARCH="native"
  fi
  echo "พบ NVIDIA GPU: $GPU_NAME | VRAM: ${VRAM_MB} MiB | CUDA arch: $CUDA_ARCH"

  # หา nvcc
  for p in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/cuda/bin; do
    [ -x "$p/nvcc" ] && export PATH="$p:$PATH" && break
  done
  if have nvcc; then
    NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -n1)"
    NV_MAJ="${NVCC_VER% *}"; NV_MIN="${NVCC_VER#* }"
    echo "พบ CUDA Toolkit $NV_MAJ.$NV_MIN"
    BACKEND="cuda"
    # การ์ด Blackwell (compute 12.x) ต้องใช้ CUDA 12.8+
    if echo "$CUDA_ARCH" | grep -qE '(^|;)12[0-9]'; then
      if [ "$NV_MAJ" -lt 12 ] || { [ "$NV_MAJ" -eq 12 ] && [ "$NV_MIN" -lt 8 ]; }; then
        warn "การ์ดรุ่นนี้ (RTX 50-series) ต้องใช้ CUDA Toolkit 12.8 ขึ้นไป แต่เครื่องมี $NV_MAJ.$NV_MIN"
        warn "ติดตั้งได้ที่ https://developer.nvidia.com/cuda-downloads แล้วรัน install.sh อีกครั้ง"
        warn "ตอนนี้จะ build แบบ CPU ไปก่อน"
        BACKEND="cpu"
      fi
    fi
  else
    warn "พบ GPU NVIDIA แต่ไม่พบ CUDA Toolkit (nvcc)"
    warn "ติดตั้งได้ที่ https://developer.nvidia.com/cuda-downloads แล้วรัน install.sh อีกครั้ง (จะ build ใหม่ให้เอง)"
    warn "ตอนนี้จะ build แบบ CPU ไปก่อน"
  fi
else
  warn "ไม่พบ NVIDIA GPU → ใช้ CPU (fork นี้รองรับเฉพาะ CUDA / Metal / CPU การ์ด AMD และ Intel จะไม่ถูกใช้)"
fi
ok "Backend ที่จะใช้: $BACKEND"

# =============================================================================
step "2/6 ติดตั้งเครื่องมือที่จำเป็น"
# =============================================================================
SUDO=""
if [ "$(id -u)" -ne 0 ]; then have sudo && SUDO="sudo"; fi

MISSING=()
for c in git cmake python3 curl; do have "$c" || MISSING+=("$c"); done
have c++ || have g++ || have clang++ || MISSING+=("compiler")
python3 -m venv --help >/dev/null 2>&1 || MISSING+=("python3-venv")

if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "ยังขาด: ${MISSING[*]}"
  if [ "$OS" = "Darwin" ]; then
    have brew || die "ไม่พบ Homebrew - ติดตั้งจาก https://brew.sh แล้วรันใหม่ (และรัน: xcode-select --install)"
    brew install git cmake python
  elif have apt-get; then
    $SUDO apt-get update
    $SUDO apt-get install -y git cmake build-essential python3 python3-venv python3-pip curl ca-certificates
  elif have dnf; then
    $SUDO dnf install -y git cmake gcc-c++ make python3 python3-pip curl ca-certificates
  elif have pacman; then
    $SUDO pacman -Sy --noconfirm git cmake base-devel python python-pip curl
  else
    die "ไม่รู้จักตัวจัดการแพ็กเกจ - ติดตั้งเองให้ครบ: ${MISSING[*]}"
  fi
fi
ok "เครื่องมือครบแล้ว"

# =============================================================================
step "3/6 ดึงซอร์ส llama.cpp (PrismML fork, tag $REPO_TAG)"
# =============================================================================
OLD_TAG=""
[ -f "$STAMP" ] && OLD_TAG="$(cut -d'|' -f1 "$STAMP")"
if [ -d "$SRC_DIR/.git" ] && [ "$OLD_TAG" = "$REPO_TAG" ]; then
  ok "มีซอร์สรุ่นนี้อยู่แล้ว ข้ามขั้นตอนนี้"
else
  rm -rf "$SRC_DIR"
  git clone --depth 1 --branch "$REPO_TAG" "$REPO_URL" "$SRC_DIR"
  ok "clone เสร็จ"
fi

# =============================================================================
step "4/6 Build (อาจใช้เวลา 5-20 นาที)"
# =============================================================================
NEW_STAMP="$REPO_TAG|$BACKEND|$CUDA_ARCH"
if [ -x "$SRC_DIR/build/bin/llama-server" ] && [ "$(cat "$STAMP" 2>/dev/null || true)" = "$NEW_STAMP" ]; then
  ok "build รุ่นนี้พร้อมแล้ว ข้ามขั้นตอนนี้"
else
  # จำนวนงานพร้อมกัน: จำกัดตาม RAM กัน build ค้าง/โดน kill
  PER_JOB_GB=2; [ "$BACKEND" = "cuda" ] && PER_JOB_GB=4
  JOBS=$(( MEM_GB / PER_JOB_GB ))
  [ "$JOBS" -gt "$CORES" ] && JOBS="$CORES"
  [ "$JOBS" -lt 1 ] && JOBS=1
  echo "ใช้ $JOBS งานพร้อมกัน"

  CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF)
  case "$BACKEND" in
    cuda)  CMAKE_ARGS+=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH") ;;
    metal) CMAKE_ARGS+=(-DGGML_METAL=ON) ;;
    cpu)   CMAKE_ARGS+=(-DGGML_CUDA=OFF -DGGML_NATIVE=ON) ;;
  esac

  rm -rf "$SRC_DIR/build"
  ( cd "$SRC_DIR"
    cmake -B build "${CMAKE_ARGS[@]}"
    cmake --build build --target llama-server llama-cli -j "$JOBS" )
  [ -x "$SRC_DIR/build/bin/llama-server" ] || die "build เสร็จแต่ไม่พบ llama-server"
  echo "$NEW_STAMP" > "$STAMP"
  ok "build สำเร็จ"
fi
echo "$BACKEND" > "$INSTALL_DIR/.backend"

# =============================================================================
step "5/6 ดาวน์โหลดโมเดล"
# =============================================================================
mkdir -p "$MODEL_DIR"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# ถือว่าไฟล์สมบูรณ์ถ้าใหญ่กว่า 5 GB (ไฟล์จริง ~7.16 GB PQ2_0 / ~5.9 GB PTQ1_0)
model_ok() { [ -f "$MODEL_PATH" ] && [ "$(wc -c < "$MODEL_PATH")" -gt 5000000000 ]; }

if model_ok; then
  ok "มีไฟล์โมเดลอยู่แล้ว ข้ามขั้นตอนนี้"
else
  [ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  . "$VENV_DIR/bin/activate"
  pip install -q -U pip huggingface_hub
  if have hf; then HF_CMD="hf"; elif have huggingface-cli; then HF_CMD="huggingface-cli"; else die "ติดตั้ง huggingface_hub ไม่สำเร็จ"; fi
  echo "กำลังโหลด $HF_REPO (ถ้าหลุดกลางทาง รันสคริปต์ซ้ำได้ จะโหลดต่อให้)"
  "$HF_CMD" download "$HF_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
  deactivate || true
  model_ok || die "ดาวน์โหลดไม่สมบูรณ์ (ไฟล์เล็กเกินไป) - รันสคริปต์ซ้ำอีกครั้ง หรือตรวจสอบ HF_TOKEN ถ้า repo เป็น private"
  ok "ดาวน์โหลดเสร็จ"
fi

# =============================================================================
step "6/6 สร้างตัวเปิดใช้งาน (start.sh)"
# =============================================================================
cat > "$INSTALL_DIR/start.sh" <<'STARTEOF'
#!/usr/bin/env bash
# เปิดเซิร์ฟเวอร์ Bonsai - ตั้งค่าอัตโนมัติตามเครื่อง
#   ปรับได้:  PORT=9000 HOST=0.0.0.0 CTX=65536  =0 ./start.sh
#   ส่ง flag เพิ่มต่อท้ายได้  เช่น ./start.sh --temp 0.7
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="$DIR/llama.cpp/build/bin/llama-server"
MODEL="$(ls "$DIR"/models/*.gguf | head -n1)"
BACKEND="$(cat "$DIR/.backend" 2>/dev/null || echo cpu)"
HOST="${HOST:-127.0.0.1}"     # ใช้ 0.0.0.0 ถ้าต้องการให้เครื่องอื่นเข้าถึง (ระวังเรื่องความปลอดภัย)
PORT="${PORT:-8080}"

ARGS=(-m "$MODEL" --host "$HOST" --port "$PORT" -fa on --jinja
      --chat-template-kwargs '{"reasoning_effort":"medium"}')

case "$BACKEND" in
  cuda)
    VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -n1 | tr -d ' ')"
    if   [ "$VRAM_MB" -ge 22000 ]; then CTX_DEF=65536
    elif [ "$VRAM_MB" -ge 11000 ]; then CTX_DEF=32768
    else CTX_DEF=16384; fi
    ARGS+=(-ngl 99 -c "${CTX:-$CTX_DEF}" --cache-type-k q8_0 --cache-type-v q8_0)
     _DEF=1 ;;
  metal)
    MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    if [ "$MEM_GB" -ge 24 ]; then CTX_DEF=32768; else CTX_DEF=16384; fi
    ARGS+=(-ngl 99 -c "${CTX:-$CTX_DEF}")
     _DEF=0 ;;   # ผู้ทำโมเดลทดสอบ   บน CUDA เท่านั้น
  *)
    ARGS+=(-ngl 0 -c "${CTX:-8192}")
     _DEF=0 ;;
esac

#   (speculative decoding ช่วยเร่งความเร็ว) - เปิดอัตโนมัติบน CUDA, บังคับด้วย  =1/0
if [ "${ :-$ _DEF}" = "1" ]; then
  ARGS+=(--spec-type draft-  --spec-draft-n-max 2)
fi

echo "Backend: $BACKEND | เปิดที่ http://$HOST:$PORT"
exec "$SERVER" "${ARGS[@]}" "$@"
STARTEOF
chmod +x "$INSTALL_DIR/start.sh"
ok "สร้าง $INSTALL_DIR/start.sh แล้ว"

# =============================================================================
printf '\n%s================ ติดตั้งเสร็จเรียบร้อย ================%s\n' "$G$B" "$N"
echo "เริ่มใช้งาน:   $INSTALL_DIR/start.sh"
echo "แล้วเปิดเบราว์เซอร์ที่  http://127.0.0.1:8080"
if [ "$BACKEND" = "cpu" ] && [ "$OS" = "Linux" ]; then
  echo
  warn "ตอนนี้ใช้ CPU จะช้ามาก (โมเดล 27B) ถ้ามีการ์ด NVIDIA ให้ติดตั้ง CUDA Toolkit แล้วรัน install.sh อีกครั้ง"
fi
```

---

## Manual installation, step by step

Prefer to do it yourself? Four commands' worth of work.

### 1. Build PrismML's llama.cpp

```bash
git clone https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp
git checkout prism-b10743-adfffbe

# CUDA — set your own compute capability (RTX 3090 = 86, RTX 4090 = 89, RTX 5090 = 120, A100 = 80)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-cli -j "$(nproc)"

# CPU only
# cmake -B build -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
# cmake --build build --target llama-server llama-cli -j "$(nproc)"

# Apple Silicon
# cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
# cmake --build build --target llama-server llama-cli -j "$(sysctl -n hw.ncpu)"
```

`-DLLAMA_CURL=OFF` is deliberate: it removes the dependency on the Hugging Face download path inside llama.cpp, since you download the file yourself in the next step.

### 2. Download the model

```bash
python3 -m venv venv && . venv/bin/activate
pip install -U huggingface_hub

hf download hotdogs/Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc \
    --local-dir models
```

Interrupted? Re-run the same command — it resumes. Add `HF_TOKEN=hf_...` if the repo were private.

For vision as well:

```bash
hf download hotdogs/Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc \
    Ternary-Bonsai-2-27B-mmproj-BF16.gguf --local-dir models
```

### 3. Run it

```bash
./llama.cpp/build/bin/llama-server \
  -m models/Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc-PQ2_0.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 32768 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --spec-type draft-  --spec-draft-n-max 2
```

Open <http://127.0.0.1:8080> for the built-in chat UI (file uploads, model switching, `/props`, and the OpenAI-compatible API on the same port).

### 4. Smoke test

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in Thai and in one sentence explain what   speculative decoding does."}],'
       "max_tokens":256,"temperature":1.0,"top_p":0.95}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print("---"); print(d.get("usage"))'
```

If the server starts but the model never answers, you are almost certainly running mainline llama.cpp, not the PrismML fork.

---

## Running the server

`install.sh` writes `~/bonsai/start.sh`, which auto-detects the backend and picks a context size:

| Detection | Flags added | Default context |
|---|---|---|
| NVIDIA GPU with ≥ 22 GB VRAM | `-ngl 99 --cache-type-k q8_0 --cache-type-v q8_0` +   on | 65,536 |
| NVIDIA GPU with ≥ 11 GB VRAM | same | 32,768 |
| NVIDIA GPU, less VRAM | same | 16,384 |
| Apple Silicon with ≥ 24 GB RAM | `-ngl 99`,   off | 32,768 |
| Apple Silicon, less RAM | `-ngl 99`,   off | 16,384 |
| CPU | `-ngl 0`, no KV quant,   off | 8,192 |

Override anything from the environment or append extra flags:

```bash
PORT=9000 HOST=0.0.0.0 CTX=65536  =0 ./start.sh      # expose on the LAN, no  
./start.sh --temp 0.7 --top-k 40 --alias bonsai        # extra llama-server flags
```

**Security:** the default is `HOST=127.0.0.1` on purpose. `HOST=0.0.0.0` exposes an unfiltered, unauthenticated model to your whole network — only do it behind a firewall or reverse proxy you control.

---

## Flag reference

| Flag | Why it is there |
|---|---|
| `-m <file>` | The GGUF. Use the `-PQ2_0.gguf` variant unless you specifically want the PTQ1_0 pack. |
| `-ngl 99` | Offload every layer. The model is small enough for one GPU at moderate context. |
| `-c 32768` | Context size. Up to 262,144 is supported; memory grows with it. |
| `-fa on` | Flash attention. Faster and lower memory; required for the quantised KV cache to be worthwhile. |
| `--cache-type-k q8_0 --cache-type-v q8_0` | 8-bit KV cache — roughly halves KV memory versus F16. Drop both if you see quality drift on long contexts. |
| `--jinja` | Use the GGUF's embedded chat template (needed for tools, vision tags and `reasoning_effort`). |
| `--chat-template-kwargs '{"reasoning_effort":"medium"}'` | Controls thinking length. Values: `xhigh` (base default), `medium`, `low`. See below. |
| `--spec-type draft-  --spec-draft-n-max 2` | Speculative decoding with the   head. Remove the `--spec-*` flags for CPU/Metal. |
| `--mmproj <file>` | Adds the vision projector (see [Vision](#vision)). |
| `-t N` | CPU threads when you are not fully offloading. |
| `--parallel 1` | Keep at 1 when benchmarking; raise for multi-user serving. |

Leave the **sampler at the file's defaults** — `temperature 1.0`, `top_k 20`, `top_p 0.95`. They are baked into the GGUF's metadata (`general.sampling.*`) and were the settings the base model was evaluated at.

---

## Reasoning effort and sampling

The chat template exposes three efforts. Budget matters more than anything else here.

| `reasoning_effort` | Behaviour | When to use |
|---|---|---|
| `xhigh` (template default) | Longest thinking, most careful | Hard prompts, offline/batch work, no token pressure |
| **`medium`** | Balanced | **Default recommendation.** Best quality per token |
| `low` | Brief thinking | Chat, short answers, latency-sensitive serving |

The Swift-1.5 fine-tune's whole purpose is to make reasoning cheaper — it was measured to cut refusals from 98/100 to 23/100 and to close reasoning earlier. At scale **0.7** you get a strong share of that effect, so `medium` or `low` effort is the sensible default; `xhigh` is available when you have budget and want the base's longest thinking.

The base model's own measurements (RTX 3090, thinking on, file sampler defaults, 4,096-token budget) are the reason `medium` is the default in `start.sh`:

| Setting (base v2, 150 hard prompts) | Score | Ran out of budget |
|---|---|---|
| default effort, 16,384 tokens | 0.941 | — |
| default effort, 4,096 tokens | 0.513 | 110 / 150 |
| **`medium`, 4,096 tokens** | **0.977** | 0 |

The same trap exists upstream in Bonsai 2: with a tight budget, default effort spends the whole budget inside the reasoning block and never emits the answer. **If you cap `max_tokens` low, set `reasoning_effort` to `medium` or `low`.** If you want to disable thinking entirely, pass `enable_thinking=false` in the same `--chat-template-kwargs` object.

---

## Hardware and context sizing

Rough guide for the `-PQ2_0.gguf` file (6.67 GiB of weights). Add KV cache on top, but note the cache is cheap here: only **16 of the 64 blocks are full attention** (`full_attention_interval = 4`); the other 48 are GDN linear attention with a **fixed-size state that does not grow with context**. A full-attention token costs `4 KV heads × 256 (K) + 4 KV heads × 256 (V)` = 2,048 values = **64 KiB at F16 / ~34 KiB at q8_0**, so:

| Context | KV at F16 | KV at q8_0 (+ weights) |
|---|---|---|
| 32k | 2.15 GB | **1.14 GB** (≈ 7.8 GB total) |
| 65k | 4.29 GB | **2.28 GB** (≈ 9.0 GB total) |
| 131k | 8.59 GB | **4.56 GB** (≈ 11.2 GB total) |

The weights dominate, not the context — which is why a 12 GB card can hold a useful context and a 24 GB card can go very long. The only context-independent extra memory is the GDN state, which scales with `--parallel`, not with `-c`.

| GPU / memory | Context | Notes |
|---|---|---|
| 8 GB (RTX 3060 Ti / 4060) | not fully offloadable | 6.67 GiB of weights + ~0.5 GB of compute buffers will not fit. Partial offload (`-ngl 24`), or CPU |
| 12 GB (RTX 3060 / 4070) | 16k, up to 65k | 6.67 GiB weights + 1.14 GB KV at 32k + ~0.6 GB buffers ≈ 8.4 GiB — comfortable. 65k needs q8_0 KV and a small `--ubatch-size` |
| 16 GB (RTX 4060 Ti 16G / 4080 mobile) | 65k | Good daily driver |
| 24 GB (RTX 3090 / 4090) | 131k–196k | Where   pays off most |
| 48 GB+ (A6000, 2×3090, 5090 cf.) | 262k | Full context, `--parallel` possible |
| CPU only (32 GB+ RAM) | 8k | Works, but expect single-digit tok/s. A 27B at 2.13 bpw still has to move 7 GB per token batch. |

One buffer deserves a note because the vocabulary is 248,320 tokens: the output buffer is roughly `n_ubatch × 248,320 × 4 bytes` ≈ 0.5 GB at the default 512 ubatch. Lower it (`--ubatch-size 128`) if you are fighting for the last few hundred MB of VRAM; it costs a little speed.

  adds a   head (0.42B params). Budget an extra ~0.5 GB VRAM when it is enabled.

---

## Calling the API

`llama-server` speaks the OpenAI API. Note the `"model"` field is ignored — one server, one model — but clients still expect it.

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bonsai",
    "messages": [
      {"role": "system", "content": "You are a concise, direct assistant."},
      {"role": "user", "content": "อธิบายความต่างระหว่าง   กับ Medusa แบบสั้น ๆ"}
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 1024,
    "chat_template_kwargs": {"reasoning_effort": "medium"}
  }' | python3 -m json.tool
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="bonsai",
    messages=[{"role": "user", "content": "Write a Rust function that parses a /proc/meminfo dump."}],
    temperature=1.0, top_p=0.95, max_tokens=2048,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}},
)
print(r.choices[0].message.content)
```

Streaming: add `"stream": true`. Token accounting: the `usage` block reports `prompt_tokens`, `completion_tokens` including reasoning tokens.

---

## Docker

CUDA image with the PrismML fork built in:

```bash
docker run --gpus all --rm -p 8080:8080 \
  -v "$HOME/bonsai/models:/models:ro" \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc-PQ2_0.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 32768 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --spec-type draft-  --spec-draft-n-max 2
```

The upstream image is **mainline** llama.cpp and will fail to load PQ2_0. Build your own from the fork:

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y git cmake build-essential && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch prism-b10743-adfffbe \
      https://github.com/PrismML-Eng/llama.cpp /src
RUN cmake -S /src -B /src/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /src/build --target llama-server -j"$(nproc)"
EXPOSE 8080
ENTRYPOINT ["/src/build/bin/llama-server", "--host", "0.0.0.0", "--port", "8080"]
```

---

## Vision

Both GGUFs carry the vision token definitions in their chat template, and this repo ships the matching projector:

```bash
./llama.cpp/build/bin/llama-server \
  -m models/Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc-PQ2_0.gguf \
  --mmproj models/Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
  -ngl 99 -c 32768 -fa on --jinja
```

Then send an OpenAI-style image content part:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}},
    {"type": "text", "text": "อธิบายภาพนี้เป็นภาษาไทย"}
  ]}],
  "max_tokens": 512
}'
```

The projector is BF16 and untested here — the base model card explicitly declares no vision evaluation, so treat vision output as unreviewed.

---

## How this build was made

### Step 1 — the base

[`OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF`](https://huggingface.co/OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF), which is PrismML's ternary Bonsai 2 (itself derived from Qwen3.8-27B) with the **Heretic LoRA baked in** by adjusting ternary codes while preserving the original block scales. 34 matrices were updated; the remaining 817 tensors were unchanged. This is an approximate merge, not an exact floating-point LoRA merge.

- **Refusal removal:** 0/100 refusals (was 95/100 in the original), KL divergence 0.0135.
- **Perplexity (WikiText-2 test, 512-token context, 4,080 tokens):** PQ2_0 = 10.1460, PTQ1_0 = 10.1435.
- **Japanese (harmless_alpaca_ja test):** PQ2_0 = 16.7833, PTQ1_0 = 16.7744.
- **16/16 short functional checks passed** on both packings (arithmetic, JSON, translation, reading comprehension).

### Step 2 — the Swift adapter

[`hotdogs/swift_unc_qwen3.8-27B_lora`](https://huggingface.co/hotdogs/swift_unc_qwen3.8-27B_lora) is a **weight-diff SVD** extraction: given two checkpoints with identical architecture, compute `Δ = W_swift_unc − W_qwen3.8` per tensor, take a truncated SVD of the largest ones, and store `lora_A = Σ_r^½ V_rᵀ`, `lora_B = U_r Σ_r^½`. Nothing is trained; the delta is compressed.

The Swift-1.5-Qwen3.8-27B-Uncensored model is an **abliterated** derivative of [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) (UkisAI's reasoning-efficiency fine-tune of Qwen3.8-27B), with the refusal direction `r` (from [orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored), Arditi et al. 2024) applied to 131 tensors via `W' = W − r(rᵀW)` (plus `embed_tokens`).

| Parameter | Value |
|---|---|
| Rank | 16 (uniform, exact truncated SVD) |
| Tensors | 204 (`lora_A`/`lora_B` pairs, float32) |
| Refusals (Heretic eval) | 98/100 → 23/100 |
| KL divergence | 0.0884 |
| Reconstruction error (40-tensor sample) | mean 0.37 |

### Step 3 — the merge

The Swift adapter was merged into the Heretic base at **scale 0.25** (70% of the adapter's full effect). This is a *blend*, not a substitution: the result sits between the Heretic base behaviour (abliterated) and the Swift-1.5 uncensored fine-tune.

> Note: the Heretic base is **already abliterated** (the abliteration is in the released ternary packing). The Swift adapter's own abliteration component (the `self_attn.o_proj`, `linear_attn.out_proj`, `mlp.down_proj` deltas) therefore partially duplicates an edit that is already present — a mild double-application of the same direction, not a new one.

---

## What to expect, and what is honest about it

**What should carry over**

- Refusal behaviour: the base's abliteration is in the released ternary packing, and the Swift adapter's abliteration component reconstructs at 96–99% energy at rank 16.
- Packaging: everything else is PrismML's release, requantised nowhere — PQ2_0 and the Hadamard latent are untouched, which is why the fork requirement and   behaviour are identical to the base.

**What is approximate, and you should expect it to be**

- The Swift adapter is not a lossless copy of its source model. The Swift adapter approximates its `mlp.gate_proj` / `mlp.up_proj` deltas at ~60–66% retained energy.
- At scale **0.25** you get roughly 70% of the Swift adapter's effect. Reasoning traces should be noticeably shorter than the base (that is Swift-1.5's whole purpose), but do **not** expect the full Swift personality or the full abliteration effect from the Swift side.
- Both adapters were extracted against **stock Qwen3.8-27B**, while this base is **already abliterated**. Their abliteration components therefore partially duplicate an edit that is already present.
- **No benchmark numbers are claimed for this merged build.** The tables in the base model's card (refusals 0/100, KL 0.0135, perplexity 10.14, 16/16 functional checks) describe the *base*, measured by OS-Software. Merging the adapter at 0.7 moves weights, so treat those figures as indicative, not as measurements of this file. If you need numbers for this build, run your own harness and publish them — the card will be updated when someone does.
- Vision is included but untested (see [Vision](#vision)).
- CUDA is the only backend the base was validated on. Metal and CPU are untested upstream.

**Speed:** the   head should give the same ~40% decode gain as the base, since the head and the draft/verify path are unchanged. Gains depend on the content: reasoning, code and JSON accept drafts at a much higher rate than free prose, where speculation can be neutral. Measure it yourself with `--parallel 1`, greedy, and compare.

---

## คู่มือภาษาไทย

### นี่คืออะไร

โมเดล **Ternary Bonsai 2 27B** เวอร์ชัน **heretic abliterated** ของ PrismML/OS-Software นำ **LoRA ตัวเดียวมารวมที่สเกล 0.25**:

- [`hotdogs/swift_unc_qwen3.8-27B_lora`](https://huggingface.co/hotdogs/swift_unc_qwen3.8-27B_lora) — การปลดการปฏิเสธ + การปรับให้คิดสั้นลงแบบ Swift-1.5 (rank 16, 204 tensors)

ขนาดไฟล์ประมาณ **6.7 GB** ทำงานบนการ์ดจอเดียวได้ รองรับบริบทสูงสุด **262,144 โทเคน** และ มี **  head** ช่วยเร่งความเร็ว

> ⚠️ **ต้องใช้งาน fork ของ PrismML** — ไฟล์นี้ใช้ฟอร์แมต **PQ2_0** ซึ่งมีเฉพาะใน fork ของ PrismML เท่านั้น

### ติดตั้งแบบอัตโนมัติ (แนะนำ)

```bash
# 1) บันทึกสคริปต์ด้านบนเป็น install.sh  หรือดาวน์โหลด install.sh จาก repo นี้
# 2) รัน
bash install.sh

# 3) เปิดเซิร์ฟเวอร์
~/bonsai/start.sh
# แล้วเปิดเบราว์เซอร์ที่ http://127.0.0.1:8080
```

สคริปต์จะทำให้ทุกอย่าง: ตรวจ OS/CPU/RAM/การ์ดจอ → ติดตั้งเครื่องมือที่ขาด → clone และ build fork ของ PrismML → ดาวน์โหลดโมเดล → สร้าง `start.sh` ให้

รันซ้ำได้ไม่พัง: ถ้า build เสร็จแล้วหรือไฟล์โมเดลครบแล้ว มันจะข้ามขั้นตอนนั้นไป และถ้าดาวน์โหลดหลุดกลางทาง รันซ้ำจะโหลดต่อให้

ปรับค่าได้ด้วย environment variable:

```bash
INSTALL_DIR=/data/bonsai bash install.sh    # เปลี่ยนที่ติดตั้ง
FORCE_CPU=1 bash install.sh                 # บังคับ CPU แม้มีการ์ดจอ
HF_TOKEN=hf_xxx bash install.sh             # ถ้า repo เป็น private
```

### ติดตั้งเองทีละขั้น (สรุป)

```bash
# 1) build fork ของ PrismML (ต้องเป็น tag prism-b10743-adfffbe ขึ้นไป)
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp
git checkout prism-b10743-adfffbe
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF
cmake --build build --target llama-server llama-cli -j"$(nproc)"

# 2) ดาวน์โหลดโมเดล
python3 -m venv venv && . venv/bin/activate && pip install -U huggingface_hub
hf download hotdogs/Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc --local-dir models

# 3) เปิดเซิร์ฟเวอร์
./build/bin/llama-server -m models/Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc-PQ2_0.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --spec-type draft-  --spec-draft-n-max 2
```

`CMAKE_CUDA_ARCHITECTURES` ใส่ตามการ์ดของคุณ: RTX 3060/3090 = `86`, RTX 4090 = `89`, RTX 5090 = `120`, A100 = `80`

### คำแนะนำการตั้งค่า

1. **วาง sampler ไว้ที่ค่าของไฟล์** — `temperature 1.0`, `top_k 20`, `top_p 0.95` (ฝังอยู่ใน metadata ของ GGUF แล้ว)
2. **ถ้าจำกัด `max_tokens` ให้ตั้ง `reasoning_effort` เป็น `medium` หรือ `low`** — ไม่งั้นโมเดลจะคิดจนหมดโควตาแล้วไม่ทันได้ตอบ (ที่ 4,096 โทเคน คะแนนต่างกันกว่าเท่าตัว: 0.513 → 0.977)
3. **ค่าเริ่มต้นของ `start.sh` ตั้งบริบทตามการ์ดจอให้อัตโนมัติ**: VRAM ≥ 22GB → 65,536 / ≥ 11GB → 32,768 / น้อยกว่า → 16,384
4. **อย่าตั้ง `HOST=0.0.0.0`** ถ้าไม่ได้อยู่หลัง firewall — เซิร์ฟเวอร์ไม่มีระบบล็อกอิน

### ความคาดหวังที่ตรงไปตรงมา

- นี่คือการ **ผสม** ไม่ใช่การแทนที่: ได้อิทธิพลจาก LoRA Swift 70% บวกกับพฤติกรรมของ base ที่ปลดการปฏิเสธแล้ว
- LoRA เป็น *การประมาณ* ไม่ใช่สำเนาเต็มรูปแบบ (gate/up deltas เก็บพลังงานได้แค่ ~60–66% ที่ rank 16)
- **ยังไม่มีตัวเลข benchmark ของโมเดลที่รวมแล้วนี้** — ตัวเลขใน card ของ base (refusals 0/100, KL 0.0135, perplexity 10.14) เป็นของ base ก่อนรวม LoRA
- การรวม LoRA **ไม่แตะ** packaging — PQ2_0 และ Hadamard latent ยังเหมือน base

---

## Safety

This model has had its refusal behaviour deliberately removed, and the merge adds further uncensored-fine-tune influence at 0.25. It **will** comply with requests that a stock model declines. It is published for research on alignment robustness, red-teaming, and users who need an unfiltered local model.

You are responsible for what you do with it. Do not put it in a user-facing product without your own safety and moderation layer. Do not use it to cause harm to people. Local does not mean consequence-free.

Underlying licenses: base weights and PrismML's release are Apache-2.0; the Swift adapter derives from a model under the **Swift Open License v1.0** (free below US$1M annual revenue, enterprise licence above). If you plan commercial use, check the upstream terms for the constituent models, not just this card's `license:` tag.

---

## Credits

- **Base model:** [`OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF`](https://huggingface.co/OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF) — Heretic LoRA baked into the ternary weights, abliteration parameters, and the benchmarks quoted above
- **Ternary base + runtime:** [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (MIT) for the PQ2_0 format, Hadamard latent, and   support
- **Original model:** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)
- **Merged adapter:** [`hotdogs/swift_unc_qwen3.8-27B_lora`](https://huggingface.co/hotdogs/swift_unc_qwen3.8-27B_lora) (from [ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored- ](https://huggingface.co/ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored- ) ← [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) + [orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored), [Arditi et al. 2024](https://arxiv.org/abs/2406.11717))

Not affiliated with or endorsed by PrismML, Qwen, BoldingBuilds, OS-Software, UkisAI, or ajgazin.

## Citation

```bibtex
@misc{hotdogs_ternary_bonsai_swift,
  title  = {Ternary Bonsai 2 27B Uncensored Heretic + Swift (PQ2\_0,  )},
  author = {hotdogs},
  year   = {2026},
  note   = {Swift LoRA merged at scale 0.25 on top of
            OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF},
  url    = {https://huggingface.co/hotdogs/Ternary-Bonsai-2-27B-Uncensored-Heretic-swift_unc}
}
```
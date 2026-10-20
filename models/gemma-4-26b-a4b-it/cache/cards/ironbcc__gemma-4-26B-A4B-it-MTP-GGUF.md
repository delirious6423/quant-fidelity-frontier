---
license: apache-2.0
license_link: https://ai.google.dev/gemma/terms
library_name: gguf
pipeline_tag: text-generation
language:
  - en
base_model:
  - google/gemma-4-26B-A4B-it
  - google/gemma-4-26B-A4B-it-assistant
base_model_relation: quantized
quantized_by: ironbcc
tags:
  - gguf
  - gemma4
  - mtp
  - speculative-decoding
  - moe
extra_gated_heading: Access Gemma on Hugging Face
extra_gated_prompt: To access Gemma on Hugging Face, you're required to review and agree to Google's usage license. Please ensure you're logged in.
extra_gated_button_content: Acknowledge license
---

# gemma-4-26B-A4B-it MTP GGUF

Speculative-decoding bundle for `google/gemma-4-26B-A4B-it` using Google's official MTP drafter (`gemma-4-26B-A4B-it-assistant`), packaged for `llama.cpp` and **LM Studio**.

Achieves ~**1.55x throughput** vs no-MTP on Apple M5 Max (Q2_K drafter, n=3, thinking-off).

## Files

| File | Role | Size |
|---|---|---|
| `gemma-4-26B-A4B-it-Q8_0.gguf` | target (26B-A4B MoE, 128 experts) | 26 GB |
| `gemma-4-26B-A4B-it-assistant-Q2_K.gguf` | drafter (recommended) | 278 MB |
| `gemma-4-26B-A4B-it-assistant-Q4_K_M.gguf` | drafter (balanced) | 310 MB |
| `gemma-4-26B-A4B-it-assistant-Q8_0.gguf` | drafter (high-precision) | 440 MB |
| `gemma-4-26B-A4B-it-assistant-F16.gguf` | drafter (reference) | 816 MB |

Target re-hosted from [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) (Unsloth Dynamic 2.0 quant, Apache-2.0). Drafter built from [`google/gemma-4-26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant) via [llama.cpp PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398) (am17an, WIP).

## Requirements

llama.cpp built from am17an's `gemma4-mtp` branch — Gemma-4 MTP not yet merged to master.

```bash
git clone -b gemma4-mtp https://github.com/am17an/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build -j --target llama-server llama-quantize
```

## Run

```bash
llama-server \
  -m  gemma-4-26B-A4B-it-Q8_0.gguf \
  -md gemma-4-26B-A4B-it-assistant-Q2_K.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 99 -c 8192 -fa on \
  --host 0.0.0.0 --port 8080
```

Or pull directly:
```bash
llama-server \
  -hf ironbcc/gemma-4-26B-A4B-it-MTP-GGUF \
  -hfd ironbcc/gemma-4-26B-A4B-it-MTP-GGUF:Q2_K \
  --spec-type draft-mtp --spec-draft-n-max 3 -ngl 99 -fa on
```

## LM Studio

1. Search this repo, download target + drafter.
2. Load target.
3. Load settings → Speculative Decoding → select drafter file.

(Requires LM Studio with am17an's PR merged or custom llama.cpp runtime. As of 2026-05, mainline LM Studio runtime doesn't yet have `draft-mtp` for Gemma-4 — track upstream merge.)

## Client request — recommended

```json
{
  "messages": [{"role": "user", "content": "Your prompt"}],
  "temperature": 1.0, "top_p": 0.95, "top_k": 64,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

`enable_thinking: false` skips `<|channel>thought` block. **+12% throughput**, +9 pp accept rate, direct answer.
For reasoning tasks (math, code review), enable thinking — slower but better.

## Benchmark (Mac M5 Max, 500-token gen, long prompt)

| Config | tok/s | Accept |
|---|---|---|
| baseline (no MTP) | 92.5 | — |
| MTP Q8_0 drafter | 121.7 | 65.9% |
| MTP Q4_K_M drafter | 127.9 | 67.7% |
| **MTP Q2_K drafter** | **131.2** | **70.3%** |
| MTP + thinking off | **143.2** | **74.6%** |

Lower drafter quant = faster (less bandwidth) and counterintuitively higher accept (quantization rounds toward target argmax).

## Use with Hermes (CLI agent)

Hermes uses LM Studio's native `/api/v1/models` for model validation, which `llama-server` doesn't expose. A 60-line Python shim adapts it.

### 1. Launcher — `mtp-server.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
GGUF=~/gemma4-build/gguf
LLAMA=~/gemma4-build/llama.cpp/build/bin/llama-server
exec "$LLAMA" \
  -m  "$GGUF/gemma-4-26B-A4B-it-Q8_0.gguf" \
  -md "$GGUF/gemma-4-26B-A4B-it-assistant-Q2_K.gguf" \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 99 -c 262144 -fa on \
  --host 127.0.0.1 --port 8080 \
  --alias "gemma-4-26B-A4B-it-MTP"
```

`-c 262144` = full 256K context (Gemma-4 max). RSS ~31 GB at boot, grows w/ usage. Drop to `-c 65536` if low-memory.

### 2. LM Studio shim — `lmstudio-shim.py`

```python
#!/usr/bin/env python3
"""Reverse proxy adapting llama-server to LM Studio's native API for Hermes."""
import http.server, json, socketserver, urllib.request, urllib.error, os
UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8080")
PORT = int(os.environ.get("PORT", "8081"))

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _proxy(self, method):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method=method)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "transfer-encoding"):
                req.add_header(k, v)
        try:
            r = urllib.request.urlopen(req, timeout=600)
            self.send_response(r.status)
            data = r.read()
            for k, v in r.headers.items():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _synth_models(self):
        with urllib.request.urlopen(UPSTREAM + "/v1/models", timeout=5) as r:
            src = json.loads(r.read())
        out = []
        for m in src.get("data", []):
            meta = m.get("meta", {}) or {}
            out.append({
                "id": m["id"], "type": "llm", "publisher": "ironbcc",
                "arch": "gemma4", "compatibility_type": "gguf",
                "quantization": "Q8_0", "state": "loaded",
                "max_context_length": meta.get("n_ctx_train", 8192),
                "loaded_context_length": meta.get("n_ctx", 8192),
                "capabilities": {
                    "reasoning": {"allowed_options": ["off", "low", "medium", "high"]},
                    "chat": True, "tool_use": True,
                },
            })
        r = json.dumps({"object": "list", "data": out, "models": out}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(r)))
        self.end_headers()
        self.wfile.write(r)

    def do_GET(self):
        if self.path.startswith("/api/v1/models"): return self._synth_models()
        self._proxy("GET")
    def do_POST(self):
        if self.path.startswith("/api/v1/models/load"):
            r = b'{"ok": true, "already_loaded": true}'
            self.send_response(200); self.send_header("Content-Length", str(len(r)))
            self.end_headers(); self.wfile.write(r); return
        self._proxy("POST")
    def do_DELETE(self): self._proxy("DELETE")
    def do_PUT(self):    self._proxy("PUT")

class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True

if __name__ == "__main__":
    print(f"lmstudio-shim :{PORT} -> {UPSTREAM}", flush=True)
    S(("127.0.0.1", PORT), H).serve_forever()
```

Synthesizes `GET /api/v1/models` from llama-server's `/v1/models`, no-ops `POST /api/v1/models/load`, proxies everything else.

### 3. Start both

```bash
nohup ./mtp-server.sh    >/tmp/mtp-srv.log  2>&1 & disown
nohup python3 ./lmstudio-shim.py >/tmp/mtp-shim.log 2>&1 & disown
```

Verify:
```bash
curl -s http://127.0.0.1:8081/api/v1/models | jq '.models[].id'
# -> "gemma-4-26B-A4B-it-MTP"
```

### 4. Add to `~/.hermes/config.yaml`

```yaml
model_aliases:
  gemma-4-mtp: { model: gemma-4-26B-A4B-it-MTP, provider: lmstudio, base_url: http://127.0.0.1:8081/v1 }
```

Optional default:
```yaml
model:
  default: gemma-4-26B-A4B-it-MTP
  provider: lmstudio
  base_url: http://127.0.0.1:8081/v1
```

### 5. Run

```bash
hermes --model gemma-4-mtp
```

Hermes hits `:8081/api/v1/models` for validation → shim returns LM Studio shape with our model id → Hermes accepts → chat goes to `:8081/v1/chat/completions` → shim proxies to llama-server on `:8080`.

### Stop

```bash
pkill -f "llama-server|lmstudio-shim"
```

## License

Apache-2.0 (inherits from Google Gemma terms). Accept the Gemma license on HF before download.

## Attribution

- `google/gemma-4-26B-A4B-it` — base model, Google.
- `google/gemma-4-26B-A4B-it-assistant` — MTP drafter, Google.
- `unsloth/gemma-4-26B-A4B-it-GGUF` — target Q8_0, Unsloth (Dynamic 2.0 quant).
- llama.cpp MTP infra: am17an (Aman Gupta), PR #23398.

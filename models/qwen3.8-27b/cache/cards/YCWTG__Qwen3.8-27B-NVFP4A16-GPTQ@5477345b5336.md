---
library_name: transformers
base_model:
- Qwen/Qwen3.8-27B
tags:
  - qwen3.8
  - dense
  - GPTQ
  - NVFP4A16
  - quantized
  - instruct
license: apache-2.0
pipeline_tag: image-text-to-text
---

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/685e122d50df66f41587d406/XU7ovrDdgsNFAzahuyvI1.png" alt="Tone">
</p>

Language [中文](https://huggingface.co/YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ/blob/main/README_zh.md)|English
## Model Details

This model is an **NVFP4A16 quantized** version of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) generated with [llm-compressor](https://github.com/vllm-project/llm-compressor). The **MTP module has been removed** to reduce the model size so it can fit on GPUs with 32 GB of VRAM, such as the RTX 5090.
Please follow the license of the original model.

This model uses Thinking mode by default. To enable Instruct mode, please add `{%- set enable_thinking = false %}` in [chat_template.jinja](https://huggingface.co/YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ/blob/main/chat_template.jinja), and make sure --reasoning-parser qwen3 has been removed from the model launch command. Like [Instruct-mode](#instruct-mode).

### Quantization Strategy

| Layer Type      | Precision | Description                                     |
| --------------- | :-------- | ----------------------------------------------- |
| `lm_head`       |    16-bit | Keep the output projection at full precision   |
| `.*visual.*`    |    16-bit | Keep the vision encoder at full precision      |
| `linear_attn.*` |    16-bit | Keep linear attention layers at full precision |
| `mtp.*`         |    16-bit | Remove MTP prediction layers    |

### Model Comparison

<style>
.vl-table th{
    font-size:15px!important;
    line-height:1.2;
    text-align:left!important;
}

.vl-table td:not(.benchmark-cell):not([colspan]){
    font-size:15px;
    line-height:1.2;
    vertical-align:middle;
    text-align:left!important;
}

.vl-table .benchmark-cell{
    padding:12px 10px 12px 18px!important;
    vertical-align:middle;
    text-align:left!important;
}

.vl-table .benchmark-capability{
    font-size:15px;
    font-weight:600;
    line-height:1.22;
    color:#171717;
}

.vl-table .benchmark-name{
    margin-top:4px;
    font-size:11px;
    font-weight:400;
    line-height:1.2;
    color:#6B6B6B;
}

.vl-table .metric-label{
    font-size:10px;
    font-weight:400;
    line-height:1.1;
    color:#777;
    margin-bottom:4px;
}

.vl-table .metric-value{
    font-size:15px;
    line-height:1.15;
    color:#171717;
}
</style>

<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1200px;margin:0;padding:16px 0">

<table class="vl-table" style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px">

<thead>
<tr>

<th style="
    width:30%;
    padding:10px 7px;
    text-align:left;
    font-weight:600;
    border-bottom:2px solid #0A2EFE;
    color:#0A2EFE;
"></th>

<th style="
    width:40%;
    padding:10px 7px;
    text-align:left;
    font-weight:500;
    border-bottom:2px solid #0A2EFE;
    color:#0A2EFE;
    font-size:14px;
    background:rgba(10,46,254,0.08);
">
Qwen3.8-27B-NVFP4A16-GPTQ
</th>

<th style="
    width:40%;
    padding:10px 7px;
    text-align:left;
    font-weight:500;
    border-bottom:2px solid #0A2EFE;
    color:#0A2EFE;
    font-size:14px;
">
Qwen3.8-27B
</th>

</tr>
</thead>

<tbody>

<tr>

<td class="benchmark-cell" style="
    padding:7px 7px;
    padding-left:20px;
    text-align:left;
    border-bottom:1px solid rgba(128,128,128,0.15);
">
    <div class="benchmark-capability">Model Size</div>
</td>

<td style="
    padding:7px 7px;
    text-align:left;
    border-bottom:1px solid rgba(128,128,128,0.15);
    background:rgba(10,46,254,0.08);
    vertical-align:middle;
">
    <div class="metric-label">50%↓↓</div>
    <div class="metric-value"><strong>27.7 GB</strong></div>
</td>

<td style="
    padding:7px 7px;
    text-align:left;
    border-bottom:1px solid rgba(128,128,128,0.15);
    vertical-align:middle;
">
    <div class="metric-value">55.6 GB</div>
</td>

</tr>

<tr>

<td class="benchmark-cell" style="
    padding:7px 7px;
    padding-left:20px;
    text-align:left;
    border-bottom:1px solid rgba(128,128,128,0.15);
">
    <div class="benchmark-capability">Multidisciplinary reasoning</div>
    <div class="benchmark-name">HLE</div>
</td>

<td style="
    padding:7px 7px;
    text-align:left;
    border-bottom:1px solid rgba(128,128,128,0.15);
    background:rgba(10,46,254,0.08);
    vertical-align:middle;
">
    17.0
</td>

<td style="
    padding:7px 7px;
    text-align:left;
    border-bottom:1px solid rgba(128,128,128,0.15);
    vertical-align:middle;
">
    24.3
</td>

</tr>

</tbody>
</table>

</div>



## Quickstart

### vLLM Usage
[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient inference and serving engine for LLMs.


Directly talk to the model

With vLLM already installed, create a file named `example.py`, copy the example code below into it, and then run ```python example.py``` in terminal.

```python
import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


# Configuration
DEFAULTS = {
    "model": "YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ",
    "served_model_name": "YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ",
    "host": "localhost",
    "port": 8000,
    "max_model_len": 29632,
    "enable_auto_tool_choice": True,
    "tool_call_parser": "qwen3_coder",
    "max_num_seqs": 1,
    "reasoning_parser": "qwen3",
    "default_chat_template_kwargs": '{"enable_thinking": true}',
    "allowed_local_media_path": "/home/ycwtg/image",
}

RUNTIME = {
    "gpu_memory_utilization": 0.98,
    "startup_timeout_sec": 1800,
    "healthcheck_timeout_sec": 3,
    "healthcheck_interval_sec": 1,
    "chat_timeout_sec": 600,
}

# The API is always local; global HTTP_PROXY settings must not intercept it.
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

SERVE_VALUE_ARGS = (
    "served_model_name", "host", "port", "max_model_len",
    "tool_call_parser", "max_num_seqs", "reasoning_parser",
    "default_chat_template_kwargs",
)
CLIENT_VALUE_ARGS = ("model", *SERVE_VALUE_ARGS)
BOOL_ARGS = ("enable_auto_tool_choice",)


def cli_flag(name):
    return "--" + ("max_num_seqs" if name == "max_num_seqs" else name.replace("_", "-"))


def value_options(args, names):
    return [part for name in names for part in (cli_flag(name), str(getattr(args, name)))]


def boolean_options(args, explicit_false=False):
    return [
        cli_flag(name) if getattr(args, name) else "--no-" + cli_flag(name)[2:]
        for name in BOOL_ARGS
        if explicit_false or getattr(args, name)
    ]


def multiline_input():
    print('User (type "END" on a single line to send, "exit" to quit):')
    lines = []
    while True:
        line = input()
        text = line.strip()
        if text.lower() in {"exit", "quit"}:
            return None
        if text == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def resolve_client_host(host):
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def launch_vllm(args):
    vllm = shutil.which("vllm", path=os.path.dirname(sys.executable)) or shutil.which("vllm")
    if not vllm:
        raise RuntimeError("vllm command not found. Activate an environment that has vllm installed.")

    cmd = [vllm, "serve", args.model, *value_options(args, SERVE_VALUE_ARGS)]
    media_path = args.allowed_local_media_path
    if media_path is not None and (not isinstance(media_path, str) or media_path.strip()):
        cmd += ["--allowed-local-media-path", str(media_path)]
    cmd += ["--gpu-memory-utilization", str(RUNTIME["gpu_memory_utilization"]), *boolean_options(args)]

    print("Launching vLLM:")
    print(" ".join(cmd))
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(vllm) + os.pathsep + env.get("PATH", "")
    try:
        return subprocess.Popen(cmd, env=env)
    except FileNotFoundError as e:
        raise RuntimeError("vllm command not found. Activate an environment that has vllm installed.") from e


def stop_vllm(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def wait_vllm_ready(base_url, timeout_sec=RUNTIME["startup_timeout_sec"], proc=None):
    deadline = time.time() + timeout_sec
    req = urllib.request.Request(url=f"{base_url}/v1/models")
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            return False
        try:
            with LOCAL_HTTP.open(req, timeout=RUNTIME["healthcheck_timeout_sec"]) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        time.sleep(RUNTIME["healthcheck_interval_sec"])
    return False


def chat_once(base_url, model_name, messages):
    payload = {"model": model_name, "messages": messages, "skip_special_tokens": False}
    req = urllib.request.Request(
        url=f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with LOCAL_HTTP.open(req, timeout=RUNTIME["chat_timeout_sec"]) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]


def chat_loop(base_url, model_name):
    print("\n===== Chat Started =====\n")
    messages = []

    while True:
        user_text = multiline_input()
        if user_text is None:
            break

        messages.append({"role": "user", "content": user_text})
        try:
            assistant_msg = chat_once(base_url, model_name, messages)
        except Exception as e:
            print(f"\nRequest failed: {e}\n")
            messages.pop()
            continue

        content = assistant_msg.get("content")
        tool_calls = assistant_msg.get("tool_calls")

        if content:
            print(f"\nAssistant:\n{content}\n")
        elif tool_calls:
            print("\nAssistant(tool_calls):")
            print(json.dumps(tool_calls, ensure_ascii=False, indent=2))
            print()
        else:
            print("\nAssistant:\n(empty response)\n")

        normalized_msg = {"role": "assistant", "content": content or ""}
        if tool_calls:
            normalized_msg["tool_calls"] = tool_calls
        messages.append(normalized_msg)


def build_client_command(args):
    return [
        sys.executable,
        os.path.abspath(__file__),
        "--_client",
        *value_options(args, CLIENT_VALUE_ARGS),
        *boolean_options(args, explicit_false=True),
    ]


def spawn_chat_terminal(args):
    client_cmd = build_client_command(args)

    if os.name == "nt":
        terminal_cmd = ["cmd", "/c", "start", "", "cmd", "/k", subprocess.list2cmdline(client_cmd)]
    elif shutil.which("ptyxis"):
        terminal_cmd = ["ptyxis", "--standalone", "--new-window", "--title=vLLM Chat", "--", *client_cmd]
    elif shutil.which("gnome-terminal"):
        terminal_cmd = ["gnome-terminal", "--", *client_cmd]
    elif shutil.which("x-terminal-emulator"):
        terminal_cmd = ["x-terminal-emulator", "-e", *client_cmd]
    else:
        return False

    try:
        terminal_proc = subprocess.Popen(terminal_cmd)
        if terminal_cmd[0] == "ptyxis":
            time.sleep(0.5)
            if terminal_proc.poll() is not None:
                print(f"Failed to open Ptyxis (exit code {terminal_proc.returncode}).")
                return False
        return True
    except Exception as e:
        print(f"Failed to open a new terminal automatically: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal local vLLM chat script")
    parser.add_argument("--_client", action="store_true", help=argparse.SUPPRESS)

    def add(name, *flags, **kwargs):
        parser.add_argument(
            *(flags or (f"--{name.replace('_', '-')}",)), dest=name, default=DEFAULTS[name], **kwargs
        )

    add("model")
    add("served_model_name")
    add("host")
    add("port", type=int)
    add("max_model_len", type=int)
    add("max_num_seqs", "--max-num-seqs", "--max_num_seqs", type=int)
    add("enable_auto_tool_choice", action=argparse.BooleanOptionalAction)
    add("allowed_local_media_path", help="Optional local media path. Leave empty to disable.")
    add("tool_call_parser")
    add("reasoning_parser")
    add("default_chat_template_kwargs")
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = f"http://{resolve_client_host(args.host)}:{args.port}"
    if args._client:
        print(f"Waiting for model service: {base_url}")
        if wait_vllm_ready(base_url):
            chat_loop(base_url, args.served_model_name)
        else:
            print("Model service did not become ready.")
        return

    proc = launch_vllm(args)
    atexit.register(stop_vllm, proc)
    terminal_opened = spawn_chat_terminal(args)

    print(f"Waiting for service to become ready: {base_url}")
    if not wait_vllm_ready(base_url, proc=proc):
        print(f"vLLM failed to become ready (exit code: {proc.poll()}). Check server logs above.")
        stop_vllm(proc)
        sys.exit(1)

    if terminal_opened:
        print("Model is ready. Opened a new terminal for chat; this terminal keeps server logs.")
        print("Press Ctrl+C here to stop vLLM.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("\nInterrupted. Stopping vLLM...")
    else:
        print("No supported terminal found. Falling back to chat in this terminal.")
        chat_loop(base_url, args.served_model_name)


if __name__ == "__main__":
    main()
```

Directly use the OpenAPI

### Instruct Mode

```shell
vllm serve --model YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ --served-model-name YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ --host localhost --port 8000 --max-model-len 29632 --enable-auto-tool-choice --tool-call-parser qwen3_coder --gpu-memory-utilization 0.98 --max_num_seqs 1 --allowed-local-media-path /home/ycwtg/image
```

### Thinking Mode

```shell
vllm serve --model YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ --served-model-name YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ --host localhost --port 8000 --max-model-len 29632 --enable-auto-tool-choice --tool-call-parser qwen3_coder --gpu-memory-utilization 0.98 --max_num_seqs 1 --allowed-local-media-path /home/ycwtg/image --reasoning-parser qwen3
```

### Text-Only Mode

```shell
vllm serve --model YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ --served-model-name YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ --host localhost --port 8000 --max-model-len 33245 --enable-auto-tool-choice --tool-call-parser qwen3_coder --gpu-memory-utilization 0.98 --max_num_seqs 1 --reasoning-parser qwen3  --language-model-only
```

See [its documentation](https://docs.vllm.ai/en/stable/getting_started/installation/index.html) for more details. 

The following will create API endpoints at `http://localhost:8000/v1`.

## Generate the Model

See [code here.](https://huggingface.co/YCWTG/Qwen3.8-27B-NVFP4A16-GPTQ/blob/main/generate_model.py)

## Ethical Considerations and Limitations

The model can produce factually incorrect output, and should not be relied on to produce factually accurate information. Because of the limitations of the pretrained model and the finetuning datasets, it is possible that this model could generate lewd, biased or otherwise offensive outputs.

Therefore, before deploying any applications of the model, developers should perform safety testing.


## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model. Please consult an attorney before using this model for commercial purposes.
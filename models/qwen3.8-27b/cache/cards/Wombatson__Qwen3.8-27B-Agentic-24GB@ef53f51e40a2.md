
---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- incoai/Qwen3.8-27B-DFlash2
---
# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->

This model is processed and compressed by Claude Sonnet 5.5 with high reasoning level on Hermes Desktop on 24GB M6 Mac Mini. It uses for the main model of profile using llama.cpp with Dflash2 support inside Hermes Desktop.

## Model Details

I used Qwen/Qwen3.8-27B to process. Qwen3.8-27B-DFlash2-Q4_K_M.gguf is downloaded from https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/tree/main directly. 
read my prompts and configs below:

Qwen3.8-27B dense slimming - run log
====================================

Task: convert Qwen3.8-27B (HF safetensors) to GGUF on a 24 GB Apple Silicon Mac,
extract the vision/video encoder as an mmproj Q8_0, build a custom IQ3_S mix
with an imatrix, wire it to a DFlash2 draft model, run it through the
Hermes-managed llama.cpp, and write the whole workflow as the skill
dense-slimming-tool.md.

TIME COST
---------
Session started: 2026-11-06 14:03 HKT (Asia/Hong_Kong)
This log written: 2026-11-06 19:08 HKT
Wall clock: about 5 h 05 min (includes the follow-up requests below).
Main machine time:
  vision encoder -> mmproj Q8_0 ........ ~1 min
  text model -> BF16 GGUF (54.7 GB) .... ~60 min (disk bound)
  Q4_K_M temp + imatrix (150 chunks) ... ~50 min
  quantize to IQ3_S mix ................ ~17 min per run, 3 runs
                                         (2 failed: invalid type name,
                                          missing imatrix for MTP layer blk.64)
  Hermes integration + tests ........... the rest

TOKEN COST (Hermes session, model anthropic/claude-sonnet-5.5 via Nous)
-----------------------------------------------------------------------
Figures read from the session database at the time of writing:
  input tokens ........ 228
  output tokens ....... 60,543
  cache read tokens ... 8,152,601
  cache write tokens .. 1,379,803
  messages: 223
  estimated cost: about USD 3.66
These are a snapshot; later turns of the session add a little more.

MY PROMPT
---------
Original request (verbatim):

  i have Qwen3.8-27B and Qwen3.8-27B-DFlash2-Q4_K_M.gguf on /volume/abc.
  extract the video encoder, convert to gguf and compress it to q8_0 .
  use llama.cpp
  "
  llama-quantize \
    --imatrix qwen3.8-27b-coding-imatrix.dat \
    --token-embedding-type q4_k_m \
    --output-tensor-type q6_k \
    --tensor-type "blk\.[0-3]\.ffn_.*=q5_k_s" \
    --tensor-type "blk\.6[0-3]\.ffn_.*=q5_k_s" \
    --tensor-type "attn_k=q6_k" \
    --tensor-type "attn_v=q6_k" \
    --tensor-type "attn_q=iq4_xxs" \
    --tensor-type "attn_output=iq4_xxs" \
    --tensor-type "ssm_.*=q5_k_s" \
    --tensor-type "ffn_down=iq4_xxs" \
    --tensor-type "ffn_gate=iq3_xxs" \
    --tensor-type "ffn_up=iq3_xxs" \
    Qwen3.8-27B-BF16.gguf \
    Qwen3.8-27B-Agentic-24GB.gguf IQ3_S
  "
  build a profile with these two models
  two script may help:"
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
  cmake --build build -j
  "
  "
  ./build/bin/llama-server \
    -hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M \
    -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \
    --spec-type draft-dflash \
    --spec-draft-n-max 7
  "
  build a whole workflow as dense-slimming-tool.md which is a skill

Follow-up prompts (verbatim):
  - progress
  - the main model of new profile should be provided by llama.cpp inside hermes desktop
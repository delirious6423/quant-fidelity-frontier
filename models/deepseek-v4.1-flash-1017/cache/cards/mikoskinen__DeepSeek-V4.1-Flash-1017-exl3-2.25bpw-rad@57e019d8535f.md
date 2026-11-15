---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- radiance
- exl3
- rocm
- rdna4
---

# DeepSeek V4.1 Flash 1017, EXL3 2.25 bpw for Radiance

This is a Radiance container (`.rad`) of
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) for four AMD
Radeon AI PRO R9700 cards, with the release's vision tower, so the server answers text and pictures.
It's the file behind the DeepSeek results in
[r9700-exact-serving](https://github.com/mikoskinen/r9700-exact-serving), and that repository's README
has the commands that start it with the same engine image and settings.

| | |
|---|---|
| Container | `deepseek-v4.1-flash-1017-exl3-2.25bpw.rad`, 273,238,496,160 bytes (254.5 GiB), sha256 `985ea2241317b511a218107bac5ef26ababcf529d2edb74cf1ac9faf4af2cf3d` |
| Files here | the container in six parts, `.part-00` to `.part-04` at 45 GiB each and `.part-05` at 29.5 GiB, because the Hub accepts files up to 50 GB |
| Routed experts | EXL3 from the `2.25bpw` branch of [turboderp/DeepSeek-V4.1-Flash-1017-exl3](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3) (commit 9183eaae), 2 bits in layers 6 to 34 and 2.5 bits in layers 0 to 5 and 35 to 39 |
| Every other text weight | converted from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with the recipe and imatrix in [Build it yourself](#build-it-yourself): the attention and shared experts in q6_K, q5_K and q8_0, the Engram table in iq4_nl, and the DSpark drafter with its MXFP4 experts as the release has them |
| Pictures | the release's vision tower and aligner in bf16 and the router biases for image rows in f32, 304 weights and 925.6 MiB |
| `dspark-dlw-q8_0.rad` | 602,046,464 bytes, sha256 `e71e034781d795bac6ab719343366ecd4b5702a727aff17640609a40313e1601`, a weight sidecar with 43 of the DSpark drafter's dense tensors in q8_0, made with `rad-sidecar` from the container |
| `dspark-conf.f32` | 21,504 bytes, sha256 `a97178fa6374686d95f833e38ad8aaf51f989cdbb1fe2312ce7b60aefc1e5b6d`, the release's `mtp.2.confidence_head.proj.weight` as raw float32, the drafter's confidence head |
| Engine | a Radiance fork with the radiance-exl3 plugin, as the image `ghcr.io/mikoskinen/r9700-exact-serving:deepseek-5f60dee1`, started with `--vision-tower-placement ram` |

## Download and join

The join needs about 322 GB of free disk:

```bash
hf download mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad --local-dir ~/models/ds41-exl3
cd ~/models/ds41-exl3
for p in deepseek-v4.1-flash-1017-exl3-2.25bpw.rad.part-0[0-5]; do
  cat "$p" >> deepseek-v4.1-flash-1017-exl3-2.25bpw.rad && rm "$p"
done
sha256sum -c SHA256SUMS
```

## Build it yourself

The conversion runs on the CPU inside the engine images.
[run/deepseek/build/](https://github.com/mikoskinen/r9700-exact-serving/tree/main/run/deepseek/build)
in the GitHub repository has the scripts that made this file, the recipe, the sidecar recipe, the list
of converted tensors and both chat templates, and the commands below run in that folder. You also need
the release in `$REL`, Unsloth's imatrix (`imatrix_unsloth.gguf_file` from
[unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF),
commit 3efffb93, sha256 `22ed7e8f07a471c8ec773a8062e5e9d72831cc179b9baf63ebf6792c8b3fd8f5`) and, in
`$BASE`, a 2.25 container that already has turboderp's routed experts, such as this file.

The Engram tables come from the release's tokenizer with `data/engram_tables.py` of the repository's
`deepseek/current` branch, which needs numpy. The body imatrix leaves out the routed experts' entries
and adds the entries of the plugin's fused weights:

```bash
git clone --depth 1 -b deepseek/current https://github.com/mikoskinen/r9700-exact-serving.git engine
python3 engine/data/engram_tables.py "$REL" engram-tables.safetensors
python3 imatrix_body.py imatrix_unsloth.gguf_file imatrix-body.gguf
```

The first conversion writes the text model from the release, except the routed experts of the 40 main
layers, which it copies from `$BASE` with `--reuse`. `changed-all.txt` lists the 3,613 release tensors
that it converts.
The run that made this file mounted an empty directory as the EXL3 source, because every expert came
from `--reuse`, and its result had the sha256 `e168865a044e44318e659e81342b0afa88c21873b169f49c1a4ade53eff5e11b`:

```bash
mkdir -p out empty
docker run --rm --network none --user "$(id -u):$(id -g)" -w /c \
  -v "$REL":/c/hf-src/DeepSeek-V4.1-Flash-1017:ro -v "$PWD/empty":/c/turboderp/DeepSeek-V4.1-Flash-1017-exl3:ro \
  -v "$PWD/engram-tables.safetensors":/c/engram-tables.safetensors:ro \
  -v "$PWD/deepseek41-exl3-full225.recipe":/c/recipe:ro -v "$PWD/chat_template.jinja":/c/chat_template.jinja:ro \
  -v "$PWD/imatrix-body.gguf":/c/imatrix-body.gguf:ro -v "$PWD/changed-all.txt":/c/changed.txt:ro \
  -v "$BASE":/base.rad:ro -v "$PWD/out":/out -e EXL3_SRC=turboderp/DeepSeek-V4.1-Flash-1017-exl3 \
  --entrypoint rad-convert ghcr.io/mikoskinen/r9700-exact-serving:deepseek-5e86593e \
  hf-src/DeepSeek-V4.1-Flash-1017 --add engram-tables.safetensors --recipe recipe \
  --tokenizer hf-src/DeepSeek-V4.1-Flash-1017/tokenizer.json -v --chat-template chat_template.jinja \
  --imatrix imatrix-body.gguf --reuse /base.rad --changed changed.txt \
  -o /out/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
```

The second conversion adds the vision tower in place. It keeps the 32,701 weights where they are,
appends the 304 tower weights and writes the chat template with the picture marker, which gives this
file:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" -w /c \
  -v "$REL":/c/hf-src/DeepSeek-V4.1-Flash-1017:ro -v "$PWD/empty":/c/turboderp/DeepSeek-V4.1-Flash-1017-exl3:ro \
  -v "$PWD/engram-tables.safetensors":/c/engram-tables.safetensors:ro \
  -v "$PWD/deepseek41-exl3-full225.recipe":/c/recipe:ro -v "$PWD/chat_template-vis.jinja":/c/chat_template.jinja:ro \
  -v "$PWD/out":/out -e EXL3_SRC=turboderp/DeepSeek-V4.1-Flash-1017-exl3 \
  --entrypoint rad-convert ghcr.io/mikoskinen/r9700-exact-serving:deepseek-5f60dee1 \
  hf-src/DeepSeek-V4.1-Flash-1017 --add engram-tables.safetensors --recipe recipe \
  --tokenizer hf-src/DeepSeek-V4.1-Flash-1017/tokenizer.json -v --chat-template chat_template.jinja \
  --reuse /out/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad --in-place
```

The drafter sidecar comes from the finished container:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/out":/m -v "$PWD/dlw-q8_0.recipe":/r.recipe:ro \
  --entrypoint rad-sidecar ghcr.io/mikoskinen/r9700-exact-serving:deepseek-5f60dee1 \
  /m/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad -o /m/dspark-dlw-q8_0.rad --recipe /r.recipe
```

A sidecar records the container's file name, so one written from a container with another name has
the same 43 weights as the sidecar here but a different sha256. The engine checks the container's
length and the entry names and shapes.

## License

MIT, the license of DeepSeek-V4.1-Flash-1017 and of turboderp's EXL3 repository. The full text is in
`LICENSE`.

---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- basalt
- gsq-rco
- iq3_s
- moe
- mtp
---

# Qwen3.8-Flash-Next GSQ-RCO IQ3_S for Basalt

[ISTA-DASLab's GSQ-RCO IQ3_S quantization](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), packed as one `.basalt` file for
[Basalt](https://github.com/jesdga95/basalt). Basalt is a local inference engine for NVIDIA Blackwell (sm_120) GPUs; it keeps the
experts that do not fit in VRAM in system RAM. Its own published packs carry NVIDIA's NVFP4 and Unsloth's GGUF quantizations; this one
carries ISTA-DASLab's GSQ-RCO.

## What is in the file

`Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt`, 87.6 GB:

- **The model:** all 1,224 tensors of ISTA-DASLab's `IQ3_S` GGUF (routed experts, n-gram table, dense weights), copied byte for byte.
  Nothing was re-quantized.
- **MTP drafter:** the BF16 `mtp.*` tensors of Qwen's checkpoint (revision `de4b8e4d`), with the tuned (self-distilled) dense part and
  the draft vocabulary of Basalt's own `Qwen3.8-FN-UD-IQ3_XXS` pack. The GGUF itself has no MTP head.
- **Vision projector:** ISTA-DASLab's `mmproj-Qwen3.8-Flash-Next-BF16.gguf`.

Built with the converter of Basalt v0.5.0. Its `--verify` found every GGUF tensor byte-identical and every section checksum correct. It
also prints 29 lines such as `mtp/dense holds fc_embedding.weight, which the source does not have`. Those are the drafter tensors,
which come from Qwen's checkpoint and not from the GGUF, so they are expected.

## Run it

You need an NVIDIA Blackwell GPU (RTX 50 series or RTX PRO), driver R580 or newer, the NVIDIA Container Toolkit, Linux x86-64 with AVX2,
and about 60 GB of free RAM: the experts the GPUs do not hold (50.4 GB) stay there, page-locked.

```sh
hf download willfalco/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Basalt --local-dir Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Basalt
cd Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Basalt
docker compose -f Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-basalt.yaml up -d
docker logs -f mod0
```

The first start builds Basalt v0.5.0 from its git tag, with FFmpeg 6 added for video input (10 to 20 minutes). The server then answers
on port 8080 with the OpenAI and Anthropic APIs, as model `mod0` (Basalt refuses other model names). It has no API key; add
`--api-key=KEY` to `command:` to require one.

What the compose file sets:

- `--gpu=0,1`: two 16 GB cards. The first runs every layer, the second holds and computes experts. For one card use `--gpu=0`.
- 256K context, int8 KV cache, 2 requests at once, each allowed the whole 256K.
- Image and video input.
- Default sampling for requests that send none: temperature 1.0, top_p 0.95, top_k 20.
- Basalt's options that change answers (route tail skip, route prior, relaxed drafting) stay off, as they are by default.

## Speed on our machine

2x RTX 5070 Ti 16 GB (PCIe 4.0 x8 each), Ryzen 9 3950X (AVX2, no AVX-512), 128 GB DDR4, the compose file above. Read from the server's
log during ordinary use, not a benchmark:

| | |
|---|---|
| One request, prose | 117 to 140 tok/s |
| One request whose answer repeats its context | up to 260 to 330 tok/s |
| Two requests at once | about 190 tok/s in total |
| Prompt reading | 2,090 tok/s at 8K tokens, 4,456 at 35K |
| Loading the weights | 52 GiB in 21 s |

## Rebuild it yourself

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-basalt-prep.sh` makes the same file from ISTA-DASLab's GGUF. It builds Basalt v0.5.0, fetches the MTP
tensors from Qwen's checkpoint, reads the tuned drafter's two sections from Basalt's UD-IQ3_XXS pack by HTTP range (SHA-256 checked),
converts and verifies. How to run it is at the top of the script.

## Credits

- **[Qwen](https://huggingface.co/Qwen)** for Qwen3.8-Flash-Next.
- **[ISTA-DASLab](https://huggingface.co/ISTA-DASLab)** for the GSQ-RCO quantization, and for the IQ3_S GGUF and the vision projector in
  this file.
- **[jesdga](https://github.com/jesdga95)** for [Basalt](https://github.com/jesdga95/basalt), its converter, and the tuned MTP drafter
  and draft vocabulary from the [official Basalt packs](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt).
- **[Niko1221](https://github.com/Niko1221)** for [Strata](https://github.com/Niko1221/Strata), the engine Basalt forks.
- **Ian Ranson and contributors** for [Infernix](https://github.com/Wallawalla47/Infernix), whose kernels Basalt adapts.
- **[ggml and llama.cpp](https://github.com/ggml-org/llama.cpp)** for the GGUF format and its quantization types.
- **[NInfer](https://github.com/Neroued/ninfer)**, whose ideas Basalt builds on.

## License

The weights are under the [Qwen Community License 1.0](LICENSE). It allows use, redistribution and derivative works as long as the
license notice comes along. Products with more than 100 million monthly active users or US$20 million monthly revenue must show the
model name, and companies running a Model-as-a-Service or AI work assistant business need a separate license from Qwen for commercial
use; read the full text before you rely on it.

ISTA-DASLab publishes its GGUF under Apache-2.0 ([copy](LICENSE-APACHE-2.0.txt)). This file repackages those tensors, unchanged, into
Basalt's file format. Basalt itself is MIT-licensed; the compose file builds it from source, and no Basalt code is included here.

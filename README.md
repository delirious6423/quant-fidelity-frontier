# QFF: Quant Fidelity Frontier

What publishers claim about their quants, pinned to the line they said it.

QFF lines up every Hugging Face quant of a model by its publisher's own claims (KLD, perplexity, top-1, benchmarks,
size) and shows which quants are efficient. Nothing is re-measured: each number is copied verbatim from the model card,
with a link to the line it came from.

| Model | Repos | Claimed values |
|---|---|---|
| [Qwen/Qwen3.8-Flash-Next](https://delirious6423.github.io/quant-fidelity-frontier/qwen3.8-flash-next/) | 453 | 2,980 |
| [Qwen/Qwen3.8-27B](https://delirious6423.github.io/quant-fidelity-frontier/qwen3.8-27b/) | 1,220 | 4,747 |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://delirious6423.github.io/quant-fidelity-frontier/deepseek-v4.1-flash/) | 133 | 1,121 |

**Site:** https://delirious6423.github.io/quant-fidelity-frontier/

Only compare numbers within one measurement set; the default overview mixes methods and is for a first glance.

## For agents
A read-only MCP server lets agents search claims, get a set's efficiency frontier, or find the best quant under a size budget:
```bash
claude mcp add qff -e QFF_DATA=https://delirious6423.github.io/quant-fidelity-frontier/ -- python3 qff_mcp/qff_mcp.py
```
More in [`qff_mcp/README.md`](qff_mcp/README.md).

## Support
QFF is free. If it saves you time, you can [tip on Ko-fi](https://ko-fi.com/delirious6423). Tips pay for upkeep and new models and never influence the data ([sponsor policy](docs/SPONSORS.md)).

## Discovery and dataset publishing
See [the rollout guide](docs/DISCOVERY.md) for repository metadata, search verification, static explainers and the reproducible Hugging Face export.

## More
- [Method: claim rules, gates, tests](docs/METHOD.md) · [How a model is ingested](docs/FASTPATH.md)
- [Developing: layout, build, hosting](docs/DEVELOPING.md)
- Found a wrong value or a missing quant? Open an issue with the repo and the card line.

Licence: code MIT ([LICENSE](LICENSE)); the compiled dataset CC BY 4.0. The claims belong to their publishers, and the cached model cards (`snapshots/`, `models/*/cache/cards/`) belong to their authors; they are kept only so every claim can be checked against its source.

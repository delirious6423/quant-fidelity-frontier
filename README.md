# QFF: Quant Fidelity Frontier

What publishers claim about their quants, pinned to the line they said it.

QFF lines up every Hugging Face quant of a model by its publisher's own claims (KLD, perplexity, top-1, benchmarks,
size) and shows which quants are efficient. Nothing is re-measured: each number is copied verbatim from the model card,
with a link to the line it came from.

| Model | Repos | Claimed values |
|---|---|---|
| Qwen/Qwen3.8-Flash-Next | 453 | 2,980 |
| Qwen/Qwen3.8-27B | 1,220 | 4,747 |

**Site:** https://delirious6423.github.io/quant-fidelity-frontier/

Only compare numbers within one measurement set; the default overview mixes methods and is for a first glance.

## For agents
A read-only MCP server lets agents search claims, get a set's efficiency frontier, or find the best quant under a size budget:
```bash
claude mcp add qff -e QFF_DATA=https://delirious6423.github.io/quant-fidelity-frontier/ -- python3 qff_mcp/qff_mcp.py
```
More in [`qff_mcp/README.md`](qff_mcp/README.md).

## More
- [Method: claim rules, gates, tests](docs/METHOD.md)
- [Developing: layout, build, hosting](docs/DEVELOPING.md)
- Found a wrong value or a missing quant? Open an issue with the repo and the card line.

Licence: code MIT ([LICENSE](LICENSE)); the compiled dataset CC BY 4.0. The claims belong to their publishers, and the cached model cards (`snapshots/`, `models/*/cache/cards/`) belong to their authors; they are kept only so every claim can be checked against its source.

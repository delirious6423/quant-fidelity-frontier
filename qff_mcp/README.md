# QFF MCP server

Read-only MCP tools over the QFF (Quant Fidelity Frontier) data, so agents can query publisher claims about
Hugging Face quants without scraping the site. Every number is a publisher's claim copied verbatim from a model card,
with `src` pointing at the card line. Nothing was re-measured, and numbers from different measurement sets are not
comparable.

## Tools
| Tool | What it answers |
|---|---|
| `list_models` | Which base models QFF covers, capture dates, counts |
| `search_quants(model, query, fmt, max_size_gb, has_metric, publisher)` | Claim rows matching filters |
| `get_repo(model, repo)` | Everything for one repo: method, methodology, flags, card pin, all rows |
| `list_sets(model)` | Measurement sets (one measurer, one method) and the not-comparable overview |
| `frontier(model, set_id, metric, max_x)` | Efficient quants in one set, and what beats each of the others |
| `best_under(model, max_size_gb, metric)` | Best quant under a size budget in each set, listed side by side, never ranked across sets |

## Run locally (stdio)
```bash
pip install mcp
QFF_DATA=https://delirious6423.github.io/quant-fidelity-frontier/ python3 qff_mcp/qff_mcp.py
```
Claude Code: `claude mcp add qff -e QFF_DATA=https://delirious6423.github.io/quant-fidelity-frontier/ -- python3 /path/to/qff_mcp/qff_mcp.py`

Claude Desktop (`claude_desktop_config.json`):
```json
{"mcpServers": {"qff": {"command": "python3", "args": ["/path/to/qff_mcp/qff_mcp.py"],
  "env": {"QFF_DATA": "https://delirious6423.github.io/quant-fidelity-frontier/"}}}}
```
`QFF_DATA` can also be a local folder built by `python3 assemble.py` (default: `../_site`).

## Host it (streamable HTTP)
`python3 qff_mcp/qff_mcp.py --http 8000` serves MCP at `/mcp`. `space/` holds a ready Hugging Face Space (Docker)
that does this; add the Space URL + `/mcp` as a custom connector.

## Tests
`python3 test_mcp.py` starts the server over real stdio and checks every tool against `data.json` independently:
rows are real and filters hold, strings are verbatim, the frontier equals an independent Pareto computation on every
comparable set, the not-comparable overview is refused for ranking, and budgets are respected.

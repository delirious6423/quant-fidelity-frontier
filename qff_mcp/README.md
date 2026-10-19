# QFF MCP server

Read-only MCP tools over the QFF (Quant Fidelity Frontier) data, so agents can query publisher claims about
Hugging Face quants without scraping the site. Every number is a publisher's claim copied verbatim from a model card,
with `src` pointing at the card line. Nothing was re-measured. Numerical ranking is permitted only when source evidence
verifies the same base model/reference, dataset and token selection, metric definition, context, settings, and compatible
implementation. Quantization publisher and measurement publisher are exposed separately; neither establishes compatibility.

## Tools
| Tool | What it answers |
|---|---|
| `list_models` | Which base models QFF covers, capture dates, counts |
| `search_quants(model, query, fmt, max_size_gb, has_metric, publisher, limit)` | Claim rows matching filters |
| `get_repo(model, repo)` | Everything for one repo: method, methodology, flags, card pin, all rows |
| `list_sets(model, comparable_only)` | Evidence-based classifications per metric; by default includes sets with at least one verified metric |
| `frontier(model, set_id, metric, max_x)` | Efficient quants in one set, and what beats each of the others |
| `best_under(model, max_size_gb, metric)` | Best quant under a size budget in each verified set; excluded sets include their reasons |
| `compare_measurements(model, left_id, right_id, metric, other_model)` | Pairwise compatibility, including cross-publisher and cross-model records |

## Comparison contract

All comparison results include `classification`, `label`, `explanation`, `evidence`,
`verified_matching_conditions`, `differences`, `unknowns`, and `ranking_permitted`:

- `same_protocol`: Same Protocol — Comparable. Numerical ranking is permitted for the requested metric.
- `different_or_unverified_protocol`: Different or Unverified Protocol — Not Directly Comparable.
- `different_base_model`: Different Base Model — Fidelity Not Directly Comparable. Side-by-side values do not rank
  model quality or fidelity across different reference distributions.

`list_sets` returns `comparisons` keyed by metric; its legacy `comparable` field requires every reported metric to pass.
The `comparable_only` filter includes sets with at least one eligible metric. Use the specific metric's result before
ranking. `frontier` and `best_under` recompute compatibility from each point's protocol evidence, ignoring stale flags.
Whole groups require all-pairs compatibility, not a connected chain of compatible pairs. An overview remains visible
with `comparable_only=false`; it can be ranked only if its actual metric population passes the same evidence checks.

`search_quants` exposes stable `measurement_id` values, protocol assertions, `measurement_owner`,
`quantization_owner`, and `claim_provenance`. Its aggregate `comparison` describes returned rows only.
Use two IDs with `compare_measurements` to obtain the precise pairwise assessment. Copied claims require verified
original attribution. Missing data, contradictory methodology, or a missing comparison engine fail closed.

Different sets or publishers may be comparable when their evidence verifies compatible protocols. Entries from
`best_under` must not be ordered against each other unless their returned aggregate comparison permits ranking.
If a requested metric prefix resolves to different named series (for example, `kld:english` and `kld:korean`),
the aggregate comparison fails closed while independently verified per-set winners remain available.
No method, reference checkpoint, or dataset is inferred merely from publisher identity.

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
`python3 tests/test_mcp.py` starts the server over real stdio and checks every tool against `data.json` independently:
rows are real and filters hold, strings are verbatim, the frontier equals an independent Pareto computation on every
comparable set, the not-comparable overview is refused for ranking, and budgets are respected.

The server imports `comparison.py` from the checkout root. For a Space, copy both `qff_mcp.py` and `comparison.py`
next to `space/Dockerfile` as described in `space/README.md`. Older published JSON without protocol assertions
remains readable, but its measurements are ineligible for rankings until rebuilt with evidence metadata.

Regression checks: `python3 -m unittest discover -s tests -p 'test_mcp_comparison.py' -v`.

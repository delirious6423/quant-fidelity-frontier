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
| `list_benchmarks(query, model, min_models, limit, offset)` | Reviewed benchmark names, exact units, model/source/report counts; `min_models=2` finds shared tasks |
| `search_benchmarks(benchmark, unit, model, query, publisher, max_size_gb, min_score, include_reference, limit, offset)` | Source-linked task scores across models, with size/score filters and bounded pagination |

All nine tools advertise read-only, non-destructive, idempotent annotations. The server exposes tools;
resource and prompt listings correctly return empty lists.

## Reviewed task benchmarks

These two tools read the same reviewed `benchmarks.sources` payloads as the model pages, including tasks
that are available for only one model. They do not scrape chart labels or infer equivalence from legacy `bench:` keys.

For example, call `list_benchmarks(query="GPQA")`, then
`search_benchmarks(benchmark="GPQA-Diamond", max_size_gb=80, min_score=30)`.
Benchmark names must exactly match discovery results. If a name has multiple units, `unit` is required;
fractions, percentages and unspecified published scales never blend or silently convert.

Every independent report remains a separate point, even when multiple sources describe the same build.
Results retain `raw`, `size_raw`, numeric `score` and `size_gb`, score/size source links and lines,
`source_repo`, `card_sha256`, `revision`, capture date, evaluation context and notes.
`publisher` selects the reporting card's owner; `repo` still identifies the evaluated build.
Reference builds are included unless `include_reference=false`.

`limit` is 1–200 and `offset` is nonnegative. `matches` counts all filtered reports before pagination;
pass `next_offset` to continue. Ordering is captured model/source/row order, never score order.
Point IDs and offsets refer to the current capture; restart pagination after a data refresh.
Withheld source tables remain visible in `withheld_sources` with their reasons.
Benchmark tools always return `ranking_permitted=false`: shared task names do not establish matched
prompting, sample selection, evaluation settings, or a cross-source leaderboard.

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
pip install "mcp==2.3.0"
QFF_DATA=https://delirious6423.github.io/quant-fidelity-frontier/ python3 qff_mcp/qff_mcp.py
```
Claude Code: `claude mcp add qff -e QFF_DATA=https://delirious6423.github.io/quant-fidelity-frontier/ -- python3 /path/to/qff_mcp/qff_mcp.py`

Claude Desktop (`claude_desktop_config.json`):
```json
{"mcpServers": {"qff": {"command": "python3", "args": ["/path/to/qff_mcp/qff_mcp.py"],
  "env": {"QFF_DATA": "https://delirious6423.github.io/quant-fidelity-frontier/"}}}}
```
`QFF_DATA` can also be a local folder built by `python3 web/assemble.py` (default: `../_site`).

The server checks for a new snapshot on the first request after five minutes and
invalidates cached model data even when model counts are unchanged. If refreshing
fails, that request returns an error and the next request retries; stale data is
never silently presented as fresh. Restarting also refreshes immediately.
Manifest paths and redirects must remain inside the configured data source.
JSON reads are limited to 1 MB for the manifest and 32 MB per model.

## Host it (streamable HTTP)
`python3 qff_mcp/qff_mcp.py --http 8000` serves MCP at `/mcp`. `space/` holds a ready Hugging Face Space (Docker)
that does this; add the Space URL + `/mcp` as a custom connector.

Hugging Face supplies `SPACE_HOST`, which is added to the exact Host and HTTPS Origin allowlists.
For another host, set `QFF_PUBLIC_URL=https://mcp.example.org` before starting. This must be an exact
HTTPS origin, with no credentials, path, query or wildcard. Invalid configuration stops startup.
Localhost remains available on the chosen port; other hosts and origins are rejected. DNS-rebinding
protection stays enabled. Data-only site updates refresh automatically; new tools or server fixes require
updating the separately hosted MCP code. No public endpoint is assumed by this scaffold.

The Docker recipe and CI pin the current stable SDK (`2.3.0`, released 2026-10-02).
The server supports its 2026-07-28 discovery protocol and legacy clients, and retains compatibility
with the maintenance SDK `1.30.0`. Both generations' HTTP startup and client result types are covered
by transport tests. HTTP tool calls use ordinary JSON responses: complete protocol evidence can exceed
the latest SDK client's default 1 MiB SSE-event limit, and these read-only calls require no event stream.
The limit remains enabled on clients. Dependency upgrades should run the full MCP suites before changing
the pin. See the [official release notes](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.3.0)
and [migration guide](https://py.sdk.modelcontextprotocol.io/migration/).

## Tests
`python3 tests/test_mcp.py` starts the server over real stdio and checks search, provenance and rankings against every model's `data.json` independently:
rows are real and filters hold, strings are verbatim, the frontier equals an independent Pareto computation on every
comparable set, the not-comparable overview is refused for ranking, and budgets are respected.

The server imports `comparison.py` from the checkout root. For a Space, copy both `qff_mcp.py` and `comparison.py`
next to `space/Dockerfile` as described in `space/README.md`. Older published JSON without protocol assertions
remains readable, but its measurements are ineligible for rankings until rebuilt with evidence metadata.

Regression checks: `python3 -m unittest discover -s tests -p 'test_mcp_comparison.py' -v`.
Loading and freshness checks: `python3 tests/test_mcp_loading.py`.
Reviewed benchmarks: `python3 tests/test_mcp_benchmarks.py` verifies every published report against the
assembled model files over actual stdio, plus unit separation, repeated reports, filters and pagination.
HTTP and tool discovery: `python3 tests/test_mcp_http.py` performs real initialization, discovery and all
nine tool calls over both transports, including a Space hostname, an explicit public origin and hostile headers.
It also checks modern SDK discovery and the largest published model's full set and search responses using
the client's default transport limits.

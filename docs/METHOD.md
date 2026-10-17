# Method: claims, gates and tests

## Rules for a claim
1. It must be in the publisher's own card (or a cited docs page) at the captured version.
2. The string is kept exactly as printed (`0.0771 ± 0.0007`, `86.95 ± 0.14 %`), never rounded or retyped.
3. It cites its line, and records the reference model, eval data, context and tool as stated, or `not stated`.
4. Numbers a publisher reports for someone else's file are `cross` rows; copies of other cards are labelled as copies.
5. Flags describe what the card says, with a line reference. No guesses about motives.

## Fail-closed gates
`build.py` refuses to build if any gate fails:
- `verify.py`: each value appears verbatim on its cited line, with the right sign and a unit next to every size.
- `verify_columns.py`: table values sit under a header of the right metric, in the cited row.
- `verify_images.py`: values read from chart images match marker positions (only for models with image claims).
- Plausibility and sign assertions inside `build.py`.

## Tests
`test_gates.py` (gate mutations), `test_claims.py` (review regressions), `test_ui.py` (Playwright, desktop and phone),
`test_mcp.py` (MCP answers checked against the data). CI runs all of them (`.github/workflows/verify.yml`).

## How to read the site
- **Overview (default):** every claim on one chart, for a first glance; labelled *not comparable*.
- **Measurement sets:** one measurer, reference, dataset and method. Only here can quants be ranked (frontier, "Beaten by").
- **Ledger:** every row with its source link. **Publisher notes:** method, flags, copies, cards with no claims.

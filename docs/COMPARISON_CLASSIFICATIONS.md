# Evidence-based comparison classifications

QFF compares published claims; it does not independently rerun the measurements. Publisher names do not establish or prohibit compatibility.

| Classification | Meaning | Direct numerical ranking |
|---|---|---|
| Same Protocol — Comparable | Evidence establishes the same or directly equivalent base model, reference checkpoint/distribution, dataset, token selection, metric definition/direction/units/aggregation, context, implementation and settings | Permitted for that metric and population |
| Different or Unverified Protocol — Not Directly Comparable | At least one condition differs, is contradictory, or lacks sufficient evidence | Unavailable; values remain visible |
| Different Base Model — Fidelity Not Directly Comparable | Confirmed different base models or reference distributions | Unavailable; KLD ordering does not rank quality or cross-model fidelity |

A different experiment identifier is **not** proof of a different reference distribution. Unknown details, empty payloads and placeholder strings cannot establish equality. Contradictory evidence fails closed rather than selecting a convenient assertion.

## Audit findings

The original `build_sets` marked all curated sets comparable. `auto_sets` grouped by measurement owner, identical methodology strings, panel and size basis, then marked surviving groups comparable. Even an unnamed reference or missing dataset could therefore enable rankings. Chart lines, Pareto frontiers, the “Beaten by” column, default metric sorting and MCP recommendations trusted that flag.

Automatic sets already distinguished measured competitor files from quoted/copied claims. Those exclusions, original set membership/IDs/order, canonical metric order, published values and source citations are preserved. Numeric coincidence detection remains an existing conservative copy flag; identical rounded values do not establish protocol equivalence or prove an independent replication.

The audit also covered homepage examples, the mixed ledger, the GLM benchmark panel, static publisher comparison bars, MCP packaging and existing mobile interactions. Unverified metric ranking is now blocked in all of these paths; source-linked values remain available. Name/size sorting and ordinary chart filters remain available.

## Rules and evidence schema

`comparison.py` is the shared pure-Python classifier. `classify_pair`, `classify_group`, `classify_by_metric` and `summarize_comparisons` return classification, explanation, evidence, verified matching conditions, confirmed differences, unknowns, limitations and `ranking_permitted`.

Each `protocol` condition is an assertion with `value` and a nonempty `evidence` list containing source URLs. Multiple assertions preserve contradictions. `protocol.metrics[metric]` supplies explicitly scoped metric definitions and corpus/settings overrides. A metric name alias alone never establishes its definition. Optimization direction must be known before ranking; uncertainty about absolute KL orientation within a documented common implementation remains visible separately.

`identity_scope` limits opaque artifact/implementation identities to a documented shared experiment. For different experiments, equality is unverified unless exact global identities or cited direct equivalence establish it. An equivalence records `condition`, two `values`, their aligned `identity_scopes` when scoped, and evidence. It applies to that exact endpoint pair; it is never transitively chained.

A group requires **every pair** to pass. Exact protocol/provenance memoization preserves the full pair counts, failures and evidence; it does not merge graph components. Source-bound cross-measurement-publisher views use complete-link groups with compatible metric and size basis. Two groups may overlap when compatibility is non-transitive.

The schema also supports future versioned QFF protocols: use exact versioned checkpoint, token-selection, implementation and settings identities, cite the standard and evidence that the measurement followed it, and bind that evidence to its source. A protocol name, same software family, or claim to use “BF16” alone cannot fill missing conditions. No special publisher allowlist is used.

## Source binding and current coverage

`protocol_metadata.py` loads reviewed assertions from `protocol_evidence/<model>.json`. Each binding specifies the measurement card and its exact SHA-256. Every evidence quote must equal the cited line range of the cached source and match its hash. Changed sources, broken quotes and unreviewed row-level methodology overrides withdraw compatibility without dropping claims. Conflicting bindings are retained as contradictions.

The first reviewed real protocols are Qwen3.8-27B AtomicChat and Archsloth:

- **AtomicChat KLD and top-1:** its card expressly says it downloaded competitors’ files and measured them itself against the same original weights and held-out text (L190–193). L329–351 identifies the BF16 GGUF reference, shared logits, `eval_neutral`, 87 chunks, 4096 context and reproduction command. All 27 existing chart points remain eligible, including the lmstudio-community and ggml-org files measured by AtomicChat. The exact target repositories for these two competitor files remain unverified; the source names their publishers. The existing unsloth row lacks an extracted size and remains outside the size plot, as before.
- **Archsloth English KLD:** its source explicitly documents a common teacher, text and chunks for candidates, with 512 context and evaluation commands. Only the mapped English KL metric is registered. Korean KL, top-p and perplexity are not silently equated with it.
- **AtomicChat versus Archsloth:** B. Different context and unverified cross-experiment reference/data/implementation equivalence prohibit direct ranking.

These are claims of relative compatibility **within the declared experiment**, not certification of reproducibility. Public checkpoint/dataset/token hashes, exact evaluator revisions and some absolute KL definitions are not documented. Scoped assertions preserve those limitations and cannot be equated to independent measurements merely sharing familiar labels. All other existing protocols remain B until evidence is reviewed; this does not say their methods are necessarily different or their measurements invalid.

Publisher URLs using `main` remain mutable. Evidence also links to the captured source at a repository commit and records the cached SHA-256. Neither a content hash nor a mutable HF URL is described as an immutable upstream checkpoint revision.

## Ownership and copied claims

`quantization_owner` identifies the file publisher. `measurement_owner` identifies the evaluator when established. `claim_provenance.claim_publisher` identifies the card carrying the claim. Legacy `measured_by` remains for compatibility; for copied/quoted claims the new evaluator field is unknown unless the original evaluator is verified. Copying a claim never creates an independent measurement.

A copied or quoted claim can only pass the reusable classifier after verified original measurement attribution with supporting evidence; existing automatic chart sets continue excluding copied/quoted rows. Missing provenance never authorizes a publisher to inherit another experiment’s protocol.

## Website and MCP behavior

The initial measurement set, saved/deep-link precedence, metric order and log/linear defaults are unchanged. A restrained status appears above the plot; methodology, differences, unknowns, reproduction limits and source links expand on demand. The overview retains all existing points and clearly states that its protocols are mixed. Verified-set links locate the exact measurement by `measurement_id`, including when quant and evaluator differ.

Classification and ranking eligibility are per metric. A set may support KLD comparison while its PPL claims remain unverified. The legacy aggregate `comparable` requires every reported metric to pass. No frontend ranking falls back to a legacy boolean.

MCP recomputes eligibility from protocol evidence. `list_sets`, `search_quants`, `get_repo`, `frontier` and `best_under` expose comparison metadata. `compare_measurements` supports explicit pairs, including across base models. Prefix metric selection in `best_under` preserves the actual corpus overlay; incompatible or differently resolved series cannot authorize an aggregate ranking. Older datasets without evidence remain readable but cannot authorize rankings.

GitHub Pages continues serving static HTML/CSS/JS/JSON. Python runs at build time; no database or mandatory server was added. The MCP Space packaging includes the same classifier, but deployment is separate and requires approval.

## Validation and previews

See [validation results](COMPARISON_VALIDATION.md) and [desktop/mobile before-and-after previews](previews/comparisons/README.md).

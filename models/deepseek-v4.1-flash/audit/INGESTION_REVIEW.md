# DeepSeek V4.1 Flash ingestion review

136 repositories catalogued: 130 tagged plus six verified supplements. 133 pinned cards captured; three HTTP 401 failures excluded and recorded in fetch_failures.json. Source hashes and commit identifiers are retained.

61 distinct cards have independent A/B maps. 42 resolver maps adjudicate differences against exact pinned source text. Resolver manifests pin source, mapping and candidate bytes. 72 cards remain explicitly unreviewed; this is not a claim of exhaustive reading of all captured cards. No model inference or new quality measurements were performed.

Final generated dataset: 1,166 source-linked values, 130 own variants, 46 repositories with quality claims, 297 displayed rows. No unresolved disagreements or quarantined entries remain. Existing Qwen dataset totals are unchanged.

Native mixed-precision checkpoints are distinguished from BF16 references. Engram-only, Engram-excluded, decoder-only and routed-expert sizes remain scoped. Named corpora, context lengths, full-vocabulary and top-k methods, quoted figures and modified-weight models are kept distinct. Scientific notation and top-1 higher-is-better ranking are supported.

The final integrated source and column gates pass, including the wave-4 measured-control and wave-5 size-label issues documented in resolver notes. Resolver-local candidate files are applied before shared candidates, avoiding candidate-ID collisions. All review agents used the requested same model; independent passes are not a different-model cold review.

Validation: all model builds, source and column gates, claim regressions, panel/mapping/combined-column regressions, static assembly and MCP checks. Browser validation is deferred to PR CI: local browser downloads returned an unavailable-site response. This branch has not been deployed.

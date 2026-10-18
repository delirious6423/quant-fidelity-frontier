# DeepSeek V4.1 Flash ingestion

- Capture all tagged quant cards and verified supplemental repositories at pinned commits.
- Retain SHA-256 hashes and source metadata; exclude failed fetches from claim extraction.
- Map core claims in two independent runs, keeping test panels and storage bases separate.
- Quarantine disagreements and gate failures; label unread cards explicitly.
- Review model-specific reference/size handling; build all models and run repository verification.
- Submit a reviewable branch with documented counts and limitations.

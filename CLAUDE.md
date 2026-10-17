# QFF working notes (for agents)

- Claims only: never type a number into `verified/`. Values are copied by code (`qcl_map.py apply`, `qcl_watch.py merge`) from card text. A gate failure means the entry is quarantined, never "fixed" by hand-editing a number.
- Never rebuild a card from memory or a truncated tool result. Cards are saved byte-for-byte by `qcl_mcpsave.py` and pinned by SHA-256.
- Data roots: repo root = Qwen3.8-Flash-Next; `models/<slug>/` = every other model. Always pass `QCL_ROOT=models/<slug>` for those.
- Before committing: `build.py` for every data root, then `test_gates.py`, `test_claims.py`, `test_ui.py` (both sites via `QCL_SITE`), `test_mcp.py`.
- The repo is public: commit no secrets, tokens, personal data, local paths or scratch files.
- Pages deploys (`deploy-pages` workflow) are approved by the owner. Publishing the Hugging Face MCP Space still needs the owner's explicit go-ahead in chat.
- Agent fan-out: max 20 concurrent subagents; mapping workers claim queues by atomic `mv pending/X claimed/X` and write only their claimed names.
- Status and next steps: `PROGRESS.md`.

# Existing regression verification

Validation used the eight live models at starting commit
`670acdfb0c84e4ebf609d656ac06912dfaba8f2a`, with the new runner files present.
All source inputs were repository snapshots. No production state machine was run,
no remote workflow was dispatched, and no published claim was edited.

| Check | Result |
|---|---|
| `build.py`, every live model | 8/8 passed |
| `assemble.py /tmp/qff-runner-site` | Passed; 30 files, 29.2 MB |
| Existing gate counterfactuals | 15 passed |
| Existing claims checks | 69 passed |
| Fetch security | 13 tests passed |
| Source pins | 12 tests passed |
| Mapping provenance | 14 tests passed |
| QA/release receipts | 52 tests passed |
| Mapping panels | 22 tests passed |
| Column captions | 8 tests passed |
| Current-source and image flags | Passed |
| Gate integrity | 20 tests passed |
| README | 4 tests passed |
| Assembly safety | 19 tests passed |
| Workflow security | 9 tests passed after registering every new runner suite; triggers unchanged |
| Discovery/export | 14 canonical pages; 3,655 claims preserved exactly across 8 models |
| Model explorer UI | All 8 model suites passed; 786 checks, desktop and phone |
| MCP stdio, installed 1.29.0 | 51 checks passed |
| MCP stdio, pinned 2.3.0 | 51 checks passed |
| Tracked model data before/after | Unchanged |

Machine-readable command/results record: [existing-validation.json](existing-validation.json).
New runner/fault-injection and shadow-replay results are recorded separately.
These are measured local test outcomes, not production performance or model quality
measurements. No tokens or costs were measured.

## Reproduce

From the repository root, with existing project dependencies and Chromium installed:

```bash
python3 pipeline/qcl_live.py > /tmp/qff-live-models.txt
while IFS= read -r model; do
  QCL_ROOT="$model" python3 build.py || exit 1
done < /tmp/qff-live-models.txt
python3 assemble.py /tmp/qff-runner-site
python3 tests/test_gates.py
python3 tests/test_claims.py
python3 tests/test_fetch_security.py
python3 tests/test_source_pins.py
python3 tests/test_mapping_provenance.py
python3 tests/test_qa_release.py
python3 tests/test_mapping_panels.py
python3 tests/test_column_caption.py
python3 tests/test_current.py
python3 tests/test_image_flag.py
python3 tests/test_gate_integrity.py
python3 tests/test_readme.py
python3 -m unittest discover -s tests -p 'test_assembly_safety.py'
python3 -m unittest discover -s tests -p 'test_workflow_security.py'
QFF_DATA=/tmp/qff-runner-site python3 tests/test_mcp.py
while IFS= read -r model; do
  QCL_SITE="$model/site" python3 tests/test_ui.py || exit 1
done < /tmp/qff-live-models.txt
git diff --exit-code -- models
```

Discovery currently hardcodes `_site`. In this validation `_site` did not exist,
so a temporary symlink to `/tmp/qff-runner-site` was created for
`python3 tests/test_discovery.py` and removed afterward. Do not overwrite an
existing `_site` directory. Alternatively assemble the ignored local `_site`
directory using the existing command before running discovery.

The environment already supplied Pillow, NumPy, Playwright and MCP 1.29.0.
Chromium was absent and installed into `/tmp/qff-browsers`; browser checks used
`PLAYWRIGHT_BROWSERS_PATH=/tmp/qff-browsers`. The syscall sandbox denied loopback
sockets and Chromium startup, and stalled MCP stdio initialization. The unchanged
fetch-security, browser and MCP suites then passed outside that sandbox. Browser
tests abort external requests and inspect only locally built pages. Raw local logs
are under `/tmp/qff-runner-validation`; these temporary diagnostics are not a
persistence or deployment mechanism.

The pinned MCP version is installed independently with:

```bash
python3 -m venv /tmp/qff-mcp-v2
/tmp/qff-mcp-v2/bin/pip install 'mcp==2.3.0'
QFF_DATA=/tmp/qff-runner-site /tmp/qff-mcp-v2/bin/python tests/test_mcp.py
```

## Focused chart correction found by new-model validation

A synthetic new model exposed an existing `build.py:kld_chart` division by zero:
a maximum size at or below 20 GB made both x-axis bounds equal; a single exact
power-of-ten KLD value likewise made both logarithmic y-axis bounds equal. Two
small guards create finite axes only for those degenerate bounds. The focused
regression is in `tests/test_runner.py`; no claim or source fixture was edited to
make this pass.

After the correction, every live model was rebuilt again. All eight builds passed
and all **40 model-site files were byte-identical** to the prior builds. A fresh
assembly into `/tmp/qff-runner-site-after-axis-guard` was compared to the previous
`/tmp/qff-runner-site`: all **30 assembled files were byte-identical**. Therefore
no browser rerun was necessary for these unchanged pages. `git diff --exit-code
-- models` remained clean. This correction changes only previously failing
degenerate chart inputs; existing website output, defaults, claims and provenance
are unchanged.

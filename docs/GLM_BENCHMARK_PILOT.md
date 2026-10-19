# GLM benchmark pilot

Scope: GLM-5.3-Flash only, behind `benchmark_pilot` in its model configuration.
Existing fidelity charts remain available. No deployment is part of this change.

The pilot reads existing gated claims from mbehr90's table, never authored values.
GSM8K chat has the NVFP4 build and source FP8 reference; GSM8K lm-eval, ARC-C,
and AI2D vision each have one result. Missing results are not zeros.
Disk sizes are converted by the existing builder from GiB to decimal GB.
Score units are not explicit in the card, so the axis says “Published score”.
No frontier, cross-source ranking, or significance claim is displayed.

The old other-model filter rejected every row named GLM, including this model's
own FP8 reference. The pilot restores only the exact base-model target from
this source card; all other filtering stays unchanged.

Controls: benchmark, zero-baseline/zoomed linear axis, file-size shading, result
selection by chart or accessible picker. Source table remains without JavaScript.
No runtime server or API is required; this is compatible with GitHub Pages.

Checks: `tests/test_benchmarks.py` verifies real captured source values and unsafe
input rejection; `tests/test_benchmarks_ui.py` checks 320/390/1366px layouts and
interactions and saves screenshots under `test-results/`. Existing multi-model
gates and UI tests remain required before merging. Review and deployment remain
separate steps.

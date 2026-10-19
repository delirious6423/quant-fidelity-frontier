"""Exercise dispatch validation and check the verification-to-deployment boundary."""
import os
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"


def scripts(text):
    """Extract the simple literal run scalars used in these workflows, without YAML dependencies."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)(?:- )?run: (.*)$", line)
        if not match:
            continue
        if match[2] not in ("|", "|-", ">", ">-"):
            yield match[2]
            continue
        block = []
        for following in lines[index + 1:]:
            if following.strip() and len(following) - len(following.lstrip()) <= len(match[1]):
                break
            block.append(following)
        yield textwrap.dedent("\n".join(block))


class WorkflowSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fetch = (WORKFLOWS / "fetch.yml").read_text()
        cls.verify = (WORKFLOWS / "verify.yml").read_text()
        cls.pages = (WORKFLOWS / "pages.yml").read_text()
        cls.validation = next(script for script in scripts(cls.fetch) if "MODEL_SLUG" in script and "fullmatch" in script)

    def validate(self, model, slug, cwd=None):
        with tempfile.TemporaryDirectory() as empty:
            return subprocess.run(["bash", "-e", "-c", self.validation], cwd=cwd or empty,
                env={**os.environ, "MODEL_ID": model, "MODEL_SLUG": slug, "PYTHONPATH": str(ROOT)},
                capture_output=True, text=True)

    def test_valid_real_model_names_are_accepted(self):
        for model, slug in (("Qwen/Qwen3.8-27B", "qwen3.8-27b"),
                            ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-v4.1-flash"),
                            ("zai-org/GLM-5.3-Flash", "glm-5.3-flash")):
            with self.subTest(model=model):
                result = self.validate(model, slug)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_shell_and_path_inputs_are_rejected(self):
        bad = ('../outside', '/absolute', 'x/../../outside', '$(printf injected)',
               'x"; printf injected; #', 'x\ninjected', '-option', '.', '..', '', 'bad slug')
        for value in bad:
            with self.subTest(value=value):
                self.assertNotEqual(self.validate("test/model", value).returncode, 0)
                self.assertNotEqual(self.validate(value, "valid").returncode, 0)

    def test_dispatch_expressions_never_become_shell_source(self):
        for script in scripts(self.fetch):
            self.assertNotIn("${{", script)
        self.assertIn("MODEL_ID: ${{ inputs.model }}", self.fetch)
        self.assertIn("MODEL_SLUG: ${{ inputs.slug }}", self.fetch)
        self.assertLess(self.fetch.index("Validate model identity"), self.fetch.index("HF_TOKEN:"))

    def test_all_shell_blocks_parse(self):
        for workflow in WORKFLOWS.glob("*.yml"):
            for script in scripts(workflow.read_text()):
                with self.subTest(workflow=workflow.name, script=script[:60]):
                    result = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_deployment_uses_the_verified_artifact_without_rebuilding(self):
        self.assertRegex(self.pages, r"(?m)^  verify:\n")
        self.assertIn("uses: ./.github/workflows/verify.yml", self.pages)
        self.assertIn("publish_pages_artifact: true", self.pages)
        self.assertIn("needs: verify", self.pages)
        self.assertNotIn("actions/checkout", self.pages)
        self.assertNotIn("build.py", self.pages)
        self.assertIn("workflow_call:", self.verify)
        self.assertIn("if: ${{ inputs.publish_pages_artifact == true }}", self.verify)
        self.assertGreater(self.verify.index("actions/upload-pages-artifact"), self.verify.index("test_mcp_comparison.py"))
        self.assertNotRegex(self.verify, r"(?m)^\s+ref:", "checkout must use the caller's SHA, not a mutable branch")

    def test_live_model_enumeration_failure_is_not_hidden_in_a_shell_loop(self):
        self.assertIn("python3 pipeline/qcl_live.py > live-models.txt", self.verify)
        self.assertNotIn("for d in $(python3 pipeline/qcl_live.py)", self.verify)

    def test_failed_fetch_keeps_completed_ledger_checkpoints(self):
        self.assertIn("id: fetch", self.fetch)
        self.assertIn("!cancelled() && (steps.fetch.outcome == 'success' || steps.fetch.outcome == 'failure')", self.fetch)
        self.assertIn('git add -- "models/$MODEL_SLUG/cache"', self.fetch)

    def test_dispatch_rejects_existing_slug_model_and_ledger_identity_mismatches(self):
        fixtures = [
            ("model.json", {"model": "old/model", "slug": "known"}),
            ("model.json", {"model": "new/model", "slug": "other"}),
            ("cache/index.json", {"model": "old/model", "repos": {}}),
            ("cache/index.json", {"repos": {}}),
        ]
        for filename, data in fixtures:
            with self.subTest(filename=filename, data=data), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "models/known" / filename
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(data))
                before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
                result = self.validate("new/model", "known", cwd=directory)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual({str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}, before)

    def test_every_regression_suite_is_part_of_release_verification(self):
        missing = sorted(path.name for path in (ROOT / 'tests').glob('test_*.py')
                         if path.name not in self.verify)
        self.assertEqual(missing, [], 'new regressions must run before publishing the verified artifact')


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""pipeline/ci_local.py runs the verify.yml battery locally from the workflow file itself: every `run:` step of every
job is listed in the workflow's order, hosted-runner plumbing is skipped, package installs are skipped unless asked,
and a `pip install X && CMD` line keeps CMD. The listing is checked against the workflow so the runner cannot drift."""
import os, re, subprocess, sys, unittest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import ci_local

WF = yaml.safe_load(open(os.path.join(ROOT, ".github", "workflows", "verify.yml"), encoding="utf-8"))


class CiLocalTests(unittest.TestCase):
    def test_every_run_step_of_every_job_is_listed_once_in_order(self):
        for job_name, job in WF["jobs"].items():
            steps = ci_local.run_steps(job_name, job)
            self.assertEqual(len(steps), len(job["steps"]), job_name)
            for wf_step, step in zip(job["steps"], steps):
                if "uses" in wf_step:
                    self.assertIn("skip", step); self.assertTrue(step["skip"].startswith("uses "))
                elif "script" in step:
                    for line in step["script"].splitlines():
                        self.assertIn(line, wf_step["run"], (job_name, step["name"]))

    def test_every_test_suite_in_the_workflow_is_executed_locally(self):
        wanted = set(re.findall(r"tests/(test_[a-z0-9_]+\.py)", open(os.path.join(ROOT, ".github", "workflows", "verify.yml")).read()))
        wanted |= set(re.findall(r"-p '(test_[a-z0-9_]+\.py)'", open(os.path.join(ROOT, ".github", "workflows", "verify.yml")).read()))
        ran = set()
        for job_name, job in WF["jobs"].items():
            for step in ci_local.run_steps(job_name, job):
                if "script" in step:
                    ran |= set(re.findall(r"tests/(test_[a-z0-9_]+\.py)", step["script"])) | set(re.findall(r"-p '(test_[a-z0-9_]+\.py)'", step["script"]))
        self.assertEqual(sorted(wanted - ran), [], "suites the workflow runs that the local runner would not")

    def test_installs_are_skipped_unless_asked_and_commands_after_them_survive(self):
        steps = ci_local.run_steps("site", WF["jobs"]["site"])
        mcp = next(s for s in steps if s["name"].startswith("MCP server answers"))
        self.assertNotIn("pip install", mcp["script"]); self.assertIn("python3 tests/test_mcp.py", mcp["script"])
        skipped = [s for s in steps if "skip" in s and "package install" in s["skip"]]
        self.assertTrue(skipped, "pip install steps are skipped by default")
        with_pip = ci_local.run_steps("site", WF["jobs"]["site"], allow_pip=True)
        self.assertTrue(any("pip install" in s.get("script", "") for s in with_pip))
        restore = next(s for s in steps if s["name"].startswith("Restore every live model"))
        self.assertIn("artifact restore", restore["skip"])

    def test_model_job_steps_take_the_model_dir_from_the_environment_only(self):
        for step in ci_local.run_steps("model", WF["jobs"]["model"]):
            if "script" in step:
                self.assertNotIn("${{", step["script"]); 
        self.assertIn("MODEL_DIR", WF["jobs"]["model"]["env"].keys().__iter__().__next__())

    def test_list_runs_and_names_each_live_model(self):
        p = subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "ci_local.py"), "--list"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-800:])
        live = subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "qcl_live.py")], cwd=ROOT, capture_output=True, text=True).stdout.split()
        for d in live:
            self.assertIn(f"model {os.path.basename(d)}", p.stdout)
        self.assertIn("RUN  plan", p.stdout); self.assertIn("RUN  site", p.stdout)
        p = subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "ci_local.py"), "--list", "--models", "no-such-model"], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()

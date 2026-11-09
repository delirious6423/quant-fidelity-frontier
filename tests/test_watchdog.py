"""The watchdog reports withdrawn values nothing restored and stalled scheduled tasks; its issue speaks only on news."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import qcl_watchdog as W

NOW = dt.datetime(2026, 10, 21, 12, 0, tzinfo=dt.timezone.utc)
OLD, NEW = "a" * 64, "b" * 64


def variant(name, *metrics, src=None, size=None):
    return {"name": name, "m": {k: "0.1" for k in metrics}, "L": 3, **({"src": src} if src else {}),
            **({"size": size, "unit": "GB"} if size is not None else {})}


def entry(repo, pin, variants, **extra):
    return {"repo": repo, "sha": "main", "source_sha256": pin, "fmt": "GGUF", "runtime": None, "weights_modified": False,
            "method": None, "methodology": None, "variants": variants, **extra}


def write_part(root, name, entries):
    (root / "verified" / name).write_text("R = []\n" + "".join(f"R.append({e!r})\n" for e in entries))


def git(cwd, *args, date=None):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", **({"GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date} if date else {})}
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="qff-watchdog-"); self.addCleanup(self.tmp.cleanup)
        self.code = Path(self.tmp.name)
        self.root = self.code / "models" / "m"
        (self.root / "verified").mkdir(parents=True); (self.root / "snapshots").mkdir()
        index = {"repos": {
            "hand/changed": {"sha": "main", "sha256": NEW, "file": "x.md", "fetched_at": "2026-10-21T01:00:00Z"},
            "hand/recent": {"sha": "main", "sha256": NEW, "file": "x.md", "lastModified": "2026-10-21T10:00:00.000Z"},
            "hand/restored": {"sha": "main", "sha256": NEW, "file": "x.md", "fetched_at": "2026-10-20T01:00:00Z"},
            "hand/sizes": {"sha": "main", "sha256": NEW, "file": "x.md", "fetched_at": "2026-10-20T01:00:00Z"},
            "img/charts": {"sha": "main", "sha256": NEW, "file": "x.md", "fetched_at": "2026-10-20T01:00:00Z"},
            "hand/current": {"sha": "main", "sha256": OLD, "file": "x.md", "fetched_at": "2026-10-20T01:00:00Z"}}}
        (self.root / "snapshots/index.json").write_text(json.dumps(index))
        write_part(self.root, "part1.py", [
            entry("hand/changed", OLD, [variant("Q4", "mean_kld", "ppl"), variant("Q8", "mean_kld")],
                  baseline={"ppl": "4.5", "L": 2}),                                   # 4 reviewed values
            entry("hand/recent", OLD, [variant("Q4", "mean_kld", "ppl")]),
            entry("hand/restored", OLD, [variant("Q4", "mean_kld")]),
            entry("hand/sizes", OLD, [variant("Q4", size=10.5), variant("Q8", size=20.1)]),
            entry("img/charts", OLD, [variant("2 bpw", "mean_kld", "ppl", src="chart_image")]),
            entry("hand/current", OLD, [variant("Q4", "mean_kld", "ppl")])])
        write_part(self.root, "part8_machine.py", [
            entry("hand/changed", NEW, [variant("Q4", "mean_kld")]),                 # a re-mapping found 1 of 4
            entry("hand/restored", NEW, [variant("Q4", "mean_kld", "ppl")])])        # a re-mapping found more
        self.logs = self.code / "logs" / "scan"; self.logs.mkdir(parents=True)
        (self.logs / "20261021T1103Z-live.jsonl").write_text("{}\n")
        # main: a deployed commit, then only operational commits (logs, locks) -> QA has nothing to publish
        (self.code / "audit").mkdir()
        git(self.code, "init", "-q", "-b", "main")
        git(self.code, "add", "-A"); git(self.code, "commit", "-qm", "data", date="2026-10-21T02:00:00Z")
        self.deployed = git(self.code, "rev-parse", "HEAD")
        (self.code / "audit/deployed.json").write_text(json.dumps({"commit": self.deployed}))
        (self.code / "qa.lock").write_text("x")
        git(self.code, "add", "-A"); git(self.code, "commit", "-qm", "receipt and lock", date="2026-10-21T03:00:00Z")
        patcher = patch.object(W, "CODE", str(self.code)); patcher.start(); self.addCleanup(patcher.stop)

    def report(self, now=NOW, first_seen=None):
        return W.report(now=now, roots=["models/m"], logs=str(self.logs), first_seen=first_seen)


class Withdrawn(Fixture):
    def rows(self, first_seen=None):
        return {r["repo"]: r for r in W.withdrawn(str(self.root), NOW, 6, first_seen, "m")}

    def test_reviewed_values_a_remapping_did_not_bring_back_are_listed(self):
        r = self.rows()["hand/changed"]
        self.assertEqual((r["reviewed"], r["current"], r["lost"]), (4, 1, 3))
        self.assertEqual(r["parts"], ["part1.py"]); self.assertTrue(r["hand_reviewed"]); self.assertFalse(r["image"])
        self.assertTrue(r["overdue"]); self.assertEqual(r["hours"], 11.0)

    def test_restored_current_and_unknown_repos_are_not_listed(self):
        rows = self.rows()
        self.assertNotIn("hand/restored", rows)
        self.assertNotIn("hand/current", rows)

    def test_withdrawn_sizes_count_like_published_values(self):
        self.assertEqual(self.rows()["hand/sizes"]["lost"], 2)

    def test_image_only_claims_are_marked(self):
        r = self.rows()["img/charts"]
        self.assertTrue(r["image"]); self.assertEqual(r["lost"], 2)

    def test_a_card_that_changed_this_hour_is_within_grace(self):
        r = self.rows()["hand/recent"]
        self.assertFalse(r["overdue"]); self.assertEqual(r["since"], "2026-10-21T10:00Z")

    def test_saved_first_seen_time_wins_over_a_refreshed_card_time(self):
        r = self.rows({"m/hand/recent": "2026-10-20T00:00Z"})["hand/recent"]
        self.assertTrue(r["overdue"]); self.assertEqual(r["since"], "2026-10-20T00:00Z")


class Tasks(Fixture):
    def test_live_updates_need_a_recent_log(self):
        self.assertFalse(W.live_task(NOW, 3, str(self.logs))["overdue"])
        (self.logs / "20261021T1103Z-live.jsonl").unlink()
        (self.logs / "20261021T0603Z-live.jsonl").write_text("{}\n")
        self.assertTrue(W.live_task(NOW, 3, str(self.logs))["overdue"])

    def test_idle_qa_is_not_stalled_but_old_unpublished_changes_are(self):
        self.assertEqual(W.qa_task(NOW, 4)["overdue"], False)          # only a lock and receipt since the deployed commit
        (self.code / "logs/scan/20261021T0700Z-qa.jsonl").write_text("{}\n")
        git(self.code, "add", "-A"); git(self.code, "commit", "-qm", "log", date="2026-10-21T07:00:00Z")
        self.assertFalse(W.qa_task(NOW, 4)["overdue"])                 # logs are not publication changes
        (self.root / "snapshots/index.json").write_text("{}")
        git(self.code, "add", "-A"); git(self.code, "commit", "-qm", "collect", date="2026-10-21T09:00:00Z")
        qa = W.qa_task(NOW, 4)
        self.assertFalse(qa["overdue"]); self.assertEqual(qa["hours"], 3.0)
        self.assertTrue(W.qa_task(NOW + dt.timedelta(hours=2), 4)["overdue"])

    def test_qa_fails_loud_without_a_comparable_deployed_commit(self):
        (self.code / "audit/deployed.json").write_text(json.dumps({"commit": "f" * 40}))
        self.assertTrue(W.qa_task(NOW, 4)["overdue"])
        (self.code / "audit/deployed.json").write_text("{}")
        self.assertTrue(W.qa_task(NOW, 4)["overdue"])


class Issue(Fixture):
    def run_issue(self, rep, open_issue, fail=None):
        calls = []
        def gh(*args, input=None):
            calls.append((args, input))
            if fail and args[:2] == fail: raise W.GhError("boom")
            return "https://github.com/x/y/issues/7\n"
        with patch.object(W, "gh", gh):
            result = W.issue(rep, open_issue)
        return result, [c[0][:2] for c in calls], calls

    def test_report_ages_do_not_change_the_problem_digest(self):
        a, b = self.report(), self.report(NOW + dt.timedelta(hours=1))
        self.assertEqual(W.problems(a), W.problems(b))
        self.assertIn("| [hand/changed](https://huggingface.co/hand/changed) | 4 | 1 | 3 | hand review |", W.markdown(a))

    def test_issue_opens_then_updates_quietly_then_comments_on_news_then_closes(self):
        rep = self.report()
        result, kinds, calls = self.run_issue(rep, None)
        self.assertEqual(kinds, [("issue", "create")]); self.assertTrue(result.startswith("opened"))
        body = calls[0][1]
        existing = {"number": 7, "title": W.TITLE, "body": body}
        self.assertEqual(W.saved_state(existing)["m/hand/recent"], "2026-10-21T10:00Z")   # first-seen times persist
        result, kinds, _ = self.run_issue(self.report(NOW + dt.timedelta(hours=1), W.saved_state(existing)), existing)
        self.assertEqual(kinds, [("issue", "edit")])                                       # same problems: no comment
        (self.root / "verified/part1.py").write_text("R = []\n")                            # nothing withdrawn any more
        (self.logs / "20261021T1103Z-live.jsonl").unlink()                                  # but live updates stopped
        result, kinds, _ = self.run_issue(self.report(), existing)
        self.assertEqual(kinds, [("issue", "edit"), ("issue", "comment")]); self.assertIn("commented", result)
        (self.logs / "20261021T1103Z-live.jsonl").write_text("{}\n")
        result, kinds, _ = self.run_issue(self.report(), existing)
        self.assertEqual(result, "closed #7"); self.assertEqual(kinds, [("issue", "comment"), ("issue", "close")])
        result, kinds, _ = self.run_issue(self.report(), None)
        self.assertEqual(result, "all clear; no issue open"); self.assertEqual(kinds, [])

    def test_gh_failure_is_raised_not_swallowed(self):
        with self.assertRaises(W.GhError):
            self.run_issue(self.report(), None, fail=("issue", "create"))

    def test_only_the_workflows_own_issue_is_adopted(self):
        seen = []
        def gh(*args, input=None):
            seen.append(args); return json.dumps([{"number": 3, "title": W.TITLE + " (copy)", "body": ""}])
        with patch.object(W, "gh", gh):
            self.assertIsNone(W.find_issue())
        self.assertIn("--author", seen[0]); self.assertEqual(seen[0][seen[0].index("--author") + 1], "app/github-actions")

    def test_issue_body_stays_under_githubs_limit(self):
        rep = self.report()
        rows = rep["withdrawn"]["m"]
        rep["withdrawn"]["m"] = [dict(rows[0], repo=f"org/repo-{i:04d}-" + "x" * 60) for i in range(3000)]
        body = W.issue_body(rep)
        self.assertLessEqual(len(body), W.MAX_BODY); self.assertIn("more repos", body)


if __name__ == "__main__":
    unittest.main()

"""Failure-injection tests for complete, non-destructive static publication."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assemble


def digest(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


class AssemblySafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.out = self.base / "published"
        self.watchlist = {"models": [{"slug": "first", "model": "test/first", "status": "live"},
                                     {"slug": "second", "model": "test/second", "status": "live"},
                                     {"slug": "pending", "model": "test/pending", "status": "ingesting"}]}
        self.write_watchlist()
        for slug in ("first", "second"):
            root = self.repo / "models" / slug
            site = root / "site"
            site.mkdir(parents=True)
            (root / "model.json").write_text(json.dumps({"slug": slug, "model": "test/" + slug}))
            (site / "index.html").write_text("<!doctype html><title>" + slug + "</title>")
            (site / "artifact.html").write_text("private preview")
            (site / "data.json").write_text(json.dumps({"model": "test/" + slug,
                "captured_at": "2026-10-19T00:00:00Z", "repos": [], "rows": [], "sets": []}))
            (site / "sitemap.xml").write_text(f'<urlset xmlns="{assemble.SITEMAP_NS}"><url><loc>{assemble.BASE}{slug}/</loc></url></urlset>')
        assemble.assemble(self.out, self.repo)
        self.before = digest(self.out)

    def write_watchlist(self):
        (self.repo / "watchlist.json").write_text(json.dumps(self.watchlist))

    def site(self, slug="second"):
        return self.repo / "models" / slug / "site"

    def rejected(self, pattern):
        with self.assertRaisesRegex((ValueError, OSError), pattern):
            assemble.assemble(self.out, self.repo)
        self.assertEqual(digest(self.out), self.before, "last good output changed on failure")
        self.assertFalse(list(self.base.glob(".published-staging-*")))
        self.assertFalse(list(self.base.glob(".published-previous-*")))

    def test_success_is_complete_and_removes_only_previous_output(self):
        self.assertEqual([m["slug"] for m in json.loads((self.out / "models.json").read_text())["models"]], ["first", "second"])
        self.assertFalse((self.out / "first/artifact.html").exists())
        self.assertFalse((self.out / "pending").exists())
        (self.out / "obsolete.txt").write_text("previous generated content")
        self.site("first").joinpath("index.html").write_text("<!doctype html><title>changed</title>")
        assemble.assemble(self.out, self.repo)
        self.assertFalse((self.out / "obsolete.txt").exists())
        self.assertIn("changed", (self.out / "first/index.html").read_text())

    def test_source_and_ancestor_destinations_are_rejected_without_changes(self):
        before = digest(self.repo)
        for destination in (self.repo, self.base, self.repo / "models", self.site(), self.repo / "web/new"):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                assemble.assemble(destination, self.repo)
        self.assertEqual(digest(self.repo), before)

    def test_unrelated_nonempty_destination_is_preserved(self):
        unrelated = self.base / "notes"
        unrelated.mkdir()
        (unrelated / "sentinel.txt").write_text("never delete")
        with self.assertRaisesRegex(ValueError, "not a QFF assembly"):
            assemble.assemble(unrelated, self.repo)
        self.assertEqual((unrelated / "sentinel.txt").read_text(), "never delete")

    def test_empty_temp_destination_remains_supported(self):
        empty = self.base / "empty"
        empty.mkdir()
        assemble.assemble(empty, self.repo)
        self.assertEqual(digest(empty), self.before)

    def test_missing_live_config_does_not_silently_publish_partial_site(self):
        shutil.rmtree(self.repo / "models/second")
        self.rejected("live models have no model.json")

    def test_missing_live_build_does_not_silently_publish_partial_site(self):
        shutil.rmtree(self.site())
        self.rejected("missing a regular site directory")

    def test_incomplete_live_build_is_rejected(self):
        self.site().joinpath("index.html").unlink()
        self.rejected("missing index.html")

    def test_data_identity_mismatch_is_rejected(self):
        data = json.loads(self.site().joinpath("data.json").read_text())
        data["model"] = "different/model"
        self.site().joinpath("data.json").write_text(json.dumps(data))
        self.rejected("model identity mismatch")

    def test_invalid_data_schema_is_rejected(self):
        data = json.loads(self.site().joinpath("data.json").read_text())
        data["sets"] = {}
        self.site().joinpath("data.json").write_text(json.dumps(data))
        self.rejected("sets must be a list")

    def test_invalid_json_preserves_last_good_output(self):
        self.site().joinpath("data.json").write_text("{invalid")
        self.rejected("Expecting property name")

    def test_slug_path_traversal_is_rejected(self):
        root = self.site().parent
        (root / "model.json").write_text(json.dumps({"slug": "../../outside", "model": "test/second"}))
        self.rejected("slug must match")

    def test_watchlist_mismatch_is_rejected(self):
        self.watchlist["models"][1]["model"] = "other/model"
        self.write_watchlist()
        self.rejected("model identity does not match watchlist")

    def test_duplicate_watchlist_slug_is_rejected(self):
        self.watchlist["models"].append(dict(self.watchlist["models"][0]))
        self.write_watchlist()
        self.rejected("duplicate slug")

    def test_hidden_nonstatic_file_is_rejected(self):
        self.site().joinpath(".env").write_text("synthetic secret")
        self.rejected("non-static file")

    def test_source_symlink_is_rejected(self):
        self.site().joinpath("leak.txt").symlink_to(self.repo / "watchlist.json")
        self.rejected("symlinks are not publishable")

    def test_destination_symlink_is_rejected(self):
        link = self.base / "alias"
        link.symlink_to(self.out, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            assemble.assemble(link, self.repo)
        self.assertEqual(digest(self.out), self.before)

    def test_render_failure_keeps_last_good_output_and_cleans_staging(self):
        with patch.object(assemble, "render_site", side_effect=ValueError("injected rendering failure")):
            self.rejected("injected rendering failure")

    def test_commit_failure_restores_last_good_output(self):
        replace = assemble.os.replace
        def fail_commit(source, destination):
            if "-staging-" in Path(source).name:
                raise OSError("injected commit failure")
            return replace(source, destination)
        with patch.object(assemble.os, "replace", side_effect=fail_commit):
            self.rejected("injected commit failure")

    def test_backup_cleanup_failure_does_not_report_a_failed_publication(self):
        remove = assemble.shutil.rmtree
        def fail_backup_cleanup(path, *args, **kwargs):
            if "-previous-" in Path(path).name:
                raise OSError("injected cleanup failure")
            return remove(path, *args, **kwargs)
        self.site("first").joinpath("index.html").write_text("<!doctype html><title>new valid build</title>")
        with patch.object(assemble.shutil, "rmtree", side_effect=fail_backup_cleanup), contextlib.redirect_stderr(io.StringIO()) as warning:
            result = assemble.assemble(self.out, self.repo)
        self.assertEqual(result["out"], str(self.out))
        self.assertIn("new valid build", self.out.joinpath("first/index.html").read_text())
        self.assertIn("Assembled successfully", warning.getvalue())
        backups = list(self.base.glob(".published-previous-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(digest(backups[0]), self.before)


if __name__ == "__main__":
    unittest.main()

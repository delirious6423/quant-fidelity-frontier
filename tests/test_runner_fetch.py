"""Offline fault injection for the scripted direct-fetch integration."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'pipeline'))
import runner_fetch as R


class RunnerFetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'shadow'
        self.ledger = Path(self.tmp.name) / 'ledger'
        R.atomic(self.root / 'model.json', {'model': 'test/base', 'slug': 'test', 'name': 'Test'})
        self.rev = 'a' * 40
        self.cat = [{'id': 'test/one', 'sha': self.rev}, {'id': 'test/two', 'sha': self.rev}]
        self.calls = collections.Counter()
        self.fail = {}
        self.patch = patch.object(R.fetch, 'get', self.get)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def get(self, url, token, stop=None, **kwargs):
        R.fetch.check_stopped(stop)
        self.calls[url] += 1
        if url in self.fail:
            what = self.fail[url]
            if what == 429:
                stop.set()
                raise R.fetch.RateLimited(url, '0')
            if what == 'terminate':
                raise KeyboardInterrupt('fault injection')
            return what, b'private response body', {}
        if '/api/models?' in url:
            return 200, json.dumps(self.cat).encode(), {}
        if '/tree/' in url:
            return 200, b'[{"type":"file","size":123}]', {}
        if '/api/models/' in url:
            return 200, json.dumps({'sha': self.rev}).encode(), {}
        return 200, b'| Quant | Mean KLD |\n|---|---|\n| Q4 | 0.125 |\n', {}

    def run_fetch(self):
        return R.collect(self.root, self.ledger, 'test/base', workers=1, base='https://example.test')

    def card_url(self, repo):
        return f'https://example.test/{repo}/raw/{self.rev}/README.md'

    def test_complete_unchanged_and_revision_only_change(self):
        R.atomic(self.root / 'cache/not_fetched.json', ['test/one'])
        self.assertEqual(self.run_fetch()['status'], 'complete')
        self.assertEqual(R.read(self.root / 'cache/not_fetched.json'), [])
        self.assertEqual(self.run_fetch()['changed'], [])
        self.assertEqual(self.calls[self.card_url('test/one')], 1)
        self.rev = 'b' * 40
        self.cat = [dict(x, sha=self.rev) for x in self.cat]
        self.assertEqual(self.run_fetch()['changed'], [])
        snapshot = R.read(self.root / 'snapshots/index.json')
        self.assertEqual(snapshot['repos']['test/one']['sha'], self.rev)
        self.assertEqual(snapshot['repos']['test/one']['sha256'], hashlib.sha256((self.root / snapshot['repos']['test/one']['file']).read_bytes()).hexdigest())

    def test_rate_limit_retains_progress_and_resume(self):
        self.fail[self.card_url('test/two')] = 429
        self.assertEqual(self.run_fetch()['status'], 'blocked')
        self.assertFalse((self.root / 'snapshots/index.json').exists())
        self.assertIn('test/one', R.read(self.ledger / 'index.json')['repos'])
        self.fail.clear()
        self.assertEqual(self.run_fetch()['status'], 'complete')
        self.assertEqual(self.calls[self.card_url('test/one')], 1)

    def test_cooldown_prevents_requests(self):
        self.ledger.mkdir()
        R.atomic(self.ledger / 'rate_limit.json', {'resume_after': 10 ** 12})
        self.assertEqual(self.run_fetch()['unresolved'][0]['status'], 'rate_limit_cooldown')
        self.assertFalse(self.calls)

    def test_partial_tree_and_portable_restored_checkpoint(self):
        tree = f'https://example.test/api/models/test/two/tree/{self.rev}?recursive=true'
        self.fail[tree] = 503
        self.assertEqual(self.run_fetch()['status'], 'blocked')
        moved = Path(self.tmp.name) / 'restored'
        shutil.copytree(self.ledger, moved)
        shutil.rmtree(self.ledger)
        self.ledger = moved
        self.fail.clear()
        self.assertEqual(self.run_fetch()['status'], 'complete')
        self.assertEqual(self.calls[self.card_url('test/two')], 1)
        raw = (self.ledger / 'index.json').read_text()
        self.assertNotIn(self.tmp.name, raw)

    def test_failed_changed_fetch_cannot_stamp_old_snapshot_current(self):
        self.run_fetch()
        before = (self.root / 'snapshots/index.json').read_bytes()
        self.rev = 'b' * 40
        self.cat = [dict(x, sha=self.rev) for x in self.cat]
        self.fail[self.card_url('test/one')] = 403
        outcome = self.run_fetch()
        self.assertEqual(outcome['status'], 'blocked')
        self.assertIn({'repo': 'test/one', 'status': 'access_denied'}, outcome['unresolved'])
        self.assertEqual((self.root / 'snapshots/index.json').read_bytes(), before)
        self.assertEqual(R.read(self.root / 'audit/runner_fetch.json')['status'], 'blocked')
        self.assertNotIn('private response body', json.dumps(outcome))

    def test_hash_mismatch_is_refetched_not_trusted(self):
        self.run_fetch()
        rec = R.read(self.ledger / 'index.json')['repos']['test/one']
        (self.ledger / rec['file']).write_bytes(b'corrupted')
        self.assertEqual(self.run_fetch()['status'], 'complete')
        self.assertEqual(self.calls[self.card_url('test/one')], 2)

    def test_missing_and_genuine_empty_remain_distinct(self):
        self.fail[self.card_url('test/one')] = 404
        with patch.object(R.fetch, 'get', side_effect=lambda url, token, **kw: (200, b'', {}) if url == self.card_url('test/two') else self.get(url, token, **kw)):
            self.assertEqual(self.run_fetch()['status'], 'complete')
        records = R.read(self.ledger / 'index.json')['repos']
        self.assertEqual(records['test/one']['readme_status'], 404)
        self.assertEqual(records['test/two']['readme_status'], 200)
        self.assertEqual(records['test/one']['bytes'], 0)
        self.assertEqual(records['test/two']['bytes'], 0)

    def test_explicit_and_previous_coverage_retained(self):
        self.run_fetch()
        self.cat = []
        result = R.collect(self.root, self.ledger, 'test/base', include_repos=['test/extra'], workers=1, base='https://example.test')
        self.assertEqual(result['repos'], 3)
        self.assertEqual(set(R.read(self.root / 'snapshots/index.json')['repos']), {'test/one', 'test/two', 'test/extra'})

    def test_catalogue_pagination_cycle_and_traversal_rejected(self):
        import threading
        def page(url, token, **kw):
            return 200, json.dumps(self.cat).encode(), {'Link': f'<{url}>; rel="next"'}
        with patch.object(R.fetch, 'get', page), self.assertRaisesRegex(ValueError, 'cycle'):
            R.catalogue('https://example.test', 'test/base', None, threading.Event())
        self.cat = [{'id': '../../escape', 'sha': self.rev}]
        self.assertEqual(self.run_fetch()['status'], 'blocked')
        self.assertFalse((self.root / 'snapshots/index.json').exists())

    def test_termination_after_card_checkpoint_resumes_without_refetch(self):
        tree = f'https://example.test/api/models/test/one/tree/{self.rev}?recursive=true'
        self.fail[tree] = 'terminate'
        with self.assertRaises(KeyboardInterrupt):
            self.run_fetch()
        self.fail.clear()
        self.assertEqual(self.run_fetch()['status'], 'complete')
        self.assertEqual(self.calls[self.card_url('test/one')], 1)

    def test_catalogue_and_tree_pagination_are_complete(self):
        original = self.get
        def pages(url, token, **kwargs):
            if url.endswith('/catalogue-page-two'):
                return 200, json.dumps(self.cat[1:]).encode(), {}
            if '/api/models?' in url:
                return 200, json.dumps(self.cat[:1]).encode(), {'Link': '</catalogue-page-two>; rel="next"'}
            if '/tree/' in url:
                return 200, b'[{"type":"file","size":123}]', {'Link': '</tree-page-two>; rel="next"'}
            if url.endswith('/tree-page-two'):
                return 200, b'[{"type":"file","size":456}]', {}
            return original(url, token, **kwargs)
        with patch.object(R.fetch, 'get', pages):
            self.assertEqual(self.run_fetch()['repos'], 2)
        records = R.read(self.ledger / 'index.json')['repos']
        self.assertEqual(records['test/one']['tree_pages'], 2)
        self.assertEqual(records['test/one']['total_bytes'], 579)

    def test_failed_explicit_inclusion_survives_next_run_without_cli_flag(self):
        self.fail[self.card_url('test/extra')] = 403
        result = R.collect(self.root, self.ledger, 'test/base', include_repos=['test/extra'], workers=1, base='https://example.test')
        self.assertEqual(result['status'], 'blocked')
        self.fail.clear()
        self.assertEqual(self.run_fetch()['repos'], 3)
        self.assertIn('test/extra', R.read(self.root / 'snapshots/index.json')['repos'])

    def test_collector_contention_and_nested_symlinks_rejected(self):
        import fcntl
        self.ledger.mkdir()
        with open(self.ledger / 'collector.lock', 'a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_fetch()
        self.assertFalse(self.calls)
        (self.root / 'cache').symlink_to(self.ledger, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            self.run_fetch()

    def test_production_root_and_untrusted_local_paths_refused(self):
        with self.assertRaisesRegex(ValueError, 'isolated'):
            R.collect(CODE / 'models/qwen3.8-flash-next', self.ledger, 'test/base')
        for name in ('../outside', '/etc/passwd'):
            with self.assertRaises(ValueError):
                R.local(self.ledger, name)


if __name__ == '__main__':
    unittest.main(verbosity=2)

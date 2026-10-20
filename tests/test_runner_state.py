"""Fault injection and contention for the durable local journal."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from runner_state import State
from runner_workspace import Workspace, CODE, restore, write


class Journal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'state.sqlite'
        self.s = State(self.path)

    def add(self, n=1):
        return [self.s.enqueue(f'm{i}:A', f'm{i}', 'A', 'v1', {'i': i}) for i in range(n)]

    def test_more_tasks_than_concurrency_and_no_restart_repeat(self):
        self.add(25)
        self.assertEqual(len(self.s.tasks()), 25)
        while True:
            batch = [self.s.claim(limit=3) for _ in range(3)]
            self.assertIsNone(self.s.claim(limit=3))
            batch = [b for b in batch if b]
            if not batch:
                break
            for c in batch:
                self.s.complete(c['id'], c['token'], {'ok': 1})
        restored = State(self.path)
        self.assertIsNone(restored.claim())
        self.assertEqual(sum(t['state'] == 'succeeded' for t in restored.tasks()), 25)

    def test_claim_crash_lease_backoff_stale_worker(self):
        self.add()
        c = self.s.claim(now=100, lease_seconds=1)
        self.assertIsNone(self.s.claim(now=102))
        second = self.s.claim(now=105)
        with self.assertRaises(ValueError):
            self.s.complete(c['id'], c['token'], {}, now=106)
        self.s.complete(second['id'], second['token'], {}, now=106)

    def test_result_persisted_before_ack_duplicate_and_conflict(self):
        self.add()
        c = self.s.claim()
        self.s.complete(c['id'], c['token'], {'x': 1})
        restored = State(self.path)
        self.assertEqual(restored.complete(c['id'], c['token'], {'x': 1}), 'duplicate')
        with self.assertRaises(ValueError):
            restored.complete(c['id'], c['token'], {'x': 2})
        self.assertEqual(restored.tasks()[0]['state'], 'blocked')

    def test_source_inflight_and_version_invalidation(self):
        self.add()
        c = self.s.claim()
        new = self.s.enqueue('m0:A', 'm0', 'A', 'v2', {'i': 0})
        with self.assertRaises(ValueError):
            self.s.complete(c['id'], c['token'], {})
        self.assertEqual(self.s.claim()['id'], new)

    def test_atomic_contention(self):
        self.add(30)
        with concurrent.futures.ThreadPoolExecutor(12) as pool:
            claims = list(pool.map(lambda _: State(self.path).claim(limit=8), range(12)))
        ids = [c['id'] for c in claims if c]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)

    def test_bounded_retry_and_reason(self):
        self.add()
        for now in (100, 200, 300):
            c = self.s.claim(now=now)
            self.s.fail(c['id'], c['token'], 'transient', now=now)
        self.assertIsNone(self.s.claim(now=500))
        self.assertEqual(self.s.tasks()[0]['state'], 'blocked')
        with self.assertRaises(ValueError):
            self.s.retry(c['id'], '')
        self.s.retry(c['id'], 'provider repaired')
        self.assertIsNotNone(self.s.claim(now=500))

    def test_fair_between_models_and_stages(self):
        for i in range(20):
            self.s.enqueue(f'a:{i}', 'a', 'A', 'v1', {'i': i})
        self.s.enqueue('b:C', 'b', 'C', 'v1', {})
        a = self.s.claim(); self.s.complete(a['id'], a['token'], {})
        self.assertEqual(self.s.claim()['stage'], 'C')

    def test_actual_process_exit_after_claim_and_after_persist(self):
        self.add()
        script = "import os,sys;sys.path.insert(0,sys.argv[1]);from runner_state import State;s=State(sys.argv[2]);s.claim(now=1,lease_seconds=1);os._exit(19)"
        p = subprocess.run([sys.executable, '-c', script, str(CODE / 'pipeline'), str(self.path)])
        self.assertEqual(p.returncode, 19)
        self.s.claim(now=10)
        c = self.s.claim(now=20)
        script = "import os,sys;sys.path.insert(0,sys.argv[1]);from runner_state import State;s=State(sys.argv[2]);s.complete(sys.argv[3],sys.argv[4],{'ok':1},now=21);os._exit(23)"
        p = subprocess.run([sys.executable, '-c', script, str(CODE / 'pipeline'), str(self.path), c['id'], c['token']])
        self.assertEqual(p.returncode, 23)
        self.assertEqual(self.s.tasks()[0]['state'], 'succeeded')

    def test_backup_restore_partial_qa(self):
        root = Path(self.tmp.name) / 'run'; root.mkdir()
        write(root / 'runner.json', {'version': 1, 'mode': 'shadow', 'models': []})
        s = State(root / 'state.sqlite')
        for i in range(3):
            s.enqueue(f'qa:{i}', 'm', 'review', 'v1', {'i': i})
        c = s.claim(); s.complete(c['id'], c['token'], {'verdict': 'ok'})
        w = Workspace(root)
        w.backup(Path(self.tmp.name) / 'backup', s)
        new = restore(Path(self.tmp.name) / 'backup', Path(self.tmp.name) / 'restored')
        restored = State(new.path / 'state.sqlite')
        self.assertEqual([t['state'] for t in restored.tasks()], ['succeeded', 'pending', 'pending'])
        self.assertNotEqual(restored.claim()['id'], c['id'])

    def test_publication_and_authoritative_paths_rejected(self):
        with self.assertRaises(ValueError):
            Workspace.create(CODE / 'work' / 'unsafe', [])
        root = Path(self.tmp.name) / 'run'; root.mkdir()
        write(root / 'runner.json', {'version': 1, 'mode': 'shadow', 'models': []})
        w = Workspace(root)
        for script in ('qcl_qa.py', 'qcl_next.py', 'qcl_scan.py', 'git', 'deploy_state.py'):
            with self.assertRaises(ValueError):
                w.run('unused', script, 'publish')
        with self.assertRaises(ValueError):
            w.publish()
        (root / 'escape').symlink_to(CODE / 'models', target_is_directory=True)
        with self.assertRaises(ValueError):
            Workspace(root)


if __name__ == '__main__':
    unittest.main()

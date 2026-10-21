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

    def test_filtered_queries_match_full_listing_without_payload_decoding(self):
        ids = self.add(6)
        self.s.enqueue('m0:B', 'm0', 'B', 'v1', {'i': 'b'})
        c = self.s.claim(); self.s.complete(c['id'], c['token'], {'ok': 1})
        full = self.s.tasks()
        self.assertEqual([t['id'] for t in self.s.tasks(model='m0')], [t['id'] for t in full if t['model'] == 'm0'])
        self.assertEqual([t['id'] for t in self.s.tasks(stage='B')], [t['id'] for t in full if t['stage'] == 'B'])
        self.assertEqual([t['id'] for t in self.s.tasks(states=('succeeded',))], [t['id'] for t in full if t['state'] == 'succeeded'])
        self.assertEqual(self.s.tasks(states=()), [])
        light = self.s.tasks(with_payload=False)
        self.assertEqual([t['id'] for t in light], [t['id'] for t in full])
        self.assertTrue(all(t['payload'] is None for t in light))
        self.assertEqual(light[0]['result'], {'ok': 1})
        self.assertEqual(self.s.get(ids[3])['payload'], {'i': 3})
        self.assertIsNone(self.s.get('missing'))
        self.assertEqual(self.s.current('m0:B')['payload'], {'i': 'b'})
        self.assertIsNone(self.s.current('m9:A'))
        self.assertEqual(self.s.counts(), {'succeeded': 1, 'pending': 6})
        # A superseded slot resolves to its active replacement only.
        new = self.s.enqueue('m0:B', 'm0', 'B', 'v2', {'i': 'b2'})
        self.assertEqual(self.s.current('m0:B')['id'], new)
        self.assertEqual(len(self.s.tasks(active=False)), 8)

    def test_enqueue_burst_is_one_durable_commit_or_nothing(self):
        class Interrupted(Exception):
            pass
        with self.assertRaises(Interrupted):
            with self.s.transaction():
                self.add(5)
                self.assertEqual(len(self.s.tasks()), 5)  # visible inside the burst
                raise Interrupted
        self.assertEqual(State(self.path).tasks(), [])  # nothing persisted from an interrupted burst
        with self.s.transaction():
            ids = self.add(5)
            with self.s.transaction():  # nested bursts join the outer commit
                self.s.enqueue('m0:A', 'm0', 'A', 'v2', {'i': 'new'})
        fresh = State(self.path)
        self.assertEqual(len(fresh.tasks()), 5)
        self.assertEqual(fresh.counts(), {'pending': 5})
        self.assertNotIn(ids[0], {t['id'] for t in fresh.tasks()})  # superseded inside the burst
        self.assertEqual(json.loads(fresh.payload_text(fresh.current('m0:A')['id'])), {'i': 'new'})
        self.assertIsNotNone(fresh.claim())
        with self.assertRaises(KeyError):
            fresh.payload_text('missing')

    def test_install_preserves_json_types_and_repairs_malformed_values(self):
        from runner_workspace import install, read
        path = Path(self.tmp.name) / 'map.json'
        write(path, {'weights_modified': 0})
        self.assertTrue(install(path, {'weights_modified': False}))
        self.assertIs(read(path)['weights_modified'], False)
        self.assertFalse(install(path, {'weights_modified': False}))

    def test_light_listing_does_not_read_payload_column(self):
        import sqlite3
        from unittest.mock import patch
        self.add(2)
        original = self.s._open
        def open_checked():
            db = original()
            db.set_authorizer(lambda op, table, column, *_:
                sqlite3.SQLITE_DENY if op == sqlite3.SQLITE_READ and table == 'tasks' and column == 'payload'
                else sqlite3.SQLITE_OK)
            return db
        with patch.object(self.s, '_open', open_checked):
            self.assertEqual(len(self.s.tasks(with_payload=False)), 2)

    def test_claim_scope_large_eligibility_fairness_and_durable_commits(self):
        import sqlite3
        from unittest.mock import patch
        with self.s.transaction():
            ids = [self.s.enqueue(f'a:{i}', 'a', 'map-A', 'v1', {}) for i in range(30)]
            other = self.s.enqueue('b', 'b', 'map-C', 'v1', {})
            missing = self.s.enqueue('missing', 'c', 'review', 'v1', {})
        original = self.s._open
        def limited():
            db = original()
            if hasattr(db, 'setlimit'):  # Python 3.11+ can lower the build's limit for this test
                db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 32)
            return db
        # Exceeds even the common 32766-variable build, with a deliberately low test limit.
        eligible = ids + [other] + [f'absent-{i}' for i in range(33000)]
        with patch.object(self.s, '_open', limited), self.s.claim_scope(eligible) as claim:
            a = claim(limit=2)
            b = claim(limit=2)
            self.assertEqual(b['id'], other)
            self.assertIsNone(claim(limit=2))
            fresh = State(self.path)
            self.assertEqual(fresh.get(a['id'])['state'], 'running')
            fresh.complete(a['id'], a['token'], {})
            fresh.complete(b['id'], b['token'], {})
            count = 2
            while c := claim(limit=2):
                fresh.complete(c['id'], c['token'], {})
                count += 1
            self.assertEqual(count, 31)
        self.assertEqual(self.s.get(missing)['attempts'], 0)
        self.assertIsNone(self.s.claim(eligible_ids=[]))
        self.assertEqual(self.s.claim(eligible_ids=[missing])['id'], missing)

    def test_scoped_claim_contention_invalidation_and_lease_recovery(self):
        ids = self.add(20)
        def take(_):
            worker = State(self.path)
            with worker.claim_scope(ids) as claim:
                return claim(limit=4, now=100, lease_seconds=1)
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            claims = [c for c in pool.map(take, range(8)) if c]
        self.assertEqual(len({c['id'] for c in claims}), 4)
        replaced = self.s.enqueue('m0:A', 'm0', 'A', 'v2', {})
        with self.s.claim_scope(ids) as claim:
            # Scope never makes an invalidated source eligible again.
            got = []
            while c := claim(now=200):
                got.append(c['id'])
                self.s.complete(c['id'], c['token'], {}, now=201)
            self.assertNotIn(ids[0], got)
            self.assertNotIn(replaced, got)
            while c := claim(now=210):
                self.s.complete(c['id'], c['token'], {}, now=211)
        self.assertEqual(self.s.get(replaced)['state'], 'pending')

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

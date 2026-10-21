"""End-to-end orchestration counterfactuals, all in isolated temporary workspaces."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
import qcl_runner as R
from runner_replay import fixture, fake_response, replay
from runner_state import State
from runner_workspace import Workspace, read, write, CODE


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='runner-test-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.w = fixture(self.base / 'run')
        self.s = State(self.w.path / 'state.sqlite')

    def prepare(self):
        R.reconcile(self.w, self.s)

    def finish(self):
        for _ in range(4):
            self.prepare()
            while c := self.s.claim(limit=3):
                R.complete(self.w, self.s, c, fake_response(c['payload']))
        self.prepare()

    def test_old_new_shadow_all_claims_and_provenance(self):
        result = replay(self.base / 'replay')
        self.assertTrue(result['shadow']['equal'])
        self.assertEqual(result['shadow']['restart_repeated_stage_operations'], 0)
        self.assertEqual(result['scripted_dispatch']['completed'], 24)

    def test_partial_response_files_never_orphan_or_starve(self):
        self.prepare()
        tasks = self.s.tasks()
        responses = self.base / 'responses'; responses.mkdir()
        # The oldest request has no file; ready work later in the queue must run.
        for t in tasks[1:4]:
            write(responses / (t['id'] + '.json'), fake_response(t['payload']))
        self.assertEqual(R.run_responses(self.w, self.s, responses, 3), 3)
        self.assertFalse(any(t['state'] == 'running' for t in self.s.tasks()))
        self.assertEqual(self.s.tasks()[0]['state'], 'pending')

    def test_expired_running_file_adapter_recovers_without_manual_claim(self):
        self.prepare()
        c = self.s.claim(now=1, lease_seconds=1)
        responses = self.base / 'responses'; responses.mkdir()
        write(responses / (c['id'] + '.json'), fake_response(c['payload']))
        self.assertEqual(R.run_responses(self.w, State(self.s.path), responses, 3), 0)
        task = next(t for t in self.s.tasks() if t['id'] == c['id'])
        self.assertEqual(task['state'], 'failed')  # waiting for bounded backoff
        with self.s.connect() as db:
            db.execute('UPDATE tasks SET available=0 WHERE id=?', (c['id'],))
        self.assertEqual(R.run_responses(self.w, State(self.s.path), responses, 3), 1)

    def test_version_bump_cannot_keep_old_maps_active(self):
        self.finish()
        old_ids = {t['id'] for t in self.s.tasks() if t['stage'].startswith('map-')}
        with patch.object(R, 'version', return_value='new-extractor-version'):
            self.prepare()
            active = [t for t in self.s.tasks() if t['stage'].startswith('map-')]
            self.assertTrue(active)
            self.assertTrue(all(t['id'] not in old_ids and t['state'] == 'pending' for t in active))
            self.assertEqual(R.records(self.w.model('fixture'))[0]['variants'], [])  # image-only map remains

    def test_source_changes_same_numbers_reject_late_result_and_requeue(self):
        self.prepare()
        c = self.s.claim()
        root = self.w.model('fixture'); repo = c['payload']['source']['repo']
        idx = read(root / 'snapshots/index.json'); meta = idx['repos'][repo]
        p = root / meta['file']; p.write_text(p.read_text().replace('Mean KLD', 'Mean KL (millinats)'))
        meta.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=len(p.read_bytes()))
        write(root / 'snapshots/index.json', idx)
        with self.assertRaises(ValueError):
            R.complete(self.w, self.s, c, fake_response(c['payload']))
        self.prepare()
        self.assertFalse(any(t['id'] == c['id'] for t in self.s.tasks()))
        self.assertTrue(any(t['payload']['source']['sha256'] == meta['sha256'] for t in self.s.tasks()))

    def test_partial_qa_persist_restore_and_wrong_hold_regeneration(self):
        self.prepare()
        for _ in range(2):
            while c := self.s.claim():
                R.complete(self.w, self.s, c, fake_response(c['payload']))
            self.prepare()
        review = next(t for t in self.s.tasks() if t['stage'] == 'review')
        c = self.s.claim(eligible_ids=[review['id']])
        R.complete(self.w, self.s, c, {'id': c['payload']['fingerprint'], 'verdict': 'wrong', 'reason': 'Fixture injected attribution error [L7].'})
        self.w.backup(self.base / 'backup', self.s)
        restored = R.restore(self.base / 'backup', self.base / 'restored')
        state = State(restored.path / 'state.sqlite')
        R.reconcile(restored, state)
        self.assertEqual(next(t for t in state.tasks() if t['id'] == c['id'])['state'], 'succeeded')
        holds = read(restored.model('fixture') / 'audit/holds.json')
        self.assertEqual(holds[0]['repo'], c['payload']['claim']['repo'])
        found = next(r for r in R.records(restored.model('fixture')) if r['repo'] == holds[0]['repo'])
        self.assertEqual(found['variants'], [])
        self.assertTrue(any(t['stage'] == 'review' and t['state'] == 'pending' for t in state.tasks()))

    def test_revision_only_update_reuses_completed_mapping_and_qa(self):
        self.finish()
        before = {t['id'] for t in self.s.tasks()}
        root = self.w.model('fixture')
        idx = read(root / 'snapshots/index.json')
        for meta in idx['repos'].values():
            meta['sha'] = 'b' * 40
        write(root / 'snapshots/index.json', idx)
        self.prepare()
        self.assertEqual({t['id'] for t in self.s.tasks()}, before)
        self.assertIsNone(self.s.claim())
        self.assertTrue(read(self.w.path / 'review-reuse.json'))

    def test_revision_only_with_over_twenty_and_inflight_qa_keeps_sample(self):
        self.finish()
        root = self.w.model('fixture')
        write(self.w.path / 'baseline-fixture.json', [])
        (self.w.path / 'review-sample.json').unlink()
        def bundles(root):
            meta, raw = R.source(root, 'Publisher/normal-GGUF')
            return [R.boundary.prepare_review('Base/Model', meta, raw,
                    {'repo': meta['repo'], 'kind': 'own', 'entry': 'Q4_K_M', 'key': 'bench:Fixture' + str(i),
                     'raw': '0.1250', 'lines': [7]}, mapping_version=R.version(), extractor_version=R.version())
                    for i in range(30)]
        with patch.object(R, 'review_bundles', side_effect=bundles):
            self.prepare()
            sample = read(self.w.path / 'review-sample.json')
            self.assertEqual(len(sample['items']), 20)
            c = self.s.claim()
            before = {t['id'] for t in self.s.tasks()}
            idx = read(root / 'snapshots/index.json')
            for meta in idx['repos'].values():
                meta['sha'] = 'c' * 40
            write(root / 'snapshots/index.json', idx)
            self.prepare()
            self.assertEqual(read(self.w.path / 'review-sample.json'), sample)
            self.assertEqual({t['id'] for t in self.s.tasks()}, before)
            R.complete(self.w, self.s, c, fake_response(c['payload']))
            while c := self.s.claim():
                R.complete(self.w, self.s, c, fake_response(c['payload']))
            for meta in idx['repos'].values():
                meta['sha'] = 'd' * 40
            write(root / 'snapshots/index.json', idx)
            self.prepare()
            self.assertEqual({t['id'] for t in self.s.tasks()}, before)
            self.assertIsNone(self.s.claim())

    def test_revision_only_update_preserves_partial_mapping(self):
        self.prepare()
        c = self.s.claim()
        R.complete(self.w, self.s, c, fake_response(c['payload']))
        before = {t['id'] for t in self.s.tasks()}
        root = self.w.model('fixture'); idx = read(root / 'snapshots/index.json')
        for meta in idx['repos'].values():
            meta['sha'] = 'b' * 40
        write(root / 'snapshots/index.json', idx)
        self.prepare()
        self.assertTrue(before <= {t['id'] for t in self.s.tasks()})
        self.assertEqual(next(t for t in self.s.tasks() if t['id'] == c['id'])['state'], 'succeeded')

    def test_crash_after_sample_before_enqueue_then_revision_change(self):
        self.finish()
        sample = read(self.w.path / 'review-sample.json')
        self.assertTrue(sample['items'])
        with self.s.connect() as db:
            db.execute("DELETE FROM tasks WHERE stage='review'")
        self.assertEqual(set(R.status(self.w, self.s)['sample_blockers']), set(sample['items']))
        root = self.w.model('fixture'); idx = read(root / 'snapshots/index.json')
        for meta in idx['repos'].values():
            meta['sha'] = 'e' * 40
        write(root / 'snapshots/index.json', idx)
        self.prepare()
        review = [t for t in self.s.tasks() if t['stage'] == 'review']
        self.assertEqual({t['payload']['fingerprint'] for t in review}, set(sample['items']))
        self.assertTrue(all(t['state'] == 'pending' for t in review))
        with self.assertRaises(ValueError):
            R.verify(self.w, self.s)

    def test_loaded_config_cannot_escape_or_verify_incomplete_ingestion(self):
        config = read(self.w.path / 'runner.json'); config['models'] = [str(CODE / 'models/qwen3.8-27b')]
        write(self.w.path / 'runner.json', config)
        with self.assertRaises(ValueError):
            Workspace(self.w.path)
        new = Workspace.create_ingest(self.base / 'ingest', 'new-model', 'Publisher/New-Model')
        self.assertEqual(R.plan(new)['models']['new-model']['collection'], 'required or blocked')
        with self.assertRaises(ValueError):
            R.verify(new, State(new.path / 'state.sqlite'))

    def test_plan_is_nonmutating_and_discovers_work(self):
        before = {str(p): p.read_bytes() for p in self.w.path.rglob('*') if p.is_file()}
        planned = R.plan(self.w)
        self.assertGreater(len(planned['models']['fixture']['mapping_repos']), 0)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.w.path.rglob('*') if p.is_file()})

    def test_failed_fetch_does_not_block_other_models_or_stamp_fresh(self):
        other = self.w.path / 'models/blocked'
        write(other / 'model.json', {'model': 'Publisher/Blocked', 'slug': 'blocked'})
        write(other / 'audit/runner_fetch.json', {'status': 'blocked'})
        self.w.config['models'].insert(0, 'blocked'); write(self.w.path / 'runner.json', self.w.config)
        result = R.reconcile(self.w, self.s)
        self.assertEqual(result['fetch_blockers'], ['blocked'])
        self.assertTrue(self.s.tasks())
        self.assertTrue(all(t['model'] == 'fixture' for t in self.s.tasks()))

    def test_small_model_and_exact_decade_static_chart_regression(self):
        import ast
        import html
        import math
        tree = ast.parse((CODE / 'web' / 'build.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kld_chart')
        ns = {'math': math, 'owner': lambda r: r.split('/')[0], 'E': html.escape,
              'BASIS_LABEL': {}, 'REF_TXT': {'bf16': 'vs BF16'}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<chart-regression>', 'exec'), ns)
        row = {'kind': 'own', 'size_gb': 4, 'metrics': {'mean_kld': {'v': 0.1, 'raw': '0.1'}},
               'measured_by': 'Publisher/Model', 'variant': 'Q4', 'size_raw': '4 GB', 'basis': 'disk', 'ref_class': 'bf16'}
        chart, _ = ns['kld_chart']([row])
        self.assertIn('<circle', chart)
        self.assertNotIn('nan', chart)

    def test_orchestration_never_scans_the_whole_journal_per_task(self):
        """Completion and enqueue look tasks up by id/slot; a reconcile loads bounded listings."""
        with patch.object(State, 'tasks', autospec=True, side_effect=State.tasks) as listing:
            self.prepare()
            per_reconcile = listing.call_count
            self.assertLessEqual(per_reconcile, 12)
            self.assertTrue(all(call.kwargs.get('with_payload', True) is False or call.kwargs.get('states') or call.kwargs.get('model') or call.kwargs.get('stage')
                                for call in listing.call_args_list), 'every listing is filtered or payload-free')
            requests = self.w.path / 'requests'
            before = {p.name: p.stat().st_mtime_ns for p in requests.glob('*.json')}
            self.assertTrue(before)
            self.prepare()  # unchanged inputs: exported requests are not rewritten
            self.assertEqual({p.name: p.stat().st_mtime_ns for p in requests.glob('*.json')}, before)
            listing.reset_mock()
            n = 0
            while c := self.s.claim(limit=3):
                R.complete(self.w, self.s, c, fake_response(c['payload'])); n += 1
            self.assertGreater(n, 3)
            self.assertEqual(listing.call_count, 0)
        self.prepare()
        done = {t['id'] for t in self.s.tasks(states=('succeeded',), with_payload=False)}
        self.assertFalse(done & {p.stem for p in requests.glob('*.json')}, 'completed requests are withdrawn')

    def test_model_identity_change_rejects_inflight_result(self):
        self.prepare()
        claim = self.s.claim()
        config = self.w.model('fixture') / 'model.json'
        model = read(config); model['model'] = 'Other/Identity'; write(config, model)
        with self.assertRaisesRegex(ValueError, 'model identity'):
            R.complete(self.w, self.s, claim, fake_response(claim['payload']))
        self.assertEqual(self.s.get(claim['id'])['state'], 'running')
        self.prepare()
        self.assertFalse(self.s.get(claim['id'])['active'])
        self.assertTrue(all(t['payload']['model'] == 'Other/Identity' for t in self.s.tasks()))

    def test_review_preparation_snapshots_version_once_but_not_across_calls(self):
        self.finish()
        with patch.object(R, 'version', return_value='v1') as version:
            old = R.review_bundles(self.w.model('fixture'))
        self.assertTrue(old)
        self.assertEqual(version.call_count, 1)
        with patch.object(R, 'version', return_value='v2'):
            new = R.review_bundles(self.w.model('fixture'))
        self.assertEqual(len(old), len(new))
        self.assertTrue(all(a['fingerprint'] != b['fingerprint'] for a, b in zip(old, new)))

    def test_corrupt_export_is_repaired_from_durable_request(self):
        self.prepare()
        request = next((self.w.path / 'requests').glob('*.json'))
        expected = request.read_bytes()
        request.write_text('{"task_id":"wrong", "request":{"source":"altered"}}')
        R.export_requests(self.w, self.s)
        self.assertEqual(request.read_bytes(), expected)
        before = request.stat().st_mtime_ns
        R.export_requests(self.w, self.s)
        self.assertEqual(request.stat().st_mtime_ns, before)

    def test_content_identity_matches_prior_definition_and_shares_evidence(self):
        import json
        self.prepare()
        bundles = R.review_bundles(self.w.model('fixture')) or [t['payload'] for t in self.s.tasks()]
        for b in bundles:
            prior = json.loads(json.dumps(b)); prior.pop('fingerprint', None); prior['source'].pop('revision', None)
            self.assertEqual(R.content_identity(b), R.digest(prior))
            self.assertIn('revision', b['source'])
        evidence = [t['payload']['source'] for t in self.s.tasks() if t['stage'] == 'map-A']
        self.assertTrue(all('text' in e and e['lines'][0]['L'] == 1 for e in evidence))

    def test_gated_build_receipt_is_not_publication_receipt(self):
        self.finish()
        receipt = R.verify(self.w, self.s)
        self.assertFalse(receipt['publication_authorized'])
        self.assertTrue(receipt['artifacts']['fixture'])
        self.assertFalse((self.w.path / 'audit/deployed.json').exists())


if __name__ == '__main__':
    unittest.main()

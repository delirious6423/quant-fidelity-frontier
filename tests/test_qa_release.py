"""QA release receipts and review gates, with no real git, gh or worker runs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import deploy_state as state
from pipeline import qcl_qa as qa  # Import-safe; main is never invoked by this suite.

OLD, TARGET, ADVANCED = 'a' * 40, 'b' * 40, 'c' * 40


def completed(code=0, stdout='', stderr=''):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def run(**overrides):
    value = dict(id=31, run_attempt=1, path=state.WORKFLOW, event='workflow_dispatch',
                 head_sha=TARGET, display_title='deploy-pages ' + TARGET,
                 status='completed', conclusion='success',
                 html_url='https://github.com/delirious6423/quant-fidelity-frontier/actions/runs/31')
    value.update(overrides)
    return value


class ReviewGate(unittest.TestCase):
    def setUp(self):
        self.items = [dict(id=f'q{i}', slug='model', repo='publisher/model', entry=f'row{i}', key='kld') for i in range(2)]
        self.verdicts = [dict(id=f'q{i}', verdict='ok', reason='Matches the metric and row at [L12].') for i in range(2)]

    def assess(self, batches=None, items=None, count=2):
        return state.assess_reviews(self.items if items is None else items, count,
                                    [self.verdicts] if batches is None else batches)

    def test_complete_decisive_review_passes(self):
        result = self.assess()
        self.assertTrue(result['ready'])
        self.assertEqual(result['counts'], dict(ok=2, wrong=0, unsure=0, missing=0))

    def test_missing_review_or_missing_item_blocks(self):
        for batches in ([], [[]], [[self.verdicts[0]]]):
            with self.subTest(batches=batches):
                result = self.assess(batches)
                self.assertFalse(result['ready'])
                self.assertIn('q1', result['unresolved'])

    def test_unsure_review_holds_that_value_without_blocking(self):
        reports = copy.deepcopy(self.verdicts)
        reports[1].update(verdict='unsure', reason='Ambiguous header [L9].')
        result = self.assess([reports])
        self.assertTrue(result['ready'])
        self.assertEqual([i['id'] for i, _ in result['held']], ['q1'])
        self.assertTrue(result['held'][0][1].startswith('reviewer unsure'))
        self.assertEqual(result['counts']['unsure'], 1)

    def test_invalid_or_unexplained_review_blocks(self):
        for change in ({'verdict': 'approved'}, {'reason': ''}, {'reason': None}):
            reports = copy.deepcopy(self.verdicts)
            reports[1].update(change)
            result = self.assess([reports])
            self.assertFalse(result['ready'], change)
            self.assertIn('q1', result['unresolved'])

    def test_duplicate_or_conflicting_review_never_last_write_wins(self):
        for verdict in ('ok', 'wrong'):
            duplicate = dict(self.verdicts[0], verdict=verdict)
            self.assertFalse(self.assess([self.verdicts, [duplicate]])['ready'])

    def test_invalid_payload_unknown_id_and_missing_sample_block(self):
        for batches in ([None], [{'q0': 'ok'}], [self.verdicts, [dict(id='other', verdict='ok', reason='[L1]')]]):
            self.assertFalse(self.assess(batches)['ready'])
        self.assertFalse(state.assess_reviews(None, 2, [self.verdicts])['ready'])
        self.assertFalse(self.assess(items=[], count=2)['ready'])
        self.assertFalse(self.assess(items=[self.items[0], self.items[0]])['ready'])
        self.assertFalse(self.assess(count=True)['ready'])

    def test_wrong_verdicts_return_exact_entries_for_quarantine(self):
        reports = copy.deepcopy(self.verdicts)
        reports[1].update(verdict='wrong', reason='The value belongs to another row [L12].')
        result = self.assess([reports])
        self.assertTrue(result['ready'])
        self.assertEqual(result['counts']['wrong'], 1)
        self.assertEqual(result['held'], [(self.items[1], reports[1]['reason'])])

    def test_zero_sample_requires_explicit_zero_state(self):
        self.assertTrue(state.assess_reviews([], 0, [])['ready'])
        self.assertFalse(state.assess_reviews([], None, [])['ready'])


class ReleaseReceipt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pending_path = Path(self.tmp.name, 'audit/pending_release.json')
        self.deployed_path = Path(self.tmp.name, 'audit/deployed.json')
        state.write_json(self.deployed_path, dict(commit=OLD, at='2026-10-01T00:00:00Z'))
        self.old_bytes = self.deployed_path.read_bytes()
        self.review = dict(reviewed=2, verdicts=dict(ok=1, wrong=1, unsure=0, missing=0), held=1, promoted=['model'])
        self.pending = state.new_pending(self.review, '2026-10-19T00:00:00Z')
        self.pending.update(target_commit=TARGET, status='dispatched')
        state.write_json(self.pending_path, self.pending)

    def reconcile(self, *runs, **kwargs):
        return state.reconcile(self.pending_path, self.deployed_path, {'workflow_runs': list(runs)}, **kwargs)

    def assert_not_deployed(self):
        self.assertEqual(self.deployed_path.read_bytes(), self.old_bytes)

    def test_preparing_release_does_not_write_deployed_or_guess_sha(self):
        pending = state.new_pending(self.review)
        self.assertIsNone(pending['target_commit'])
        state.write_json(self.pending_path, pending)
        self.assert_not_deployed()

    def test_fresh_clone_recovers_exact_introducing_commit(self):
        pending = state.new_pending(self.review)
        state.write_json(self.pending_path, pending)
        clone_pending = json.loads(self.pending_path.read_text())
        self.assertEqual(state.resolve_target(clone_pending, TARGET)['target_commit'], TARGET)
        self.assertIsNone(clone_pending['target_commit'])
        self.reconcile(run(), introducing_commit=TARGET)
        self.assertEqual(state.read_json(self.deployed_path)['commit'], TARGET)

    def test_only_successful_exact_workflow_advances_receipt(self):
        result = self.reconcile(run())
        self.assertEqual(result['outcome'], 'succeeded')
        deployed = state.read_json(self.deployed_path)
        self.assertEqual(deployed['commit'], TARGET)
        self.assertEqual(deployed['run_id'], 31)
        self.assertEqual(deployed['verdicts'], self.review['verdicts'])
        self.assertEqual(state.read_json(self.pending_path)['status'], 'succeeded')

    def test_queued_running_and_missing_run_preserve_deployed(self):
        for status in ('queued', 'in_progress', 'waiting'):
            with self.subTest(status=status):
                self.assertEqual(self.reconcile(run(status=status, conclusion=None))['outcome'], 'pending')
                self.assert_not_deployed()
        self.assertEqual(self.reconcile()['outcome'], 'pending')
        self.assert_not_deployed()

    def test_every_unsuccessful_conclusion_preserves_deployed(self):
        for conclusion in ('failure', 'cancelled', 'timed_out', 'skipped', 'action_required', None):
            with self.subTest(conclusion=conclusion):
                self.assertEqual(self.reconcile(run(conclusion=conclusion))['outcome'], 'failed')
                self.assert_not_deployed()

    def test_unrelated_workflow_event_or_sha_cannot_authorize_release(self):
        for change in ({'path': '.github/workflows/verify.yml'}, {'event': 'push'},
                       {'head_sha': ADVANCED, 'display_title': 'another run'}):
            self.assertEqual(self.reconcile(run(**change))['outcome'], 'pending')
            self.assert_not_deployed()

    def test_expected_sha_title_identifies_branch_race_as_failed(self):
        self.assertEqual(self.reconcile(run(head_sha=ADVANCED))['outcome'], 'failed')
        self.assert_not_deployed()

    def test_latest_failure_supersedes_an_older_success(self):
        result = self.reconcile(run(id=31), run(id=32, conclusion='failure'))
        self.assertEqual(result['outcome'], 'failed')
        self.assert_not_deployed()

    def test_latest_rerun_attempt_is_required(self):
        result = self.reconcile(run(run_attempt=1), run(run_attempt=2, status='in_progress', conclusion=None))
        self.assertEqual(result['outcome'], 'pending')
        self.assert_not_deployed()

    def test_paginated_run_results_preserve_latest_attempt(self):
        payload = [{'workflow_runs': [run(id=31)]}, {'workflow_runs': [run(id=32, conclusion='failure')]}]
        self.assertEqual(state.reconcile(self.pending_path, self.deployed_path, payload)['outcome'], 'failed')
        self.assert_not_deployed()

    def test_malformed_api_and_invalid_identity_fail_closed(self):
        for payload in (None, {}, {'workflow_runs': None}, [{'workflow_runs': []}, None],
                        {'workflow_runs': [run(id=None)]}, {'workflow_runs': [run(run_attempt=0)]}):
            with self.subTest(payload=payload):
                before = self.pending_path.read_bytes()
                self.assertEqual(state.reconcile(self.pending_path, self.deployed_path, payload)['outcome'], 'unknown')
                self.assert_not_deployed()
                self.assertEqual(self.pending_path.read_bytes(), before)

    def test_blank_or_malformed_sha_never_falls_back(self):
        for target in ('', 'bad', 123, 'b' * 39):
            invalid = dict(self.pending, target_commit=target)
            with self.assertRaises(ValueError): state.resolve_target(invalid, TARGET)
        with self.assertRaises(ValueError): state.resolve_target(dict(self.pending, target_commit=None), '')
        with self.assertRaises(ValueError): state.resolve_target(dict(self.pending, workflow='other.yml'))


class StopWorker(Exception):
    pass


class MockedAdapters(ReleaseReceipt):
    """Call small import-safe adapters only; never execute qcl_qa.main."""
    def setUp(self):
        super().setUp()
        for key, value in [('CODE', self.tmp.name), ('PENDING', str(self.pending_path)), ('DEPLOYED', str(self.deployed_path))]:
            item = patch.object(qa, key, value)
            item.start()
            self.addCleanup(item.stop)

    def test_dispatch_failure_persists_pending_without_deployed_change(self):
        with patch.object(qa, 'command', return_value=completed(1, stderr='not permitted')) as gh:
            self.assertEqual(qa.dispatch(self.pending), 'dispatch_failed')
        self.assertIn('inputs[expected_sha]=' + TARGET, gh.call_args.args[0])
        self.assertEqual(state.read_json(self.pending_path)['status'], 'dispatch_failed')
        self.assert_not_deployed()

    def test_dispatch_success_is_still_only_pending(self):
        with patch.object(qa, 'command', return_value=completed()) as gh:
            self.assertEqual(qa.dispatch(self.pending), 'dispatched')
        self.assertEqual(gh.call_count, 1)
        self.assert_not_deployed()

    def test_reconcile_in_fresh_clone_uses_external_success(self):
        state.write_json(self.pending_path, state.new_pending(self.review))
        replies = [completed(stdout=TARGET + '\n'), completed(stdout=json.dumps([{'workflow_runs': [run()]}]))]
        with patch.object(qa, 'command', side_effect=replies) as gh, patch.object(qa, 'push', return_value='pushed') as push:
            self.assertFalse(qa.reconcile_pending())
        self.assertEqual(state.read_json(self.deployed_path)['commit'], TARGET)
        self.assertIn('--slurp', gh.call_args.args[0])
        self.assertIn('audit/deployed.json', push.call_args.args[1])

    def test_status_api_failure_never_changes_receipts(self):
        before = self.pending_path.read_bytes()
        replies = [completed(stdout=TARGET), completed(1, stderr='network unavailable')]
        with patch.object(qa, 'command', side_effect=replies), patch.object(qa, 'push') as push, patch.object(qa, 'say', side_effect=StopWorker):
            with self.assertRaises(StopWorker): qa.reconcile_pending()
        push.assert_not_called()
        self.assert_not_deployed()
        self.assertEqual(self.pending_path.read_bytes(), before)

    def test_crash_before_dispatch_can_retry_exact_candidate(self):
        state.write_json(self.pending_path, state.new_pending(self.review))
        replies = [completed(stdout=TARGET), completed(stdout='[{"workflow_runs": []}]'), completed(stdout=TARGET), completed()]
        with patch.object(qa, 'command', side_effect=replies) as gh, patch.object(qa, 'push', return_value='pushed'), patch.object(qa, 'say', side_effect=StopWorker):
            with self.assertRaises(StopWorker): qa.reconcile_pending()
        self.assertIn('inputs[expected_sha]=' + TARGET, gh.call_args.args[0])
        self.assertEqual(state.read_json(self.pending_path)['status'], 'dispatched')
        self.assert_not_deployed()

    def test_crash_before_dispatch_and_branch_advance_requires_fresh_candidate(self):
        state.write_json(self.pending_path, state.new_pending(self.review))
        replies = [completed(stdout=TARGET), completed(stdout='[{"workflow_runs": []}]'), completed(stdout=ADVANCED)]
        with patch.object(qa, 'command', side_effect=replies) as gh, patch.object(qa, 'push', return_value='pushed'), patch.object(qa, 'say', side_effect=StopWorker):
            with self.assertRaises(StopWorker): qa.reconcile_pending()
        self.assertFalse(any('POST' in call.args[0] for call in gh.call_args_list))
        self.assertEqual(state.read_json(self.pending_path)['status'], 'failed')
        self.assert_not_deployed()

    def test_failed_release_requests_a_new_qa_attempt(self):
        state.write_json(self.pending_path, dict(self.pending, status='dispatch_failed'))
        with patch.object(qa, 'command') as command:
            self.assertTrue(qa.reconcile_pending())
        command.assert_not_called()
        self.assert_not_deployed()

    def test_candidate_push_failure_never_rebases_or_retries(self):
        replies = [completed(), completed(), completed(1), completed(), completed(1)]  # clean index, stage, changed diff, commit, push rejected
        with patch.object(qa, 'command', side_effect=replies) as git:
            result = qa.push('candidate', '.', allow_rebase=False)
        self.assertIn('push failed', result)
        self.assertFalse(any('pull' in call.args[0] for call in git.call_args_list))
        self.assertEqual(sum('push' in call.args[0] for call in git.call_args_list), 1)

    def test_failed_commit_never_pushes_previous_head(self):
        replies = [completed(), completed(), completed(1), completed(1)]
        with patch.object(qa, 'command', side_effect=replies) as git:
            self.assertEqual(qa.push('candidate', '.', allow_rebase=False), 'commit failed')
        self.assertFalse(any('push' in call.args[0] for call in git.call_args_list))


class CheckoutAndChangeSafety(unittest.TestCase):
    def test_all_publication_dependencies_invalidate_the_receipt(self):
        paths = ['homepage.py', 'benchmarks.py', 'benchmark_view.py', 'comparison.py',
                 'web_payload.py', 'gates/verify.py', 'pipeline/qcl_live.py',
                 'protocol_evidence/reviewed.json', 'site.json', 'watchlist.json',
                 'models/example/model.json', 'models/example/audit/review_pins.json',
                 'models/example/audit/holds.json', 'models/example/snapshots/source.md',
                 'models/example/verified/part8_machine.py', 'future/new_renderer.py']
        with patch.object(qa, 'command', return_value=completed(stdout='\0'.join(paths) + '\0')) as git:
            self.assertEqual(qa.changed_since_deployment(OLD, TARGET), paths)
        git.assert_called_once_with(['git', 'diff', '--name-only', '-z', OLD, TARGET, '--'])

    def test_only_known_operational_changes_do_not_trigger_release(self):
        paths = sorted(state.OPERATIONAL_FILES) + ['logs/scan/run-qa.jsonl', '']
        self.assertEqual(state.publication_changes(paths), [])
        self.assertEqual(state.publication_changes(['audit/source.json', 'logs/source.json']),
                         ['audit/source.json', 'logs/source.json'])

    def test_failed_diff_never_means_no_changes(self):
        with patch.object(qa, 'command', return_value=completed(128, stderr='fatal: bad object')):
            with self.assertRaisesRegex(ValueError, 'git fetch --unshallow origin'):
                qa.changed_since_deployment(OLD, TARGET)

    def test_invalid_commit_never_reaches_git_or_shell(self):
        for base, head in [('$(printf unsafe)', TARGET), ('', TARGET), (123, TARGET), (OLD, ''), (OLD, None)]:
            with self.subTest(base=base, head=head), patch.object(qa, 'command') as git:
                with self.assertRaises(ValueError): qa.changed_since_deployment(base, head)
                git.assert_not_called()
        with patch.object(qa, 'command') as git:
            self.assertEqual(qa.changed_since_deployment(None, TARGET), ['(first run)'])
            git.assert_not_called()

    def test_failed_comparison_stops_main_before_taking_a_lock(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(qa.os.environ, {}):
            deployed = Path(directory, 'deployed.json')
            state.write_json(deployed, {'commit': OLD})
            replies = [completed(), completed(stdout=TARGET + '\n'), completed(128, stderr='fatal: bad object')]
            with patch.object(qa, 'ST', str(Path(directory, 'missing.json'))), patch.object(qa, 'DEPLOYED', str(deployed)), \
                 patch.object(qa, 'command', side_effect=replies), patch.object(qa, 'reconcile_pending', return_value=False), \
                 patch.object(qa, 'scan') as scan, patch.object(qa, 'push') as push, \
                 patch.object(qa, 'say', side_effect=StopWorker) as say:
                with self.assertRaises(StopWorker): qa.main(['--synced'])
            report = say.call_args.args[0]['report']
            self.assertIn('git fetch --unshallow origin', report['error'])
            self.assertNotIn('skipped', report)
            scan.assert_not_called()
            push.assert_not_called()

    def test_lock_push_preserves_preexisting_staged_changes(self):
        for result in (completed(1), completed(128)):
            with self.subTest(exit_code=result.returncode), patch.object(qa, 'command', return_value=result) as git:
                status = qa.push('QA lock', 'qa.lock')
                self.assertNotIn(status, ('pushed', 'nothing to commit'))
                git.assert_called_once_with(['git', 'diff', '--cached', '--quiet'])

    def test_dirty_checkout_blocks_before_fetch_or_checkout(self):
        for status in (' M build.py\n', 'M  README.md\n', '?? personal-note.txt\n'):
            with self.subTest(status=status), patch.object(qa, 'command', return_value=completed(stdout=status)) as git:
                with self.assertRaisesRegex(ValueError, 'Commit or stash'):
                    qa.synchronize()
                git.assert_called_once_with(['git', 'status', '--porcelain=v1', '--untracked-files=normal'])

    def test_unpublished_commits_are_never_overwritten_by_sync(self):
        replies = [completed(), completed(), completed(1)]
        with patch.object(qa, 'command', side_effect=replies) as git:
            with self.assertRaisesRegex(ValueError, 'Preserve this checkout'):
                qa.synchronize()
        self.assertFalse(any('checkout' in call.args[0] for call in git.call_args_list))

    def test_checkout_failure_never_reexecutes_or_starts_qa(self):
        replies = [completed(), completed(), completed(), completed(1)]
        with patch.object(qa, 'command', side_effect=replies) as git:
            with self.assertRaisesRegex(ValueError, 'check out synchronized main'):
                qa.synchronize()
        self.assertFalse(any('config' in call.args[0] for call in git.call_args_list))


if __name__ == '__main__':
    unittest.main()

"""Fail-closed QA review and release receipts; no subprocess or network calls.

A pending receipt is committed with the candidate. Its initial target is null:
the commit that last changed that file supplies the exact SHA after pushing (or
after a crash and fresh clone). Later receipt commits store that SHA explicitly.
Only a successful, matching Pages workflow can advance the deployed receipt.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

WORKFLOW = '.github/workflows/pages.yml'
SHA = re.compile(r'^[0-9a-f]{40}$')
TERMINAL = {'succeeded', 'failed', 'dispatch_failed', 'push_failed'}
OPERATIONAL_FILES = {'qa.lock', 'scan.lock', 'ingest.lock', 'PROGRESS.md', 'PLAN.md',
                     'audit/deployed.json', 'audit/pending_release.json'}


def publication_changes(paths):
    """Conservatively invalidate a release for every non-operational change.

    A small exclusion list avoids forgetting new renderers, gates, source pins,
    model configurations or published metadata as the application grows.
    """
    return [path for path in paths if path and path not in OPERATIONAL_FILES
            and not path.startswith('logs/scan/')]


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def write_json(path, value):
    """Atomic receipts cannot become partial JSON on an interrupted write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, indent=1)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assess_reviews(items, expected_count, batches):
    """Every sampled item needs exactly one decisive, explained verdict."""
    counts = dict(ok=0, wrong=0, unsure=0, missing=0)
    issues, held, by = [], [], {}
    if not isinstance(items, list):
        items = []
        issues.append('The required review sample is missing or invalid.')
    if type(expected_count) is not int or expected_count < 0 or expected_count != len(items):
        issues.append('The review sample count does not match the saved QA state.')
    for item in items:
        id_ = item.get('id') if isinstance(item, dict) else None
        if not isinstance(id_, str) or not id_ or id_ in by:
            issues.append('Review sample IDs must be nonempty and unique.')
        else:
            by[id_] = item
    received = {}
    for batch in batches:
        if not isinstance(batch, list):
            issues.append('A verdict file is missing, malformed, or not a list.')
            continue
        for verdict in batch:
            id_ = verdict.get('id') if isinstance(verdict, dict) else None
            if not isinstance(id_, str) or id_ not in by:
                issues.append('A verdict does not identify a required sample item.')
                continue
            received.setdefault(id_, []).append(verdict)
    unresolved = []
    for id_, item in by.items():
        reports = received.get(id_, [])
        if not reports:
            counts['missing'] += 1
            unresolved.append(id_)
            continue
        if len(reports) != 1:
            issues.append(f'{id_}: duplicate verdicts require resolution.')
            counts['unsure'] += 1
            unresolved.append(id_)
            continue
        report = reports[0]
        verdict, reason = report.get('verdict'), report.get('reason')
        if verdict not in ('ok', 'wrong', 'unsure') or not isinstance(reason, str) or not reason.strip():
            issues.append(f'{id_}: verdict or source explanation is invalid.')
            verdict = 'unsure'
        counts[verdict] += 1
        if verdict == 'wrong':
            held.append((item, reason.strip()[:300]))
        elif verdict == 'unsure':
            unresolved.append(id_)
    return dict(ready=not issues and not unresolved, counts=counts,
                held=held, unresolved=unresolved, issues=issues)


def new_pending(review, now=None):
    return dict(version=1, release_id=uuid.uuid4().hex, workflow=WORKFLOW,
                target_commit=None, status='prepared', prepared_at=now or utc_now(),
                review={k: copy.deepcopy(review.get(k)) for k in ('reviewed', 'verdicts', 'held', 'promoted')})


def resolve_target(pending, introducing_commit=None):
    if not isinstance(pending, dict) or pending.get('version') != 1 or pending.get('workflow') != WORKFLOW:
        raise ValueError('Pending release receipt has an unsupported schema or workflow.')
    # An explicitly malformed SHA must not fall back to another commit.
    target = introducing_commit if pending.get('target_commit') is None else pending.get('target_commit')
    if not isinstance(target, str) or not SHA.fullmatch(target):
        raise ValueError('The exact pending release commit could not be established.')
    result = copy.deepcopy(pending)
    result['target_commit'] = target
    return result


def workflow_result(pending, payload):
    """Classify latest matching attempt, including a guarded branch-race run.

The caller retrieves only pages.yml runs from the configured repository. The
workflow run name carries expected_sha so a run on an advanced branch is
recognized as failed rather than being mistaken for a still-queued request.
"""
    target = resolve_target(pending)['target_commit']
    pages = payload if isinstance(payload, list) else [payload]
    runs = []
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get('workflow_runs'), list):
            return dict(outcome='unknown', reason='Workflow API response is incomplete or malformed.')
        runs.extend(page['workflow_runs'])
    matches = [r for r in runs if isinstance(r, dict) and r.get('path') == WORKFLOW
               and r.get('event') == 'workflow_dispatch'
               and (r.get('head_sha') == target or r.get('display_title') == 'deploy-pages ' + target)]
    if not matches:
        return dict(outcome='pending', reason='No matching Pages workflow run is visible yet.')
    if any(type(r.get('id')) is not int or r['id'] <= 0 or type(r.get('run_attempt', 1)) is not int or r.get('run_attempt', 1) < 1 for r in matches):
        return dict(outcome='unknown', reason='Matching workflow run identity is malformed.')
    latest = max(matches, key=lambda r: (r['id'], r.get('run_attempt', 1)))
    proof = {k: latest.get(k) for k in ('id', 'run_attempt', 'head_sha', 'status', 'conclusion', 'html_url')}
    if latest.get('head_sha') != target:
        return dict(outcome='failed', reason='The branch advanced before dispatch; expected SHA guard rejects this run.', run=proof)
    if latest.get('status') != 'completed':
        return dict(outcome='pending', reason='The matching Pages workflow has not completed.', run=proof)
    if latest.get('conclusion') != 'success':
        return dict(outcome='failed', reason='The matching Pages workflow did not succeed.', run=proof)
    return dict(outcome='succeeded', reason='The exact candidate completed the Pages workflow successfully.', run=proof)


def reconcile(pending_path, deployed_path, payload, introducing_commit=None, now=None):
    """Advance deployed only with successful external proof, never on dispatch."""
    pending = resolve_target(read_json(pending_path), introducing_commit)
    result = workflow_result(pending, payload)
    if result['outcome'] == 'unknown':
        return result
    if result['outcome'] == 'succeeded':
        run = result['run']
        deployed = {**pending.get('review', {}), 'commit': pending['target_commit'],
                    'at': now or utc_now(), 'workflow': WORKFLOW,
                    'run_id': run['id'], 'run_attempt': run.get('run_attempt') or 1,
                    'run_url': run.get('html_url')}
        write_json(deployed_path, deployed)
    pending['status'] = result['outcome']
    pending['last_result'] = result
    write_json(pending_path, pending)
    return result

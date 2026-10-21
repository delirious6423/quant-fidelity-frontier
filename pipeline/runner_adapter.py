#!/usr/bin/env python3
"""Live interpretation adapter for the shadow runner: one Claude request per journal task.

    python3 pipeline/runner_adapter.py --workspace /path/to/shadow --model claude-opus-5-5 \
        --workers 4 --max-usd 5 --responses /path/to/responses

Each claimed task becomes exactly one Messages API call whose only input is that
task's bundle (docs/SCRIPTED_RUNNER.md "Model request contract"): a mapping task
sends the generated mapping prompt plus the complete numbered source; a review
task sends the claim, its cited lines and the complete source, with the verdict
schema of docs/agents/REVIEW.md. The reply must be one JSON object; it is parsed
strictly, validated by qcl_runner.complete (source pins, candidate ids, schema)
and persisted in the journal before the lease is released. Validated responses
are also written to --responses, the same contract as the file adapter, so a
run can be replayed offline.

Every call's usage (input, cache, output tokens, request id, model) is recorded
as a journal event and summed against --max-usd; the adapter stops claiming
once the next call could exceed it. Rate limits and server errors back the
task off through the journal's bounded retry; malformed or refused replies
block the task with a reason. Credentials come from the SDK's normal resolution
(ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile); the
adapter never prints them. Nothing here can publish, merge or touch production.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qcl_runner as runner
import runner_model as boundary
from runner_state import State, encoded
from runner_workspace import Workspace, write

# USD per million tokens: (input, cache write, cache read, output). Override with --price when a model is missing.
PRICES = {
    'claude-opus-5-5': (4.0, 5.0, 0.20, 20.0),
    'claude-sonnet-5-5': (2.0, 2.5, 0.20, 10.0),
    'claude-haiku-5-5': (0.10, 0.125, 0.01, 0.50),
}
REVIEW_SCHEMA = ('Answer with ONE JSON object, no prose: {"id": "<the evidence fingerprint given above>", '
                 '"verdict": "ok" | "wrong" | "unsure", "reason": "<one short sentence citing [Lnn]>"}. '
                 'ok: the raw text is on the cited line and the card says it is this metric, for this row, in this role '
                 '(own = this repo\'s file; ladder = the publisher\'s other builds; cross = another publisher\'s file the '
                 'author measured; quoted = copied from elsewhere). wrong: anything else, for example a count published '
                 'as a score, a row that is not a quant, a different metric or model, or a dropped qualifier such as ~ or '
                 'a range. unsure: the card is ambiguous.')


def numbered(source):
    return '\n'.join(f"{line['L']}: {line['text']}" for line in source['lines'])


def messages_for(bundle):
    """The complete, self-contained request for one task. Source text is quoted data, never instructions."""
    if bundle['kind'] == 'mapping':
        text = (bundle['prompt'] + '\n\n## Complete source (line: text), the exact bytes pinned above\n'
                + numbered(bundle['source']))
    else:
        claim = bundle['claim']
        cited = '\n'.join(f"{n}: {bundle['source']['lines'][n - 1]['text']}" for n in claim['lines'])
        text = ('## Value to review\n' + json.dumps({k: claim[k] for k in ('repo', 'kind', 'entry', 'key', 'raw', 'lines')})
                + '\nEvidence fingerprint (copy into "id"): ' + bundle['fingerprint']
                + '\nEntry as published: ' + json.dumps(bundle['claim'].get('entry_evidence'), sort_keys=True)
                + '\n\n## Cited lines\n' + cited + '\n\n## Complete source (line: text)\n' + numbered(bundle['source'])
                + '\n\n## Output\n' + REVIEW_SCHEMA)
    return bundle['instructions'], [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]


def extract_json(text):
    """The reply must be one JSON object; tolerate a fenced block around it, nothing else."""
    text = text.strip()
    match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
    if match:
        text = match.group(1)
    return boundary.parse_response(text.encode('utf-8'))


def cost_usd(usage, price):
    pin, pwrite, pread, pout = price
    return (usage.get('input_tokens', 0) * pin + usage.get('cache_creation_input_tokens', 0) * pwrite
            + usage.get('cache_read_input_tokens', 0) * pread + usage.get('output_tokens', 0) * pout) / 1e6


def usage_dict(message):
    u = message.usage
    return {'input_tokens': u.input_tokens or 0, 'output_tokens': u.output_tokens or 0,
            'cache_creation_input_tokens': getattr(u, 'cache_creation_input_tokens', 0) or 0,
            'cache_read_input_tokens': getattr(u, 'cache_read_input_tokens', 0) or 0}


class Adapter:
    def __init__(self, workspace, state, *, client, model, effort='high', max_tokens=16000, price=None,
                 max_usd=None, responses=None, fallbacks=True):
        self.w, self.state, self.client, self.model = workspace, state, client, model
        self.effort, self.max_tokens, self.fallbacks = effort, max_tokens, fallbacks
        self.price = price or PRICES.get(model)
        if self.price is None:
            raise ValueError('no price table for this model; pass --price in,cache_write,cache_read,out')
        self.max_usd, self.responses = max_usd, Path(responses) if responses else None
        self.spent, self.calls, self.lock = self.spent_so_far(), 0, threading.Lock()

    def spent_so_far(self):
        with self.state.connect() as db:
            rows = db.execute("SELECT reason FROM events WHERE event='usage'").fetchall()
        return sum(json.loads(r['reason']).get('usd', 0.0) for r in rows)

    def budget_left(self):
        return self.max_usd is None or self.spent < self.max_usd

    def call(self, bundle):
        system, messages = messages_for(bundle)
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages,
                      output_config={'effort': self.effort})
        if self.fallbacks:
            return self.client.beta.messages.create(betas=['server-side-fallback-2026-07-01'], fallbacks='default', **kwargs)
        return self.client.messages.create(**kwargs)

    def record(self, task_id, message, usd, outcome):
        with self.state.connect() as db:
            self.state.event(db, task_id, 'usage', encoded({'model': getattr(message, 'model', self.model), 'usd': round(usd, 6),
                                                            'request_id': getattr(message, '_request_id', None),
                                                            'stop_reason': getattr(message, 'stop_reason', None),
                                                            'outcome': outcome, **usage_dict(message)}))

    def one(self, claim):
        import anthropic
        task_id, bundle = claim['id'], claim['payload']
        try:
            message = self.call(bundle)
        except anthropic.RateLimitError as error:
            self.state.fail(task_id, claim['token'], 'rate limited: ' + str(getattr(error, 'status_code', 429)))
            return 'rate_limited'
        except anthropic.APIStatusError as error:
            blocked = error.status_code < 500 and error.status_code != 408  # a 4xx other than throttling is not transient
            self.state.fail(task_id, claim['token'], f'api error {error.status_code}', blocked=blocked)
            return 'api_error'
        except anthropic.APIConnectionError:
            self.state.fail(task_id, claim['token'], 'connection error')
            return 'connection_error'
        usd = cost_usd(usage_dict(message), self.price)
        with self.lock:
            self.spent += usd; self.calls += 1
        if message.stop_reason == 'refusal':
            self.record(task_id, message, usd, 'refusal')
            self.state.fail(task_id, claim['token'], 'model declined the request', blocked=True)
            return 'refusal'
        if message.stop_reason == 'max_tokens':
            self.record(task_id, message, usd, 'truncated')
            self.state.fail(task_id, claim['token'], 'reply truncated at max_tokens', blocked=True)
            return 'truncated'
        text = ''.join(block.text for block in message.content if block.type == 'text')
        try:
            response = extract_json(text)
            with self.lock:  # API calls run in parallel; validation and installation take the orchestrator lock one at a time
                runner.complete(self.w, self.state, claim, response)
        except (ValueError, TypeError, KeyError) as error:
            self.record(task_id, message, usd, 'invalid: ' + str(error)[:200])
            self.state.fail(task_id, claim['token'], 'invalid reply: ' + str(error)[:300], blocked=True)
            return 'invalid'
        self.record(task_id, message, usd, 'persisted')
        if self.responses:
            write(self.responses / (task_id + '.json'), response)
        return 'persisted'

    def run(self, *, workers=4, lease_seconds=900, max_tasks=None):
        """Claim and answer until no eligible task is left, the task cap or the budget is reached."""
        from concurrent.futures import ThreadPoolExecutor
        outcomes, done = {}, 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while self.budget_left() and (max_tasks is None or done < max_tasks):
                claims = []
                while len(claims) < workers and (max_tasks is None or done + len(claims) < max_tasks):
                    claim = self.state.claim(lease_seconds=lease_seconds, limit=workers)
                    if claim is None:
                        break
                    claims.append(claim)
                if not claims:
                    break
                for outcome in pool.map(self.one, claims):
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
                done += len(claims)
        return {'tasks': done, 'outcomes': outcomes, 'calls': self.calls, 'spent_usd': round(self.spent, 4),
                'max_usd': self.max_usd, 'model': self.model}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workspace', required=True)
    ap.add_argument('--model', default='claude-opus-5-5')
    ap.add_argument('--effort', default='high', choices=('low', 'medium', 'high', 'xhigh', 'max'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lease', type=int, default=900)
    ap.add_argument('--max-tasks', type=int)
    ap.add_argument('--max-usd', type=float, help='stop claiming once recorded spend reaches this amount')
    ap.add_argument('--price', help='USD per million tokens as in,cache_write,cache_read,out')
    ap.add_argument('--responses', help='also save each validated response as <task-id>.json (file-adapter contract)')
    ap.add_argument('--no-fallbacks', action='store_true', help='do not opt into server-side refusal fallbacks')
    ap.add_argument('--max-tokens', type=int, default=16000)
    a = ap.parse_args()
    if not 1 <= a.workers <= 32:
        raise ValueError('workers must be 1..32')
    import anthropic
    price = tuple(float(x) for x in a.price.split(',')) if a.price else None
    if price is not None and len(price) != 4:
        raise ValueError('--price takes four numbers')
    w = Workspace(a.workspace)
    state = State(w.path / 'state.sqlite')
    runner.reconcile(w, state)  # export eligible work first; nothing is claimed by a reconcile
    adapter = Adapter(w, state, client=anthropic.Anthropic(), model=a.model, effort=a.effort, max_tokens=a.max_tokens,
                      price=price, max_usd=a.max_usd, responses=a.responses, fallbacks=not a.no_fallbacks)
    result = adapter.run(workers=a.workers, lease_seconds=a.lease, max_tasks=a.max_tasks)
    result['runner'] = runner.reconcile(w, state)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        raise SystemExit(1)

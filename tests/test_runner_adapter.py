"""The live adapter against a local fake Messages endpoint: one call per task, strict validation, usage and budget."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
import qcl_runner as R
from runner_replay import fixture, fake_response
from runner_state import State

try:
    import anthropic
except ImportError:  # the adapter is optional; the rest of the runner never imports it
    anthropic = None


class FakeMessages(BaseHTTPRequestHandler):
    """Replies with the fixture adapter's answer for the task named in the prompt; records every request body."""
    requests = []
    mode = {'kind': 'ok'}

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeMessages.requests.append((self.path, dict(self.headers), body))
        mode = FakeMessages.mode['kind']
        if mode == '429':
            self.send_response(429); self.send_header('Content-Type', 'application/json'); self.send_header('retry-after', '1')
            self.end_headers(); self.wfile.write(b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}'); return
        text = body['messages'][0]['content'][0]['text']
        bundle = FakeMessages.bundles[text.split('\n', 1)[0]] if mode != 'garbage' else None
        reply = {'garbage': 'not json {', 'refusal': ''}.get(mode)
        if reply is None:
            reply = json.dumps(fake_response(bundle))
            if mode == 'fenced':
                reply = '```json\n' + reply + '\n```'
            if mode == 'bad_candidate':
                answer = fake_response(bundle); answer['entries'][0]['metrics']['mean_kld'] = 'c999'; reply = json.dumps(answer)
        message = {'id': 'msg_test', 'type': 'message', 'role': 'assistant', 'model': body['model'],
                   'content': [{'type': 'text', 'text': reply}],
                   'stop_reason': 'refusal' if mode == 'refusal' else 'end_turn', 'stop_sequence': None,
                   'usage': {'input_tokens': 1000, 'output_tokens': 50, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}}
        if mode == 'refusal':
            message['stop_details'] = {'type': 'refusal', 'category': 'general_harms', 'explanation': 'test'}
        out = json.dumps(message).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(out)))
        self.send_header('request-id', 'req_test'); self.end_headers(); self.wfile.write(out)


@unittest.skipIf(anthropic is None, 'anthropic SDK not installed')
class Adapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(('127.0.0.1', 0), FakeMessages)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = anthropic.Anthropic(api_key='test-key', base_url=f'http://127.0.0.1:{cls.server.server_port}', max_retries=0)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        import runner_adapter as A
        self.A = A
        self.tmp = tempfile.TemporaryDirectory(prefix='adapter-test-'); self.addCleanup(self.tmp.cleanup)
        self.w = fixture(Path(self.tmp.name) / 'run'); self.s = State(self.w.path / 'state.sqlite')
        R.reconcile(self.w, self.s)
        FakeMessages.bundles = {t['payload']['prompt'].split('\n', 1)[0]: t['payload'] for t in self.s.tasks()}
        FakeMessages.requests.clear(); FakeMessages.mode['kind'] = 'ok'

    def adapter(self, **kw):
        return self.A.Adapter(self.w, self.s, client=self.client, model='claude-opus-5-5',
                              responses=Path(self.tmp.name) / 'responses', **kw)

    def test_one_call_per_task_strict_validation_usage_and_file_contract(self):
        pending = len(self.s.tasks(states=('pending',), with_payload=False))
        result = self.adapter().run(workers=3)
        self.assertEqual(result['outcomes'], {'persisted': pending})
        self.assertEqual(len(FakeMessages.requests), pending)
        path, headers, body = FakeMessages.requests[0]
        self.assertTrue(path.split('?')[0].endswith('/v1/messages'), path)
        self.assertEqual(body['output_config'], {'effort': 'high'})
        self.assertNotIn('thinking', body)  # adaptive thinking is the model default; budget_tokens would be rejected
        self.assertEqual(body['fallbacks'], 'default')
        self.assertIn('server-side-fallback-2026-07-01', headers.get('anthropic-beta', ''))
        self.assertIn('## Complete source (line: text)', body['messages'][0]['content'][0]['text'])
        self.assertEqual(body['system'], self.A.boundary.INSTRUCTIONS)
        self.assertAlmostEqual(result['spent_usd'], pending * (1000 * 4.0 + 50 * 20.0) / 1e6, places=6)
        with self.s.connect() as db:
            usage = [json.loads(r['reason']) for r in db.execute("SELECT reason FROM events WHERE event='usage'")]
        self.assertEqual(len(usage), pending)
        self.assertEqual(usage[0]['request_id'], 'req_test'); self.assertEqual(usage[0]['outcome'], 'persisted')
        self.assertEqual(len(list((Path(self.tmp.name) / 'responses').glob('*.json'))), pending)
        self.assertEqual(self.s.counts(), {'succeeded': pending})
        # the file-adapter contract: the saved responses replay to the same journal state
        R.reconcile(self.w, self.s)

    def test_invalid_refused_and_rate_limited_replies_never_install(self):
        FakeMessages.mode['kind'] = 'bad_candidate'
        self.assertEqual(self.adapter().run(workers=1, max_tasks=1)['outcomes'], {'invalid': 1})
        blocked = self.s.tasks(states=('blocked',), with_payload=False)
        self.assertEqual(len(blocked), 1); self.assertIn('existing candidate', blocked[0]['reason'])
        FakeMessages.mode['kind'] = 'refusal'
        self.assertEqual(self.adapter().run(workers=1, max_tasks=1)['outcomes'], {'refusal': 1})
        FakeMessages.mode['kind'] = 'garbage'
        self.assertEqual(self.adapter().run(workers=1, max_tasks=1)['outcomes'], {'invalid': 1})
        FakeMessages.mode['kind'] = '429'
        self.assertEqual(self.adapter().run(workers=1, max_tasks=1)['outcomes'], {'rate_limited': 1})
        failed = self.s.tasks(states=('failed',), with_payload=False)
        self.assertEqual(len(failed), 1); self.assertIn('rate limited', failed[0]['reason'])
        self.assertEqual(self.s.counts().get('succeeded', 0), 0)
        self.assertEqual(len(FakeMessages.requests), 4)  # exactly one call per attempt, none repeated by the adapter

    def test_budget_cap_and_fenced_json_and_resume(self):
        FakeMessages.mode['kind'] = 'fenced'
        one_call = (1000 * 4.0 + 50 * 20.0) / 1e6
        result = self.adapter(max_usd=one_call * 2.5).run(workers=1)
        self.assertEqual(result['tasks'], 3)  # the third call starts under the cap; the fourth would not
        self.assertEqual(result['outcomes'], {'persisted': 3})
        resumed = self.adapter(max_usd=one_call * 2.5)  # recorded spend survives a restart
        self.assertAlmostEqual(resumed.spent, result['spent_usd'], places=6)
        self.assertEqual(resumed.run(workers=1)['tasks'], 0)
        self.assertEqual(self.adapter(max_usd=None).run(workers=2)['outcomes'], {'persisted': len(self.s.tasks()) - 3})


if __name__ == '__main__':
    unittest.main()

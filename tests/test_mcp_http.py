"""Real stdio/streamable-HTTP clients, every read-only tool, and public Host handling."""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from test_mcp_benchmarks import mcp, write_fixture
from test_mcp_comparison import fixture, point
if mcp._SDK_V2:
    import httpx2 as httpx
else:
    import httpx

ROOT = Path(__file__).resolve().parents[1]
CALLS = {
    'list_models': {},
    'search_quants': dict(model='alpha', limit=1),
    'get_repo': dict(model='alpha', repo='quant-a/quant'),
    'list_sets': dict(model='alpha'),
    'frontier': dict(model='alpha', set_id='test', metric='kld'),
    'best_under': dict(model='alpha', max_size_gb=10, metric='kld'),
    'compare_measurements': dict(model='alpha', left_id='a', right_id='b'),
    'list_benchmarks': dict(query='GPQA', min_models=2),
    'search_benchmarks': dict(benchmark='GPQA-Diamond', unit='as published', limit=2),
}


class HTTPConfigTests(unittest.TestCase):
    def test_original_tools_reject_nonfinite_budgets(self):
        for value in (float('nan'), float('inf'), -1, True):
            for tool, arguments in (
                (mcp.search_quants, dict(model='anything', max_size_gb=value)),
                (mcp.best_under, dict(model='anything', max_size_gb=value)),
                (mcp.frontier, dict(model='anything', set_id='anything', max_x=value)),
            ):
                with self.subTest(tool=tool.__name__, value=value), self.assertRaisesRegex(ValueError, 'finite'):
                    tool(**arguments)

    def test_malformed_public_origins_fail_closed(self):
        invalid = ('http://example.org', 'https://*.example.org', 'https://example.org/path',
                   'https://example.org?x=y', 'https://example.org#fragment',
                   'https://user:pass@example.org', 'https://example.org:bad',
                   'https://exam ple.org', 'https://example.org%2fevil')
        for url in invalid:
            with self.subTest(url=url), patch.dict(os.environ, {'QFF_PUBLIC_URL': url, 'SPACE_HOST': ''}):
                with self.assertRaises(ValueError):
                    mcp._http_security(8000)
        for host in ('*.hf.space', 'https://example.hf.space', 'example.hf.space/path'):
            with self.subTest(host=host), patch.dict(os.environ, {'QFF_PUBLIC_URL': '', 'SPACE_HOST': host}):
                with self.assertRaises(ValueError):
                    mcp._http_security(8000)

    def test_exact_public_hosts_retain_rebinding_protection(self):
        with patch.dict(os.environ, {'QFF_PUBLIC_URL': 'https://mcp.example.org', 'SPACE_HOST': 'qff-fixture.hf.space'}):
            security = mcp._http_security(8000)
        self.assertTrue(security.enable_dns_rebinding_protection)
        self.assertIn('mcp.example.org', security.allowed_hosts)
        self.assertIn('qff-fixture.hf.space', security.allowed_hosts)
        self.assertIn('https://qff-fixture.hf.space', security.allowed_origins)
        self.assertFalse(any('*' in host for host in security.allowed_hosts))


class MCPTransportTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='qff-mcp-transport-')
        cls.addClassCleanup(cls.tmp.cleanup)
        write_fixture(cls.tmp.name)
        path = Path(cls.tmp.name, 'alpha/data.json')
        data = json.loads(path.read_text())
        data.update(fixture([point('a'), point('b', 'quant-b', 5, .05)]))
        path.write_text(json.dumps(data))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cls.url = f'http://127.0.0.1:{cls.port}/mcp'
        cls.env = {**os.environ, 'QFF_DATA': cls.tmp.name,
                   'SPACE_HOST': 'qff-fixture.hf.space', 'QFF_PUBLIC_URL': 'https://mcp.example.org'}
        cls.log = tempfile.TemporaryFile(mode='w+b')
        cls.addClassCleanup(cls.log.close)
        cls.process = subprocess.Popen([sys.executable, str(ROOT / 'qff_mcp/qff_mcp.py'), '--http', str(cls.port)],
                                       env=cls.env, stdout=cls.log, stderr=cls.log)
        cls.addClassCleanup(cls.stop_server)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                cls.log.seek(0)
                raise RuntimeError('MCP server exited: ' + cls.log.read().decode())
            try:
                with socket.create_connection(('127.0.0.1', cls.port), timeout=.1):
                    return
            except OSError:
                time.sleep(.05)
        raise RuntimeError('MCP HTTP server did not become ready in 10 seconds')

    @classmethod
    def stop_server(cls):
        if cls.process.poll() is None:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait(timeout=5)

    async def check_every_tool(self, client, initialize=True):
        if initialize:
            initialized = await client.initialize()
            info = getattr(initialized, 'server_info', None) or initialized.serverInfo
        else:
            info = client.server_info
        self.assertEqual(info.name, 'qff')
        tools = (await client.list_tools()).tools
        self.assertEqual({tool.name for tool in tools}, set(CALLS))
        for tool in tools:
            annotation = tool.annotations.model_dump(by_alias=True)
            self.assertTrue(annotation['readOnlyHint'])
            self.assertFalse(annotation['destructiveHint'])
        self.assertEqual((await client.list_resources()).resources, [])
        self.assertEqual((await client.list_prompts()).prompts, [])
        results = {}
        for name, arguments in CALLS.items():
            result = await client.call_tool(name, arguments)
            self.assertFalse(bool(getattr(result, 'is_error', False) or getattr(result, 'isError', False)), f'{name}: {result}')
            results[name] = json.loads(result.content[0].text)
        self.assertEqual(len(results['list_models']['models']), 2)
        self.assertEqual(results['search_quants']['rows'][0]['measurement_id'], 'a')
        self.assertEqual(results['get_repo']['repo']['repo'], 'quant-a/quant')
        self.assertEqual(results['list_sets']['sets'][0]['id'], 'test')
        self.assertEqual(len(results['frontier']['efficient']), 2)
        self.assertEqual(results['best_under']['per_set'][0]['best']['measurement_id'], 'b')
        self.assertEqual(results['compare_measurements']['classification'], 'same_protocol')
        self.assertEqual(results['list_benchmarks']['benchmarks'][0]['model_count'], 2)
        self.assertEqual(results['search_benchmarks']['matches'], 4)
        self.assertEqual(results['search_benchmarks']['next_offset'], 2)
        self.assertFalse(results['search_benchmarks']['ranking_permitted'])
        error = await client.call_tool('compare_measurements', dict(model='alpha', left_id='a', right_id='missing'))
        self.assertTrue(bool(getattr(error, 'is_error', False) or getattr(error, 'isError', False)))
        self.assertIn('unknown measurement_id', error.content[0].text)
        error = await client.call_tool('search_benchmarks', dict(benchmark='GPQA-Diamond'))
        self.assertTrue(bool(getattr(error, 'is_error', False) or getattr(error, 'isError', False)))
        self.assertIn('select unit explicitly', error.content[0].text)
        print('MCP transport: initialize, tools/list, resources/list, prompts/list and all 9 tools passed.')

    async def test_all_nine_tools_over_stdio(self):
        parameters = StdioServerParameters(command=sys.executable,
            args=[str(ROOT / 'qff_mcp/qff_mcp.py')], env=self.env)
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await self.check_every_tool(client)

    async def test_all_nine_tools_over_http_with_space_host_and_origin(self):
        async with httpx.AsyncClient(trust_env=False,
                headers={'Host': 'qff-fixture.hf.space', 'Origin': 'https://qff-fixture.hf.space'}) as http_client:
            async with streamable_http_client(self.url, http_client=http_client) as streams:
                async with ClientSession(streams[0], streams[1]) as client:
                    await self.check_every_tool(client)

    async def test_custom_public_url_initializes_over_http(self):
        async with httpx.AsyncClient(trust_env=False,
                headers={'Host': 'mcp.example.org', 'Origin': 'https://mcp.example.org'}) as http_client:
            async with streamable_http_client(self.url, http_client=http_client) as streams:
                async with ClientSession(streams[0], streams[1]) as client:
                    initialized = await client.initialize()
                    info = getattr(initialized, 'server_info', None) or initialized.serverInfo
                    self.assertEqual(info.name, 'qff')
                    self.assertEqual(len((await client.list_tools()).tools), 9)

    @unittest.skipUnless(mcp._SDK_V2, 'Modern protocol requires SDK 2')
    async def test_sdk2_discovers_modern_protocol_and_calls_all_nine_tools(self):
        from mcp import Client
        async with httpx.AsyncClient(trust_env=False,
                headers={'Host': 'qff-fixture.hf.space', 'Origin': 'https://qff-fixture.hf.space'}) as http_client:
            transport = streamable_http_client(self.url, http_client=http_client)
            async with Client(transport, mode='auto') as client:
                self.assertEqual(client.protocol_version, '2026-07-28')
                await self.check_every_tool(client, initialize=False)

    async def test_http_refuses_unconfigured_hosts_and_origins(self):
        message = dict(jsonrpc='2.0', id=1, method='initialize', params=dict(
            protocolVersion='2025-03-26', capabilities={}, clientInfo=dict(name='qff-test', version='1')))
        async with httpx.AsyncClient(trust_env=False) as client:
            for host, origin, expected in (
                ('attacker.example.org', 'https://qff-fixture.hf.space', 421),
                ('qff-fixture.hf.space.evil.example', 'https://qff-fixture.hf.space', 421),
                ('qff-fixture.hf.space', 'https://attacker.example.org', 403),
            ):
                result = await client.post(self.url, json=message, headers=dict(
                    Host=host, Origin=origin, Accept='application/json, text/event-stream'))
                self.assertEqual(result.status_code, expected, result.text)


class PublishedHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_published_responses_work_with_default_client_limits(self):
        # Small fixtures miss SDK 2.3's default 1 MiB SSE-event limit. These
        # actual Qwen responses exceed it; ordinary JSON replies must work
        # without asking every connector to weaken its transport limits.
        site = Path(os.environ.get('QFF_DATA', ROOT / '_site')).resolve()
        path = site / 'qwen3.8-27b/data.json'
        if not path.is_file():
            self.skipTest('assemble.py must run before published-data HTTP checks')
        data = json.loads(path.read_text())
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        log = tempfile.TemporaryFile(mode='w+b')
        self.addCleanup(log.close)
        process = subprocess.Popen([sys.executable, str(ROOT / 'qff_mcp/qff_mcp.py'), '--http', str(port)],
            env={**os.environ, 'QFF_DATA': str(site), 'SPACE_HOST': '', 'QFF_PUBLIC_URL': ''},
            stdout=log, stderr=log)
        def stop():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        self.addCleanup(stop)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.seek(0)
                self.fail('MCP server exited: ' + log.read().decode())
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.1):
                    break
            except OSError:
                await asyncio.sleep(.05)
        else:
            self.fail('MCP HTTP server did not become ready in 10 seconds')
        response_types = []
        async def record_response(response):
            if response.request.method == 'POST' and response.status_code == 200:
                response_types.append(response.headers.get('content-type', ''))
        async with httpx.AsyncClient(trust_env=False, event_hooks={'response': [record_response]}) as http_client:
            async with streamable_http_client(f'http://127.0.0.1:{port}/mcp', http_client=http_client) as streams:
                async with ClientSession(streams[0], streams[1]) as client:
                    await client.initialize()
                    for name, arguments, field, expected in (
                        ('list_sets', dict(model=data['model'], comparable_only=False), 'sets', len(data['sets'])),
                        ('search_quants', dict(model=data['model'], limit=200), 'rows', min(200, len(data['rows']))),
                    ):
                        result = await client.call_tool(name, arguments)
                        self.assertFalse(bool(getattr(result, 'is_error', False) or getattr(result, 'isError', False)))
                        self.assertEqual(len(json.loads(result.content[0].text)[field]), expected)
                        print(f'Published HTTP: {name}, {len(result.content[0].text.encode())} bytes, {expected} records.')
        self.assertGreaterEqual(len(response_types), 3)
        self.assertTrue(all(kind.startswith('application/json') for kind in response_types), response_types)


if __name__ == '__main__':
    unittest.main()

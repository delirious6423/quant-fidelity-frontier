#!/usr/bin/env python3
"""Measure repeated MCP responses over real stdio and HTTP, plus handler profiles.

Run after assemble.py with the pinned MCP SDK 2.3.0:
    python3 tests/benchmark_mcp.py --output mcp-performance.json

The fixed workload asks every model for all set classifications and the best
quant under 100 GB. One warmup round precedes timed rounds; import, process start,
and model loading are excluded. Response hashes detect any changed answers.
Use --server to measure an isolated experimental server with comparison.py beside
it, and identical --site input to compare runs. Output contains no local paths.
This is an opt-in benchmark, not a CI timing assertion.
"""
import argparse,asyncio,cProfile,hashlib,importlib.metadata,importlib.util,json,os,pathlib,pstats,socket,statistics,subprocess,sys,tempfile,time
from collections import defaultdict
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
import httpx2
ROOT=pathlib.Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--server',type=pathlib.Path,default=ROOT/'qff_mcp/qff_mcp.py')
parser.add_argument('--site',type=pathlib.Path,default=ROOT/'_site')
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--rounds',type=int,default=5)
args=parser.parse_args()
if args.rounds < 1:parser.error('--rounds must be positive')
SERVER=args.server.resolve();SITE=args.site.resolve();OUT=args.output;ROUNDS=args.rounds
manifest=json.loads((SITE/'models.json').read_text())
CALLS=[(n,dict(model=m['slug'],**a)) for m in manifest['models'] for n,a in [('list_sets',dict(comparable_only=False)),('best_under',dict(max_size_gb=100))]]
output={'stretch_target_speedup':100,'workload':{'models':len(manifest['models']),'calls_per_round':len(CALLS),'warmup_rounds':1,'measured_rounds':ROUNDS,'tools':['list_sets(comparable_only=False)','best_under(max_size_gb=100)'],'sdk':importlib.metadata.version('mcp'),'python':sys.version.split()[0],'server_sha256':hashlib.sha256(SERVER.read_bytes()).hexdigest(),'data_sha256':{m['slug']:hashlib.sha256((SITE/m['data']).read_bytes()).hexdigest() for m in manifest['models']}}}
def stats(x):return {'median_ms':statistics.median(x)*1000,'min_ms':min(x)*1000,'max_ms':max(x)*1000,'samples_ms':[v*1000 for v in x]}
async def measure(c):
 await c.initialize(); times=[];bytool=defaultdict(list);fingerprints={};sizes={}
 for r in range(ROUNDS+1):
  start=time.perf_counter()
  for n,a in CALLS:
   t=time.perf_counter();v=await c.call_tool(n,a);elapsed=time.perf_counter()-t
   assert not bool(getattr(v,'is_error',False) or getattr(v,'isError',False)),v
   if r:bytool[n].append(elapsed)
   raw=v.content[0].text.encode();key=a['model']+'/'+n
   digest=hashlib.sha256(raw).hexdigest()
   if key in fingerprints:assert fingerprints[key]==digest
   fingerprints[key]=digest;sizes[key]=len(raw)
  if r:times.append(time.perf_counter()-start)
 return {'round':stats(times),'per_call':{k:stats(v) for k,v in bytool.items()},'result_sha256':fingerprints,'result_text_bytes':sizes}
async def main():
 env={**os.environ,'QFF_DATA':str(SITE),'QFF_PUBLIC_URL':'','SPACE_HOST':''}
 with tempfile.TemporaryFile(mode='w+') as log:
  async with stdio_client(StdioServerParameters(command=sys.executable,args=[str(SERVER)],env=env),errlog=log) as (r,w):
   async with ClientSession(r,w) as c:output['stdio']=await measure(c)
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 with tempfile.TemporaryFile() as log:
  process=subprocess.Popen([sys.executable,str(SERVER),'--http',str(port)],env=env,stdout=log,stderr=log)
  try:
   for _ in range(100):
    try:
     with socket.create_connection(('127.0.0.1',port),timeout=.1):break
    except OSError:await asyncio.sleep(.05)
   async with httpx2.AsyncClient(trust_env=False) as h:
    async with streamable_http_client(f'http://127.0.0.1:{port}/mcp',http_client=h) as streams:
     async with ClientSession(streams[0],streams[1]) as c:output['http']=await measure(c)
  finally:process.terminate();process.wait(timeout=5)
 # Attribute warm actual handler work; source reading and import are excluded.
 sys.path.insert(0,str(SERVER.parent));s=importlib.util.spec_from_file_location('subject',SERVER);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.SRC=str(SITE)
 for n,a in CALLS:getattr(m,n)(**a)
 times=[];prof=cProfile.Profile()
 for _ in range(ROUNDS):
  t=time.perf_counter()
  for n,a in CALLS:getattr(m,n)(**a)
  times.append(time.perf_counter()-t)
 prof.enable()
 for n,a in CALLS:getattr(m,n)(**a)
 prof.disable();ps=pstats.Stats(prof)
 top=sorted(ps.stats.items(),key=lambda v:v[1][3],reverse=True)[:12]
 output['direct']={'round':stats(times),'profile_top':[{'function':k[2],'calls':v[1],'cumulative_seconds':v[3]} for k,v in top]}
 OUT.write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps({k:output[k]['round'] for k in ('stdio','http','direct')},indent=2))
asyncio.run(main())

"""Repeatable V8 JSON parsing benchmark over the five real model pages.

Usage: python3 tests/benchmark_payload.py --site _site --output result.json
This isolates chart-data parsing; it is not a network or whole-page latency test.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


def measure(site):
    results = []
    for path in sorted(site.glob('*/index.html')):
        html = path.read_bytes()
        match = re.search(rb'<script type="application/json" id="exdata">(.*?)</script>', html, re.S)
        if not match:
            continue
        payload = match[1]
        # V8 warmup and nine independent batches; no disk I/O in the timing.
        program = r'''
const fs = require('fs');
const data = fs.readFileSync(0, 'utf8');
for (let i=0;i<10;i++) JSON.parse(data);
const samples=[];
let count=0;
for (let batch=0;batch<9;batch++) {
  const start=performance.now();
  for (let i=0;i<30;i++) count+=JSON.parse(data).sets.length;
  samples.push((performance.now()-start)/30);
}
samples.sort((a,b)=>a-b);
process.stdout.write(JSON.stringify({median_ms:samples[4],min_ms:samples[0],max_ms:samples[8],count}));
'''
        run = subprocess.run(['node', '-e', program], input=payload, capture_output=True, check=True)
        timing = json.loads(run.stdout)
        results.append(dict(model=path.parent.name, html_bytes=len(html),
                            html_gzip_bytes=len(gzip.compress(html, mtime=0)),
                            inline_bytes=len(payload), parse=timing,
                            data_sha256=hashlib.sha256((path.parent/'data.json').read_bytes()).hexdigest()))
    if not results:
        raise ValueError('No model chart payloads found; build and assemble the site first')
    return dict(workload='real-model inline chart JSON; V8, 10 warmups, 9 x 30 parses',
                node=subprocess.check_output(['node', '--version'], text=True).strip(), models=results)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',type=Path,default=Path('_site'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=json.dumps(measure(args.site),indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(report+'\n')
    print(report)

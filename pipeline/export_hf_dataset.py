#!/usr/bin/env python3
"""Export assembled public data to a reviewable HF dataset directory; never upload."""
import argparse
import json
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--site', default='_site')
p.add_argument('--out', default='dist/hf-dataset')
a = p.parse_args()
site, out = Path(a.site), Path(a.out)
manifest = json.loads((site / 'models.json').read_text())
out.mkdir(parents=True, exist_ok=True)
coverage = []
count = 0
with (out / 'claims.jsonl').open('w') as f:
    for model in manifest['models']:
        data = json.loads((site / model['data']).read_text())
        coverage.append(f'- {data["model"]}: captured {data["captured_at"]}; {len(data["rows"])} claim rows')
        for row in data['rows']:
            # Stable scalar columns for Dataset Viewer; original nested objects stay lossless JSON.
            record = {'model': data['model'], 'captured_at': data['captured_at'], 'repo': row['repo'],
                      'variant': row.get('variant'), 'kind': row['kind'], 'size_gb': row.get('size_gb'),
                      'basis': row.get('basis'), 'src': row['src'], 'sha': row['sha'],
                      'source_revision_pinned': bool(re.fullmatch(r'[0-9a-fA-F]{40}', row['sha'] or '')),
                      'claim_json': json.dumps(row, ensure_ascii=False)}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
        # Preserve the full original set membership and repo methodology separately.
        (out / (model['slug'] + '.json')).write_text(json.dumps(data, ensure_ascii=False))
(out / 'README.md').write_text('''---
pretty_name: QFF Quantization Fidelity Claims
license: cc-by-4.0
language:
- en
tags:
- quantization
- llm
- provenance
- kld
configs:
- config_name: default
  data_files:
  - split: train
    path: claims.jsonl
---
# Quant Fidelity Frontier

Publisher-reported quantization claims, transcribed from captured sources. This is an evidence catalogue, not independently reproduced benchmark results. The `train` split is a packaging convention, not a recommended training use.

## Coverage
''' + '\n'.join(coverage) + '''

## Provenance and schema
`src` links to the publisher source; `sha` may be a commit or a mutable ref such as `main`. `source_revision_pinned` is true only for a full commit hash. Mutable URLs can change after capture; do not describe them as immutable citations. Captured snapshots remain in the source project for audit. `claim_json` preserves the complete original claim, raw metrics, units and flags. Each model's companion JSON contains repository methodology and measurement sets. Join through the original claim fields and set point metadata; never infer comparability from the shared model name alone.

## Limitations
The overview mixes methods. Compare only compatible measurement sets, references and size bases. Missing metrics are unknown. File size is not total VRAM usage. Copied claims are not independent replications. Coverage is bounded by capture dates and review status; missing or quarantined claims are not proof of poor quality.

## Attribution and licensing
The compiled dataset is CC BY 4.0, as declared by the source project. Attribute Quant Fidelity Frontier and retain original publisher source links. Code is MIT. Source model cards and model weights retain their owners' terms; this export does not relicense them and does not include cached card files or model weights.

[Project and correction requests](https://github.com/delirious6423/quant-fidelity-frontier) · [Interactive comparisons](https://delirious6423.github.io/quant-fidelity-frontier/)
''')
print(json.dumps({'rows': count, 'models': len(coverage), 'out': str(out)}))

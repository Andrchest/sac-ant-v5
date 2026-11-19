#!/usr/bin/env python3
"""Export small primary results and a provenance-checked derived campaign index.

No original run files are modified. Excludes models, videos and replay buffers.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
OUT = ROOT / 'research' / 'campaign'
INDEX = ART / 'campaign' / 'campaign_summary.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(index=INDEX, out=OUT, root=ROOT):
    meta = json.loads(index.read_text(encoding='utf-8'))
    evidence = {'original_index_sha256': digest(index), 'sources': {}}
    out.mkdir(parents=True, exist_ok=True)
    for item in meta['experiments']:
        if item.get('status') != 'done':
            continue
        source_dir = Path(item['run_dir'])
        if not source_dir.is_absolute():
            source_dir = root / source_dir
        target = out / 'runs' / item['id']
        target.mkdir(parents=True, exist_ok=True)
        summary_path = source_dir / 'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        evidence['sources'][item['id']] = {'summary_sha256': digest(summary_path)}
        for filename in ('summary.json', 'evaluation_all.csv'):
            original = source_dir / filename
            if not original.is_file():
                raise FileNotFoundError(original)
            (target / filename).write_bytes(original.read_bytes())
            evidence['sources'][item['id']][filename + '_sha256'] = digest(original)
        item['checkpoints'] = summary.get('checkpoints', [])
        item['run_dir'] = target.relative_to(root).as_posix()
    for seed in (50, 51, 52):
        matches = list((root / 'artifacts').glob(f'xp_sac_{seed}_*best{seed}/summary.json'))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one summary for seed {seed}: {matches}')
        target = out / 'best' / f'seed_{seed}'
        target.mkdir(parents=True, exist_ok=True)
        evidence['sources'][f'best_{seed}'] = {}
        for filename in ('summary.json', 'evaluation_all.csv'):
            original = matches[0].parent / filename
            (target / filename).write_bytes(original.read_bytes())
            evidence['sources'][f'best_{seed}'][filename + '_sha256'] = digest(original)
    (out / 'derived_summary.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    (out / 'provenance.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    return meta


def export_publication(root=ROOT):
    run = root / 'artifacts/ant_sac_full_20261021T194131399885Z'
    ref = root / 'research/campaign/reference'
    ref.mkdir(parents=True, exist_ok=True)
    for name in ('summary.json', 'evaluation_all.csv', 'unseen_reset_validation.csv',
                 'unseen_reset_validation.json'):
        (ref / name).write_bytes((run / name).read_bytes())
    public = root / 'research/publication'
    public.mkdir(parents=True, exist_ok=True)
    (public / 'index.html').write_bytes((run / 'index.html').read_bytes())
    report = root / 'artifacts/night_20261021T224130+0300/report.md'
    (public / 'report.md').write_bytes(report.read_bytes())


if __name__ == '__main__':
    print(f'Exported {len(export()["experiments"])} campaign records into {OUT}')
    export_publication()

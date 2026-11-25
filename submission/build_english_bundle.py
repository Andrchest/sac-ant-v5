"""Translate Markdown in the compact archive; preserve all other payload bytes and paths."""
from pathlib import Path
import hashlib
import json
import re
import os
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/ant_sac_full_20261021T194131399885Z'
PREFIX = 'sac-ant-v5/'


def main():
    source = RUN / 'sac_ant_v5_submission_compact.zip'
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupt source')
        files = {i.filename.removeprefix(PREFIX): z.read(i) for i in z.infolist()}
    manifest = json.loads(files.pop('MANIFEST.json'))
    if set(files) != set(manifest['sha256']):
        raise ValueError('Unexpected source paths')
    for n, data in files.items():
        if hashlib.sha256(data).hexdigest() != manifest['sha256'][n]:
            raise ValueError(f'Source hash mismatch: {n}')
    original = dict(files)
    replacements = {
        'README.md': ROOT / 'README.md',
        'PRESENTATION_DRAFT.md': ROOT / 'submission/PRESENTATION_DRAFT.md',
        'README_SUBMISSION.md': ROOT / 'submission/README_SUBMISSION.md',
        'artifacts/campaign/campaign_report.md': ROOT / 'artifacts/campaign/campaign_report.md',
    }
    for name, path in replacements.items():
        files[name] = path.read_bytes()
    for name, data in files.items():
        if name.endswith('.md') and re.search('[\u0400-\u04ff]', data.decode('utf-8')):
            raise ValueError(f'Untranslated Cyrillic in {name}')
        if not name.endswith('.md') and data != original[name]:
            raise ValueError(f'Unexpected non-Markdown change: {name}')
    if set(files) != set(original):
        raise ValueError('Payload paths changed')
    manifest['parent_archive_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['markdown_language'] = 'English'
    manifest['sha256'] = {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())}
    files['MANIFEST.json'] = json.dumps(manifest, indent=2).encode()
    target = RUN / 'sac_ant_v5_submission_english.zip'
    temp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w') as z:
        for n, b in sorted(files.items()):
            info = zipfile.ZipInfo(PREFIX + n)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, b)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupt output')
        for n, digest in manifest['sha256'].items():
            if hashlib.sha256(z.read(PREFIX + n)).hexdigest() != digest:
                raise ValueError(n)
        if any(n.endswith('00_START_HERE.html') for n in z.namelist()):
            raise ValueError('HTML navigation must not be included')
    os.replace(temp, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(f'{digest}  {target.name}\n')
    # Publish the translated standalone draft, too.
    (RUN / 'PRESENTATION_DRAFT.md').write_bytes(files['PRESENTATION_DRAFT.md'])
    (RUN / 'README_SUBMISSION.md').write_bytes(files['README_SUBMISSION.md'])
    print(f'PASS: {sum(n.endswith(".md") for n in files)} English Markdown files; '
          f'paths preserved; non-Markdown bytes unchanged; no HTML navigation')
    print(target, target.stat().st_size, digest)


if __name__ == '__main__':
    main()

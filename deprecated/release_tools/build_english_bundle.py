"""Translate compact archive Markdown and drop obsolete hand-in assets."""
from pathlib import Path
import hashlib
import json
import re
import os
import sys
from tempfile import TemporaryDirectory
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT))
from deprecated.ablation_campaign import write_markdown

RUN = ROOT / 'artifacts/ant_sac_full_20261021T194131399885Z'
PREFIX = 'sac-ant-v5/'
CYRILLIC = re.compile('[\u0400-\u052f]')
OBSOLETE_NAMES = {'PRESENTATION_DRAFT.md', 'make_presentation_video.py',
                  'make_slide.py', 'presentation_demo.mp4', 'results_slide.png'}


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
    obsolete = {name for name in files if name.rsplit('/', 1)[-1] in OBSOLETE_NAMES}
    removed_paths = obsolete | {'README_SUBMISSION.md', 'submission/README_SUBMISSION.md'}
    for name in removed_paths:
        files.pop(name, None)
    expected_paths = set(original) - removed_paths

    video_path = f'artifacts/{RUN.name}/best_config/seed_50/rollout.mp4'
    local_video = RUN / 'best_config/seed_50/rollout.mp4'
    if obsolete and video_path not in files and local_video.is_file():
        files[video_path] = local_video.read_bytes()
        expected_paths.add(video_path)

    if obsolete:
        bundle_guide = files['BUNDLE_README.md'].decode('utf-8').replace('\r\n', '\n')
        bundle_guide = bundle_guide.replace(
            'the superseded `rollout.mp4`, the earlier\n`results_slide.png`, the intermediate ',
            'the intermediate ')
        if video_path in files:
            bundle_guide = bundle_guide.replace('presentation_demo.mp4', 'rollout.mp4')
        else:
            bundle_guide = re.sub(r'(?m)^- Open `[^\n]*/presentation_demo\.mp4` in a video player\.\n',
                                  '', bundle_guide)
            bundle_guide = bundle_guide.replace('CSV tables and recorded demo video.', 'and CSV tables.')
        files['BUNDLE_README.md'] = bundle_guide.encode('utf-8')

    if 'support/watch_sac.py' in original:
        files['README.md'] = (ROOT / 'README.md').read_bytes()
    elif obsolete or 'deprecated/__init__.py' not in original:
        # The bundled guide documents the original layout and commands.
        files['README.md'] = files['BUNDLE_README.md']
    for name in ('REPORT.md', 'OVERNIGHT_REPORT.md'):
        if name in original and CYRILLIC.search(original[name].decode('utf-8')):
            files[name] = (ROOT / 'REPORT.md').read_bytes()

    campaign_report = 'artifacts/campaign/campaign_report.md'
    if CYRILLIC.search(files[campaign_report].decode('utf-8')):
        summary = json.loads(files['artifacts/campaign/campaign_summary.json'])
        with TemporaryDirectory(prefix='sac-campaign-report-') as directory:
            report = Path(directory) / 'campaign_report.md'
            write_markdown(report, summary)
            files[campaign_report] = report.read_bytes()

    for name, data in files.items():
        if name.endswith('.md') and CYRILLIC.search(data.decode('utf-8')):
            raise ValueError(f'Untranslated Cyrillic in {name}')
        if name in original and not name.endswith('.md') and data != original[name]:
            raise ValueError(f'Unexpected non-Markdown change: {name}')
    if set(files) != expected_paths:
        raise ValueError('Unexpected change to payload paths')
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
    print(f'PASS: {sum(n.endswith(".md") for n in files)} English Markdown files, verified archive integrity')
    print(target, target.stat().st_size, digest)


if __name__ == '__main__':
    main()

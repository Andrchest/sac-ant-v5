"""Download the served ZIP, verify it, then run its standalone inference demo."""
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import urllib.request
import zipfile

base = 'http://100.64.62.13:8008/'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(base, timeout=10) as response:
    assert b'sac_ant_v5_bundle.zip' in response.read()
with opener.open(base + 'sac_ant_v5_bundle.zip.sha256', timeout=10) as response:
    digest = response.read().decode().split()[0]
with opener.open(base + 'sac_ant_v5_bundle.zip', timeout=20) as response:
    assert response.status == 200
    payload = response.read()
    assert len(payload) == int(response.headers['Content-Length'])
assert len(payload) < 10_000_000
assert hashlib.sha256(payload).hexdigest() == digest
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    assert not any('replay_buffer' in n for n in names)
    assert not any(part in {'.env', '.git', '.venv', '.pi'} for n in names for part in Path(n).parts)
    manifest = json.loads(archive.read('sac-ant-v5/MANIFEST.json'))
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256(archive.read('sac-ant-v5/' + name)).hexdigest() == expected
    with tempfile.TemporaryDirectory(prefix='sac-bundle-check-') as temporary:
        archive.extractall(temporary)
        demo = Path(temporary) / 'sac-ant-v5/demo_policy.py'
        subprocess.run([sys.executable, str(demo), '--episodes', '1'], check=True, timeout=120)
print('HTTP DOWNLOAD + ZIP INTEGRITY + STANDALONE INFERENCE PASSED')
print('Archive bytes:', len(payload), 'SHA256:', digest)

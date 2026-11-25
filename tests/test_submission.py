import hashlib
import json
import math
from pathlib import Path


def test_toy_sac_critic_update_is_standalone_and_correct():
    notebook = json.loads(Path('sac_ant_v5.ipynb').read_text(encoding='utf-8'))
    cells = [c for c in notebook['cells'] if c.get('id') == 'toy-sac-update']
    assert len(cells) == 1 and cells[0]['cell_type'] == 'code'
    source = cells[0]['source']
    namespace = {}
    exec(''.join(source) if isinstance(source, list) else source, namespace)
    result = namespace['toy_sac_critic_update']()
    assert math.isclose(result['target'], 4.7026, abs_tol=1e-10)
    assert math.isclose(result['q_after'], 2.54052, abs_tol=1e-10)
    assert math.isclose(result['loss_after'], 0.64 * result['loss_before'], abs_tol=1e-10)


def test_notebook_has_unique_ids_and_training_cells():
    import nbformat
    notebook = nbformat.read('sac_ant_v5.ipynb', as_version=4)
    nbformat.validate(notebook)
    ids = [cell['id'] for cell in notebook.cells]
    assert len(ids) == len(set(ids))
    assert {'config', 'train', 'summary', 'toy-sac-update'} <= set(ids)


def test_followup_provenance_hash_has_shipped_source():
    base = Path('research/followup')
    provenance = json.loads((base / 'provenance.json').read_text())
    assert hashlib.sha256((base / 'source_status.json').read_bytes()).hexdigest() == provenance['status_sha256']


def test_report_withdraws_unverifiable_stochastic_result():
    report = Path('REPORT.md').read_text(encoding='utf-8')
    assert 'previously quoted stochastic evaluation is withdrawn' in report
    assert '5685 ± 129' not in report

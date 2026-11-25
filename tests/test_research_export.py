import hashlib
from importlib import import_module
import json

export = import_module('additional analysis.export_research_data').export


def test_derived_index_recovers_checkpoints_without_modifying_original(tmp_path):
    original = tmp_path / 'artifacts' / 'campaign' / 'campaign_summary.json'
    original.parent.mkdir(parents=True)
    run = tmp_path / 'artifacts' / 'xp_conv'
    run.mkdir()
    checkpoint = {'step': 250000, 'return_mean': 42.}
    (run / 'summary.json').write_text(json.dumps({'checkpoints': [checkpoint]}))
    (run / 'evaluation_all.csv').write_text('policy,return_\nsac_final,42\n')
    for seed in (50, 51, 52):
        p = tmp_path / 'artifacts' / f'xp_sac_{seed}_test_best{seed}'
        p.mkdir()
        (p / 'summary.json').write_text('{}')
        (p / 'evaluation_all.csv').write_text('policy,return_\nsac_final,1\n')
    original.write_text(json.dumps({'experiments': [{'id': 'conv', 'status': 'done', 'run_dir': str(run)}]}))
    before = original.read_bytes()
    output = tmp_path / 'research' / 'campaign'
    derived = export(original, output, tmp_path)
    assert original.read_bytes() == before
    assert derived['experiments'][0]['checkpoints'] == [checkpoint]
    assert derived['experiments'][0]['run_dir'] == 'research/campaign/runs/conv'
    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['original_index_sha256'] == hashlib.sha256(before).hexdigest()
    assert (output / 'runs/conv/evaluation_all.csv').read_bytes() == (run / 'evaluation_all.csv').read_bytes()

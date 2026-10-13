import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent_state_lab.cli import main
from recurrent_state_lab.experiment import Protocol, datasets, run
from recurrent_state_lab.report import render_report
from recurrent_state_lab.zoology import mqar


def test_mqar_query_targets_match_the_earlier_dictionary():
    data = mqar(examples=32, length=32, pairs=4, seed=51)
    assert int((data.labels != -100).sum()) == 128
    for tokens, targets in zip(data.inputs, data.labels):
        mapping = {int(tokens[i]): int(tokens[i+1]) for i in range(0, 8, 2)}
        for position in torch.nonzero(targets != -100).flatten():
            assert int(position) >= 8
            assert mapping[int(tokens[position])] == int(targets[position])


def test_data_generation_repeats_without_changing_callers_rng():
    torch.manual_seed(19)
    np.random.seed(21)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    first, second = mqar(seed=43), mqar(seed=43)
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(torch.get_rng_state(), torch_before)
    after = np.random.get_state()
    assert numpy_before[0] == after[0]
    assert np.array_equal(numpy_before[1], after[1])
    assert numpy_before[2:] == after[2:]


@pytest.mark.parametrize('values', [dict(length=15), dict(pairs=5), dict(vocab=16),
                                  dict(examples=0), dict(length=True)])
def test_invalid_mqar_dimensions_refused(values):
    with pytest.raises(ValueError):
        mqar(**values)


def test_exact_dataset_rows_separated():
    _, _, _, manifest = datasets(Protocol(train_examples=64, test_examples=16,
                                         validation_examples=16))
    seen = set()
    for entry in manifest.values():
        assert not seen.intersection(entry['row_sha256'])
        seen.update(entry['row_sha256'])


@pytest.mark.parametrize('protocol', [Protocol(epochs=0), Protocol(epochs=33),
    Protocol(seeds=(7, 7)), Protocol(seeds=(True,)), Protocol(train_examples=5000),
    Protocol(test_lengths=(16, 128)), Protocol(learning_rate=float('nan'))])
def test_recipe_limits_are_enforced(protocol):
    with pytest.raises(ValueError):
        protocol.validate()


def test_existing_output_refused_without_a_failure_file(tmp_path):
    output = tmp_path/'existing'
    output.mkdir()
    sentinel = output/'keep.txt'
    sentinel.write_text('keep', encoding='utf-8')
    with pytest.raises(SystemExit) as error:
        main(['--output', str(output)])
    assert error.value.code == 2
    with pytest.raises(FileExistsError):
        run(output, Protocol())
    assert sentinel.read_text(encoding='utf-8') == 'keep'
    assert sorted(p.name for p in output.iterdir()) == ['keep.txt']


def test_runtime_failure_recorded_only_in_new_owned_output(tmp_path, monkeypatch):
    from recurrent_state_lab import experiment
    monkeypatch.setattr(experiment, '_run', lambda *args: (_ for _ in ()).throw(ValueError('controlled failure')))
    output = tmp_path/'new'
    with pytest.raises(ValueError, match='controlled failure'):
        run(output, Protocol())
    receipt = json.loads((output/'failure.json').read_text(encoding='utf-8'))
    assert receipt['error_type'] == 'ValueError'
    assert receipt['message'] == 'controlled failure'


def test_report_does_not_embed_executable_data():
    rendered = render_report({'note': '</script><script>throw "unsafe"</script>', 'runs': []})
    assert '</script><script>throw' not in rendered
    assert '\\u003c/script>' in rendered
    assert '<meta name="artifact-revision"' in rendered

import copy

import pytest
import torch

from recurrent_state_lab.experiment import controls, parameter_hash
from recurrent_state_lab.zoology import RecallModel, mqar


@pytest.mark.parametrize('loops', [1, 2])
def test_learned_model_full_split_step_and_reset(loops):
    torch.manual_seed(41)
    model = RecallModel(loops).eval()
    result = controls(model, mqar(examples=2, length=32, seed=29).inputs)
    assert result['state_offsets'] == [32] * loops
    assert result['returned_state_bytes'] == 9216 * loops


def test_shared_block_matches_parameters_but_keeps_two_states():
    torch.manual_seed(7)
    one = RecallModel(1).eval()
    torch.manual_seed(7)
    two = RecallModel(2).eval()
    assert parameter_hash(one) == parameter_hash(two)
    assert sum(p.numel() for p in one.parameters()) == sum(p.numel() for p in two.parameters())
    assert two.base.backbone.layers[0] is two.base.backbone.layers[1]
    tokens = mqar(examples=1).inputs
    with torch.no_grad():
        _, states = two.stream(tokens)
    assert len(states) == 2
    assert states[0].recurrent_state.data_ptr() != states[1].recurrent_state.data_ptr()
    assert states[0].offset == states[1].offset == 16


@pytest.mark.parametrize('loops', [1, 2])
def test_whole_model_gradients_survive_split_streaming(loops):
    torch.manual_seed(51)
    full = RecallModel(loops).eval()
    split = copy.deepcopy(full)
    tokens = mqar(examples=2, seed=3).inputs
    full(tokens).square().mean().backward()
    first, states = split.stream(tokens[:, :7])
    second, _ = split.stream(tokens[:, 7:], states)
    torch.cat([first, second], dim=1).square().mean().backward()
    full_grads = dict(full.named_parameters())
    for name, p in split.named_parameters():
        assert p.grad is not None and full_grads[name].grad is not None
        torch.testing.assert_close(p.grad, full_grads[name].grad, atol=3e-5, rtol=3e-4)


def test_interleaved_model_requests_do_not_replace_each_others_state():
    model = RecallModel(2).eval()
    tokens = mqar(examples=2, length=32).inputs
    with torch.no_grad():
        a1, state_a = model.stream(tokens[:1, :9])
        b1, state_b = model.stream(tokens[1:, :13])
        b2, _ = model.stream(tokens[1:, 13:], state_b)
        a2, _ = model.stream(tokens[:1, 9:], state_a)
        torch.testing.assert_close(torch.cat([a1, a2], 1), model(tokens[:1]), atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(torch.cat([b1, b2], 1), model(tokens[1:]), atol=3e-4, rtol=3e-4)


def test_wrong_loop_state_count_refused():
    model = RecallModel(2).eval()
    with pytest.raises(ValueError, match='one independent'):
        model.stream(torch.zeros(1, 1, dtype=torch.long), state=(None,))

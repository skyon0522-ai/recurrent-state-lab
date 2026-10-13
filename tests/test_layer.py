"""Independent unchanged-operation oracle and explicit-state regression checks."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import pytest
import torch
from torch.nn import functional as F

from recurrent_state_lab import reference as ref
from recurrent_state_lab._source import load_symbols, source_record
from recurrent_state_lab.reference import CPUReferenceLayer, LayerState


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def layer(**kwargs):
    torch.manual_seed(37)
    return CPUReferenceLayer(hidden_size=12, head_dim=3, num_heads=2, expand_v=2, **kwargs)


def initial(model, batch=2, gradients=False):
    torch.manual_seed(97)
    recurrence = (torch.randn(batch, model.num_heads, model.head_v_dim, model.head_k_dim) * .1).requires_grad_(gradients)
    convolution = tuple((torch.randn(batch, dim, model.conv_size) * .1).requires_grad_(gradients)
        for dim in (model.key_dim, model.key_dim, model.value_dim)) if model.use_short_conv else None
    return LayerState(recurrence, convolution, 7, model._state_owner)


def oracle(model, x, state=None):
    """Compose unchanged refs directly, independently of wrapper/cache shims.

    In particular convolution uses the upstream decoding reference over the
    whole sequence; the tested full-sequence layer uses its prefill binding.
    """
    projected = [model.q_proj(x), model.k_proj(x), model.v_proj(x)]
    convolution = None
    if model.use_short_conv:
        convolution, values = [], []
        for index, (raw, module) in enumerate(zip(projected, (model.q_conv1d, model.k_conv1d, model.v_conv1d))):
            cache = state.conv_state[index].clone() if state is not None else raw.new_zeros(raw.shape[0], raw.shape[-1], model.conv_size)
            value = ref._conv["causal_conv1d_update_ref"](raw.transpose(1, 2), cache,
                module.weight[:, 0], module.bias, module.activation)
            values.append(value.transpose(1, 2))
            convolution.append(cache)
        convolution = tuple(convolution)
    else:
        values = [F.silu(raw) for raw in projected]
    q, k = [value.reshape(*x.shape[:2], model.num_heads, model.head_k_dim) for value in values[:2]]
    v = values[2].reshape(*x.shape[:2], model.num_heads, model.head_v_dim)
    q, k = [value / torch.sqrt(value.square().sum(-1, keepdim=True) + 1e-6) for value in (q, k)]
    g = ref._gate_ref(model.a_proj(x), model.A_log, model.dt_bias)
    beta = model.b_proj(x).sigmoid() * (2 if model.allow_neg_eigval else 1)
    output, final = ref._naive["naive_recurrent_gated_delta_rule"](q, k, v, beta, g,
        initial_state=state.recurrent_state.transpose(-1, -2) if state is not None else None,
        output_final_state=True)
    if model.use_gate:
        gate = model.g_proj(x).reshape_as(output)
        if model.o_norm.activation in ("silu", "swish"):
            output = ref._gated_rms_ref(output, model.o_norm.weight, None, z=gate, eps=model.o_norm.eps)
        else:
            output = ref._rms_ref(output, model.o_norm.weight, None, eps=model.o_norm.eps) * gate.sigmoid()
    else:
        output = ref._rms_ref(output, model.o_norm.weight, None, eps=model.o_norm.eps)
    output = model.o_proj(output.reshape(*x.shape[:2], model.value_dim))
    return output, LayerState(final.transpose(-1, -2).contiguous(), convolution,
        (state.offset if state is not None else 0) + x.shape[1], model._state_owner)


def assert_state(left, right):
    assert left.offset == right.offset
    torch.testing.assert_close(left.recurrent_state, right.recurrent_state, atol=3e-5, rtol=3e-5)
    assert (left.conv_state is None) == (right.conv_state is None)
    for a, b in zip(left.conv_state or (), right.conv_state or ()):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)


def stream(model, x, state, mode, style):
    if style == "full":
        return model(x, state, mode)
    parts = (x[:, :2], x[:, 2:5], x[:, 5:]) if style == "split" else x.split(1, dim=1)
    outputs = []
    for part in parts:
        output, state = model(part, state, mode)
        outputs.append(output)
    return torch.cat(outputs, dim=1), state


def test_fixed_sources_and_original_class_bodies():
    vendor = Path(ref.__file__).parent / "_vendor"
    manifest = json.loads((vendor / "fla/manifest.json").read_text())
    for row in manifest:
        assert hashlib.sha256((vendor / row["package_path"]).read_bytes()).hexdigest() == row["sha256"]
        if row["selected_symbols"]:
            record = source_record(row["package_path"], row["selected_symbols"])
            assert record["sha256"] == row["sha256"]
    source = ast.parse((vendor / "fla/layers/gated_deltanet.py").read_bytes())
    original = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "GatedDeltaNet")
    forward = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    for implementation in (ref._ChunkLayer, ref._RecurrentLayer):
        actual = ast.parse(textwrap.dedent(inspect.getsource(implementation.forward))).body[0]
        assert ast.dump(actual, include_attributes=False) == ast.dump(forward, include_attributes=False)
    assert ref._ChunkLayer.forward.__code__.co_code == ref._RecurrentLayer.forward.__code__.co_code


@pytest.mark.parametrize("width", (1, 4))
@pytest.mark.parametrize("length", (1, 3, 4, 9, 67))
@pytest.mark.parametrize("gate", (None, "swish", "sigmoid"))
@pytest.mark.parametrize("negative", (False, True))
def test_full_outputs_and_rectangular_state_against_unchanged_refs(width, length, gate, negative):
    model = layer(conv_size=width, use_gate=gate is not None,
        output_gate_activation=gate or "swish", allow_neg_eigval=negative, conv_bias=True)
    x = torch.randn(2, length, 12)
    state = initial(model)
    expected, expected_state = oracle(model, x, state)
    for mode in ("recurrent", "chunk"):
        actual, actual_state = model(x, state, mode)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        assert_state(actual_state, expected_state)
        assert actual_state.recurrent_state.shape == (2, 2, 6, 3)


@pytest.mark.parametrize("width", (1, 4))
@pytest.mark.parametrize("mode", ("recurrent", "chunk"))
@pytest.mark.parametrize("style", ("split", "step"))
@pytest.mark.parametrize("autograd", (False, True))
def test_full_split_step_outputs_state_and_dispatch(width, mode, style, autograd):
    model = layer(conv_size=width).eval()
    x = torch.randn(2, 13, 12)
    state = initial(model)
    with torch.set_grad_enabled(autograd):
        expected, expected_state = oracle(model, x, state)
        output, final = stream(model, x, state, mode, style)
    torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)
    assert_state(final, expected_state)
    assert final.offset == 20


@pytest.mark.parametrize("width", (1, 4))
@pytest.mark.parametrize("gate", (None, "silu", "sigmoid"))
@pytest.mark.parametrize("mode", ("recurrent", "chunk"))
@pytest.mark.parametrize("style", ("full", "split", "step"))
def test_input_parameter_and_initial_state_gradient_parity(width, gate, mode, style):
    kwargs = {"conv_size": width, "use_gate": gate is not None,
        "output_gate_activation": gate or "swish", "allow_neg_eigval": True, "conv_bias": True}
    expected_model, actual_model = layer(**kwargs), layer(**kwargs)
    torch.manual_seed(59)
    raw = torch.randn(2, 9, 12)
    all_gradients = []
    states = []
    for model, use_oracle in ((expected_model, True), (actual_model, False)):
        x = raw.clone().requires_grad_()
        state = initial(model, gradients=True)
        output, final = oracle(model, x, state) if use_oracle else stream(model, x, state, mode, style)
        loss = output.square().mean() + final.recurrent_state.square().mean()
        loss = loss + sum(tensor.square().mean() for tensor in final.conv_state)
        differentiable = [x, *model.parameters(), state.recurrent_state, *state.conv_state]
        gradients = torch.autograd.grad(loss, differentiable)
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        all_gradients.append(gradients)
        states.append(final)
    for actual, expected in zip(all_gradients[1], all_gradients[0]):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    assert_state(states[0], states[1])


def test_small_norm_qk_and_value_first_conversion():
    torch.manual_seed(101)
    q, k = [torch.randn(1, 3, 2, 3) * 1e-8 for _ in range(2)]
    v, beta, g = torch.randn(1, 3, 2, 5), torch.randn(1, 3, 2), torch.randn(1, 3, 2)
    a, dt, state = torch.randn(2), torch.randn(2), torch.randn(1, 2, 5, 3)
    expected, final = ref._naive["naive_recurrent_gated_delta_rule"](
        q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6),
        k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6), v, beta.sigmoid() * 2,
        ref._gate_ref(g, a, dt), initial_state=state.transpose(-1, -2), output_final_state=True)
    output, actual = ref._rule_binding("recurrent")(q, k, v, g, beta, state, True,
        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True, state_v_first=True, use_gate_in_kernel=True, A_log=a, dt_bias=dt)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(actual, final.transpose(-1, -2))


def test_reset_isolation_immutability_detachment_and_fixed_storage():
    model = layer().eval()
    a, b = torch.randn(2, 11, 12), torch.randn(2, 8, 12)
    parameters = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        whole_a, final_a = model(a)
        whole_b, final_b = model(b)
        head_a, carry_a = model(a[:, :2])
        head_b, carry_b = model(b[:, :3])
        frozen = [carry_a.recurrent_state.clone(), *[t.clone() for t in carry_a.conv_state]]
        tail_a, streamed_a = stream(model, a[:, 2:], carry_a, "recurrent", "step")
        tail_b, streamed_b = stream(model, b[:, 3:], carry_b, "chunk", "step")
        reset_a, reset_state = model(a, state=None)
    torch.testing.assert_close(torch.cat((head_a, tail_a), 1), whole_a)
    torch.testing.assert_close(torch.cat((head_b, tail_b), 1), whole_b, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(reset_a, whole_a)
    assert_state(streamed_a, final_a)
    assert_state(streamed_b, final_b)
    assert_state(reset_state, final_a)
    for before, after in zip(frozen, (carry_a.recurrent_state, *carry_a.conv_state)):
        assert torch.equal(before, after)
    for before, after in zip((carry_a.recurrent_state, *carry_a.conv_state), (streamed_a.recurrent_state, *streamed_a.conv_state)):
        assert before.data_ptr() != after.data_ptr()
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, parameters[name])
    payload = []
    with torch.no_grad():
        for length in (1, 17, 129):
            _, state = model(torch.randn(2, length, 12))
            tensors = (state.recurrent_state, *state.conv_state)
            assert sum(t.untyped_storage().nbytes() for t in tensors) == state.nbytes
            payload.append(state.nbytes)
    assert len(set(payload)) == 1
    _, attached = model(a.requires_grad_())
    detached = attached.detach()
    assert attached.recurrent_state.grad_fn is not None
    assert all(t.grad_fn is None for t in (detached.recurrent_state, *detached.conv_state))


def test_no_convolution_and_default_dimensions():
    with pytest.warns(UserWarning, match="ShortConvolution"):
        model = layer(use_short_conv=False)
    x = torch.randn(2, 9, 12)
    expected, state = oracle(model, x)
    actual, final = model(x, mode="chunk")
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    assert_state(state, final)
    assert final.conv_state is None
    default = CPUReferenceLayer()
    output, final = default(torch.randn(2, 9, 48))
    assert output.shape == (2, 9, 48) and final.recurrent_state.shape == (2, 4, 12, 12)
    assert final.nbytes == 9216


@pytest.mark.parametrize("bad", (
    torch.randn(12), torch.randn(1, 2, 11), torch.randn(0, 2, 12),
    torch.randn(1, 0, 12), torch.randn(1, 2, 12, dtype=torch.float64),
    torch.full((1, 2, 12), float("nan")), torch.empty(1, 2, 12, device="meta"),
))
def test_invalid_inputs_are_refused(bad):
    with pytest.raises(ValueError):
        layer()(bad)


@pytest.mark.parametrize("options", (
    {"attention_mask": torch.ones(1, 2)}, {"cu_seqlens": torch.tensor([0, 2])},
    {"mode": "fused_recurrent"}, {"use_cache": False}, {"output_attentions": True},
    {"cp_context": object()},
))
def test_unsupported_runtime_options_are_refused(options):
    with pytest.raises(ValueError):
        layer()(torch.randn(1, 2, 12), **options)


@pytest.mark.parametrize("options", (
    {"hidden_size": 0}, {"head_dim": 0}, {"num_heads": 0}, {"conv_size": 0},
    {"expand_v": .1}, {"expand_v": float("nan")}, {"norm_eps": 0},
    {"num_v_heads": 8}, {"output_gate_activation": "relu"},
))
def test_invalid_config_is_refused(options):
    with pytest.raises(ValueError):
        CPUReferenceLayer(**options)


def test_invalid_state_and_parameter_precision_are_refused():
    model = layer()
    x = torch.randn(2, 3, 12)
    state = initial(model)
    alternatives = [
        initial(layer()), replace(state, offset=-1), replace(state, offset=True),
        replace(state, recurrent_state=torch.randn(2, 2, 3, 6)),
        replace(state, recurrent_state=state.recurrent_state.double()),
        replace(state, recurrent_state=torch.full_like(state.recurrent_state, float("nan"))),
        replace(state, conv_state=None), replace(state, conv_state=state.conv_state[:2]),
        replace(state, conv_state=(torch.randn(2, 6, 3), *state.conv_state[1:])),
        replace(state, conv_state=(state.conv_state[0].double(), *state.conv_state[1:])),
    ]
    for bad in alternatives:
        with pytest.raises(ValueError):
            model(x, bad)
    with pytest.raises(ValueError):
        model(x[:1], state)
    with pytest.raises(ValueError):
        model.double()(x)


def test_loader_refuses_path_escape_and_missing_or_duplicate_symbols():
    for path, names in (("../outside.py", ["x"]), ("fla/ops/gated_delta_rule/naive.py", ["missing"]),
                        ("fla/ops/gated_delta_rule/naive.py", []),
                        ("fla/ops/gated_delta_rule/naive.py", ["naive_recurrent_gated_delta_rule"] * 2)):
        with pytest.raises(ValueError):
            load_symbols(path, names, {})


@pytest.mark.parametrize("mode", ("recurrent", "chunk"))
@pytest.mark.parametrize("autograd,training,length", ((True, True, 3), (True, False, 3), (False, False, 1), (False, False, 65)))
def test_requested_mode_is_actual_reference_dispatch(monkeypatch, mode, autograd, training, length):
    observed = []
    for name in ("naive_recurrent_gated_delta_rule", "naive_chunk_gated_delta_rule"):
        original = ref._naive[name]
        def tracked(*args, _name=name, _original=original, **kwargs):
            observed.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setitem(ref._naive, name, tracked)
    model = layer().train(training)
    with torch.set_grad_enabled(autograd):
        model(torch.randn(2, length, 12), mode=mode)
    assert observed == [f"naive_{mode}_gated_delta_rule"]


@pytest.mark.parametrize("bias", (False, True))
def test_original_fused_qkv_convolution_path_matches_separate_cached_path(monkeypatch, bias):
    model = layer(conv_bias=bias)
    x = torch.randn(2, 9, 12)
    assert model._use_fused_qkv_conv(last_state=None, use_cache=False, cu_seqlens=None)
    called_widths = []
    namespace = ref._ChunkLayer.forward.__globals__
    original = namespace["causal_conv1d"]
    def tracked(x, **kwargs):
        called_widths.append(x.shape[-1])
        return original(x, **kwargs)
    monkeypatch.setitem(namespace, "causal_conv1d", tracked)
    fused, _, returned_cache = ref._ChunkLayer.forward(model, x, use_cache=False)
    separate, _ = model(x, mode="chunk")
    torch.testing.assert_close(fused, separate, atol=3e-5, rtol=3e-5)
    assert returned_cache is None
    assert called_widths == [2 * model.key_dim + model.value_dim]


def test_dense_precision_boundary_and_noncontiguous_input():
    model = layer()
    x = torch.randn(2, 9, 24)[..., ::2]
    assert not x.is_contiguous()
    expected, expected_state = oracle(model, x)
    output, state = model(x)
    torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)
    assert_state(state, expected_state)
    with pytest.raises(ValueError, match="autocast"):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(x)
    with pytest.raises(ValueError):
        model(x.to_sparse())
    with pytest.raises(ValueError):
        CPUReferenceLayer(use_gate="false")

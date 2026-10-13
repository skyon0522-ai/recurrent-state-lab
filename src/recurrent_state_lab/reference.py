"""Explicit-state CPU FP32 composition of pinned FLA reference operations.

FLA's class bodies and callable references are loaded without modification.
These narrow bindings adapt layout, flags, and cache ownership; they are not
optimized FLA kernels or a native upstream CPU backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from ._source import load_symbols


_TORCH = {"torch": torch, "nn": nn, "F": F, "rearrange": rearrange}
_naive = load_symbols("fla/ops/gated_delta_rule/naive.py", (
    "naive_recurrent_gated_delta_rule", "naive_chunk_gated_delta_rule"), _TORCH)
_conv = load_symbols("fla/tests/modules/test_conv.py", (
    "_conv_ref_compute_dtype", "causal_conv1d_ref", "causal_conv1d_update_ref"),
    {**_TORCH, "_CONV_REF_FP32_DTYPES": (torch.float16, torch.bfloat16)})
_rms_ref = load_symbols("fla/modules/norm/layernorm.py", ("rms_norm_ref",), _TORCH)["rms_norm_ref"]
_gated_rms_ref = load_symbols("fla/modules/norm/layernorm_gated.py", ("rms_norm_ref",), _TORCH)["rms_norm_ref"]
_gate_ref = load_symbols("fla/ops/gated_delta_rule/gate.py", ("naive_gdn_gate",), _TORCH)["naive_gdn_gate"]


def _causal_conv1d(x, weight, bias=None, residual=None, initial_state=None,
                   output_final_state=False, activation=None, backend=None,
                   cu_seqlens=None, chunk_indices=None):
    if cu_seqlens is not None or chunk_indices is not None:
        raise ValueError("packed convolution is unsupported by the CPU reference")
    channels_first = x.transpose(1, 2)
    history = initial_state[..., 1:] if initial_state is not None else None
    output = _conv["causal_conv1d_ref"](channels_first, weight, bias, history, activation=activation)
    final = None
    if output_final_state:
        raw = torch.cat([initial_state, channels_first], -1) if initial_state is not None else channels_first
        width = weight.shape[-1]
        # Clone the tail: retaining a view would retain storage growing with T.
        final = F.pad(raw, (max(0, width - raw.shape[-1]), 0))[..., -width:].clone()
    output = output.transpose(1, 2)
    return (output + residual if residual is not None else output), final


def _causal_conv1d_update(x, cache, weight, bias=None, residual=None, activation=None):
    # Original ShortConvolution.step receives a call-private clone of the cache.
    output = _conv["causal_conv1d_update_ref"](x.transpose(1, 2), cache, weight, bias, activation)
    output = output.transpose(1, 2)
    return (output + residual if residual is not None else output), cache


_ShortConvolution = load_symbols("fla/modules/causal_conv1d/ops.py", ("ShortConvolution",), {
    **_TORCH, "warnings": warnings,
    # Retain the original constructor label, but bind both calls to Torch.
    # Its optional environment override cannot select an unrelated GPU backend.
    "os": SimpleNamespace(environ={}), "causal_conv1d_fn_cuda": None,
    "causal_conv1d": _causal_conv1d, "causal_conv1d_update": _causal_conv1d_update,
})["ShortConvolution"]


def _rms_norm(x, weight, bias, residual=None, eps=1e-5, prenorm=False, residual_in_fp32=False):
    return _rms_ref(x, weight, bias, residual=residual, eps=eps, prenorm=prenorm, upcast=True)


def _rms_norm_gated(x, g, weight, bias, activation="swish", residual=None,
                    prenorm=False, residual_in_fp32=False, eps=1e-5):
    if residual is not None or prenorm:
        raise ValueError("residual/prenorm gated RMS is outside this layer reference")
    if activation in ("swish", "silu"):
        return _gated_rms_ref(x, weight, bias, z=g, eps=eps)
    if activation == "sigmoid":
        return _rms_ref(x, weight, bias, eps=eps, upcast=True) * g.sigmoid()
    raise ValueError(f"unsupported output gate activation: {activation!r}")


_RMSNorm = load_symbols("fla/modules/norm/layernorm.py", ("RMSNorm",),
    {**_TORCH, "rms_norm": _rms_norm})["RMSNorm"]
_FusedRMSNormGated = load_symbols("fla/modules/norm/fused_norm_gate.py", ("FusedRMSNormGated",),
    {**_TORCH, "rms_norm_gated": _rms_norm_gated})["FusedRMSNormGated"]
_cache_helpers = load_symbols("fla/layers/utils.py", (
    "require_cache_layer_idx", "get_layer_cache", "update_layer_cache",
    "unpad_hidden_states", "repad_hidden_states"), {
        **_TORCH, "_LAYER_IDX_REQUIRED_MSG": "{cls} requires layer_idx when a cache is supplied",
    })


def _rule_binding(mode):
    def rule(q, k, v, g, beta, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False,
             allow_neg_eigval=False, state_v_first=False, cu_seqlens=None,
             use_gate_in_kernel=False, A_log=None, dt_bias=None):
        if cu_seqlens is not None:
            raise ValueError("packed recurrence is unsupported by the CPU reference")
        if use_qk_l2norm_in_kernel:
            q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        if use_beta_sigmoid_in_kernel:
            beta = beta.sigmoid() * (2 if allow_neg_eigval else 1)
        if use_gate_in_kernel:
            g = _gate_ref(g, A_log, dt_bias)
        if initial_state is not None and state_v_first:
            initial_state = initial_state.transpose(-1, -2)
        if mode == "recurrent":
            output, final = _naive["naive_recurrent_gated_delta_rule"](
                q, k, v, beta, g, initial_state=initial_state, output_final_state=output_final_state)
        else:
            output, final = _naive["naive_chunk_gated_delta_rule"](
                q, k, v, g, beta, initial_state=initial_state, output_final_state=output_final_state)
        if final is not None and state_v_first:
            final = final.transpose(-1, -2).contiguous()
        return output, final
    return rule


def _original_layer(mode):
    kernel = _rule_binding(mode)
    return load_symbols("fla/layers/gated_deltanet.py", ("GatedDeltaNet",), {
        **_TORCH, **_cache_helpers, "math": math, "warnings": warnings,
        "ShortConvolution": _ShortConvolution, "RMSNorm": _RMSNorm,
        "FusedRMSNormGated": _FusedRMSNormGated, "causal_conv1d": _causal_conv1d,
        "chunk_gated_delta_rule": kernel, "fused_recurrent_gated_delta_rule": kernel,
    })["GatedDeltaNet"]


_ChunkLayer = _original_layer("chunk")
_RecurrentLayer = _original_layer("recurrent")


@dataclass(frozen=True)
class LayerState:
    """One dense batch's state, explicitly carried by its caller.

    Recurrent layout is [B,H,V,K], convolution layout is [B,D,W]. The state
    retains its autograd graph until detach() is called. Keep batch order fixed.
    nbytes counts tensor payload, excluding graph/Python overhead.
    """
    recurrent_state: torch.Tensor
    conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    offset: int
    layer_id: object

    @property
    def nbytes(self) -> int:
        tensors = (self.recurrent_state, *(self.conv_state or ()))
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def detach(self) -> LayerState:
        return LayerState(self.recurrent_state.detach(),
            tuple(tensor.detach() for tensor in self.conv_state) if self.conv_state is not None else None,
            self.offset, self.layer_id)


class _CallCache:
    def __init__(self, state):
        self.offset = state.offset if state is not None else 0
        self.value = None if state is None else {
            "recurrent_state": state.recurrent_state,
            "conv_state": tuple(tensor.clone() for tensor in state.conv_state) if state.conv_state is not None else None,
        }

    def __len__(self):
        return int(self.value is not None)

    def __getitem__(self, layer_idx):
        if layer_idx != 0 or self.value is None:
            raise IndexError(layer_idx)
        return self.value

    def update(self, layer_idx, recurrent_state, conv_state, offset):
        if layer_idx != 0:
            raise ValueError("only one layer is represented by a call-local cache")
        self.offset += offset
        self.value = {"recurrent_state": recurrent_state, "conv_state": conv_state}


class CPUReferenceLayer(_ChunkLayer):
    """Dense CPU FP32 GatedDeltaNet with functional, caller-owned state.

    mode='recurrent' and mode='chunk' select the actual unchanged naive rule.
    Each mode binds both kernel names used by the untouched upstream forward;
    upstream automatic mode selection cannot override the requested comparison.
    Padding, packing, grouped value heads, and GPU/mixed precision are refused.
    """
    def __init__(self, hidden_size=48, head_dim=12, num_heads=4, expand_v=1,
                 conv_size=4, use_gate=True, use_short_conv=True, conv_bias=False,
                 norm_eps=1e-5, allow_neg_eigval=False, output_gate_activation="swish",
                 num_v_heads=None):
        for name, value in (("hidden_size", hidden_size), ("head_dim", head_dim),
                            ("num_heads", num_heads), ("conv_size", conv_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("use_gate", use_gate), ("use_short_conv", use_short_conv),
                            ("conv_bias", conv_bias), ("allow_neg_eigval", allow_neg_eigval)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        if isinstance(expand_v, bool) or not isinstance(expand_v, (int, float)) or not math.isfinite(expand_v) or expand_v <= 0:
            raise ValueError("expand_v must be positive and finite")
        if not isinstance(norm_eps, (int, float)) or not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite")
        if num_v_heads is not None and (not isinstance(num_v_heads, int) or isinstance(num_v_heads, bool) or num_v_heads != num_heads):
            raise ValueError("grouped value heads are unsupported by the CPU reference")
        super().__init__(hidden_size=hidden_size, head_dim=head_dim, num_heads=num_heads,
            expand_v=expand_v, conv_size=conv_size, use_gate=use_gate,
            use_short_conv=use_short_conv, conv_bias=conv_bias, norm_eps=norm_eps,
            allow_neg_eigval=allow_neg_eigval, output_gate_activation=output_gate_activation,
            layer_idx=0, mode="chunk")
        self._state_owner = object()

    @staticmethod
    def _tensor(tensor, shape, name):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.is_nested or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be CPU float32")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain finite values")

    def forward(self, x, state=None, mode="recurrent", *, attention_mask=None, cu_seqlens=None, **kwargs):
        if attention_mask is not None or cu_seqlens is not None or kwargs:
            raise ValueError("padding, packed inputs, and extra kernel/cache options are unsupported")
        if mode not in ("recurrent", "chunk"):
            raise ValueError("mode must be 'recurrent' or 'chunk'")
        if torch.is_autocast_enabled("cpu"):
            raise ValueError("CPU autocast is unsupported; use float32 reference execution")
        if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.is_nested or x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("input must be a nonempty [B,T,hidden_size] tensor")
        batch, length, _ = x.shape
        self._tensor(x, (batch, length, self.hidden_size), "input")
        if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("layer parameters must be CPU float32")
        if state is not None:
            if not isinstance(state, LayerState) or state.layer_id is not self._state_owner:
                raise ValueError("state must belong to this layer")
            if not isinstance(state.offset, int) or isinstance(state.offset, bool) or state.offset < 0:
                raise ValueError("state offset must be a nonnegative integer")
            self._tensor(state.recurrent_state, (batch, self.num_heads, self.head_v_dim, self.head_k_dim), "recurrent state")
            if self.use_short_conv:
                if not isinstance(state.conv_state, tuple) or len(state.conv_state) != 3:
                    raise ValueError("state must contain all three convolution histories")
                for tensor, dim, name in zip(state.conv_state, (self.key_dim, self.key_dim, self.value_dim), ("q", "k", "v")):
                    self._tensor(tensor, (batch, dim, self.conv_size), f"{name} convolution state")
            elif state.conv_state is not None:
                raise ValueError("a layer without convolution cannot consume convolution state")
        cache = _CallCache(state)
        implementation = _RecurrentLayer if mode == "recurrent" else _ChunkLayer
        output, _, _ = implementation.forward(self, x, past_key_values=cache, use_cache=True)
        final = LayerState(cache.value["recurrent_state"], cache.value["conv_state"], cache.offset, self._state_owner)
        if not bool(torch.isfinite(output).all() and torch.isfinite(final.recurrent_state).all()):
            raise RuntimeError("CPU reference produced non-finite output or state")
        return output, final

"""Pinned Zoology data/model bodies; CPU bindings and explicit streaming extension.

Original code remains Apache-2.0 in _vendor/zoology. No CUDA, telemetry or
configuration-loader imports are needed by the selected unchanged bodies.
"""
from dataclasses import dataclass
from functools import lru_cache, partial
import math
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._source import load_symbols


@lru_cache(maxsize=1)
def originals():
    bindings = dict(torch=torch, nn=nn, F=F, math=math, np=np, partial=partial,
                    dataclass=dataclass, Dict=Dict)
    segment = load_symbols('zoology/data/utils.py', ['DataSegment'], bindings)['DataSegment']
    data = load_symbols('zoology/data/multiquery_ar.py', ['multiquery_ar'],
                        {**bindings, 'DataSegment': segment})['multiquery_ar']
    mlp = load_symbols('zoology/mixers/mlp.py', ['MLP'], bindings)['MLP']
    token = load_symbols('zoology/model.py', ['TokenEmbeddings'], bindings)['TokenEmbeddings']

    def zero_depth(p, mode):
        if p != 0 or mode != 'row':
            raise ValueError('Only the original zero-drop-path configuration is supported.')
        return nn.Identity()

    model = load_symbols('zoology/model.py',
        ['_init_weights', 'TransformerBlock', 'LMBackbone', 'LanguageModel'],
        {**bindings, 'StochasticDepth': zero_depth,
         'TokenEmbeddings': partial(token, device='cpu')})
    return dict(data=data, segment=segment, token=token, mlp=mlp, **model)


def mqar(*, examples=512, length=16, pairs=2, vocab=128, seed=17):
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in (examples, length, pairs, vocab, seed)):
        raise ValueError('MQAR dimensions and seed must be integers.')
    if examples < 1 or length < 4 or length % 2 or pairs < 1 or pairs * 4 > length or vocab <= length:
        raise ValueError('Require positive examples/pairs, even length >= 4*pairs, and vocab > length.')
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return originals()['data'](vocab_size=vocab, num_examples=examples,
                input_seq_len=length, num_kv_pairs=pairs, seed=seed,
                power_a=1.0, random_non_queries=False)
    finally:
        np.random.set_state(numpy_state)


class _Mixer(nn.Module):
    def __init__(self, d_model, layer_idx=0):
        super().__init__()
        from .reference import CPUReferenceLayer
        self.layer = CPUReferenceLayer(hidden_size=d_model, head_dim=12,
                                       num_heads=4, expand_v=1, conv_size=4)

    def forward(self, x):
        return self.layer(x, mode='chunk')[0]


class RecallModel(nn.Module):
    """Unchanged Zoology LM bodies with a shared FLA block and separate loop state."""
    def __init__(self, loops=1, vocab=128, width=48):
        super().__init__()
        if loops not in (1, 2) or width != 48 or vocab != 128:
            raise ValueError('This bounded demo supports loops 1/2, width48 and vocab128.')
        source = originals()
        config = SimpleNamespace(
            vocab_size=vocab, d_model=width, n_layers=1, multiplier=1,
            max_position_embeddings=0, learnable_word_embeddings=True,
            embedding_init_type='default', pad_vocab_size_multiple=1,
            block_type='TransformerBlock', resid_dropout=0, embed_dropout=0,
            drop_path=0, layer_norm_epsilon=1e-5,
            sequence_mixer=SimpleNamespace(instantiate=lambda **kw: _Mixer(**kw)),
            state_mixer=SimpleNamespace(instantiate=lambda **kw: source['mlp'](**kw, hidden_mult=2)),
        )
        self.base = source['LanguageModel'](config)
        block = self.base.backbone.layers[0]
        self.base.backbone.layers = nn.ModuleList([block] * loops)
        self.loops = loops

    def forward(self, tokens):
        return self.base(tokens)

    def stream(self, tokens, state=None):
        if self.training:
            raise ValueError('Streaming is an evaluation path; call model.eval().')
        if state is not None and (not isinstance(state, tuple) or len(state) != self.loops):
            raise ValueError('Provide one independent returned state for each loop.')
        states = (None,) * self.loops if state is None else state
        backbone = self.base.backbone
        hidden = backbone.embeddings(tokens)
        residual = None
        next_states = []
        # Same operations/order as Zoology TransformerBlock.forward, adding only
        # an explicit state argument to the FLA sequence mixer for each loop.
        for block, cached in zip(backbone.layers, states):
            dropped = block.drop_path1(block.dropout1(hidden))
            residual = dropped + residual if residual is not None else dropped
            hidden = block.norm1(residual.to(block.norm1.weight.dtype))
            hidden, cached = block.sequence_mixer.layer(hidden, state=cached, mode='recurrent')
            dropped = block.drop_path2(block.dropout2(hidden))
            residual = dropped + residual if residual is not None else dropped
            hidden = block.norm2(residual.to(block.norm2.weight.dtype))
            hidden = block.state_mixer(hidden)
            next_states.append(cached)
        dropped = backbone.drop_f(hidden)
        residual = dropped + residual if residual is not None else dropped
        hidden = backbone.ln_f(residual.to(backbone.ln_f.weight.dtype))
        return self.base.lm_head(hidden), tuple(next_states)

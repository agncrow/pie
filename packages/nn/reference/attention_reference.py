"""nn's attention, transformers and dropout held to PyTorch (torch 2.8,
float64).

Each functional case applies one of PyTorch's functions to fixed inputs;
each layer case makes one of PyTorch's modules from torch.manual_seed and
runs it. A layer case in training draws its dropout from the generator
after its parameters, as a module made and then run draws: pie draws the
same numbers from a torch generator seeded alike (nn.dropout_noise). The script
writes:

  data/at_inputs.csv        every input: its name, its sizes and its
                            numbers (rounded to four decimals, so the same
                            double in both; a mask's -inf as written),
                            joined by spaces
  expected/at_cases.csv     each case: its name, its inputs (up to seven,
                            `none` past them), its seed (a layer case's;
                            -1 for a function), the distance its row may
                            be from pie's (1e-12 unless the case says why)
                            and its row: the numbers of its value, then
                            the gradient of sum(value * c) with respect to
                            each input — c whole eighths, as reference.py's
                            — and, for a layer, first the parameters torch
                            drew (pie's must be the same numbers) and the
                            gradient with respect to them after the
                            value's

A value of two parts (an attention's output and its weights) is the one
then the other, flat. held_attention.co computes each row with pie and
compares.

Run from pie's root: ~/.venvs/gpt/bin/python -I packages/nn/reference/attention_reference.py
"""
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261110)
inputs = {}
cases = []
NEG = float('-inf')


def given(name, *dims, scale=1.0):
    inputs[name] = torch.tensor(np.round(rng.normal(0.0, scale, dims), 4))


def fixed(name, values):
    inputs[name] = torch.tensor(values)


def weights(n):
    return torch.tensor([float((k * 7) % 11 - 5) / 4.0 + 0.125 for k in range(n)])


def case(name, fn, names, within=1e-12):
    cases.append((name, fn, None, names, -1, within, False))


def layer(name, make, run, names, seed, train=False, within=1e-12):
    cases.append((name, make, run, names, seed, within, train))


# inputs
given('q4', 2, 2, 3, 4)
given('k4', 2, 2, 5, 4)
given('v4', 2, 2, 5, 3)
given('sq4', 2, 2, 4, 4)
given('q3', 3, 3, 4)
given('k3', 3, 6, 4)
given('v3', 3, 6, 2)
given('m35', 3, 5)
given('m2135', 2, 1, 3, 5)
fixed('pad35', [[0.0, 0.0, 0.0, NEG, NEG], [0.0, 0.0, NEG, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
given('x482', 4, 2, 8)
given('xb', 2, 4, 8)
given('kv', 6, 2, 5)
given('vv', 6, 2, 3)
given('xu', 4, 8)
given('mem', 5, 2, 8)
given('memb', 2, 5, 8)
given('src', 6, 2, 8)
given('tgt', 4, 2, 8)
given('m44', 4, 4)
given('m46', 4, 6)
fixed('causal4', torch.triu(torch.full((4, 4), NEG), diagonal=1).tolist())
fixed('causal6', torch.triu(torch.full((6, 6), NEG), diagonal=1).tolist())
fixed('kpm26', [[0.0, 0.0, 0.0, 0.0, 0.0, NEG], [0.0, NEG, 0.0, 0.0, 0.0, 0.0]])
fixed('kpm25', [[0.0, 0.0, 0.0, 0.0, NEG], [NEG, 0.0, 0.0, 0.0, 0.0]])
fixed('kpm24', [[0.0, 0.0, 0.0, NEG], [0.0, 0.0, 0.0, 0.0]])
given('xd', 3, 40)
given('xc4', 2, 6, 3, 2)
given('xc3', 3, 5, 4)
given('xc5', 2, 3, 2, 2, 2)
given('xr', 5, 2, 3)
given('xrb', 2, 5, 3)
given('m45', 4, 5)

# scaled_dot_product_attention: a float mask is added, a causal one is
# torch's own lower triangle; the scale is 1/√E unless given
case('sdpa', lambda q, k, v: F.scaled_dot_product_attention(q, k, v), ['q4', 'k4', 'v4'])
case('sdpa_mask', lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m), ['q4', 'k4', 'v4', 'm35'])
case('sdpa_broadcast', lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m), ['q4', 'k4', 'v4', 'm2135'])
case('sdpa_padded', lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m), ['q4', 'k4', 'v4', 'pad35'])
case('sdpa_causal', lambda q: F.scaled_dot_product_attention(q, q * 0.5, q, is_causal=True), ['sq4'])
case('sdpa_scale', lambda q, k, v: F.scaled_dot_product_attention(q, k, v, scale=0.3), ['q3', 'k3', 'v3'])

# multi-head attention: the output, then the weights where asked for
def attended(mha, q, k, v, **kw):
    out, w = mha(q, k, v, **kw)
    return out.reshape(-1) if w is None else torch.cat([out.reshape(-1), w.reshape(-1)])


layer('MHA_self', lambda: nn.MultiheadAttention(8, 2), lambda m, x: attended(m, x, x, x), ['x482'], 1)
layer('MHA_heads', lambda: nn.MultiheadAttention(8, 4), lambda m, x, msk: attended(m, x, x, x, attn_mask=msk, average_attn_weights=False), ['x482', 'm44'], 2)
layer('MHA_batch_first', lambda: nn.MultiheadAttention(8, 2, batch_first=True), lambda m, x: attended(m, x, x, x, need_weights=False), ['xb'], 3)
layer('MHA_kv_dims', lambda: nn.MultiheadAttention(8, 2, kdim=5, vdim=3), lambda m, x, k, v, p: attended(m, x, k, v, key_padding_mask=p), ['x482', 'kv', 'vv', 'kpm26'], 4)
layer('MHA_bias_kv_zero', lambda: nn.MultiheadAttention(8, 2, add_bias_kv=True, add_zero_attn=True), lambda m, x, msk: attended(m, x, x, x, attn_mask=msk), ['x482', 'causal4'], 5)
layer('MHA_unbatched', lambda: nn.MultiheadAttention(8, 2, bias=False), lambda m, x: attended(m, x, x, x), ['xu'], 6)
layer('MHA_cross', lambda: nn.MultiheadAttention(8, 2), lambda m, x, mem, msk: attended(m, x, mem, mem, attn_mask=msk, need_weights=False), ['x482', 'mem', 'm45'], 7)

# the transformer's layers, encoders and decoders (dropout 0: run as
# made, in training)
layer('Encoder_layer', lambda: nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0), lambda m, x: m(x).reshape(-1), ['x482'], 8)
layer('Encoder_layer_prenorm', lambda: nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, activation='gelu', norm_first=True, batch_first=True), lambda m, x: m(x).reshape(-1), ['xb'], 9)
layer('Encoder_layer_masked', lambda: nn.TransformerEncoderLayer(8, 4, dim_feedforward=12, dropout=0.0, bias=False), lambda m, x, msk, p: m(x, src_mask=msk, src_key_padding_mask=p).reshape(-1), ['src', 'causal6', 'kpm26'], 10)
layer('Decoder_layer', lambda: nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0), lambda m, x, mem, msk, p: m(x, mem, tgt_mask=msk, memory_key_padding_mask=p).reshape(-1), ['tgt', 'mem', 'causal4', 'kpm25'], 11)
layer('Decoder_layer_prenorm', lambda: nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0, norm_first=True, batch_first=True, activation='gelu'), lambda m, x, mem: m(x, mem).reshape(-1), ['xb', 'memb'], 12)
layer('Encoder', lambda: nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0), 2, norm=nn.LayerNorm(8), enable_nested_tensor=False), lambda m, x, msk: m(x, mask=msk).reshape(-1), ['src', 'causal6'], 13)
layer('Decoder', lambda: nn.TransformerDecoder(nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0), 2), lambda m, x, mem, msk: m(x, mem, tgt_mask=msk).reshape(-1), ['tgt', 'mem', 'causal4'], 14)
layer('Transformer', lambda: nn.Transformer(8, 2, num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=16, dropout=0.0), lambda m, s, t, msk: m(s, t, tgt_mask=msk).reshape(-1), ['src', 'tgt', 'causal4'], 15)
layer('Transformer_batch_first', lambda: nn.Transformer(8, 2, num_encoder_layers=1, num_decoder_layers=2, dim_feedforward=12, dropout=0.0, batch_first=True, norm_first=True), lambda m, s, t: m(s, t).reshape(-1), ['xb', 'memb'], 16)

# dropout drawn from the generator in training, after the parameters
layer('Dropout', lambda: nn.Dropout(0.3), lambda m, x: m(x).reshape(-1), ['xd'], 17, train=True)
layer('Dropout1d', lambda: nn.Dropout1d(0.5), lambda m, x: m(x).reshape(-1), ['xc3'], 18, train=True)
layer('Dropout2d', lambda: nn.Dropout2d(0.4), lambda m, x: m(x).reshape(-1), ['xc4'], 19, train=True)
layer('Dropout3d', lambda: nn.Dropout3d(0.5), lambda m, x: m(x).reshape(-1), ['xc5'], 20, train=True)
layer('Dropout_model', lambda: nn.Sequential(nn.Linear(40, 6), nn.Dropout(0.25), nn.ReLU(), nn.Linear(6, 3), nn.Dropout(0.5)), lambda m, x: m(x).reshape(-1), ['xd'], 21, train=True)
layer('MHA_dropout', lambda: nn.MultiheadAttention(8, 2, dropout=0.25), lambda m, x: attended(m, x, x, x), ['x482'], 22, train=True)
layer('MHA_dropout_sdpa', lambda: nn.MultiheadAttention(8, 2, dropout=0.25, batch_first=True), lambda m, x: attended(m, x, x, x, need_weights=False), ['xb'], 23, train=True)
layer('Encoder_layer_dropout', lambda: nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.2), lambda m, x, msk: m(x, src_mask=msk).reshape(-1), ['x482', 'causal4'], 24, train=True)
layer('Decoder_layer_dropout', lambda: nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.3, batch_first=True), lambda m, x, mem: m(x, mem).reshape(-1), ['xb', 'memb'], 25, train=True)
layer('Transformer_dropout', lambda: nn.Transformer(8, 2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=12, dropout=0.1), lambda m, s, t: m(s, t).reshape(-1), ['src', 'tgt'], 26, train=True)
layer('LSTM_dropout', lambda: nn.LSTM(3, 4, num_layers=3, dropout=0.5), lambda m, x: m(x)[0].reshape(-1), ['xr'], 27, train=True)
layer('GRU_dropout_bidirectional', lambda: nn.GRU(3, 4, num_layers=2, dropout=0.4, bidirectional=True, batch_first=True), lambda m, x: m(x)[0].reshape(-1), ['xrb'], 28, train=True)


def joined(xs):
    return ' '.join(repr(float(v)) for v in xs)


rows = []
for name, fn, run, names, seed, within, train in cases:
    leaves = [inputs[n].clone().requires_grad_(True) for n in names]
    theta = []
    if seed >= 0:
        torch.manual_seed(seed)
        module = fn()
        params = list(module.parameters())
        theta = [v for p in params for v in p.detach().reshape(-1).tolist()]
        # in training, as made; the cases without dropout run as made too
        module.train(True)
        value = run(module, *leaves)
        wrt = params + leaves
    else:
        value = fn(*leaves).reshape(-1)
        wrt = leaves
    s = (value * weights(value.numel())).sum()
    gs = torch.autograd.grad(s, wrt, allow_unused=True)
    row = theta + value.tolist()
    for g, w in zip(gs, wrt):
        row += (torch.zeros(w.numel()) if g is None else g.reshape(-1)).tolist()
    rows.append([name, ' '.join(names + ['none'] * (7 - len(names))), seed, repr(within), joined(row)])

os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
os.makedirs(os.path.join(HERE, 'expected'), exist_ok=True)
with open(os.path.join(HERE, 'data/at_inputs.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['name', 'dims', 'values'])
    for name, t in inputs.items():
        out.writerow([name, str(list(t.shape)), joined(t.reshape(-1).tolist())])
with open(os.path.join(HERE, 'expected/at_cases.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['case', 'inputs', 'seed', 'within', 'row'])
    for r in rows:
        out.writerow(r)
print('torch %s: %d cases, %d inputs' % (torch.__version__, len(rows), len(inputs)))

"""nn's convolutions, poolings, paddings, resamplings and recurrent layers
held to PyTorch (torch 2.8, float64).

Each functional case applies one of PyTorch's functions to fixed inputs;
each layer case makes one of PyTorch's modules, or a Sequential of them,
from torch.manual_seed and runs it. The script writes:

  data/sp_inputs.csv        every input: its name, its sizes and its
                            numbers (rounded to four decimals, so the same
                            double in both), joined by spaces
  expected/sp_cases.csv     each case: its name, its inputs (up to seven,
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

A recurrent layer's value is its output, then h_n, then c_n (an LSTM's).
held_spatial.co computes each row with pie and compares.

Run from pie's root: ~/.venvs/gpt/bin/python -I packages/nn/reference/spatial_reference.py
"""
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261109)
inputs = {}


def given(name, *dims, positive=False, scale=1.0):
    v = np.round(rng.normal(0.0, scale, dims), 4)
    if positive:
        v = np.round(np.abs(v) + 0.1, 4)
    inputs[name] = torch.tensor(v)


def weights(n):
    return torch.tensor([float((k * 7) % 11 - 5) / 4.0 + 0.125 for k in range(n)])


given('x1d', 2, 3, 7)
given('w1d', 4, 3, 3)
given('b4', 4)
given('x1g', 2, 4, 9)
given('w1g', 6, 2, 3)
given('b6', 6)
given('x2d', 2, 3, 5, 6)
given('w2d', 4, 3, 3, 3)
given('w2e', 4, 3, 3, 2)
given('x2g', 1, 4, 5, 5)
given('w2g', 6, 2, 3, 3)
given('x3d', 1, 2, 4, 4, 4)
given('w3d', 3, 2, 2, 3, 2)
given('b3', 3)
given('xt1', 2, 3, 5)
given('wt1', 3, 2, 3)
given('b2', 2)
given('xt2', 2, 4, 3, 4)
given('wt2', 4, 3, 3, 2)
given('xt3', 1, 2, 2, 3, 2)
given('wt3', 2, 2, 2, 2, 2)
given('xp1', 2, 3, 8)
given('xp2', 2, 3, 7, 6)
given('xp3', 1, 2, 5, 4, 4)
given('xpos', 2, 3, 7, 6, positive=True)
given('xpad', 2, 3, 4, 5)
given('xpad3', 1, 2, 3, 4, 4)
given('xps', 2, 8, 3, 4)
given('xpu', 2, 2, 6, 4)
given('xcs', 2, 6, 3, 2)
given('xi2', 2, 3, 4, 5)
given('xi3', 1, 2, 3, 4, 3)
given('xc', 3, 4)
given('hc', 3, 5)
given('cc', 3, 5)
given('wih1', 5, 4, scale=0.5)
given('whh1', 5, 5, scale=0.5)
given('bih1', 5)
given('bhh1', 5)
given('wih4', 20, 4, scale=0.5)
given('whh4', 20, 5, scale=0.5)
given('bih4', 20)
given('bhh4', 20)
given('wih3', 15, 4, scale=0.5)
given('whh3', 15, 5, scale=0.5)
given('bih3', 15)
given('bhh3', 15)
given('xr', 2, 4, 3)
given('h0r', 4, 2, 5)
given('xu', 5, 3)
given('xl', 4, 2, 3)
given('h0l', 4, 2, 2)
given('c0l', 4, 2, 5)
given('xs', 5, 2, 3)
given('xg', 2, 5, 3)
given('h0g', 2, 2, 4)
given('xcnn', 2, 3, 5, 6)
given('x1s', 2, 2, 9)
# zeros where a layer starts from PyTorch's default state
inputs['zh4'] = torch.zeros(1, 2, 4)
inputs['zh5'] = torch.zeros(3, 5)
inputs['zu4'] = torch.zeros(1, 4)

cases = []


def case(name, fn, names, within=1e-12):
    cases.append((name, fn, names, -1, within))


def layer(name, make, names, seed, within=1e-12):
    cases.append((name, make, names, seed, within))


# convolutions: pie gathers the windows and multiplies by the weight as one
# matrix, as PyTorch's float64 kernel does, its sums in another order
case('conv1d', lambda x, w, b: F.conv1d(x, w, b, stride=2, padding=1), ['x1d', 'w1d', 'b4'])
case('conv1d_groups', lambda x, w, b: F.conv1d(x, w, b, padding='same', dilation=2, groups=2), ['x1g', 'w1g', 'b6'])
case('conv2d', lambda x, w, b: F.conv2d(x, w, b, stride=(2, 1), padding=1), ['x2d', 'w2d', 'b4'])
case('conv2d_same', lambda x, w, b: F.conv2d(x, w, b, padding='same', dilation=(1, 2)), ['x2d', 'w2e', 'b4'])
case('conv2d_groups', lambda x, w: F.conv2d(x, w, groups=2), ['x2g', 'w2g'])
case('conv3d', lambda x, w, b: F.conv3d(x, w, b, stride=(1, 2, 1), padding=(1, 0, 1)), ['x3d', 'w3d', 'b3'])
case('conv_transpose1d', lambda x, w, b: F.conv_transpose1d(x, w, b, stride=2, padding=1, output_padding=1), ['xt1', 'wt1', 'b2'])
case('conv_transpose2d', lambda x, w, b: F.conv_transpose2d(x, w, b, stride=(2, 1), padding=(1, 0), output_padding=(1, 0), groups=2, dilation=(1, 2)), ['xt2', 'wt2', 'b6'])
case('conv_transpose3d', lambda x, w: F.conv_transpose3d(x, w, stride=2), ['xt3', 'wt3'])
case('unfold', lambda x: F.unfold(x, (2, 3), padding=1, stride=(2, 1)), ['x2d'])
case('fold', lambda x: F.fold(F.unfold(x, (2, 3), padding=1, stride=(2, 1)), (5, 6), (2, 3), padding=1, stride=(2, 1)), ['x2d'])
# poolings
case('max_pool1d', lambda x: F.max_pool1d(x, 3, stride=2, padding=1), ['xp1'])
case('max_pool2d', lambda x: F.max_pool2d(x, (3, 2), stride=(2, 1), padding=(1, 0), dilation=(1, 2), ceil_mode=True), ['xp2'])
case('max_pool3d', lambda x: F.max_pool3d(x, 2), ['xp3'])
case('avg_pool1d', lambda x: F.avg_pool1d(x, 3, stride=2, padding=1, ceil_mode=True, count_include_pad=False), ['xp1'])
case('avg_pool2d', lambda x: F.avg_pool2d(x, 3, stride=2, padding=1, ceil_mode=True), ['xp2'])
case('avg_pool2d_divisor', lambda x: F.avg_pool2d(x, 3, stride=2, padding=1, divisor_override=4), ['xp2'])
case('avg_pool3d', lambda x: F.avg_pool3d(x, (2, 3, 2), stride=(2, 1, 2), padding=(1, 1, 0)), ['xp3'])
case('adaptive_avg_pool1d', lambda x: F.adaptive_avg_pool1d(x, 3), ['x1d'])
case('adaptive_avg_pool2d', lambda x: F.adaptive_avg_pool2d(x, (3, 4)), ['xp2'])
case('adaptive_avg_pool3d', lambda x: F.adaptive_avg_pool3d(x, (2, 3, 3)), ['xp3'])
case('adaptive_max_pool1d', lambda x: F.adaptive_max_pool1d(x, 4), ['x1d'])
case('adaptive_max_pool2d', lambda x: F.adaptive_max_pool2d(x, (3, 4)), ['xp2'])
case('lp_pool1d', lambda x: F.lp_pool1d(x, 2.0, 3, stride=2), ['xp1'])
case('lp_pool2d', lambda x: F.lp_pool2d(x, 3.0, 2), ['xpos'])
# paddings: a negative width cuts
case('pad_constant', lambda x: F.pad(x, (1, 2, 0, 1), value=0.5), ['xpad'])
case('pad_cut', lambda x: F.pad(x, (-1, 1, 2, -1)), ['xpad'])
case('pad_reflect', lambda x: F.pad(x, (2, 1, 1, 3), mode='reflect'), ['xpad'])
case('pad_replicate', lambda x: F.pad(x, (2, 1, 3, 0), mode='replicate'), ['xpad'])
case('pad_circular', lambda x: F.pad(x, (1, 2, 2, 1), mode='circular'), ['xpad'])
case('pad_reflect1d', lambda x: F.pad(x, (2, 3), mode='reflect'), ['xp1'])
case('pad_replicate3d', lambda x: F.pad(x, (1, 1, 2, 0, 1, 1), mode='replicate'), ['xpad3'])
# rearrangements
case('pixel_shuffle', lambda x: F.pixel_shuffle(x, 2), ['xps'])
case('pixel_unshuffle', lambda x: F.pixel_unshuffle(x, 2), ['xpu'])
case('channel_shuffle', lambda x: F.channel_shuffle(x, 3), ['xcs'])
# resamplings: nearest reads the position PyTorch computes in single precision
case('nearest', lambda x: F.interpolate(x, size=(7, 3)), ['xi2'])
case('nearest_double', lambda x: F.interpolate(x, scale_factor=2.0), ['xi2'])
case('nearest_scale', lambda x: F.interpolate(x, scale_factor=1.7), ['xi2'])
case('nearest_exact', lambda x: F.interpolate(x, size=(6, 8), mode='nearest-exact'), ['xi2'])
case('linear', lambda x: F.interpolate(x, size=8, mode='linear'), ['x1d'])
case('linear_corners', lambda x: F.interpolate(x, size=8, mode='linear', align_corners=True), ['x1d'])
case('bilinear', lambda x: F.interpolate(x, size=(7, 8), mode='bilinear'), ['xi2'])
case('bilinear_scale', lambda x: F.interpolate(x, scale_factor=(1.5, 2.3), mode='bilinear', align_corners=True), ['xi2'])
case('trilinear', lambda x: F.interpolate(x, size=(5, 6, 4), mode='trilinear'), ['xi3'])
case('area', lambda x: F.interpolate(x, size=(3, 4), mode='area'), ['xp2'])
# one step of each recurrent cell
case('rnn_tanh_cell', lambda x, h, wi, wh, bi, bh: torch.rnn_tanh_cell(x, h, wi, wh, bi, bh), ['xc', 'hc', 'wih1', 'whh1', 'bih1', 'bhh1'])
case('rnn_relu_cell', lambda x, h, wi, wh, bi, bh: torch.rnn_relu_cell(x, h, wi, wh, bi, bh), ['xc', 'hc', 'wih1', 'whh1', 'bih1', 'bhh1'])
case('lstm_cell', lambda x, h, c, wi, wh, bi, bh: torch.cat([t.reshape(-1) for t in torch.lstm_cell(x, (h, c), wi, wh, bi, bh)]), ['xc', 'hc', 'cc', 'wih4', 'whh4', 'bih4', 'bhh4'])
case('gru_cell', lambda x, h, wi, wh, bi, bh: torch.gru_cell(x, h, wi, wh, bi, bh), ['xc', 'hc', 'wih3', 'whh3', 'bih3', 'bhh3'])

# layers, their parameters drawn by torch's generator as PyTorch's modules
# draw them
layer('Conv1d', lambda: nn.Conv1d(3, 4, 3, stride=2, padding=1), ['x1d'], 1)
layer('Conv2d_reflect', lambda: nn.Conv2d(3, 4, (3, 2), padding=1, padding_mode='reflect'), ['x2d'], 2)
layer('Conv2d_circular_same', lambda: nn.Conv2d(4, 6, 3, padding='same', padding_mode='circular', groups=2, bias=False), ['x2g'], 3)
layer('Conv3d_replicate', lambda: nn.Conv3d(2, 3, 2, padding=1, padding_mode='replicate'), ['x3d'], 4)
layer('ConvTranspose2d', lambda: nn.ConvTranspose2d(4, 6, (3, 2), stride=(2, 1), padding=(1, 0), output_padding=(1, 0), groups=2, dilation=(1, 2)), ['xt2'], 5)
layer('ConvTranspose1d', lambda: nn.ConvTranspose1d(3, 2, 3, stride=2, padding=1, output_padding=1), ['xt1'], 6)
layer('RNN_bidirectional', lambda: nn.RNN(3, 5, num_layers=2, bidirectional=True, batch_first=True), ['xr', 'h0r'], 7)
layer('RNN_relu_unbatched', lambda: nn.RNN(3, 4, nonlinearity='relu'), ['xu', 'zu4'], 8)
layer('LSTM_projected', lambda: nn.LSTM(3, 5, num_layers=2, proj_size=2, bidirectional=True), ['xl', 'h0l', 'c0l'], 9)
layer('LSTM', lambda: nn.LSTM(3, 4), ['xs', 'zh4', 'zh4'], 10)
layer('GRU_batch_first', lambda: nn.GRU(3, 4, num_layers=2, batch_first=True), ['xg', 'h0g'], 11)
layer('GRU_no_bias', lambda: nn.GRU(3, 4, bias=False), ['xl', 'zh4'], 12)
layer('RNNCell', lambda: nn.RNNCell(4, 5), ['xc', 'hc'], 13)
layer('LSTMCell', lambda: nn.LSTMCell(4, 5), ['xc', 'hc', 'cc'], 14)
layer('GRUCell', lambda: nn.GRUCell(4, 5), ['xc', 'zh5'], 15)
layer('CNN', lambda: nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(4 * 2 * 3, 5)), ['xcnn'], 16)
layer('Windows', lambda: nn.Sequential(nn.ReflectionPad2d((1, 0, 2, 1)), nn.Conv2d(3, 8, 3), nn.PixelShuffle(2), nn.AdaptiveAvgPool2d((3, 4)), nn.Upsample(scale_factor=2, mode='bilinear'), nn.ZeroPad2d(1), nn.AvgPool2d(2), nn.Unflatten(1, (1, 2))), ['x2d'], 17)
layer('Sequence', lambda: nn.Sequential(nn.Conv1d(2, 4, 3), nn.LPPool1d(2, 2), nn.Upsample(scale_factor=1.5, mode='linear'), nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False), nn.ConstantPad1d((1, 2), 0.25), nn.ConvTranspose1d(4, 2, 2, stride=2), nn.AdaptiveMaxPool1d(5)), ['x1s'], 18)


def flat(value):
    return value.reshape(-1)


def joined(xs):
    return ' '.join(repr(float(v)) for v in xs)


rows = []
for name, fn, names, seed, within in cases:
    leaves = [inputs[n].clone().requires_grad_(True) for n in names]
    theta = []
    if seed >= 0:
        torch.manual_seed(seed)
        module = fn()
        params = list(module.parameters())
        theta = [v for p in params for v in p.detach().reshape(-1).tolist()]
        # a recurrent layer's value as pie's nn.recurrent answers it: the
        # output, h_n, then c_n (a cell's output is its h_n)
        if isinstance(module, nn.LSTM):
            out, (h, c) = module(leaves[0], (leaves[1], leaves[2]))
            value = torch.cat([out.reshape(-1), h.reshape(-1), c.reshape(-1)])
        elif isinstance(module, nn.LSTMCell):
            h, c = module(leaves[0], (leaves[1], leaves[2]))
            value = torch.cat([h.reshape(-1), h.reshape(-1), c.reshape(-1)])
        elif isinstance(module, nn.RNNCellBase):
            h = module(leaves[0], leaves[1])
            value = torch.cat([h.reshape(-1), h.reshape(-1)])
        elif isinstance(module, nn.RNNBase):
            out, h = module(leaves[0], leaves[1])
            value = torch.cat([out.reshape(-1), h.reshape(-1)])
        else:
            value = module(leaves[0]).reshape(-1)
        wrt = params + leaves
    else:
        value = flat(fn(*leaves))
        wrt = leaves
    s = (value * weights(value.numel())).sum()
    gs = torch.autograd.grad(s, wrt, allow_unused=True)
    row = theta + value.tolist()
    for g, w in zip(gs, wrt):
        row += (torch.zeros(w.numel()) if g is None else g.reshape(-1)).tolist()
    rows.append([name, ' '.join(names + ['none'] * (7 - len(names))), seed, repr(within), joined(row)])

os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
os.makedirs(os.path.join(HERE, 'expected'), exist_ok=True)
with open(os.path.join(HERE, 'data/sp_inputs.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['name', 'dims', 'values'])
    for name, t in inputs.items():
        out.writerow([name, str(list(t.shape)), joined(t.reshape(-1).tolist())])
with open(os.path.join(HERE, 'expected/sp_cases.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['case', 'inputs', 'seed', 'within', 'row'])
    for r in rows:
        out.writerow(r)
print('torch %s: %d cases, %d inputs' % (torch.__version__, len(rows), len(inputs)))

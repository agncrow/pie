"""nn.functional held to PyTorch's torch.nn.functional, in float64.

Each case is one of PyTorch's functions applied to fixed inputs. The script
writes the inputs (data/<name>.csv, each a matrix: a vector is one row, a
tensor of three axes its rows laid one after another), the list of cases
(expected/cases.csv: the case, whether it answers a tensor or a number, the
inputs it takes, how many of them have their gradient compared and the
distance its row may be from pie's, 1e-12 unless the case says why), and one
row per case (expected/<case>.csv): the numbers of the value, then the
gradient of sum(value * c) -- of the value itself where it is a number --
with respect to each compared input, c the weights below (whole eighths,
the same double in both). held.co computes each row with pie and compares.

The constants a case reads beside its inputs -- class indices, class
weights, running statistics, labels of 1 and -1, embedding ids -- are inputs
too (data/), handed to held.co's cases as one record.

Run: ~/.venvs/gpt/bin/python -I packages/nn/reference/reference.py (torch 2.8)
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))


def write(path, rows):
    # a header row: Crow's loaders read the first row as the names
    path = os.path.join(HERE, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [list(r) for r in rows]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['c%d' % i for i in range(len(rows[0]))])
        for r in rows:
            w.writerow([repr(float(x)) if isinstance(x, (float, np.floating)) else x for x in r])


rng = np.random.default_rng(20261107)


def normal(r, c, scale=1.0):
    return np.round(rng.normal(0.0, scale, (r, c)), 4)


def uniform(r, c, lo, hi):
    return np.round(rng.uniform(lo, hi, (r, c)), 4)


data = {}
# the kinks of the activations: 0, ±0.5, ±1, ±3, 6
data['a'] = np.array([[-4.0, -3.0, -1.0, -0.5], [-0.25, 0.0, 0.25, 0.5], [1.0, 3.0, 6.0, 7.5]])
# far out, where an exponential overflows
data['ext'] = np.array([[-800.0, -50.0, -20.0, -0.001], [0.001, 20.0, 50.0, 800.0]])
data['b'] = normal(4, 5, 1.5)
data['b2'] = normal(4, 5, 1.5)
data['b3'] = normal(4, 5, 1.5)
data['b6'] = normal(3, 6, 1.5)
data['big'] = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]])
zrow = normal(4, 5, 1.5)
zrow[2] = 0.0
data['zrow'] = zrow
data['p'] = uniform(4, 5, 0.05, 0.95)
data['soft'] = uniform(4, 5, 0.0, 1.0)
data['bits'] = (rng.uniform(size=(4, 5)) < 0.5).astype(float)
data['pm45'] = np.where(rng.uniform(size=(4, 5)) < 0.5, -1.0, 1.0)
data['pm4'] = np.array([[1.0, -1.0, -1.0, 1.0]])
data['x4'] = normal(1, 4)
data['y4'] = normal(1, 4)
data['cls'] = np.array([[1.0, 0.0, 4.0, 2.0]])
data['ign'] = np.array([[1.0, -100.0, 4.0, 2.0]])
data['cls1'] = np.array([[3.0]])
data['w5'] = np.array([[0.5, 1.0, 2.0, 0.25, 1.5]])
data['x5'] = normal(1, 5, 1.5)
data['img'] = normal(6, 4, 1.5)  # [2, 3, 4]
data['w3'] = normal(1, 3, 0.5) + 1.0
data['b3c'] = normal(1, 3, 0.5)
data['rm'] = np.array([[0.1, -0.2, 0.3]])
data['rv'] = np.array([[0.5, 1.5, 2.0]])
data['g'] = normal(8, 3, 1.5)  # [2, 4, 3]
data['w4'] = normal(1, 4, 0.5) + 1.0
data['b4'] = normal(1, 4, 0.5)
data['wl5'] = normal(1, 5, 0.5) + 1.0
data['bl5'] = normal(1, 5, 0.5)
data['wl34'] = normal(3, 4, 0.5) + 1.0
data['bl34'] = normal(3, 4, 0.5)
data['wlin'] = normal(3, 5, 0.5)
data['blin'] = normal(1, 3, 0.5)
data['x3d'] = normal(8, 5)  # [2, 4, 5]
data['x2'] = normal(4, 3)
data['wbil'] = normal(10, 3, 0.5)  # [2, 5, 3]
data['bbil'] = normal(1, 2, 0.5)
data['emb'] = normal(6, 3)
data['ids'] = np.array([[0.0, 5.0, 2.0], [2.0, 1.0, 4.0]])
data['counts'] = rng.poisson(2.0, (4, 5)).astype(float)
data['var'] = uniform(4, 5, 0.1, 2.0)
data['wone'] = np.array([[0.25]])
data['wch'] = normal(1, 5, 0.3)

for name, m in data.items():
    write('data/%s.csv' % name, m.tolist())


def t(name):
    return torch.tensor(data[name], dtype=torch.float64)


cls = t('cls').flatten().long()
ign = t('ign').flatten().long()
cls1 = t('cls1').flatten().long()[0]
w5 = t('w5').flatten()
rm = t('rm').flatten()
rv = t('rv').flatten()
pm4 = t('pm4').flatten()
pm45 = t('pm45')
ids = t('ids').long()

cases = []


def case(name, fn, inputs, grads=None, number=False, within=1e-12):
    cases.append((name, fn, inputs, len(inputs) if grads is None else grads, number, within))


def loss(name, fn, inputs, grads=None):
    case(name, fn, inputs, grads, number=True)


def img(x):
    return x.reshape(2, 3, 4)


# activations
case('relu', lambda x: F.relu(x), ['a'])
case('relu6', lambda x: F.relu6(x), ['a'])
case('hardtanh', lambda x: F.hardtanh(x), ['a'])
case('hardtanh_half', lambda x: F.hardtanh(x, -0.5, 0.5), ['a'])
case('leaky_relu', lambda x: F.leaky_relu(x), ['a'])
case('leaky_relu_02', lambda x: F.leaky_relu(x, 0.2), ['a'])
case('elu', lambda x: F.elu(x), ['a'])
case('elu_2', lambda x: F.elu(x, 2.0), ['a'])
case('elu_ext', lambda x: F.elu(x), ['ext'])
case('selu', lambda x: F.selu(x), ['a'])
case('celu', lambda x: F.celu(x), ['a'])
case('celu_2', lambda x: F.celu(x, 2.0), ['a'])
case('gelu', lambda x: F.gelu(x), ['a'])
case('gelu_tanh', lambda x: F.gelu(x, approximate='tanh'), ['a'])
case('gelu_ext', lambda x: F.gelu(x), ['ext'])
case('sigmoid', lambda x: torch.sigmoid(x), ['a'])
case('sigmoid_ext', lambda x: torch.sigmoid(x), ['ext'])
case('silu', lambda x: F.silu(x), ['a'])
case('silu_ext', lambda x: F.silu(x), ['ext'])
case('tanh', lambda x: torch.tanh(x), ['b'])
case('tanh_ext', lambda x: torch.tanh(x), ['ext'])
case('softsign', lambda x: F.softsign(x), ['a'])
case('softplus', lambda x: F.softplus(x), ['b'])
case('softplus_2_5', lambda x: F.softplus(x, 2.0, 5.0), ['a'])
case('softplus_ext', lambda x: F.softplus(x), ['ext'])
case('mish', lambda x: F.mish(x), ['a'])
case('mish_ext', lambda x: F.mish(x), ['ext'])
case('logsigmoid', lambda x: F.logsigmoid(x), ['a'])
case('logsigmoid_ext', lambda x: F.logsigmoid(x), ['ext'])
case('tanhshrink', lambda x: F.tanhshrink(x), ['b'])
case('softshrink', lambda x: F.softshrink(x), ['a'])
case('softshrink_1', lambda x: F.softshrink(x, 1.0), ['a'])
case('hardshrink', lambda x: F.hardshrink(x), ['a'])
case('hardshrink_1', lambda x: F.hardshrink(x, 1.0), ['a'])
case('threshold', lambda x: F.threshold(x, 0.5, -2.0), ['a'])
# (PyTorch's CPU backward multiplies by 1.0f / 6.0f, a sixth rounded to
# float32 and 3e-8 from the slope; pie's gradient is the slope, 1/6)
case('hardsigmoid', lambda x: F.hardsigmoid(x), ['a'], within=1e-7)
case('hardswish', lambda x: F.hardswish(x), ['a'])
case('prelu_one', lambda x, w: F.prelu(x, w.flatten()), ['a', 'wone'])
case('prelu_channels', lambda x, w: F.prelu(x, w.flatten()), ['b', 'wch'])
case('narrow', lambda x: torch.narrow(x, 1, 1, 3), ['b'])
case('narrow_0', lambda x: torch.narrow(x, 0, 1, 2), ['b'])
case('glu', lambda x: F.glu(x), ['b6'])
case('glu_0', lambda x: F.glu(x, 0), ['b'])
# softmax and its kin
case('softmax', lambda x: F.softmax(x, -1), ['b'])
case('softmax_0', lambda x: F.softmax(x, 0), ['b'])
case('softmax_big', lambda x: F.softmax(x, -1), ['big'])
case('softmax_3d', lambda x: F.softmax(img(x), 1), ['img'])
case('log_softmax', lambda x: F.log_softmax(x, -1), ['b'])
case('log_softmax_0', lambda x: F.log_softmax(x, 0), ['b'])
case('log_softmax_big', lambda x: F.log_softmax(x, -1), ['big'])
case('softmin', lambda x: F.softmin(x, -1), ['b'])
case('normalize', lambda x: F.normalize(x), ['b'])
case('normalize_p1', lambda x: F.normalize(x, p=1.0), ['b'])
case('normalize_p3', lambda x: F.normalize(x, p=3.0), ['b'])
case('normalize_0', lambda x: F.normalize(x, dim=0), ['b'])
case('normalize_zero', lambda x: F.normalize(x), ['zrow'])
# layers as functions
case('linear', lambda x, w, b: F.linear(x, w, b.flatten()), ['b', 'wlin', 'blin'])
case('linear_nobias', lambda x, w: F.linear(x, w), ['b', 'wlin'])
case('linear_3d', lambda x, w, b: F.linear(x.reshape(2, 4, 5), w, b.flatten()), ['x3d', 'wlin', 'blin'])
case('bilinear', lambda x1, x2, w, b: F.bilinear(x1, x2, w.reshape(2, 5, 3), b.flatten()), ['b', 'x2', 'wbil', 'bbil'])
case('embedding', lambda w: F.embedding(ids, w), ['emb'])
case('one_hot', lambda: F.one_hot(ids, 6).double(), [])
case('dropout_off', lambda x: F.dropout(x, 0.5, training=False), ['b'])
case('dropout_none', lambda x: F.dropout(x, 0.0), ['b'])
# normalization
case('layer_norm', lambda x: F.layer_norm(x, [5]), ['b'])
case('layer_norm_affine', lambda x, w, b: F.layer_norm(x, [5], w.flatten(), b.flatten()), ['b', 'wl5', 'bl5'])
case('layer_norm_2d', lambda x, w, b: F.layer_norm(img(x), [3, 4], w, b), ['img', 'wl34', 'bl34'])
case('rms_norm', lambda x: F.rms_norm(x, [5]), ['b'])
case('rms_norm_w', lambda x, w: F.rms_norm(x, [5], w.flatten()), ['b', 'wl5'])
case('rms_norm_eps', lambda x: F.rms_norm(x, [5], eps=1e-5), ['b'])
case('batch_norm_train', lambda x, w, b: F.batch_norm(img(x), None, None, w.flatten(), b.flatten(), training=True), ['img', 'w3', 'b3c'])
case('batch_norm_eval', lambda x, w, b: F.batch_norm(img(x), rm, rv, w.flatten(), b.flatten()), ['img', 'w3', 'b3c'])
case('batch_norm_2d', lambda x: F.batch_norm(x, None, None, training=True), ['b'])
case('group_norm', lambda x, w, b: F.group_norm(x.reshape(2, 4, 3), 2, w.flatten(), b.flatten()), ['g', 'w4', 'b4'])
case('instance_norm', lambda x: F.instance_norm(img(x)), ['img'])
case('instance_norm_affine', lambda x, w, b: F.instance_norm(img(x), weight=w.flatten(), bias=b.flatten()), ['img', 'w3', 'b3c'])
case('cosine_similarity', lambda x, y: F.cosine_similarity(x, y), ['b', 'b2'])
case('cosine_similarity_0', lambda x, y: F.cosine_similarity(x, y, 0), ['b', 'b2'])
case('pairwise_distance', lambda x, y: F.pairwise_distance(x, y), ['b', 'b2'])
case('pairwise_distance_p1', lambda x, y: F.pairwise_distance(x, y, 1.0), ['b', 'b2'])
# the losses, each term
case('mse_each', lambda x, y: F.mse_loss(x, y, reduction='none'), ['b', 'b2'])
case('bce_each', lambda x, y: F.binary_cross_entropy(x, y, reduction='none'), ['p', 'soft'])
case('nll_each', lambda x: F.nll_loss(x, ign, w5, reduction='none'), ['b'])
case('ce_each', lambda x: F.cross_entropy(x, ign, w5, reduction='none', label_smoothing=0.1), ['b'])
case('ce_prob_each', lambda x, y: F.cross_entropy(x, F.softmax(y, 1), w5, reduction='none', label_smoothing=0.2), ['b', 'b2'])
case('kl_each', lambda x, y: F.kl_div(F.log_softmax(x, 1), F.softmax(y, 1), reduction='none'), ['b', 'b2'])
case('multi_margin_each', lambda x: F.multi_margin_loss(x, cls, reduction='none'), ['b'])
case('triplet_each', lambda a, p, n: F.triplet_margin_loss(a, p, n, reduction='none'), ['b', 'b2', 'b3'])
case('cosine_embedding_each', lambda x, y: F.cosine_embedding_loss(x, y, pm4, 0.2, reduction='none'), ['b', 'b2'])
# the losses
loss('mse', lambda x, y: F.mse_loss(x, y), ['b', 'b2'])
loss('mse_sum', lambda x, y: F.mse_loss(x, y, reduction='sum'), ['b', 'b2'])
loss('l1', lambda x, y: F.l1_loss(x, y), ['b', 'b2'])
loss('l1_sum', lambda x, y: F.l1_loss(x, y, reduction='sum'), ['b', 'b2'])
loss('smooth_l1', lambda x, y: F.smooth_l1_loss(x, y), ['b', 'b2'])
loss('smooth_l1_half', lambda x, y: F.smooth_l1_loss(x, y, beta=0.5), ['b', 'b2'])
loss('smooth_l1_0', lambda x, y: F.smooth_l1_loss(x, y, beta=0.0), ['b', 'b2'])
loss('huber', lambda x, y: F.huber_loss(x, y), ['b', 'b2'])
loss('huber_half', lambda x, y: F.huber_loss(x, y, delta=0.5), ['b', 'b2'])
loss('bce', lambda x, y: F.binary_cross_entropy(x, y), ['p', 'soft'])
loss('bce_w', lambda x, y: F.binary_cross_entropy(x, y, w5), ['p', 'soft'])
loss('bce_sum', lambda x, y: F.binary_cross_entropy(x, y, reduction='sum'), ['p', 'soft'])
loss('bce_logits', lambda x, y: F.binary_cross_entropy_with_logits(x, y), ['b', 'soft'])
loss('bce_logits_w', lambda x, y: F.binary_cross_entropy_with_logits(x, y, w5, pos_weight=w5 * 2.0), ['b', 'soft'])
loss('nll', lambda x: F.nll_loss(x, cls), ['b'])
loss('nll_w', lambda x: F.nll_loss(x, cls, w5), ['b'])
loss('nll_ign', lambda x: F.nll_loss(x, ign), ['b'])
loss('nll_ign_w', lambda x: F.nll_loss(x, ign, w5), ['b'])
loss('nll_sum', lambda x: F.nll_loss(x, ign, w5, reduction='sum'), ['b'])
loss('ce', lambda x: F.cross_entropy(x, cls), ['b'])
loss('ce_w', lambda x: F.cross_entropy(x, cls, w5), ['b'])
loss('ce_ign', lambda x: F.cross_entropy(x, ign), ['b'])
loss('ce_ls', lambda x: F.cross_entropy(x, cls, label_smoothing=0.1), ['b'])
loss('ce_ls_w_ign', lambda x: F.cross_entropy(x, ign, w5, label_smoothing=0.1), ['b'])
loss('ce_sum', lambda x: F.cross_entropy(x, ign, w5, reduction='sum', label_smoothing=0.1), ['b'])
loss('ce_prob', lambda x, y: F.cross_entropy(x, F.softmax(y, 1)), ['b', 'b2'])
loss('ce_prob_ls', lambda x, y: F.cross_entropy(x, F.softmax(y, 1), label_smoothing=0.2), ['b', 'b2'])
loss('ce_prob_w', lambda x, y: F.cross_entropy(x, F.softmax(y, 1), w5), ['b', 'b2'])
loss('ce_one', lambda x: F.cross_entropy(x.flatten(), cls1), ['x5'])
loss('kl', lambda x, y: F.kl_div(F.log_softmax(x, 1), F.softmax(y, 1), reduction='batchmean'), ['b', 'b2'])
loss('kl_mean', lambda x, y: F.kl_div(F.log_softmax(x, 1), F.softmax(y, 1), reduction='mean'), ['b', 'b2'])
loss('kl_sum', lambda x, y: F.kl_div(F.log_softmax(x, 1), F.softmax(y, 1), reduction='sum'), ['b', 'b2'])
loss('kl_log_target', lambda x, y: F.kl_div(F.log_softmax(x, 1), F.log_softmax(y, 1), reduction='batchmean', log_target=True), ['b', 'b2'])
loss('poisson', lambda x, y: F.poisson_nll_loss(x, y), ['b', 'counts'])
# (the target's gradient where Stirling's term is masked is NaN in PyTorch:
# ln(0) behind the mask; only the input's is compared)
loss('poisson_full', lambda x, y: F.poisson_nll_loss(x, y, full=True), ['b', 'counts'], grads=1)
loss('poisson_rates', lambda x, y: F.poisson_nll_loss(x, y, log_input=False), ['p', 'counts'])
loss('gaussian', lambda x, y, v: F.gaussian_nll_loss(x, y, v), ['b', 'b2', 'var'])
loss('gaussian_full', lambda x, y, v: F.gaussian_nll_loss(x, y, v, full=True), ['b', 'b2', 'var'])
loss('margin_ranking', lambda x, y: F.margin_ranking_loss(x.flatten(), y.flatten(), pm4, 0.5), ['x4', 'y4'])
loss('hinge_embedding', lambda x: F.hinge_embedding_loss(x, pm45), ['b'])
loss('hinge_embedding_2', lambda x: F.hinge_embedding_loss(x, pm45, 2.0), ['b'])
loss('soft_margin', lambda x: F.soft_margin_loss(x, pm45), ['b'])
loss('multilabel_soft_margin', lambda x, y: F.multilabel_soft_margin_loss(x, y), ['b', 'bits'])
loss('multilabel_soft_margin_w', lambda x, y: F.multilabel_soft_margin_loss(x, y, w5), ['b', 'bits'])
loss('cosine_embedding', lambda x, y: F.cosine_embedding_loss(x, y, pm4, 0.2), ['b', 'b2'])
loss('triplet', lambda a, p, n: F.triplet_margin_loss(a, p, n), ['b', 'b2', 'b3'])
loss('triplet_swap', lambda a, p, n: F.triplet_margin_loss(a, p, n, swap=True), ['b', 'b2', 'b3'])
loss('triplet_p1', lambda a, p, n: F.triplet_margin_loss(a, p, n, p=1.0), ['b', 'b2', 'b3'])
loss('multi_margin', lambda x: F.multi_margin_loss(x, cls), ['b'])
loss('multi_margin_p2', lambda x: F.multi_margin_loss(x, cls, p=2), ['b'])
loss('multi_margin_w', lambda x: F.multi_margin_loss(x, cls, margin=0.5, weight=w5), ['b'])


def weights(n):
    return torch.tensor([float((k * 7) % 11 - 5) / 4.0 + 0.125 for k in range(n)])


listed = []
for name, fn, inputs, grads, number, within in cases:
    leaves = [t(i).requires_grad_(True) for i in inputs]
    value = fn(*leaves)
    flat = value.reshape(-1)
    s = value if number else (flat * weights(flat.numel())).sum()
    row = [float(v) for v in flat.tolist()]
    if grads > 0:
        gs = torch.autograd.grad(s, leaves[:grads], allow_unused=True)
        for leaf, g in zip(leaves, gs):
            g = torch.zeros_like(leaf) if g is None else g
            row += [float(v) for v in g.reshape(-1).tolist()]
    write('expected/%s.csv' % name, [row])
    padded = (inputs + ['none'] * 4)[:4]
    listed.append([name, 'number' if number else 'tensor'] + padded + [grads, repr(within)])

with open(os.path.join(HERE, 'expected/cases.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['case', 'answers', 'x', 'y', 'z', 'u', 'grads', 'within'])
    w.writerows(listed)


# ---- the layers, as models: nn's records over one theta ------------------
#
# Each model's parameters are drawn here (a fixed generator, four places)
# and written in PyTorch's order of parameters (data/theta_<case>.csv), the
# order nn.named lays theta in. The model runs on its inputs in training or
# evaluation, after a batch run in training first where the case says
# (moving a batch norm's buffers); its row (expected/model_<case>.csv) is
# the output's numbers, the gradient of sum(output * c) with respect to the
# parameters and to each input of numbers, then each batch norm's running
# mean, running variance and count of batches after the run, in the order
# the model's modules come. expected/models.csv lists the cases with the
# parameters' names and sizes, as named_parameters gives them.
import torch.nn as tnn


class Pair(tnn.Module):
    # nn.module[map[dec = ..., enc = ...]]: parts by name, which pie lays in
    # the names' order, so they are registered here in that order
    def __init__(self):
        super().__init__()
        self.dec = tnn.Linear(4, 3)
        self.enc = tnn.Linear(5, 4)

    def forward(self, x):
        return self.dec(F.relu(self.enc(x)))


models = []


def model(name, make, inputs, pre='none', mode='train'):
    models.append((name, make, inputs, pre, mode))


def batch_norm_net():
    return tnn.Sequential(tnn.Linear(5, 4), tnn.BatchNorm1d(4), tnn.Tanh(), tnn.Linear(4, 2))


model('mlp', lambda: tnn.Sequential(tnn.Linear(5, 4), tnn.ReLU(), tnn.LayerNorm(4), tnn.Linear(4, 3)), ['b'])
model('mlp_nobias', lambda: tnn.Sequential(tnn.Linear(5, 4, bias=False), tnn.GELU(approximate='tanh'), tnn.Linear(4, 2, bias=False)), ['b'])
model('bn_train', batch_norm_net, ['b'])
model('bn_eval', batch_norm_net, ['b'], pre='b2', mode='eval')
model('bn_twice', batch_norm_net, ['b'], pre='b2')
model('bn_image', lambda: tnn.Sequential(tnn.BatchNorm1d(3, momentum=0.3)), ['img'])
model('bn_plain', lambda: tnn.Sequential(tnn.BatchNorm1d(5, affine=False), tnn.Linear(5, 2)), ['b'])
model('norms', lambda: tnn.Sequential(tnn.GroupNorm(1, 3), tnn.PReLU(3), tnn.InstanceNorm1d(3, affine=True), tnn.GroupNorm(3, 3)), ['img'])
model('embedding', lambda: tnn.Sequential(tnn.Embedding(6, 3), tnn.Flatten(), tnn.Linear(9, 2)), ['ids'])
model('rms', lambda: tnn.Sequential(tnn.Linear(5, 4), tnn.RMSNorm(4), tnn.SiLU(), tnn.Linear(4, 3)), ['b'])
model('pair', Pair, ['b'])
model('bilinear', lambda: tnn.Bilinear(3, 5, 2), ['x2', 'b'])
model('flatten', lambda: tnn.Sequential(tnn.Flatten(), tnn.Linear(12, 2), tnn.LogSoftmax(dim=1)), ['img'])
model('dropout_eval', lambda: tnn.Sequential(tnn.Linear(5, 4), tnn.Dropout(0.5), tnn.ReLU()), ['b'], mode='eval')
model('activations', lambda: tnn.Sequential(tnn.ELU(0.5), tnn.Hardtanh(-0.8, 0.9), tnn.Softplus(2.0), tnn.LeakyReLU(0.1), tnn.Softmax(dim=0)), ['b'])
model('nested', lambda: tnn.Sequential(tnn.Sequential(tnn.Linear(5, 4), tnn.Mish()), tnn.Sequential(tnn.LayerNorm(4, bias=False), tnn.Linear(4, 2))), ['b'])

mrng = np.random.default_rng(20261108)


def shaped(name):
    if name == 'img':
        return t('img').reshape(2, 3, 4)
    if name == 'ids':
        return t('ids').long()
    return t(name)


listed = []
for name, make, inputs, pre, mode in models:
    torch.manual_seed(0)
    mod = make()
    params = list(mod.parameters())
    n = sum(p.numel() for p in params)
    theta = np.round(mrng.normal(0.0, 0.5, n), 4)
    with torch.no_grad():
        torch.nn.utils.vector_to_parameters(torch.tensor(theta), params)
    write('data/theta_%s.csv' % name, [theta.tolist()])
    if pre != 'none':
        mod.train()
        with torch.no_grad():
            mod(shaped(pre))
    mod.train(mode == 'train')
    xs = [shaped(i) for i in inputs]
    numbers = [x.requires_grad_(True) for x in xs if x.is_floating_point()]
    out = mod(*xs)
    flat = out.reshape(-1)
    s = (flat * weights(flat.numel())).sum()
    gs = torch.autograd.grad(s, params + numbers)
    row = [float(v) for v in flat.tolist()]
    for g in gs:
        row += [float(v) for v in g.reshape(-1).tolist()]
    for b in mod.modules():
        if isinstance(b, tnn.BatchNorm1d):
            row += b.running_mean.tolist() + b.running_var.tolist() + [float(b.num_batches_tracked)]
    write('expected/model_%s.csv' % name, [row])
    # (a model of no parameters lists `none`: an empty cell is missing)
    names = ' '.join('%s:%s' % (k, 'x'.join(str(d) for d in p.shape)) for k, p in mod.named_parameters()) or 'none'
    listed.append([name] + (inputs + ['none'])[:2] + [pre, mode, len(numbers), names])

with open(os.path.join(HERE, 'expected/models.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['case', 'x', 'y', 'pre', 'mode', 'takes', 'names'])
    w.writerows(listed)


# ---- nn.init: what is fixed of it --------------------------------------
#
# What PyTorch fixes: the gains, the fans, the identities (dirac, eye) and
# the inverse error function trunc_normal draws through. The draws
# themselves are torch's generator's, held by rng_reference.py.
import torch.nn.init as tinit

gains = [tinit.calculate_gain(n) for n in ['linear', 'conv1d', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'selu']]
gains.append(tinit.calculate_gain('leaky_relu', 0.2))
write('expected/init_gains.csv', [gains])
fans = []
for shape in [(3, 4), (8, 3, 5), (6, 2, 3, 3)]:
    fans += list(tinit._calculate_fan_in_and_fan_out(torch.empty(shape)))
write('expected/init_fans.csv', [[float(f) for f in fans]])
write('expected/init_dirac3.csv', [tinit.dirac_(torch.empty(4, 2, 3), groups=2).reshape(-1).tolist()])
write('expected/init_dirac4.csv', [tinit.dirac_(torch.empty(2, 2, 3, 3)).reshape(-1).tolist()])
write('expected/init_dirac5.csv', [tinit.dirac_(torch.empty(3, 2, 3, 2, 3)).reshape(-1).tolist()])
write('expected/init_eye.csv', [tinit.eye_(torch.empty(3, 5)).reshape(-1).tolist()])
ys = [-0.999, -0.9, -0.5, -1e-8, 0.0, 1e-6, 0.3, 0.7, 0.95, 0.999999]
write('data/init_ys.csv', [ys])
write('expected/init_erfinv.csv', [torch.erfinv(torch.tensor(ys)).tolist()])

print('%d cases and %d models, torch %s' % (len(cases), len(models), torch.__version__))

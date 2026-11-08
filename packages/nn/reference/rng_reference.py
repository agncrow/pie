"""nn.rng and nn.data held to torch's CPU generator (torch 2.8).

A model of the generator in Python -- MT19937 seeded as manual_seed seeds
it, and each draw as torch's CPU kernels make it of the generator's words --
is checked here against torch itself, draw by draw, then writes the rows
held_rng.co and held_data.co compare pie's draws with:

  expected/rng_ints.csv    a case per row: its name and its ints, joined
                           by spaces (an int64 past 2^53 is no float)
  expected/rng_<case>.csv  a case's floats, one row

The words, the permutations, the ints, the uniforms and the Bernoulli
draws are pure arithmetic and agree with torch exactly. A normal and an
exponential go through libm (sin, cos, log, log1p): the model calls each
separately, as pie does, and so agrees with torch exactly where torch's
build calls them separately too. macOS's torch fuses a sine and a cosine
of one angle into one call (__sincos_stret), whose sine differs from
sin's in its last digit about once in 1,500: the check reports those
and holds them to one unit in the last place.

Run from pie's root: ~/.venvs/gpt/bin/python -I packages/nn/reference/rng_reference.py
"""
import csv
import math
import os
from fractions import Fraction

import torch

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
M32 = 0xffffffff


def fma(a, b, c):
    # one rounding of a * b + c, as the compiled kernels' fused multiply-add
    return float(Fraction(a) * Fraction(b) + Fraction(c))


class MT:
    """torch's CPU generator: mt19937 and the cached normal."""

    def __init__(self, seed):
        self.mt = [seed & M32]
        for j in range(1, 624):
            p = self.mt[-1]
            self.mt.append((1812433253 * (p ^ (p >> 30)) + j) & M32)
        self.i = 624
        self.cached = None

    def twist(self):
        mt = self.mt
        for k in range(624):
            y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff)
            mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ (0x9908b0df if y & 1 else 0)
        self.i = 0

    def u32(self):
        if self.i >= 624:
            self.twist()
        y = self.mt[self.i]
        self.i += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= y >> 18
        return y & M32

    def u64(self):
        a = self.u32()
        b = self.u32()
        return (a << 32) | b

    def unit(self):
        return (self.u64() & (2 ** 53 - 1)) * 2.0 ** -53


def randperm(g, n):
    r = list(range(n))
    for i in range(n - 1):
        z = g.u32() % (n - i)
        r[i], r[z + i] = r[z + i], r[i]
    return r


def randint(g, low, high, n):
    span = high - low
    if span < 2 ** 32:
        return [g.u32() % span + low for _ in range(n)]
    return [g.u64() % span + low for _ in range(n)]


def random_int64(g):
    return g.u64() % 2 ** 63


def uniform(g, n, a=0.0, b=1.0):
    return [fma(g.unit(), b - a, a) for _ in range(n)]


def normal(g, n, mu=0.0, sigma=1.0):
    if n >= 16:
        data = [g.unit() for _ in range(n)]

        def sixteen(at):
            for j in range(8):
                u1 = 1 - data[at + j]
                u2 = data[at + j + 8]
                radius = math.sqrt(-2 * math.log(u1))
                theta = 2.0 * math.pi * u2
                data[at + j] = fma(radius * math.cos(theta), sigma, mu)
                data[at + j + 8] = fma(radius * math.sin(theta), sigma, mu)

        for i in range(0, n - 15, 16):
            sixteen(i)
        if n % 16 != 0:
            for i in range(16):
                data[n - 16 + i] = g.unit()
            sixteen(n - 16)
        return data
    out = []
    for _ in range(n):
        if g.cached is not None:
            out.append(fma(g.cached, sigma, mu))
            g.cached = None
            continue
        u1 = g.unit()
        u2 = g.unit()
        r = math.sqrt(-2.0 * math.log1p(-u2))
        theta = 2.0 * math.pi * u1
        g.cached = r * math.sin(theta)
        out.append(fma(r * math.cos(theta), sigma, mu))
    return out


def bernoulli(g, n, p=0.5):
    return [1.0 if g.unit() < p else 0.0 for _ in range(n)]


def exponential(g, n, lambd=1.0):
    return [-1.0 / lambd * math.log1p(-g.unit()) for _ in range(n)]


def multinomial(g, w, n, replacement=False):
    k = len(w)
    if not replacement or n == 1:
        q = exponential(g, k, 1.0)
        keys = [w[i] / q[i] for i in range(k)]
        return sorted(range(k), key=lambda i: -keys[i])[:n]
    total = 0.0
    cdf = []
    for x in w:
        total += x
        cdf.append(total)
    cdf = [c / total for c in cdf]
    cdf[-1] = 1.0
    out = []
    for _ in range(n):
        u = g.unit()
        lo, hi = 0, k
        while hi - lo > 0:
            mid = lo + (hi - lo) // 2
            if cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        out.append(lo)
    return out


def ulps(a, b):
    if a == b:
        return 0
    return abs(a - b) / math.ulp(max(abs(a), abs(b)))


ints = []
floats = {}
apart = []


def hold_ints(name, mine, theirs):
    assert mine == theirs, (name, mine[:8], theirs[:8])
    ints.append((name, mine))


def hold_floats(name, mine, theirs, exact=True):
    assert len(mine) == len(theirs), name
    far = [ulps(a, b) for a, b in zip(mine, theirs)]
    if exact:
        assert max(far) == 0, (name, max(far))
    else:
        assert max(far) <= 1, (name, max(far))
        off = sum(1 for d in far if d)
        if off:
            apart.append('%s: %d of %d one ulp from torch' % (name, off, len(far)))
    floats[name] = mine


def gen(seed=None):
    t = torch.Generator()
    if seed is not None:
        t.manual_seed(seed)
    return t


def empty(n):
    return torch.empty(n, dtype=torch.float64)


# the words, through the permutations: across twists, from many seeds
for seed, n in [(0, 10), (42, 100), (7, 1), (7, 2), (2 ** 31 + 5, 1000), (123456789, 3000), (2 ** 40 + 7, 50)]:
    hold_ints('randperm_%d_%d' % (seed, n), randperm(MT(seed), n), torch.randperm(n, generator=gen(seed)).tolist())
# a generator nothing seeded: torch's default seed, 67280421310721
hold_ints('randperm_default_10', randperm(MT(67280421310721), 10), torch.randperm(10, generator=gen()).tolist())
hold_ints('randint_7_0_10', randint(MT(7), 0, 10, 20), torch.randint(0, 10, (20,), generator=gen(7)).tolist())
hold_ints('randint_8_m3_4', randint(MT(8), -3, 4, 2000), torch.randint(-3, 4, (2000,), generator=gen(8)).tolist())
hold_ints('randint_17_wide', randint(MT(17), -5, 2 ** 40, 6), torch.randint(-5, 2 ** 40, (6,), generator=gen(17)).tolist())
hold_ints('randint_17_2_32', randint(MT(17), 3, 2 ** 32 + 3, 6), torch.randint(3, 2 ** 32 + 3, (6,), generator=gen(17)).tolist())
m = MT(7)
t = gen(7)
hold_ints('random_int64_7', [random_int64(m) for _ in range(3)],
          [torch.empty((), dtype=torch.int64).random_(generator=t).item() for _ in range(3)])
w6 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.05]
hold_ints('multinomial_29_4', multinomial(MT(29), w6, 4), torch.multinomial(torch.tensor(w6), 4, generator=gen(29)).tolist())
hold_ints('multinomial_29_1r', multinomial(MT(29), w6, 1, True),
          torch.multinomial(torch.tensor(w6), 1, replacement=True, generator=gen(29)).tolist())
hold_ints('multinomial_31_40r', multinomial(MT(31), w6, 40, True),
          torch.multinomial(torch.tensor(w6), 40, replacement=True, generator=gen(31)).tolist())
w3 = [3.0, 0.0, 1.0]
hold_ints('multinomial_32_3', multinomial(MT(32), w3, 2), torch.multinomial(torch.tensor(w3), 2, generator=gen(32)).tolist())

hold_floats('rand_5_7', uniform(MT(5), 7), torch.rand(7, generator=gen(5)).tolist())
hold_floats('uniform_11_100', uniform(MT(11), 100, -2.0, 3.0), empty(100).uniform_(-2.0, 3.0, generator=gen(11)).tolist())
hold_floats('bernoulli_19_50', bernoulli(MT(19), 50, 0.3), empty(50).bernoulli_(0.3, generator=gen(19)).tolist())
hold_floats('exponential_23_50', exponential(MT(23), 50, 2.5), empty(50).exponential_(2.5, generator=gen(23)).tolist(), exact=False)
for seed, n, mu, sigma in [(3, 5, 0.0, 1.0), (3, 16, 0.0, 1.0), (4, 37, 0.5, 2.0), (9, 1000, -1.0, 0.25), (99, 15, 0.1, 0.37)]:
    hold_floats('normal_%d_%d' % (seed, n), normal(MT(seed), n, mu, sigma),
                empty(n).normal_(mu, sigma, generator=gen(seed)).tolist(), exact=False)
# the cached normal: one number holds its pair's other for the next call,
# a fill of 16 or more neither reads it nor clears it, a seed clears it
m = MT(6)
t = gen(6)
mine = normal(m, 1) + normal(m, 20) + normal(m, 3) + normal(m, 2)
theirs = []
for n in [1, 20, 3, 2]:
    theirs += empty(n).normal_(generator=t).tolist()
hold_floats('normal_cached', mine, theirs, exact=False)
# one generator through draws of every kind, in turn
m = MT(12)
t = gen(12)
mine = uniform(m, 3) + [float(x) for x in randperm(m, 10)] + normal(m, 5) + bernoulli(m, 4) + normal(m, 1) + [float(x) for x in randint(m, 0, 100, 3)]
theirs = (torch.rand(3, generator=t).tolist() + [float(x) for x in torch.randperm(10, generator=t).tolist()] +
          empty(5).normal_(generator=t).tolist() + empty(4).bernoulli_(0.5, generator=t).tolist() +
          empty(1).normal_(generator=t).tolist() + [float(x) for x in torch.randint(0, 100, (3,), generator=t).tolist()])
hold_floats('mixed_12', mine, theirs, exact=False)


# --- torch.nn.init and the modules' own draws, in float64 --------------------
I = torch.nn.init
nn_ = torch.nn


def drawn(f, *shape):
    t = torch.empty(*shape)
    f(t)
    return t.flatten().tolist()


init_cases = [
    ('init_uniform', lambda: drawn(lambda t: I.uniform_(t, -0.5, 2.0, generator=gen(5)), 3, 4), 0),
    ('init_normal', lambda: drawn(lambda t: I.normal_(t, 1.0, 2.0, generator=gen(6)), 100), 1),
    ('init_normal_7', lambda: drawn(lambda t: I.normal_(t, 1.0, 2.0, generator=gen(6)), 7), 1),
    ('init_xavier_uniform', lambda: drawn(lambda t: I.xavier_uniform_(t, generator=gen(7)), 20, 5), 0),
    ('init_xavier_normal', lambda: drawn(lambda t: I.xavier_normal_(t, gain=2.0, generator=gen(8)), 7, 3), 1),
    ('init_kaiming_normal', lambda: drawn(lambda t: I.kaiming_normal_(t, mode='fan_out', nonlinearity='relu', generator=gen(9)), 10, 10), 1),
    ('init_kaiming_uniform', lambda: drawn(lambda t: I.kaiming_uniform_(t, a=0.2, generator=gen(10)), 10, 25), 0),
    ('init_trunc_normal', lambda: drawn(lambda t: I.trunc_normal_(t, generator=gen(11)), 50), 1e-11),
    ('init_trunc_normal_ab', lambda: drawn(lambda t: I.trunc_normal_(t, 0.5, 2.0, -1.0, 3.0, generator=gen(12)), 40), 1e-11),
    ('init_orthogonal_tall', lambda: drawn(lambda t: I.orthogonal_(t, generator=gen(13)), 5, 3), 1e-12),
    ('init_orthogonal_wide', lambda: drawn(lambda t: I.orthogonal_(t, 2.0, generator=gen(14)), 3, 5), 1e-12),
    ('init_orthogonal_kernel', lambda: drawn(lambda t: I.orthogonal_(t, generator=gen(15)), 4, 2, 3), 1e-12),
]


def sparse16():
    torch.manual_seed(16)
    return drawn(lambda t: I.sparse_(t, 0.3), 10, 4)


def params(seed, made):
    torch.manual_seed(seed)
    model = made()
    return [x for p in model.parameters() for x in p.detach().flatten().tolist()]


init_cases += [
    ('init_sparse', sparse16, 1),
    ('params_seq_1', lambda: params(1, lambda: nn_.Sequential(nn_.Linear(5, 4), nn_.ReLU(), nn_.Linear(4, 3))), 0),
    ('params_mixed_2', lambda: params(2, lambda: nn_.Sequential(nn_.Embedding(3, 4), nn_.Linear(4, 2, bias=False), nn_.LayerNorm(2), nn_.PReLU())), 1),
    ('params_bilinear_3', lambda: params(3, lambda: nn_.Bilinear(3, 2, 4)), 0),
    ('params_embeddings_4', lambda: params(4, lambda: nn_.Sequential(nn_.Embedding(3, 3), nn_.Embedding(2, 3))), 1),
]
# held_rng.co reads how far each case may be: 0 exactly, 1 an ulp (libm),
# or a distance (erfinv's, the QR's)
within = {}
for name, f, far in init_cases:
    floats[name] = f()
    within[name] = far
# the model's own answers for the draws that are pure arithmetic or libm's
m = MT(5)
assert floats['init_uniform'] == uniform(m, 12, -0.5, 2.0)
m = MT(6)
assert max(ulps(a, b) for a, b in zip(floats['init_normal_7'], normal(m, 7, 1.0, 2.0))) <= 1

# --- torch.utils.data: loaders, samplers and splits ---------------------------
from torch.utils.data import (BatchSampler, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler,
                              TensorDataset, WeightedRandomSampler, random_split)


def model_pass(kind, g, count=0, indices=(), weights=(), replacement=False, num_samples=0):
    """a sampler's positions drawn from g (RandomSampler's generator=None
    case is the caller's: it seeds a fresh generator from the process's)"""
    if kind == 'sequential':
        return list(range(count))
    if kind == 'subset':
        return [indices[i] for i in randperm(g, len(indices))]
    if kind == 'weighted':
        return multinomial(g, list(weights), num_samples, replacement)
    n = count
    num = num_samples or n
    if replacement:
        return [g.u32() % n for _ in range(num)]
    out = []
    for _ in range(num // n):
        out += randperm(g, n)
    return out + randperm(g, n)[:num % n]


def model_batches(order, size, drop_last):
    out = [order[i:i + size] for i in range(0, len(order), size)]
    if drop_last and out and len(out[-1]) < size:
        out = out[:-1]
    return out


def loader_epochs(ds, epochs, **kw):
    dl = DataLoader(ds, **kw)
    return [[b[0].tolist() for b in dl] for _ in range(epochs)]


def flat(epochs):
    return [i for e in epochs for b in e for i in b]


def sizes(epochs):
    return [len(b) for e in epochs for b in e]


ten = TensorDataset(torch.arange(10))
# a shuffled loader handed a generator: per epoch, the base seed (a 64-bit
# word), the permutation and the remainder's permutation, all from it
for bs, drop in [(4, False), (3, True), (5, True)]:
    theirs = loader_epochs(ten, 3, batch_size=bs, shuffle=True, drop_last=drop, generator=gen(3))
    m = MT(3)
    mine = []
    for e in range(3):
        random_int64(m)
        mine.append(model_batches(model_pass('random', m, 10), bs, drop))
    assert mine == theirs, (bs, drop, mine, theirs)
    hold_ints('loader_3_%d_%s' % (bs, drop), flat(mine), flat(theirs))
    hold_ints('loader_sizes_3_%d_%s' % (bs, drop), sizes(mine), sizes(theirs))
# a shuffled loader handed none, after torch.manual_seed(21): the base seed,
# then the sampler's seed, from the process's generator; the permutation
# from a generator seeded by it; then the process's generator draws on
torch.manual_seed(21)
theirs = loader_epochs(ten, 2, batch_size=4, shuffle=True)
after = torch.randperm(10).tolist()
m = MT(21)
mine = []
for e in range(2):
    random_int64(m)
    local = MT(random_int64(m))
    mine.append(model_batches(model_pass('random', local, 10), 4, False))
hold_ints('loader_process_21', flat(mine), flat(theirs))
hold_ints('loader_process_21_after', randperm(m, 10), after)
# an ordered loader still draws its base seed from the process's generator
torch.manual_seed(22)
theirs = loader_epochs(ten, 2, batch_size=4)
after = torch.randperm(10).tolist()
m = MT(22)
for e in range(2):
    random_int64(m)
hold_ints('loader_ordered_22_after', randperm(m, 10), after)
# samplers seeded alone, in a loader handed a generator of its own
w10 = [0.5, 1.0, 0.25, 2.0, 1.5, 0.75, 0.1, 3.0, 1.25, 0.6]
cases = [
    ('random_replacement', dict(kind='random', count=10, replacement=True, num_samples=40),
     lambda g: RandomSampler(range(10), replacement=True, num_samples=40, generator=g)),
    ('random_25', dict(kind='random', count=10, num_samples=25), lambda g: RandomSampler(range(10), num_samples=25, generator=g)),
    ('random_7', dict(kind='random', count=10, num_samples=7), lambda g: RandomSampler(range(10), num_samples=7, generator=g)),
    ('subset', dict(kind='subset', indices=[5, 7, 9, 2]), lambda g: SubsetRandomSampler([5, 7, 9, 2], generator=g)),
    ('weighted', dict(kind='weighted', weights=w10, replacement=True, num_samples=12),
     lambda g: WeightedRandomSampler(w10, 12, generator=g)),
    ('weighted_unreplaced', dict(kind='weighted', weights=w10, replacement=False, num_samples=6),
     lambda g: WeightedRandomSampler(w10, 6, replacement=False, generator=g)),
]
for name, kw, made in cases:
    s = made(gen(9))
    theirs = loader_epochs(ten, 2, batch_size=3, sampler=s, generator=gen(4))
    m = MT(9)
    mine = [model_batches(model_pass(g=m, **kw), 3, False) for _ in range(2)]
    hold_ints('sampler_%s' % name, flat(mine), flat(theirs))
    hold_ints('sampler_sizes_%s' % name, sizes(mine), sizes(theirs))
# samplers given no generator draw from the process's: a random one
# through a generator seeded from it, the others from it directly
for name, kw, made in cases:
    torch.manual_seed(31)
    theirs = loader_epochs(ten, 2, batch_size=3, sampler=made(None), generator=gen(4))
    after = torch.randperm(10).tolist()
    m = MT(31)
    mine = []
    for e in range(2):
        if kw['kind'] == 'random':
            mine.append(model_batches(model_pass(g=MT(random_int64(m)), **kw), 3, False))
        else:
            mine.append(model_batches(model_pass(g=m, **kw), 3, False))
    hold_ints('sampler_process_%s' % name, flat(mine), flat(theirs))
    hold_ints('sampler_process_%s_after' % name, randperm(m, 10), after)
# random_split: counts, fractions, a remainder handed out from the first
for name, n, lengths in [('counts', 10, [3, 3, 4]), ('fractions', 10, [0.5, 0.3, 0.2]),
                         ('remainder', 11, [0.25, 0.25, 0.5]), ('thirds', 10, [1 / 3, 1 / 3, 1 / 3]), ('whole', 7, [1.0])]:
    parts = [list(p.indices) for p in random_split(range(n), lengths, generator=gen(5))]
    perm = randperm(MT(5), n)
    sizes_ = [len(p) for p in parts]
    mine = []
    at = 0
    for k in sizes_:
        mine.append(perm[at:at + k])
        at += k
    assert mine == parts, (name, mine, parts)
    hold_ints('split_%s' % name, [i for p in mine for i in p], [i for p in parts for i in p])
    hold_ints('split_sizes_%s' % name, sizes_, sizes_)

os.makedirs(os.path.join(HERE, 'expected'), exist_ok=True)
with open(os.path.join(HERE, 'expected/rng_ints.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['case', 'values'])
    for name, values in ints:
        out.writerow([name, ' '.join(str(v) for v in values)])
with open(os.path.join(HERE, 'expected/rng_init.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['case', 'within'])
    for name, far in within.items():
        out.writerow([name, repr(float(far))])
for name, values in floats.items():
    with open(os.path.join(HERE, 'expected/rng_%s.csv' % name), 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['c%d' % i for i in range(len(values))])
        out.writerow([repr(v) for v in values])
print('the model agrees with torch %s: %d int cases, %d float cases' % (torch.__version__, len(ints), len(floats)))
for line in apart:
    print('  ' + line)

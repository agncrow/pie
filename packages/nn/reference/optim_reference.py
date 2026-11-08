"""nn.optim and nn.lr_scheduler held to torch.optim (torch 2.8, float64).

Each optimizer case starts from one theta of 8 numbers split, where the case
says, into PyTorch parameter groups, and takes 12 steps on the gradient
field g(theta) = a*theta - b + c*theta^3 (computed in the same order of
operations as held_optim.co computes it, so the field adds no difference of
its own); its row (expected/optim_<case>.csv) is theta after each step. A
case with a schedule steps it after each optimizer step, as PyTorch's loop
does. Each schedule case (expected/sched_<case>.csv) is the first group's
learning rate after the schedule is made and after each of 30 steps, then,
where the schedule cycles it, the momentum (or beta1) the same way.
expected/optim.csv and expected/sched.csv list the cases. expected/clip.csv
holds a gradient clipped by norm (2, 1, 3 and the largest number) and by
value, as torch.nn.utils clips it.

Run from pie's root: ~/.venvs/gpt/bin/python -I packages/nn/reference/optim_reference.py
"""
import csv
import math
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))


def write(path, rows):
    path = os.path.join(HERE, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [list(r) for r in rows]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['c%d' % i for i in range(len(rows[0]))])
        for r in rows:
            w.writerow([repr(float(x)) for x in r])


rng = np.random.default_rng(20261109)
theta0 = np.round(rng.normal(0.0, 1.0, 8), 4)
fa = np.round(rng.uniform(0.5, 2.0, 8), 4)
fb = np.round(rng.normal(0.0, 1.0, 8), 4)
fc = np.round(rng.uniform(0.0, 0.1, 8), 4)
write('data/optim_field.csv', [theta0, fa, fb, fc])
A, B, C = torch.tensor(fa), torch.tensor(fb), torch.tensor(fc)


def field(th):
    # a .* th - b + c .* th .* th .* th, each operation rounded as Crow's is
    return ((A * th - B) + ((C * th) * th) * th)


O = torch.optim
L = torch.optim.lr_scheduler

# name, make(params-or-groups) -> optimizer, groups ('one', 'two', 'swapped'),
# schedule (make(optimizer) -> scheduler, or None)
optim_cases = [
    ('sgd', lambda p: O.SGD(p, lr=0.1), 'one', None),
    ('sgd_momentum', lambda p: O.SGD(p, lr=0.05, momentum=0.9), 'one', None),
    ('sgd_dampening', lambda p: O.SGD(p, lr=0.05, momentum=0.9, dampening=0.3, weight_decay=0.01), 'one', None),
    ('sgd_nesterov', lambda p: O.SGD(p, lr=0.05, momentum=0.8, nesterov=True), 'one', None),
    ('sgd_maximize', lambda p: O.SGD(p, lr=0.01, momentum=0.5, maximize=True), 'one', None),
    ('adam', lambda p: O.Adam(p, lr=0.05), 'one', None),
    ('adam_amsgrad', lambda p: O.Adam(p, lr=0.05, amsgrad=True, weight_decay=0.1), 'one', None),
    ('adam_decoupled', lambda p: O.Adam(p, lr=0.05, weight_decay=0.1, decoupled_weight_decay=True), 'one', None),
    ('adamw', lambda p: O.AdamW(p, lr=0.05), 'one', None),
    ('adamw_amsgrad', lambda p: O.AdamW(p, lr=0.05, betas=(0.8, 0.99), eps=1e-6, weight_decay=0.05, amsgrad=True), 'one', None),
    ('adam_maximize', lambda p: O.Adam(p, lr=0.01, betas=(0.3, 0.9), maximize=True), 'one', None),
    ('adamax', lambda p: O.Adamax(p, lr=0.05, weight_decay=0.01), 'one', None),
    ('nadam', lambda p: O.NAdam(p, lr=0.05), 'one', None),
    ('nadam_decoupled', lambda p: O.NAdam(p, lr=0.05, weight_decay=0.1, decoupled_weight_decay=True, momentum_decay=0.01), 'one', None),
    ('radam', lambda p: O.RAdam(p, lr=0.05), 'one', None),
    ('radam_wd', lambda p: O.RAdam(p, lr=0.05, betas=(0.9, 0.99), weight_decay=0.1), 'one', None),
    ('radam_decoupled', lambda p: O.RAdam(p, lr=0.05, weight_decay=0.1, decoupled_weight_decay=True), 'one', None),
    ('rmsprop', lambda p: O.RMSprop(p, lr=0.01), 'one', None),
    ('rmsprop_centered', lambda p: O.RMSprop(p, lr=0.01, alpha=0.9, centered=True, momentum=0.5, weight_decay=0.01), 'one', None),
    ('adagrad', lambda p: O.Adagrad(p, lr=0.1), 'one', None),
    ('adagrad_decay', lambda p: O.Adagrad(p, lr=0.1, lr_decay=0.01, weight_decay=0.01, initial_accumulator_value=0.1), 'one', None),
    ('adadelta', lambda p: O.Adadelta(p, lr=1.0), 'one', None),
    ('adadelta_wd', lambda p: O.Adadelta(p, lr=0.5, rho=0.8, eps=1e-5, weight_decay=0.01), 'one', None),
    ('rprop', lambda p: O.Rprop(p, lr=0.01), 'one', None),
    ('rprop_bounds', lambda p: O.Rprop(p, lr=0.05, etas=(0.3, 1.5), step_sizes=(0.001, 0.1)), 'one', None),
    ('asgd', lambda p: O.ASGD(p, lr=0.01), 'one', None),
    ('asgd_t0', lambda p: O.ASGD(p, lr=0.05, lambd=0.001, alpha=0.5, t0=3, weight_decay=0.01), 'one', None),
    ('groups_sgd', lambda p: O.SGD(p, lr=0.05, momentum=0.9), 'two', None),
    ('groups_adam', lambda p: O.Adam(p, lr=0.05), 'swapped', None),
    ('adam_cosine', lambda p: O.Adam(p, lr=0.05), 'one', lambda o: L.CosineAnnealingLR(o, T_max=6)),
    ('sgd_one_cycle', lambda p: O.SGD(p, lr=0.05, momentum=0.9), 'one', lambda o: L.OneCycleLR(o, max_lr=0.1, total_steps=12)),
    ('adam_one_cycle', lambda p: O.Adam(p, lr=0.05), 'two', lambda o: L.OneCycleLR(o, max_lr=0.1, total_steps=12, three_phase=True)),
    ('groups_step', lambda p: O.SGD(p, lr=0.05, momentum=0.9), 'two', lambda o: L.StepLR(o, step_size=3, gamma=0.5)),
]


def groups_of(kind):
    th = torch.tensor(theta0)
    if kind == 'one':
        ps = [th.clone().requires_grad_(False)]
        return ps, ps
    a = th[0:5].clone()
    b = th[5:8].clone()
    if kind == 'two':
        return [a, b], [{'params': [a]}, {'params': [b], 'lr': 0.01, 'weight_decay': 0.1}]
    return [a, b], [{'params': [b], 'lr': 0.02}, {'params': [a]}]


listed = []
for name, make, kind, sched in optim_cases:
    ps, given = groups_of(kind)
    opt = make(given)
    s = sched(opt) if sched else None
    row = []
    for k in range(12):
        th = torch.cat([p.detach() for p in ps])
        g = field(th)
        at = 0
        for p in ps:
            p.grad = g[at:at + p.numel()].clone()
            at += p.numel()
        opt.step()
        if s is not None:
            s.step()
        row += torch.cat([p.detach() for p in ps]).tolist()
    write('expected/optim_%s.csv' % name, [row])
    listed.append([name, kind])

with open(os.path.join(HERE, 'expected/optim.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['case', 'groups'])
    w.writerows(listed)


# ---- the schedules: the learning rate, and a momentum a schedule cycles

def decay(epoch):
    return 0.95 ** epoch


def tenth(epoch):
    return 0.9


falling = [1.0, 0.9, 0.85, 0.86, 0.84, 0.8, 0.8, 0.79, 0.81, 0.8, 0.7, 0.7, 0.69, 0.7, 0.7,
           0.71, 0.72, 0.6, 0.6, 0.6, 0.59, 0.58, 0.6, 0.61, 0.6, 0.5, 0.5, 0.5, 0.5, 0.49]
rising = [0.1, 0.2, 0.25, 0.3, 0.31, 0.32, 0.33, 0.5, 0.55, 0.5, 0.52, 0.53, 0.54, 0.7, 0.69,
          0.7, 0.71, 0.72, 0.9, 0.9, 0.95, 0.96, 0.97, 0.97, 1.1, 1.1, 1.15, 1.2, 1.2, 1.2]
write('data/sched_metrics.csv', [falling, rising])

# name, optimizer ('sgd' with momentum 0.9 or 'adam'), make(o) -> scheduler, the
# metrics a plateau case reads ('none', 'falling', 'rising'), whether it cycles
# the momentum
sched_cases = [
    ('step', 'sgd', lambda o: L.StepLR(o, step_size=4, gamma=0.5), 'none', False),
    ('multi_step', 'sgd', lambda o: L.MultiStepLR(o, milestones=[3, 7, 7, 12], gamma=0.3), 'none', False),
    ('constant', 'sgd', lambda o: L.ConstantLR(o, factor=0.25, total_iters=6), 'none', False),
    ('linear', 'sgd', lambda o: L.LinearLR(o, start_factor=0.2, end_factor=0.9, total_iters=10), 'none', False),
    ('exponential', 'sgd', lambda o: L.ExponentialLR(o, gamma=0.9), 'none', False),
    ('polynomial', 'sgd', lambda o: L.PolynomialLR(o, total_iters=12, power=2.0), 'none', False),
    ('cosine', 'sgd', lambda o: L.CosineAnnealingLR(o, T_max=10, eta_min=0.001), 'none', False),
    ('warm_restarts', 'sgd', lambda o: L.CosineAnnealingWarmRestarts(o, T_0=5, T_mult=2, eta_min=0.002), 'none', False),
    ('warm_restarts_one', 'sgd', lambda o: L.CosineAnnealingWarmRestarts(o, T_0=4), 'none', False),
    ('cyclic', 'sgd', lambda o: L.CyclicLR(o, base_lr=0.01, max_lr=0.1, step_size_up=4, mode='triangular2'), 'none', True),
    ('cyclic_exp', 'sgd', lambda o: L.CyclicLR(o, base_lr=0.01, max_lr=0.1, step_size_up=3, step_size_down=5, mode='exp_range', gamma=0.95, cycle_momentum=False), 'none', False),
    ('cyclic_adam', 'adam', lambda o: L.CyclicLR(o, base_lr=0.001, max_lr=0.01, step_size_up=5, base_momentum=0.85, max_momentum=0.95), 'none', True),
    ('one_cycle', 'sgd', lambda o: L.OneCycleLR(o, max_lr=0.2, total_steps=30), 'none', True),
    ('one_cycle_three', 'adam', lambda o: L.OneCycleLR(o, max_lr=0.2, total_steps=30, pct_start=0.25, anneal_strategy='linear', three_phase=True, div_factor=10.0, final_div_factor=100.0), 'none', True),
    ('one_cycle_epochs', 'sgd', lambda o: L.OneCycleLR(o, max_lr=0.3, epochs=5, steps_per_epoch=6, cycle_momentum=False), 'none', False),
    ('lambda', 'sgd', lambda o: L.LambdaLR(o, lr_lambda=decay), 'none', False),
    ('multiplicative', 'sgd', lambda o: L.MultiplicativeLR(o, lr_lambda=tenth), 'none', False),
    ('sequential', 'sgd', lambda o: L.SequentialLR(o, [L.ConstantLR(o, factor=0.5, total_iters=3), L.ExponentialLR(o, gamma=0.9)], milestones=[4]), 'none', False),
    ('sequential_cosine', 'sgd', lambda o: L.SequentialLR(o, [L.LinearLR(o, start_factor=0.1, total_iters=5), L.CosineAnnealingLR(o, T_max=20)], milestones=[5]), 'none', False),
    ('chained', 'sgd', lambda o: L.ChainedScheduler([L.ConstantLR(o, factor=0.5, total_iters=3), L.ExponentialLR(o, gamma=0.9)]), 'none', False),
    ('plateau', 'sgd', lambda o: L.ReduceLROnPlateau(o, mode='min', factor=0.5, patience=2, threshold=0.01, cooldown=1, min_lr=0.001), 'falling', False),
    ('plateau_max_abs', 'sgd', lambda o: L.ReduceLROnPlateau(o, mode='max', threshold_mode='abs', threshold=0.1, patience=1, factor=0.3), 'rising', False),
]

metrics = {'falling': falling, 'rising': rising}
listed = []
for name, which, make, reads, cycles in sched_cases:
    p = torch.zeros(1, requires_grad=True)
    o = O.SGD([p], lr=0.1, momentum=0.9) if which == 'sgd' else O.Adam([p], lr=0.1)

    def momentum():
        g = o.param_groups[0]
        return g['momentum'] if which == 'sgd' else g['betas'][0]

    s = make(o)
    lrs = [o.param_groups[0]['lr']]
    ms = [momentum()]
    for k in range(30):
        p.grad = torch.zeros(1)
        o.step()
        if reads == 'none':
            s.step()
        else:
            s.step(metrics[reads][k])
        lrs.append(o.param_groups[0]['lr'])
        ms.append(momentum())
    write('expected/sched_%s.csv' % name, [lrs + (ms if cycles else [])])
    listed.append([name, which, reads, 'true' if cycles else 'false'])

with open(os.path.join(HERE, 'expected/sched.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['case', 'optimizer', 'metrics', 'cycles'])
    w.writerows(listed)

# gradient clipping: a model's gradient (data/clip_g.csv, laid as theta:
# each parameter in state_dict order) clipped by its total norm and by
# value; each row of expected/clip.csv is the total norm (0 for the clip by
# value), then the gradient clipped
model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
g = np.round(rng.normal(0.0, 1.0, sum(q.numel() for q in model.parameters())), 4)
write('data/clip_g.csv', [g])


def given_grads():
    at = 0
    for q in model.parameters():
        q.grad = torch.tensor(g[at:at + q.numel()]).reshape(q.shape).clone()
        at += q.numel()


clip_cases = [(0.5, 2.0), (100.0, 2.0), (0.5, math.inf), (0.7, 1.0), (0.9, 3.0)]
rows = []
for max_norm, p in clip_cases:
    given_grads()
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=p, foreach=False)
    rows.append([float(total)] + torch.cat([q.grad.flatten() for q in model.parameters()]).tolist())
given_grads()
torch.nn.utils.clip_grad_value_(model.parameters(), 0.3, foreach=False)
rows.append([0.0] + torch.cat([q.grad.flatten() for q in model.parameters()]).tolist())
write('expected/clip.csv', rows)

print('%d optimizer cases, %d schedule cases and %d clippings, torch %s' % (len(optim_cases), len(sched_cases), len(rows), torch.__version__))

"""nn.safetensors held to Hugging Face's safetensors library (0.7, through
safetensors.torch, torch 2.8).

The script writes, with the library itself:

  data/st_state.csv           the numbers of a state of four tensors and
                              their shapes, row by row
  data/st_<dtype>.safetensors that state saved in F64 (with metadata), F32,
                              F16 and BF16, as torch's .to(dtype) rounds
                              each number; and a state of whole numbers
                              saved in I64
  data/st_all.safetensors     a tensor of each dtype the format has
  expected/st_<dtype>.csv     the numbers each file holds, as float64
  expected/st_all.csv         each tensor of st_all: its name, its shape
                              and its numbers as float64, joined by spaces
  data/st_model.safetensors   the state_dict of a model torch made: a
                              linear layer, a tanh, a linear layer
  data/st_model_x.csv, expected/st_model_y.csv
                              an input and what that model makes of it

held_safetensors.co saves the same state and compares the bytes with the
library's, and loads every file and compares the numbers with torch's.

Run from pie's root: ~/.venvs/gpt/bin/python -I packages/nn/reference/safetensors_reference.py
"""
import csv
import os

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261108)


def path(name):
    p = os.path.join(HERE, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# the edges of the narrow formats: signed zeros, half's largest number and
# its overflow's midpoint, half's smallest subnormal and the midpoints
# around it, a number just past half's and bfloat16's midpoints (one
# rounding and two part there), infinities and a NaN
edges = [0.0, -0.0, 65504.0, 65519.99, 65520.0, -65520.0, 2.0 ** -24, 2.0 ** -25, 3 * 2.0 ** -26, 1e-8,
         1 + 2.0 ** -11 + 2.0 ** -40, 1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11, 1 + 2.0 ** -8 + 2.0 ** -40,
         -3.75, 6.1e-05, 1e30, -1e-30, 3.4e38, 3.5e38, float('inf'), float('-inf'), float('nan')]
state = {
    'weight': torch.tensor(rng.normal(0.0, 1.0, (3, 4)) * np.array([1e-6, 1.0, 300.0, 1e5])),
    'bias': torch.tensor(rng.uniform(-1.0, 1.0, 4)),
    'emb.weight': torch.tensor(rng.normal(0.0, 0.02, (2, 3, 2))),
    'edges': torch.tensor(edges, dtype=torch.float64),
}
with open(path('data/st_state.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['name', 'dims', 'values'])
    for name, t in state.items():
        out.writerow([name, str(list(t.shape)), ' '.join(repr(float(x)) for x in t.flatten().tolist())])

dtypes = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16}
for word, dtype in dtypes.items():
    held = {k: v.to(dtype) for k, v in state.items()}
    save_file(held, path('data/st_%s.safetensors' % word.lower()), metadata={'format': 'pt'} if word == 'F64' else None)
    back = load_file(path('data/st_%s.safetensors' % word.lower()))
    with open(path('expected/st_%s.csv' % word.lower()), 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['name', 'values'])
        for name in sorted(back):
            out.writerow([name, ' '.join(repr(float(x)) for x in back[name].double().flatten().tolist())])
ints = {'counts': torch.tensor([0, -1, 2 ** 40, -2 ** 53, 7], dtype=torch.int64), 'steps': torch.tensor([[3], [4]], dtype=torch.int64)}
save_file(ints, path('data/st_i64.safetensors'))

every = {
    'f64': torch.tensor([1.5, -2.25, 1e300], dtype=torch.float64),
    'f32': torch.tensor([0.1, -3.0e38, 1e-40], dtype=torch.float32),
    'f16': torch.tensor([[0.1, -65504.0], [6e-8, 1.0]], dtype=torch.float16),
    'bf16': torch.tensor([0.1, -3.0e38, 9.2e-41, float('inf')], dtype=torch.bfloat16),
    'i64': torch.tensor([-(2 ** 62), 5], dtype=torch.int64),
    'i32': torch.tensor([-(2 ** 31), 2 ** 31 - 1], dtype=torch.int32),
    'i16': torch.tensor([-32768, 32767], dtype=torch.int16),
    'i8': torch.tensor([-128, 127], dtype=torch.int8),
    'u64': torch.tensor([2 ** 62, 0], dtype=torch.uint64),
    'u32': torch.tensor([2 ** 32 - 1], dtype=torch.uint32),
    'u16': torch.tensor([65535, 1], dtype=torch.uint16),
    'u8': torch.tensor([255, 0, 9], dtype=torch.uint8),
    'flag': torch.tensor([True, False, True]),
    'one': torch.tensor(2.5, dtype=torch.float64),
}
save_file(every, path('data/st_all.safetensors'), metadata={'note': 'a tensor of each dtype'})
back = load_file(path('data/st_all.safetensors'))
with open(path('expected/st_all.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['name', 'dims', 'values'])
    for name in sorted(back):
        t = back[name]
        out.writerow([name, str(list(t.shape)), ' '.join(repr(float(x)) for x in t.double().flatten().tolist())])

# a model PyTorch made and saved: pie loads its state into the same model
# and runs it on the same input
torch.manual_seed(3)
model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)).double()
save_file(model.state_dict(), path('data/st_model.safetensors'))
x = torch.tensor(rng.normal(0.0, 1.0, (5, 3)))
with open(path('data/st_model_x.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['c%d' % i for i in range(3)])
    for r in x.tolist():
        out.writerow([repr(v) for v in r])
with open(path('expected/st_model_y.csv'), 'w', newline='') as f:
    out = csv.writer(f)
    y = model(x).detach().flatten().tolist()
    out.writerow(['c%d' % i for i in range(len(y))])
    out.writerow([repr(v) for v in y])
print('safetensors %s: %d files written' % (__import__('safetensors').__version__, len(dtypes) + 2))

# Neural networks: `use nn`

Pie's `nn` is PyTorch's `torch.nn`, `torch.nn.functional` and
`torch.optim`, written in Crow over crow's tensors and `grad`. Its names
are PyTorch's, its layers draw their first parameters as PyTorch's draw
them, and every function is held to PyTorch's answers in float64
(`packages/nn/held*.co`, the parity programs under `tests/parity/`).

A program imports the parts it uses:

| import | PyTorch's |
|---|---|
| `use nn` | `torch.nn`: the layers, a model's parameters, its forward pass |
| `use nn.functional as F` | `torch.nn.functional` |
| `use nn.optim` | `torch.optim` |
| `use nn.lr_scheduler as lrs` | `torch.optim.lr_scheduler` |
| `use nn.rng` | the CPU generator behind `torch.manual_seed` |
| `use nn.init` | `torch.nn.init` |
| `use nn.data` | `torch.utils.data`'s loaders and samplers |
| `use nn.safetensors` | `safetensors.torch` |

## The functional forms

`F` holds the activations, the losses with their reductions and options,
the normalisations, attention, the convolutions and poolings, as
`torch.nn.functional` names them. A list of numbers is read as a tensor
where a form takes one.

```crow
use math
use nn.functional as F

log[F.relu[[-1.0, 0.5, 2.0]]]
log[F.softmax[[1.0, 2.0, 3.0]]]
log[fmt[F.gelu[[1.0]][0], decimals = 6]]
log[F.mse_loss[[1.0, 2.0], [1.0, 4.0]]]
log[F.mse_loss[[1.0, 2.0], [1.0, 4.0], reduction = 'sum']]
log[fmt[F.cross_entropy[[[2.0, 0.0, -1.0]], [0.0]], decimals = 6]]
log[fmt[F.cross_entropy[[[2.0, 0.0, -1.0]], [0.0], label_smoothing = 0.1], decimals = 6]]
```

```text
[0, 0.5, 2]
[0.09003057317038045, 0.2447284710547976, 0.6652409557748218]
0.841345
2
4
0.169846
0.336513
```

## A model is one vector of parameters

A PyTorch module owns its parameters and changes them in place. A Crow
value never changes under the program, so pie holds a model as two
values: the model, a record of its layers, and `theta`, every parameter
laid end to end in one vector (as `torch.nn.utils.parameters_to_vector`
lays them). `nn.parameters` draws theta as PyTorch's modules draw their
parameters when they are made, from a generator a seed names, so a model
starts from the numbers its PyTorch twin starts from after
`torch.manual_seed`:

```crow
use math
use nn
use nn.rng

model = nn.sequential[[nn.linear[3, 4], nn.relu[], nn.linear[4, 2]]]
log[nn.size[model]]
for p in nn.named[model]
    log['`p.path` `p.dims`']
theta = nn.parameters[model, rng.manual_seed[0]].values
log[fmt[theta[0], decimals = 6]]
x = tensor[[[1.0, 2.0, 3.0], [0.0, -1.0, 0.5]]]
log[nn.forward[model, theta, x].shape]
state = nn.state[model, theta]
log[state['2.bias']]
log[max[abs[nn.from_state[model, state] - theta]]]
```

```text
26
0.weight [4, 3]
0.bias [4]
2.weight [2, 4]
2.bias [2]
0.542770
[2, 2]
[-0.27397979706160946, 0.05824307538022577]
0
```

After `torch.manual_seed(0)` in float64, PyTorch's
`nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))` starts from
these numbers: its first weight is 0.542770 and its last bias is the one
shown, to the last digit. `nn.named` is `named_parameters`, `nn.state` the
state dict by PyTorch's names, and `nn.from_state` lays a state dict's tensors back into theta.
A model whose forward pass is code — a residual stream, a weight two
layers share — is `nn.module[map[…]]` (PyTorch's `ModuleDict`) or
`nn.module_list[[…]]`, run by a function the program writes, each part
reached by `nn.part[model, name]` and run by `nn.forward`.

## Training

`grad[loss][model, theta, x, y]` is `loss.backward()`: the gradient of
every parameter at once, as one vector beside theta. An optimiser is a
value too: `optim.step[opt, g]` answers it moved one step, its theta the
new parameters. A schedule wraps the optimiser and moves its rate at each
`lrs.step`. Here a small network learns XOR with Adam on a cosine
schedule:

```crow
use math
use nn
use nn.functional as F
use nn.optim, nn.rng
use nn.lr_scheduler as lrs

fn loss[model, theta, x, y]
    return F.mse_loss[nn.forward[model, theta, x], y]

model = nn.sequential[[nn.linear[2, 8], nn.tanh[], nn.linear[8, 1]]]
x = tensor[[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]]
y = tensor[[[0.0], [1.0], [1.0], [0.0]]]
opt = optim.adam[nn.parameters[model, rng.manual_seed[7]].values, lr = 0.05]
opt = lrs.cosine_annealing_lr[opt, 300]
first = loss[model, opt.theta, x, y]
for k in range[300]
    gs = grad[loss][model, opt.theta, x, y]
    opt = optim.step[opt, gs[0]]
    opt = lrs.step[opt]
last = loss[model, opt.theta, x, y]
log[last lt first / 100.0]
out = nn.forward[model, opt.theta, x]
log[min[(out - 0.5) .* (y - 0.5)] gt 0.0]
```

```text
true
true
```

Every optimiser of `torch.optim` is there (SGD, Adam, AdamW, Adamax,
NAdam, RAdam, RMSprop, Adagrad, Adadelta, Rprop, ASGD) with their
options and parameter groups, every schedule of `lr_scheduler`, and
`nn.clip_grad_norm` and `nn.clip_grad_value`; `held_optim.co` holds each
step to PyTorch's.

## Random numbers are torch's

A seed names the same numbers in pie as in PyTorch. `nn.rng` is torch's
CPU generator written in Crow — a Mersenne Twister seeded as
`torch.manual_seed` seeds it, each distribution drawn as ATen draws it —
and a generator is a value: a draw answers its numbers beside the
generator moved past them.

```crow
use math
use nn.rng

a = rng.normal[rng.manual_seed[0], 3]
log[a.values]
b = rng.uniform[a.generator, 2]
log[b.values]
log[rng.randperm[rng.manual_seed[0], 6].values]
```

```text
[1.5409961082440433, -0.2934289057609464, -2.1787893820745574]
[0.6450241201227648, 0.7911478921803037]
[2, 5, 3, 0, 1, 4]
```

These are PyTorch's numbers to the last digit: the first line is
`torch.randn(3, dtype=torch.float64)` after `torch.manual_seed(0)`, the
second `torch.rand(2, dtype=torch.float64)` drawn next, and the third
`torch.randperm(6)` after the seed again.

## Dropout reads the noise drawn for its pass

A PyTorch dropout draws its mask inside the forward pass. A function
`grad` differentiates draws nothing, and a step's loss and its gradient
must be one pass's, so in pie the masks are an input: `nn.dropout_noise`
draws every mask a training pass over `x` draws, in PyTorch's order, and
the forward pass reads them as its `noise`. `nn.eval[model]` is the model
in evaluation, which drops nothing and draws nothing.

```crow
use math
use nn
use nn.rng

model = nn.sequential[[nn.linear[4, 3], nn.dropout[0.5], nn.linear[3, 2]]]
drawn = nn.parameters[model, rng.manual_seed[7]]
x = math.ones[5, 4]
noise = nn.dropout_noise[model, drawn.generator, x]
log[len[noise.values]]
on = nn.forward[model, drawn.values, x, noise = noise.values]
off = nn.forward[nn.eval[model], drawn.values, x]
log[on.shape]
log[off.shape]
```

```text
15
[5, 2]
[5, 2]
```

## Attention and transformers

`F.scaled_dot_product_attention` is softmax(q·kᵀ/√d + mask)·v, with
`is_causal` the triangle; `nn.multihead_attention`, the transformer's
encoder and decoder layers, their stacks and `nn.transformer` are
PyTorch's modules, run by `nn.attend`, `nn.encode`, `nn.decode` and
`nn.transform`. A mask is a float mask: 0 where a position takes part, −∞
where it does not.

```crow
use math
use nn
use nn.functional as F
use nn.rng

q = tensor[[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]
out = F.scaled_dot_product_attention[q, q, q, is_causal = true]
log[flatten[out]]
layer = nn.transformer_encoder_layer[8, 2, dim_feedforward = 16, dropout = 0.0, batch_first = true]
theta = nn.parameters[layer, rng.manual_seed[0]].values
log[nn.size[layer]]
src = math.ones[1, 5, 8]
log[nn.encode[layer, theta, src].shape]
```

```text
[1, 0, 0.33023845067334306, 0.6697615493266569, 0.7517449217422769, 0.7517449217422769]
600
[1, 5, 8]
```

The attention's numbers are `F.scaled_dot_product_attention(q, q, q,
is_causal=True)`'s, and the layer holds the 600 parameters of
`nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0,
batch_first=True)`, drawn as it draws them.

## Data

`nn.data` is `torch.utils.data`'s loaders and samplers. An epoch answers
its batches as positions and the loader past them, for the next epoch;
`data.rows` gathers a tensor's rows at a batch's positions:

```crow
use math
use nn.data

loader = data.loader[10, batch_size = 4, shuffle = true, seed = 3]
first = data.epoch[loader]
for b in first.batches
    log[b]
second = data.epoch[first.loader]
log[second.batches[0]]
x = math.reshape[math.arange[0.0, 20.0, 1.0], 10, 2]
log[data.rows[x, first.batches[2]]]
```

```text
[7, 4, 2, 9]
[1, 5, 0, 8]
[6, 3]
[1, 2, 0, 5]
[[12, 13], [6, 7]]
```

These are the batches `DataLoader(range(10), batch_size=4, shuffle=True,
generator=torch.Generator().manual_seed(3))` hands out, epoch after
epoch; a loader with no seed draws from the process's generator, as
PyTorch's does.

## Weights in and out

`nn.safetensors` writes the bytes Hugging Face's `safetensors.torch`
writes, and reads every file it writes; with `nn.state` and
`nn.from_state`, a model PyTorch trained runs in pie on its own weights.
`save_file` and `load_file` do the same through a file. The 308 bytes are
those `safetensors.torch.save` writes of the same state dict in float32.

```crow
use math
use nn
use nn.rng, nn.safetensors

model = nn.sequential[[nn.linear[2, 3], nn.tanh[], nn.linear[3, 1]]]
theta = nn.parameters[model, rng.manual_seed[1]].values
bytes = safetensors.save[nn.state[model, theta], dtype = 'F32']
log[len[bytes]]
back = nn.from_state[model, safetensors.load[bytes]]
log[max[abs[back - theta]] lt 0.0000001]
```

```text
308
true
```

## On the GPU

A program that writes `gpu[]` before it makes its tensors trains on the
GPU with nothing else changed: theta, the batch and everything computed
from them are placed there, and the forward pass, `grad`'s backward pass
and the optimiser's steps run there (crow's spec, GPU Support). The GPU
computes in f32, so its run agrees with the CPU's to f32's digits; where
no GPU answers, every tensor stays on the CPU and the program runs as
before.

```crow
use math
use nn
use nn.functional as F
use nn.optim, nn.rng

fn loss[model, theta, x, y]
    return F.mse_loss[nn.forward[model, theta, x], y]

gpu[]
model = nn.sequential[[nn.linear[2, 16], nn.tanh[], nn.linear[16, 1]]]
x = tensor[[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]]
y = tensor[[[0.0], [1.0], [1.0], [0.0]]]
opt = optim.adam[nn.parameters[model, rng.manual_seed[7]].values, lr = 0.05]
first = loss[model, opt.theta, x, y]
for k in range[200]
    opt = optim.step[opt, grad[loss][model, opt.theta, x, y][0]]
log[loss[model, opt.theta, x, y] lt first / 10.0]
log[opt.theta.device eq x.device]
```

```text
true
true
```

Each operation on the GPU costs tens of microseconds however small it is,
and reading a number back is a round trip of about a millisecond and a
half, so a small model like this one trains faster on the CPU; the GPU is
for wide layers, long sequences and big batches (`docs/pie-design.md`, On
the GPU).

## gpt

`packages/gpt` is nanoGPT on `nn`: the model as `nn.module` parts run by
its own forward function, trained with AdamW from the parameters nanoGPT
starts from, its losses and its generated text held to nanoGPT's
(`packages/gpt/held.co`). `packages/gpt/example.co` trains one on a
character corpus and samples from it.

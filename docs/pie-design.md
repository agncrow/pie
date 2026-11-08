# Pie: design

Pie is Crow's home for numbers that are more than the language's own: the
statistics, the networks and their training, the classical models, the
optimisers, graphs and retrieval. It is written in Crow, and every answer
is the same in `crow run -i` and `crow run`.

## Where the line with crow is (2026-11-07, Vijay)

Crow keeps what its compiler must know and what only Rust can be: the
tensor and the frame (kinds the compiler types, with their shapes and
columns), their operators, the kernels under them (element-wise math,
reductions, the matrix product, linear algebra, random numbers, the GPU)
and `grad`. Pie keeps every algorithm written over them. Pie depends on
crow; crow never depends on pie, so nothing is in both.

`grad` stays in the language because a network is written with the
language's own operators: `x * w + b` is differentiated because the
compiler and both backends trace `+` and `*`. Crow has no operator a
library can redefine, so an autograd written in Crow would have made
every model `add[mul[x, w], b]`. Pie has no tape of its own — one
autograd, crow's — and what `grad` could not yet differentiate is added to
`grad` in crow.

Later, one module at a time and measured before anything is deleted, the
library-shaped kernels (signal processing, special functions, sparse
arrays, the decompositions) may move into pie as Crow, as Magpie's kernels
did.

## The packages

| Package | What it is | Held to |
|---|---|---|
| `nn` | tensors' activations, losses and layers; modules, the model as one parameter vector, initialisers (`nn.init`), the functional forms; `nn.optim` (SGD, Adam, AdamW, Adamax, NAdam, RAdam, RMSprop, Adagrad, Adadelta, ASGD, Rprop) and `nn.lr_scheduler`; gradient clipping; torch's generator (`nn.rng`); loaders, samplers and splits (`nn.data`); weights saved and loaded as safetensors (`nn.safetensors`) | PyTorch (float64), and the safetensors library byte for byte |
| `stats` | descriptive statistics, correlation, tests, fits | scipy |
| `models` | the classical models, metrics, cross-validation | scikit-learn |
| `optimize` | scalar and multivariate minimisers, roots | scipy.optimize |
| `symbolic` | symbolic regression | PySR |
| `gpt` | nanoGPT | PyTorch |
| `graph` | graphs, paths, ranks | networkx |
| `retrieval` | TF-IDF, BM25 | scikit-learn, rank_bm25 |
| `vectors` | an exact vector index | faiss |

## A model is one parameter vector

PyTorch's module holds its parameters as tensors it owns and changes in
place, and `loss.backward()` writes a gradient into each. Crow's values do
not change under the program's feet, and a list holds one kind, so pie
holds a model as two values:

- the **spec**: a record of the layers — each a `Layer` of its kind, its
  sizes and options, and where its parameters lie;
- the **parameters**: ONE vector, `theta`, every layer's weights laid end
  to end (as `torch.nn.utils.parameters_to_vector` lays them).

A forward pass reads a layer's weight as `reshape[take[theta, at], rows,
columns]`; `grad[loss][theta, x, y]` answers one vector, the gradient of
every parameter at once; an optimiser is arithmetic over vectors (Adam is
element-wise, so the whole model is one update, exactly PyTorch's per
tensor update); `nn.named[spec, theta]` gives the parameters back by
PyTorch's names (`0.weight`, `0.bias`, …), which is the state dict, and
loading PyTorch's state dict lays the tensors into `theta` in the same
order. The one place this reads differently from PyTorch is the step:

```
g = grad[loss][theta, x, y]          --- loss.backward()
opt = optim.step[opt, theta, g]      --- optimizer.step()
theta = opt.theta
```

## A forward pass hands its numbers flat

A tensor's rank is in its type (crow's Design Note "rank is in the type"):
a function's copy reads what it is handed by the rank its type says, and a
tensor whose rank nothing states is checked where it is handed — it reaches
the copy of its rank, or the call stops. A network's layers change the
rank (a linear layer keeps it, `flatten` lowers it, an embedding raises
it), so a forward pass recursing over a sequential model cannot hold one
rank per call. Each layer therefore takes and hands its numbers flat, as a
vector beside their sizes, a list of ints (`_run[layer, theta, v, dims]`),
and lays them out itself, with literal sizes, at the rank it computes at:
`reshape[v, len[v] / ins, ins]` for a linear layer, `[N, C, M]` for a
layer over channels, `[before, size, after]` around a softmax's axis.
`_after[layer, dims]` says the sizes a layer hands on.

`nn.forward` lays the last layer's numbers out at the sizes the model ends
with, so its answer has the right shape but no rank a type states: state
it before indexing it, comparing it or handing it to a function that has
copies for several ranks (`reshape[y, n, k]`, or `flatten[y]` for a
vector), as `held_models.co` does. A loss takes it as it is.

## Random numbers are torch's

A seed in PyTorch names numbers: `torch.manual_seed(0)` and then
`nn.Linear(3, 4)` is one weight matrix on every machine. Pie draws from
the same generator, so a seed names the same numbers in pie. `nn.rng` is
torch's CPU generator written in Crow: a Mersenne Twister (MT19937)
seeded as torch seeds it, a 64-bit draw made of two 32-bit words, and
each distribution made from them as ATen makes it — a uniform double from
53 bits as `fma(u, b − a, a)`; normals by Box–Muller, one pair at a time
below sixteen numbers (the second of a pair kept in the generator for the
next draw, as torch keeps it) and sixteen at a time from sixteen on;
`randperm` by torch's swaps; `randint`, `bernoulli`, `exponential` and
`multinomial` as ATen draws them. `reference/rng_reference.py` writes the
model of the generator, checks it against torch, and writes what
`held_rng.co` holds pie to: every int and every float the same.

A generator is a value: a draw takes one and answers its numbers beside
the generator moved past them (`rng.Floats[values, generator]`), so what a
program draws is decided by what it hands each draw, in both modes alike.
`nn.parameters[model, g]` draws a model's parameters as its PyTorch
modules draw them when they are made — a linear layer's weight by
kaiming's bound and then its bias, an embedding's normals, a norm's ones
and zeros — so a pie model starts from the numbers a PyTorch model starts
from after `torch.manual_seed(s)`, in float64
(`torch.set_default_dtype(torch.float64)`). A float32 model draws
float32's numbers (24 bits for a uniform, single precision's functions for
a normal); pie holds float64, so a float32 model's weights come in
through safetensors instead.

A DataLoader draws too — a base seed for its workers at every epoch, then
its sampler's order — and which generator it draws from depends on what
it was handed: the loader's own (which the sampler `shuffle` made
shares), a sampler's own, or the process's (a random sampler through a
generator seeded from it). `nn.data` follows each path, so a seeded pie
loader hands out PyTorch's batches in PyTorch's order, epoch after epoch
(`held_data.co`). An epoch answers its batches as positions; `data.rows`
gathers a tensor's rows at them.

## Weights in and out

`nn.safetensors` reads and writes the format PyTorch models are shared in
(Hugging Face's): a file pie saves is the file `safetensors.torch` writes
of the same tensors, byte for byte — in F64, F32, F16, BF16 or I64, each
number rounded as torch's `.to(dtype)` rounds a float64 (half precision at
once, bfloat16 through single precision) — and pie loads every dtype the
format has. `nn.state[model, theta]` and `nn.from_state[model, state]`
are the state dict both ways, so a model PyTorch trained runs in pie on
its own weights (`held_safetensors.co`: a model torch made and saved, run
in pie, is within 1e-15 of torch's answer).

## What crow had to grow

`grad` reads a tensor's numbers only through the operations on its list.
To build every layer over one vector, crow's `grad` differentiates `take`
(numbers at positions, the gather under an embedding, a convolution's
patches, a pooling window, a layer's slice of `theta`) — its backward
spreads the gradient back to the positions, adding where a position is
taken twice. A convolution is then `take` (im2col) and `*`; pooling is
`take` and `max` along an axis; attention's heads are takes of the
projections; a recurrent layer is a loop over the steps, which `grad`
runs in order.

## Held to PyTorch

Each package's `reference/reference.py` runs PyTorch in float64 on fixed
inputs and writes what pie is held to (`reference/expected/`); `held.co`
compares. The references run under the virtual environment named by
`PIE_PYTHON` (`~/.venvs/gpt/bin/python`, torch 2.8, by default); the
expected files are committed, so the tests need no Python.

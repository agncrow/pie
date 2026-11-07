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
| `nn` | tensors' activations, losses and layers; modules, the model as one parameter vector, initialisers, the functional forms | PyTorch (float64) |
| `optim` | SGD (momentum, Nesterov, weight decay), Adam, AdamW, RMSprop, Adagrad, and the learning-rate schedules; gradient clipping | PyTorch |
| `data` | datasets of tensors or frames, batches, seeded shuffles, splits | PyTorch's DataLoader order, given the order |
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

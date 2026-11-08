# Pie

Crow's numbers beyond the language's own, written in Crow: networks and
their training, as PyTorch builds and trains them — tensors' layers and
losses, a model as one parameter vector differentiated by `grad`, the
optimisers and their schedules, datasets and batches, saved weights — with
the statistics, the classical models, the optimisers of scipy, symbolic
regression, graphs and retrieval beside them. Every answer is the same in
`crow run -i` and `crow run`, and each package is held to the library it
follows: PyTorch in float64, scikit-learn, scipy, networkx, faiss.

| Package | What it is |
|---|---|
| [`packages/nn`](packages/nn) | activations, losses, layers (linear, convolutions, poolings, paddings, resamplings, RNN/LSTM/GRU, norms, embeddings), modules, initialisers, optimisers and their schedules — PyTorch's `torch.nn`, `torch.nn.functional`, `torch.nn.init` and `torch.optim` |
| [`packages/stats`](packages/stats) | descriptive statistics, correlation, tests and fits — scipy.stats |
| [`packages/models`](packages/models/README.md) | the classical models, metrics, cross-validation — scikit-learn |
| [`packages/optimize`](packages/optimize/README.md) | minimisers and roots — scipy.optimize |
| [`packages/symbolic`](packages/symbolic/README.md) | symbolic regression — PySR |
| [`packages/gpt`](packages/gpt/README.md) | nanoGPT, held to PyTorch |
| [`packages/graph`](packages/graph/README.md) | graphs, paths, ranks — networkx |
| [`packages/retrieval`](packages/retrieval/README.md) | TF-IDF and BM25 — scikit-learn, rank_bm25 |
| [`packages/vectors`](packages/vectors/README.md) | an exact vector index — faiss |

A program installs a package as any dependency and calls it qualified:

```
deps = map[nn = 'github.com/agncrow/pie/packages/nn@v0.1.0']
```

```
use math
use nn
use nn.functional as F
use nn.optim, nn.rng

fn loss[model, theta, x, y]
    return F.mse_loss[nn.forward[model, theta, x], y]

model = nn.sequential[[nn.linear[3, 4], nn.relu[], nn.linear[4, 1]]]
x = tensor[[[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]]
y = tensor[[[1.0], [0.0]]]
opt = optim.sgd[nn.parameters[model, rng.manual_seed[0]].values, lr = 0.1]
for k in range[50]
    opt = optim.step[opt, grad[loss][model, opt.theta, x, y][0]]
log[loss[model, opt.theta, x, y]]
```

A model is a record of its layers beside one vector of parameters,
`theta`, drawn from a torch generator (`nn.rng`) as PyTorch's modules draw
it — the same numbers, after `torch.manual_seed(0)` in float64; `grad[loss]`
answers the gradient of every parameter at once, and an optimiser's step is
arithmetic over that vector. The same model and steps in PyTorch, in
float64 from `torch.manual_seed(0)`, end at the loss this prints to twelve
digits. A loader (`nn.data`) hands out PyTorch's
batches in PyTorch's order, and `nn.safetensors` saves and loads weights in
the format PyTorch models are shared in.

`./test.sh [crow]` runs every package's tests and examples in both modes.
[docs/pie-design.md](docs/pie-design.md) says where the line with crow
lies and why a model is one parameter vector.

Pie is source-available under the Business Source License 1.1
([LICENSING.md](LICENSING.md) says what that means in plain words).

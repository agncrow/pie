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
| [`packages/nn`](packages/nn) | activations, losses, layers, modules, initialisers — PyTorch's `torch.nn` |
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
use math, nn

log[nn.softmax[[1.0, 2.0, 3.0]]]
```

`./test.sh [crow]` runs every package's tests and examples in both modes.
[docs/pie-design.md](docs/pie-design.md) says where the line with crow
lies and why a model is one parameter vector.

Pie is source-available under the Business Source License 1.1
([LICENSING.md](LICENSING.md) says what that means in plain words).

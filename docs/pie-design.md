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
| `nn` | tensors' activations, losses and layers — linear, convolutions and their transposes, poolings, paddings, resamplings, recurrent layers (RNN, LSTM, GRU), attention (scaled dot-product, multi-head) and transformers (encoder and decoder layers and stacks, `nn.Transformer`), norms, embeddings, dropouts; modules, the model as one parameter vector, initialisers (`nn.init`), the functional forms; `nn.optim` (SGD, Adam, AdamW, Adamax, NAdam, RAdam, RMSprop, Adagrad, Adadelta, ASGD, Rprop) and `nn.lr_scheduler`; gradient clipping; torch's generator (`nn.rng`); loaders, samplers and splits (`nn.data`); weights saved and loaded as safetensors (`nn.safetensors`) | PyTorch (float64), and the safetensors library byte for byte |
| `stats` | descriptive statistics, correlation, tests, fits | scipy |
| `models` | the classical models, metrics, cross-validation | scikit-learn |
| `optimize` | scalar and multivariate minimisers, roots | scipy.optimize |
| `symbolic` | symbolic regression | PySR |
| `gpt` | nanoGPT's model and training, on nn (docs/gpt-design.md) | nanoGPT in PyTorch |
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

A forward pass reads a layer's weight as `reshape[slice[theta, at, at +
n], rows, columns]`; `grad[loss][model, theta, batch]` answers one vector,
the gradient of every parameter at once (the model and the batch are
records, constants of the run); an optimiser is arithmetic over vectors
(Adam is element-wise, so the whole model is one update, exactly
PyTorch's per tensor update); `nn.state[model, theta]` gives the
parameters back by PyTorch's names (`0.weight`, `0.bias`, …), which is the
state dict, and `nn.from_state[model, state]` lays a state dict's tensors
into `theta` by those names. The one place this reads differently from
PyTorch is the step:

```
g = grad[loss][model, opt.theta, batch]   --- loss.backward()
opt = optim.step[opt, g]                  --- optimizer.step(); opt.theta is the new theta
```

A model whose layers run one after another is `nn.sequential`, and
`nn.forward` runs it. A model whose forward pass is code — a residual
stream, a weight shared by two layers, nanoGPT's `forward` — is
`nn.module[map[…]]` (torch's ModuleDict: parts by name) and
`nn.module_list[[…]]` (ModuleList: parts by position), run by a function
the program writes, each part reached by `nn.part[model, name]` and run by
`nn.forward`; its state's names are the parts' paths
(`transformer.h.0.attn.c_attn.weight`), as torch's are.

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

## Windows: convolutions, poolings, paddings, resamplings

A convolution, a pooling, a padding, a fold and a resampling each read
their input at positions. `nn.windows` computes the positions as tensors
of whole numbers, in the order PyTorch's kernels read them, and the layer
is a gather and then a matrix product or a reduction:

- a convolution (1, 2 and 3 spatial axes; stride, padding, dilation,
  groups) gathers each window's numbers as a column (im2col, `F.unfold`)
  and multiplies the columns by the weight;
- a transposed convolution multiplies each input position by the weight
  and adds the products into the output positions they reach (col2im,
  `F.fold`), in the order PyTorch adds them;
- a pooling reduces each window: its largest number, its mean (with
  `ceil_mode`, `count_include_pad` and `divisor_override`), its Lp norm,
  or PyTorch's adaptive windows;
- a padding (constant, reflection, replication, circular) reads the
  position each output number copies;
- `F.interpolate` reads, for 'nearest' and 'nearest-exact', the source
  position PyTorch computes in single precision (`nn.single` rounds a
  float64 to float32 as torch's float arithmetic does), and for 'linear',
  'bilinear' and 'trilinear' weighs the two numbers about it along each
  axis, the corners nested as PyTorch nests them; 'area' is adaptive
  average pooling. 'bicubic' is not there yet.

A position that reads no number of the input, padding or a window past
the input's end, is the input's length: the one number a layer appends
after the input (0 for a convolution, −∞ for a max pooling, a constant
pad's value). `grad` differentiates a gather as it does any gather, so no
layer over windows needs anything of its own in `grad`: the gradient goes
back to the positions read, added where windows overlap.

## Recurrent layers

`nn.rnn`, `nn.lstm` and `nn.gru` are PyTorch's: `num_layers` layers, each
over the one before's output, both ways where `bidirectional`, dropout
between the layers in training, batch-first or not. Their parameters are
named, laid and drawn as PyTorch's (`weight_ih_l0`, …, `_reverse` for the
other direction; each uniform within ±1/√hidden), so a seeded pie layer
starts from torch's numbers. A layer is a loop over the steps, which
`grad` runs in order; a step is the functional cell (`F.rnn_tanh_cell`,
`F.rnn_relu_cell`, `F.lstm_cell`, `F.gru_cell`), the gates computed as
PyTorch computes them. `nn.forward` answers the output, and
`nn.recurrent[layer, theta, x, h0, c0]` answers the output with the last
hidden and cell states (`output, (h_n, c_n) = lstm(x, (h0, c0))`).
`held_spatial.co` holds every convolution, pooling, padding, resampling and
recurrent layer to PyTorch, value and gradients, to 1e-12.

## Attention and transformers

`F.scaled_dot_product_attention` is softmax(q·kᵀ·scale + mask)·v over the
last two axes, the axes before them a batch, `is_causal` the triangle.
`nn.multihead_attention` is `nn.MultiheadAttention` — its parameters
PyTorch's (`in_proj_weight`, or `q_proj_weight`, `k_proj_weight` and
`v_proj_weight` where the keys or the values are of other sizes;
`in_proj_bias`; `bias_k` and `bias_v`; `out_proj`), drawn as PyTorch draws
them — run by `nn.attend` (`F.multi_head_attention_forward`): heads,
key padding, `add_zero_attn`, batch-first, the weights averaged over the
heads or not. `nn.transformer_encoder_layer` and
`nn.transformer_decoder_layer` (post-norm or `norm_first`, relu or gelu),
their stacks (`nn.transformer_encoder`, `nn.transformer_decoder`: copies of
the layer's parameters, as torch clones them) and `nn.transformer` (every
parameter of two axes or more drawn again xavier_uniform, as
`nn.Transformer` does) run by `nn.encode`, `nn.decode` and
`nn.transform`. A mask is a float mask — 0 where a position takes part,
−∞ where it does not — added to the scores, a padding mask likewise; an
encoder or a decoder reads a mask that is the causal triangle as causal,
as torch's `_detect_is_causal_mask` does, since torch then takes another
path through the attention. `held_attention.co` holds each function and
layer to PyTorch, value and gradients, to 1e-12 (parity 008 holds it in
both modes).

## Dropout reads the noise drawn for the pass

A torch dropout draws its mask inside the forward pass. A function `grad`
differentiates draws nothing it is differentiated through, and a step's
loss and its gradient must be one pass's, so in pie the masks are an input:
`nn.dropout_noise[model, g, x]` draws every mask a training pass over x
draws — each dropout's `bernoulli(1 − p)` over its numbers, an attention's
over its weights, the dropout between a recurrent layer's layers — from a
torch generator, in the order and the layout PyTorch's forward pass draws
them (a batch-first transformer layer drops its attention's output as the
attention lays it, time-major), and the forward pass takes them as
`noise`. A layer in training with a dropout refuses a pass without its
noise; `nn.eval[model]` makes a model that draws none. A program running
its own forward function draws each part's noise in the order it runs the
parts, as gpt's `dropout_noise` does for nanoGPT's.

## On the GPU

A program that writes `gpu[]` before it makes its model trains it on the
GPU with nothing else changed: theta, the batch and every tensor computed
from them are placed there (crow's spec, GPU Support), the forward pass,
`grad`'s backward pass and the optimiser's step run there — gathers and
their gradients, a softmax's max and its gradient among them — and a
number comes back only where the program reads one. The GPU computes in
f32, so a run agrees with the CPU's to f32's digits rather than exactly:
`nn/held_gpu.co` trains a perceptron six steps of Adam on both and holds
their losses within 1e-4 (they are 3e-8 apart), and gpt's first gradient
is 2e-7 of its largest number apart. AdamW then carries that difference
on, as it carries any difference in the last digits: forty steps of gpt
end 0.5% apart.

Measured on an M4 Max, with other work running: a perceptron of 256 →
1024 → 1024 → 10 on a batch of 512 takes 5–7 ms an Adam step on the GPU
against 50–70 ms on the CPU, and a product of two 2048 × 2048 matrices
runs at 1.5 TFLOP/s against faer's 0.2 in float64. Each operation costs
10–50 µs of the GPU's time however small it is, and a read of a number is
a round trip of about 1.5 ms, so a small model trains faster on the CPU:
gpt's example, two layers 32 wide over blocks of 16, takes 0.7 s for 40
steps there and 8 s on the GPU. The GPU is for models whose operations
are large — wide layers, long blocks, big batches.

## What crow had to grow

`grad` reads a tensor's numbers only through the operations on its list,
and pie's layers needed more of them than crow had. What it could not
differentiate was added to `grad` in crow (pie has no tape of its own):

- a tensor read at a tensor of positions (`v[positions]`, the gather
  under an embedding, a convolution's columns, a pooling's windows, a
  padding), its backward adding the gradient back where a position is read
  twice; a mask and `where`'s sides; `log1p` and `expm1`;
- a collecting loop, run in order, and a list or a record holding traced
  tensors (a recurrent layer's steps and its state), lists of lists among
  them; `stack` of traced tensors along an axis (`torch.stack`), and a list
  of traced numbers made a tensor;
- what carries no derivative meets any builtin inside the function
  differentiated: a table of positions built from a tensor's sizes, a
  `str` of a size, a draw;
- a value that can fail holds its traced number, as a size read from
  parsed text does in the spatial check, and `log` prints a traced value;
- every number and tensor a differentiated function's locals hold is held
  traced throughout the compiled copy: a local given a plain value first
  and a traced one later (`p = layer.floats[0]`, then `p = 0.0`) had not
  built.

- the GPU under every operation a network computes (crow's parity
  520–522): a tensor made of a list placed by `gpu[]`, an operation
  meeting a view, a broadcast or an operand on the CPU run there, the
  matrix product a kernel of 64 × 64 tiles reading each operand through
  its strides, reductions along a long axis shared by a workgroup, a
  gather and its gradient, the gradient of a softmax's max found there,
  and the loss's sum left there until read — each had sent its numbers
  back to the CPU, and a network trained under `gpu[]` had never reached
  the GPU.

Each is a parity program in crow, in both modes (crow's parity 490–522).
And the interpreter had asked the table for a row's written name by a scan
inside a scan at every traced call, three fifths of a GPT's training time:
the question reads an index now, as the table's other per-call questions
do.

## Held to PyTorch

Each package's `reference/reference.py` runs PyTorch in float64 on fixed
inputs and writes what pie is held to (`reference/expected/`); `held.co`
compares. The references run under the virtual environment named by
`PIE_PYTHON` (`~/.venvs/gpt/bin/python`, torch 2.8, by default); the
expected files are committed, so the tests need no Python.

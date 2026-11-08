# A GPT in Crow, held to nanoGPT

`packages/gpt` is nanoGPT (github.com/karpathy/nanoGPT) — its `model.py`
and the training of its `train.py` — written in Crow on pie's `nn`, module
for module and name for name, and held to nanoGPT in float64: the same
weights from the same seed, the same logits, the same losses step by step,
the same ids drawn. This note is the design; the package is the code.

## Why it was written again

The first port (crow's `packages/gpt`, 2026-10-11) spelled its 36 weight
tensors out as parameters of the loss, because `grad` took tensors one
parameter each and a list of tensors was not one. Pie's nn holds a model as
a record of its layers beside ONE vector of parameters (pie-design.md, "A
model is one parameter vector"), which `grad` differentiates whole, so a
model of any depth is a value. The port on nn is nanoGPT's own structure:

- its modules are nanoGPT's, so its state is nanoGPT's state: a checkpoint
  loads by its names, with no table mapping one naming to another;
- its weights are drawn from a torch generator in the order nanoGPT's
  `GPT(config)` draws them, so a seed names the same model in both;
- it trains with nanoGPT's AdamW over nanoGPT's parameter groups, and
  generates with nanoGPT's temperature, top-k and multinomial — from the
  generator, so a sampled continuation is held as exactly as a greedy one;
- dropout is nanoGPT's, drawn where nanoGPT's forward pass draws it.

## The model is nanoGPT's modules

`gpt[config]` builds what `GPT(config)` builds, from nn's layers:

```
transformer = nn.module[map[
    wte = nn.embedding[vocab_size, n_embd],
    wpe = nn.embedding[block_size, n_embd],
    drop = nn.dropout[dropout],
    h = nn.module_list[blocks],          --- transformer.h.0, .1, …
    ln_f = nn.layer_norm[[n_embd], bias = bias]]]
```

each block a module of `ln_1`, `attn` (`c_attn`, `c_proj`, `attn_dropout`,
`resid_dropout`), `ln_2` and `mlp` (`c_fc`, `gelu`, `c_proj`, `dropout`).
`nn.module_list` is torch's ModuleList: its parts named by position and run
by the forward function the program writes. The state's names are
nanoGPT's — `transformer.h.0.attn.c_attn.weight`, `transformer.ln_f.weight`,
… — and `lm_head.weight` is `transformer.wte.weight`, as nanoGPT ties them:
the model holds the one tensor, `state` writes it under both names, and
`load` reads wte's (nn.from_state reads the names the model has and passes
over the others). The reference's checkpoint leaves `lm_head.weight` out,
since safetensors keeps no tensor twice.

The forward pass is a function the program writes, as nanoGPT's `forward`
is: `logits` adds `wte(idx)` and `wpe(arange(t))`, drops them, runs each
block — `x + attn(ln_1(x))`, then `x + mlp(ln_2(x))` — and `ln_f`, and
multiplies by wte's weight. Attention splits `c_attn`'s output into
queries, keys and values and views them as heads `[B, nh, T, hs]` — `split`,
`view` and `transpose(1, 2)` are one gather at computed positions
(`_heads`), and the heads are joined back by another (`_joined`) — around
`F.scaled_dot_product_attention` with `is_causal`. One copy of a function
holds one rank (pie-design.md, "A forward pass hands its numbers flat"), so
the last product runs on rows `[B·T, C]` and is laid back as `[B, T, V]`.

## The weights are nanoGPT's draws

`parameters[net, g]` reproduces the order of `GPT.__init__`:

1. each module is made as torch makes it, drawing its default: an
   embedding's normal, a linear layer's kaiming uniform and its bias's
   uniform (`_made_linear`) — numbers made only to be replaced, drawn so
   the generator moves past them;
2. `apply(_init_weights)` meets the modules in order — wte, wpe, each
   block's c_attn, c_proj, c_fc and c_proj, then lm_head — and draws each
   weight normal of deviation 0.02, every bias 0; wte's numbers are
   lm_head's draw, the last, since the two are one tensor;
3. each `c_proj.weight` is drawn again, of deviation 0.02/√(2·n_layer)
   (GPT-2's scaling for depth).

A norm's weight is 1. The weights agree with torch's to two units in the
last place: torch's normal goes through libm's log, cos and sin, whose
last digit two builds may round apart (nn.rng holds a draw to one unit),
and the scaling by the deviation rounds once more — 22 of the 27,136
numbers differ, the widest by 1.55 units.

## Training and generation

`configure_optimizers` is nanoGPT's: every parameter of two axes or more
decayed, the norms' weights and the biases not — two parameter groups of
`nn.optim.adamw` over the ranges of theta they name (`nn.ranges`), CPU's
single-tensor AdamW. `train.get_batch` draws the offsets with
`torch.randint(len(data) − block_size, (B,))`; a step is

```
loss = batch_loss[net, o.theta, b]
o = optim.step[o, grad[batch_loss][net, o.theta, b]]
```

the batch a record, so `grad` answers theta's gradient alone. `generate`
crops to the last block_size ids, divides the last position's logits by
the temperature, sets those below the k-th largest to −∞, and draws from
the softmax by `torch.multinomial` (nn.rng's: an exponential for each
number, the largest ratio drawn), one draw per sequence in order.

**Dropout reads the noise drawn for the pass.** A torch module draws its
dropout masks inside its forward pass; a Crow function differentiated by
`grad` draws nothing, and the loss and its gradient must come from one
pass. So, as nn's layers do (pie-design.md), the noise is an input:
`dropout_noise[net, g, b, t]` draws what a training pass over `[b, t]`
draws, in nanoGPT's order — transformer.drop's over the embeddings, then
for each block the attention's over its weights `[B, nh, T, T]`
(`dropout_p` of scaled_dot_product_attention), resid_dropout's and the
MLP's, each a `bernoulli(1 − p)` of its numbers — and the forward pass
reads it. A batch carries its noise (`train.Batch`), drawn after its
offsets as train.py's forward pass draws after `get_batch`; `generate`
draws a pass's noise before its multinomial. A model is made in training,
as a torch module is; `model.eval[net]` draws none, and a model in
training with a dropout refuses a pass without its noise.

## Held to

`held.co` against `reference/reference.py` — nanoGPT's model written again
in PyTorch, float64, two layers, two heads, 32 wide, a block of 16, no
biases, over the characters of a slice of tiny Shakespeare:

- the weights drawn from seed 1337, to two units in the last place;
- the logits and the loss of the first block at the reference's own
  weights (its safetensors checkpoint, loaded by name), to 1e-9 relative;
- the loss before each of twenty steps of SGD, one block a step, and after
  them, to 1e-9; fifty ids generated at the weights they reach (seed 7,
  temperature 0.8, top-k 10), exactly;
- the loss before each of twenty steps of nanoGPT's AdamW (weight decay
  0.1, lr 0.01, betas 0.9 and 0.95) on batches of four blocks get_batch
  draws (seed 1338), to 1e-9;
- with shakespeare_char's dropout of 0.2: five steps of AdamW in training
  (seed 1339), to 1e-9; ten ids generated still in training (seed 9),
  exactly; the loss in eval after them, to 1e-9.

Not "to the last digit": a matrix product's accumulation order is the
BLAS's, which neither faer nor PyTorch states. The draws are held on Apple
silicon, where torch draws every distribution from its own generator in
order; a torch built with Intel's MKL may draw a bernoulli or an
exponential through MKL's generator (seeded by one draw from torch's), and
its dropout masks and samples are other numbers there. `crow test
--packages gpt` runs the tests and the example in both modes.

## What the port found

In crow, each fixed with its test (crow's parity 515 and the table's unit
tests):

- **A differentiated function's numbers are held traced throughout its
  copy.** A number or a tensor a local holds, given a traced value later —
  `p = layer.floats[0]`, then `p = 0.0` where the layer does not train;
  `wd = [2, 3, 4]`, then `wd = slice[…]` — had been a plain `f64` or tensor
  in the compiled traced copy, and the program did not build.
- **The interpreter asked the table for a row's written name by a scan,
  twice over, at every traced call** (`table::written_name`: each row's
  growing form found by a scan of its own): three fifths of the training's
  time. It reads an index built once now, and the example's CPU time
  halved; the lowering's per-call lookups by name (`core_row`,
  `row_taking`, `module`, `form_answer`, `refusal`, `miscounted`) read the
  index too, a fifth of the lowering.

In pie:

- `rng.multinomial` read its ratio at a rank no type stated, once
  `generate` had handed `rng.Floats` a matrix elsewhere (a record's field
  holding tensors of two ranks loses its rank): it flattens the ratio, and
  generation answers its own record, `Generated`.
- `chars.tokens` answered ids that can fail (a map read per character): it
  builds the list by `push`, which holds plain ids and refuses a character
  the vocabulary lacks where it is read.

The first port's findings stand in crow's history: a traced copy taking its
parameter immutable (parity 245), a test's call refining a parameter typed
with an unknown inside (`type_by_tests`), a list refused for holding two
kinds.

## What is next

GPT-2's own tokenizer, a byte-level BPE held to tiktoken, and GPT-2's
weights loaded by these names for inference; batched matrix products, so
attention runs heads as one product rather than a product per head and a
gather to lay them out. The same programs run under `gpu[]` (pie-design.md,
On the GPU), where the example's small model is slower than on the CPU —
each of its many small operations costs the GPU's fixed time — and the
heads' positions, computed on the device, are read back to be checked at
each gather: a model the size of nanoGPT's shakespeare_char is where the
GPU pays.

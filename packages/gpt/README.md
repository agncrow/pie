# gpt

nanoGPT (github.com/karpathy/nanoGPT) written in Crow on pie's `nn`,
module for module and name for name, and held to it in float64
(docs/gpt-design.md). A GPT is token and position embeddings, blocks of a
layer norm, causal self-attention and an MLP added to the stream, a final
norm and the logits through the embedding's own weight — nanoGPT's
`transformer.wte`, `transformer.h.0.attn.c_attn`, … — so a nanoGPT
checkpoint loads by its names, and its parameters are drawn from a torch
generator as nanoGPT's `GPT(config)` draws them. `grad` differentiates the
loss over the model's one parameter vector, and nanoGPT's AdamW
(`configure_optimizers`) steps it.

`held.co` holds the package to `reference/reference.py` (nanoGPT's model
written again, PyTorch in float64, no dropout): the weights drawn from seed
1337, each within two units in its last place; the logits and the loss at
the reference's own weights, read from its safetensors checkpoint; twenty
steps of SGD; fifty ids generate draws at temperature 0.8 and top-k 10; and
twenty steps of AdamW on batches get_batch draws — numbers to 1e-9
relative, the tokens and the ids exactly. Each answers the same in
`crow run -i` and `crow run`.

```
deps = map[gpt = 'github.com/agncrow/pie/packages/gpt@v0.1.0']
```

A program imports the parts it uses:

```
use nn.rng
use gpt.chars, gpt.model, gpt.train

net = model.gpt[model.config[block_size = 16, vocab_size = 59, n_layer = 2, n_head = 2, n_embd = 32, bias = false]]
theta = model.parameters[net, rng.manual_seed[1337]].values
opt = model.configure_optimizers[net, theta, 0.1, 0.01, 0.9, 0.95]
fitted = train.fit_batches[net, opt, ids, 400, 8, rng.manual_seed[1338]]
made = model.generate[net, fitted.opt.theta, prompt, 100, rng.manual_seed[42], temperature = 0.8, top_k = 10]
```

## The parts

- `chars.co` — a character tokenizer, as nanoGPT's `shakespeare_char`
  prepares its data: `vocab[text]`, `tokens[text, vocabulary]`,
  `text_of[ids, vocabulary]`.
- `model.co` — nanoGPT's `model.py`: `config[…]` (GPTConfig, GPT-2's
  defaults), `gpt[config]` (the modules), `parameters[net, g]` (the numbers
  `GPT(config)` draws from a torch generator), `load[net, state]` and
  `state[net, theta]` (a checkpoint by nanoGPT's names, `lm_head.weight`
  tied to `transformer.wte.weight`), `logits[net, theta, idx]`,
  `loss[net, theta, idx, targets]`, `generate[net, theta, idx,
  max_new_tokens, g, temperature, top_k]`, `configure_optimizers[net, theta,
  weight_decay, learning_rate, beta1, beta2]`.
- `train.co` — nanoGPT's `train.py`: `get_batch[ids, g, batch_size,
  block_size]`, `in_order[ids, i, block_size]`, `batch_loss[net, theta,
  batch]`, `fit[net, opt, ids, steps]` (the blocks in order) and
  `fit_batches[net, opt, ids, steps, batch_size, g]`, each answering the
  losses and the optimizer after the last step.
- `example.co` — nanoGPT's shakespeare_char run, made small: four hundred
  steps of AdamW on batches of eight blocks, then a hundred characters
  continued from the text's first line.
- `reference/` — `reference.py` and what it writes: the vocabulary, the
  tokens, the state at seed 1337 (`expected/init.safetensors`), the logits
  and the loss at it, the SGD and AdamW losses, the generated ids. To write
  them again, from pie's root: `~/.venvs/gpt/bin/python -I
  packages/gpt/reference/reference.py`.

"""gpt held to nanoGPT (github.com/karpathy/nanoGPT, model.py) in float64.

nanoGPT's model is written again below module for module, so that its
modules, their names and the order they draw from torch's generator are
nanoGPT's: GPTConfig, LayerNorm with an optional bias, CausalSelfAttention
over F.scaled_dot_product_attention, MLP, Block and GPT with its
_init_weights, forward, configure_optimizers and generate. The
configuration is small — a block of 16, two layers, two heads, 32 wide, no
biases, no dropout — over the characters of a slice of tiny Shakespeare.

The script writes, under expected/:

  vocab.txt            the vocabulary, the characters in order
  tokens.csv           the first 17 ids of the text
  init.safetensors     the state GPT(config) draws from seed 1337, as
                       nanoGPT's checkpoint holds it (lm_head.weight, which
                       is wte's, left out: safetensors keeps no tensor twice)
  logits0.csv          the logits of the first block at that state
  loss0.csv            the loss of the first block at that state
  sgd_losses.csv       the loss before each of twenty steps of SGD (lr 0.1),
                       one block a step, the blocks in order
  sgd_after.csv        the loss of the first block after them
  generated.csv        fifty ids generate draws after the first block at the
                       SGD's weights (seed 7, temperature 0.8, top_k 10)
  adamw_losses.csv     the loss before each of twenty steps of
                       configure_optimizers' AdamW (weight decay 0.1, lr
                       0.01, betas 0.9 and 0.95) from the same state, each
                       step a batch of four blocks get_batch draws (seed
                       1338)
  dropout_losses.csv   the same model with shakespeare_char's dropout, 0.2
                       (its weights the same state: a dropout draws nothing
                       when it is made): the loss before each of five steps
                       of AdamW in training (seed 1339), each forward pass
                       drawing its dropouts after its batch's offsets
  dropout_generated.csv  ten ids generate draws after the first block's
                       first eight ids, still in training (seed 9): each
                       step's forward pass draws its dropouts before the
                       multinomial
  dropout_after.csv    the loss of the first block after them, in eval

held.co loads the state and compares the weights gpt.model.parameters
draws, then each of the rest. Run from pie's root:
  ~/.venvs/gpt/bin/python -I packages/gpt/reference/reference.py
"""
import csv
import inspect
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import save_file

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))


# the model: nanoGPT's model.py ------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    # a layer norm whose bias may be left out
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # the embedding and the output share their weight
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # the residual projections, scaled for the depth (GPT-2's)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # (generation reads the last position alone)
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# the runs ---------------------------------------------------------------------

def write(name, rows):
    # (a header row: Crow's loaders read the first row as the names)
    path = os.path.join(HERE, 'expected', name)
    rows = list(rows)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['c%d' % i for i in range(len(rows[0]))])
        for r in rows:
            w.writerow(['%.17g' % x if isinstance(x, float) else x for x in r])


T = 16
text = open(os.path.join(HERE, 'data', 'shakespeare_50k.txt'), encoding='utf-8').read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
config = GPTConfig(block_size=T, vocab_size=len(chars), n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False)
open(os.path.join(HERE, 'expected', 'vocab.txt'), 'w', encoding='utf-8').write(''.join(chars))
write('tokens.csv', [[int(t) for t in data[0:T + 1].tolist()]])

torch.manual_seed(1337)
model = GPT(config)
state0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
save_file({k: v.contiguous() for k, v in state0.items() if k != 'lm_head.weight'}, os.path.join(HERE, 'expected', 'init.safetensors'))

x0 = data[0:T].unsqueeze(0)
y0 = data[1:T + 1].unsqueeze(0)
with torch.no_grad():
    logits0, loss0 = model(x0, y0)
write('logits0.csv', logits0[0].tolist())
write('loss0.csv', [[float(loss0)]])

# twenty steps of SGD, the blocks in order
opt = torch.optim.SGD(model.parameters(), lr=0.1)
losses = []
for step in range(20):
    xb = data[step * T:(step + 1) * T].unsqueeze(0)
    yb = data[step * T + 1:(step + 1) * T + 1].unsqueeze(0)
    _, loss = model(xb, yb)
    losses.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
write('sgd_losses.csv', [[x] for x in losses])
with torch.no_grad():
    _, after = model(x0, y0)
write('sgd_after.csv', [[float(after)]])

# fifty ids drawn after the first block
torch.manual_seed(7)
out = model.generate(x0, 50, temperature=0.8, top_k=10)
write('generated.csv', [[int(i) for i in out[0, T:].tolist()]])

# twenty steps of nanoGPT's AdamW from the same state, on batches of four
# blocks drawn as train.py's get_batch draws them
model.load_state_dict(state0)
opt = model.configure_optimizers(0.1, 0.01, (0.9, 0.95), 'cpu')
torch.manual_seed(1338)
losses = []
for step in range(20):
    ix = torch.randint(len(data) - T, (4,))
    xb = torch.stack([data[i:i + T] for i in ix])
    yb = torch.stack([data[i + 1:i + 1 + T] for i in ix])
    _, loss = model(xb, yb)
    losses.append(loss.item())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
write('adamw_losses.csv', [[x] for x in losses])

adamw_last = losses[-1]

# the same model with shakespeare_char's dropout: five steps of AdamW in
# training, ten ids generated in training, then the first block's loss in
# eval
config_d = GPTConfig(block_size=T, vocab_size=len(chars), n_layer=2, n_head=2, n_embd=32, dropout=0.2, bias=False)
torch.manual_seed(1337)
model_d = GPT(config_d)
assert all(torch.equal(model_d.state_dict()[k], state0[k]) for k in state0)
opt = model_d.configure_optimizers(0.1, 0.01, (0.9, 0.95), 'cpu')
torch.manual_seed(1339)
losses = []
for step in range(5):
    ix = torch.randint(len(data) - T, (4,))
    xb = torch.stack([data[i:i + T] for i in ix])
    yb = torch.stack([data[i + 1:i + 1 + T] for i in ix])
    _, loss = model_d(xb, yb)
    losses.append(loss.item())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
write('dropout_losses.csv', [[x] for x in losses])
torch.manual_seed(9)
out_d = model_d.generate(x0[:, :8], 10)
write('dropout_generated.csv', [[int(i) for i in out_d[0, 8:].tolist()]])
model_d.eval()
with torch.no_grad():
    _, after_d = model_d(x0, y0)
write('dropout_after.csv', [[float(after_d)]])

print('vocab', len(chars), 'loss0 %.12f' % float(loss0), 'sgd after %.12f' % float(after), 'adamw last %.12f' % adamw_last, 'dropout after %.12f' % float(after_d))
print('generated:', repr(''.join(chars[i] for i in out[0, T:].tolist())))

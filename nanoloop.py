"""
nanoloop: the smallest readable looped transformer.

A normal GPT stacks N different layers, like an assembly line with N stations.
A looped GPT has ONE station and sends the work through it K times. Same
weights, more passes, more "thinking" per token. You pick K at inference.

    tokens -> embed -> prelude -> [ core ] x K -> coda -> logits
                          |          ^
                          +-- e -----+   (input injection: re-add e every loop)

One file, PyTorch only. Train, sample and sweep K from the command line:

    python nanoloop.py --mode compare                  # chain task: standard vs looped
    python nanoloop.py --task shakespeare --steps 3000 # char-level Shakespeare
    python nanoloop.py --mode sample --loops 8         # sample from the last checkpoint
"""
import argparse
import os
import random
import time
import urllib.request
from dataclasses import asdict, dataclass, fields, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    task: str = "chain"          # "chain" (toy multi-hop lookup) or "shakespeare" (char LM)
    n_embd: int = 128            # width of every token's vector
    n_head: int = 4              # attention heads
    n_core: int = 1              # distinct layers inside the shared core (looped as a unit)
    loops: int = 4               # K: how many times the core runs (1 = a standard 3-layer GPT)
    inject: bool = True          # re-add the prelude output e before every loop
    random_loops: bool = False   # train with K ~ Uniform{1..loops} per step instead of fixed K
    hops: int = 4                # chain task: max lookups needed to answer
    block_size: int = 64         # context length in characters
    batch_size: int = 128
    steps: int = 6000
    lr: float = 5e-4
    weight_decay: float = 0.0
    seed: int = 1337
    device: str = "auto"         # auto picks mps, then cuda, then cpu
    out: str = "out/model.pt"
    vocab: str = ""              # filled in from the data; saved with the checkpoint


# ----------------------------------------------------------------------------- model

def rope(x):
    """Rotary position embedding: rotate each (even, odd) pair of q/k features by an
    angle proportional to the position. The dot product q.k then depends only on
    the DISTANCE between two tokens, so "look 2 characters back" is one skill that
    works everywhere, not one skill per absolute position."""
    T, D = x.shape[-2], x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device) / D))
    ang = torch.arange(T, device=x.device)[:, None] * freqs  # (T, D/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(-2)


class Block(nn.Module):
    """One transformer layer: tokens talk (attention), then each token thinks (MLP).
    Pre-LayerNorm + residual: x = x + f(norm(x)), so a block only adds an update."""

    def __init__(self, c):
        super().__init__()
        self.n_head = c.n_head
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)   # query, key, value in one matmul
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, C/heads) so each head attends independently
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)  # causal: no peeking ahead
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class LoopedGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        V = len(c.vocab)
        self.tok = nn.Embedding(V, c.n_embd)             # char id -> vector
        self.prelude = Block(c)                          # runs once: read the input
        self.core = nn.ModuleList(Block(c) for _ in range(c.n_core))  # runs K times: think
        self.coda = Block(c)                             # runs once: write the answer
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, V, bias=False)   # vector -> score per char

    def forward(self, idx, loops=None, targets=None):
        K = loops or self.c.loops
        e = self.prelude(self.tok(idx))  # e = "the embedded input" (positions come from rope)
        # With injection, h starts at zero and every loop sees h + e, so the
        # model never loses the question while it iterates. Without it, h starts
        # at e and the original input fades as the loops rewrite h.
        h = torch.zeros_like(e) if self.c.inject else e
        for _ in range(K):
            if self.c.inject:
                h = h + e
            for block in self.core:  # the SAME weights every iteration
                h = block(h)
        logits = self.head(self.ln_f(self.coda(h)))
        if targets is None:
            return logits, None
        # ignore_index=-1 lets the chain task score only the answer character
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, loops=None, temperature=1.0):
        for _ in range(n):
            logits, _ = self(idx[:, -self.c.block_size:], loops)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ----------------------------------------------------------------------------- data

CHAIN_VOCAB = ".abcdefghijklmnopqrstuvwxyz0123456789=;?"  # "." is padding


def chain_example(rng, max_hops, n_chains=2):
    """A pointer-chasing puzzle: two chains like a=7;b=a;c=b interleaved, then one
    question per variable, e.g. 'a=7;x=2;b=a;c=b;c?x?a?b?'. 'c?' needs 3 lookups
    in a row (c->b->a->7), 'a?' needs 1. Answers never appear in the input, so
    every '?' must be worked out. A model can resolve one more link of every
    chain per layer (or per loop), which is why this task is hungry for depth.
    Returns the text and [(index of '?', answer digit, hops needed)]."""
    lens = [rng.randint(1, max_hops) for _ in range(n_chains)]
    names = iter(rng.sample("abcdefghijklmnopqrstuvwxyz", sum(lens)))
    chains, answers = [], {}
    for n in lens:
        vs, digit = [next(names) for _ in range(n)], str(rng.randrange(10))
        targets = vs[1:] + [digit]  # vs[0]=vs[1], ..., vs[-1]=digit
        for i, v in enumerate(vs):  # vs[0] is n hops from the digit, vs[-1] is 1 hop
            answers[v] = (digit, n - i)
        # written like code: every variable is defined before it is used (a=7;b=a;c=b)
        chains.append([f"{v}={t}" for v, t in reversed(list(zip(vs, targets)))])
    statements = []
    while any(chains):  # interleave the chains at random, keeping each chain's order
        statements.append(rng.choice([ch for ch in chains if ch]).pop(0))
    queries = list(answers)
    rng.shuffle(queries)
    text = ";".join(statements) + ";" + "".join(f"{q}?" for q in queries)
    start = len(text) - 2 * len(queries)
    return text, [(start + 2 * i + 1, *answers[q]) for i, q in enumerate(queries)]


def chain_batch(rng, c, n=None):
    """x: puzzles left-padded with '.'. y: the answer digit at every '?', -1 elsewhere
    (-1 = "don't score this position"). hops: lookups needed at each '?', 0 elsewhere."""
    n = n or c.batch_size
    stoi = {ch: i for i, ch in enumerate(c.vocab)}
    x = torch.zeros(n, c.block_size, dtype=torch.long)
    y = torch.full((n, c.block_size), -1, dtype=torch.long)
    hops = torch.zeros(n, c.block_size, dtype=torch.long)
    for b in range(n):
        text, qs = chain_example(rng, c.hops)
        off = c.block_size - len(text)
        x[b, off:] = torch.tensor([stoi[ch] for ch in text])
        for i, digit, h in qs:
            y[b, off + i], hops[b, off + i] = stoi[digit], h
    return x, y, hops


def shakespeare_text():
    path = "data/tinyshakespeare.txt"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    return open(path).read()


def text_batch(data, rng, c):
    """Random windows of the text; target = the same window shifted one char left."""
    ix = [rng.randrange(len(data) - c.block_size - 1) for _ in range(c.batch_size)]
    x = torch.stack([data[i:i + c.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + c.block_size + 1] for i in ix])
    return x, y


# ----------------------------------------------------------------------------- train / eval

def pick_device(name):
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def chain_accuracy(model, c, loops, n=1000, seed=0):
    """Accuracy per hop count on a fixed held-out set (same seed -> same puzzles)."""
    model.eval()
    dev = next(model.parameters()).device
    x, y, hops = chain_batch(random.Random(seed), c, n=n)
    correct = model(x.to(dev), loops)[0].argmax(-1).cpu() == y
    model.train()
    return {h: correct[hops == h].float().mean().item() for h in range(1, c.hops + 1)}


def train(c):
    torch.manual_seed(c.seed)
    rng = random.Random(c.seed)
    dev = pick_device(c.device)
    if c.task == "chain":
        c.vocab = CHAIN_VOCAB
        get_batch = lambda: chain_batch(rng, replace(c, hops=level))[:2]  # noqa: E731
    else:
        text = shakespeare_text()
        c.vocab = "".join(sorted(set(text)))
        stoi = {ch: i for i, ch in enumerate(c.vocab)}
        data = torch.tensor([stoi[ch] for ch in text])
        split = int(0.9 * len(data))  # last 10% is held out for validation
        train_data, val_data = data[:split], data[split:]
        get_batch = lambda: text_batch(train_data, rng, c)  # noqa: E731
    model = LoopedGPT(c).to(dev)
    print(f"{sum(p.numel() for p in model.parameters()):,} params | K={c.loops} "
          f"inject={c.inject} random_loops={c.random_loops} | {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    # Curriculum for the chain task: start with 1-hop puzzles and unlock one more hop
    # each time training accuracy passes 85%. Without it, deep puzzles are pure noise
    # until the basic lookup exists, and the model stalls at "guess one of the digits".
    level, acc = 1, 0.0
    t0 = time.time()
    for step in range(1, c.steps + 1):
        x, y = get_batch()
        K = rng.randint(1, c.loops) if c.random_loops else c.loops
        x, y = x.to(dev), y.to(dev)
        logits, loss = model(x, K, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # loops can blow up gradients
        opt.step()
        if c.task == "chain":
            scored = y >= 0
            acc = 0.95 * acc + 0.05 * (logits.argmax(-1)[scored] == y[scored]).float().mean().item()
            if acc > 0.85 and level < c.hops:
                level, acc = level + 1, 0.0
                print(f"step {step:5d} | curriculum: now up to {level} hops")
        if step % max(1, c.steps // 10) == 0 or step == 1:
            msg = f"step {step:5d} | loss {loss.item():.3f}"
            if c.task == "chain":
                msg += f" | train acc {acc:.2f} at <={level} hops"
            if c.task == "shakespeare":
                vrng = random.Random(0)
                vx, vy = zip(*(text_batch(val_data, vrng, c) for _ in range(10)))
                model.eval()
                with torch.no_grad():
                    vl = model(torch.cat(vx).to(dev), c.loops, torch.cat(vy).to(dev))[1]
                model.train()
                msg += f" | val {vl.item():.3f}"
            print(msg + f" | {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(c.out) or ".", exist_ok=True)
    torch.save({"config": asdict(c), "model": model.state_dict()}, c.out)
    return model


def load(path, device="auto"):
    ckpt = torch.load(path, map_location="cpu")
    c = Config(**ckpt["config"])
    model = LoopedGPT(c)
    model.load_state_dict(ckpt["model"])
    return model.to(pick_device(device)).eval(), c


def sweep(models, max_loops=8):
    """Print chain accuracy (mean over hop counts, and per hop) for K = 1..max_loops."""
    for name, (model, c) in models.items():
        print(f"\n{name} (trained at K={c.loops}{', random K' if c.random_loops else ''})")
        print("  K | mean  | " + " | ".join(f"{h}-hop" for h in range(1, c.hops + 1)))
        for K in range(1, max_loops + 1):
            a = chain_accuracy(model, c, K)
            print(f"  {K} | {sum(a.values()) / len(a):.3f} | "
                  + " | ".join(f"{v:.3f}" for v in a.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="train", choices=["train", "sample", "sweep", "compare"])
    for f in fields(Config):  # every Config field becomes a flag, e.g. --loops 8 --inject 0
        cast = (lambda s: s.lower() in ("1", "true", "yes")) if f.type is bool else f.type
        p.add_argument(f"--{f.name}", type=cast, default=None)
    args = vars(p.parse_args())
    mode = args.pop("mode")
    overrides = {k: v for k, v in args.items() if v is not None}
    c = Config(**overrides)

    if mode == "train":
        model = train(c)
        if c.task == "chain":
            sweep({"model": (model, c)})
    elif mode == "compare":
        # Same weights (prelude + core + coda); only the number of core passes differs.
        sc = Config(**{**overrides, "loops": 1, "random_loops": False, "out": "out/standard.pt"})
        lc = Config(**{**overrides, "out": "out/looped.pt"})
        std, looped = train(sc), train(lc)
        sweep({"standard": (std, sc), "looped": (looped, lc)})
    elif mode == "sweep":
        sweep({c.out: load(c.out, c.device)})
    else:  # sample
        model, mc = load(c.out, c.device)
        dev = next(model.parameters()).device
        K = overrides.get("loops", mc.loops)
        if mc.task == "chain":
            rng = random.Random(c.seed)
            for _ in range(3):
                text, qs = chain_example(rng, mc.hops)
                padded = text.rjust(mc.block_size, ".")  # pad exactly like training did
                off = mc.block_size - len(text)
                x = torch.tensor([[mc.vocab.index(ch) for ch in padded]], device=dev)
                guess = model(x, K)[0][0].argmax(-1).tolist()
                print(f"{text}   (K={K})")
                for i, digit, h in qs:
                    g = mc.vocab[guess[off + i]]
                    print(f"  {text[i - 1]}? -> {g}  truth {digit}  ({h}-hop{'' if g == digit else '  WRONG'})")
        else:
            torch.manual_seed(c.seed)
            x = torch.tensor([[mc.vocab.index("\n")]], device=dev)
            print("".join(mc.vocab[i] for i in model.generate(x, 500, K, temperature=0.8)[0].tolist()))


if __name__ == "__main__":
    main()

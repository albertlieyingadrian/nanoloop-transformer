"""Tiny checks. Run with `python test_nanoloop.py` or `pytest`."""
import random

import torch

import nanoloop as nl


def tiny_config(**kw):
    return nl.Config(vocab=nl.CHAIN_VOCAB, n_embd=32, n_head=2, block_size=48, batch_size=8, **kw)


def test_shapes():
    c = tiny_config()
    x, y, hops = nl.chain_batch(random.Random(0), c)
    logits, loss = nl.LoopedGPT(c)(x, targets=y)
    assert logits.shape == (8, 48, len(nl.CHAIN_VOCAB))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert ((y >= 0) == (hops > 0)).all() and (y[:, -1] >= 0).all()  # only answers are scored


def test_loops_reuse_weights():
    # The parameter count does not depend on K: looping adds compute, not weights.
    n = lambda c: sum(p.numel() for p in nl.LoopedGPT(c).parameters())  # noqa: E731
    assert n(tiny_config(loops=1)) == n(tiny_config(loops=8))


def test_k_sweep_smoke():
    c = tiny_config(hops=2)
    model = nl.LoopedGPT(c)
    for K in range(1, 9):
        accs = nl.chain_accuracy(model, c, K, n=16)
        assert set(accs) == {1, 2} and all(0.0 <= a <= 1.0 for a in accs.values())


def test_chain_answer_is_reachable():
    # Following the chain by hand from the query must land on the answer digit.
    text, qs = nl.chain_example(random.Random(3), max_hops=3)
    table = dict(s.split("=") for s in text.split(";")[:-1])
    for i, digit, hops in qs:
        assert text[i] == "?"
        v = text[i - 1]
        for _ in range(hops):
            v = table[v]
        assert v == digit


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

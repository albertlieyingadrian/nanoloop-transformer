# nanoloop-transformer

The smallest readable looped transformer. One file of PyTorch (`nanoloop.py`):
a char-level GPT whose middle block runs K times with the same weights. You
choose K at inference.

```
standard GPT (3 layers, 3 sets of weights)     looped GPT (same 3 sets of weights)

  chars                                          chars
    |                                              |
 [ layer 1 ]                                   [ prelude ]  runs once: read
    |                                              |------------+ e
 [ layer 2 ]                                   [  core   ] <--+ |  runs K times
    |                                              |  ^  h+e  | |  (input injection:
 [ layer 3 ]                                       |  +-------+-+   re-add e each loop)
    |                                              |
  next char                                    [  coda   ]  runs once: write
                                                   |
                                                 next char
```

A standard model with 3 layers can do about 3 steps of "thinking" per token.
The looped model has the same parameter count but can do 1 + K + 1 steps,
because the core block is reused. Think of a for-loop around a function instead
of 3 copies of slightly different functions.

## Quickstart

```bash
git clone https://github.com/albertlieyingadrian/nanoloop-transformer && cd nanoloop-transformer
pip install -r requirements.txt   # just torch
python nanoloop.py --mode compare            # chain task: standard vs looped, ~10 min on a Mac (MPS)
python nanoloop.py --mode sample --out out/looped.pt --loops 1   # watch it fail with 1 loop
python nanoloop.py --mode sample --out out/looped.pt --loops 4   # and succeed with 4
python nanoloop.py --task shakespeare --steps 3000                # char-level Shakespeare, ~2 min
python nanoloop.py --mode sample --out out/model.pt --loops 4     # sample text
python test_nanoloop.py                       # 4 tiny checks
```

Every field of `Config` is a flag: `--loops 8 --inject 0 --random_loops 1 --n_core 2 --hops 6 ...`

## The depth-hungry task

Two chains of variables, written like code (defined before use) and interleaved,
then one question per variable:

```
z=4;m=5;u=z;k=m;y=k;z?k?m?y?u?
```

`y?` needs 3 lookups in a row (y -> k -> m -> 5). Answers never appear in the
input, so each `?` has to be worked out. Each layer (or each loop) can resolve
one more link of every chain, so k-hop questions need about k passes.
Training uses a curriculum: 1-hop puzzles first, and one more hop is unlocked
each time training accuracy passes 85%.

## Results (real runs, Apple M-series MPS, seed 1337, 6000 steps, batch 128)

Chain task, hops 1 to 4, held-out set of 1000 puzzles. Accuracy at each `?`:

| model | params | trained K | eval K | mean | 1-hop | 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|---|---|---|---|
| standard (prelude + core + coda) | 605k | 1 | 1 | 0.606 | 0.766 | 0.565 | 0.535 | 0.556 |
| **looped (same weights)** | 605k | 4 | 4 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| looped, no input injection | 605k | 4 | 4 | 0.606 | 0.806 | 0.582 | 0.526 | 0.511 |
| looped, random K in 1..4 | 605k | 1-4 | 4 | 0.689 | 0.947 | 0.659 | 0.602 | 0.548 |
| unshared, 4 distinct core layers | 1.2M | 1 | 1 | 0.937 | 0.980 | 0.967 | 0.920 | 0.882 |

Sweeping K at inference on the looped model (trained at K=4):

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| looped mean | 0.513 | 0.833 | 0.979 | 1.000 | 1.000 | 0.999 | 0.998 | 0.993 |
| 2-hop | 0.387 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.996 | 0.988 |
| 4-hop | 0.339 | 0.535 | 0.915 | 1.000 | 1.000 | 0.998 | 0.996 | 0.996 |
| standard mean (forced to loop) | 0.606 | 0.594 | 0.575 | 0.561 | 0.561 | 0.558 | 0.557 | 0.557 |

What this shows:

- **Loops buy depth without parameters.** Same 605k weights: 0.606 when the core
  runs once, 1.000 when it runs 4 times. It even beats a model with twice the
  weights and the same compute (0.937).
- **You can watch the hops resolve.** With K=1 only 1-hop questions are right;
  2-hop questions become right at K=2; 4-hop questions need K=3 to 4.
- **Extra loops past training are safe here.** K=5 to 8 stays at 0.99 or above:
  the core has learned a step that is stable when repeated.
- **Input injection matters.** Without re-adding `e` every loop the model never
  gets past 2-hop puzzles in 6000 steps (0.606).
- **Random K is slower to train.** At 6000 steps it only reached the 3-hop
  stage. In a 12000-step run it reached 0.93 and stayed flat from K=3 to K=8.
  Training at many depths makes it robust to K, but each depth gets fewer updates.
- **A standard model gets nothing from extra passes.** Forcing its core to run
  more than once makes it slightly worse.

Shakespeare (char-level, 3000 steps, batch 128, context 64): validation loss
**1.601** at K=1 vs **1.584** at K=4. Text isn't depth-hungry at this scale, so
looping helps only a little there.

```
FRIAR LAURENCE:
To yield is must not be request him.

ROMEO:
Then at my lordship, if we that comes offend'st be embrace.
```

Getting the chain task to train at all took four fixes, all now in the code:
dense supervision (one question per variable, not one per puzzle), the
curriculum, rotary position embeddings (so "look 2 characters back" is one
skill instead of one per position), and the defined-before-use order. Before
those fixes, every model sat at 0.55 accuracy: it learned to guess one of the
two digits in the prompt. `program.md` has the log.

## Files

- `nanoloop.py`: model, data, training, K sweep, sampling.
- `test_nanoloop.py`: shapes, K doesn't change the parameter count, K-sweep smoke test, puzzle answers are reachable.
- `program.md`: a brief for running small experiments and logging them.
- `results.tsv`: the runs behind the tables above.

## Limitations

- Toy scale: 605k parameters, one synthetic task and a small Shakespeare run, one seed per row of the table (seed
  1337). Treat gaps under about 5 points as noise until you rerun with another seed.
- Times are for an Apple M-series laptop (MPS). On a plain CPU expect the 6000-step chain runs to take several times
  longer; `--steps`, `--batch_size` and `--n_embd` shrink them.
- No KV cache, no mixed precision, no early exit: the code is written to be read, not to be fast.

## Where to go next

- The full course this file grew out of, with tests per stage, baselines and six ways to read the loops:
  https://github.com/albertlieyingadrian/build-your-own-looped-transformer

## Credits and further reading

- Universal Transformers (2018), https://arxiv.org/abs/1807.03819: the original shared-weights, looped transformer.
- Looped Transformers are Better at Learning Learning Algorithms (ICLR 2024), https://arxiv.org/abs/2311.12424, code
  https://github.com/Leiay/looped_transformer: input injection and the train-K / test-K story.
- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn, 2025),
  https://arxiv.org/abs/2502.05171: prelude / recurrent core / coda at 3.5B parameters with random K.
- Scaling Latent Reasoning via Looped Language Models (Ouro, 2025), https://arxiv.org/abs/2510.25741: looped LMs
  pretrained at scale with learned early exit.
- Inkling 2 Mini, https://huggingface.co/blog/inkling2mini: a released model that runs its middle layers three times.
- The shape of this repo: nanoGPT (https://github.com/karpathy/nanoGPT), micrograd
  (https://github.com/karpathy/micrograd) and autoresearch (https://github.com/karpathy/autoresearch). No code is
  copied from them.
- Tiny Shakespeare comes from the char-rnn repo, https://github.com/karpathy/char-rnn, downloaded on first use.

## License

MIT

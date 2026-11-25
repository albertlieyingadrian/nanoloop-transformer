# program.md

A short brief for running small experiments with `nanoloop.py`. Hand it to
yourself (or to a research assistant) before a session. One idea per run, one
line per result.

## The question

Does running the same block more times buy depth you didn't pay for in
parameters? The chain task answers this cleanly: a k-hop puzzle needs about k
lookups in a row, and each lookup needs about one attention layer.

## Ground rules

- Change one thing per run. Keep `--seed` fixed unless you're measuring noise.
- Budget: a run should finish in about 10 minutes on a laptop (the default
  6000-step chain run takes ~5 min on Apple MPS). If it doesn't,
  shrink `--steps`, `--batch_size` or `--n_embd` before adding anything.
- The metric is chain accuracy from the K sweep (mean and per hop). For
  Shakespeare it is the `val` loss on the last log line.
- Don't edit `nanoloop.py` to make a number go up unless the edit is the idea
  being tested. If it is, keep the diff small and note it in the log.
- Run the same config twice (seeds 1337 and 1) before you believe a gap
  smaller than about 5 points.

## What to try

| knob | flag | what you learn |
|---|---|---|
| loops | `--loops 1/2/4/8` | how accuracy on k-hop puzzles tracks K |
| input injection | `--inject 0` | whether the model forgets the question as it loops |
| random K | `--random_loops 1` | whether training at many K makes the sweep smooth and lets K go past training |
| sharing schedule | `--n_core 2 --loops 2` vs `--n_core 1 --loops 4` | same depth, twice the core weights: is more sharing worse? |
| unshared baseline | `--n_core 4 --loops 1` | the same compute as K=4 with 4x the core weights: the ceiling looping is chasing |
| harder puzzles | `--hops 6 --block_size 96` | where a fixed K runs out |
| curriculum | edit the 0.85 threshold in `train` | how much the curriculum matters |
| text | `--task shakespeare --loops 1/4` | does looping help a task that isn't depth-hungry? |

Ideas past the flags (each is a small edit):

- Loop-number embedding: add a learned vector per iteration so the core knows
  which pass it is on.
- Early exit: stop looping when `h` stops changing (norm of the update below a
  threshold) and log the average K used per hop count.
- Truncated backprop: only backprop through the last few loops, as in Huginn.

## Logging

Append one line per run to `results.tsv`:

```
date	flags	steps	params	mean@trainK	1hop	2hop	3hop	4hop	notes
```

Keep failed runs in the log too; "didn't learn, stuck at 0.55" is a result.
After a batch of runs, write two or three sentences at the bottom of this file
under "Findings": what moved the number, what didn't, what to try next.

## Findings

- Before anything learned, every model sat at ~0.55 accuracy: it guessed one of
  the two digits in the prompt. Four changes fixed that: one question per
  variable (dense signal), a 1-hop-first curriculum, rotary positions, and
  writing chains in defined-before-use order. lr 5e-4 with batch 128 was more
  reliable than lr 1e-3 with batch 64.
- Same 605k weights: standard (K=1) 0.606, looped (K=4) 1.000. The unshared
  4-layer core (1.2M params, same compute) got 0.937.
- Without input injection the looped model stalls at 2 hops (0.606).
- Random K needs about twice the steps (0.927 at 12k steps) but is flat in K.
- Next: early exit by convergence of `h`; `--hops 6 --block_size 96` to see
  where K=4 runs out and whether K=6 at inference rescues it.

# Speculative Decoding Lab

An offline experiment showing why speculative sampling can preserve a target token distribution even when a cheaper draft distribution is a poor match. It uses a finite synthetic vocabulary and the Python standard library—no checkpoints, network calls, or GPU.

```sh
python demo.py
python -m pip install pytest ruff
python -m pytest -q
ruff check .
```

For draft probabilities `q` and target probabilities `p`, propose `x` from `q`, then accept it with probability `min(1, p[x]/q[x])`. Accepted mass at `x` is `min(p[x], q[x])`. On rejection, resample from the normalized positive residual `max(0, p-q)`. Rejection supplies exactly the missing mass, so the output distribution remains `p` in exact arithmetic. The expected acceptance rate is `sum(min(p, q))`.

The demo compares two proposals against one target. Their expected acceptance is 90% and 40% respectively; observed token fractions in both runs should be close to `[0.6, 0.3, 0.1]`. Tests cover partial and disjoint support, equal distributions, zero-weight categories and malformed inputs.

This is the one-token building block, not a multi-token verifier or a benchmark for transformer throughput. It does not simulate KV caches, batched target inference, GPU overhead or speedups.

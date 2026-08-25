# Serving Qwen3.8-27B for Crux

Two engines behind one router on four H20 cards.

```
card 0,1 ──> engine A :30000 ─┐
                              ├── router :30080 ── clients
card 2,3 ──> engine B :30001 ─┘
```

```bash
./engine.sh 0,1 30000
./engine.sh 2,3 30001
./router.sh
```

## Why two TP=2 engines and not four TP=1 replicas

Total KV capacity is roughly the same either way. What differs is how long a
*single* request is allowed to get, and that is the binding constraint here.

Measured across 393 real Crux trajectories:

| percentile | context |
|---|---|
| median | 16K |
| P90 | 47K |
| P99 | 113K |
| max | 155K |

Images cost another 1-2K each on top. A TP=1 replica holds ~811K tokens, so
three full-length sessions fill it and the fourth queues behind them. Pooled
across two cards the engine holds 1.85M tokens -- more than double, because
pooling also cuts fragmentation -- and it takes six concurrent long sessions to
wedge it. Measured throughput agrees:

| concurrency | TP=1 | TP=2 |
|---|---|---|
| 1 | 55.4 tok/s | 67.8 tok/s |
| 16 | 670.5 tok/s | 1099.2 tok/s |
| 48 | 1446.7 tok/s | 2469.2 tok/s |

Two of these together sustained 96 concurrent requests at 5086 tok/s with
2.12s latency, all 96 succeeding.

## Why a router and not two ports

Handing people two ports means they pick inconsistently, and a client that
reconnects picks again. The second turn of a conversation then lands on the
engine that never saw the first turn, re-prefills the entire history, and pays
full price for a prefix that is already computed next door.

For agent traffic that prefix is nearly the whole request -- the system prompt
and the transcript so far are identical turn over turn, and only the last few
hundred tokens are new. So `--policy cache_aware`: the router keeps a prefix
tree per worker and sends each request to whichever engine holds the longest
match.

Affinity is a default, not a lock. When one engine falls 64 requests *and* 1.5x
behind the other, the balance thresholds override cache affinity, so a single
heavy user cannot pin everyone else behind their own cache.

## Why MTP

Speculative decoding does not make the GPU faster. It makes one forward pass
emit more than one token when the draft head guesses right, which converts
memory-bandwidth stalls into tokens.

This checkpoint ships `mtp.safetensors` and declares `mtp_num_hidden_layers=1`,
so the draft head was trained against these exact weights -- `--speculative-
algorithm NEXTN` reuses it rather than loading a separate draft model.
`--speculative-eagle-topk 1` keeps the draft a linear chain, which is all one
MTP layer supports; tree verification needs more layers.

`--speculative-adaptive` matters more than the step count. Speculation is a bet
on idle compute: it wins at batch size 1 and loses once the batch already
saturates the GPU, because every rejected draft token is compute spent for
nothing. Benchmark runs swing between both extremes within a single job, so
`num_steps` has to track the live acceptance rate instead of being pinned to
whatever looked good on an idle box.

## Things that broke on the way here

- **`ninja` not on PATH.** SGLang shells out to it when compiling CUDA graphs;
  having it installed in the venv is not enough.
- **Compile caches on the shared mount.** They are keyed to the driver and GPU,
  so they belong on node-local `/scratch`, not cpfs.
- **Thinking chain leaking into `content`.** Needs `--reasoning-parser qwen3`.
- **Wrong tool parser name.** vLLM calls it `qwen3_xml`, SGLang `qwen3_coder`.

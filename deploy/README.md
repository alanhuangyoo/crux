# Serving Qwen3.8-27B for Crux

Two engines behind one router on four H20 cards.

```
card 0,1 ──> engine A :30000 ─┐
                              ├── router :30080 ── clients
card 2,3 ──> engine B :30001 ─┘
```

```bash
systemctl start crux-engine@a crux-engine@b crux-router crux-tunnel
```

Card assignment lives in `engines/<name>.env` (`CARDS=0,1`, `PORT=30000`), not
in the unit name -- systemd escapes commas in instance names and `0,1` is the
natural way to write a card pair.

## Reaching it

| from | url |
|---|---|
| inside the cluster | `http://192.168.21.45:30080/v1` |
| public | `http://8.210.147.108:18077/v1` |

Both need `Authorization: Bearer <key>`. The router rejects anything else --
verified: 401 with no key, 401 with a wrong one, 200 with the right one. That
check matters because the public path terminates at the router, so the key is
the only thing standing in front of the GPUs.

The public path is `:18077 -> socat -> jump 127.0.0.1:18078 -> reverse ssh
tunnel -> :30080`. The socat hop exists because sshd binds `-R` forwards to
loopback unless `GatewayPorts` is on, and turning that on would expose every
future forward on that box rather than just this one.

`ServerAliveInterval` on the tunnel is not optional. Without it the tunnel
stays "up" on the cluster side long after the jump box has dropped it from its
NAT table, and the public endpoint blackholes instead of reconnecting.

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

## Why MTP is off

The checkpoint ships `mtp.safetensors` and declares `mtp_num_hidden_layers=1`,
so a trained draft head is right there and `--speculative-algorithm NEXTN`
reuses it without loading a second model. It was worth trying. It loses.

Same hardware, same weights, 256 output tokens per request, prompts sharing a
long prefix the way agent traffic does. Decode is per stream; throughput is
aggregate:

| concurrency | no MTP | MTP | MTP + ReplaySSM |
|---|---|---|---|
| 1 | **96.7 tok/s** | 77.5 | 76.3 |
| 8 | **88.2 tok/s** | 59.4 | 65.9 |
| 32 | **68.7 tok/s** | 33.3 | 35.3 |
| 64 | **70.2 tok/s** | 28.7 | 30.6 |

The draft head is good. Accept length measured 2.9 of 4 draft tokens at an
accept rate of 0.65 -- speculation is guessing right two thirds of the time.
It still runs at half speed, and the arithmetic says why:

| | per cycle | tokens | rate |
|---|---|---|---|
| plain decode | 10.3 ms | 1 | 96.7/s |
| speculative | 38.5 ms | 2.94 | 76.3/s |

A verify cycle is one target forward plus three single-layer draft passes, so
it should cost about what a plain forward costs -- call it 11 ms for 2.94
tokens, roughly 267 tok/s. It costs 38.5 ms. **Around 28 ms per cycle is not
model forward time at all.**

That gap is the architecture. 48 of the 64 layers are gated-delta-net linear
attention, not softmax attention. Rejecting a draft token in a KV model means
dropping cache entries; in a recurrent model it means restoring SSM state, so
the verify path snapshots full state per draft step -- 116 live mamba states
for 32 requests, across 48 layers, every step.
`--enable-linear-replayssm-spec` exists precisely to replace those snapshots
with a raw-input window, and it recovered only 6-11%.

Lowering `num_steps` does not rescue it either: the 28 ms is largely fixed per
cycle, so cutting the draft chain to 1 keeps most of the overhead while
dropping the yield from 2.94 tokens to about 1.6.

`engine.sh` carries the exact flags to re-test this after an sglang upgrade.
Two things to know before doing so: MTP captures a second set of CUDA graphs
and OOMs at `mem-fraction-static 0.88`, and `--speculative-adaptive` asserts
outright on 0.5.18 (`shared logits buffer holds 192 rows but caller needs
384`).

## Why systemd and not `nohup`

A backgrounded ssh child dies with its parent shell -- silently, and at the
worst possible moment. Worse, when the scheduler hits an OOM it SIGQUITs
itself, so the failure mode is not a crash loop anyone would notice but simply
no engine, with the port closed and clients timing out. `Restart=on-failure`
turns both into a blip. For something other people are meant to depend on, that
is the difference between a service and a demo.

## Things that broke on the way here

- **`ninja` not on PATH.** SGLang shells out to it when compiling CUDA graphs;
  having it installed in the venv is not enough.
- **Compile caches on the shared mount.** They are keyed to the driver and GPU,
  so they belong on node-local `/scratch`, not cpfs.
- **Thinking chain leaking into `content`.** Needs `--reasoning-parser qwen3`.
- **Wrong tool parser name.** vLLM calls it `qwen3_xml`, SGLang `qwen3_coder`.
- **OOM during CUDA graph capture, at a `mem-fraction-static` the plain engine
  was happy with.** MTP captures a second set of graphs -- draft and verify on
  top of target -- and the scheduler had settled on `max_running_requests=48`
  while still capturing decode graphs up to batch 256, i.e. memory spent on
  shapes that never run. Capping decode graphs at 64 and dropping the static
  fraction to 0.78 fixed it; the KV pool gives up ~300K tokens out of 1.85M,
  against a P99 context of 113K.

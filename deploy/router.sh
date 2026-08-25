#!/usr/bin/env bash
#
# One endpoint in front of both engines, routed by prefix cache.
#
# Without this, sharing the deployment means handing people two ports and
# hoping they pick consistently. They won't, and neither will a client that
# reconnects -- so a conversation's second turn lands on the engine that has
# never seen its first turn, re-prefills the whole history, and pays full
# price for a prefix it already computed next door. With agent traffic that
# prefix is most of the request: the system prompt and the transcript so far
# are identical turn to turn, and only the last few hundred tokens are new.
#
# cache_aware keeps a per-worker prefix tree and sends a request to whichever
# engine already holds the longest match, falling back to load balancing when
# the match is weak (cache-threshold) or when one engine has fallen far enough
# behind the other (balance-*-threshold).
#
# The balance thresholds are far below their defaults (64 / 1.5) because the
# defaults do not fire in practice. Measured at 96 concurrent requests over a
# shared prefix: 203 requests to one engine, 2 to the other, while the idle
# card sat there and aggregate throughput matched a single engine exactly.
# Both conditions have to hold to trigger balancing, so a 64-request absolute
# gap is a very high bar when each engine only runs 48 at a time -- the queue
# has to be deeper than the engine before affinity ever yields.
#
# 16 and 1.25 keep affinity for real prefix hits while making it structurally
# impossible for one engine to idle through a backlog on the other.
set -euo pipefail
B=/mnt/cpfs/users/xiaohuang
export PATH="$B/envs/sglang/bin:$PATH"

exec $B/envs/sglang/bin/python -m sglang_router.launch_router \
  --host 0.0.0.0 --port 30080 \
  --worker-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --policy cache_aware \
  --cache-threshold 0.3 \
  --balance-abs-threshold 16 \
  --balance-rel-threshold 1.25 \
  --max-payload-size 536870912 \
  --request-timeout-secs 3600 \
  --worker-startup-timeout-secs 1800 \
  --prometheus-port 30090 \
  --log-level info \
  --api-key sk-crux-iM-eVeNJmh1_crsLPfiBInwFaU410pNM

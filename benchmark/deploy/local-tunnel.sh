#!/usr/bin/env bash
#
# Forwards the h20-45 services to localhost, over the jump host:
#   :30080  SGLang router (raw model API)
#   :3000   new-api       (gateway console: users, keys, quota)
#   :8317   CLIProxyAPI   (CLI-subscription upstreams)
#
# The jump box cannot reach the GPU subnet itself, so this hops through h20-43,
# which can -- ~/.ssh/config already routes h20-43 via ProxyJump jump, so the
# two hops are implicit here.
#
# ServerAlive* is what makes this survive a laptop sleep or a NAT timeout:
# without it the forward stays "up" locally long after the far side has dropped
# it, and every request hangs instead of reconnecting.
#
# ExitOnForwardFailure matters too -- otherwise a stale ssh already holding
# :30080 lets this one start "successfully" while forwarding nothing.
set -euo pipefail
exec /usr/bin/ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=20 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -L 30080:192.168.21.45:30080 \
  -L 3000:192.168.21.45:3000 \
  -L 8317:192.168.21.45:8317 \
  h20-43

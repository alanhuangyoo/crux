#!/usr/bin/env bash
#
# Same engine as sgl2.sh plus MTP speculative decoding.
#
# TP=2 rather than two TP=1 replicas: total KV capacity is the same either way,
# but TP pools it, and what binds here is how long a *single* request can get.
# Measured over 393 real trajectories: median 16K tokens, P90 47K, P99 113K,
# max 155K -- and that is before images, which cost 1-2K each. A TP=1 replica
# holds ~811K tokens, so three full-length sessions fill it; pooled across two
# cards it takes six, and a few long sessions can no longer wedge an engine.
#
# MTP speculative decoding is deliberately OFF. It is implemented, it works,
# and it is a net loss on this model -- measured, not assumed. See
# deploy/README.md for the table; the short version is that a verify cycle
# costs ~38.5ms and returns 2.94 tokens where a plain forward costs 10.3ms and
# returns 1, so speculation runs at roughly half the decode speed of not
# speculating, at every concurrency from 1 to 64.
#
# The draft head is not the problem: accept length measured 2.9 of 4 tokens at
# an accept rate of 0.65. The problem is that 48 of the 64 layers are
# gated-delta-net linear attention. Rejecting a draft token in a KV model means
# dropping cache entries; in a recurrent one it means restoring SSM state, and
# that machinery costs ~28ms per cycle that is not model forward time at all.
# --enable-linear-replayssm-spec, which exists precisely for this, recovered
# only 6-11% of it.
#
# To re-test after an sglang upgrade, add:
#   --speculative-algorithm NEXTN --speculative-num-steps 3 \
#   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
#   --enable-linear-replayssm-spec
# and drop --mem-fraction-static to 0.78; MTP captures a second set of CUDA
# graphs and OOMs at 0.88. Note --speculative-adaptive still asserts on 0.5.18
# ("shared logits buffer holds 192 rows but caller needs 384").

# --enable-cache-report is the only way to know whether the prefix cache is
# actually being reused. Without it usage.prompt_tokens_details comes back
# empty and a cache hit is indistinguishable from a quiet engine -- measured
# over 24 sessions, turn 1 took 5.81s and turns 2+ took 3.08s, which is
# obviously the cache working, but "obviously" is not a number.
# The chat template is a patched copy of the checkpoint's own. Qwen3.5 accepts
# reasoning_effort in low/medium/xhigh and raises on anything else; every
# OpenAI-shaped client sends minimal/low/medium/high. Codex's default of "high"
# therefore 400s before the model runs. The copy maps high->xhigh and
# minimal/none->low ahead of the original validation, which is left intact so a
# genuine typo still fails loudly.
set -euo pipefail
CARDS="${1:?usage: sgl3.sh <card,card> <port>}"
PORT="${2:?usage: sgl3.sh <card,card> <port>}"
B=/mnt/cpfs/users/xiaohuang

export CUDA_VISIBLE_DEVICES="$CARDS"
export HF_HOME=$B/cache/hf
# SGLang shells out to `ninja` when compiling CUDA graphs, so the venv bin has
# to be on PATH -- importing the module is not enough.
export PATH="$B/envs/sglang/bin:$PATH"
# Compilation caches are keyed to the driver and GPU, so they stay on this
# machine's local disk rather than the shared mount.
SAFE=${CARDS//,/_}
export TORCHINDUCTOR_CACHE_DIR=/scratch/crux-inductor-$SAFE
export TRITON_CACHE_DIR=/scratch/crux-triton-$SAFE
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# The SSM state dtype is a runtime activation, not the weights -- those stay
# FP8 either way. It defaults to float32 from the model config, and on SM90
# nothing forces that (the bf16 requirement is an SM100-only guard), while H20
# runs bf16 at roughly 2.5x its fp32 rate. Since decode here is compute-bound
# rather than bandwidth-bound -- measured at 100% SM against 32% memory -- the
# state math is worth trying in bf16. Left unset by default until accuracy is
# checked; long sequences are where a narrower state would drift.
SSM_ARGS=()
if [ -n "${SSM_DTYPE:-}" ]; then
  SSM_ARGS=(--mamba-ssm-dtype "$SSM_DTYPE")
fi

exec $B/envs/sglang/bin/python -m sglang.launch_server \
  --model-path $B/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT" \
  --tp 2 \
  --context-length 262144 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.88 \
  --cuda-graph-max-bs-decode 64 \
  --enable-cache-report \
  --chat-template $B/models/qwen3.8-27b-openai-effort.jinja \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  "${SSM_ARGS[@]}" \
  --api-key sk-crux-iM-eVeNJmh1_crsLPfiBInwFaU410pNM

#!/usr/bin/env bash
#
# Same engine as sgl2.sh plus MTP speculative decoding.
#
# One TP=4 engine rather than two TP=2 ones. The earlier split was chosen to
# pool KV cache for long contexts, on the assumption that decode here is
# memory-bandwidth bound. It is not: under load this deployment sits at 100% SM
# with 32% memory-bandwidth utilisation, which is compute-bound -- H20 pairs
# H100-class bandwidth with roughly 15% of its compute, and 48 of this model's
# 64 layers are gated-delta-net linear attention, which spends arithmetic to
# save memory. That trade gives back the resource we have and consumes the one
# we lack.
#
# Raising TP is the only lever that adds compute per forward pass, and it
# roughly doubles decode:
#
#   decode tok/s   TP=2    TP=4
#   concurrency 1  65.1    144.2
#   concurrency 8  63.2    131.3
#   KV pool        1.85M   3.94M tokens
#
# NVLink is fully connected here (NV18), so the wider all-reduce costs little.
# What this buys is task completions rather than throughput: the median task was
# spending about 750s of its 900s budget generating reasoning, and halving that
# is what lets a task finish instead of being cut off.
#
# TP is derived from the card list, so the env file is the only thing to edit.

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
TP=$(awk -F, "{print NF}" <<< "$CARDS")

SSM_ARGS=()
if [ -n "${SSM_DTYPE:-}" ]; then
  SSM_ARGS=(--mamba-ssm-dtype "$SSM_DTYPE")
fi

exec $B/envs/sglang/bin/python -m sglang.launch_server \
  --model-path $B/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT" \
  --tp "$TP" \
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

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
# The checkpoint ships mtp.safetensors and declares mtp_num_hidden_layers=1, so
# the draft head is already trained against these exact weights -- NEXTN reuses
# it instead of loading a separate draft model. topk=1 keeps the draft a linear
# chain, which is all a single MTP layer can support; a tree needs more layers.
#
# --speculative-adaptive matters more here than the step count. Speculation is
# a bet on idle compute: it wins when the GPU is waiting on memory bandwidth
# (one user, batch of 1) and loses when the batch is already saturating it,
# because every rejected draft token is compute spent for nothing. Benchmark
# runs swing between both extremes, so num_steps has to follow the accept rate
# rather than being pinned to whatever looked good on an idle box.
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

exec $B/envs/sglang/bin/python -m sglang.launch_server \
  --model-path $B/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT" \
  --tp 2 \
  --context-length 262144 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-adaptive \
  --api-key sk-crux-iM-eVeNJmh1_crsLPfiBInwFaU410pNM

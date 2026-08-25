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
# --enable-linear-replayssm-spec is what makes any of this pay off, and the
# reason is the architecture: 48 of the 64 layers are gated-delta-net linear
# attention, not softmax attention. Rejecting a draft token in a KV model means
# dropping cache entries; in a recurrent one it means restoring SSM state, so
# the default verify path snapshots the full state per draft step. Measured at
# batch 32 that was 116 live mamba states for 32 requests -- ~3.6 copies each,
# across 48 layers, every decode step.
#
# The cost is not draft quality. Accept length measured 2.6-3.1 of 4 at an
# accept rate of 0.53-0.67, which is a good head; the engine was still 2.5x
# slower than no speculation at all, because state management ate the gain and
# then some. ReplaySSM replaces those snapshots with a per-slot raw-input
# window and recomputes instead of copying. It requires the linear draft chain
# above, and is mutually exclusive with --enable-linear-replayssm (same ring
# storage, incompatible cursor protocol).
#
# Two memory notes, both learned by watching this OOM at 0.85. MTP captures a
# second set of CUDA graphs -- draft and verify on top of target -- so it needs
# noticeably more room outside the KV pool than the plain engine does. And the
# scheduler settles on max_running_requests=48, which makes every decode graph
# captured above that batch size memory spent on a shape that never runs. The
# lower static fraction costs ~300K tokens of KV, out of 1.85M against a P99
# context of 113K; the batch-size cap costs nothing at all.
#
# --speculative-adaptive is deliberately NOT set, and that is a compromise.
#
# Speculation is a bet on idle compute: it wins when the GPU is stalled on
# memory bandwidth (one user, batch of 1) and loses once the batch already
# saturates compute, because every rejected draft token is work spent for
# nothing. Adaptive stepping is the principled answer -- follow the live accept
# rate instead of pinning num_steps to whatever looked good on an idle box.
#
# It does not run on sglang 0.5.18. The adaptive controller builds a second
# target graph runner that asks the shared logits buffer for twice the rows it
# was allocated:
#
#   eagle_worker_v2.build_adaptive_runtime_state -> decode_cuda_graph_runner
#   AssertionError: shared logits buffer holds 192 rows but caller needs 384
#
# The ratio is fixed, so shrinking the graph batch size does not help -- the
# buffer is sized before the controller asks. Until that is fixed upstream,
# num_steps stays pinned and the high-concurrency cost has to be measured
# rather than designed around: see deploy/abtest.py, which reports TTFT and
# per-stream decode separately at concurrency on both sides of saturation.
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
  --mem-fraction-static 0.78 \
  --cuda-graph-max-bs-decode 64 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-linear-replayssm-spec \
  --api-key sk-crux-iM-eVeNJmh1_crsLPfiBInwFaU410pNM

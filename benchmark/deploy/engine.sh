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
# envs/sglang was destroyed by an editable install of the qwen4 branch whose
# source tree was later deleted, so `import sglang` fails there and any engine
# started from it dies immediately. envs/sglang-rel is a clean `sglang[all]`
# that resolved to the same released 0.5.18. Override with VENV= if needed.
VENV="${VENV:-$B/envs/sglang-rel}"

export CUDA_VISIBLE_DEVICES="$CARDS"
export HF_HOME=$B/cache/hf
# Both SGLang's CUDA-graph compilation and flashinfer's sampling module shell
# out to `ninja` by name, so the venv bin has to be on PATH -- invoking
# bin/python directly does not put it there.
#
# The flashinfer one is the dangerous half: it JIT-builds on the first
# *sampled* token, which is after startup, after /health_generate returns 200,
# and after any greedy (temperature=0) probe has answered correctly. An engine
# missing this passes every check and then dies on the benchmark's first real
# request. Validate a new environment with temperature > 0, not with a greedy
# smoke test.
export PATH="$VENV/bin:$PATH"
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

# Speculative decoding with a separately trained drafter, per the published
# SGLang recipe for this model.
#
# An earlier attempt used the checkpoint's own MTP head (NEXTN) and lost: it
# drafts by running a model layer, which costs the one resource this box does
# not have -- measured at 100% SM against 32% memory bandwidth. A trained
# drafter reads the target's hidden states at a few tap layers instead, so it
# buys a much higher accept rate for far less compute. That is why the recipe
# names it and not MTP.
#
# Which drafter depends on the sglang build: 0.5.18 registers DFlashDraftModel
# (DFlash 1) and DSparkDraftModel, but not DFlash2DraftModel -- loading the
# published DFlash2 weights fails with "Cannot find model module".
#
# Left unset by default, because every speculative variant measured worse here,
# and worse in proportion to how much compute the draft costs:
#
#   decode tok/s        conc 1   conc 8   conc 16   accept rate
#   no speculation       140.2    128.3     119.3   --
#   MTP (own head)        98.0     90.8       --    0.65
#   DSpark (recipe)      101.4     73.2      46.5   0.15-0.26
#
# Speculation trades compute for latency, and this box has no spare compute to
# trade -- 100% SM against 32% memory bandwidth. The published recipes that
# recommend it were measured on H200 and RTX PRO 6000, where the balance is the
# other way round. Same flag, opposite conclusion, because the hardware differs.
#
# Note SGLang pins --max-running-requests to 48 whenever speculation is on and
# the flag is unset; set it explicitly if a higher ceiling is needed.
SPEC_ARGS=()
if [ -n "${SPEC_DRAFT_PATH:-}" ]; then
  SPEC_ARGS=(
    --speculative-algorithm "${SPEC_ALGO:-DSPARK}"
    --speculative-draft-model-path "$SPEC_DRAFT_PATH"
    --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS:-8}"
    --speculative-draft-model-quantization unquant
  )
fi

exec $VENV/bin/python -m sglang.launch_server \
  --model-path $B/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT" \
  --tp "$TP" \
  --context-length 262144 \
  --kv-cache-dtype "${KV_DTYPE:-fp8_e4m3}" \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.88 \
  --cuda-graph-max-bs-decode 64 \
  --enable-cache-report \
  --chat-template $B/models/qwen3.8-27b-openai-effort.jinja \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  "${SSM_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  --api-key sk-crux-iM-eVeNJmh1_crsLPfiBInwFaU410pNM

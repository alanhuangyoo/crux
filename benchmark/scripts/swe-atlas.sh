#!/usr/bin/env bash
# SWE-Atlas-QnA: 124 codebase-comprehension questions across 11 repositories in
# Go, Python, C and TypeScript. A different task shape from the other two -- the
# answer is a claim about an existing repository, not an edit -- which is why it
# is worth running: every fix so far was measured on runs whose success is "the
# code now passes tests", and a benchmark where nothing is edited says whether
# any of it was about editing.
#
# Two things this run does differently, both forced by the box:
#
#   .env.atlas   the grader is an LLM judge, defaulted by the task files to
#                anthropic/claude-opus-4-5 -- what the public leaderboard uses.
#                There is no Anthropic credential here, so the judge is pointed
#                at the same local endpoint the agent uses. These scores are
#                internally comparable and are NOT comparable to the published
#                SWE-Atlas numbers.
#
#   verifier x3  the judge's default budget is 900s. With the model name unset
#                to something the endpoint does not serve, the client retried
#                until it hit that, and all 124 trials came back "Verifier
#                execution timed out after 900" with the agent already finished
#                and its answer written.
cd /scratch/crux
export PATH="$HOME/.local/bin:$PATH"
SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="/scratch/crux/src:${SP}"
exec python3 -m crux.cli bench \
  --dataset scale-ai/swe-atlas-qna \
  --model openai/qwen3.8-27b \
  --attempts 1 --concurrent 12 --allow-concurrent-jobs \
  --agent-timeout-multiplier 4 \
  --verifier-timeout-multiplier 3 \
  --jobs-dir /scratch/atlas \
  --env-file /scratch/crux/.env.atlas

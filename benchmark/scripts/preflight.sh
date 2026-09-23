#!/bin/bash
# Check an arm's configuration before it costs hours, not after.
#
# Five arms were thrown away in one session, none of them for a reason the
# launch command could show: tool calls arriving as text because the server had
# no parser, a forgotten `pi_env`, an escalated ceiling that overflowed a 32K
# context, containers removed out from under a running arm, and an attempt count
# that made the run too slow to be worth finishing. On this hardware the GPUs sit
# at 100% and a turn takes eight seconds, so a wasted arm is two to three hours
# that no amount of concurrency buys back.
#
# Usage: preflight.sh   (run it before every launch)
set -uo pipefail
fail=0
say() { printf '  %-44s %s\n' "$1" "$2"; }

# The launcher that is actually about to run: every check below reads this one
# file. Two of them still read `/tmp/confirm.sh` after the output-token check
# had been moved off it, so the model name and the arm settings came from
# whichever round last wrote that file. Pass it explicitly: preflight.sh <launcher>
LAUNCHER="${1:-}"
if [ -z "$LAUNCHER" ]; then
  LAUNCHER=$(ls -t /tmp/one.sh /tmp/ab.sh /tmp/budget-probe.sh 2>/dev/null | head -1)
fi

echo "== endpoint =="
BASE=$(grep -oP '(?<=^OPENAI_BASE_URL=).*' /scratch/crux/.env 2>/dev/null)
# The name the launcher will ask for: MODEL from the environment, as one.sh
# exports it; else the launcher's own default (`${MODEL:-...}`); else a literal
# `--model openai/<name>`. A plain match on `--model openai/` reads one.sh's
# `"${MODEL:-...` as the name.
MODEL="${MODEL:-$(grep -oP '(?<=MODEL:-)[^}"]+' "$LAUNCHER" 2>/dev/null | head -1)}"
[ -n "$MODEL" ] || MODEL=$(grep -oP '(?<=--model openai/)[^ "$]+' "$LAUNCHER" 2>/dev/null | head -1)
# Removed first: when the endpoint is down, curl writes nothing, and the file
# a previous run left behind answered for it -- a dead endpoint reported the
# model list it had served days before.
rm -f /tmp/pf-models.json
code=$(curl -s -o /tmp/pf-models.json -w '%{http_code}' -m 15 "$BASE/models")
[ "$code" = "200" ] && say "reachable at $BASE" "ok" || { say "reachable at $BASE" "FAIL ($code)"; fail=1; }

win=$(python3 -c "
import json
try: j=json.load(open('/tmp/pf-models.json'))
except Exception: raise SystemExit
for e in j.get('data') or []:
    if e.get('id')=='$MODEL': print(e.get('max_model_len') or '')
" 2>/dev/null)
if [ -n "$win" ]; then
  say "context window ($MODEL)" "$win"
else
  served=$(python3 -c "
import json
try: print(' '.join(e.get('id','') for e in json.load(open('/tmp/pf-models.json')).get('data') or []))
except Exception: pass
" 2>/dev/null)
  say "context window ($MODEL)" "UNKNOWN - endpoint serves: ${served:-nothing readable}"; fail=1
fi

# The failure that cost the most: the model emits tool calls, the server does
# not parse them, and pi sees text. Nine trials took zero actions.
tc=$(curl -s -m 90 "$BASE/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"List files in /app. Use the bash tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\"required\":[\"command\"]}}}]}" \
  | python3 -c "import sys,json;j=json.load(sys.stdin);print(len((j['choices'][0]['message'].get('tool_calls') or [])))" 2>/dev/null)
[ "${tc:-0}" -ge 1 ] && say "server returns native tool_calls" "ok" || { say "server returns native tool_calls" "FAIL - needs --tool-call-parser"; fail=1; }

echo "== completion budget vs window =="
# (The launcher was resolved at the top.) This read
# `/tmp/confirm.sh` unconditionally -- a leftover from another round -- so for a
# day it answered "ok" about a script nobody was running, which is the same
# shape of hole this whole file exists to close. Pass the launcher explicitly to
# be sure: preflight.sh <launcher>
mx=$(grep -oP '(?<=PI_MAX_OUTPUT_TOKENS=)[0-9]+' "$LAUNCHER" 2>/dev/null | head -1)
say "launcher being checked" "${LAUNCHER:-NONE FOUND}"
if [ -n "$win" ] && [ -n "$mx" ]; then
  # pi_agent derives the model's own ceiling as min(32768, window // 4), and the
  # starting cap has to sit below it or the truncation-recovery escalation has
  # nowhere to go -- a silent no-op that cost a full round once.
  ceil=$(( win / 4 )); [ "$ceil" -gt 32768 ] && ceil=32768
  if [ "$mx" -ge "$ceil" ]; then
    say "PI_MAX_OUTPUT_TOKENS=$mx vs model ceiling $ceil" "FAIL - escalation cannot fire"; fail=1
  elif [ "$mx" -gt $((win / 2)) ]; then
    say "PI_MAX_OUTPUT_TOKENS=$mx vs window $win" "FAIL - leaves under half for input"; fail=1
  else
    say "PI_MAX_OUTPUT_TOKENS=$mx vs window $win, ceiling $ceil" "ok"
  fi
elif [ -z "$mx" ]; then
  say "PI_MAX_OUTPUT_TOKENS set" "FAIL - unset or launcher unreadable"; fail=1
else
  say "PI_MAX_OUTPUT_TOKENS=$mx" "UNCHECKED - window unknown (see endpoint)"; fail=1
fi

echo "== arm settings =="
for k in PI_TIME_BUDGET_SEC PI_STOP_BUDGET_SHARE; do
  v=$(grep -oP "(?<=$k=)[^,\\ ]+" "$LAUNCHER" 2>/dev/null | head -1)
  [ -n "$v" ] && say "$k" "$v" || { say "$k" "MISSING"; fail=1; }
done

echo "== the code that will actually run =="
# The check this was missing, and it cost a whole arm. A `rsync src/ ...` issued
# from the wrong directory merged pi's TypeScript over the harness package
# without --delete: the crux package still imported, every trial still ran and
# scored, and the change under test was simply not there. The arm came back
# 16.7 points below its own control and looked like a regression.
SRC=/scratch/crux-next-src
if [ -d "$SRC/crux" ] && [ ! -d "$SRC/core" ]; then
  say "harness source tree" "clean ($(ls "$SRC" | tr '\n' ' '))"
else
  say "harness source tree" "POLLUTED - $(ls "$SRC" 2>/dev/null | head -4 | tr '\n' ' ')"; fail=1
fi
# Compare what is on the box against what is committed in the checkout. This
# script runs on the box, which cannot see the checkout, so the expected sum is
# passed in -- `EXPECTED_HARNESS_SUM=$(benchmark/scripts/harness-sum.sh)`.
# Without it the check says so rather than quietly passing.
REMOTE_SUM=$(cd "$SRC" && find . -name '*.py' -not -path '*__pycache__*' | LC_ALL=C sort | xargs cat 2>/dev/null | md5sum | cut -d' ' -f1)
if [ -z "${EXPECTED_HARNESS_SUM:-}" ]; then
  say "harness matches the checkout" "UNCHECKED (pass EXPECTED_HARNESS_SUM)"
elif [ "$EXPECTED_HARNESS_SUM" = "$REMOTE_SUM" ]; then
  say "harness matches the checkout" "ok"
else
  say "harness matches the checkout" "STALE - rsync benchmark/src/ before launching"; fail=1
fi

echo "== box =="
n=$(docker ps -q | wc -l)
[ "$n" -le 4 ] && say "containers already running" "$n" \
  || { say "containers already running" "$n - another arm is live, or orphans"; fail=1; }
# Orphans from a killed arm keep competing for a GPU that is already at 100%.
# Identify them by creation time against the live run's start, never by "not in
# the list of live trials" -- that races with harbor creating the next trial,
# and removing containers out from under a running arm is what voided one of the
# five. A snapshot taken a second before a removal loop is not evidence.
say "how to clear orphans" "docker rm -f containers created before the run started"
u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1 | tr -dc 0-9)
say "gpu utilisation" "${u}% (100% means throughput is the floor, not concurrency)"

echo
[ "$fail" = 0 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT FAILED - fix the above before spending hours"
exit "$fail"

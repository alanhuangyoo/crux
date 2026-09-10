# Next round, ready to launch

Everything below is built, verified and staged. The only thing missing is the
model endpoint, which is stopped.

## Restarting the endpoint (h20-43, cards 0-3)

    python -m sglang.launch_server \
      --model-path /mnt/cpfs/users/xiaohuang/models/Qwen3.8-27B-FP8 \
      --served-model-name qwen3.8-27b --host 0.0.0.0 --port 30000

from `/mnt/cpfs/users/xiaohuang/envs/sglang-rel/bin/python`. Check for other
users' jobs on the cards in the same command that starts it, not before it.

## 1. Validate the Claude Code recovery port  (nothing has tested it)

`aed614fbd` made the truncation recovery unconditional and took Claude Code's
own wording, including the clause pi was missing -- "break remaining work into
smaller pieces". `/scratch/crux/pi-nvm-budget.tar.gz` carries it, verified in
the bundle.

The failure it targets: 12 of 13 crux failures that ended under ten tool calls
ended on `stopReason: "length"`. `regex-chess` and `polyglot-rust-c` each scored
zero after two turns and no tool calls at all.

    bash /tmp/targeted.sh z-recovery 3

Read: do `regex-chess` and `polyglot-rust-c` still end in two turns? The score
on those two is not the question -- Claude Code cannot solve them either. The
question is whether the run continues.

## 2. Full-set confirmation  (the only number that can be reported)

    bash /tmp/confirm.sh          # 89 tasks x 2, jobs-dir /scratch/y-confirm

Last attempt reached 5/178 before the endpoint stopped. Reference points:
pi 53.9 | before this round 59.0 | crux-terminus 71.9 | Claude Code 73.0 | goal 80.
The targeted arm suggests about +10, which would be 69, and that estimate comes
from a subset chosen for being where crux is worst -- expect regression toward
the mean.

## 3. The seven that did not move

`circuit-fibsqrt`, `install-windows-3.11`, `qemu-alpine-ssh`, `qemu-startup`,
`winning-avg-corewars`, and the two that regressed, `sanitize-git-repo` and
`video-processing`.

Do not open this with another corpus-wide ratio. Twenty have been tried and
rejected, and the last three were rejected specifically because one trajectory
made them look obvious. See `benchmark/docs/pi-is-the-harness-now.md`.

## Concurrency

Twelve per arm, and no more than 29 containers on the box at once -- that is the
measured inflection. Check `docker ps -q | wc -l` inside the same command that
launches.

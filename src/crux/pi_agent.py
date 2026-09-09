"""pi, with the prompt sections this project measured.

Two things pushed the work here rather than onto crux's own Terminus base.

pi already has what crux does not: an interactive terminal UI, session resume,
and a clean split between agent core, model layer and front-end. Building that
onto a benchmark scaffold means fighting its design -- Terminus's `run()` starts
a fresh Chat every call, so multi-turn would need the chat construction
monkeypatched.

And pi exposes `--append-system-prompt`, which makes a prompt change a one
variable experiment: the same base, the same model, the same tasks, with and
without the sections. Comparing crux against pi compares two codebases at once
and cannot attribute anything.

harbor's own Pi agent declares only a `thinking` flag, so the flag is added the
same way crux extends Terminus -- by subclassing at the supported extension
point, not by patching harbor.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import tempfile
from pathlib import Path

from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment
from typing_extensions import override

from crux.prompts import build_sections

logger = logging.getLogger(__name__)

# `<<FINAL_ANSWER>>`-style instructions name the file they want. Read from the
# instruction rather than hardcoded, so this stays inert on a benchmark that
# asks for nothing.
_ANSWER_PATH = re.compile(r"(/[\w./-]*answer[\w.-]*\.(?:txt|md|json))")


def _answer_path(instruction: str) -> str | None:
    m = _ANSWER_PATH.search(instruction or "")
    return m.group(1) if m else None


def _final_answer(pi_log: Path) -> str:
    """The last thing the agent said, from pi's own session log.

    Prefers the text inside `<<FINAL_ANSWER>>` markers when the agent used
    them, because that is what the task asked for and what a grader expects to
    read; otherwise the last assistant text, which is the answer given to the
    wrong channel.
    """
    if not pi_log.is_file():
        return ""
    last = ""
    try:
        with pi_log.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"role"' not in line or "assistant" not in line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                message = event.get("message") or {}
                if message.get("role") != "assistant":
                    continue
                text = " ".join(
                    str(b.get("text", ""))
                    for b in (message.get("content") or [])
                    if isinstance(b, dict) and b.get("type") == "text"
                ).strip()
                if text:
                    last = text
    except OSError:
        return ""
    marked = re.search(r"<<FINAL_ANSWER>>(.*?)(?:<</?FINAL_ANSWER>>|$)", last, re.S)
    return (marked.group(1) if marked else last).strip()


# Where the sections land inside the environment. pi reads the file at startup,
# so it has to exist before the agent command runs.
_REMOTE_PROMPT_PATH = "/tmp/crux-sections.md"

# A prebuilt $HOME/.nvm holding node and pi, so a trial does not have to reach
# the network to get its agent. Built once with `crux bake-pi`; see
# `_install_offline` for what it is worth.
_PI_BUNDLE = os.environ.get("CRUX_PI_BUNDLE", "/scratch/crux/pi-nvm.tar.gz")
_REMOTE_BUNDLE = "/tmp/pi-nvm.tar.gz"

# The submit section directs the model through `crux submit`, a tool that only
# exists inside crux's own image, so it is off by default here.
_DEFAULT_SECTIONS = ("scoring", "harness")


class CruxPiAgent(Pi):
    """pi with crux's measured prompt sections appended.

    `sections` selects them, comma separated, and an empty string runs stock pi
    -- which is the control arm, and has to be reachable through the same code
    path as the treatment so that a difference between the two is the sections
    and not the plumbing.
    """

    CLI_FLAGS = Pi.CLI_FLAGS + [
        CliFlag(
            "append_system_prompt",
            cli="--append-system-prompt",
            type="str",
        ),
    ]

    @staticmethod
    def name() -> str:
        return "crux-pi"

    def __init__(self, *args, sections: str | None = None,
                 bundle: str | None = None, pi_env: str | None = None, **kwargs):
        # Settings for the pi process itself, as `KEY=value,KEY=value`. They
        # cannot be passed through the environment of the run: harbor builds
        # that env from the model connection alone, so a variable exported on
        # the runner reaches nothing inside the container -- the same shape as
        # a tmux `-e` that the container's tmux rejects, and it looks armed
        # either way. See `_deliver_pi_env`.
        self._pi_env = dict(
            kv.split("=", 1) for kv in (pi_env or "").split(",") if "=" in kv
        )
        # Which prebuilt install to unpack. Passing it as a kwarg rather than
        # reading the env means a control arm and a treatment arm reach the
        # same code by the same path, and differ only in the bundle -- which is
        # the whole point of running them against each other.
        self._bundle_path = bundle or _PI_BUNDLE
        raw = _DEFAULT_SECTIONS if sections is None else tuple(
            s.strip() for s in str(sections).split(",") if s.strip()
        )
        self._sections = raw
        self._section_text = build_sections(raw) if raw else ""
        if self._section_text:
            kwargs.setdefault("append_system_prompt", _REMOTE_PROMPT_PATH)
        super().__init__(*args, **kwargs)

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        if not self._section_text:
            return
        # Written as a file rather than inlined into the command: the sections
        # run to thousands of characters and contain quotes and backticks, and
        # a shell-escaped blob that long is where quoting bugs live.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(self._section_text)
            local = Path(f.name)
        try:
            await environment.upload_file(local, _REMOTE_PROMPT_PATH)
        finally:
            local.unlink(missing_ok=True)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        """Unpack a prebuilt node+pi instead of fetching one per trial.

        Upstream installs the agent from scratch in every container:

            curl raw.githubusercontent.com/nvm-sh/nvm/.../install.sh | bash
            nvm install 22
            npm install -g @earendil-works/pi-coding-agent@latest

        Three network fetches per trial, under `set -euo pipefail`, at whatever
        concurrency the run uses. Any one of them failing raises
        `NonZeroAgentExitCodeError` before the agent has run, and the trial is
        scored zero.

        Measured on Terminal-Bench 2.1: **24 of pi's 89 trials died here**, all
        of them in `curl ... nvm/install.sh`, none with a single tool call
        recorded. That is 27% of the benchmark decided by a download, and it is
        why pi's headline 53.9% was not a number about pi.

        The bundle is the same install, done once. If it is missing the
        network path still runs, because a missing file should cost a slower
        setup and not the run.
        """
        bundle = Path(getattr(self, "_bundle_path", _PI_BUNDLE))
        if not bundle.is_file():
            logger.warning(
                "pi bundle %s not found; falling back to the per-trial network "
                "install that loses ~27%% of trials", bundle,
            )
            await super().install(environment)
            return

        await environment.upload_file(source_path=bundle, target_path=_REMOTE_BUNDLE)
        # `exec_as_agent`, not `environment.exec`: the latter runs as root, so
        # on an image whose agent user is not root the bundle lands in root's
        # home and the agent -- which harbor starts with `. ~/.nvm/nvm.sh` --
        # never sees it. That is what "pi: command not found" meant on the
        # SWE-Atlas images while the same bundle worked on Terminal-Bench.
        # Upstream's install uses the same helper; matching it is the point.
        result = await self.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                # `tar` is not everywhere. Nine of forty SWE-Atlas trials failed
                # here with exit 127 -- command not found -- and the network
                # fallback could not run either because those images have no
                # curl. Python's own tarfile module needs neither a package
                # manager nor the network, and an image that holds a Python
                # repository has Python.
                f'if command -v tar >/dev/null 2>&1; then tar xzf {_REMOTE_BUNDLE} -C "$HOME"; '
                f'elif command -v python3 >/dev/null 2>&1; then '
                f'python3 -m tarfile -e {_REMOTE_BUNDLE} "$HOME"; '
                f'else echo "no tar and no python3" >&2; exit 127; fi; '
                f"rm -f {_REMOTE_BUNDLE}; "
                '. "$HOME/.nvm/nvm.sh"; '
                "pi --version; "
                # Printed so the root step below can link an absolute path.
                # Resolving `$HOME/.nvm` again as root would look in root's
                # home, which is the same mistake one layer down.
                'echo "CRUX_PI_BIN=$(dirname "$(command -v pi)")"'
            ),
        )
        out = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
        if getattr(result, "return_code", 1) == 0:
            # Put the binaries somewhere that does not depend on nvm resolving a
            # default version, on `$HOME` being what install saw, or on the run
            # shell being bash. harbor starts the agent with
            # `. ~/.nvm/nvm.sh; ... pi ...` -- a semicolon, so a missing nvm.sh
            # is only a message -- and on the SWE-Atlas images that line found
            # no `pi` even after the bundle had unpacked and reported 0.85.1.
            # A symlink on the default PATH is the one thing all of those agree
            # on. Root, because /usr/local/bin is root-owned; best effort,
            # because a working nvm path must not be lost to a failed link.
            bin_dir = ""
            for line in out.splitlines():
                if line.startswith("CRUX_PI_BIN="):
                    bin_dir = line.split("=", 1)[1].strip()
            if bin_dir:
                await environment.exec(
                    "set -eu; "
                    f"for b in pi node npm npx; do "
                    f'  [ -x {shlex.quote(bin_dir)}/"$b" ] && '
                    f'    ln -sf {shlex.quote(bin_dir)}/"$b" /usr/local/bin/"$b" || true; '
                    "done; true"
                )
            else:
                logger.warning("pi bundle unpacked but its bin dir was not reported")
        if getattr(result, "return_code", 1) != 0:
            # Say what the bundle did before falling back, so a broken bundle is
            # distinguishable from a missing one.
            logger.warning("pi bundle failed to unpack (%s); falling back", out.strip()[:200])
            await super().install(environment)
            return
        logger.info("pi installed from bundle: %s", out.strip().splitlines()[-1:] or "?")

    async def _deliver_pi_env(self, environment: BaseEnvironment) -> None:
        """Put pi's own settings where its run command will read them.

        harbor starts the agent with

            . ~/.nvm/nvm.sh; PI_CODING_AGENT_DIR=... pi --print ...

        and builds the exec environment from the model connection, so there is
        no route for an arbitrary variable. That file is sourced on every run by
        construction, which makes appending the exports to it the one place a
        setting is certain to arrive. Written once at install, marked so a
        second install replaces rather than stacks.
        """
        if not self._pi_env:
            return
        exports = "; ".join(
            f"export {k}={shlex.quote(str(v))}" for k, v in sorted(self._pi_env.items())
        )
        marker = "# crux-pi-env"
        result = await self.exec_as_agent(
            environment,
            command=(
                'f="$HOME/.nvm/nvm.sh"; '
                f"grep -q {shlex.quote(marker)} \"$f\" || "
                f"printf '%s\\n' {shlex.quote(marker)} {shlex.quote(exports)} >> \"$f\"; "
                'bash -lc \'. "$HOME/.nvm/nvm.sh"; echo READY=$PI_MAX_OUTPUT_TOKENS\' 2>/dev/null || true'
            ),
        )
        out = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
        logger.info("pi settings delivered (%s): %s", exports, out.strip()[-60:])

    @override
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context) -> None:
        """Deliver the agent's final answer to the file the task named.

        SWE-Atlas scores by reading `/logs/agent/answer.txt`; its instruction
        ends with "write your complete final answer to /logs/agent/answer.txt
        wrapped in <<FINAL_ANSWER>> tags". Terminus complies -- its whole loop
        is typing into a terminal, so writing a file is the natural move, and
        its trajectories mention the path eighteen times. pi in `--print` mode
        answers the person who asked: across 31 SWE-Atlas trials it read the
        instruction, mentioned the path three times, and **never issued a
        single tool call touching it**. Every one scored zero on
        `No answer file at /logs/agent/answer.txt, scoring 0`.

        This copies what the agent already said into the path the task named.
        It writes nothing the agent did not produce, and it does not run when
        the agent wrote the file itself or when the instruction never asked for
        one. That keeps it plumbing: the difference between an answer that was
        never given and one that was given to the wrong channel.
        """
        await self._deliver_pi_env(environment)
        await super().run(instruction, environment, context)
        path = _answer_path(instruction)
        if not path:
            return
        exists = await environment.exec(f"test -s {shlex.quote(path)} && echo yes")
        if "yes" in ((getattr(exists, "stdout", "") or "") + (getattr(exists, "stderr", "") or "")):
            return
        answer = _final_answer(self.logs_dir / "pi.txt")
        if not answer:
            logger.warning("no answer file and no final message to deliver to %s", path)
            return
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(str(Path(path).parent))}; "
                f"cat > {shlex.quote(path)} <<'CRUX_ANSWER_EOF'\n"
                f"{answer}\nCRUX_ANSWER_EOF"
            ),
        )
        logger.info("delivered the agent's final message to %s (%d chars)", path, len(answer))

    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        # pi reads the argument as text or as a path; quote it so a path with a
        # space does not silently become two arguments.
        if self._section_text and _REMOTE_PROMPT_PATH in flags:
            flags = flags.replace(
                f"--append-system-prompt {_REMOTE_PROMPT_PATH}",
                f"--append-system-prompt {shlex.quote(_REMOTE_PROMPT_PATH)}",
            )
        return flags

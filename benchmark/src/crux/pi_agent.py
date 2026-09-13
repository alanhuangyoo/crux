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
import urllib.request
from pathlib import Path

from harbor.agents.installed.base import CliFlag, NonZeroAgentExitCodeError
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment
from typing_extensions import override

from crux.prompts import build_sections

logger = logging.getLogger(__name__)

# `<<FINAL_ANSWER>>`-style instructions name the file they want. Read from the
# instruction rather than hardcoded, so this stays inert on a benchmark that
# asks for nothing.
_ANSWER_PATH = re.compile(r"(/[\w./-]*answer[\w.-]*\.(?:txt|md|json))")

# The main pi invocation, as harbor builds it. Both markers are harbor's own:
# the flags it always passes, and the redirect it always appends.
_PI_MAIN = "pi --print --mode json "
_PI_TAIL = " 2>&1 </dev/null |"
# `build_cli_flags` always ends with the option terminator, so the instruction
# is whatever follows the last one.
_TERMINATOR = " -- "

# How many times a killed run may be picked back up. A process that dies three
# times is not being interrupted, it is failing.
_MAX_RESUMES = 3

# Shell exit codes for a process that was killed rather than one that failed.
# 128+n is the shell's encoding of "died on signal n": 137 is SIGKILL, which is
# what a cgroup out-of-memory kill looks like from outside, and 143 is SIGTERM.
_KILLED_EXIT_CODES = (137, 143)
_EXIT_CODE = re.compile(r"exit (\d+)")


def _was_killed(exc: Exception) -> bool:
    """Whether this failure is the kind resuming can actually help.

    Resuming was built for a container out-of-memory kill: the process is
    destroyed mid-task with the work still on disk, so picking it back up is
    free progress. A plain non-zero exit is a different animal -- the run
    decided to stop -- and retrying it three times costs three more agent runs
    and buys nothing.

    It also costs the diagnosis. When trials started failing en masse, the
    exception they carried was the *resumed* command, `pi --print ... --continue
    -- 'Your previous process was killed partway through'`, and the original
    exit code was three layers back. The real cause was containers being removed
    out from under a running arm, and the resume machinery had papered over
    every trace of it.

    An unparseable message is treated as not-killed: retrying is the action with
    a cost, so it needs the evidence.
    """
    match = _EXIT_CODE.search(str(exc))
    if not match:
        return False
    return int(match.group(1)) in _KILLED_EXIT_CODES


_RESUME_INSTRUCTION = (
    "Your previous process was killed partway through this task -- most often "
    "because a command you started exhausted the container's memory and the "
    "kernel killed the whole group, or because a `pkill -f` pattern matched "
    "your own shell. The work you already did is still on disk and this "
    "conversation is intact.\n\n"
    "Do not start over. Check what is already there, then continue from where "
    "you stopped. Whatever you restart, bound it: this container has far less "
    "memory and CPU than `nproc` reports, so pass explicit small values rather "
    "than `-j$(nproc)` or a default worker count, and match `pkill -f` "
    "patterns so they cannot match the shell running them."
)


def _served_context_length(base_url: str | None, model_id: str) -> int | None:
    """Ask the endpoint how much context it actually serves.

    `/v1/models` reports `max_model_len`, which is the one number that decides
    whether a request is accepted. Returns None on anything unexpected -- an
    endpoint that will not answer is not a reason to fail a trial, it just means
    pi keeps the default it would have had anyway.
    """
    if not base_url:
        return None
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - advisory lookup, never fatal
        logger.warning("could not read context length from %s: %s", url, exc)
        return None
    for entry in payload.get("data") or []:
        if entry.get("id") != model_id:
            continue
        for key in ("max_model_len", "context_length", "max_context_length"):
            value = entry.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None


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
                 bundle: str | None = None, pi_env: str | None = None,
                 no_thinking: str | None = None, **kwargs):
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
        # Whether to turn the model's reasoning off at the chat template.
        # Off unless asked for; see `_build_custom_models_json`.
        self._no_thinking = str(no_thinking or "").strip().lower() in ("1", "true", "yes")
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
                # fallback could not run either, because those images have no
                # curl. Python's tarfile module needs neither a package manager
                # nor the network, and an image holding a Python repository has
                # Python.
                f'if command -v tar >/dev/null 2>&1; then tar xzf {_REMOTE_BUNDLE} -C "$HOME"; '
                f'elif command -v python3 >/dev/null 2>&1; then '
                f'python3 -m tarfile -e {_REMOTE_BUNDLE} "$HOME"; '
                f'else echo "no tar and no python3" >&2; exit 127; fi; '
                f"rm -f {_REMOTE_BUNDLE}; "
                # Resolved by glob and run by absolute path. Sourcing nvm.sh and
                # calling `pi` needs nvm to select a version, which it does not
                # do on every image: the same bundle unpacked cleanly and then
                # died on `pi: command not found` under `set -eu`, so the check
                # that was meant to prove the install had worked was the thing
                # that failed it.
                'b=$(ls -d "$HOME"/.nvm/versions/node/*/bin 2>/dev/null | head -1); '
                '[ -n "$b" ] || { echo "no node bin dir after unpack" >&2; exit 1; }; '
                # The bundled node is glibc-linked, and three of eleven
                # SWE-Atlas images are Alpine: musl reports the mismatch as
                # `env: can't execute 'node': No such file or directory`, which
                # reads like a missing binary rather than an incompatible one.
                # pi's package is pure JavaScript, so one package and two node
                # binaries cover both; the musl build simply replaces the one in
                # place.
                'if [ -e /lib/ld-musl-x86_64.so.1 ] && [ -x "$HOME/.nvm/crux-musl/node" ]; then '
                '  cp "$HOME/.nvm/crux-musl/node" "$b/node"; echo CRUX_PI_LIBC=musl; '
                'else echo CRUX_PI_LIBC=glibc; fi; '
                # node invoked directly rather than through pi's `#!/usr/bin/env
                # node` shebang, which needs node on PATH -- the thing this
                # install is in the middle of arranging.
                '"$b/node" "$b/pi" --version; '
                # And the PATH written into the file harbor's run line sources,
                # which is stronger than the /usr/local/bin symlink below: it
                # does not depend on nvm selecting a version, on that directory
                # being writable, or on it being on the PATH the run inherits.
                # Two SWE-Atlas trials installed cleanly and still died on
                # `pi: command not found` at run time with the symlink in place.
                'f="$HOME/.nvm/nvm.sh"; grep -q "# crux-pi-path" "$f" 2>/dev/null || '
                'printf \'%s\\n\' "# crux-pi-path" "export PATH=\\"$b:\\$PATH\\"" >> "$f"; '
                # Recorded because three SWE-Atlas trials installed cleanly and
                # still died at run time on `pi: command not found`, with both
                # the symlink and the PATH in place. The remaining explanation
                # is that install and run do not share a HOME, and that is not
                # answerable from a trial directory after the fact.
                'echo "CRUX_PI_BIN=$b"; '
                'echo "CRUX_PI_WHO=$(id -un 2>/dev/null) HOME=$HOME PATH=$PATH"'
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

    def _build_custom_models_json(self, access, model_id):
        """Declare what the model can do, so pi stops clamping the flag away.

        harbor registers a custom endpoint as `{"id": model_id}` and nothing
        else. pi reads `model.reasoning` as false, `getSupportedThinkingLevels`
        returns `["off"]`, and `clampThinkingLevel` turns every `--thinking`
        into `off` before a request is built. Both arms of the round that was
        meant to test the reasoning level recorded `"thinkingLevel":"off"` in
        their own session logs -- the kwarg was passed, accepted, and discarded
        one layer below where anyone was looking, and the arms it produced were
        the same configuration run twice.

        Measured against this deployment (SGLang, Qwen3.8-27B), three samples
        each on one reasoning-heavy prompt, reasoning_content characters:

            baseline                       9333  10481   9423
            reasoning_effort=low           4171   4686   6216
            thinking_budget=128            9492   9364  10566
            chat_template enable_thinking  false: 0 0 0

        `reasoning_effort` halves it with no overlap between the groups.
        `thinking_budget` is ignored by this server, so pi's
        `thinkingTokenBudgetField` has nothing to talk to here. A first reading
        of a single short prompt said `reasoning_effort` was ignored too; it was
        noise, and three samples on a prompt that actually needs reasoning is
        what the question required.

        This matters because reasoning and the answer share one `max_tokens`
        here: 50 of 441 trials contain two or more consecutive turns of 20,000+
        thinking characters that emit nothing the loop can run, and 41 of those
        50 failed.

        Declaring it is not sufficient on its own. pi reads a reasoning model as
        an OpenAI reasoning model, so `useDeveloperRole` becomes
        `model.reasoning && compat.supportsDeveloperRole`, and
        `supportsDeveloperRole` is auto-detected from the base URL -- a bare
        `host:port` reads as standard OpenAI, so it is true. The system prompt
        then goes out as `role: "developer"`, which this server rejects:

            {"message":"Unexpected message role.","type":"BadRequest"} -> 400

        Every turn, so a smoke run of three tasks died at turn one with an
        empty assistant message and no exception -- the shape that reaches the
        scoreboard as three ordinary zeros. The compat override is what keeps
        the role at `system` while the model still counts as reasoning.
        """
        models_json = super()._build_custom_models_json(access, model_id)
        if not models_json:
            return models_json
        window = _served_context_length(access.configured_base_url, model_id)
        for provider in models_json.get("providers", {}).values():
            for model in provider.get("models", []):
                model["reasoning"] = True
                if window:
                    # Tell pi what the window actually is. harbor registers a
                    # custom endpoint as `{"id": ...}`, so pi falls back to a
                    # default that has nothing to do with this server, and on a
                    # 32K deployment that default is catastrophic in two ways at
                    # once: the loop does not compact until the input is already
                    # too big, and `escalatedMaxTokens` reads the window as
                    # roomy and raises the ceiling into space that is not there.
                    # Both arrive as the same opaque failure:
                    #
                    #   400: 32899 = 16515 from the input and 16384 for the
                    #        completion, against a limit of 32768
                    #
                    # Read from the server rather than configured, because a
                    # number typed here is a number that goes stale.
                    model["contextWindow"] = window
                    # A quarter of the window, capped at 32K.
                    #
                    # `min(16384, window // 2)` was written when the window was
                    # 32768, where half of it was the only sane answer. Served at
                    # the model's own 262144 the 16384 is what binds, and it binds
                    # on the wrong thing: output truncation measured 4.3% of turns
                    # at the larger window, and every truncated turn costs a
                    # recovery cycle. A quarter still leaves three quarters of the
                    # window for the history that produced the answer.
                    model.setdefault("maxTokens", min(32768, max(4096, window // 4)))
                compat = model.setdefault("compat", {})
                compat.setdefault("supportsDeveloperRole", False)
                # This server does not need the model's own reasoning sent back
                # to it -- the answer and the tool calls are the conversation.
                # Measured over one 88-task run, reasoning was 39.7% of every
                # character living in the replayed conversation, and 89% of it
                # on `regex-chess`. The trials that failed sat at a median
                # maximum context of 31,948 against a 32,768 window; the ones
                # that solved sat at 20,787.
                compat.setdefault("replaysReasoning", False)
                if self._no_thinking:
                    # `reasoning_effort` biases this model, it does not cap it.
                    # Measured on the five tasks where reasoning is what loses
                    # the run: with effort=low the median thinking block is
                    # still 41,888-54,778 characters, against 47,291-56,964
                    # without it, and every run still ends on `length`. The
                    # output ceiling makes it worse rather than better, exactly
                    # as designed -- 8,192 truncates, the loop silently raises
                    # to 16,384, and the model spends that on thinking too.
                    # More room was the wrong lever.
                    #
                    # Which wire format, measured against this server, two
                    # samples each on one reasoning-heavy prompt (characters of
                    # reasoning_content):
                    #
                    #   baseline                                  9730  11358
                    #   chat_template_kwargs{enable_thinking:0}      0      0
                    #   ... plus preserve_thinking                   0      0
                    #   chat_template_args{enable_thinking:0}     9211   9046
                    #   top-level enable_thinking:false           9011   8828
                    #
                    # So `chat_template_args` and the top-level flag are both
                    # ignored here, which rules out pi's "baseten" and "qwen"
                    # formats. The first version of this shipped the `baseten`
                    # field, the probe ran with reasoning fully on, and only
                    # counting thinking blocks in its own trajectories caught
                    # it -- the models.json inside the container was exactly
                    # what it was meant to be.
                    compat["thinkingFormat"] = "chat-template"
                    kwargs = compat.setdefault("chatTemplateKwargs", {})
                    kwargs.setdefault("enable_thinking", False)
                    kwargs.setdefault("preserve_thinking", True)
        return models_json

    @override
    async def exec_as_agent(self, environment, command, **kwargs):
        """Pick a killed run back up instead of scoring it a zero.

        pi runs inside the trial container; Terminus drives the same container
        from outside. That is not a stylistic difference -- a cgroup out-of-
        memory kill takes every process in the group, so a task that exhausts
        memory ends Terminus's command and ends pi's *run*. Measured over 501
        trials, seven died on exit 137 across `rstan-to-pystan`,
        `install-windows-3.11` and `mcmc-sampling-stan`; the trajectories show
        two causes and neither is pi's own footprint, which is 89 MB against a
        2 GiB limit. Some are `-j$(nproc)` inside a container where `nproc`
        reports the host's 192 cores, and some are the agent's own `pkill -f`
        matching the shell that issued it -- pi diagnosed both, in its own
        words, in the run that then died.

        `pi --continue` restores the session, verified end to end: a run told
        to write BANANA into one file, then resumed with a second instruction
        that never repeats the word, writes BANANA into the second file and
        keeps one session file. So the recovery is real rather than a fresh
        agent that happens to share a directory.

        Only the main pi invocation is retried, and only three times.
        """
        if _PI_MAIN not in command:
            return await super().exec_as_agent(environment, command, **kwargs)

        attempt = 0
        while True:
            try:
                return await super().exec_as_agent(environment, command, **kwargs)
            except NonZeroAgentExitCodeError as exc:
                attempt += 1
                resumed = self._resume_command(command)
                if resumed is None or attempt > _MAX_RESUMES or not _was_killed(exc):
                    raise
                logger.warning(
                    "pi exited non-zero (%s); resuming session, attempt %d of %d",
                    str(exc)[:120],
                    attempt,
                    _MAX_RESUMES,
                )
                command = resumed

    @staticmethod
    def _resume_command(command: str) -> str | None:
        """Rewrite the run command to resume rather than restart.

        Returns None when the command is not the shape this knows how to
        rewrite, so an unexpected template raises the original error instead of
        running something half-rebuilt.
        """
        head, tail_sep, tail = command.partition(_PI_TAIL)
        if not tail_sep:
            return None
        prefix, term, _instruction = head.rpartition(_TERMINATOR)
        if not term:
            return None
        if _PI_MAIN not in prefix:
            return None
        if "--continue " not in prefix:
            prefix = prefix.replace(_PI_MAIN, _PI_MAIN + "--continue ", 1)
        return f"{prefix}{_TERMINATOR}{shlex.quote(_RESUME_INSTRUCTION)} {tail_sep}{tail}"

    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        # pi reads the argument as text or as a path; quote it so a path with a
        # space does not silently become two arguments.
        if self._section_text and _REMOTE_PROMPT_PATH in flags:
            flags = flags.replace(
                f"--append-system-prompt {_REMOTE_PROMPT_PATH}",
                f"--append-system-prompt {shlex.quote(_REMOTE_PROMPT_PATH)}",
            )
        # `--` ends option parsing, so a task whose text begins with a hyphen is
        # read as the prompt rather than as a flag. harbor builds the command as
        # `pi --print ... '<instruction>'` with no terminator, and pi's parser
        # rejects any single-hyphen argument:
        #
        #     Error: Unknown option: - You are given a PyTorch state dictionary
        #
        # `pytorch-model-recovery` states its task as a markdown list, so its
        # first character is "-". Every pi trial on it died before taking an
        # action, in every arm, while six other scaffolds solved it. pi already
        # honours `--`; nothing was passing it one.
        #
        # The first version of this appended `"-- "` to a string that does not
        # end in one, which glued the terminator to the previous argument:
        #
        #     --append-system-prompt /tmp/crux-sections.md--  '- You are given'
        #
        # so the flag took a path that does not exist, no terminator was ever
        # parsed, and the task died exactly as before -- three more trials, same
        # zero, from the fix. rstrip-then-join is the whole correction.
        return f"{flags.rstrip()} -- " if flags.strip() else "-- "

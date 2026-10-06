"""Crux on pi: the harbor agent that runs pi inside a task container.

pi brings a terminal UI, session resume, and a clean split between agent core,
model layer and front-end, and it exposes `--append-system-prompt`, so a prompt
change is a single-variable experiment on the same base, model and tasks.
harbor's own Pi agent is extended at its supported extension points rather
than patched.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import tempfile
import tomllib
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

    Resuming is for an out-of-memory kill: the process is destroyed mid-task with
    the work still on disk, so picking it back up is free progress. A plain
    non-zero exit means the run decided to stop, and retrying it costs runs and
    hides the original cause. An unparseable message is treated as not-killed.
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

# A prebuilt $HOME/.nvm holding node and pi, so a trial does not reach the
# network to get its agent. Built once with `crux bake-pi`.
_PI_BUNDLE = os.environ.get("CRUX_PI_BUNDLE", "/scratch/crux/pi-nvm.tar.gz")
_REMOTE_BUNDLE = "/tmp/pi-nvm.tar.gz"

# The submit section directs the model through `crux submit`, a tool that only
# exists inside crux's own image, so it is off by default here.
_DEFAULT_SECTIONS = ("scoring", "harness")

# How long before harbor's kill pi's own clock runs out. pi winds down on its
# deadline, and it starts a few seconds after harbor's clock does.
_DEADLINE_MARGIN_SEC = 60


def _harbor_agent_limit_sec(task_dir: Path, trial_dir: Path) -> float | None:
    """The agent timeout harbor will enforce on this trial, computed its way.

    Mirrors `Trial._compute_agent_timeout_sec`: the override if one is set, else
    the task's `[agent] timeout_sec`, capped by `max_timeout_sec`, times the
    agent timeout multiplier -- which falls back to `timeout_multiplier`, since
    the trial's config.json leaves out anything left at its default. None when
    any piece cannot be read, so the configured budget stands.
    """
    try:
        task = tomllib.loads((task_dir / "task.toml").read_text())
        config = json.loads((trial_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    agent = config.get("agent") or {}
    base = agent.get("override_timeout_sec") or (task.get("agent") or {}).get("timeout_sec")
    if not base:
        return None
    cap = agent.get("max_timeout_sec")
    multiplier = config.get("agent_timeout_multiplier")
    if multiplier is None:
        multiplier = config.get("timeout_multiplier", 1.0)
    return min(float(base), float(cap) if cap else float("inf")) * float(multiplier)


def _time_budget(configured: str | None, limit_sec: float | None) -> str | None:
    """`PI_TIME_BUDGET_SEC` for one trial, given harbor's limit on it.

    harbor's limit varies by task (the task's own timeout times the multiplier),
    so a fixed budget either overruns it or stops short of it. A configured number
    is capped at the limit less a margin, which only ever shortens it, so runs
    stay comparable; `task` takes the whole limit.
    """
    if limit_sec is None:
        return None if configured == "task" else configured
    ceiling = max(1, int(limit_sec) - _DEADLINE_MARGIN_SEC)
    if configured == "task":
        return str(ceiling)
    try:
        seconds = float(configured) if configured is not None else 0.0
    except ValueError:
        return configured
    if seconds <= 0:
        return configured
    return str(min(int(seconds), ceiling))


class CruxPiAgent(Pi):
    """pi with Crux's prompt sections appended.

    `sections` selects them, comma separated; an empty string runs stock pi, so
    the control and treatment go through the same code path.
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
        # Settings for the pi process itself, as `KEY=value,KEY=value`. harbor builds
        # the run's environment from the model connection alone, so they are delivered
        # into the container instead; see `_deliver_pi_env`.
        self._pi_env = dict(
            kv.split("=", 1) for kv in (pi_env or "").split(",") if "=" in kv
        )
        # Which prebuilt install to unpack. A kwarg rather than an env var, so two
        # arms reach the same code by the same path and differ only in the bundle.
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

        Upstream installs nvm, node and pi from the network in every container, and
        any failed download fails the trial before the agent has run. The bundle is
        the same install, done once. If it is missing, the network path still runs.
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
        # `exec_as_agent`, not `environment.exec`: the latter runs as root, so on an
        # image whose agent user is not root the bundle would land in root's home.
        result = await self.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                # `tar` is not on every image; Python's tarfile needs neither a package
                # manager nor the network.
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
                # The bundled node is glibc-linked; on musl (Alpine) images the musl build
                # replaces it in place. pi's package is pure JavaScript, so one package and two
                # node binaries cover both.
                'if [ -e /lib/ld-musl-x86_64.so.1 ] && [ -x "$HOME/.nvm/crux-musl/node" ]; then '
                '  cp "$HOME/.nvm/crux-musl/node" "$b/node"; echo CRUX_PI_LIBC=musl; '
                'else echo CRUX_PI_LIBC=glibc; fi; '
                # node invoked directly rather than through pi's `#!/usr/bin/env
                # node` shebang, which needs node on PATH -- the thing this
                # install is in the middle of arranging.
                '"$b/node" "$b/pi" --version; '
                # The PATH is written into the file harbor's run line sources, so it does not
                # depend on nvm selecting a version or on the PATH the run inherits.
                'f="$HOME/.nvm/nvm.sh"; grep -q "# crux-pi-path" "$f" 2>/dev/null || '
                'printf \'%s\\n\' "# crux-pi-path" "export PATH=\\"$b:\\$PATH\\"" >> "$f"; '
                # Recorded so the trial directory shows where pi was installed.
                'echo "CRUX_PI_BIN=$b"; '
                'echo "CRUX_PI_WHO=$(id -un 2>/dev/null) HOME=$HOME PATH=$PATH"'
            ),
        )
        out = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
        if getattr(result, "return_code", 1) == 0:
            # Also link the binaries into /usr/local/bin, the one location that does not
            # depend on nvm, `$HOME` or the run shell. Best effort: a working nvm path must
            # not be lost to a failed link.
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

    def _pace_to_harbor(self, environment: BaseEnvironment) -> None:
        """Fit pi's time budget to the limit harbor enforces on this trial.

        harbor hands the limit only to its own oracle agent, so it is worked
        out here from the task's task.toml -- the environment directory's
        parent -- and the trial's config.json. See `_time_budget`.
        """
        configured = self._pi_env.get("PI_TIME_BUDGET_SEC")
        if configured is None:
            return
        env_dir = getattr(environment, "environment_dir", None)
        limit = (
            _harbor_agent_limit_sec(Path(env_dir).parent, Path(self.logs_dir).parent)
            if env_dir is not None
            else None
        )
        budget = _time_budget(configured, limit)
        if budget is None:
            logger.warning("time budget %r needs harbor's limit and it could not be read; running without one", configured)
            self._pi_env = {k: v for k, v in self._pi_env.items() if k != "PI_TIME_BUDGET_SEC"}
            return
        self._pi_env = {**self._pi_env, "PI_TIME_BUDGET_SEC": budget}
        logger.info("time budget: configured %s, harbor limit %s, using %s", configured, limit, budget)

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

        Some benchmarks score an answer file named in the instruction. pi in `--print`
        mode answers in its output, so this copies what the agent already said into
        that path. It writes nothing the agent did not produce, and it does not run
        when the agent wrote the file itself or the instruction asked for none.
        """
        self._pace_to_harbor(environment)
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
        """Declare what the model can do, so pi does not clamp it away.

        harbor registers a custom endpoint as `{"id": model_id}` and nothing else, so
        pi would read the model as non-reasoning and turn every `--thinking` into
        `off`. This declares reasoning, the served context window and an output
        ceiling.

        It also keeps the system prompt on the `system` role: pi treats a reasoning
        model as an OpenAI reasoning model and would send `developer`, which
        OpenAI-compatible servers such as SGLang reject.
        """
        models_json = super()._build_custom_models_json(access, model_id)
        if not models_json:
            return models_json
        window = _served_context_length(access.configured_base_url, model_id)
        for provider in models_json.get("providers", {}).values():
            for model in provider.get("models", []):
                model["reasoning"] = True
                if window:
                    # Tell pi the window the server actually serves, read from the server rather
                    # than configured: compaction and the output-ceiling escalation both depend on
                    # it, and a default unrelated to this server would let requests overflow.
                    model["contextWindow"] = window
                    # A quarter of the window, capped at 32K: room for long answers while
                    # leaving three quarters of the window for the history that produced them.
                    model.setdefault("maxTokens", min(32768, max(4096, window // 4)))
                compat = model.setdefault("compat", {})
                compat.setdefault("supportsDeveloperRole", False)
                # The answer and the tool calls are the conversation; replaying the model's
                # own reasoning would only spend context.
                compat.setdefault("replaysReasoning", False)
                if self._no_thinking:
                    # `reasoning_effort` biases this model rather than capping it, so turning
                    # reasoning off goes through the chat template. On SGLang only
                    # `chat_template_kwargs` reaches the template; `chat_template_args` and a
                    # top-level `enable_thinking` are ignored.
                    compat["thinkingFormat"] = "chat-template"
                    kwargs = compat.setdefault("chatTemplateKwargs", {})
                    kwargs.setdefault("enable_thinking", False)
                    kwargs.setdefault("preserve_thinking", True)
        return models_json

    @override
    async def exec_as_agent(self, environment, command, **kwargs):
        """Pick a killed run back up instead of scoring it a zero.

        pi runs inside the trial container, so a cgroup out-of-memory kill -- from a
        build using every host core, or a broad `pkill -f` -- ends the whole run, not
        just a command. `pi --continue` restores the session, so the work carries on.
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
        # `--` ends option parsing, so a task whose text begins with a hyphen (a
        # markdown list, say) is read as the prompt rather than as a flag. The flags
        # are right-stripped first, so the terminator stands as its own argument.
        return f"{flags.rstrip()} -- " if flags.strip() else "-- "

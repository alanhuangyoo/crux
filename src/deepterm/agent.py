"""DeepTerm agent — Harbor BaseAgent implementation.

Design notes
------------
The loop is deliberately minimal (see docs/RESEARCH.md §5): the Tmax paper's
finding is that architectural novelty does not move Terminal-Bench scores —
prompt quality, tool ergonomics, and context management do. So the structure
here stays close to mini-swe-agent's proven shape:

    LLM -> bash command -> execute in environment -> feed output back

and the optimization work happens in the prompt templates and the observation
formatting, not in the control flow.

``SUPPORTS_ATIF = True`` is a hard requirement for leaderboard eligibility:
Harbor's CI rejects any submission whose passing trials lack an ATIF
trajectory (docs/RESEARCH.md §4).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from deepterm import __version__

# The model is asked to emit exactly one bash block per turn. Keeping the
# contract this narrow is what keeps the parser (and the failure modes) simple.
ACTION_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

# Sentinel the model prints when it considers the task finished.
DONE_MARKER = "DEEPTERM_TASK_COMPLETE"


class DeepTermAgent(BaseAgent):
    """A bash-loop terminal agent, tuned for DeepSeek models."""

    SUPPORTS_ATIF: bool = True
    SUPPORTS_WINDOWS: bool = False

    def __init__(
        self,
        *args,
        step_limit: int = 60,
        wall_time_limit_sec: int = 0,
        command_timeout_sec: int = 300,
        max_output_chars: int = 8000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.step_limit = int(step_limit)
        self.wall_time_limit_sec = int(wall_time_limit_sec)
        self.command_timeout_sec = int(command_timeout_sec)
        self.max_output_chars = int(max_output_chars)

        self.messages: list[dict] = []
        self.n_steps = 0
        self._start_time = 0.0

    @staticmethod
    @override
    def name() -> str:
        return "deepterm"

    @override
    def version(self) -> str:
        return __version__

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install.

        The loop runs host-side and reaches into the sandbox via
        ``environment.exec``, so the task container stays exactly as the
        benchmark built it. That matters: mutating the environment during
        setup is the kind of thing the /judge pass flags as harness cheating.
        """
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._start_time = time.time()
        self.messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._instance_prompt(instruction)},
        ]

        while not self._limits_exceeded():
            self.n_steps += 1

            reply = await self._query_model()
            self.messages.append({"role": "assistant", "content": reply})

            if DONE_MARKER in reply:
                break

            action = self._parse_action(reply)
            if action is None:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "No bash block found. Reply with exactly one "
                            "```bash``` block, or print "
                            f"{DONE_MARKER} if the task is done."
                        ),
                    }
                )
                continue

            result = await environment.exec(
                action, timeout_sec=self.command_timeout_sec
            )
            self.messages.append(
                {"role": "user", "content": self._format_observation(result)}
            )

        self._write_trajectory()
        context.metadata = {
            "n_steps": self.n_steps,
            "elapsed_sec": round(time.time() - self._start_time, 1),
            "agent_version": self.version(),
        }

    # ---- internals -------------------------------------------------------

    def _limits_exceeded(self) -> bool:
        if 0 < self.step_limit <= self.n_steps:
            return True
        if 0 < self.wall_time_limit_sec <= (time.time() - self._start_time):
            return True
        return False

    def _system_prompt(self) -> str:
        # TODO(phase-3): this is the single highest-leverage file in the repo.
        return (
            "You are a terminal agent. You solve tasks by running bash "
            "commands in a Linux container.\n\n"
            "Reply with exactly one ```bash``` code block per turn. The block "
            "is executed and you receive its stdout, stderr and exit code.\n"
            "Commands are non-interactive: never launch an editor, pager, or "
            "anything that waits for input.\n"
            f"When the task is fully complete, reply with {DONE_MARKER}."
        )

    def _instance_prompt(self, instruction: str) -> str:
        return f"Task:\n\n{instruction}"

    def _parse_action(self, reply: str) -> str | None:
        match = ACTION_RE.search(reply)
        return match.group(1).strip() if match else None

    def _format_observation(self, result) -> str:
        """Render an ExecResult back to the model.

        Long outputs are truncated in the middle — the head carries the
        command's intent and the tail carries the error, and it is the middle
        that is safe to drop.
        """
        parts = [f"exit_code: {result.return_code}"]
        for label, stream in (("stdout", result.stdout), ("stderr", result.stderr)):
            text = (stream or "").strip()
            if not text:
                continue
            if len(text) > self.max_output_chars:
                half = self.max_output_chars // 2
                omitted = len(text) - 2 * half
                text = (
                    f"{text[:half]}\n\n... [{omitted} chars omitted] ...\n\n"
                    f"{text[-half:]}"
                )
            parts.append(f"{label}:\n{text}")
        return "\n\n".join(parts)

    async def _query_model(self) -> str:
        """Call the model.

        TODO(phase-1): wire to litellm against ``self.model_name`` and record
        token counts + cost into AgentContext.
        """
        raise NotImplementedError(
            "Model client not wired yet — see Phase 1 in README.md"
        )

    def _write_trajectory(self) -> None:
        """Emit the ATIF trajectory Harbor's CI requires.

        TODO(phase-2): build a proper ``harbor.models.trajectories.Trajectory``
        (Step/ToolCall/Observation) instead of this raw message dump.
        """
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = Path(self.logs_dir) / "messages.json"
        path.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False))

"""Crux agent — Harbor BaseAgent implementation.

The loop is deliberately minimal. The Tmax paper's finding is that
architectural novelty does not move Terminal-Bench scores; prompt quality,
tool ergonomics, and context management do (docs/RESEARCH.md §5). So the
control flow stays close to mini-swe-agent's proven shape —

    model -> one bash command -> execute -> feed the output back

— and the tuning work happens in the prompt templates and in how observations
are rendered, not in the plumbing.

``SUPPORTS_ATIF = True`` is a hard requirement for leaderboard eligibility:
Harbor's CI rejects submissions whose passing trials carry no ATIF trajectory.
"""

from __future__ import annotations

import re
import time
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from crux import __version__
from crux.model import ModelClient
from crux.trajectory import TrajectoryRecorder

# One bash block per turn. Keeping the contract this narrow keeps the parser —
# and the ways it can go wrong — small.
ACTION_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)

# Printed by the model when it considers the task finished.
DONE_MARKER = "CRUX_TASK_COMPLETE"

SYSTEM_PROMPT = """\
You are a terminal agent solving a task inside a Linux container. You act by \
running bash commands.

Rules:
- Reply with exactly ONE ```bash code block per turn. It is executed and you \
receive its stdout, stderr, and exit code.
- Commands are non-interactive. Never start an editor, a pager, or anything \
that waits for input. Use non-interactive flags (-y, --no-pager, --yes).
- Long-running commands should be given a sensible timeout so a hang does not \
consume the whole budget.
- Verify your work before declaring completion: re-read the file you wrote, \
re-run the test you fixed.
- When the task is fully complete and verified, reply with {done_marker} and \
no bash block.
"""

INSTANCE_PROMPT = """\
Task:

{instruction}
"""


class CruxAgent(BaseAgent):
    """A bash-loop terminal agent."""

    SUPPORTS_ATIF: bool = True
    SUPPORTS_WINDOWS: bool = False

    def __init__(
        self,
        *args,
        step_limit: int = 80,
        wall_time_limit_sec: int = 0,
        command_timeout_sec: int = 300,
        max_output_chars: int = 8000,
        max_consecutive_format_errors: int = 3,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.step_limit = int(step_limit)
        self.wall_time_limit_sec = int(wall_time_limit_sec)
        self.command_timeout_sec = int(command_timeout_sec)
        self.max_output_chars = int(max_output_chars)
        self.max_consecutive_format_errors = int(max_consecutive_format_errors)

        self._client = ModelClient(
            model_name=self.model_name or "",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.messages: list[dict] = []
        self.n_steps = 0
        self._start_time = 0.0

    @staticmethod
    @override
    def name() -> str:
        return "crux"

    @override
    def version(self) -> str:
        return __version__

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install.

        The loop runs host-side and reaches into the sandbox through
        ``environment.exec``, so the task container stays exactly as the
        benchmark built it. That matters: mutating the environment during
        setup is what the /judge pass flags as harness cheating.
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
        recorder = TrajectoryRecorder(
            logs_dir=self.logs_dir,
            agent_name=self.name(),
            agent_version=self.version(),
            model_name=self.model_name,
            session_id=self.session_id,
        )

        system_prompt = SYSTEM_PROMPT.format(done_marker=DONE_MARKER)
        instance_prompt = INSTANCE_PROMPT.format(instruction=instruction)
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        recorder.record_system(system_prompt)
        recorder.record_user(instance_prompt)

        exit_reason = "completed"
        n_format_errors = 0

        try:
            while True:
                if 0 < self.step_limit <= self.n_steps:
                    exit_reason = "step_limit"
                    break
                if 0 < self.wall_time_limit_sec <= time.time() - self._start_time:
                    exit_reason = "wall_time_limit"
                    break

                self.n_steps += 1
                reply = await self._client.complete(self.messages)
                self.messages.append({"role": "assistant", "content": reply.content})

                if DONE_MARKER in reply.content:
                    recorder.record_agent_step(
                        message=reply.content,
                        reasoning_content=reply.reasoning_content,
                        prompt_tokens=reply.usage.prompt_tokens,
                        completion_tokens=reply.usage.completion_tokens,
                        cached_tokens=reply.usage.cached_tokens,
                        cost_usd=reply.usage.cost_usd,
                    )
                    self._publish(context, recorder)
                    break

                command = self._parse_action(reply.content)

                if command is None:
                    n_format_errors += 1
                    if 0 < self.max_consecutive_format_errors <= n_format_errors:
                        exit_reason = "repeated_format_error"
                        recorder.record_agent_step(
                            message=reply.content,
                            reasoning_content=reply.reasoning_content,
                            prompt_tokens=reply.usage.prompt_tokens,
                            completion_tokens=reply.usage.completion_tokens,
                            cached_tokens=reply.usage.cached_tokens,
                            cost_usd=reply.usage.cost_usd,
                        )
                        self._publish(context, recorder)
                        break
                    nudge = (
                        "No bash block found. Reply with exactly one ```bash "
                        f"block, or {DONE_MARKER} if the task is done."
                    )
                    self.messages.append({"role": "user", "content": nudge})
                    recorder.record_agent_step(
                        message=reply.content,
                        reasoning_content=reply.reasoning_content,
                        output=nudge,
                        prompt_tokens=reply.usage.prompt_tokens,
                        completion_tokens=reply.usage.completion_tokens,
                        cached_tokens=reply.usage.cached_tokens,
                        cost_usd=reply.usage.cost_usd,
                    )
                    self._publish(context, recorder)
                    continue

                n_format_errors = 0
                result = await environment.exec(
                    command, timeout_sec=self.command_timeout_sec
                )
                observation = self._format_observation(result)
                self.messages.append({"role": "user", "content": observation})

                recorder.record_agent_step(
                    message=reply.content,
                    reasoning_content=reply.reasoning_content,
                    command=command,
                    output=observation,
                    exit_code=result.return_code,
                    prompt_tokens=reply.usage.prompt_tokens,
                    completion_tokens=reply.usage.completion_tokens,
                    cached_tokens=reply.usage.cached_tokens,
                    cost_usd=reply.usage.cost_usd,
                )
                self._publish(context, recorder)

        except Exception as exc:
            exit_reason = f"error: {type(exc).__name__}"
            raise
        finally:
            # Runs on the timeout/kill path too, so the trial still leaves a
            # trajectory and usage numbers behind.
            recorder.set_notes(f"exit_reason={exit_reason}")
            self._publish(context, recorder, exit_reason=exit_reason)

    # ---- internals -------------------------------------------------------

    def _publish(
        self,
        context: AgentContext,
        recorder: TrajectoryRecorder,
        exit_reason: str | None = None,
    ) -> None:
        """Mirror the running totals into the AgentContext.

        Updated every step rather than once at the end: if the harness kills
        the trial, whatever was reported last is what gets recorded.
        """
        context.n_input_tokens = recorder.total_prompt_tokens
        context.n_cache_tokens = recorder.total_cached_tokens
        context.n_output_tokens = recorder.total_completion_tokens
        context.cost_usd = recorder.total_cost_usd
        context.metadata = {
            "n_steps": self.n_steps,
            "elapsed_sec": round(time.time() - self._start_time, 1),
            "agent_version": self.version(),
            **({"exit_reason": exit_reason} if exit_reason else {}),
        }

    def _parse_action(self, reply: str) -> str | None:
        match = ACTION_RE.search(reply)
        return match.group(1).strip() if match else None

    def _format_observation(self, result) -> str:
        """Render an ExecResult back to the model.

        Long output is truncated in the middle: the head carries what the
        command was doing and the tail carries how it ended, and the middle is
        the part that can be dropped without losing the thread.
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

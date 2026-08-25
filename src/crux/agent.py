"""Crux agent — Harbor BaseAgent implementation.

Architecture follows Terminus 2, the strongest open scaffold on this benchmark
(80.4% on terminal-bench@2.1): a persistent tmux session, a batch of keystrokes
per turn, and the terminal screen as the observation. Crux reuses Harbor's own
``TmuxSession`` and ``TerminusXMLPlainParser`` rather than reimplementing them —
they are already battle-tested against these tasks, and rewriting them would
only add new bugs.

v1 did the naive thing instead: one ``exec`` per model call, no session state.
Both smoke-test tasks burned through the 80-step budget without finishing, which
is what motivated the rewrite — a turn that can only carry one command spends
most of its budget on `cd` and `ls`.

``SUPPORTS_ATIF = True`` is a hard requirement for leaderboard eligibility:
Harbor's CI rejects submissions whose passing trials carry no ATIF trajectory.
"""

from __future__ import annotations

import time
import re
from pathlib import Path
from typing import override

from harbor.agents.base import BaseAgent
from harbor.agents.terminus_2.terminus_xml_plain_parser import TerminusXMLPlainParser
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from crux import __version__
from crux.config import build_config
from crux.context import (
    HANDOFF_PROMPT,
    build_resume_messages,
    should_compact,
)
from crux.model import ModelClient
from crux.prompts import (
    APPLY_PATCH_SECTION,
    FORMAT_ERROR_PROMPT,
    INSTANCE_PROMPT,
    SYSTEM_PROMPT,
    TIMEOUT_PROMPT,
    VERIFY_PROMPT,
)
from crux.trajectory import TrajectoryRecorder

APPLY_PATCH_SRC = Path(__file__).parent / "resources" / "apply_patch.py"
APPLY_PATCH_DEST = "/usr/local/bin/apply_patch"


class CruxAgent(BaseAgent):
    """A tmux-driven terminal agent."""

    SUPPORTS_ATIF: bool = True
    SUPPORTS_WINDOWS: bool = False

    def __init__(self, *args, **kwargs):
        # Harbor forwards --ak key=value pairs here. Anything the config knows
        # about is a knob; everything else belongs to BaseAgent.
        from crux.config import CruxConfig

        own = {k: kwargs.pop(k) for k in list(kwargs) if k in CruxConfig.model_fields}
        super().__init__(*args, **kwargs)
        self.config = build_config(**own)

        self._client = ModelClient(
            model_name=self.model_name or "",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            request_timeout_sec=self.config.request_timeout_sec,
            max_retries=self.config.max_retries,
        )
        self._parser = TerminusXMLPlainParser()
        self._session: TmuxSession | None = None
        self._apply_patch_ready = False

        self.messages: list[dict] = []
        self.n_steps = 0
        self.n_compactions = 0
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
        """Start the tmux session and install the editing helper.

        Nothing else is added to the image. The loop runs host-side and reaches
        in through the environment, so the task container stays as the
        benchmark built it — mutating it beyond the agent's own tooling is what
        the /judge pass flags as harness cheating.
        """
        self._session = TmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "crux.pane",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
            pane_width=self.config.pane_width,
            pane_height=self.config.pane_height,
            extra_env=self.extra_env,
            user=environment.default_user,
        )
        await self._session.start()

        if self.config.enable_apply_patch:
            self._apply_patch_ready = await self._install_apply_patch(environment)

    async def _install_apply_patch(self, environment: BaseEnvironment) -> bool:
        """Copy the helper in. Returns whether it is actually usable.

        Task images vary, and a few have no python3. The prompt only advertises
        the tool when this returns True: describing a tool that is not on PATH
        costs the agent turns on command-not-found before it falls back.
        """
        try:
            await environment.upload_file(APPLY_PATCH_SRC, APPLY_PATCH_DEST)
            await environment.exec(f"chmod +x {APPLY_PATCH_DEST}")
            probe = await environment.exec(
                f"{APPLY_PATCH_DEST} '*** Begin Patch\n*** End Patch' 2>&1 || true"
            )
            output = (probe.stdout or "") + (probe.stderr or "")
            # An empty patch is rejected on purpose; reaching that specific
            # complaint proves the interpreter ran the script.
            if "no file operations" in output:
                return True
            self.logger.warning("apply_patch unusable in this image: %s", output[:200])
        except Exception as exc:
            self.logger.warning("apply_patch install failed: %s", exc)
        return False

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        assert self._session is not None, "setup() must run before run()"
        self._start_time = time.time()

        recorder = TrajectoryRecorder(
            logs_dir=self.logs_dir,
            agent_name=self.name(),
            agent_version=self.version(),
            model_name=self.model_name,
            session_id=self.session_id,
        )

        system_prompt = SYSTEM_PROMPT.format(
            apply_patch_section=APPLY_PATCH_SECTION if self._apply_patch_ready else ""
        )
        terminal_state = await self._session.capture_pane()
        first_prompt = INSTANCE_PROMPT.format(
            instruction=instruction, terminal_state=terminal_state
        )
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]
        recorder.record_system(system_prompt)
        recorder.record_user(first_prompt)

        exit_reason = "completed"
        n_format_errors = 0
        n_verify_rounds = 0

        try:
            while True:
                if 0 < self.config.step_limit <= self.n_steps:
                    exit_reason = "step_limit"
                    break
                if 0 < self.config.wall_time_limit_sec <= time.time() - self._start_time:
                    exit_reason = "wall_time_limit"
                    break

                if self.config.enable_compaction and should_compact(
                    self.model_name or "",
                    self.messages,
                    self.config.compaction_threshold_tokens,
                    self.config.max_input_tokens,
                    self.config.context_window_cap,
                ):
                    await self._compact(instruction, system_prompt, recorder)

                self.n_steps += 1
                reply = await self._client.complete(self.messages)
                self.messages.append({"role": "assistant", "content": reply.content})

                parsed = self._parse(reply.content)

                if parsed.error:
                    n_format_errors += 1
                    if 0 < self.config.max_consecutive_format_errors <= n_format_errors:
                        exit_reason = "repeated_format_error"
                        self._record(recorder, reply, parsed)
                        self._publish(context, recorder)
                        break
                    nudge = FORMAT_ERROR_PROMPT.format(
                        error=parsed.error,
                        terminal_state=await self._session.capture_pane(),
                    )
                    self.messages.append({"role": "user", "content": nudge})
                    self._record(recorder, reply, parsed, observation=nudge)
                    self._publish(context, recorder)
                    continue

                n_format_errors = 0
                timed_out, elapsed = await self._send(parsed.commands)
                terminal_state = await self._session.capture_pane()

                self._record(
                    recorder,
                    reply,
                    parsed,
                    observation=terminal_state,
                    commands=parsed.commands,
                )

                if not parsed.is_task_complete:
                    # The model went back to work rather than re-asserting, so
                    # a later claim is a fresh one and earns a fresh challenge.
                    # This must not reset while a challenge is in flight: the
                    # reply to one carries verifying commands of its own, and
                    # resetting on those would re-challenge forever.
                    n_verify_rounds = 0
                else:
                    # A completion claim is challenged, not taken at face
                    # value: most of them were wrong. Accept it only once the
                    # model has been asked for evidence and stood by it.
                    if (
                        self.config.verify_before_complete
                        and n_verify_rounds < self.config.max_verify_rounds
                    ):
                        n_verify_rounds += 1
                        challenge = VERIFY_PROMPT.format(
                            instruction=instruction, terminal_state=terminal_state
                        )
                        self.messages.append({"role": "user", "content": challenge})
                        self._publish(context, recorder)
                        continue
                    exit_reason = (
                        "completed_verified" if n_verify_rounds else "completed"
                    )
                    self._publish(context, recorder)
                    break

                if timed_out:
                    follow_up = TIMEOUT_PROMPT.format(
                        duration=elapsed, terminal_state=terminal_state
                    )
                else:
                    follow_up = INSTANCE_PROMPT.format(
                        instruction=instruction, terminal_state=terminal_state
                    )
                self.messages.append({"role": "user", "content": follow_up})
                self._publish(context, recorder)

        except Exception as exc:
            exit_reason = f"error: {type(exc).__name__}"
            raise
        finally:
            # Also runs on the timeout/kill path, so a trial that the harness
            # cuts short still leaves a complete trajectory and usage numbers.
            recorder.set_notes(f"exit_reason={exit_reason}")
            self._publish(context, recorder, exit_reason=exit_reason)

    # ---- internals -------------------------------------------------------

    async def _compact(self, instruction, system_prompt, recorder) -> None:
        """Summarize the run so far and restart the conversation from it.

        The terminal is the real state and survives untouched, so the summary
        only has to carry what the screen cannot show: what was tried, what was
        learned, and what already failed. If the handoff call itself fails the
        run continues uncompacted — a degraded context beats ending the trial,
        which would score zero.
        """
        assert self._session is not None
        self.n_compactions += 1

        handoff = self.messages + [
            {"role": "user", "content": HANDOFF_PROMPT.format(instruction=instruction)}
        ]
        try:
            summary = await self._client.complete(handoff)
        except Exception as exc:
            self.logger.warning("compaction failed, continuing uncompacted: %s", exc)
            return

        terminal_state = await self._session.capture_pane()
        self.messages = build_resume_messages(
            system_prompt=system_prompt,
            instruction=instruction,
            summary=summary.content,
            terminal_state=terminal_state,
        )
        recorder.record_agent_step(
            message=summary.content,
            prompt_tokens=summary.usage.prompt_tokens,
            completion_tokens=summary.usage.completion_tokens,
            cached_tokens=summary.usage.cached_tokens,
            cost_usd=summary.usage.cost_usd,
            extra={"event": "compaction", "compaction_index": self.n_compactions},
        )
        self.logger.info("compacted context (#%d)", self.n_compactions)

    @staticmethod
    def _close_commands(response: str) -> str:
        """Close a <commands> block the model closed with the wrong tag.

        Observed in every v2 run: the block opens correctly, the keystrokes are
        valid, and then it closes with </jobs> or </tasks>, or with nothing at
        all. Tag drift over a long generation is not something prompting
        reliably fixes, and the commands themselves are fine.
        """
        if "<commands>" not in response or "</commands>" in response:
            return response
        idx = response.rfind("</keystrokes>")
        if idx == -1:
            return response
        cut = idx + len("</keystrokes>")
        # Drop the mis-named closer, if one is there, before inserting the real one.
        tail = re.sub(r"^\s*</[A-Za-z_][\w.-]*>", "", response[cut:], count=1)
        return response[:cut] + "\n</commands>" + tail

    @staticmethod
    def _normalize(response: str) -> str:
        """Restore the <response> envelope when the model omits it.

        Dropping an outer wrapper while producing every section correctly is
        one of the most common things models do with nested-tag formats, and it
        was the dominant failure in v2's first run: every turn carried valid
        analysis, plan and commands, and every turn was thrown away for want of
        eleven characters. Rejecting those replies teaches the model nothing —
        it just reproduces them — so repair here rather than spend turns
        scolding.
        """
        response = CruxAgent._close_commands(response)
        if "<response>" in response or "<commands>" not in response:
            return response
        start = response.find("<analysis>")
        if start == -1:
            start = response.find("<commands>")
        end = response.rfind("</task_complete>")
        end = end + len("</task_complete>") if end != -1 else len(response)
        end = max(end, response.rfind("</commands>") + len("</commands>"))
        return f"<response>{response[start:end]}</response>"

    def _parse(self, response: str):
        """Parse a reply, repairing a missing envelope or a truncated tail.

        A response cut off by the output limit still usually carries valid
        commands; discarding it would waste the turn and, worse, teach nothing
        — the model would likely produce the same too-long reply again.
        """
        response = self._normalize(response)
        result = self._parser.parse_response(response)
        if not result.error:
            return result
        # salvage_truncated_response hands back a cleaned string, not a result,
        # so it has to go through the parser again.
        cleaned, _multiple = self._parser.salvage_truncated_response(response)
        if cleaned:
            salvaged = self._parser.parse_response(cleaned)
            if not salvaged.error and salvaged.commands:
                return salvaged
        return result

    async def _send(self, commands) -> tuple[bool, float]:
        """Send one batch of keystrokes. Returns (timed_out, elapsed_sec)."""
        assert self._session is not None
        started = time.time()
        for command in commands:
            duration = min(float(command.duration), self.config.max_command_timeout_sec)
            try:
                await self._session.send_keys(
                    command.keystrokes,
                    block=False,
                    min_timeout_sec=duration,
                    max_timeout_sec=self.config.max_command_timeout_sec,
                )
            except TimeoutError:
                return True, time.time() - started
        return False, time.time() - started

    def _record(self, recorder, reply, parsed, observation=None, commands=None) -> None:
        recorder.record_agent_step(
            message=reply.content,
            reasoning_content=reply.reasoning_content,
            command="\n".join(c.keystrokes for c in commands) if commands else None,
            output=observation,
            exit_code=None,
            prompt_tokens=reply.usage.prompt_tokens,
            completion_tokens=reply.usage.completion_tokens,
            cached_tokens=reply.usage.cached_tokens,
            cost_usd=reply.usage.cost_usd,
            extra={
                "analysis": parsed.analysis or None,
                "plan": parsed.plan or None,
                "n_commands": len(parsed.commands),
                "parse_error": parsed.error or None,
            },
        )

    def _publish(
        self,
        context: AgentContext,
        recorder: TrajectoryRecorder,
        exit_reason: str | None = None,
    ) -> None:
        """Mirror running totals into AgentContext.

        Updated every step rather than once at the end: if the harness kills
        the trial, whatever was reported last is what gets recorded.
        """
        context.n_input_tokens = recorder.total_prompt_tokens
        context.n_cache_tokens = recorder.total_cached_tokens
        context.n_output_tokens = recorder.total_completion_tokens
        context.cost_usd = recorder.total_cost_usd
        context.metadata = {
            "n_steps": self.n_steps,
            "n_compactions": self.n_compactions,
            "elapsed_sec": round(time.time() - self._start_time, 1),
            "agent_version": self.version(),
            "apply_patch_ready": self._apply_patch_ready,
            # The full config travels with the result. A score is not
            # interpretable without knowing exactly which knobs produced it,
            # and that is the whole basis of the ablation set.
            "config": self.config.model_dump(),
            **({"exit_reason": exit_reason} if exit_reason else {}),
        }

"""ATIF trajectory recording.

Harbor's leaderboard CI rejects any submission whose passing trials lack an
ATIF trajectory, and the `/judge` pass audits those trajectories for reward
hacking. So this is not optional instrumentation — it is the artifact the
submission is validated against (docs/RESEARCH.md §4).

Two things follow from that, and they drive the design here:

* The file is rewritten after **every** step, not once at the end. A trial that
  hits the harness timeout is killed outright; if the trajectory were written
  only on the happy path, exactly the runs that need explaining would have
  nothing to show.
* Each write goes to a temp file and is then renamed. A kill in the middle of
  a write would otherwise leave truncated JSON, which fails validation just as
  hard as no file at all.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.trajectory_utils import format_trajectory_json

TRAJECTORY_FILENAME = "trajectory.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TrajectoryRecorder:
    """Accumulates ATIF steps and keeps trajectory.json on disk current."""

    def __init__(
        self,
        logs_dir: Path,
        agent_name: str,
        agent_version: str,
        model_name: str | None,
        session_id: str | None = None,
    ):
        self._path = Path(logs_dir) / TRAJECTORY_FILENAME
        self._agent = Agent(
            name=agent_name, version=agent_version, model_name=model_name
        )
        self._model_name = model_name
        self._session_id = session_id
        self._steps: list[Step] = []
        self._notes: str | None = None

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.total_cost_usd = 0.0

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    def _next_id(self) -> int:
        return len(self._steps) + 1

    def record_system(self, message: str) -> None:
        self._steps.append(
            Step(
                step_id=self._next_id(),
                timestamp=_now(),
                source="system",
                message=message,
            )
        )
        self.flush()

    def record_user(self, message: str) -> None:
        self._steps.append(
            Step(
                step_id=self._next_id(),
                timestamp=_now(),
                source="user",
                message=message,
            )
        )
        self.flush()

    def record_agent_step(
        self,
        *,
        message: str,
        reasoning_content: str | None = None,
        command: str | None = None,
        output: str | None = None,
        exit_code: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
        extra: dict | None = None,
    ) -> None:
        """Record one model turn, plus the command it ran and what came back."""
        self.total_prompt_tokens += prompt_tokens or 0
        self.total_completion_tokens += completion_tokens or 0
        self.total_cached_tokens += cached_tokens or 0
        self.total_cost_usd += cost_usd or 0.0

        tool_calls = None
        observation = None
        if command is not None:
            call_id = f"call_{self._next_id()}"
            tool_calls = [
                ToolCall(
                    tool_call_id=call_id,
                    function_name="bash",
                    arguments={"command": command},
                )
            ]
            observation = Observation(
                results=[
                    ObservationResult(
                        source_call_id=call_id,
                        content=output,
                        extra={"exit_code": exit_code},
                    )
                ]
            )
        elif output is not None:
            # A turn we rejected before executing anything — the parse failure
            # is still feedback the model saw, so it belongs in the record.
            observation = Observation(results=[ObservationResult(content=output)])

        self._steps.append(
            Step(
                step_id=self._next_id(),
                timestamp=_now(),
                source="agent",
                model_name=self._model_name,
                message=message,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                observation=observation,
                metrics=Metrics(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cost_usd=cost_usd,
                ),
                # The model's own analysis and plan for the turn. The /judge
                # pass reads trajectories to decide whether a pass was earned;
                # stated intent is what makes a batch of commands legible.
                extra={k: v for k, v in (extra or {}).items() if v is not None}
                or None,
            )
        )
        self.flush()

    def set_notes(self, notes: str) -> None:
        self._notes = notes
        self.flush()

    def build(self) -> Trajectory:
        return Trajectory(
            session_id=self._session_id,
            agent=self._agent,
            steps=self._steps,
            notes=self._notes,
            final_metrics=FinalMetrics(
                total_prompt_tokens=self.total_prompt_tokens,
                total_completion_tokens=self.total_completion_tokens,
                total_cached_tokens=self.total_cached_tokens,
                total_cost_usd=self.total_cost_usd,
                total_steps=len(self._steps),
            ),
        )

    def flush(self) -> None:
        """Write the trajectory so far, atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(format_trajectory_json(self.build().to_json_dict()))
        tmp.replace(self._path)

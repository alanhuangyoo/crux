"""Configuration and ablation variants.

Crux is a modified mini-SWE-agent, so a "config" here is the mini-swe-agent
config dict that gets serialized to YAML and written into the task container.
Only the fields Crux actually changes are set; everything else stays at
upstream's defaults, which is the point — the base agent scores 76.2% on the
public leaderboard and is not what needs fixing.

Variants exist so an ablation isolates one change at a time. A score is not
interpretable without knowing which configuration produced it.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from crux.prompts import SYSTEM_TEMPLATE, build_instance_template


class CruxConfig(BaseModel):
    """The knobs Crux adds on top of mini-SWE-agent."""

    variant: str = Field(default="default")

    grading_section: bool = Field(
        default=True,
        description=(
            "Tell the model that scoring is per-task all-or-nothing and have "
            "it enumerate and prove the requirements. In our own full run 30 "
            "of 70 tasks reached 60% or more of their checks while only 3 "
            "scored, which is what this is aimed at."
        ),
    )
    apply_patch: bool = Field(
        default=True,
        description=(
            "Install apply_patch and prefer it over sed for edits. Measured at "
            "97.5% success over 601 calls, against sed's habit of silently "
            "matching the wrong line."
        ),
    )
    survival: bool = Field(
        default=True,
        description=(
            "Warn about the container memory cap. Exceeding it is a SIGKILL, "
            "not a catchable error, and the trial scores zero -- 10% of trials "
            "in one evaluation."
        ),
    )
    file_tools: bool = Field(
        default=True,
        description=(
            "Describe the crux file tools (read/grep/edit/write) and prefer "
            "them over cat/sed. Off by default in lean variants because the "
            "model does not take them up: over 206 tool calls in one "
            "evaluation, read/grep/edit/write together accounted for under 2%, "
            "while the prompt describing them was resent every turn."
        ),
    )
    toolkit: bool = Field(
        default=True,
        description=(
            "Install the crux file tools and prefer them over cat/grep/sed. "
            "Line-numbered paginated reads, bounded output, and edits whose "
            "anchor must be unique — the last of which turns sed's silent "
            "wrong-line edit into an error message."
        ),
    )
    step_limit: int = Field(
        default=0,
        description="Upstream default of 0 means unbounded; the harness times out.",
    )
    cost_limit: float = Field(
        default=0.0, description="0 disables mini-swe-agent's own cost cap."
    )
    num_retries: int = Field(
        default=8,
        description=(
            "litellm retries per model call, inside the container where the "
            "agent actually runs. A 429 there ends the trial outright -- which "
            "scores zero and cannot be excluded -- and under the concurrency a "
            "benchmark run needs, rate limits are routine rather than "
            "exceptional."
        ),
    )
    reasoning_effort: Optional[Literal["low", "medium", "xhigh"]] = Field(
        default=None,
        description=(
            "Qwen3.5's chat template accepts low / medium / xhigh, defaulting "
            "to xhigh. Left unset here so the model's own default applies and "
            "hosted models that know nothing about the parameter are "
            "unaffected. Worth setting because running out of time is the "
            "single largest cause of failure and reasoning tokens are most of "
            "a turn: measured on one prompt, low spent 134 thinking tokens "
            "against xhigh's 198. Note that turning thinking off entirely is "
            "not the cheap end of the same axis -- it moves the reasoning into "
            "the visible answer, which came out longer, not shorter."
        ),
    )
    request_timeout: int = Field(
        default=600,
        description="Per-call timeout. Long enough that a slow provider is waited out.",
    )
    max_tokens: int = Field(
        default=16384,
        description=(
            "Output budget per call. Reasoning models spend it on the think "
            "block before writing anything, so a small budget returns "
            "finish_reason=length with content=None -- a turn that did nothing "
            "and reads to the agent as a format error. 8192 was not enough: on "
            "the ten-task slice it produced two RepeatedFormatError trials that "
            "never emitted a single command, and burned 4 more turns inside a "
            "trial that then timed out at 66.7% complete. A truncated turn "
            "costs a full round trip and returns nothing, so the budget is "
            "cheaper raised than spent."
        ),
    )


VARIANTS: dict[str, dict] = {
    "default": {},
    # The base agent as upstream ships it. This is the number Crux has to beat
    # to justify existing at all.
    "stock": {
        "grading_section": False,
        "apply_patch": False,
        "toolkit": False,
        "file_tools": False,
        "survival": False,
    },
    # Isolates each addition against default.
    "no_grading": {"grading_section": False},
    "no_apply_patch": {"apply_patch": False},
    "no_toolkit": {"toolkit": False, "file_tools": False},
    # Drops the file tools the model never picked up while keeping the
    # checklist and `crux submit`, which is what turns "I think I am done"
    # into a re-run of every bound check.
    "lean": {"file_tools": False},
}


def build_config(**kwargs) -> CruxConfig:
    """Resolve a variant name plus explicit overrides into a config."""
    variant = kwargs.pop("variant", "default")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Known: {sorted(VARIANTS)}")
    # Harbor passes --ak values as strings; pydantic coerces them.
    return CruxConfig(variant=variant, **{**VARIANTS[variant], **kwargs})


def to_mini_config(cfg: CruxConfig) -> dict:
    """Render a CruxConfig as a mini-swe-agent config dict."""
    model_kwargs = {
        "drop_params": True,
        "num_retries": cfg.num_retries,
        "timeout": cfg.request_timeout,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.reasoning_effort is not None:
        # chat_template_kwargs is not an OpenAI parameter, so it has to ride in
        # extra_body -- litellm passes that through untouched, where drop_params
        # would otherwise discard an unrecognised top-level key.
        model_kwargs["extra_body"] = {
            "chat_template_kwargs": {"reasoning_effort": cfg.reasoning_effort}
        }
    return {
        # mini-swe-agent's own litellm settings. The agent runs inside the task
        # container, so retry behaviour has to be configured here — anything
        # host-side never sees its calls.
        "model": {
            "model_kwargs": model_kwargs,
        },
        "agent": {
            "system_template": SYSTEM_TEMPLATE,
            "instance_template": build_instance_template(
                grading=cfg.grading_section,
                apply_patch=cfg.apply_patch,
                toolkit=cfg.toolkit,
                file_tools=cfg.file_tools,
                survival=cfg.survival,
            ),
            "step_limit": cfg.step_limit,
            "cost_limit": cfg.cost_limit,
        }
    }

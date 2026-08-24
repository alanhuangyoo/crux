"""Agent configuration and named variants.

Every knob that could plausibly change a score lives here, in one typed object
that is stamped into each trajectory. That is what makes an ablation
answerable after the fact: a run's score is meaningless unless you can say
exactly which configuration produced it, and reconstructing that from a shell
history six runs later does not work.

Variants are named so a sweep reads as `-a crux --ak variant=no_compaction`
rather than five separate `--ak` flags that are easy to get subtly wrong.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CruxConfig(BaseModel):
    """Everything that shapes a run. Serialized into the ATIF trajectory."""

    variant: str = Field(
        default="default", description="Name of the configuration variant."
    )

    # --- loop bounds ---------------------------------------------------
    step_limit: int = Field(
        default=60,
        description=(
            "Model turns before giving up. With batched commands a turn does "
            "far more than v1's single command, so this is lower than v1's 80 "
            "while allowing strictly more work."
        ),
    )
    wall_time_limit_sec: int = Field(
        default=0, description="Wall-clock cap; 0 disables. The harness has its own."
    )
    max_consecutive_format_errors: int = Field(
        default=3, description="Unparseable replies in a row before giving up."
    )

    # --- terminal ------------------------------------------------------
    pane_width: int = Field(default=160, description="tmux pane width.")
    pane_height: int = Field(default=40, description="tmux pane height.")
    max_command_timeout_sec: float = Field(
        default=60.0,
        description=(
            "Cap on a single batch's wait. Bounded so one hung command cannot "
            "eat the whole budget; the model can always wait again."
        ),
    )

    # --- context management --------------------------------------------
    enable_compaction: bool = Field(
        default=True,
        description=(
            "Summarize and restart the conversation as the context fills. "
            "Without it, long tasks fail on context length rather than on "
            "difficulty, which caps the achievable score outright."
        ),
    )
    compaction_threshold_tokens: int = Field(
        default=12000,
        description="Compact once fewer than this many input tokens remain free.",
    )
    max_input_tokens: int = Field(
        default=128000,
        description=(
            "Assumed context window. Only a fallback: the real value is read "
            "from litellm's model map when it knows the model."
        ),
    )

    # --- tools ---------------------------------------------------------
    enable_apply_patch: bool = Field(
        default=True,
        description=(
            "Install the apply_patch helper into the task container. "
            "Structured patches fail far less often than heredoc-and-sed "
            "editing, which is where a lot of otherwise-solved tasks are lost."
        ),
    )

    # --- model ---------------------------------------------------------
    temperature: float | None = Field(default=None)
    max_tokens: int | None = Field(default=None)
    request_timeout_sec: int = Field(default=300)
    max_retries: int = Field(
        default=6,
        description=(
            "A trial lost to a 429 counts as reward 0 and cannot be excluded "
            "from a submission, so waiting out a rate limit always beats "
            "letting it kill the run."
        ),
    )


# Ablation variants. Each isolates one mechanism against `default` so a score
# delta has a single cause; changing two knobs at once makes the result
# uninterpretable.
VARIANTS: dict[str, dict] = {
    "default": {},
    # Does compaction actually buy anything, or do tasks finish inside one
    # context anyway?
    "no_compaction": {"enable_compaction": False},
    # Is structured patching worth installing a helper, versus leaving the
    # model to edit with heredocs and sed?
    "no_apply_patch": {"enable_apply_patch": False},
    # Are failures caused by the turn budget, or by the model being stuck?
    # If doubling the budget does not move the score, the budget was not it.
    "steps_120": {"step_limit": 120},
    # A taller pane shows more state per turn but costs tokens on every turn.
    "pane_60": {"pane_height": 60},
    # Ablates both mechanisms at once: the v2 architecture with none of the
    # additions, i.e. the closest thing to a plain Terminus-style baseline.
    "bare": {"enable_compaction": False, "enable_apply_patch": False},
}


def build_config(**kwargs) -> CruxConfig:
    """Resolve a config from a variant name plus explicit overrides.

    Explicit kwargs win over the variant's values, so a sweep can pin one
    extra knob without having to declare a whole new variant.
    """
    variant = kwargs.pop("variant", "default")
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Known: {sorted(VARIANTS)}"
        )
    # Harbor passes --ak values as strings; let pydantic coerce them.
    return CruxConfig(variant=variant, **{**VARIANTS[variant], **kwargs})

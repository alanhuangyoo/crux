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
        default=150,
        description=(
            "Model turns before giving up. Set high on purpose: every task in "
            "the first clean run hit the old limit of 60 while still making "
            "real progress, and Harbor records an agent timeout and grades the "
            "trial anyway rather than erroring it. So the per-task timeout is "
            "the real governor, and a low limit only throws away work that "
            "would have been scored."
        ),
    )
    wall_time_limit_sec: int = Field(
        default=0, description="Wall-clock cap; 0 disables. The harness has its own."
    )
    max_consecutive_format_errors: int = Field(
        default=6,
        description=(
            "Unparseable replies in a row before giving up. Lenient on "
            "purpose: abandoning a trial scores zero, while another nudge only "
            "costs a turn out of a bounded budget."
        ),
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
    context_window_cap: int = Field(
        default=200000,
        description=(
            "Upper bound on the window used for the compaction decision, "
            "whatever the model map reports. litellm puts deepseek-v4-flash at "
            "1M tokens, so a threshold measured as free-space-remaining never "
            "fired and one trial died on ContextWindowExceededError instead. "
            "Capping the assumed window makes compaction trigger on a sane "
            "conversation length rather than on a number we cannot trust."
        ),
    )

    # --- completion gating ----------------------------------------------
    verify_before_complete: bool = Field(
        default=True,
        description=(
            "Challenge a completion claim before accepting it. In the first "
            "full run the agent declared completion 28 times and passed the "
            "verifier 3 times; a claim costs the model nothing to make, so it "
            "has to be paid for with evidence."
        ),
    )
    max_verify_rounds: int = Field(
        default=2,
        description=(
            "How many times a completion claim can be challenged. Bounded so "
            "a model that keeps re-asserting cannot spend the whole budget "
            "arguing with itself."
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
    # The single biggest lever found so far: 25 of 28 completion claims in the
    # first full run were false. This measures what challenging them is worth.
    "no_verify_gate": {"verify_before_complete": False},
    # Is structured patching worth installing a helper, versus leaving the
    # model to edit with heredocs and sed?
    "no_apply_patch": {"enable_apply_patch": False},
    # The old default, kept as the control for raising it. If 60 scores the
    # same as 150, the budget was never the constraint and the extra spend is
    # waste; if it scores worse, the raise is paying for itself.
    "steps_60": {"step_limit": 60},
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

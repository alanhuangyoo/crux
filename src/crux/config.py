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
    step_limit: int = Field(
        default=0,
        description="Upstream default of 0 means unbounded; the harness times out.",
    )
    cost_limit: float = Field(
        default=0.0, description="0 disables mini-swe-agent's own cost cap."
    )


VARIANTS: dict[str, dict] = {
    "default": {},
    # The base agent as upstream ships it. This is the number Crux has to beat
    # to justify existing at all.
    "stock": {"grading_section": False, "apply_patch": False},
    # Isolates each addition against default.
    "no_grading": {"grading_section": False},
    "no_apply_patch": {"apply_patch": False},
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
    return {
        "agent": {
            "system_template": SYSTEM_TEMPLATE,
            "instance_template": build_instance_template(
                grading=cfg.grading_section, apply_patch=cfg.apply_patch
            ),
            "step_limit": cfg.step_limit,
            "cost_limit": cfg.cost_limit,
        }
    }

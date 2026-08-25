"""Variants are the unit of the ablation set, so their contract is pinned.

The one that matters most is `stock`: it renders upstream's own prompt, and if
Crux cannot beat that number it has no reason to exist.
"""

import pytest
import yaml

from crux.config import VARIANTS, CruxConfig, build_config, to_mini_config
from crux.prompts import APPLY_PATCH_SECTION, GRADING_SECTION


def rendered(variant="default"):
    return to_mini_config(build_config(variant=variant))["agent"]["instance_template"]


def test_default_enables_both_changes():
    cfg = build_config()
    assert cfg.grading_section is True and cfg.apply_patch is True
    body = rendered()
    assert "all or nothing" in body
    assert "apply_patch <<'PATCH'" in body


def test_stock_variant_is_upstream_prompt():
    body = rendered("stock")
    assert GRADING_SECTION not in body
    assert APPLY_PATCH_SECTION not in body
    # Upstream's format contract must survive: its parser depends on both.
    assert "bash tool call" in body
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in body


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_variant_renders_valid_config(name):
    cfg = to_mini_config(build_config(variant=name))
    # The dict is serialized to YAML and written into the container; a template
    # that breaks the dump would fail at trial time, not here.
    assert yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False)) == cfg
    agent = cfg["agent"]
    assert agent["system_template"] and agent["instance_template"]


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_format_contract_survives_every_variant(name):
    body = rendered(name)
    assert "bash tool call" in body
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in body
    assert "{{task}}" in body


def test_variants_differ_from_default():
    """A variant identical to default would spend a full run proving nothing."""
    default = build_config()
    for name in VARIANTS:
        if name == "default":
            continue
        cfg = build_config(variant=name)
        assert any(
            getattr(cfg, f) != getattr(default, f)
            for f in CruxConfig.model_fields
            if f != "variant"
        ), f"variant {name!r} is identical to default"


def test_ablations_isolate_one_section_each():
    assert GRADING_SECTION not in rendered("no_grading")
    assert APPLY_PATCH_SECTION in rendered("no_grading")
    assert APPLY_PATCH_SECTION not in rendered("no_apply_patch")
    assert GRADING_SECTION in rendered("no_apply_patch")


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_config(variant="typo")


def test_harbor_passes_strings():
    cfg = build_config(step_limit="90", apply_patch="false")
    assert cfg.step_limit == 90 and cfg.apply_patch is False


def test_toolkit_section_present_by_default():
    body = rendered()
    assert "crux read" in body and "crux edit" in body
    # sed stays available but is demoted rather than removed: some edits are
    # genuinely easier as a regex over many lines.
    assert "last resort" in body


def test_stock_has_no_crux_tools():
    """`stock` must be upstream's prompt, or it is not a baseline."""
    body = rendered("stock")
    assert "crux read" not in body
    assert "crux edit" not in body
    assert "apply_patch" not in body


def test_no_toolkit_variant_keeps_the_other_changes():
    body = rendered("no_toolkit")
    assert "crux read" not in body
    assert "all or nothing" in body
    assert "apply_patch <<'PATCH'" in body


def test_stock_is_exactly_upstream_with_nothing_added():
    """The baseline must carry none of Crux's sections.

    Each of these leaked into `stock` at least once while being added. A
    baseline contaminated by the thing it is measuring is worse than no
    baseline, because it looks like a result.
    """
    body = rendered("stock")
    for marker in (
        "all or nothing",      # grading
        "Staying alive",       # survival
        "crux read",           # toolkit
        "crux todo",           # todo
        "apply_patch",         # patch tool
        "last resort",         # the sed caveat
    ):
        assert marker not in body, f"{marker!r} leaked into the baseline prompt"


def test_no_placeholder_markers_survive_rendering():
    """An unreplaced __MARKER__ would ship to the model as literal text."""
    for variant in VARIANTS:
        body = rendered(variant)
        assert "__" not in body.replace("__init__", ""), f"{variant} has a raw marker"


def test_model_kwargs_carry_retries_into_the_container():
    """The agent runs inside the task container with its own litellm.

    A 429 there ends the trial outright — reward 0, not excludable — and host
    -side retry logic never sees those calls. One run lost
    torch-pipeline-parallelism to exactly this.
    """
    cfg = to_mini_config(build_config())
    kwargs = cfg["model"]["model_kwargs"]
    assert kwargs["num_retries"] >= 5
    assert kwargs["timeout"] >= 300
    # drop_params is upstream's; losing it breaks models that reject unknown
    # sampling parameters.
    assert kwargs["drop_params"] is True


def test_stock_keeps_the_retry_settings():
    """`stock` ablates the prompt, not the harness.

    Leaving the baseline to die on rate limits would make it look worse for a
    reason that has nothing to do with what is being compared.
    """
    assert to_mini_config(build_config(variant="stock"))["model"]["model_kwargs"][
        "num_retries"
    ] >= 5

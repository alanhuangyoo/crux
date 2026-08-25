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
    assert "```mswea_bash_command" in body
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
    assert "```mswea_bash_command" in body
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

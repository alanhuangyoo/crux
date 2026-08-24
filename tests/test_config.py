"""Variants are the unit of the ablation set, so their contract is pinned here.

A typo in a variant name silently running `default` would produce a sweep whose
rows all look the same and mean nothing.
"""

import pytest

from crux.config import VARIANTS, CruxConfig, build_config


def test_default_variant():
    cfg = build_config()
    assert cfg.variant == "default"
    assert cfg.enable_compaction is True
    assert cfg.enable_apply_patch is True


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_variant_builds(name):
    cfg = build_config(variant=name)
    assert cfg.variant == name


def test_variants_differ_from_default():
    """A variant that changes nothing would waste a full run to prove nothing."""
    default = build_config()
    for name in VARIANTS:
        if name == "default":
            continue
        cfg = build_config(variant=name)
        differing = {
            f
            for f in CruxConfig.model_fields
            if f != "variant" and getattr(cfg, f) != getattr(default, f)
        }
        assert differing, f"variant {name!r} is identical to default"


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_config(variant="typo")


def test_explicit_kwargs_win_over_variant():
    cfg = build_config(variant="steps_60", step_limit=200)
    assert cfg.step_limit == 200


def test_harbor_passes_strings():
    """Harbor forwards --ak values as strings; they must coerce."""
    cfg = build_config(step_limit="90", enable_compaction="false")
    assert cfg.step_limit == 90
    assert cfg.enable_compaction is False


def test_salvage_returns_a_string_not_a_result():
    """Pins the upstream contract that broke the first v2 run.

    salvage_truncated_response returns (cleaned_text, has_multiple_blocks);
    treating it as a ParseResult raised AttributeError on every trial.
    """
    from harbor.agents.terminus_2.terminus_xml_plain_parser import (
        TerminusXMLPlainParser,
    )

    parser = TerminusXMLPlainParser()
    truncated = (
        "<response><analysis>a</analysis><plan>b</plan><commands>"
        '<keystrokes duration="0.1">ls\n</keystrokes></commands></response>'
        " trailing junk that never closed"
    )
    result = parser.salvage_truncated_response(truncated)
    assert isinstance(result, tuple) and len(result) == 2
    cleaned, _ = result
    assert cleaned is None or isinstance(cleaned, str)

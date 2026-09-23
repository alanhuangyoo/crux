"""The launchers and the preflight that checks them agree with each other.

preflight.sh exists because arms were lost to settings that looked right and
were not. Twice the hole was preflight itself answering from a stale file --
`/tmp/confirm.sh` from another round, and a model list left by a previous run
when the endpoint was down.
"""

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
LAUNCHERS = ("one.sh", "ab.sh", "budget-probe.sh")


def code_lines(name):
    return [line for line in (SCRIPTS / name).read_text().splitlines() if not line.lstrip().startswith("#")]


def test_every_launcher_gives_the_same_time_budget():
    # A number is capped per task at harbor's limit less a minute (pi_agent),
    # so 14400 is four hours only where harbor allows four hours.
    found = set()
    for name in LAUNCHERS:
        found.update(re.findall(r"PI_TIME_BUDGET_SEC=([0-9a-z]+)", (SCRIPTS / name).read_text()))
    assert found == {"14400"}


def test_preflight_reads_only_the_launcher_it_was_given():
    assert not [line for line in code_lines("preflight.sh") if "confirm.sh" in line]


def test_preflight_resolves_the_launcher_before_the_first_check_uses_it():
    lines = code_lines("preflight.sh")
    resolved = next(i for i, line in enumerate(lines) if line.startswith('LAUNCHER="${1:-}"'))
    first_use = next(i for i, line in enumerate(lines) if '"$LAUNCHER"' in line)
    assert resolved < first_use


def test_preflight_does_not_answer_from_a_previous_runs_model_list():
    lines = code_lines("preflight.sh")
    removed = next(i for i, line in enumerate(lines) if line.strip() == "rm -f /tmp/pf-models.json")
    fetched = next(i for i, line in enumerate(lines) if "-o /tmp/pf-models.json" in line)
    assert removed < fetched

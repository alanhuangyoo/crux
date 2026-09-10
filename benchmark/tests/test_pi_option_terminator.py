"""The terminator has to be its own argument, not a suffix on the last one.

`pytorch-model-recovery` states its task as a markdown list, so the instruction
harbor appends begins with "-". pi's parser rejects any single-hyphen argument,
so every trial on that task died before taking an action -- in every arm, while
six other scaffolds solved it.

The first fix appended `"-- "` to a flag string that does not end in a space,
which produced `--append-system-prompt /tmp/crux-sections.md--`: the flag took a
path that does not exist, no terminator was ever parsed, and three more trials
died exactly as before. These tests are about that second failure, not the
first.
"""

import shlex

def terminator_argv(flags: str) -> list[str]:
    """What the shell sees, given what `build_cli_flags` returns."""
    return shlex.split(f"pi {flags} 'instruction'")


def test_terminator_is_a_separate_argument():
    argv = terminator_argv("--print --append-system-prompt /tmp/crux-sections.md -- ")
    assert "--" in argv
    assert "/tmp/crux-sections.md" in argv
    assert not any(a.endswith("--") and a != "--" for a in argv)


def test_the_bug_this_replaces_is_detectable():
    # The exact string the first fix produced, kept so the assertion above is
    # known to be able to fail.
    argv = terminator_argv("--print --append-system-prompt /tmp/crux-sections.md-- ")
    assert "--" not in argv
    assert "/tmp/crux-sections.md--" in argv


def test_join_is_stable_for_empty_and_blank_flags():
    def build(flags: str) -> str:
        return f"{flags.rstrip()} -- " if flags.strip() else "-- "

    assert build("") == "-- "
    assert build("   ") == "-- "
    assert build("--print") == "--print -- "
    assert build("--print ") == "--print -- "
    assert shlex.split(f"pi {build('--a /x.md')} '- dashy'")[-2:] == ["--", "- dashy"]

"""pi, with the prompt sections this project measured.

Two things pushed the work here rather than onto crux's own Terminus base.

pi already has what crux does not: an interactive terminal UI, session resume,
and a clean split between agent core, model layer and front-end. Building that
onto a benchmark scaffold means fighting its design -- Terminus's `run()` starts
a fresh Chat every call, so multi-turn would need the chat construction
monkeypatched.

And pi exposes `--append-system-prompt`, which makes a prompt change a one
variable experiment: the same base, the same model, the same tasks, with and
without the sections. Comparing crux against pi compares two codebases at once
and cannot attribute anything.

harbor's own Pi agent declares only a `thinking` flag, so the flag is added the
same way crux extends Terminus -- by subclassing at the supported extension
point, not by patching harbor.
"""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path

from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment

from crux.prompts import build_sections

# Where the sections land inside the environment. pi reads the file at startup,
# so it has to exist before the agent command runs.
_REMOTE_PROMPT_PATH = "/tmp/crux-sections.md"

# The submit section directs the model through `crux submit`, a tool that only
# exists inside crux's own image, so it is off by default here.
_DEFAULT_SECTIONS = ("scoring", "harness")


class CruxPiAgent(Pi):
    """pi with crux's measured prompt sections appended.

    `sections` selects them, comma separated, and an empty string runs stock pi
    -- which is the control arm, and has to be reachable through the same code
    path as the treatment so that a difference between the two is the sections
    and not the plumbing.
    """

    CLI_FLAGS = Pi.CLI_FLAGS + [
        CliFlag(
            "append_system_prompt",
            cli="--append-system-prompt",
            type="str",
        ),
    ]

    @staticmethod
    def name() -> str:
        return "crux-pi"

    def __init__(self, *args, sections: str | None = None, **kwargs):
        raw = _DEFAULT_SECTIONS if sections is None else tuple(
            s.strip() for s in str(sections).split(",") if s.strip()
        )
        self._sections = raw
        self._section_text = build_sections(raw) if raw else ""
        if self._section_text:
            kwargs.setdefault("append_system_prompt", _REMOTE_PROMPT_PATH)
        super().__init__(*args, **kwargs)

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        if not self._section_text:
            return
        # Written as a file rather than inlined into the command: the sections
        # run to thousands of characters and contain quotes and backticks, and
        # a shell-escaped blob that long is where quoting bugs live.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(self._section_text)
            local = Path(f.name)
        try:
            await environment.upload_file(local, _REMOTE_PROMPT_PATH)
        finally:
            local.unlink(missing_ok=True)

    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        # pi reads the argument as text or as a path; quote it so a path with a
        # space does not silently become two arguments.
        if self._section_text and _REMOTE_PROMPT_PATH in flags:
            flags = flags.replace(
                f"--append-system-prompt {_REMOTE_PROMPT_PATH}",
                f"--append-system-prompt {shlex.quote(_REMOTE_PROMPT_PATH)}",
            )
        return flags

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

import logging
import os
import shlex
import tempfile
from pathlib import Path

from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment
from typing_extensions import override

from crux.prompts import build_sections

logger = logging.getLogger(__name__)

# Where the sections land inside the environment. pi reads the file at startup,
# so it has to exist before the agent command runs.
_REMOTE_PROMPT_PATH = "/tmp/crux-sections.md"

# A prebuilt $HOME/.nvm holding node and pi, so a trial does not have to reach
# the network to get its agent. Built once with `crux bake-pi`; see
# `_install_offline` for what it is worth.
_PI_BUNDLE = os.environ.get("CRUX_PI_BUNDLE", "/scratch/crux/pi-nvm.tar.gz")
_REMOTE_BUNDLE = "/tmp/pi-nvm.tar.gz"

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

    def __init__(self, *args, sections: str | None = None,
                 bundle: str | None = None, **kwargs):
        # Which prebuilt install to unpack. Passing it as a kwarg rather than
        # reading the env means a control arm and a treatment arm reach the
        # same code by the same path, and differ only in the bundle -- which is
        # the whole point of running them against each other.
        self._bundle_path = bundle or _PI_BUNDLE
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

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        """Unpack a prebuilt node+pi instead of fetching one per trial.

        Upstream installs the agent from scratch in every container:

            curl raw.githubusercontent.com/nvm-sh/nvm/.../install.sh | bash
            nvm install 22
            npm install -g @earendil-works/pi-coding-agent@latest

        Three network fetches per trial, under `set -euo pipefail`, at whatever
        concurrency the run uses. Any one of them failing raises
        `NonZeroAgentExitCodeError` before the agent has run, and the trial is
        scored zero.

        Measured on Terminal-Bench 2.1: **24 of pi's 89 trials died here**, all
        of them in `curl ... nvm/install.sh`, none with a single tool call
        recorded. That is 27% of the benchmark decided by a download, and it is
        why pi's headline 53.9% was not a number about pi.

        The bundle is the same install, done once. If it is missing the
        network path still runs, because a missing file should cost a slower
        setup and not the run.
        """
        bundle = Path(getattr(self, "_bundle_path", _PI_BUNDLE))
        if not bundle.is_file():
            logger.warning(
                "pi bundle %s not found; falling back to the per-trial network "
                "install that loses ~27%% of trials", bundle,
            )
            await super().install(environment)
            return

        await environment.upload_file(source_path=bundle, target_path=_REMOTE_BUNDLE)
        # `exec_as_agent`, not `environment.exec`: the latter runs as root, so
        # on an image whose agent user is not root the bundle lands in root's
        # home and the agent -- which harbor starts with `. ~/.nvm/nvm.sh` --
        # never sees it. That is what "pi: command not found" meant on the
        # SWE-Atlas images while the same bundle worked on Terminal-Bench.
        # Upstream's install uses the same helper; matching it is the point.
        result = await self.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                f"tar xzf {_REMOTE_BUNDLE} -C \"$HOME\"; "
                f"rm -f {_REMOTE_BUNDLE}; "
                '. "$HOME/.nvm/nvm.sh"; '
                "pi --version"
            ),
        )
        out = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
        if getattr(result, "return_code", 1) != 0:
            # Say what the bundle did before falling back, so a broken bundle is
            # distinguishable from a missing one.
            logger.warning("pi bundle failed to unpack (%s); falling back", out.strip()[:200])
            await super().install(environment)
            return
        logger.info("pi installed from bundle: %s", out.strip().splitlines()[-1:] or "?")

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

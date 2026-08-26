"""Crux — mini-SWE-agent with a modified prompt and a better file-editing tool.

Crux subclasses Harbor's ``MiniSweAgent`` rather than reimplementing an agent
loop. The base agent sits on the public Terminal-Bench leaderboard at 76.2%;
its loop, parser, trajectory export and format contract are all upstream's and
are left untouched. Writing that machinery again from scratch was tried first
and reached 4.62% — the gap is not prompt tuning, it is the iterations already
baked into the upstream agent.

Two changes, each from evidence in a full 70-task run of our own:

* The prompt now states that scoring is per-task all-or-nothing and makes the
  model enumerate and prove the task's requirements. That run finished with 30
  tasks at 60% or more of their checks and only 3 scored — the model stops when
  the work looks basically done, which is exactly where the points go.
* ``apply_patch`` is installed and preferred over sed. Across 601 calls it
  succeeded 97.5% of the time, and its failures were bad paths and stale
  context rather than silently mangled files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment

from crux import __version__
from crux.config import CruxConfig, build_config, to_mini_config

RESOURCES = Path(__file__).parent / "resources"

# (source, destination, probe, expected substring). The probe matters because
# the prompt already describes these tools: if one were missing, the model
# would spend turns on command-not-found before falling back to sed.
HELPERS = [
    (
        "apply_patch.py",
        "/usr/local/bin/apply_patch",
        "/usr/local/bin/apply_patch '*** Begin Patch\n*** End Patch'",
        # An empty patch is rejected on purpose; reaching that specific
        # complaint proves the interpreter ran the script.
        "no file operations",
    ),
    (
        "crux_tool.py",
        "/usr/local/bin/crux",
        "/usr/local/bin/crux read /nonexistent-probe-path",
        "no such file",
    ),
]


class CruxAgent(MiniSweAgent):
    """mini-SWE-agent, tuned for Terminal-Bench's all-or-nothing scoring."""

    @staticmethod
    @override
    def name() -> str:
        return "crux"

    @override
    def version(self) -> str:
        return __version__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Harbor forwards --ak key=value pairs here. Split ours out before
        # handing the rest to the base agent, which has its own.
        own = {k: kwargs.pop(k) for k in list(kwargs) if k in CruxConfig.model_fields}
        self.crux_config = build_config(**own)
        # `config` is upstream's injection point: the dict is serialized to
        # YAML and written into the container as mini-swe-agent's config.
        kwargs.setdefault("config", to_mini_config(self.crux_config))
        super().__init__(*args, **kwargs)
        self.helpers_ready: dict[str, bool] = {}

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        await self._repair_python_too_old(environment)
        # The grading section tells the agent to finish with `crux submit`
        # whether or not the toolkit section is present, so gating the binary
        # on `toolkit` alone left 26 of 89 tasks in one evaluation running a
        # command that did not exist. Install it whenever any enabled section
        # names it.
        needs_crux = (
            self.crux_config.toolkit
            or self.crux_config.file_tools
            or self.crux_config.grading_section
        )
        wanted = {
            "apply_patch": self.crux_config.apply_patch,
            "crux": needs_crux,
        }
        for source, dest, probe, expected in HELPERS:
            name = Path(dest).name
            if not wanted.get(name):
                continue
            self.helpers_ready[name] = await self._install_helper(
                environment, source, dest, probe, expected
            )

    async def _repair_python_too_old(self, environment: BaseEnvironment) -> None:
        """Reinstall against a newer interpreter only when the image's is too old.

        Upstream installs with a bare `uv tool install mini-swe-agent`, which
        resolves against whatever python3 the image ships. Several ship 3.10,
        where a dependency's `from typing import NotRequired` fails at import:
        the agent never starts and the trial is lost outright.

        The check is the interpreter's own version rather than a probe of the
        installed entry point. An earlier version probed `mini --help` and
        reported healthy on images that then failed at run time; the version
        number cannot be wrong that way.

        It also has to be a check rather than an unconditional reinstall,
        because agent setup has its own 360s budget and downloading an
        interpreter into every container spends enough of it to blow that
        budget on slow images -- trading one class of lost trial for another.
        """
        probe = await self.exec_as_agent(
            environment,
            command=(
                "python3 -c 'import sys; "
                'print("CRUX_PY_OK" if sys.version_info >= (3, 11) '
                "else \"CRUX_PY_OLD\")' 2>/dev/null || echo CRUX_PY_OLD"
            ),
        )
        output = (getattr(probe, "stdout", "") or "") + (
            getattr(probe, "stderr", "") or ""
        )
        if "CRUX_PY_OLD" not in output:
            return

        self.logger.warning(
            "task image ships python < 3.11; reinstalling mini-swe-agent "
            "against a pinned interpreter"
        )
        await self.exec_as_agent(
            environment,
            command='if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
            'else export PATH="$HOME/.local/bin:$PATH"; fi; '
            "uv tool install --force --python 3.12 mini-swe-agent "
            "--with litellm --with orjson --with fastapi",
        )

    async def _install_helper(
        self,
        environment: BaseEnvironment,
        source: str,
        dest: str,
        probe: str,
        expected: str,
    ) -> bool:
        """Copy a helper in and confirm it actually runs in this image."""
        try:
            await environment.upload_file(RESOURCES / source, dest)
            await environment.exec(f"chmod +x {dest}")
            result = await environment.exec(f"{probe} 2>&1 || true")
            output = (result.stdout or "") + (result.stderr or "")
            if expected in output:
                return True
            self.logger.warning(
                "%s unusable in this image: %s", Path(dest).name, output[:200]
            )
        except Exception as exc:
            self.logger.warning("%s install failed: %s", Path(dest).name, exc)
        return False

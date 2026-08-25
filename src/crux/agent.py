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

APPLY_PATCH_SRC = Path(__file__).parent / "resources" / "apply_patch.py"
APPLY_PATCH_DEST = "/usr/local/bin/apply_patch"


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
        self._apply_patch_ready = False

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        if self.crux_config.apply_patch:
            self._apply_patch_ready = await self._install_apply_patch(environment)

    async def _install_apply_patch(self, environment: BaseEnvironment) -> bool:
        """Copy the helper in and confirm it runs.

        Task images vary and a few have no usable python3. The probe matters
        because the prompt already describes the tool: if it were missing, the
        model would spend turns on command-not-found before falling back to sed.
        """
        try:
            await environment.upload_file(APPLY_PATCH_SRC, APPLY_PATCH_DEST)
            await environment.exec(f"chmod +x {APPLY_PATCH_DEST}")
            probe = await environment.exec(
                f"{APPLY_PATCH_DEST} '*** Begin Patch\n*** End Patch' 2>&1 || true"
            )
            output = (probe.stdout or "") + (probe.stderr or "")
            # An empty patch is rejected on purpose; reaching that specific
            # complaint proves the interpreter ran the script.
            if "no file operations" in output:
                return True
            self.logger.warning(
                "apply_patch unusable in this image: %s", output[:200]
            )
        except Exception as exc:
            self.logger.warning("apply_patch install failed: %s", exc)
        return False

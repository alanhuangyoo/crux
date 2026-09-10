"""Run the Terminus agent against this machine instead of a container.

The benchmark scaffold is the better one -- it drives a live tmux session and
can enter an ssh session or a REPL, which the one-command-per-turn base cannot
express at all. But every harbor environment is a container or a cloud sandbox,
and `crux solve` runs on the user's own machine, in their own repo.

So this is the missing piece: the same agent, the same prompt, the same tools,
pointed at a local shell. Terminus touches only seven members of an
environment, so this implements those rather than subclassing BaseEnvironment,
whose constructor wants a TrialPaths, an EnvironmentConfig and a trial id that
do not exist outside a benchmark run.

Nothing here sandboxes anything. Commands run as the invoking user with their
full authority, which is the point -- and the reason `crux solve` gates
dangerous ones before they reach this class.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LocalExecResult:
    """Mirrors harbor's ExecResult for the fields Terminus reads."""

    return_code: int
    stdout: str
    stderr: str


class _LocalTrialPaths:
    """The two paths Terminus writes into, rooted under a session directory.

    Terminus records a tmux pane dump and an asciinema cast per run. In a
    benchmark those land in the trial directory; here they land beside the
    session so `crux solve --replay` can find them later.
    """

    def __init__(self, root: Path):
        self.agent_dir = root
        root.mkdir(parents=True, exist_ok=True)


class LocalEnvironment:
    """A shell on this machine, shaped like the environment Terminus expects."""

    def __init__(
        self,
        cwd: Path | str | None = None,
        session_root: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self.session_id = f"crux-{uuid.uuid4().hex[:8]}"
        self.default_user = None          # whoever invoked us; no su, no docker exec
        self.logger = logger or logging.getLogger("crux.local")
        root = session_root or (Path.home() / ".crux" / "sessions" / self.session_id)
        self.trial_paths = _LocalTrialPaths(root)
        # Where container-absolute tool paths land instead. The agent installs
        # its helpers at /usr/local/bin, which is correct inside a task image
        # and needs root on a laptop. Redirecting into the session and putting
        # that on PATH keeps one code path for both, and keeps a local run from
        # writing outside its own directory.
        self.bin_dir = root / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)

    # --- lifecycle ---------------------------------------------------------

    async def start(self, force_build: bool = False) -> None:
        if shutil.which("tmux") is None:
            raise RuntimeError(
                "tmux is not installed. The Terminus agent drives a live tmux "
                "session; install it with your package manager (brew install "
                "tmux / apt-get install tmux)."
            )

    async def stop(self, delete: bool = False) -> None:
        # The agent kills its own tmux session; nothing else to tear down.
        return None

    # --- the surface Terminus actually uses --------------------------------

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
        user: str | int | None = None,
        **kwargs,
    ) -> LocalExecResult:
        """Run a command in a shell, as the invoking user.

        `user` is accepted and ignored: switching users would need sudo, and a
        CLI that silently escalates is worse than one that cannot.
        """
        merged = {**os.environ, **(env or {})}
        for prefix in ("/usr/local/bin/", "/usr/bin/"):
            if prefix in command:
                command = command.replace(prefix, f"{self.bin_dir}/")
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd or self.cwd),
            env=merged,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return LocalExecResult(124, "", f"timed out after {timeout_sec}s")
        return LocalExecResult(
            proc.returncode if proc.returncode is not None else -1,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return Path(path).expanduser().is_dir()

    def _local_target(self, target_path: str) -> Path:
        """Map a container-absolute tool path into this session's bin dir."""
        target = Path(target_path).expanduser()
        for prefix in ("/usr/local/bin", "/usr/bin"):
            try:
                return self.bin_dir / target.relative_to(prefix)
            except ValueError:
                continue
        return target

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """A copy, since 'the environment' is the same filesystem."""
        dst = self._local_target(target_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(source_path).expanduser(), dst)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        dst = Path(target_path).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(source_path).expanduser(), dst)

    # --- convenience -------------------------------------------------------

    def describe(self) -> str:
        return f"local shell in {self.cwd} (session {self.session_id})"

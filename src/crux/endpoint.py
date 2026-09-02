"""Reaching the model, including the ssh tunnel it usually sits behind.

The model this project was tuned against is served by sglang on an eval box and
bound to that box's loopback, so a laptop reaches it through a forwarded port.
That forward is easy to forget and, worse, easy not to notice: when it drops,
the agent reports "Connection error" and retries, which reads like the model is
down rather than like a tunnel needing one command.

So the tunnel is the CLI's job. `CRUX_TUNNEL=<ssh host>` in the env file is
enough for `crux chat` and `crux repl` to check the port before starting and
open the forward themselves if it is closed.

Started detached, deliberately: it has to outlive the command that opened it,
because the point is that the next `crux chat` finds the port already up.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

# Enough for ssh to authenticate and bind on a warm connection; a cold one that
# needs longer will be reported rather than waited on forever.
_TUNNEL_WAIT_SEC = 15.0

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Whether something is accepting connections there.

    A TCP connect, not an HTTP request: the server answers an unauthenticated
    request with 401, and an earlier version of this check treated anything but
    200 as unreachable and restarted a tunnel that was working.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _split(base_url: str) -> tuple[str, int] | None:
    """Host and port of an endpoint, or None if it names neither."""
    parsed = urlparse(base_url)
    if not parsed.hostname:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.hostname, port


def tunnel_command(ssh_host: str, port: int) -> list[str]:
    """The forward, with keepalives so a sleeping laptop drops it cleanly."""
    return [
        "ssh",
        "-N",
        "-L",
        f"{port}:127.0.0.1:{port}",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        ssh_host,
    ]


def ensure_reachable(base_url: str, ssh_host: str | None = None, quiet: bool = False) -> bool:
    """Make sure the endpoint answers, opening a tunnel if that is what is missing.

    Returns whether the port is open at the end. Never raises: a caller that
    cannot reach the model should say so in its own terms.
    """
    where = _split(base_url)
    if where is None:
        return False
    host, port = where
    if _port_open(host, port):
        return True

    ssh_host = ssh_host or os.environ.get("CRUX_TUNNEL", "")
    if not ssh_host or host not in _LOOPBACK:
        return False

    if not quiet:
        print(f"model endpoint {host}:{port} is closed; opening a tunnel to {ssh_host}", file=sys.stderr)
    try:
        subprocess.Popen(
            tunnel_command(ssh_host, port),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # outlives this command, which is the point
        )
    except OSError as exc:
        if not quiet:
            print(f"could not start ssh: {exc}", file=sys.stderr)
        return False

    deadline = time.monotonic() + _TUNNEL_WAIT_SEC
    while time.monotonic() < deadline:
        if _port_open(host, port, timeout=1.0):
            if not quiet:
                print(f"tunnel up on {host}:{port}", file=sys.stderr)
            return True
        time.sleep(0.5)
    if not quiet:
        print(
            f"tunnel to {ssh_host} did not come up within {_TUNNEL_WAIT_SEC:.0f}s.\n"
            f"Try it by hand to see why:\n  {' '.join(tunnel_command(ssh_host, port))}",
            file=sys.stderr,
        )
    return False

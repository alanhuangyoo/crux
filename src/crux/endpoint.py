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

import http.client
import os
import signal
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
    """Whether a TCP connection is accepted there.

    True of a healthy tunnel and also of a dead one: `ssh -N -L` keeps its
    local listener after the remote side is gone, so the handshake succeeds and
    nothing traverses. Kept as the cheap first half of `_alive`.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _speaks_http(host: str, port: int, timeout: float = 4.0) -> bool:
    """Whether something on the far side answers an HTTP request.

    Any status counts, 401 and 404 included. That is the whole point: the
    server answers an unauthenticated request with 401, and an earlier version
    of this check demanded 200, decided a working tunnel was broken and tore it
    down. What distinguishes alive from dead is not the status, it is whether a
    status comes back at all.

    The failure this catches is the one a TCP probe cannot see. A laptop that
    slept, or a forward whose remote end died, leaves `ssh -N -L` listening
    locally with nothing behind it: the connect succeeds, the request hangs or
    resets, and the agent reports "Connection error" through litellm three
    times and gives up -- which reads like the model is down.
    """
    conn = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/v1/models", headers={"Accept": "application/json"})
        conn.getresponse().read(1)
        return True
    except Exception:  # noqa: BLE001 - any failure to get a status means dead
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def _alive(host: str, port: int, timeout: float = 4.0) -> bool:
    """A port that both accepts a connection and answers HTTP on it."""
    return _port_open(host, port, timeout=min(timeout, 2.0)) and _speaks_http(host, port, timeout)


def _listeners_on(port: int) -> list[int]:
    """PIDs listening on a local port, via lsof, empty if lsof is unavailable."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for tok in out.split():
        try:
            pids.append(int(tok))
        except ValueError:
            pass
    return sorted(set(pids))


def _is_ssh(pid: int) -> bool:
    """Whether a pid is an ssh client, so nothing else is ever killed here.

    Three shapes have to match, and only the first is obvious:

        ssh                                     a plain client
        /usr/bin/ssh                            the same, absolute
        ssh: /Users/me/.ssh/cm-81e5... [mux]    a multiplexing master

    The third is what actually holds the port on a machine with
    `ControlMaster auto`, and taking the basename of it yields
    "cm-81e5...[mux]", which matches nothing. That mistake made the kill a
    no-op and left the replacement ssh to die on "Address already in use".
    """
    try:
        comm = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    if not comm:
        return False
    return comm.startswith("ssh:") or comm.rsplit("/", 1)[-1].split()[0] == "ssh"


def _cancel_forward(ssh_host: str, port: int) -> bool:
    """Ask a multiplexing master to drop just this forward.

    Preferred over killing the master, which would take every other session
    sharing that connection with it -- on this machine that includes the shell
    commands driving the benchmark. Both the configured target and the old
    default are tried, because the forward being cancelled is by definition the
    one made under a previous configuration.
    """
    if not ssh_host:
        return False
    for target in dict.fromkeys([tunnel_target(), "127.0.0.1"]):
        try:
            r = subprocess.run(
                ["ssh", "-O", "cancel", "-L", f"{port}:{target}:{port}", ssh_host],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            time.sleep(0.4)
            if not _listeners_on(port):
                return True
    return not _listeners_on(port)


def _kill_stale_tunnel(port: int, ssh_host: str = "") -> bool:
    """Take down a local forward that is listening and carrying nothing.

    Without this, `ensure_reachable` sees the port open, declines to start a
    tunnel, and every turn fails against a listener that will never work.

    Found by port rather than by command line, because the common case here is
    the one a command-line match cannot see. With `ControlMaster auto` in
    ~/.ssh/config -- which most people who ssh to the same box all day have --
    the forward is held by a multiplexing master whose argv is

        ssh: /Users/me/.ssh/cm-81e5e342ce3f... [mux]

    with no `-L` in it at all. Matching on `-L <port>:` finds nothing, the kill
    silently does nothing, and the replacement ssh dies on
    "bind [127.0.0.1]:30000: Address already in use".

    Only ssh is ever signalled: if something else is on the port, that is a
    conflict to report, not to resolve by killing it.
    """
    # The polite route first: it leaves every other session on the shared
    # connection alone.
    if _cancel_forward(ssh_host, port):
        return True

    killed = False
    for pid in _listeners_on(port):
        if not _is_ssh(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed = True
        except (ProcessLookupError, PermissionError):
            pass
    if killed:
        # ControlPersist means the master can outlive the TERM briefly; give it
        # a moment, then check, then insist.
        for _ in range(10):
            time.sleep(0.3)
            if not _listeners_on(port):
                return True
        for pid in _listeners_on(port):
            if _is_ssh(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        time.sleep(0.5)
    return killed


def _port_holder(port: int) -> str:
    """A human description of whatever holds the port, for an error message."""
    pids = _listeners_on(port)
    if not pids:
        return ""
    try:
        out = subprocess.run(
            ["ps", "-p", ",".join(str(p) for p in pids), "-o", "pid=,comm="],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ", ".join(str(p) for p in pids)
    return "; ".join(line.strip() for line in out.splitlines())


def _split(base_url: str) -> tuple[str, int] | None:
    """Host and port of an endpoint, or None if it names neither."""
    parsed = urlparse(base_url)
    if not parsed.hostname:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.hostname, port


def tunnel_target() -> str:
    """Where the forward lands on the far side of the ssh host.

    Defaults to the ssh host's own loopback, which is right when the model is
    served on the box you can reach. It stops being right the moment the model
    moves: on this deployment the jump host is the eval box and the model is on
    a different machine, so `-L 30000:127.0.0.1:30000` forwards to a port
    nothing is listening on.

    The failure is quiet and slow to read. ssh binds the local port and the
    forward looks up: a TCP connect succeeds, `curl` returns nothing, and the
    agent reports "Connection error" three times through litellm -- which reads
    like the model is down rather than like a forward pointing at the wrong
    host.

    `CRUX_TUNNEL_TARGET=<host or ip>` in the env file is the whole fix.
    """
    return os.environ.get("CRUX_TUNNEL_TARGET", "127.0.0.1")


def tunnel_command(ssh_host: str, port: int, target: str | None = None) -> list[str]:
    """The forward, with keepalives so a sleeping laptop drops it cleanly."""
    return [
        "ssh",
        "-N",
        "-L",
        f"{port}:{target or tunnel_target()}:{port}",
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
    if _alive(host, port):
        return True

    ssh_host = ssh_host or os.environ.get("CRUX_TUNNEL", "")
    if not ssh_host or host not in _LOOPBACK:
        return False

    # A listener with nothing behind it has to go first, or the new ssh cannot
    # bind and ExitOnForwardFailure kills it immediately.
    if _port_open(host, port):
        if not quiet:
            print(
                f"{host}:{port} accepts connections but does not answer; "
                "the forward is stale, replacing it",
                file=sys.stderr,
            )
        if not _kill_stale_tunnel(port, ssh_host) and not quiet:
            holder = _port_holder(port)
            print(
                f"could not clear {host}:{port}"
                + (f" -- held by {holder}" if holder else "")
                + "\nNothing was killed: only an ssh forward is ever taken down here.",
                file=sys.stderr,
            )

    if not quiet:
        print(
            f"model endpoint {host}:{port} is closed; opening a tunnel to "
            f"{ssh_host} -> {tunnel_target()}:{port}",
            file=sys.stderr,
        )
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
        if _alive(host, port, timeout=2.0):
            if not quiet:
                print(f"tunnel up on {host}:{port}", file=sys.stderr)
            return True
        time.sleep(0.5)
    if not quiet:
        print(
            f"tunnel to {ssh_host} did not come up within {_TUNNEL_WAIT_SEC:.0f}s.\n"
            f"It forwards to {tunnel_target()}:{port} on that host; if the model is\n"
            f"served somewhere else, set CRUX_TUNNEL_TARGET to where it actually is.\n"
            f"Try it by hand to see why:\n  {' '.join(tunnel_command(ssh_host, port))}",
            file=sys.stderr,
        )
    return False

"""The tunnel check, which is the difference between working and 'Connection error'."""

from __future__ import annotations

import socket
import threading

from crux.endpoint import (
    _alive,
    _port_open,
    _speaks_http,
    _split,
    ensure_reachable,
    tunnel_command,
)


def test_split_reads_host_and_port():
    assert _split("http://127.0.0.1:30000/v1") == ("127.0.0.1", 30000)
    assert _split("https://api.example.com/v1") == ("api.example.com", 443)
    assert _split("not a url") is None


def _http_server(status_line: bytes):
    """A socket that answers one request with a fixed status line, then closes."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            try:
                conn.recv(4096)
                conn.sendall(status_line + b"Content-Length: 0\r\n\r\n")
            except OSError:
                pass
            finally:
                conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return srv, srv.getsockname()[1]


def test_an_unauthenticated_server_still_counts_as_reachable():
    """The regression this guards: sglang answers 401 without a key.

    A first version of this check demanded 200 and tore down a working tunnel.
    What distinguishes alive from dead is not the status, it is whether a
    status comes back at all -- so 401 and 404 both count.
    """
    srv, port = _http_server(b"HTTP/1.1 401 Unauthorized\r\n")
    try:
        assert _speaks_http("127.0.0.1", port) is True
        assert _alive("127.0.0.1", port) is True
        assert ensure_reachable(f"http://127.0.0.1:{port}/v1") is True
    finally:
        srv.close()


def test_a_listener_that_answers_nothing_is_not_reachable():
    """The failure a TCP probe cannot see, and the reason this check changed.

    `ssh -N -L` keeps its local listener after the remote side is gone, or
    after a laptop sleeps. The connect succeeds and nothing traverses, so the
    old TCP-only check called it healthy, declined to reopen the forward, and
    every turn failed with "Connection error" three times through litellm --
    which reads like the model being down.
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    threading.Thread(target=lambda: srv.accept(), daemon=True).start()
    try:
        assert _port_open("127.0.0.1", port) is True   # the connect still works
        assert _speaks_http("127.0.0.1", port) is False
        assert _alive("127.0.0.1", port) is False
    finally:
        srv.close()


def test_the_tunnel_target_is_configurable(monkeypatch):
    """Where the forward lands is not always the ssh host's own loopback.

    On this deployment the jump host is the eval box and the model is on
    another machine, so the default `-L p:127.0.0.1:p` forwards to a port
    nothing is listening on -- a forward that looks up and carries nothing.
    """
    monkeypatch.delenv("CRUX_TUNNEL_TARGET", raising=False)
    assert "30000:127.0.0.1:30000" in " ".join(tunnel_command("box", 30000))
    monkeypatch.setenv("CRUX_TUNNEL_TARGET", "192.168.21.40")
    assert "30000:192.168.21.40:30000" in " ".join(tunnel_command("box", 30000))


def test_no_tunnel_configured_is_a_clean_false(monkeypatch):
    """A closed port with nowhere to tunnel to reports, rather than hanging."""
    monkeypatch.delenv("CRUX_TUNNEL", raising=False)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()  # nothing listening now
    assert ensure_reachable(f"http://127.0.0.1:{port}/v1", quiet=True) is False


def test_a_remote_host_is_never_tunnelled(monkeypatch):
    """Only a loopback endpoint means 'served elsewhere, forwarded here'.

    The port check is stubbed rather than aimed at an unroutable address: some
    sandboxed networks accept a connection to anything, which made an earlier
    version of this test measure the network instead of the rule.
    """
    import crux.endpoint as endpoint

    monkeypatch.setenv("CRUX_TUNNEL", "somewhere")
    monkeypatch.setattr(endpoint, "_port_open", lambda *a, **k: False)
    started = []
    monkeypatch.setattr(endpoint.subprocess, "Popen", lambda *a, **k: started.append(a))
    assert ensure_reachable("http://model.internal:8000/v1", quiet=True) is False
    assert started == [], "a non-loopback endpoint must not be tunnelled"


def test_a_loopback_endpoint_is_tunnelled(monkeypatch):
    """The other half of the same rule, so the guard cannot pass by never firing."""
    import crux.endpoint as endpoint

    monkeypatch.setenv("CRUX_TUNNEL", "box")
    monkeypatch.setattr(endpoint, "_port_open", lambda *a, **k: False)
    started = []
    monkeypatch.setattr(endpoint, "_TUNNEL_WAIT_SEC", 0.0)
    monkeypatch.setattr(endpoint.subprocess, "Popen", lambda *a, **k: started.append(a[0]))
    ensure_reachable("http://127.0.0.1:30000/v1", quiet=True)
    assert started and "box" in started[0]


def test_the_forward_keeps_alive():
    cmd = tunnel_command("box", 30000)
    assert "30000:127.0.0.1:30000" in cmd
    assert "ServerAliveInterval=30" in cmd
    assert "ExitOnForwardFailure=yes" in cmd

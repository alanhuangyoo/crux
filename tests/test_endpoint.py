"""The tunnel check, which is the difference between working and 'Connection error'."""

from __future__ import annotations

import socket
import threading

from crux.endpoint import _port_open, _split, ensure_reachable, tunnel_command


def test_split_reads_host_and_port():
    assert _split("http://127.0.0.1:30000/v1") == ("127.0.0.1", 30000)
    assert _split("https://api.example.com/v1") == ("api.example.com", 443)
    assert _split("not a url") is None


def test_an_unauthenticated_server_still_counts_as_reachable():
    """The regression this guards: sglang answers 401 without a key.

    A first version of this check made an HTTP request and treated anything but
    200 as down, so it tore down a tunnel that was working. A TCP connect is the
    question actually being asked.
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    threading.Thread(target=lambda: srv.accept(), daemon=True).start()
    try:
        assert _port_open("127.0.0.1", port) is True
        assert ensure_reachable(f"http://127.0.0.1:{port}/v1") is True
    finally:
        srv.close()


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

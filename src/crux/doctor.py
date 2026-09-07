"""Check the things that have actually broken, in the order they break.

Every entry here is a fault that happened on this project and cost hours,
because each one reported itself somewhere other than where it was. That is the
pattern worth building against: the message you get is almost never the message
you need.

    what it said                          what was wrong
    ----------------------------------------------------------------------
    Connection error (x3, then give up)   an ssh forward pointing at a host
                                          that never served the model
    Failed to start tmux session.         one apt-get carrying two packages,
      Error: None                         where the one with no candidate
                                          took down the one that had it
    container exited (126)                a compose file supplying `command`
                                          to an image that sets ENTRYPOINT
    no tasks matched the filter(s),       a dataset name with no org, and a
      then a list identical to yours      prefix added anyway
    Verifier execution timed out          a judge pointed at the loopback of
      after 900                           the container it runs in
    Docker compose command failed         a YAML file a stray shell expansion
                                          had eaten a line out of

Read-only. Every check either reports or suggests; none of them changes
anything, so running it while a benchmark is in flight is safe.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from crux import ui

OK, WARN, BAD, SKIP = "ok", "warn", "bad", "skip"

_MARK = {
    OK: f"{ui.GREEN}✓{ui.RESET}",
    WARN: f"{ui.YELLOW}!{ui.RESET}",
    BAD: f"{ui.RED}✗{ui.RESET}",
    SKIP: f"{ui.GREY}-{ui.RESET}",
}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "", fix: str = "") -> None:
        self.rows.append((status, name, detail, fix))

    def render(self) -> str:
        w = max((len(n) for _, n, _, _ in self.rows), default=10)
        out = []
        for status, name, detail, fix in self.rows:
            out.append(f"  {_MARK[status]} {name:<{w}}  {detail}")
            if fix and status in (WARN, BAD):
                for line in fix.splitlines():
                    out.append(f"      {ui.GREY}{line}{ui.RESET}")
        bad = sum(1 for s, *_ in self.rows if s == BAD)
        warn = sum(1 for s, *_ in self.rows if s == WARN)
        out.append("")
        if bad:
            out.append(ui.error(f"  {bad} broken") + (ui.warn(f", {warn} to look at") if warn else ""))
        elif warn:
            out.append(ui.warn(f"  {warn} to look at, nothing broken"))
        else:
            out.append(ui.ok("  everything the agent needs is here"))
        return "\n".join(out)

    @property
    def worst(self) -> int:
        if any(s == BAD for s, *_ in self.rows):
            return 1
        return 0


def _env_file() -> Path:
    return Path(os.environ.get("CRUX_HOME", Path.home() / ".crux")) / "env"


def _load_env() -> dict[str, str]:
    """The env file as a dict, without exporting it."""
    p = _env_file()
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _check_env(rep: Report) -> dict[str, str]:
    env = _load_env()
    p = _env_file()
    if not p.exists():
        rep.add(BAD, "env file", f"{p} does not exist",
                "Create it with at least:\n"
                "  OPENAI_BASE_URL=http://<host>:<port>/v1\n"
                "  OPENAI_API_KEY=<key or EMPTY>")
        return env
    have = [k for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY") if env.get(k)]
    missing = [k for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY") if not env.get(k)]
    if missing:
        rep.add(BAD, "env file", f"{p}: missing {', '.join(missing)}")
    else:
        rep.add(OK, "env file", f"{p}  ({len(env)} keys, {', '.join(have)} set)")

    # The fault that cost a whole SWE-Atlas run: two base-url variables that
    # disagree, with the agent reading one and a task's verifier the other.
    base = env.get("OPENAI_BASE_URL", "")
    alt = env.get("OPENAI_API_BASE", "")
    if base and alt and base != alt:
        rep.add(WARN, "base url", f"OPENAI_BASE_URL={base}  OPENAI_API_BASE={alt}",
                "Two names for one endpoint, pointing at different places. Tasks that\n"
                "wire a verifier as ${OPENAI_API_BASE} will use the second one; the\n"
                "agent uses the first. All 124 SWE-Atlas trials were lost to exactly\n"
                "this, reported as 'Verifier execution timed out after 900'.")
    elif base:
        rep.add(OK, "base url", base)
    return env


def _check_tunnel(rep: Report, env: dict[str, str]) -> None:
    from crux.endpoint import (
        _alive, _port_holder, _port_open, _speaks_http, _split, tunnel_target,
    )

    base = env.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    if not base:
        rep.add(SKIP, "tunnel", "no base url to check")
        return
    where = _split(base)
    if where is None:
        rep.add(BAD, "tunnel", f"cannot parse {base}")
        return
    host, port = where
    local = host in ("127.0.0.1", "localhost", "::1")
    ssh_host = env.get("CRUX_TUNNEL") or os.environ.get("CRUX_TUNNEL", "")

    if not local:
        rep.add(SKIP, "tunnel", f"{host} is not loopback; no forward needed")
        return
    if not ssh_host:
        rep.add(WARN, "tunnel", "endpoint is loopback but CRUX_TUNNEL is unset",
                "Set CRUX_TUNNEL=<ssh host> and crux will open the forward itself.")
        return

    target = env.get("CRUX_TUNNEL_TARGET") or tunnel_target()
    tcp = _port_open(host, port)
    http = _speaks_http(host, port) if tcp else False
    if tcp and http:
        rep.add(OK, "tunnel", f"{host}:{port} -> {ssh_host} -> {target}:{port}")
    elif tcp and not http:
        rep.add(BAD, "tunnel", f"{host}:{port} accepts connections but answers nothing",
                f"A stale forward. Held by: {_port_holder(port) or 'unknown'}\n"
                "`crux repl` clears this itself now; by hand it is\n"
                f"  ssh -O cancel -L {port}:{target}:{port} {ssh_host}")
    else:
        rep.add(WARN, "tunnel", f"{host}:{port} closed (crux opens it on demand)",
                f"  ssh -N -L {port}:{target}:{port} {ssh_host}")

    # The forward can be up and still point at nothing, which is the failure
    # that reads as "the model is down".
    if ssh_host and shutil.which("ssh"):
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_host,
                 f"curl -s -m 8 -o /dev/null -w '%{{http_code}}' http://{target}:{port}/v1/models"],
                capture_output=True, text=True, timeout=30,
            )
            code = (r.stdout or "").strip()[-3:]
            if code.isdigit() and code != "000":
                rep.add(OK, "far side", f"{ssh_host} reaches {target}:{port} (HTTP {code})")
            else:
                rep.add(BAD, "far side", f"{ssh_host} cannot reach {target}:{port}",
                        "The forward lands where nothing is serving. If the model moved,\n"
                        "point CRUX_TUNNEL_TARGET at the machine that has it.")
        except (OSError, subprocess.SubprocessError):
            rep.add(SKIP, "far side", f"could not ask {ssh_host}")


def _check_model(rep: Report, env: dict[str, str]) -> None:
    import urllib.error
    import urllib.request

    base = (env.get("OPENAI_BASE_URL") or "").rstrip("/")
    key = env.get("OPENAI_API_KEY", "EMPTY")
    if not base:
        rep.add(SKIP, "model", "no base url")
        return

    def call(path: str, body=None, timeout=60):
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode() if body else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        return json.load(urllib.request.urlopen(req, timeout=timeout))

    try:
        served = [m.get("id") for m in call("/models").get("data", [])]
    except Exception as exc:  # noqa: BLE001
        rep.add(BAD, "model", f"/models failed: {exc}")
        return
    rep.add(OK, "model", f"serving {', '.join(served) or '(none)'}")

    if not served:
        return
    t0 = time.monotonic()
    try:
        r = call("/chat/completions", {
            "model": served[0],
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 64, "temperature": 0,
        }, timeout=180)
    except Exception as exc:  # noqa: BLE001
        rep.add(BAD, "generation", f"a completion failed: {exc}")
        return
    el = time.monotonic() - t0
    usage = r.get("usage") or {}
    n = int(usage.get("completion_tokens") or 0)
    rate = n / el if el > 0 else 0
    status = OK if rate > 20 else WARN
    rep.add(status, "generation",
            f"{n} tokens in {ui.human_secs(el)} ({rate:.0f} tok/s single stream)",
            "Under 20 tok/s usually means the endpoint is saturated by something\n"
            "else -- a benchmark run will do it. Interactive turns will be slow."
            if status == WARN else "")


# The prompt the sampling probe uses. It has to have more than one good answer,
# or the probe reports a determinism it never tested: the first version asked
# the model to name a command for listing files, got `ls` four times out of
# four, and called the endpoint deterministic -- on the very deployment whose
# measured behaviour is three different answers in five. A check with one
# overwhelming right answer cannot fail, which is the fault this project found
# in the agent's own checklists, reproduced in its own tooling.
_SAMPLING_PROBE = (
    "Write one short bash command that lists python files modified today. "
    "Output only the command."
)

# Enough draws to see a difference without making `crux doctor` slow.
_SAMPLING_DRAWS = 5


def _check_sampling(rep: Report, env: dict[str, str]) -> None:
    """Whether anything sets a sampling temperature, and what that costs.

    Nothing did, for the whole life of this project. Terminus passes a
    temperature only when one is explicitly configured and crux never
    configured one, so every number it produced was sampled at the server
    default -- the maximum-variance setting -- and nothing anywhere said so.

    What that bought, measured: 60% of SWE-bench failures solve on a plain
    re-run with nothing changed, 15-16% of tasks flip between two runs of one
    configuration, and an 89-task run resolves about ±8 points. It is also why
    two separate gates looked effective and neither survived attribution: in a
    system this noisy, anything selected on failure looks better re-run.

    Probed rather than read off a config, because the default lives on the
    server and the client cannot see it.
    """
    import urllib.request

    base = (env.get("OPENAI_BASE_URL") or "").rstrip("/")
    key = env.get("OPENAI_API_KEY", "EMPTY")
    if not base:
        rep.add(SKIP, "sampling", "no base url")
        return

    def ask(extra):
        body = {
            "model": env.get("CRUX_MODEL", "qwen3.8-27b"),
            "messages": [{"role": "user", "content": _SAMPLING_PROBE}],
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
            **extra,
        }
        req = urllib.request.Request(
            base + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        d = json.load(urllib.request.urlopen(req, timeout=90))
        return (d["choices"][0]["message"].get("content") or "").strip()

    try:
        seen = {ask({}) for _ in range(_SAMPLING_DRAWS)}
    except Exception as exc:  # noqa: BLE001
        rep.add(SKIP, "sampling", f"could not probe: {exc}")
        return
    if len(seen) == 1:
        rep.add(OK, "sampling", "the default is deterministic here")
    else:
        rep.add(WARN, "sampling",
                f"no temperature set: {_SAMPLING_DRAWS} samples gave {len(seen)} different answers",
                "Every run inherits the server default, which is the highest-variance\n"
                "setting. Measured downstream: 60% of failures solve on a plain re-run,\n"
                "15-16% of tasks flip between identical configurations, and an 89-task\n"
                "run resolves only ±8 points.\n"
                "  crux bench --agent-kwarg temperature=0.2 ...\n"
                "(0.2 rather than 0: Qwen3's card warns greedy decoding in thinking mode\n"
                "can fall into repetition loops.)")


def _check_local_tools(rep: Report) -> None:
    for tool, why in (("tmux", "Terminus types into a live tmux session"),
                      ("docker", "benchmark trials each get a container"),
                      ("git", "used by several task images"),
                      ("lsof", "how a stale forward is found by port")):
        path = shutil.which(tool)
        if path:
            rep.add(OK, tool, path)
        else:
            status = BAD if tool == "tmux" else WARN
            rep.add(status, tool, f"not on PATH -- {why}")

    try:
        import harbor  # noqa: F401

        rep.add(OK, "harbor", getattr(__import__("harbor"), "__version__", "installed"))
    except ImportError:
        rep.add(BAD, "harbor", "not importable",
                "  uv tool install 'harbor[modal]'")


def _check_agent_tools(rep: Report) -> None:
    """The two helpers the prompt tells the model to run.

    An earlier version of the prompt named `crux submit` while the binary was
    installed only under a flag, so 26 of 89 trials ran a command that did not
    exist. Whether the file is there is worth one stat call.
    """
    from crux import terminus_agent

    res = Path(terminus_agent.__file__).parent / "resources"
    for name in ("crux_tool.py", "apply_patch.py"):
        p = res / name
        if p.is_file():
            rep.add(OK, name, f"{p.stat().st_size} bytes")
        else:
            rep.add(BAD, name, f"missing from {res}")


def _check_compose_patch(rep: Report) -> None:
    """The harbor file this project edits, and whether it still parses.

    It has been corrupted once, by a shell expansion inside an unquoted
    heredoc, and the symptom was every container create and destroy failing
    with `could not find expected ':'` -- across every run at once.
    """
    try:
        import harbor

        f = Path(harbor.__file__).parent / "environments" / "docker" / "docker-compose-prebuilt.yaml"
    except ImportError:
        rep.add(SKIP, "compose file", "harbor not importable")
        return
    if not f.is_file():
        rep.add(WARN, "compose file", f"not found at {f}")
        return
    text = f.read_text(errors="replace")
    try:
        import yaml  # type: ignore

        doc = yaml.safe_load(text)
        services = (doc or {}).get("services", {})
        main = services.get("main", {})
        has_ep = "entrypoint" in main
        rep.add(OK, "compose file",
                f"parses, {len(services)} service(s)"
                + (", entrypoint cleared" if has_ep else ", entrypoint NOT cleared"))
        if not has_ep:
            rep.add(WARN, "compose entrypoint",
                    "images that set their own ENTRYPOINT will exit 126",
                    "harbor supplies its keepalive as `command` and assumes no\n"
                    "ENTRYPOINT. Add `entrypoint: []` under services.main.")
    except ImportError:
        rep.add(SKIP, "compose file", "pyyaml not installed; cannot parse")
    except Exception as exc:  # noqa: BLE001
        rep.add(BAD, "compose file", f"does not parse: {exc}",
                f"Restore it: {f}.bak-crux")


def cmd_doctor(args) -> int:
    rep = Report()
    print(ui.rule("crux doctor"))
    env = _check_env(rep)
    _check_local_tools(rep)
    _check_agent_tools(rep)
    _check_tunnel(rep, env)
    if not getattr(args, "offline", False):
        _check_model(rep, env)
        _check_sampling(rep, env)
    else:
        rep.add(SKIP, "model", "--offline")
    _check_compose_patch(rep)
    print(rep.render())
    return rep.worst

"""`crux chat` — the interactive agent, on pi's front-end.

pi supplies what a benchmark scaffold cannot: a terminal UI, multi-turn
conversation, session resume, interrupt. crux supplies the parts that were
measured -- the prompt sections and the model configuration -- through pi's own
`--append-system-prompt`, which is a supported extension point rather than a
fork.

Being explicit about what this is: a wrapper. The alternative was writing a REPL
onto crux's Terminus base, whose `run()` builds a fresh chat on every call, so
multi-turn would need its chat construction monkeypatched. A hand-rolled loop
that nobody uses is worth less than a good one that carries our changes.

The same sections run under `crux bench -a crux.pi_agent:CruxPiAgent`, so what
is measured on the benchmark is what runs here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from crux.endpoint import ensure_reachable
from crux.prompts import build_sections

PI_CONFIG = Path.home() / ".pi" / "agent" / "models.json"
PROVIDER = "crux-local"


def _endpoint() -> tuple[str, str, str]:
    """Base URL, api key and model, from the environment or the .env file."""
    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("CRUX_MODEL")
    if not (base and key):
        for candidate in (Path(".env"), Path.home() / ".crux" / "env"):
            if not candidate.exists():
                continue
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if "=" not in line or line.strip().startswith("#"):
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if k == "OPENAI_BASE_URL" and not base:
                    base = v
                elif k == "OPENAI_API_KEY" and not key:
                    key = v
                elif k == "CRUX_MODEL" and not model:
                    model = v
                elif k == "CRUX_TUNNEL":
                    # Exported rather than returned: the tunnel is looked up by
                    # whoever needs the endpoint, not only by this caller.
                    os.environ.setdefault("CRUX_TUNNEL", v)
            if base and key:
                break
    return base or "", key or "", model or "qwen3.8-27b"


def ensure_pi_provider(base: str, key: str, model: str, path: Path = PI_CONFIG) -> Path:
    """Write pi's provider entry, leaving any others alone.

    Two details that cost time when they were wrong: the file lives under
    ~/.pi/agent/, not ~/.pi/; and an OpenAI-compatible server like sglang needs
    `compat.supportsDeveloperRole: false`, or pi sends a `developer` role the
    server does not understand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {}
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
    providers = doc.setdefault("providers", {})
    providers[PROVIDER] = {
        "baseUrl": base.rstrip("/"),
        "apiKey": key,
        "api": "openai-completions",
        "compat": {"supportsDeveloperRole": False},
        "models": [{"id": model, "reasoning": True, "maxTokens": 65536}],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def cmd_chat(args) -> int:
    """Launch the interactive agent."""
    if shutil.which("pi") is None:
        print(
            "pi is not installed. It provides the terminal UI:\n"
            "  npm install -g @earendil-works/pi-coding-agent",
            file=sys.stderr,
        )
        return 1

    base, key, model = _endpoint()
    if not base or not key:
        print(
            "no model endpoint found. Set OPENAI_BASE_URL and OPENAI_API_KEY,\n"
            "or put them in ./.env or ~/.crux/env.",
            file=sys.stderr,
        )
        return 1
    model = args.model or model
    if not ensure_reachable(base):
        print(
            f"cannot reach the model at {base}.\n"
            "If it is served on an eval box, set CRUX_TUNNEL=<ssh host> in\n"
            "~/.crux/env and this will open the forward itself.",
            file=sys.stderr,
        )
        return 1
    ensure_pi_provider(base, key, model)

    sections = [s.strip() for s in (args.sections or "").split(",") if s.strip()]
    command = ["pi", "--provider", PROVIDER, "--model", model]
    written = None
    if sections:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8", prefix="crux-sections-"
        ) as f:
            f.write(build_sections(sections))
            written = Path(f.name)
        command += ["--append-system-prompt", str(written)]
    if args.resume:
        command.append("--resume")
    elif args.cont:
        command.append("--continue")
    command += args.rest or []

    print(f"crux chat  model={model}  endpoint={base}")
    print(f"  sections: {', '.join(sections) if sections else '(none — stock pi)'}")
    try:
        return subprocess.call(command)
    finally:
        if written is not None:
            written.unlink(missing_ok=True)

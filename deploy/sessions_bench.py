#!/usr/bin/env python3
"""Measure the router the way agent traffic actually arrives.

abtest.py sends every request behind one shared prefix, which is a pathological
input for cache-aware routing: there is exactly one right engine for the cache
and exactly one wrong answer for load. Real traffic is N distinct conversations,
each with its own long prefix, each growing a few hundred tokens per turn. That
shape is what decides whether affinity and balance can be had at the same time.

So: N sessions, each seeded with its own multi-thousand-token history, each
taking T sequential turns that append to it. Between sessions the prefixes
share only the system prompt; within a session the prefix is nearly the whole
request. Cache hit rate comes from the server's own accounting rather than
being inferred from latency.
"""
import argparse, asyncio, json, statistics, time, urllib.request

KEY = "sk-crux-iM-eVeNJmh1_crsLPfiBInwFaU410pNM"

SYSTEM = (
    "You are a systems engineer working inside a Linux container. Verify every "
    "claim against the filesystem before relying on it, and chain shell commands "
    "so one round trip does as much work as possible.\n"
)


def history(session: int) -> list:
    """A distinct multi-turn transcript per session, ~3-4K tokens of prefix."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for turn in range(6):
        msgs.append({
            "role": "user",
            "content": f"[session {session} step {turn}] Inspect the service at "
                       f"/srv/app-{session}-{turn} and report what its entrypoint "
                       f"does. " + f"Context line {session}.{turn}: " + "x" * 400,
        })
        msgs.append({
            "role": "assistant",
            "content": f"Checked /srv/app-{session}-{turn}. The entrypoint is a "
                       f"shell wrapper that execs gunicorn on port {8000 + turn}. "
                       + "Detail: " + "y" * 400,
        })
    return msgs


async def turn(url: str, model: str, msgs: list, out_tokens: int):
    body = json.dumps({
        "model": model, "messages": msgs, "max_tokens": out_tokens,
        "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    def blocking():
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    try:
        d = await loop.run_in_executor(None, blocking)
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:100]

    dt = time.perf_counter() - t0
    u = d.get("usage") or {}
    m = d["choices"][0]["message"]
    text = (m.get("content") or "") + (m.get("reasoning_content") or "")
    # cached_tokens is what proves the prefix cache was actually reused; a
    # latency drop alone could just as easily be a quieter engine.
    details = u.get("prompt_tokens_details") or {}
    return {
        "latency": dt,
        "prompt": u.get("prompt_tokens", 0),
        "cached": (details or {}).get("cached_tokens", 0) or 0,
        "completion": u.get("completion_tokens", 0),
        "text": text,
    }, None


async def session(url: str, model: str, sid: int, turns: int, out_tokens: int):
    msgs = history(sid)
    out = []
    for t in range(turns):
        msgs.append({"role": "user", "content":
                     f"[session {sid} follow-up {t}] Given the above, what should "
                     f"be checked next? Answer in two sentences."})
        r, err = await turn(url, model, msgs, out_tokens)
        if err:
            out.append({"err": err})
            break
        out.append(r)
        msgs.append({"role": "assistant", "content": r["text"] or "ok"})
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--out-tokens", type=int, default=200)
    a = ap.parse_args()

    t0 = time.perf_counter()
    res = await asyncio.gather(*(
        session(a.url, a.model, s, a.turns, a.out_tokens) for s in range(a.sessions)
    ))
    wall = time.perf_counter() - t0

    flat = [r for s in res for r in s if "err" not in r]
    errs = [r for s in res for r in s if "err" in r]
    if not flat:
        print("all failed:", {e["err"] for e in errs})
        return

    # Turn 1 of each session is a cold prefix; later turns should hit cache.
    first = [s[0] for s in res if s and "err" not in s[0]]
    later = [r for s in res for r in s[1:] if "err" not in r]

    def hit(rs):
        p = sum(r["prompt"] for r in rs)
        return (sum(r["cached"] for r in rs) / p * 100) if p else 0.0

    print(f"sessions={a.sessions} turns={a.turns}  wall={wall:.1f}s  "
          f"ok={len(flat)} failed={len(errs)}")
    print(f"  throughput      {sum(r['completion'] for r in flat) / wall:8.1f} tok/s")
    print(f"  latency p50     {statistics.median(r['latency'] for r in flat):8.2f}s")
    print(f"  latency p95     {sorted(r['latency'] for r in flat)[int(len(flat)*0.95)-1]:8.2f}s")
    print(f"  cache hit  turn 1 (cold)  {hit(first):5.1f}%")
    print(f"  cache hit  turns 2+       {hit(later):5.1f}%")
    if later:
        print(f"  latency turn 1  {statistics.median(r['latency'] for r in first):8.2f}s")
        print(f"  latency turns 2+{statistics.median(r['latency'] for r in later):8.2f}s")


if __name__ == "__main__":
    asyncio.run(main())

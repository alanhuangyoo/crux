#!/usr/bin/env python3
"""A/B two engines on the metrics that actually bind: TTFT and decode speed.

Total throughput hides the thing we are trying to change. Speculative decoding
does not make the GPU faster; it makes each forward pass emit more than one
token when the draft guesses right. That shows up as decode tok/s per stream
and not at all in aggregate throughput once the batch is large enough to
saturate compute -- so the two numbers have to be reported separately, at
concurrency levels on both sides of saturation.

Prompts are deliberately near-identical with a varying tail: that is the shape
of agent traffic (shared system prompt + transcript, few hundred new tokens)
and it is also what makes prefix caching and MTP accept rates realistic. A
random-prompt benchmark would understate both.
"""
import argparse, asyncio, concurrent.futures, json, statistics, sys, time
import urllib.request

KEY = "sk-crux-iM-eVeNJmh1_crsLPfiBInwFaU410pNM"

SHARED = """You are a careful systems engineer working inside a Linux container.
You reason step by step, you never guess at file contents, and you verify each
claim against the actual filesystem before you rely on it. When you write shell
commands you chain them so that one round trip does as much work as possible.
""" * 12


async def one(session_id: int, url: str, model: str, out_tokens: int):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SHARED},
            {"role": "user", "content": f"Task {session_id}: explain in detail how to find every "
                                        f"file larger than {session_id + 1} megabytes under /var "
                                        f"and summarise them by directory. Think it through."},
        ],
        "max_tokens": out_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )

    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    ttft = None
    n = 0

    def blocking():
        nonlocal ttft, n
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    # Reasoning models emit the chain through a separate field;
                    # both count as decoded tokens for a speed measurement.
                    if delta.get("content") or delta.get("reasoning_content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        n += 1

    try:
        await loop.run_in_executor(None, blocking)
    except Exception as e:  # noqa: BLE001 - a failed stream is a data point
        return {"ok": False, "err": str(e)[:120]}

    dur = time.perf_counter() - t0
    if ttft is None or n == 0:
        return {"ok": False, "err": "no tokens"}
    decode = (n - 1) / (dur - ttft) if dur > ttft and n > 1 else 0.0
    return {"ok": True, "ttft": ttft, "total": dur, "n": n, "decode": decode}


async def sweep(url: str, model: str, conc: int, out_tokens: int):
    # Every request is a blocking urllib call handed to the default executor,
    # which sizes itself at min(32, cpu_count + 4). On any real box that is 32
    # -- so without this, asking for 96 concurrent requests measures 32, and
    # throughput appears to plateau at whatever 32 streams can pull. It looks
    # exactly like server saturation and it is entirely client-side.
    loop = asyncio.get_running_loop()
    prev = getattr(loop, "_default_executor", None)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=conc + 8)
    loop.set_default_executor(pool)
    try:
        res = await asyncio.gather(*(one(i, url, model, out_tokens) for i in range(conc)))
    finally:
        loop.set_default_executor(prev) if prev else None
        pool.shutdown(wait=False)
    good = [r for r in res if r["ok"]]
    if not good:
        errs = {r["err"] for r in res}
        return {"conc": conc, "ok": 0, "err": list(errs)[:2]}
    wall = max(r["total"] for r in good)
    return {
        "conc": conc,
        "ok": len(good),
        "ttft_p50": statistics.median(r["ttft"] for r in good),
        "ttft_p95": sorted(r["ttft"] for r in good)[int(len(good) * 0.95) - 1] if len(good) > 1 else good[0]["ttft"],
        "decode_p50": statistics.median(r["decode"] for r in good),
        "tput": sum(r["n"] for r in good) / wall,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", required=True, help="label=http://host:port")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--conc", default="1,8,32,64")
    ap.add_argument("--out-tokens", type=int, default=256)
    a = ap.parse_args()

    targets = [u.split("=", 1) for u in a.url]
    levels = [int(x) for x in a.conc.split(",")]

    print(f"{'engine':<10} {'conc':>5} {'ok':>5} {'TTFT p50':>9} {'TTFT p95':>9} {'decode/req':>11} {'total tok/s':>12}")
    print("-" * 68)
    for label, url in targets:
        # One warm pass so prefix cache and CUDA graphs are not being paid for
        # inside the measurement.
        await sweep(url, a.model, 2, 32)
        for c in levels:
            r = await sweep(url, a.model, c, a.out_tokens)
            if not r["ok"]:
                print(f"{label:<10} {c:>5} {'FAIL':>5}  {r.get('err')}")
                continue
            print(f"{label:<10} {r['conc']:>5} {r['ok']:>5} "
                  f"{r['ttft_p50']:>8.2f}s {r['ttft_p95']:>8.2f}s "
                  f"{r['decode_p50']:>9.1f}/s {r['tput']:>10.1f}/s")
        print()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)

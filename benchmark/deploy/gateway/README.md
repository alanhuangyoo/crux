# Gateway: who Codex actually talks to

```
codex ──Responses──▶ CPA :8317 ──Chat──▶ router :30080 ──▶ Qwen3.8-27B
```

new-api (`:3000`) is deployed and works, but is **not** in the Codex path. Why
not is the whole content of this file.

## new-api cannot carry a Responses stream

Its Responses conversion emits the item-level events and never the
response-level ones. Measured on a real stream: sequence numbers ran 2..50,
with no `response.created` (seq 0/1) and no `response.completed` at the end.
Codex treats a stream with no terminal event as a dropped connection, which is
exactly the symptom -- `stream disconnected before completion: stream closed
before response.completed`.

This is upstream issue **#6951**, open. `latest` on Docker Hub is
`v1.0.0-rc.25`, built two days *before* the issue was filed, so no released
image contains a fix.

`pass_through_body_enabled` does not rescue it. A channel pointing at CPA --
which speaks Responses natively and does emit the full lifecycle -- was created
with passthrough on, channel 1 disabled to force routing through it, and the
stream still came back without the terminal events. new-api regenerates the
stream on every path. That channel is still in the database, disabled, ready
for the day #6951 lands.

Non-streaming `/v1/responses` through new-api is fine; only streaming is
affected.

## Base URL needs `/v1`

`http://192.168.21.45:3000` sends `POST /responses`, which the SPA fallback
route answers with 1166 bytes of `<!doctype html>` in 78µs. The access log
tells them apart at a glance:

```
| api | 200 | 215ms | GET  /api/channel/test/1     <- real API
| web | 200 |  78µs | POST /responses              <- the HTML fallback
```

## reasoning_effort: low and medium only

OpenAI clients offer `minimal/low/medium/high`. This checkpoint accepts
`low/medium/xhigh` and raises on anything else -- so Codex's default of `high`
fails before the model runs.

`low` and `medium` exist in both vocabularies and work as-is. `minimal` and
`none` are mapped down to `low` by a `payload.override` rule in
`cpa-config.yaml`.

**`high` cannot be fixed at the gateway.** Two attempts, both undone by CPA:

1. Rewrite `high` -> `xhigh` via `payload.override`. CPA normalises effort back
   into OpenAI's own vocabulary before calling the upstream, and `xhigh` is not
   in it. The error gives this away -- send `xhigh` and the model reports
   `Unexpected reasoning effort high`.
2. Delete the field via `payload.filter`, so the template's own
   `reasoning_effort|default('xhigh')` would apply. The converter reads the
   original request, not the filtered payload, so `high` still arrives.

The asymmetry is the whole story: the low end is fixable because `low` survives
normalisation, and the high end is not because `xhigh` does not.

Patching the checkpoint's chat template does work, and was rejected on purpose:
vocabulary translation belongs in a gateway, and a patched checkpoint silently
loses the fix the next time the weights are re-downloaded.

The practical cost is small. Measured reasoning tokens on one prompt: xhigh
198, medium 238, low 134 -- `medium` is the neutral setting that injects no
instruction at all and lets the model decide, while `xhigh` only adds a "think
carefully, validate assumptions" line.

## Codex config

```toml
model_provider = "micuapi"
model = "gpt-5.6-sol"
model_reasoning_effort = "medium"

[model_providers.micuapi]
name = "MicuAPI"
base_url = "http://192.168.21.45:8317/v1"
wire_api = "responses"
requires_openai_auth = true
```

`gpt-5.6-sol`, `gpt-5.6-terra` and `gpt-5.5` are aliases of `qwen3.8-27b`
declared in `cpa-config.yaml`, so no client-side model renaming is needed.

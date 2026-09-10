# Patches applied to the installed harbor

Not part of crux; recorded so a fresh eval box can be brought to the same state
and so the reason survives the next `uv tool upgrade harbor`, which reverts them.

## harbor-docker-compose-prebuilt.yaml

Replaces
`harbor/environments/docker/docker-compose-prebuilt.yaml`.
Backup on the box: same path + `.bak-crux`.

harbor supplies its keepalive as a compose `command` and assumes a prebuilt
image has no ENTRYPOINT. The SWE-Atlas images set `ENTRYPOINT ["/bin/bash"]`,
so the two concatenate into

    /bin/bash sh -c "sleep infinity"

bash then runs the file `sh` as a script and the container exits 126,
"cannot execute binary file". All 124 SWE-Atlas trials failed this way before
the container ever started. Clearing the entrypoint explicitly fixes it and is
a no-op for every Terminal-Bench and SWE-bench image, all of which have a null
entrypoint.

## SWE-Atlas: the judge's endpoint

Not a patch -- an env file, `.env.atlas`, because the deviation belongs to the
run and not to the installed harbor.

SWE-Atlas grades with an LLM judge and its task files wire it as

    [verifier.env]
    EVAL_BASE_URL = "${OPENAI_API_BASE}"
    EVAL_MODEL    = "${EVAL_MODEL:-anthropic/claude-opus-4-5-20251101}"

This box has two base-url variables and they do not agree:

    OPENAI_BASE_URL=http://192.168.21.40:30000/v1   reachable
    OPENAI_API_BASE=http://127.0.0.1:30000/v1       the loopback of whatever
                                                    container reads it

The agent reads `OPENAI_BASE_URL` and never noticed. The judge reads the other
one: every call failed to connect, retried 8 times per rubric, and all 124
trials died at the 900s verifier timeout -- with the agent finished and its
answer already written. The failure reads as "the verifier is slow".

`.env.atlas` sets `EVAL_BASE_URL` and `OPENAI_API_BASE` to the reachable
address, and sets `EVAL_MODEL` to the local model since there is no Anthropic
credential here. Measured on the fixed endpoint: 3.6s per rubric with thinking
on, 0.9s with it off, valid JSON both ways -- so the script is left alone.

Scores from this run are internally comparable and are NOT comparable to the
published SWE-Atlas numbers, which are judged by claude-opus-4-5.

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

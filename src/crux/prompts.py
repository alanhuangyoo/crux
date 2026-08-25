"""Prompt templates for Crux.

Crux is mini-SWE-agent with a modified prompt. The base agent is on the public
Terminal-Bench leaderboard at 76.2%; the loop, the parser, the trajectory
export, and the format contract are all upstream's and are left alone. What
changes here is only what a full run of our own showed the model getting wrong.

The format contract — one ```mswea_bash_command``` block per turn, and
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` to finish — is upstream's parser
talking, so it is reproduced exactly.
"""

# Unchanged from upstream except for the last paragraph. The base agent's
# framing is fine; what it never tells the model is what "done" has to mean.
SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or \
commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning \
process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.
"""

# The grading paragraph is the substantive change. A full 70-task run finished
# with 30 tasks at 60% or more of their checks and only 3 scored, because the
# scoring is per-task all-or-nothing and the model has no way to know that. It
# stops when the work looks basically done, which is exactly where the points
# are lost.
GRADING_SECTION = """\
## How this is graded

A hidden test suite decides the outcome, and scoring is **all or nothing**: \
passing nine checks out of ten scores exactly the same as passing none. There \
is no partial credit to settle for, so "basically working" is a failing state.

That makes the requirement you did not think of the one that decides the task. \
Before you finish, re-read the task statement and list every requirement it \
states or implies — including malformed input, boundary values, empty \
collections, concurrent access, and error paths. Those are what a thorough \
test suite reaches for first, and handling only the happy path is the most \
common way to score zero on a task that was nearly solved.

Then prove each item with a command whose output demonstrates it: feed the \
adversarial input, re-read the file you wrote, re-run the test, check the exit \
code. Do not rely on remembering that you did something earlier — show that it \
holds now.

Keep that list in `crux todo`, with a `--verify` command on every item you \
can express as one. Across a full evaluation this agent declared itself \
finished 28 times and was right 3 times: not from dishonesty, but from having \
done most of the work and lost track of the rest.

**Run `crux todo verify` immediately before you finish.** If it exits \
non-zero, you are not finished, whatever your recollection says — go fix what \
it reports and run it again.
"""

# apply_patch is offered instead of sed because it measurably works better.
# Across 601 calls in one full run it succeeded 97.5% of the time; the failures
# were bad paths and stale context, not mangled files. sed edits fail silently
# by matching the wrong line, which the model then has to notice.
APPLY_PATCH_SECTION = """\
### Edit files with apply_patch (preferred over sed)

`apply_patch` reads a patch on stdin and locates each hunk by its surrounding \
context rather than by line number, so it still applies when your picture of \
the file is slightly out of date, and it fails loudly instead of silently \
editing the wrong line.

```mswea_bash_command
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: src/app.py
@@ def handler():
-    return None
+    return build_response()
*** End Patch
PATCH
```

Headers are `*** Add File: <path>` (every following line prefixed `+`), \
`*** Delete File: <path>`, and `*** Update File: <path>` (optionally followed \
by `*** Move to: <path>`). Inside a hunk ` ` is context, `-` removes, `+` \
adds. Give about three lines of context on each side, and use \
`@@ <enclosing def or class>` when that context repeats in the file. Paths are \
relative. A failed patch changes nothing, so it is safe to correct and retry.
"""

# A shell-only agent reads with `sed -n`, searches with `grep`, and edits with
# `sed -i`. Each has a failure mode the model cannot see: reads come back
# without line numbers to refer to, an unbounded `cat` evicts the context it
# needed, and a sed edit that matches the wrong line reports success. These
# tools are the same set the mature terminal agents converge on (opencode, pi,
# Codex), delivered through the only channel available here.
TOOLKIT_SECTION = """\
### File tools — prefer these over cat/grep/sed

`crux read <path> [--offset N] [--limit N]` — print with line numbers, \
paginated. Output is capped and tells you what you have not seen and how to \
continue, so it will not flood your context. Also lists directories.

`crux grep <pattern> [path] [--include '*.py']` — regex search reporting \
`path:line: text`. Skips .git, node_modules and other noise.

`crux files '<glob>' [path]` — find files by name.

`crux edit <path>` — exact-text replacement, edits given as JSON on stdin:

```mswea_bash_command
crux edit src/app.py <<'JSON'
{"edits": [
  {"old": "    return None", "new": "    return build_response()"},
  {"old": "DEBUG = True", "new": "DEBUG = False"}
]}
JSON
```

Each `old` must occur **exactly once** in the file. That is the point of the \
tool: an anchor matching twice is exactly when `sed -i` edits the wrong line \
and reports success. If it is not unique, extend it with surrounding lines. \
Nothing is written unless every edit in the batch resolves, so a failure never \
leaves the file half-changed.

`crux write <path>` — write stdin to a file, creating parent directories.

The requirement checklist, with a check bound to each item:

```mswea_bash_command
crux todo add "rejects malformed input" --verify "./filter < bad.txt; test $? -ne 0"
```

`crux todo done <n>` re-runs that command and **refuses to close the item if \
it fails**, so closing one is an observation rather than a claim. \
`crux todo verify` re-runs every check at once — edits made for one \
requirement break another constantly, and this is the only way to notice \
before submitting. `crux todo list` exits non-zero while anything is open.

Write the check first, watch it fail, then make it pass. An item with no \
runnable check is allowed but proves nothing.

"""

SURVIVAL_SECTION = """\
## Staying alive

The container has a memory cap, and exceeding it does not raise an error you \\
can recover from — the kernel kills the process and the task ends there, \\
scoring zero regardless of how much you had finished. In one evaluation this \\
was 10% of all trials.

So when a file or dataset might be large, stream it rather than loading it:

- read line by line, or in chunks, instead of `f.read()` / `json.load` on the \\
whole thing
- prefer `head`, `tail`, `wc -l`, `grep`, `sed -n` over `cat` on anything you \\
have not sized first
- check the size before you commit to an approach: `ls -lh`, `wc -c`
- for structured data, iterate rows rather than materializing a full parse

The same applies to output: a command that prints a hundred megabytes will \\
push out the context you needed to use it.

"""

# Upstream's instance template, with the grading section inserted before the
# workflow and apply_patch added to the command examples. Everything else,
# including the submit sentinel, is upstream's.
INSTANCE_TEMPLATE = """\
Please solve this issue: {{task}}

You can execute bash commands and edit files to implement the necessary changes.

__GRADING_SECTION__
__SURVIVAL_SECTION__## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your \
changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Work through the requirement list above, testing edge cases and error paths
6. Submit your changes and finish your work by issuing the following command: \
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, \
you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action \
is executed in a new subshell.
   However, you can prefix any action with \
`MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment \
variables from files

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me \
check what files are in the current directory to get a better understanding of \
the codebase.

```mswea_bash_command
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```mswea_bash_command
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

__TOOLKIT_SECTION____APPLY_PATCH_SECTION__
### Edit files with sed__SED_CAVEAT__:

```mswea_bash_command
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py
```

### View file content:

```mswea_bash_command
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```mswea_bash_command
anything
```
"""


def build_instance_template(
    *, grading: bool, apply_patch: bool, toolkit: bool = True, survival: bool = True
) -> str:
    """Assemble the instance template for a variant.

    Sections are inserted or omitted rather than reworded, so an ablation
    isolates the section itself instead of a rewrite that happens to differ.
    """
    # Substituted rather than .format()-ed: the template is Jinja2, and
    # str.format collapses its `{{task}}` placeholders into `{task}`, which
    # would break the prompt inside the container rather than here.
    return (
        INSTANCE_TEMPLATE.replace(
            "__GRADING_SECTION__", GRADING_SECTION + "\n" if grading else ""
        )
        .replace("__SURVIVAL_SECTION__", SURVIVAL_SECTION if survival else "")
        .replace("__TOOLKIT_SECTION__", TOOLKIT_SECTION if toolkit else "")
        # The caveat only makes sense when the tool it points at is installed;
        # `stock` must not reference something that is not there.
        .replace(
            "__SED_CAVEAT__", " (last resort — prefer `crux edit`)" if toolkit else ""
        )
        .replace(
            "__APPLY_PATCH_SECTION__", APPLY_PATCH_SECTION + "\n" if apply_patch else ""
        )
    )

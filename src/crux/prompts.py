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

**Finish with `crux submit`, not with the echo.** It re-runs every bound check \
and only then emits the completion sentinel; if anything is open or has \
regressed it refuses and tells you what. That makes finishing a command that \
observes, rather than a judgement you make about yourself — which is the \
judgement this agent gets wrong most often.
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

```bash
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
FILE_TOOLS_SECTION = """\
### File tools — prefer these over cat/grep/sed

`crux read <path> [--offset N] [--limit N]` — print with line numbers, \
paginated. Output is capped and tells you what you have not seen and how to \
continue, so it will not flood your context. Also lists directories.

`crux grep <pattern> [path] [--include '*.py']` — regex search reporting \
`path:line: text`. Skips .git, node_modules and other noise.

`crux files '<glob>' [path]` — find files by name.

`crux edit <path>` — exact-text replacement, edits given as JSON on stdin:

```bash
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

"""

CHECKLIST_SECTION = """\
### Requirement checklist

The requirement checklist, with a check bound to each item:

```bash
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
## Turns are the scarcest resource

Every task has a wall-clock limit, and running out of it is the single largest
cause of failure -- larger than getting anything wrong. What that limit buys
you is generated tokens, not turns: on a self-hosted model a full evaluation
spent about 51,000 tokens of reasoning per task against a 600-second budget,
which is most of the budget before a single command runs.

So the budget is spent by thinking, and it is spent whether or not the thinking
was needed.

One action per reply is the format, but an action may chain commands with `&&`.
Use that whenever you do not need to read one result before deciding the next:

```mswea_bash_command
cd /app && ls -la && cat README.md 2>/dev/null | head -40 && python -V
```

Orient in one turn instead of four. Verify a fix and re-run the test in the
same turn. Split a chain only where the next command genuinely depends on what
you read.

### Look before you solve

The costliest habit is solving the whole problem in your head before touching
the machine. On puzzle-shaped tasks -- write a polyglot, find the shortest
regex, reverse this cipher -- that produced single turns of 24,000 reasoning
tokens, five minutes each, and several tasks died having run two commands.

Your first turn is for looking, not solving. Read the task files, list the
directory, run the existing tests, check what is installed. That costs seconds
and it replaces assumptions with facts -- and the facts are usually what makes
the problem smaller than it looked.

After that, think in the gaps between commands rather than all at once. If you
catch yourself reasoning at length about something you could simply run, stop
and run it. A wrong command you can see the output of is worth more than a
correct chain of reasoning you spent the budget producing.

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

__GRADING_SECTION____SURVIVAL_SECTION__## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Work through the requirement list above, testing edge cases and error paths
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
- Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

Example of a CORRECT response:
<example_response>
I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

[Makes bash tool call with {"command": "ls -la"} as arguments]
</example_response>

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

__FILE_TOOLS_SECTION____CHECKLIST_SECTION____APPLY_PATCH_SECTION__## Useful command examples

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed__SED_CAVEAT__:

{%- if system == "Darwin" -%}
<important>
You are on MacOS. For all the below examples, you need to use `sed -i ''` instead of `sed -i`.
</important>
{%- endif -%}

```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```bash
anything
```
"""


def build_instance_template(
    *,
    grading: bool,
    apply_patch: bool,
    toolkit: bool = True,
    file_tools: bool = True,
    survival: bool = True,
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
        .replace("__FILE_TOOLS_SECTION__", FILE_TOOLS_SECTION if file_tools else "")
        .replace("__CHECKLIST_SECTION__", CHECKLIST_SECTION if toolkit else "")
        # The caveat only makes sense when the tool it points at is installed;
        # `stock` must not reference something that is not there.
        .replace(
            "__SED_CAVEAT__",
            " (last resort — prefer `crux edit`)" if file_tools else "",
        )
        .replace(
            "__APPLY_PATCH_SECTION__", APPLY_PATCH_SECTION + "\n" if apply_patch else ""
        )
    )


# --- Terminus-path sections -------------------------------------------------
#
# Composed onto upstream's template at runtime rather than shipped as a copy of
# it. A frozen copy was in the tree and matched upstream exactly, which is the
# problem: it would go on matching a prompt harbor had since changed, silently.

# Measured against claude-code on the same 89 tasks, same model. Segments per
# command, counting `&&` and `;`:
#
#                    every command   the first command
#     claude-code         3.0              3.0
#     crux                2.0              1.0
#
# crux's opening move is one segment -- `ls` -- where the other agent's is
# three. Its first commands look like
#
#     ls -la /app && head -5 /app/data.csv && wc -l /app/data.csv
#     ls -la /app/ && file /app/a.out
#     python3 --version && which python3 && ls -la /app
#
# and that is the whole of the 1.85x step count and 2.8x tool calls measured
# between the two: fewer things per step means more steps. Unlike the file
# tools, this asks the model to adopt nothing new -- only to put what it was
# going to run anyway into one command.
#
# Probed before spending a run on it: five first commands from the real prompt
# against the real endpoint, thinking off.
#
#     default            [1, 1, 1, 1, 1]   median 1.0
#     batch_section=1    [3, 2, 1, 1, 1]   median 1.0
#
# Two of five move; the median does not. That is the same shape as the file
# tools -- uptake 0.3% to 4.3%, and a score of 81.1% against 78.4% at p=0.77 --
# so this is written down and left off rather than given an arm. A section
# nudges a strong prior and does not replace it.
# Read the artifact before describing it.
#
# From claude-code's trajectories: on `chess-best-move` its first move is
# `Read /app/chess_board.png`; on the four tasks it still wins its openers are
#
#     ls -la /app && head -5 /app/data.csv && wc -l /app/data.csv
#     ls -la /app/ && file /app/a.out
#
# Every one of them touches the data on the first command. crux opens with a
# bare `ls -la /app` in 35 of 39 runs and does not read anything until later --
# and in the file-tools arms the first structured read lands at 41% of the way
# through the trajectory.
#
# Probed before spending a run: six first commands, counting whether any of
# them opens a file rather than listing one.
#
#     default          [0, 0, 0, 0, 0, 0]   all `ls -la /app`
#     look_section=1   [0, 0, 0, 0, 0, 0]   all `ls -la /app`
#
# Nothing moved -- weaker even than the batching section, which moved one draw
# in five. Three prompt sections have now been written against this same habit
# and all three bounce off it. The opening `ls` is not something the prompt is
# competing with; it is what the model does when it has read nothing yet, and a
# paragraph asking otherwise is read after that decision is already made.
#
# Kept, off, and unmeasured. One minute of probing rather than six hours of GPU.
TERMINUS_LOOK_SECTION = """## Look at the thing itself, first

Your first command should show you the data, not just its name. A directory
listing tells you a file exists; the file tells you what the task is.

    ls -la /app && head -20 /app/input.csv && wc -l /app/input.csv
    ls -la /app && file /app/a.out && strings -n 8 /app/a.out | head

For a format you cannot read as text -- an image, a binary, a video -- open it
the cheapest way that returns a fact: dimensions, a header, a byte count, the
first frame. One such fact usually decides what the whole task is.
"""

TERMINUS_BATCH_SECTION = """## One command, several answers

Chain your probes. A turn costs a model call whatever it carries, so a step
that answers one question wastes the other three it could have asked:

    ls -la /app && head -5 /app/data.csv && wc -l /app/data.csv
    python3 --version && which python3 && pip list 2>/dev/null | head

This matters most on the first command. Opening with `ls` alone buys one fact
and a whole turn; opening with a chain buys the shape of the task.

Keep them separate when a later part depends on reading an earlier one, or when
one part is slow and you want its output before deciding. Chaining is for the
questions you already know you will ask.
"""

TERMINUS_HARNESS_SECTION = """## Use what the task already gives you

Read the task for anything that checks the work -- a script it names, a test
file, a make target, a sample input with a known answer. Run it early, and run
it again before you finish.

Read side by side with a run that solved `mailman` in 15 steps where this agent
took 512: the task said "an /app/eval.py script is provided to help
iterations". That run read it as its second action and executed it twice, once
to check the build and once at the end. This agent read it once and never ran
it. The grader is not visible to you, but a checker the task ships is the
closest thing to it you will get, and running it is cheaper than any amount of
reasoning about whether the work is right.

Capture the exit status of anything that matters:

    <command> 2>&1; echo "exit=$?"

Output text is not a result. A command that printed something plausible and
returned 1 looks, in the terminal, exactly like one that worked. That same
comparison run wrote `echo "exit=$?"` after every state-changing command; this
one wrote it zero times in 512 steps and inferred success from prose.

**In an existing repository, run its suite the way the repository runs it.**
`crux tests` reads that out of the CI workflow, tox.ini, the Makefile or
CONTRIBUTING and prints it. Use what it finds, flags and all.

The reason is measured. On a finished run of 89 real GitHub issues, the
failures were not trials that skipped testing: 88 of 89 ran the repository's
own suite, a median of 7 times when they solved and 10 when they failed, and
for 11 of the 16 failures the module holding the broken test was one the trial
had run. One of them ran a 1243-test sweep, saw `Ran 1243 tests OK`, and was
still failed on a module inside that sweep.

What differed was the command. The grader ran

    ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 <modules>

and of eight failures none passed `--parallel` and two passed `--settings`. A
different settings module and a different isolation policy make the same tests
not the same tests, and a pass under yours is not the pass that will be scored.

This is worth doing first because it is the one thing here you do not have to
judge. Whether your work is right is a judgement; how this repository runs its
tests is written down in it.
"""


TERMINUS_SCORING_SECTION = """## Scoring is all or nothing

A task scores 1.0 only if every requirement holds; 0.9 of the work scores
zero. Before you finish, enumerate the requirements from the task description
as a checklist and prove each one with a command whose output you can see --
not from memory of having done it earlier.

Measured here: across eight graded tasks the suites ran 206 tests, this agent
passed 176 of them, and scored on none -- every task failed one to three tests
of its own set. 94 of 97 on one. 11 of 12 on another. The work was nearly
right and the finishing was not.

The same runs show why. The agent bound 35 checks of its own against those 206
tests: five checks for a 97-test suite, one for a thirteen. A checklist
covering a sixth of what is graded will pass and mean nothing. One check per
stated requirement is the floor, not the target, because each requirement is
graded several times over:

  - the ordinary case, and the empty, zero, single-element and boundary ones
  - the numeric tolerance the task actually names, not a value that looks close
  - the failure path: malformed input rejected, the exit code it must return
  - what must NOT change: state left untouched, ordering preserved

The gap is almost never a requirement nobody attempted. It is one met earlier
and quietly broken since, one assumed rather than checked, or an edge of a
requirement that was only ever checked down the middle.
"""

TERMINUS_SUBMIT_SECTION = """## Finish by verifying, not by deciding

Run `crux submit` as your last command instead of declaring completion
yourself. It re-runs every check you bound with `crux todo add ... --verify`,
and only reports success if all of them pass; if anything is open or has
regressed it says so and you can keep working.

    crux todo add "rejects malformed input" --verify "./filter < bad.txt; test $? -ne 0"
    crux todo done 1        # re-runs the check, refuses to close if it fails
    crux submit             # re-runs everything, then finishes

Before running it, state plainly which behaviours the task asks for that you
have bound no check for. "all 3 item(s) verified" describes your checklist, not
the task -- it was reported verbatim on tasks that then failed their suites. If
that list is not empty, bind those checks rather than finish.

Only set <task_complete>true</task_complete> after `crux submit` has confirmed
it. Declaring completion is the judgement this agent gets wrong most often.
"""

# Upstream ends every template with this; the crux sections go before it so the
# task and the live terminal stay last, where the model reads them.
_TERMINUS_FOOTER = "Task Description:"



# The three sections, addressable by name, for front-ends that append them to
# another agent's system prompt. Kept here rather than beside the harbor adapter
# so that the interactive CLI does not need the benchmark harness installed --
# the same separation pi has between its agent core and its eval package.

TERMINUS_EDIT_SECTION = """## Changing part of a file

`apply_patch` is on PATH. Use it to change an existing file:

    apply_patch <<'PATCH'
    *** Begin Patch
    *** Update File: path/to/file.py
    @@
     unchanged context line
    -the line as it is now
    +the line as it should be
     unchanged context line
    *** End Patch
    PATCH

It also takes `*** Add File:`, `*** Delete File:` and `*** Move to:`. It fails
loudly when the context does not match, which is the point: the edit either
lands where you meant or it does not land.

Measured on this setup, across an 89-task run this agent made 845 file
modifications, 791 of them a heredoc that retyped the whole file and 54 a
`sed -i`. Both are worse than they look. A heredoc silently discards every
line you did not retype, so a file you meant to adjust comes back missing
whatever you forgot. A `sed -i` edits every line that matches the pattern, not
the line you had in mind, and reports nothing when it matches four.

Write a whole file with a heredoc when you are creating it. To change one that
already exists, patch it."""

# Two lines lifted from Claude Code's own system prompt
# (`src/constants/prompts.ts`, the `# Doing tasks` section), because each one
# names a failure measured in pi on this benchmark rather than a mechanism
# reasoned out here.
#
# **Verify before reporting done.** 18 of pi's 24 failed Terminal-Bench trials
# ended with `agent_settled` -- the agent deciding it was finished -- and 69 of
# 73 trials never ran a command that looks like a check at all. Claude Code
# tells the model to run the test, and to say so when it cannot.
#
# **Diagnose before abandoning.** On the tasks it failed, stock pi issued a
# median of 7 tool calls across 5 turns and stopped; Claude Code issued 40.
# Its prompt asks for a focused fix after reading the error, and explicitly
# rules out both blind retries and giving up after one failure.
#
# Claude Code's section runs to some eighty lines and most of it is about being
# Claude Code -- slash commands, feedback channels, its own tool names. These
# two are the ones with a measurement behind them here, so these two are what
# gets ported. Everything else in that section is a guess about this benchmark
# until something says otherwise.
CC_FINISH_SECTION = """\
## Finishing

Before reporting a task complete, verify it actually works: run the test,
execute the script, check the output. Minimum complexity means no gold-plating,
not skipping the finish line. If you cannot verify -- no test exists, the code
cannot be run here -- say so explicitly rather than claiming success.

If an approach fails, diagnose why before switching tactics: read the error,
check your assumptions, try a focused fix. Do not retry the identical action
blindly, and do not abandon a viable approach after a single failure. The task
is not over because the first thing you tried did not work.
"""

# Two directives from Claude Code's own `# Doing tasks` system-prompt section
# (`src/constants/prompts.ts`, getSimpleDoingTasksSection), carried over because
# they name the two failures measured in pi on this benchmark and nothing in
# pi's prompt addresses either.
#
#   - Eighteen of pi's twenty-four failed Terminal-Bench 2.1 trials ended with
#     the agent declaring itself done and the verifier disagreeing.
#   - Four more stopped after two to four actions.
#
# Claude Code answers the first with "verify it actually works: run the test,
# execute the script, check the output ... if you can't verify, say so
# explicitly rather than claiming success", and the second with "don't retry
# the identical action blindly, but don't abandon a viable approach after a
# single failure either".
#
# Worth being precise about what this is evidence for. Eight prompt sections
# written for this project were measured on Terminal-Bench and every one landed
# inside the noise, so the prior on a prompt section is poor. What is different
# here is not the wording but the provenance: this is the text a scaffold that
# leads by 17 points on the same model actually ships, and it is the largest
# untested difference between the two -- every arm so far ran pi with no
# appended prompt at all.
CC_DOING_TASKS_SECTION = """\
# Finishing and persisting

Before reporting a task complete, verify it actually works: run the test,
execute the script, check the output. If you cannot verify -- no test exists,
the code cannot be run here -- say so explicitly rather than claiming success.

If an approach fails, diagnose why before switching tactics: read the error,
check your assumptions, try a focused fix. Do not retry the identical action
blindly, and do not abandon a viable approach after a single failure.

Do not propose or make changes to code you have not read. If a file is to be
modified, read it first.
"""


# The task-solving substance of Claude Code's `# Doing tasks` section
# (`src/constants/prompts.ts`, getSimpleDoingTasksSection), ported at its real
# size rather than as two lines.
#
# Why size is the point. pi's core system prompt is 4,169 characters. Claude
# Code's is 27,960, of which roughly 12,600 could bear on a benchmark trial at
# all -- the rest is communication style for a human who is watching, a
# confirm-before-risky-actions policy that a `bypassPermissions` trial has
# nobody to honour, MCP and deferred-tool discovery, and an autonomous-tick
# mode that is feature-gated off. So the real ratio of task guidance is about
# three to one, and `getSimpleDoingTasksSection` is 7,145 characters of it.
#
# Nine prompt sections have been measured in this project and all nine landed
# inside the noise. Every one was a few hundred characters. That makes nine null
# results weak evidence about prompts in general: none of them tested a prompt
# at the size of the one being compared against.
#
# Dropped from the port, deliberately: the product bullets (/help, /issue,
# /share, the feedback channel), the time-estimate and knowledge-cutoff rules,
# the accountability-and-tone bullet, and the "default to helping" safety
# framing. None of them can move a benchmark trial, and carrying them would
# make this a test of length rather than of content.
CC_DOING_FULL_SECTION = """\
# Doing tasks

- These are software engineering tasks: fixing bugs, adding functionality,
  refactoring, explaining code. When an instruction is unclear or generic, read
  it in the context of the working directory and the code in it. If asked to
  rename a method, find the method in the code and change the code -- do not
  answer with the new name.
- You are highly capable, and ambitious tasks are often within reach. Do not
  talk yourself out of a task because it looks large.
- Do not propose or make changes to code you have not read. If a file is to be
  modified, read it first, and understand the existing code before changing it.
- Do not create files unless they are necessary for the goal. Prefer editing an
  existing file to creating a new one.
- If an approach fails, diagnose why before switching tactics: read the error,
  check your assumptions, try a focused fix. Do not retry the identical action
  blindly, and do not abandon a viable approach after a single failure.
- Do not introduce security holes -- command injection, path traversal, SQL
  injection. If you notice you have written insecure code, fix it immediately.

## How much to change

- Do not add features, refactor, or make improvements beyond what was asked. A
  bug fix does not need the surrounding code cleaned up. A simple feature does
  not need extra configurability.
- Do not add error handling, fallbacks, or validation for cases that cannot
  happen. Trust internal code and framework guarantees. Validate at system
  boundaries only.
- Do not create helpers, utilities, or abstractions for a one-time operation,
  and do not design for hypothetical future requirements. Three similar lines
  are better than a premature abstraction. Equally, no half-finished
  implementations: the right amount of complexity is what the task requires.
- Default to writing no comments. Add one only where the reason is non-obvious:
  a hidden constraint, a subtle invariant, a workaround for a specific bug. Do
  not explain what the code does -- names already do that.
- Do not remove existing comments unless you are removing the code they
  describe or you know they are wrong. A comment that looks pointless may
  encode a constraint from a past bug.
- Avoid backwards-compatibility shims, renamed unused variables, or
  "// removed" markers. If something is certainly unused, delete it.

## Finishing

- Before reporting a task complete, verify it actually works: run the test,
  execute the script, check the output. Minimum complexity means no
  gold-plating, not skipping the finish line. If you cannot verify -- no test
  exists, the code cannot be run here -- say so explicitly rather than claiming
  success.
- Report outcomes faithfully. If tests fail, say so with the output. If a
  verification step was not run, say that rather than implying it succeeded.
  Never claim all tests pass when the output shows failures, and never simplify
  a failing check to manufacture a green result. Equally, when a check did pass,
  state it plainly rather than hedging a confirmed result.
"""


PROMPT_SECTIONS = {
    "scoring": TERMINUS_SCORING_SECTION,
    "harness": TERMINUS_HARNESS_SECTION,
    "edit": TERMINUS_EDIT_SECTION,
    "submit": TERMINUS_SUBMIT_SECTION,
    "finish": CC_FINISH_SECTION,
    "doing": CC_DOING_TASKS_SECTION,
    "doing_full": CC_DOING_FULL_SECTION,
}
_SECTION_ORDER = ("scoring", "harness", "submit", "finish", "doing", "doing_full")


def build_sections(names) -> str:
    """The requested sections, in a fixed order, as one appended block.

    Order is fixed rather than following the caller so two runs asking for the
    same set produce the same bytes; otherwise an A/B could differ by section
    order with nothing recording it. An unknown name raises, because a typo that
    quietly runs the control arm while claiming the treatment is the failure
    this project keeps finding elsewhere.
    """
    wanted = set(names)
    unknown = wanted - set(PROMPT_SECTIONS)
    if unknown:
        raise ValueError(f"unknown prompt section(s): {sorted(unknown)}")
    return "\n\n".join(
        PROMPT_SECTIONS[n].strip() for n in _SECTION_ORDER if n in wanted
    )

def _escape_braces(text: str) -> str:
    """Make literal braces survive a later `str.format`.

    Sections that show JSON to the model contain `{` and `}` that are content,
    not placeholders. Anything inserted into a template upstream will format
    has to double them.
    """
    return text.replace("{", "{{").replace("}", "}}")


def build_terminus_template(
    upstream: str,
    scoring: bool = True,
    submit: bool = True,
    harness: bool = True,
    edit: bool = True,
    file_tools: bool = False,
    batch: bool = False,
    look: bool = False,
) -> str:
    """Insert the crux sections into upstream's Terminus template.

    `file_tools` defaults off, and the reason recorded for that was not true.
    It said the model did not take the tools up -- "over 206 tool calls,
    read/grep/edit/write together accounted for under 2%" -- but this function
    had no `file_tools` parameter at all, so the Terminus prompt never named
    `crux read`, `crux grep`, `crux files`, `crux edit` or `crux write`. The
    binaries were installed and never mentioned. 0% uptake was a statement
    about the prompt.

    What that costs is measurable. Across both finished runs, 70-83% of every
    command crux issues is a file operation and 0-1% of them go through a
    structured tool; reading a file alone is 41-49% of all commands, one slice
    at a time. claude-code finishes the same 89 tasks in 4,013 tool calls
    against 6,826, for the same score.

    Still off by default: it is now reachable and unmeasured, which is an arm,
    not a change to the one being scored.

    `submit` is separable from `scoring` because they are not the same claim.
    The scoring section states a fact about the grader -- partial work scores
    zero -- which nothing has contradicted. The submit section directs the model
    through `crux submit`, and across a full run that gate passed on all nine
    wrong answers and caught none of them, so it needs to be testable on its
    own.
    """
    head, sep, tail = upstream.partition(_TERMINUS_FOOTER)
    if not sep:
        raise ValueError("upstream Terminus template has no task-description footer")
    parts = [head.rstrip(), ""]
    if scoring:
        parts += [TERMINUS_SCORING_SECTION.strip(), ""]
    if submit:
        parts += [TERMINUS_SUBMIT_SECTION.strip(), ""]
    if harness:
        parts += [TERMINUS_HARNESS_SECTION.strip(), ""]
    if batch:
        parts += [TERMINUS_BATCH_SECTION.strip(), ""]
    if look:
        parts += [TERMINUS_LOOK_SECTION.strip(), ""]
    if edit:
        parts += [TERMINUS_EDIT_SECTION.strip(), ""]
    if file_tools:
        # Braces doubled. Upstream runs `.format(instruction=..., terminal_state=...)`
        # over this template, and the `crux edit` example in this section is a
        # JSON object -- `{"edits": [...]}` -- which format() reads as a field
        # name and raises KeyError('"edits"') on. That took out 16 of 89 trials
        # on the first run of this arm.
        #
        # The mini-swe-agent path does not need this: it substitutes with
        # str.replace and never formats.
        parts += [_escape_braces(FILE_TOOLS_SECTION.strip()), ""]
    parts += [sep + tail]
    return "\n".join(parts)

"""SWE-Atlas scores by reading a file pi never wrote.

Its instruction ends with "write your complete final answer to
/logs/agent/answer.txt wrapped in <<FINAL_ANSWER>> tags". Terminus complies --
its loop is typing into a terminal, so writing a file is the natural move, and
its trajectories mention the path eighteen times. pi in `--print` mode answers
the person who asked: across 31 SWE-Atlas trials it read the instruction,
mentioned the path three times, and never issued one tool call touching it.
Every trial scored zero on `No answer file at /logs/agent/answer.txt, scoring 0`.
"""

import json
from pathlib import Path

from crux.pi_agent import _answer_path, _final_answer

ATLAS = (
    "Consider the following question about the codebase...\n"
    "5. When you are confident in your answer, write your complete final answer "
    "to /logs/agent/answer.txt wrapped in <<FINAL_ANSWER>> tags. Do not only "
    "print the answer in chat."
)


def test_it_reads_the_path_out_of_the_instruction():
    assert _answer_path(ATLAS) == "/logs/agent/answer.txt"


def test_a_task_that_asks_for_no_file_gets_nothing():
    """Inert on Terminal-Bench, which asks for no answer file."""
    assert _answer_path("Fix the failing test in /app/solver.py") is None
    assert _answer_path("") is None


def _log(tmp_path, messages):
    p = tmp_path / "pi.txt"
    with p.open("w", encoding="utf-8") as fh:
        for m in messages:
            fh.write(json.dumps({"type": "message_end", "message": m}) + "\n")
    return p


def test_it_prefers_what_the_agent_marked_as_final():
    log = _log(
        Path.cwd() if False else _tmp(),
        [
            {"role": "assistant", "content": [{"type": "text", "text": "thinking out loud"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Here you go. <<FINAL_ANSWER>>42 tables<</FINAL_ANSWER>> hope that helps"}]},
        ],
    )
    assert _final_answer(log) == "42 tables"


def test_without_markers_it_takes_the_last_thing_said():
    """The answer was given, to the wrong channel."""
    log = _log(_tmp(), [
        {"role": "assistant", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "the answer is 7"}]},
    ])
    assert _final_answer(log) == "the answer is 7"


def test_a_run_that_said_nothing_delivers_nothing():
    """Better an empty file check than a file holding reasoning noise."""
    log = _log(_tmp(), [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]},
    ])
    assert _final_answer(log) == ""
    assert _final_answer(Path("/nonexistent/pi.txt")) == ""


def test_tool_results_are_not_mistaken_for_the_answer():
    log = _log(_tmp(), [
        {"role": "assistant", "content": [{"type": "text", "text": "the answer is 7"}]},
        {"role": "toolResult", "toolName": "bash", "content": [{"type": "text", "text": "ls output"}]},
    ])
    assert _final_answer(log) == "the answer is 7"


def _tmp():
    import tempfile
    return Path(tempfile.mkdtemp())

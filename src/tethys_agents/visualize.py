from __future__ import annotations

import shutil
import subprocess
import textwrap
from typing import Any


_LABELS = {
    "user": "User",
    "round": "Round",
    "tool_call": "Tool",
    "tool_result": "Result",
    "answer": "Answer",
}


def _wrap(text: str, width: int = 30, max_lines: int = 3) -> str:
    """Wrap and truncate a multi-line label for a graph node."""
    text = str(text).strip().replace('"', "'")
    if not text:
        return ""
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][:width] + "..."]
    return "\\n".join(lines)


def trace_to_dot(trace: list[dict[str, Any]]) -> str:
    """Return the DOT source for ``trace`` as a string.

    Pipe the result through any DOT renderer (``dot``, ``graph-easy``, etc.).
    """
    lines = ["digraph trace {", '  rankdir=TB; node [shape=box];']
    prev = None
    for i, ev in enumerate(trace):
        ev_type = ev.get("type")
        if ev_type == "user":
            nid = "user"
            lines.append(f'  {nid} [label="{_LABELS["user"]}: {_wrap(ev.get("text", ""))}"];')
        elif ev_type == "round":
            nid = f"r{ev.get('round', i)}"
            lines.append(f'  {nid} [label="{_LABELS["round"]} {ev.get("round", "?")}"];')
        elif ev_type == "tool_call":
            nid = f"t{i}"
            tool = ev.get("tool", "?")
            args = _wrap(str(ev.get("arguments", {})), width=32, max_lines=2)
            lines.append(f'  {nid} [label="{_LABELS["tool_call"]}: {tool}\\n{args}"];')
            rid = f"res{i}"
            result = _wrap(str(ev.get("result", "")), width=36, max_lines=3)
            lines.append(f'  {rid} [label="{_LABELS["tool_result"]}: {result}"];')
            if prev is not None:
                lines.append(f"  {prev} -> {nid};")
            lines.append(f"  {nid} -> {rid};")
            prev = rid
            continue
        elif ev_type == "answer":
            nid = "answer"
            label = "Max rounds" if ev.get("max_rounds_reached") else _LABELS["answer"]
            lines.append(f'  {nid} [label="{label}: {_wrap(ev.get("text", ""))}"];')
        else:
            continue
        if prev is not None:
            lines.append(f"  {prev} -> {nid};")
        prev = nid
    lines.append("}")
    return "\n".join(lines)


def print_trace_ascii(trace: list[dict[str, Any]]) -> None:
    """Print ``trace`` to stdout as a Unicode box-art tree.

    Pipes the DOT source through ``graph-easy --as=boxart`` when available.
    Falls back to the raw DOT source if the binary isn't on PATH.
    """
    dot = trace_to_dot(trace)
    if shutil.which("graph-easy"):
        try:
            result = subprocess.run(
                ["graph-easy", "--as=boxart"],
                input=dot,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                print(result.stdout)
                return
        except (subprocess.SubprocessError, OSError):
            pass
    print(dot)

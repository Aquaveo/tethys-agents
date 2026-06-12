"""Render a :class:`ReactAgent` trace as a DOT/Graphviz graph.

The :attr:`ReactAgent.trace` attribute accumulates a list of dict events
during ``run()``:

* ``{"type": "user", "text": "<prompt>"}``  — at the start.
* ``{"type": "round", "round": N, "model_text": "...", "had_tool_call": bool}``
  — once per LLM round.
* ``{"type": "tool_call", "tool": "name", "arguments": {...}, "result": "..."}``
  — once per tool dispatch (including the three error classes).
* ``{"type": "answer", "text": "...", "max_rounds_reached"?: True}`` — at exit.

:func:`trace_to_svg` renders that list as a top-to-bottom flowchart SVG
suitable for embedding in a web UI via ``dangerouslySetInnerHTML``.

System requirement: the Graphviz ``dot`` binary must be on ``PATH``. The
Python ``graphviz`` package is just a wrapper:

* Debian / Ubuntu: ``sudo apt-get install graphviz``
* macOS: ``brew install graphviz``
* Fedora: ``sudo dnf install graphviz``

If the binary is missing :func:`trace_to_svg` raises ``ExecutableNotFound``.
Hosts should catch that and fall back to a text-only timeline so the
chat endpoint never fails just because graphviz isn't installed.
"""

from __future__ import annotations

import textwrap
from typing import Any


_DEFAULT_NODE_ATTRS = {
    "shape": "box",
    "style": "rounded,filled",
    "fontname": "Helvetica",
    "fontsize": "11",
    "margin": "0.15,0.08",
}

_COLORS = {
    "user": "#cfe2ff",
    "round": "#d1e7dd",
    "tool_call": "#fff3cd",
    "tool_result_ok": "#e2e3e5",
    "tool_result_err": "#f8d7da",
    "answer": "#d1e7dd",
}


def _wrap(text: str, width: int = 28, max_lines: int = 4) -> str:
    """Wrap and truncate a multi-line label for a graph node."""
    text = str(text).strip()
    if not text:
        return ""
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][:width] + "…"]
    return "\\n".join(lines)


def trace_to_dot(trace: list[dict[str, Any]]):
    """Build a :class:`graphviz.Digraph` representing ``trace``.

    Args:
        trace: A list of trace dicts as accumulated by
            :class:`tethys_agents.react_agent.ReactAgent`.

    Returns:
        A :class:`graphviz.Digraph` ready to ``.pipe()`` or ``.render()``.

    Raises:
        ImportError: if the ``graphviz`` Python package isn't installed.
    """
    from graphviz import Digraph

    dot = Digraph(
        format="svg",
        graph_attr={
            "rankdir": "TB",
            # Default curved splines (ortho strips edge labels with a warning)
            "nodesep": "0.35",
            "ranksep": "0.4",
            "bgcolor": "transparent",
        },
    )
    dot.attr("node", **_DEFAULT_NODE_ATTRS)
    dot.attr("edge", fontname="Helvetica", fontsize="9", color="#6c757d")

    prev = None
    for i, ev in enumerate(trace):
        ev_type = ev.get("type")

        if ev_type == "user":
            nid = "user"
            dot.node(
                nid,
                f"👤 User\\n{_wrap(ev.get('text', ''))}",
                shape="oval",
                fillcolor=_COLORS["user"],
            )

        elif ev_type == "round":
            nid = f"r{ev.get('round', i)}"
            label = f"🤖 Round {ev.get('round', '?')}"
            text_snip = _wrap(ev.get("model_text", ""), width=32, max_lines=2)
            if text_snip:
                label += f"\\n{text_snip}"
            dot.node(nid, label, fillcolor=_COLORS["round"])

        elif ev_type == "tool_call":
            nid = f"t{i}"
            tool_name = ev.get("tool", "?")
            args = ev.get("arguments", {})
            args_str = _wrap(str(args), width=32, max_lines=2)
            dot.node(
                nid,
                f"🔧 {tool_name}\\n{args_str}",
                fillcolor=_COLORS["tool_call"],
            )
            if prev is not None:
                dot.edge(prev, nid, label="calls")

            # Result is shown as a separate "note" node so the result
            # text can be longer than the tool node itself.
            rid = f"res{i}"
            result_text = str(ev.get("result", ""))
            is_error = result_text.startswith("Error")
            dot.node(
                rid,
                _wrap(result_text, width=36, max_lines=4),
                shape="note",
                fillcolor=(
                    _COLORS["tool_result_err"]
                    if is_error
                    else _COLORS["tool_result_ok"]
                ),
            )
            dot.edge(nid, rid, label="returns")
            prev = rid
            continue  # skip the prev = nid below

        elif ev_type == "answer":
            nid = "answer"
            label = "✅ Final answer"
            if ev.get("max_rounds_reached"):
                label = "⚠️ Max rounds reached"
            text_snip = _wrap(ev.get("text", ""), width=32, max_lines=3)
            if text_snip:
                label += f"\\n{text_snip}"
            dot.node(nid, label, fillcolor=_COLORS["answer"])

        else:
            # Unknown event type — skip rather than break the whole graph.
            continue

        if prev is not None:
            dot.edge(prev, nid)
        prev = nid

    return dot


def trace_to_svg(trace: list[dict[str, Any]]) -> str:
    """Render ``trace`` to an SVG string.

    Convenience wrapper around :func:`trace_to_dot`. The returned SVG is
    safe to embed in a browser via ``dangerouslySetInnerHTML`` *as long as*
    the trace was produced by the agent itself (not by user-typed strings
    interpolated unsafely into node labels — which is the case in
    :class:`tethys_agents.react_agent.ReactAgent`).

    Raises:
        graphviz.ExecutableNotFound: if the ``dot`` binary is missing.
            Callers should catch and fall back to a text timeline.
    """
    dot = trace_to_dot(trace)
    return dot.pipe(format="svg").decode("utf-8")

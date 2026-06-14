"""Convention-based runner discovery for chat-agent harnesses.

A *runner* wraps whatever agent shape a plugin chooses (single
ReactAgent, multi-agent Crew, future custom DAGs) behind a uniform
one-turn-at-a-time interface. A harness (e.g. ``tethysdash chat``)
queries the operator's configured plugin list, picks a runner by name,
and drives a REPL loop without knowing anything about the underlying
agent shape, backstories, or domain.

The discovery pattern mirrors :func:`tethys_agents.discover.discover`:
each plugin package contributes by exposing a ``RUNNERS`` dict at
``<package>.agent``::

    # in <plugin>/agent.py
    RUNNERS = {
        "single": MySingleRunner,
        "multi":  MyCrewRunner,
    }

The harness then calls ``discover_runners(["<plugin>"])`` and gets a
``{name: class}`` mapping it can index by the operator's chosen runner
name.

Why a Protocol and not an ABC: plugin authors who do not depend on
``tethys-agents`` should still be able to ship a duck-typed runner. The
runtime-checkable Protocol lets the harness verify shape without forcing
inheritance.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Protocol, Type, runtime_checkable

log = logging.getLogger(__name__)


@runtime_checkable
class AgentRunner(Protocol):
    """One-turn-at-a-time chat runner.

    Implementations wrap whatever agent shape they want - single
    ReactAgent, multi-agent Crew, custom DAG. The harness drives them
    through this minimal interface so it never needs to know the
    underlying shape.

    Contract:
        run_turn: process one user prompt; printing is the runner's job.
        reset:    discard any per-session state (history, accumulated
                  context) so the next ``run_turn`` starts fresh.
        describe: short human-readable label shown in the harness banner.
    """

    def run_turn(self, user_msg: str) -> None: ...

    def reset(self) -> None: ...

    def describe(self) -> str: ...


def discover_runners(packages: list[str]) -> Dict[str, Type[AgentRunner]]:
    """Import each ``<package>.agent`` and harvest its ``RUNNERS`` dict.

    Error semantics mirror :func:`tethys_agents.discover.discover`:

    * ``ModuleNotFoundError`` propagates - almost always a config bug
      (typo in the package list, package not installed in this env).
      The harness gets a loud failure instead of a silent zero-runner
      boot.
    * Other ``ImportError`` causes (module exists but its top-level
      imports failed) get logged at WARNING and the package is skipped,
      so one broken plugin doesn't kill the whole discovery pass.

    Duplicate runner names across packages: first-listed package wins,
    subsequent duplicates emit a WARNING. Operators resolve collisions
    by reordering the package list or renaming a runner in their plugin.
    """
    found: Dict[str, Type[AgentRunner]] = {}

    for pkg in packages:
        module_path = f"{pkg}.agent"
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError:
            raise
        except ImportError:
            log.warning("could not import %s; skipping", module_path, exc_info=True)
            continue

        runners = getattr(mod, "RUNNERS", None)
        if not isinstance(runners, dict):
            log.warning(
                "%s has no RUNNERS dict; skipping (got %s)",
                module_path, type(runners).__name__,
            )
            continue

        for name, cls in runners.items():
            if name in found:
                log.warning(
                    "runner %r already registered; ignoring duplicate from %s",
                    name, pkg,
                )
                continue
            found[name] = cls

    return found

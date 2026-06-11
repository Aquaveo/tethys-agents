"""Convention-based tool discovery.

For each package name a host passes in, this module imports ``<package>.tools``
and harvests two shapes of "tool":

1. Already-decorated :class:`Tool` instances (explicit opt-in). Plugin
   authors who already depend on ``tethys-agents`` can use ``@tool`` to
   register their callable.
2. Public, type-annotated functions defined inside ``<package>.tools``
   itself (zero-dep convention). The discoverer wraps them with the
   ``@tool`` decorator at discovery time.

The second path lets plugin packages contribute tools without listing
``tethys-agents`` as a dependency - the framework awareness lives at the
host (the process calling ``discover``), not at the plugin author.

Filtering rules for the convention path:

* Leading-underscore names are private helpers and skipped.
* Only functions (``inspect.isfunction``) qualify - classes, instances,
  and built-ins are skipped.
* The function must be defined IN the ``tools.py`` module
  (``__module__ == "<pkg>.tools"``). Imported callables that happen to
  appear in the module namespace are skipped.
* The function must carry type annotations - without them the generated
  tool schema would be empty and the LLM couldn't call it.

Duplicate tool names: first-listed wins; subsequent duplicates emit a
WARNING log so the operator can rename one or remove a package from
the host's tool-packages list.
"""

import importlib
import inspect
import logging
from typing import Dict, List, Tuple

from tethys_agents.tool import Tool, tool as _wrap_as_tool

log = logging.getLogger(__name__)


def discover(packages: List[str]) -> List[Tool]:
    """Import each package's ``tools`` module and harvest its tools.

    Args:
        packages: explicit list of trusted Python package import paths.
            Each entry's ``<package>.tools`` module is imported and walked.

    Returns:
        Deduplicated list of :class:`Tool` instances. On duplicate names,
        first-listed wins and subsequent duplicates emit a WARNING log.
    """
    seen: Dict[str, Tuple[Tool, str]] = {}

    for pkg in packages:
        module_path = f"{pkg}.tools"
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            log.warning("could not import %s; skipping", module_path, exc_info=True)
            continue

        for attr_name, value in vars(mod).items():
            # Path 1: already-decorated Tool instances pass through.
            if isinstance(value, Tool):
                _register(seen, value, pkg)
                continue

            # Path 2: auto-wrap public, type-annotated functions defined
            # in this module. See module docstring for filter rationale.
            if (
                not attr_name.startswith("_")
                and inspect.isfunction(value)
                and getattr(value, "__module__", None) == module_path
                and getattr(value, "__annotations__", None)
            ):
                _register(seen, _wrap_as_tool(value), pkg)

    return [tool for tool, _origin in seen.values()]


def _register(seen: Dict[str, Tuple[Tool, str]], tool: Tool, pkg: str) -> None:
    """Add a tool to the seen-dict; warn-and-skip on duplicate name."""
    if tool.name in seen:
        first_pkg = seen[tool.name][1]
        log.warning(
            "tool '%s' from package '%s' shadows earlier registration "
            "from '%s'; keeping first",
            tool.name, pkg, first_pkg,
        )
        return
    seen[tool.name] = (tool, pkg)

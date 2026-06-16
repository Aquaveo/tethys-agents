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
import pkgutil
from typing import Dict, List, Tuple

from tethys_agents.tool import Tool, tool as _wrap_as_tool

log = logging.getLogger(__name__)


def discover(packages: List[str]) -> List[Tool]:
    """Import each package's ``tools`` module and harvest its tools.

    Args:
        packages: explicit list of trusted Python package import paths.
            Each entry's ``<package>.tools`` module is imported and walked.
            If ``<package>.tools`` is itself a PACKAGE (i.e. a directory
            with submodules), every submodule under it is also walked.
            This is what lets workshop plugins ship a
            ``tools/primitives.py`` + ``tools/recipes.py`` layout and have
            functions in BOTH files auto-register without any per-file
            re-export in ``tools/__init__.py``.

    Returns:
        Deduplicated list of :class:`Tool` instances. On duplicate names,
        first-listed wins and subsequent duplicates emit a WARNING log.
    """
    seen: Dict[str, Tuple[Tool, str]] = {}
    # Functions we've already wrapped, keyed by id(fn). When the package
    # walks ``<pkg>.tools`` AND its submodules, the same function appears
    # multiple times (once per submodule that re-imports it). Without
    # this short-circuit, every primitive logs a spurious "shadows
    # earlier registration" WARNING per re-export. We only want to warn
    # on actual name collisions between DIFFERENT function objects.
    seen_fn_ids: set = set()

    for pkg in packages:
        module_path = f"{pkg}.tools"
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            # Two cases:
            #   1. The parent package <pkg> itself is missing - operator
            #      config bug (typo, not installed). Re-raise so the
            #      caller hears about it.
            #   2. The parent imports fine but has no <pkg>.tools
            #      submodule - this package just doesn't contribute
            #      tools. Skip silently so it can still contribute
            #      runners (or anything else discovered separately).
            if exc.name == module_path:
                log.debug("%s has no .tools submodule; skipping", pkg)
                continue
            raise
        except ImportError:
            # Module was found but its own top-level imports failed.
            # Skip + warn so one broken plugin doesn't kill the whole
            # discovery pass for the other packages in the list.
            log.warning("could not import %s; skipping", module_path, exc_info=True)
            continue

        # Walk the entry-point module PLUS every submodule under it. When
        # ``mod`` is a single tools.py file it has no ``__path__`` and the
        # list is just [mod]; when ``mod`` is a tools/ package the
        # pkgutil walk picks up every submodule so participant-authored
        # files (tools/recipes.py, tools/my_first_tool.py, etc.) work
        # without any explicit re-export from tools/__init__.py.
        modules_to_walk = [mod]
        if hasattr(mod, "__path__"):
            for _finder, submod_name, _is_pkg in pkgutil.iter_modules(mod.__path__):
                if submod_name.startswith("_"):
                    # Skip dunder modules like __pycache__. Submodules
                    # whose own name starts with _ are also treated as
                    # private - matches the leading-underscore-name
                    # convention for functions.
                    continue
                submod_path = f"{module_path}.{submod_name}"
                try:
                    modules_to_walk.append(importlib.import_module(submod_path))
                except ImportError:
                    log.warning(
                        "could not import submodule %s; skipping",
                        submod_path, exc_info=True,
                    )
                    continue

        for m in modules_to_walk:
            for attr_name, value in vars(m).items():
                # Path 1: already-decorated Tool instances pass through.
                if isinstance(value, Tool):
                    _register(seen, value, pkg)
                    continue

                # Path 2: auto-wrap public, type-annotated functions defined
                # in this module OR in any submodule of it. The submodule
                # check keeps the "no accidentally-imported callables"
                # protection - only things defined inside the <pkg>.tools
                # tree are accepted; imported callables from unrelated
                # packages still get filtered out.
                own_module = getattr(value, "__module__", None) or ""
                if (
                    not attr_name.startswith("_")
                    and inspect.isfunction(value)
                    and (
                        own_module == module_path
                        or own_module.startswith(module_path + ".")
                    )
                    and getattr(value, "__annotations__", None)
                ):
                    # Skip silently if we've already wrapped THIS function
                    # object (re-imported across submodules). Only fall
                    # through to _register for unseen functions, where a
                    # name collision IS worth warning about.
                    if id(value) in seen_fn_ids:
                        continue
                    seen_fn_ids.add(id(value))
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

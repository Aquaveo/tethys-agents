import importlib
import logging
from typing import List

from tethys_agents.tool import Tool

log = logging.getLogger(__name__)


# TODO: this is a very simple implementation that only looks for Tool instances in <pkg>.tools.
# WE NEED A BETTER WAY TO DISCOVER TOOLS, AND ALSO A BETTER WAY TO HANDLE DUPLICATE TOOL NAMES.

def discover(packages: List[str]) -> List[Tool]:
    """Import each package's `tools` module and harvest @tool callables.

    Args:
        packages: explicit list of trusted Python package import paths.
            Each entry's `<pkg>.tools` module is imported and walked.

    Returns:
        Deduplicated list of Tool instances; on duplicate names,
        first-listed wins.
    """
    seen = {}  # tool name -> (Tool, origin_pkg)

    for pkg in packages:
        module_path = f"{pkg}.tools"
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            log.warning("could not import %s; skipping", module_path, exc_info=True)
            continue

        for value in vars(mod).values():
            if not isinstance(value, Tool):
                continue
            if value.name in seen:
                first_pkg = seen[value.name][1]
                log.warning(
                    "tool '%s' from '%s' shadows earlier registration from '%s'; keeping first",
                    value.name, pkg, first_pkg,
                )
                continue
            seen[value.name] = (value, pkg)

    return [t for t, _ in seen.values()]
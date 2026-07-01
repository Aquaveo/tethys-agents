from typing import Any
from pydantic_ai import Agent


class AgentSession:
    """One agent + its deps + its history. Works with ANY entry-point-registered agent.

    The host constructs one of these per conversation (CLI REPL, chat sidebar
    session, whatever). No plugin-specific subclassing.
    """

    def __init__(self, agent: Agent):
        self._agent = agent
        self._deps: Any = None
        self._history: list = []

    async def start(self) -> None:
        deps_cls = getattr(self._agent, "deps_type", None)
        if deps_cls is not None:
            self._deps = deps_cls()   # deps_type() must work with no args

    async def turn(self, user_msg: str) -> Any:
        result = await self._agent.run(
            user_msg, deps=self._deps, message_history=self._history,
        )
        self._history = result.all_messages()
        return result.output

    def reset(self) -> None:
        self._history.clear()

    async def close(self) -> None:
        if self._deps is not None and hasattr(self._deps, "aclose"):
            await self._deps.aclose()
        self._deps = None
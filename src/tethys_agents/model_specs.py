"""Per-model output conventions.

Different LLMs emit different exit signals and reasoning conventions:

* qwen3 wraps internal reasoning in ``<think>...</think>`` and writes raw
  prose afterwards.
* llama3.2 writes raw prose with no wrapper.
* The strict ReAct prompt (see ``react_agent.REACT_SYSTEM_PROMPT``) asks for
  ``<response>...</response>`` tags, but most modern models ignore that
  instruction unless they were specifically trained on it.

This module holds a registry mapping model names to :class:`ModelSpec`
objects so :class:`ReactAgent` can route correctly per model. Users register
their own with :func:`register_model_spec`. The pattern intentionally
mirrors the :func:`tethys_agents.tool.tool` decorator: declarative,
registered, extensible.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    """Tag conventions for a specific LLM family.

    Attributes:
        thinking_tag: Name of the model's internal-reasoning wrapper, e.g.
            ``"think"`` for qwen3. ``None`` means the model doesn't have one.
        response_tag: Name of the explicit final-answer wrapper. ``None``
            means the model doesn't use one and the agent loop should exit
            when no tool call appears in a response.
        tool_call_tag: Name of the tool-invocation tag. Almost always
            ``"tool_call"`` since this is set by the agent's system prompt,
            but kept configurable for native-tool-call models.
        strip_thinking: Whether to remove the thinking block from the
            final answer before returning it. Almost always ``True``.
    """

    thinking_tag: Optional[str] = None
    response_tag: Optional[str] = None
    tool_call_tag: str = "tool_call"
    strip_thinking: bool = True


# Default spec: assumes the strict ReAct prompt's contract.
DEFAULT_SPEC = ModelSpec(response_tag="response")


# Built-in registry. Keys are matched first by exact name, then by prefix
# (so "qwen3:latest" and "qwen3:14b" both resolve to the "qwen3" entry).
MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3": ModelSpec(thinking_tag="think", response_tag=None),
    "qwen2": ModelSpec(response_tag=None),
    "llama3.2": ModelSpec(response_tag=None),
    "llama3.1": ModelSpec(response_tag=None),
    "llama3": ModelSpec(response_tag=None),
    "llama-3": ModelSpec(response_tag=None),
    "mistral": ModelSpec(response_tag=None),
    "gpt-4": ModelSpec(response_tag=None),
    "gpt-3.5": ModelSpec(response_tag=None),
    "claude": ModelSpec(response_tag=None),
    "deepseek-r1": ModelSpec(thinking_tag="think", response_tag=None),
    "default": DEFAULT_SPEC,
}


def get_model_spec(model: str) -> ModelSpec:
    """Resolve a model name to its spec. Exact match first, then prefix.

    Args:
        model: A model identifier such as ``"qwen3:latest"`` or ``"gpt-4o"``.

    Returns:
        The matching :class:`ModelSpec`. Falls back to :data:`DEFAULT_SPEC`
        if no prefix matches.
    """
    if model in MODEL_SPECS:
        return MODEL_SPECS[model]
    for prefix, spec in MODEL_SPECS.items():
        if prefix != "default" and model.startswith(prefix):
            return spec
    return MODEL_SPECS["default"]


def register_model_spec(model_name: str, spec: ModelSpec) -> None:
    """Register a custom model spec. Overwrites any existing entry.

    Args:
        model_name: Exact or prefix key to register against.
        spec: The :class:`ModelSpec` describing the model's conventions.
    """
    MODEL_SPECS[model_name] = spec

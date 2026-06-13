import json
from typing import Optional

from colorama import Fore
from dotenv import load_dotenv

from ._ollama import OllamaProvider
from .model_specs import ModelSpec, get_model_spec
from .tool import Tool
from .tool import validate_arguments
from .utils.completions import build_prompt_structure
from .utils.completions import ChatHistory
from .utils.completions import completions_create
from .utils.completions import update_chat_history
from .utils.extraction import extract_tag_content

load_dotenv()

BASE_SYSTEM_PROMPT = ""


REACT_SYSTEM_PROMPT = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don' make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools / actions:

<tools>
%s
</tools>

Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

You will be called again with this:

<observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>

Additional constraints:

- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
"""


class ReactAgent:
    """A ReAct-loop agent that interacts with tools per the LLM's tag conventions.

    The agent's exit condition is driven by a :class:`ModelSpec` (see
    :mod:`tethys_agents.model_specs`). For models that emit explicit
    ``<response>...</response>`` tags, the loop exits on the first response.
    For models that don't (qwen3, llama3.2, gpt-4, etc.), the loop exits as
    soon as the model emits text without asking for another tool call. For
    "thinking" models that wrap reasoning in ``<think>...</think>``, the
    block is stripped from the returned answer.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = BASE_SYSTEM_PROMPT,
        model_spec: Optional[ModelSpec] = None,
    ) -> None:
        self.client = OllamaProvider().client
        self.model = model
        self.model_spec = model_spec or get_model_spec(model)
        self.system_prompt = system_prompt
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        # Per-run trace: each entry is {"type": "round"|"tool_call", ...}.
        # The host can read self.trace after run() to render a workflow
        # graph or timeline in the UI. Cleared at the top of run().
        self.trace: list[dict] = []

    def add_tool_signatures(self) -> str:
        """Collect the function signatures of all available tools."""
        return "".join([tool.fn_signature for tool in self.tools])

    def _strip_thinking(self, completion: str) -> str:
        """Clean a model completion of reasoning blocks and response-tag artifacts.
        """
        text = completion
        # Step 1: strip the thinking block if the spec asks for it.
        if self.model_spec.thinking_tag and self.model_spec.strip_thinking:
            close_tag = f"</{self.model_spec.thinking_tag}>"
            if close_tag in text:
                text = text.rsplit(close_tag, 1)[-1]
        # Step 2: extract the content between <response>...</response> tags
        # if the model chose to wrap its answer that way.
        match = extract_tag_content(text, "response")
        if match.found:
            text = match.content[0]
        return text.strip()

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        """Validate, execute, and collect results for each tool call.

        Robust against the three classes of LLM mishap that would
        otherwise crash an entire request:

        * Malformed JSON inside ``<tool_call>`` — small models emit typos
          (``)`` for ``}``, missing commas, trailing junk).
        * Unknown tool name — model hallucinates a tool not in
          ``tools_dict`` (often from a system-prompt instruction that
          referenced a tool the host didn't actually register).
        * Tool body raising — plugin bug, network failure, permission
          error, etc.

        Each becomes an Observation the LLM can read on its next turn
        and self-correct. The agent never raises out of this method.
        """
        observations = {}
        for tool_call_str in tool_calls_content:
            call_id = len(observations)

            # 1. Parse the tool_call JSON.
            try:
                tool_call = json.loads(tool_call_str)
            except json.JSONDecodeError as exc:
                print(
                    Fore.RED
                    + f"\nMalformed <tool_call> JSON ({exc}). Raw payload:\n"
                    f"{tool_call_str!r}"
                )
                msg = (
                    f"Error: your <tool_call> body was not valid JSON ({exc}). "
                    "Re-emit the tool_call with strict JSON: "
                    '{"name": "...", "arguments": {...}, "id": <int>}.'
                )
                observations[call_id] = msg
                self.trace.append({
                    "type": "tool_call",
                    "tool": "<malformed>",
                    "arguments": {"raw": tool_call_str[:120]},
                    "result": msg,
                })
                continue

            # Honor the model-supplied id if present; fall back to ordinal.
            call_id = tool_call.get("id", call_id)
            tool_name = tool_call.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                msg = "Error: tool_call missing 'name' (string) field."
                observations[call_id] = msg
                self.trace.append({
                    "type": "tool_call",
                    "tool": "<missing name>",
                    "arguments": tool_call.get("arguments", {}),
                    "result": msg,
                })
                continue

            # 2. Resolve the tool.
            tool = self.tools_dict.get(tool_name)
            if tool is None:
                print(
                    Fore.RED
                    + f"\nUnknown tool '{tool_name}'. "
                    f"Available: {sorted(self.tools_dict.keys())}"
                )
                msg = (
                    f"Error: unknown tool '{tool_name}'. "
                    f"Available tools: {sorted(self.tools_dict.keys())}. "
                    "Use one of the available names exactly."
                )
                observations[call_id] = msg
                self.trace.append({
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": tool_call.get("arguments", {}),
                    "result": msg,
                })
                continue

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

            # 3. Validate and execute.
            try:
                validated_tool_call = validate_arguments(
                    tool_call, json.loads(tool.fn_signature)
                )
                print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")
                result = tool.run(**validated_tool_call["arguments"])
                print(Fore.GREEN + f"\nTool result: \n{result}")
            except Exception as exc:
                print(
                    Fore.RED
                    + f"\nTool '{tool_name}' raised {type(exc).__name__}: {exc}"
                )
                msg = (
                    f"Error: tool '{tool_name}' raised "
                    f"{type(exc).__name__}: {exc}. "
                    "Check your arguments match the tool's schema."
                )
                observations[call_id] = msg
                self.trace.append({
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": tool_call.get("arguments", {}),
                    "result": msg,
                })
                continue

            observations[call_id] = result
            self.trace.append({
                "type": "tool_call",
                "tool": tool_name,
                "arguments": validated_tool_call.get("arguments", {}),
                "result": result if isinstance(result, str) else repr(result),
            })

        return observations

    def run(
        self,
        user_msg: str,
        max_rounds: int = 10,
    ) -> str:
        """Run the ReAct loop until an exit condition fires or max_rounds hit."""
        # Reset the per-run trace so consecutive run() calls don't accumulate
        # across requests.
        self.trace = [{"type": "user", "text": user_msg}]
        user_prompt = build_prompt_structure(
            prompt=user_msg, role="user", tag="question"
        )
        if self.tools:
            self.system_prompt += (
                "\n" + REACT_SYSTEM_PROMPT % self.add_tool_signatures()
            )

        chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=self.system_prompt,
                    role="system",
                ),
                user_prompt,
            ]
        )

        if self.tools:
            for round_idx in range(max_rounds):
                completion = completions_create(self.client, chat_history, self.model)
                cleaned = self._strip_thinking(completion)

                # Exit path 1: explicit response tag (strict ReAct contract).
                if self.model_spec.response_tag:
                    response = extract_tag_content(
                        cleaned, self.model_spec.response_tag
                    )
                    if response.found:
                        return response.content[0]

                tool_calls = extract_tag_content(
                    cleaned, self.model_spec.tool_call_tag
                )

                # Exit path 2: no tool call this round AND there is content
                # => the model is done answering. Works for qwen3 / llama / gpt /
                # claude / any model that doesn't bother with <response> tags.
                if not tool_calls.found and cleaned.strip():
                    print(
                        Fore.CYAN
                        + f"\nExited at round {round_idx + 1} "
                        f"(model_spec={self.model_spec})"
                    )
                    self.trace.append({
                        "type": "answer",
                        "text": cleaned.strip(),
                    })
                    return cleaned.strip()

                thought = extract_tag_content(cleaned, "thought")
                update_chat_history(chat_history, cleaned, "assistant")

                if thought.found:
                    print(Fore.MAGENTA + f"\nThought: {thought.content[0]}")
                else:
                    print(
                        Fore.MAGENTA
                        + f"\nRound {round_idx + 1} completion "
                        f"(no <thought> tag):\n{cleaned}"
                    )

                # Record this round in the trace BEFORE running tools so the
                # round entry sits ahead of the tool_call entries it spawns.
                self.trace.append({
                    "type": "round",
                    "round": round_idx + 1,
                    "model_text": cleaned[:500],
                    "had_tool_call": tool_calls.found,
                })

                if tool_calls.found:
                    observations = self.process_tool_calls(tool_calls.content)
                    print(Fore.BLUE + f"\nObservations: {observations}")
                    update_chat_history(chat_history, f"{observations}", "user")

        # Final fall-through: max_rounds reached. Make one synthesis call.
        final = self._strip_thinking(
            completions_create(self.client, chat_history, self.model)
        ).strip()
        self.trace.append({"type": "answer", "text": final, "max_rounds_reached": True})
        return final

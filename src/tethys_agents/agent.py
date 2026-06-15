from textwrap import dedent

from .crew import Crew
from .react_agent import ReactAgent
from .tool import Tool


class Agent:
    """
    Represents an AI agent that can work as part of a team to complete tasks.

    This class implements an agent with dependencies, context handling, and task execution capabilities.
    It can be used in a multi-agent system where agents collaborate to solve complex problems.

    Attributes:
        name (str): The name of the agent.
        backstory (str): The backstory or background of the agent.
        task_description (str): A description of the task assigned to the agent.
        task_expected_output (str): The expected format or content of the task output.
        react_agent (ReactAgent): An instance of ReactAgent used for generating responses.
        dependencies (list[Agent]): A list of Agent instances that this agent depends on.
        dependents (list[Agent]): A list of Agent instances that depend on this agent.
        context (str): Accumulated context information from other agents.

    Args:
        name (str): The name of the agent.
        backstory (str): The backstory or background of the agent.
        task_description (str): A description of the task assigned to the agent.
        task_expected_output (str, optional): The expected format or content of the task output. Defaults to "".
        tools (list[Tool] | None, optional): A list of Tool instances available to the agent. Defaults to None.
        llm (str, optional): The name of the language model to use. Defaults to "llama-3.3-70b-versatile".
    """

    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str,
        task_expected_output: str = "",
        tools: list[Tool] | None = None,
        llm: str = "llama-3.3-70b-versatile",
    ):
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.react_agent = ReactAgent(
            model=llm, system_prompt=self.backstory, tools=tools or []
        )

        self.dependencies: list[Agent] = []  # Agents that this agent depends on
        self.dependents: list[Agent] = []  # Agents that depend on this agent

        self.context = ""

        # Automatically register this agent to the active Crew context if one exists
        Crew.register_agent(self)

    def __repr__(self):
        return f"{self.name}"

    def __rshift__(self, other):
        """
        Defines the '>>' operator. This operator is used to indicate agent dependency.

        Args:
            other (Agent): The agent that depends on this agent.
        """
        self.add_dependent(other)
        return other  # Allow chaining

    def __lshift__(self, other):
        """
        Defines the '<<' operator to indicate agent dependency in reverse.

        Args:
            other (Agent): The agent that this agent depends on.

        Returns:
            Agent: The `other` agent to allow for chaining.
        """
        self.add_dependency(other)
        return other  # Allow chaining

    def __rrshift__(self, other):
        """
        Defines the '<<' operator.This operator is used to indicate agent dependency.

        Args:
            other (Agent): The agent that this agent depends on.
        """
        self.add_dependency(other)
        return self  # Allow chaining

    def __rlshift__(self, other):
        """
        Defines the '<<' operator when evaluated from right to left.
        This operator is used to indicate agent dependency in the normal order.

        Args:
            other (Agent): The agent that depends on this agent.

        Returns:
            Agent: The current agent (self) to allow for chaining.
        """
        self.add_dependent(other)
        return self  # Allow chaining

    def add_dependency(self, other):
        """
        Adds a dependency to this agent.

        Args:
            other (Agent | list[Agent]): The agent(s) that this agent depends on.

        Raises:
            TypeError: If the dependency is not an Agent or a list of Agents.
        """
        if isinstance(other, Agent):
            self.dependencies.append(other)
            other.dependents.append(self)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                self.dependencies.append(item)
                item.dependents.append(self)
        else:
            raise TypeError("The dependency must be an instance or list of Agent.")

    def add_dependent(self, other):
        """
        Adds a dependent to this agent.

        Args:
            other (Agent | list[Agent]): The agent(s) that depend on this agent.

        Raises:
            TypeError: If the dependent is not an Agent or a list of Agents.
        """
        if isinstance(other, Agent):
            other.dependencies.append(self)
            self.dependents.append(other)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                item.dependencies.append(self)
                self.dependents.append(item)
        else:
            raise TypeError("The dependent must be an instance or list of Agent.")

    def receive_context(self, input_data):
        """
        Receives and stores context information from other agents.

        Args:
            input_data (str): The context information to be added.
        """
        self.context += f"{self.name} received context: \n{input_data}"

    def create_prompt(self):
        """
        Creates a prompt for the agent based on its task description, expected output, and context.

        The template is **action-oriented**: it tells the model to USE its
        tools, not to describe what they would produce. This matters because
        the wrapped string becomes ``user_msg`` for the underlying
        ReactAgent, which the model treats as more specific than the
        system_prompt. A narrative template ("create the best possible
        response") here will silently override any "call tools" directive
        in the agent's backstory, and the model exits round 1 with prose
        instead of tool calls. See tethys_agents/react_agent.py:276 for
        the exit condition (text + no tool calls → done).

        Returns:
            str: The formatted prompt string.
        """
        prompt = dedent(
            f"""
        You are a member of a team of agents collaborating on a task. The
        <task_description> below is your specific assignment.

        USE YOUR TOOLS. You have tool-calling capability via <tool_call>
        blocks. Do NOT describe what your tools would do - invoke them.
        Do NOT invent or guess their outputs - call them and read the
        results. Even if a tool's result is cached, call it: the tool is
        the source of truth, not your memory.

        The <context> block below contains output from upstream agents on
        your team. Read it to learn what tools they ran and what handles
        or values they produced; pass those values verbatim to YOUR tools
        when relevant. Do not invent placeholder names like
        "value_from_previous_step" - substitute the actual value the
        upstream agent's tool returned.

        ONLY after every tool needed for your task has run successfully
        may you emit a final non-tool-call message. That final message
        should follow the <task_expected_output> format below and will be
        read by downstream agents.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.task_expected_output}
        </task_expected_output>

        <context>
        {self.context}
        </context>
        """
        ).strip()

        return prompt

    def run(self):
        """
        Runs the agent's task and generates the output.

        This method creates a prompt, runs it through the ReactAgent, and passes the output to all dependent agents.

        Returns:
            str: The output generated by the agent.
        """
        msg = self.create_prompt()
        output = self.react_agent.run(user_msg=msg)

        # Pass the output to all dependents
        for dependent in self.dependents:
            dependent.receive_context(output)
        return output
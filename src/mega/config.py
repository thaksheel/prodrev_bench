from dataclasses import dataclass
from typing import Any, Literal, Optional


from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Params:
    """Configuration for a multi-agent review run.

    Groups connection settings, token/loop limits, feature switches that
    trade capability for speed, sampling controls, and robustness limits.

    Attributes:
        api_key: API key used to authenticate requests.
        model: Model identifier sent with every request.
        reasoning_effort: Reasoning effort hint passed to the model. One of
            "none", "low", "medium", "high", "xhigh", or "max". Set to None
            to omit it from the request. Defaults to "none".
        base_url: Base URL of the OpenAI-compatible API.
            Defaults to "https://api.openai.com/v1".
        ceo_name: Display name of the leader (CEO) agent. Defaults to "Erza".

        max_memory: Number of recent history messages sent per request.
            Main token lever. Defaults to 6.
        max_rounds: Maximum LLM calls per agent wake-up. Defaults to 3.
        max_subordinates: Maximum employees kept from the plan, excluding
            the leader. Defaults to 2.
        max_leader_nudges: Number of times the leader is re-prompted when
            every agent is idle but no rating has been produced yet.
            Defaults to 2.

        parallel_kickoff: If True, employees start on the review immediately
            and in parallel, and the leader is woken once with all their
            reports. If False, uses the original flow where the leader
            delegates first, costing one extra sequential round trip.
            Defaults to True.
        share_file: If True, agents may read files written by other agents.
            Defaults to True.
        allow_recruiting: If True, exposes the ``add_agent`` tool so agents
            can recruit others. Adds latency. Defaults to False.
        track_todos: If True, exposes the ``change_task_status`` tool and
            injects TODO lists into prompts. Defaults to False.
        use_retrieval_memory: If True, enables ChromaDB-based retrieval of
            past context. Slow, so off by default. Defaults to False.

        tool_choice: Tool choice mode sent to the API. "required" forces
            the model to call a function. Defaults to "required".
        temperature: Sampling temperature. Only sent if set.
            Defaults to None.
        seed: Sampling seed for reproducibility. Only sent if set.
            Defaults to None.

        timeout_s: Hard time cap per sample, in seconds. Defaults to 60.0.
        llm_retries: Number of retries for a failed LLM call. Defaults to 3.
    """

    api_key: str
    model: str
    reasoning_effort: Optional[
        Literal["none", "low", "medium", "high", "xhigh", "max"]
    ] = "none"
    base_url: str = "https://api.openai.com/v1"
    ceo_name: str = "Erza"

    # --- context / loop limits (main token levers) ---
    max_memory: int = 6
    max_rounds: int = 3
    max_subordinates: int = 2
    max_leader_nudges: int = 2

    # --- speed switches ---
    parallel_kickoff: bool = True
    share_file: bool = True
    allow_recruiting: bool = False
    track_todos: bool = False
    use_retrieval_memory: bool = False

    # --- consistency ---
    tool_choice: Optional[str] = "required"
    temperature: Optional[float] = None
    seed: Optional[int] = None

    # --- robustness ---
    timeout_s: float = 60.0
    llm_retries: int = 3


@dataclass
class MASOut:
    pred: Optional[int]
    true: Optional[int]
    reasoning: str
    total_input_token: int
    total_output_token: int
    duration: float
    review: str = ""
    sample_id: str = ""
    n_llm_calls: int = 0
    error: Optional[str] = None


@dataclass
class LLMOut:
    response: Any
    message: Optional[str]
    input_token: int
    output_token: int
    model_name: str
    provider: str


class MegaPrompt:
    def __init__(self):
        self.task_ins: str = None
        self.leader_name: str = None

    def get_mega_instructions(
        self, leader_name: str, task_ins: str, max_agents: int = 3
    ):
        """Planning prompt (the one-off call that designs the team).

        Same structure as the original MegaAgent prompt, but it asks for a
        small team and short prompts, because every employee costs at least
        one extra LLM call per sample.
        """
        self.leader_name = leader_name
        self.task_ins = task_ins
        return f"""You are {leader_name}, the leader of a highly effective company. Your company's current goal is to {task_ins}
You are now recruiting employees and assigning work to them. For each employee (including yourself), write a prompt that states their name (one word, no prefix), their job, and exactly what they must do. You MUST state all their collaborators' names and jobs in the prompt. Use exactly this format (the example is for Alice in an unrelated novel-writing project):

<agent name="Alice">
You are Alice, a novelist. Your job is to write a single chapter of a novel with 1000 words according to the outline from Carol, the architect designer, and send it to David, the editor. Please only follow this routine. Your collaborators include Bob (the Boss), Carol (the architect designer) and David (the editor).
</agent>

Every employee is lazy and will not do anything that is not stated in their prompt, so each job must be non-divisible, concrete and short. Together the employees (including yourself) must cover the whole SOP, and {leader_name} makes the final decision. Use between 2 and {max_agents} employees in total, including yourself: every extra employee adds time and cost. Keep each prompt under 80 words."""

    def get_mega_prompt(self):
        operations = """
You can only output function calls in your response. DO NOT output anything else directly. For example, if you want to talk to someone, call the 'talk' function."""
        return self.task_ins + operations

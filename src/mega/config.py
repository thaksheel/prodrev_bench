from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Params:
    """Settings for the mega agent team.

    Parameters
    ----------
    api_key : str, optional
        API key. Replaced by the LLMProvider's token when MegaRunner gets one.
    model : str, default ""
        Model id. Replaced by the LLMProvider's model name when MegaRunner gets one.
    reasoning_effort : {"none", "low", "medium", "high", "xhigh", "max"}, optional, default "none"
        Reasoning effort sent with every request; None omits it.
    base_url : str, default "https://api.openai.com/v1"
        Base URL of the OpenAI-compatible API.
    ceo_name : str, default "Erza"
        Name of the leader agent.
    max_memory : int, default 6
        Recent history messages sent per request; the main token lever.
    max_rounds : int, default 3
        Maximum LLM calls per agent wake-up.
    max_subordinates : int, default 2
        Employees kept from the plan, excluding the leader.
    max_leader_nudges : int, default 2
        Times the leader is re-prompted when everyone is idle and no rating
        has been submitted.
    parallel_kickoff : bool, default True
        Employees start at once and the leader is woken once with all their
        reports. False: the leader delegates first, one extra round trip.
    share_file : bool, default True
        Agents may read files written by other agents.
    allow_recruiting : bool, default False
        Expose the ``add_agent`` tool.
    track_todos : bool, default False
        Expose the ``change_task_status`` tool and add TODO lists to prompts.
    use_retrieval_memory : bool, default False
        ChromaDB retrieval of past context. Slow.
    tool_choice : str, optional, default "required"
        "required" forces the model to call a function.
    temperature : float, optional
        Sent only when set. Replaced by the LLMProvider's temperature when
        MegaRunner gets one.
    seed : int, optional
        Sampling seed, sent only when set.
    timeout_s : float, default 60.0
        Time cap per review, in seconds.
    llm_retries : int, default 3
        Attempts per LLM call.
    """

    api_key: Optional[str] = None
    model: str = ""
    reasoning_effort: Optional[
        Literal["none", "low", "medium", "high", "xhigh", "max"]
    ] = "none"
    base_url: str = "https://api.openai.com/v1"
    ceo_name: str = "Erza"
    max_memory: int = 6
    max_rounds: int = 3
    max_subordinates: int = 2
    max_leader_nudges: int = 2
    parallel_kickoff: bool = True
    share_file: bool = True
    allow_recruiting: bool = False
    track_todos: bool = False
    use_retrieval_memory: bool = False
    tool_choice: Optional[str] = "required"
    temperature: Optional[float] = None
    seed: Optional[int] = None
    timeout_s: float = 60.0
    llm_retries: int = 3


@dataclass
class MASOut:
    """Result of one review."""

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

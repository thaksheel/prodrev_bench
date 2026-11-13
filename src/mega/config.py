from dataclasses import dataclass, field
from typing import Literal, Optional
from ..llm_provider import LLMProvider


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


@dataclass
class MegaConfig:
    """Settings for the mega agent method.

    Parameters
    ----------
    llm : LLMProvider, optional
        Model used by every team member. None uses the runner's predictor.
        Must be an "openai" provider: the agents rely on tool calling.
        Its model name, token and temperature replace those in ``params``.
    params : Params
        Team settings (size, memory, rounds, timeout, reasoning effort, ...).
    task_instruction : str
        Task description given to the leader when it designs the team.
    prompt_template : str
        Per-review project input; must contain ``{review}``.
    plan_path : str, default "./exports/mega_plan.txt"
        Saved team plan; created on the first run and reused afterwards.
    log_path : str, default "./exports/logs.out"
        Log file.
    run_dir : str, optional, default "./exports/mega_runs"
        Folder for each review's files, transcript and result.
    checkpoint_path : str, optional
        Checkpoint CSV; gets a ``_seed<N>`` suffix so each seed has its own.
    max_workers : int, default 4
        Reviews processed in parallel.
    progress : bool, default False
        Print one line per finished review.
    """

    llm: Optional[LLMProvider] = None
    params: Params = field(default_factory=Params)
    task_instruction: str = (
        "predict the overall star rating (1 to 5) a customer gave a product, using "
        "only the text of their online review. Do not use any label or information "
        "outside the review text as evidence. The final answer is an integer rating "
        "with a brief evidence-based rationale."
    )
    prompt_template: str = (
        "Score this customer online product review from 1-5: <review> {review} </review>"
    )
    plan_path: str = "./exports/mega_plan.txt"
    log_path: str = "./exports/logs.out"
    run_dir: Optional[str] = "./exports/mega_runs"
    checkpoint_path: Optional[str] = None
    max_workers: int = 4
    progress: bool = False

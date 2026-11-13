from typing import Optional
from dataclasses import dataclass

from .agent import MetaAgent
from ..llm_provider import LLMProvider


@dataclass
class MetaConfig:
    """Settings for the meta agent method (reason, ask for help, reflect, learn from experience).

    Parameters
    ----------
    llm : LLMProvider, optional
        Reasoning, reflection and experience model. None uses the runner's predictor.
    auxiliary_llm : LLMProvider, optional
        Turns help requests into search queries. None uses ``llm``.
    use_search : bool, default True
        Answer help requests with similar labelled reviews from the seed's
        few-shot pool, found with the KATE encoder. False: help requests get
        a note that no tool is available.
    search_topk : int, default 5
        Reviews retrieved per query.
    use_reflection : bool, default True
        Critique every answer and redo answers judged incorrect.
    use_experience : bool, default True
        After each review, write an experience summary by comparing the
        prediction with the author's rating and pass it to the next review,
        as in the original. This uses the evaluation labels and forces the
        reviews of a seed to run one after another.
    max_steps : int, default 15
        Reasoning calls per review before an answer is forced.
    max_help : int, default 10
        Help budget stated in the prompt.
    max_reflections : int, default 3
        Rejections before the final rating is drawn from the previous answers.
    max_retries : int, default 3
        Attempts per review when a call raises an error.
    max_workers : int, default 4
        Reviews processed in parallel; only used when ``use_experience`` is False.
    """

    llm: Optional["LLMProvider"] = None
    auxiliary_llm: Optional["LLMProvider"] = None
    use_search: bool = True
    search_topk: int = 5
    use_reflection: bool = True
    use_experience: bool = True
    max_steps: int = 15
    max_help: int = 10
    max_reflections: int = 3
    max_retries: int = 3
    max_workers: int = 4


__all__ = [
    "MetaAgent",
    "MetaConfig",
]

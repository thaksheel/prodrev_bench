from dataclasses import dataclass
from typing import Optional

from .agent import Agent
from .debate import Debate
from ..llm_provider import LLMProvider


@dataclass
class MadConfig:
    """Settings for the multi-agent debate method.

    Parameters
    ----------
    llm : LLMProvider, optional
        Model used by the debaters and the debate judge. None uses the
        runner's predictor, so MAD and the prompting methods share a model.
    references_path : str, default "./exports/references.json"
        Output of build_references.py.
    num_agents : int, default 5
        Debaters, one per star rating from 1 up.
    num_rounds : int, default 3
        Debate rounds.
    max_workers : int, default 4
        Reviews debated in parallel. Use 1 for a local Hugging Face model.
    """

    llm: Optional[LLMProvider] = None
    references_path: str = "./exports/references.json"
    num_agents: int = 5
    num_rounds: int = 3
    max_workers: int = 4


__all__ = [
    "MadConfig",
    "Agent",
    "Debate",
]

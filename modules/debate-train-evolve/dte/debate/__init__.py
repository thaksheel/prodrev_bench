"""Multi-agent debate components with structured prompting."""

from .agent import DebateAgent
from .manager import DebateManager, DebateResult
from .prompts import DebatePromptManager, DebateResponse

__all__ = [
    "DebateAgent",
    "DebateManager",
    "DebateResult",
    "DebatePromptManager",
    "DebateResponse",
]

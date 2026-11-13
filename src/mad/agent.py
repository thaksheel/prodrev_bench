import re
from typing import Any, Dict, List

from ..llm_provider import LLMProvider


def fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace(f"##{key}##", value)
    return template


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text), flags=re.S).strip()


class Agent:
    """One debater that argues for a fixed star rating.

    The agent keeps a single conversation across rounds, so in later rounds
    it still sees the review, its earlier arguments and the rebuttals.

    Parameters
    ----------
    agent_id : int
        Position of the agent in the debate.
    rating : int
        Star rating the agent argues for.
    instructions : str
        System prompt; ``##reference##`` or ``##references##`` is replaced by ``reference``.
    prompt_tmpl : str
        First-round prompt; ``##text##`` is replaced by the review and
        ``##reference##`` by ``reference``.
    rebuttal_tmpl : str
        Later-round prompt; ``##rebuddle##`` is replaced by the other agents' arguments.
    reference : str
        Description of reviews at this rating.
    llm : LLMProvider
        Model that generates the agent's arguments.
    """

    def __init__(
        self,
        agent_id: int,
        rating: int,
        instructions: str,
        prompt_tmpl: str,
        rebuttal_tmpl: str,
        reference: str,
        llm: LLMProvider,
    ):
        self.agent_id = agent_id
        self.rating = rating
        self.prompt_tmpl = prompt_tmpl
        self.rebuttal_tmpl = rebuttal_tmpl
        self.reference = reference
        self.llm = llm
        system = fill(instructions, reference=reference, references=reference)
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        self.opinions: List[str] = []
        self.calls: List[Dict[str, Any]] = []

    def open(self, text: str) -> str:
        """First round: argue for this agent's rating."""
        prompt = fill(
            self.prompt_tmpl,
            text=text,
            reference=self.reference,
            references=self.reference,
        )
        return self._say(prompt)

    def rebut(self, others: str) -> str:
        """Later rounds: agree with or rebut the other agents' arguments."""
        return self._say(fill(self.rebuttal_tmpl, rebuddle=others))

    def _say(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        sent = list(self.messages)
        out = self.llm.invoke(messages=sent)
        opinion = strip_thinking(out.response)
        self.messages.append({"role": "assistant", "content": opinion})
        self.opinions.append(opinion)
        self.calls.append(
            {
                "label": f"agent {self.agent_id} ({self.rating}-star), round {len(self.opinions)}",
                "messages": sent,
                "out": out,
            }
        )
        return opinion

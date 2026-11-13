import re, json, time 
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..llm_provider import LLMProvider
from .agent import Agent, strip_thinking

JUDGE_SYSTEM = "Follow the given instructions."
JUDGE_PROMPT = """You are the Sentiment Rating Judge. The debaters have completed their arguments.
Review Text:
"##text##"

Debate History:
##history##

Your task:
1. Evaluate all arguments from the debaters.
2. Determine which star rating (1-5) is best supported by the evidence and reasoning.
3. Provide a final judgment strictly in JSON format.

JSON Output Format:
{"Rating": <1-5>, "Reason": "<one-sentence explanation>"}

Important:
- The "Rating" must be a single integer from 1 to 5.
- The "Reason" must be one concise sentence summarizing why that rating is correct.
- Do NOT include any text outside the JSON object.
"""


def _parse_json(text: Optional[str]) -> Optional[dict]:
    """Extract the first {...} block from an LLM response."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", str(text))
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_judgement(text: str) -> Tuple[Optional[int], Optional[str]]:
    text = strip_thinking(text)
    obj = _parse_json(text) or {}
    rating = obj.get("Rating", obj.get("rating"))
    reason = obj.get("Reason", obj.get("reason"))
    if rating is None:
        m = re.search(r'"?rating"?\s*[:=]\s*"?([1-5])', text, re.I)
        rating = m.group(1) if m else None
    m = re.search(r"[1-5]", str(rating)) if rating is not None else None
    return (int(m.group()) if m else None), reason


class Debate:
    """Multi-agent debate for 1-5 star rating prediction.

    One agent per star rating argues that the review deserves its rating.
    In every round after the first, each agent sees the other agents'
    previous arguments and agrees or rebuts. A judge then reads the full
    debate and gives the final rating.

    Parameters
    ----------
    llm : LLMProvider
        Model used by every agent and the judge.
    references : dict
        Contents of references.json, with the lists ``instructions``,
        ``prompt_template``, ``rebuddle_prompt_template`` and ``references``
        (one entry per rating) and optionally ``rating``.
    num_agents : int, default 5
        Agents in the debate, one per rating from 1 up.
    num_rounds : int, default 3
        Debate rounds.

    Notes
    -----
    Each review costs ``num_agents * num_rounds + 1`` LLM calls.
    """

    def __init__(
        self,
        llm: LLMProvider,
        references: Dict[str, Any],
        num_agents: int = 5,
        num_rounds: int = 3,
    ):
        self.llm = llm
        self.references = references
        self.num_agents = num_agents
        self.num_rounds = num_rounds

    def _make_agents(self) -> List[Agent]:
        refs = self.references
        ratings = refs.get("rating", list(range(1, self.num_agents + 1)))
        return [
            Agent(
                agent_id=i,
                rating=int(ratings[i]),
                instructions=refs["instructions"][i],
                prompt_tmpl=refs["prompt_template"][i],
                rebuttal_tmpl=refs["rebuddle_prompt_template"][i],
                reference=refs["references"][i],
                llm=self.llm,
            )
            for i in range(self.num_agents)
        ]

    @staticmethod
    def _history(agents: List[Agent]) -> str:
        lines = []
        for r in range(max(len(a.opinions) for a in agents)):
            lines.append(f"Round {r + 1}:")
            for agent in agents:
                if r < len(agent.opinions):
                    lines.append(
                        f"- Debater {agent.agent_id} (argues {agent.rating} stars): "
                        f"{agent.opinions[r]}"
                    )
        return "\n".join(lines)

    def run(self, text: str) -> Dict[str, Any]:
        """Debate one review.

        Returns
        -------
        dict
            ``rating``, ``reasoning``, ``n_llm_calls``, ``input_tokens``,
            ``output_tokens``, ``duration``, ``error`` and ``calls`` (every
            call's label, messages and LLMOut).
        """
        start = time.time()
        agents = self._make_agents()
        judge_call = None
        rating, reason, error = None, None, None
        try:
            for agent in agents:
                agent.open(text)
            for _ in range(1, self.num_rounds):
                previous = [agent.opinions[-1] for agent in agents]
                for agent in agents:
                    others = " ".join(
                        f"Debater {other.agent_id} ({other.rating}-star): {opinion}"
                        for other, opinion in zip(agents, previous)
                        if other is not agent
                    )
                    agent.rebut(others)

            prompt = JUDGE_PROMPT.replace("##text##", text).replace(
                "##history##", self._history(agents)
            )
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            out = self.llm.invoke(messages=messages)
            judge_call = {"label": "judge", "messages": messages, "out": out}
            rating, reason = parse_judgement(out.response)
            if rating is None:
                error = f"Unparseable judgement: {str(out.response)[:200]!r}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        calls = [call for agent in agents for call in agent.calls]
        if judge_call is not None:
            calls.append(judge_call)
        return {
            "rating": rating,
            "reasoning": reason,
            "n_llm_calls": len(calls),
            "input_tokens": sum(call["out"].input_token or 0 for call in calls),
            "output_tokens": sum(call["out"].output_token or 0 for call in calls),
            "duration": time.time() - start,
            "error": error,
            "calls": calls,
        }

    def simulate_debate(
        self, texts: List[str], max_workers: int = 1
    ) -> List[Dict[str, Any]]:
        """Debate every review; reviews run in parallel when ``max_workers > 1``."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(tqdm(pool.map(self.run, texts), total=len(texts)))

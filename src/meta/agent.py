import random
import re
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple

from ..llm_provider import LLMProvider
from ..prompts import KATEExamples
from ..review_judge import _parse_json
from . import prompts as P

MAX_REVIEW_CHARS = 600


def strip_thinking(text) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S).strip()


def extract_last(text: str, start: str, end: str) -> Optional[str]:
    matches = re.findall(
        re.escape(start) + r"(.*?)" + re.escape(end), text or "", flags=re.S
    )
    return matches[-1].strip() if matches else None


def first_block(text: str) -> Tuple[Optional[str], str]:
    """The first <help> or <answer> block in the response, as (kind, content)."""
    found = [
        (text.find(f"<{tag}>"), tag) for tag in ("help", "answer") if f"<{tag}>" in text
    ]
    if not found:
        return None, ""
    _, tag = min(found)
    content = text.split(f"<{tag}>", 1)[1].split(f"</{tag}>", 1)[0]
    return tag, content.strip()


def parse_rating(text: str) -> Optional[int]:
    m = re.search(r"\b([1-5])\b", text or "")
    return int(m.group(1)) if m else None


def parse_verdict(text: str) -> Optional[str]:
    analysis = extract_last(text, "<analysis>", "</analysis>") or text
    m = re.search(r"answer_correctness\s*:\s*\[?\s*(correct|incorrect)", analysis, re.I)
    return m.group(1).lower() if m else None


class MetaAgent:
    """Reason, ask for help, reflect and learn from experience, for 1-5 star ratings.

    Adapted from the authors' GAIA agent. For each review the reasoner
    either answers or asks for help. A help request is turned into search
    queries by the auxiliary model and answered with similar labelled reviews
    (on the first request, reviews similar to the review itself are added).
    Every answer is critiqued by a reflection step; an answer judged incorrect
    is redone with the critique. After ``max_reflections`` rejections the
    final rating is drawn at random from the previous answers, as in the
    original. Finally, an experience summary is written by comparing the
    prediction with the true rating; the caller passes it to the next review.

    Parameters
    ----------
    llm : LLMProvider
        Reasoning, reflection and experience model.
    auxiliary_llm : LLMProvider, optional
        Turns help requests into search queries. Defaults to ``llm``.
    searcher : KATEExamples, optional
        Index of labelled reviews used to answer help requests. None
        disables search.
    use_reflection : bool, default True
    use_experience : bool, default True
        Write an experience summary when a true rating is given.
    max_steps : int, default 15
        Reasoning calls per review before an answer is forced.
    max_help : int, default 10
        Help budget stated in the prompt.
    max_reflections : int, default 3
    max_retries : int, default 3
        Attempts per review when a call raises an error.
    seed : int, default 0
        Seed for the random fallback answer.
    """

    def __init__(
        self,
        llm: LLMProvider,
        auxiliary_llm: Optional[LLMProvider] = None,
        searcher: Optional[KATEExamples] = None,
        use_reflection: bool = True,
        use_experience: bool = True,
        max_steps: int = 15,
        max_help: int = 10,
        max_reflections: int = 3,
        max_retries: int = 3,
        seed: int = 0,
    ):
        self.llm = llm
        self.auxiliary_llm = auxiliary_llm or llm
        self.searcher = searcher
        self.use_reflection = use_reflection
        self.use_experience = use_experience
        self.max_steps = max_steps
        self.max_help = max_help
        self.max_reflections = max_reflections
        self.max_retries = max_retries
        self.seed = seed

    # ------------------------------------------------------------------ #
    # One review
    # ------------------------------------------------------------------ #
    def run(
        self,
        review: str,
        true_rating: Optional[int] = None,
        experience: Optional[P.Experience] = None,
    ) -> Dict[str, Any]:
        """Rate one review.

        Returns
        -------
        dict
            ``rating``, ``reasoning``, ``experience`` (the updated summary or
            None), ``n_help``, ``n_reflections``, ``n_llm_calls``,
            ``input_tokens``, ``output_tokens``, ``duration``, ``error`` and
            ``calls`` (every call's label, messages and LLMOut).
        """
        start = time.time()
        calls: List[Dict[str, Any]] = []
        result, error = None, None
        for _ in range(self.max_retries):
            try:
                result = self._solve(review, experience, calls)
                error = None
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        new_experience = None
        if result is not None and self.use_experience and true_rating is not None:
            try:
                new_experience = self._summarize(
                    review, result, int(true_rating), experience, calls
                )
            except Exception as exc:
                error = f"experience: {type(exc).__name__}: {exc}"

        result = result or {
            "rating": None,
            "reasoning": None,
            "n_help": 0,
            "n_reflections": 0,
        }
        return {
            **result,
            "experience": new_experience,
            "n_llm_calls": len(calls),
            "input_tokens": sum(c["out"].input_token or 0 for c in calls),
            "output_tokens": sum(c["out"].output_token or 0 for c in calls),
            "duration": time.time() - start,
            "error": error,
            "calls": calls,
        }

    def _invoke(
        self,
        llm: LLMProvider,
        label: str,
        messages: List[Dict[str, str]],
        calls: List[Dict[str, Any]],
    ) -> str:
        out = llm.invoke(messages=messages)
        calls.append({"label": label, "messages": list(messages), "out": out})
        return strip_thinking(out.response)

    def _solve(
        self,
        review: str,
        experience: Optional[P.Experience],
        calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        reflections: List[str] = []
        answers: List[int] = []
        shown: set = set()
        queries: List[str] = []
        state = {"step": 0, "help": 0}

        while True:
            answer, transcript = self._reason(
                review, experience, reflections, shown, queries, state, calls
            )
            answers.append(answer)
            if not self.use_reflection:
                return self._result(answer, transcript, state, reflections)
            if len(reflections) >= self.max_reflections:
                fallback = random.Random(
                    self.seed * 1_000_003 + zlib.crc32(review.encode("utf-8"))
                )
                return self._result(
                    fallback.choice(answers), transcript, state, reflections
                )

            prompt = P.REFLECTION_PROMPT.format(
                review=review, reasoning=transcript, answer=answer
            )
            critique = self._invoke(
                self.llm,
                f"reflection {len(reflections) + 1}",
                [{"role": "user", "content": prompt}],
                calls,
            )
            if parse_verdict(critique) == "correct":
                return self._result(answer, transcript, state, reflections + [critique])
            reflections.append(critique)

    def _reason(
        self,
        review: str,
        experience: Optional[P.Experience],
        reflections: List[str],
        shown: set,
        queries: List[str],
        state: Dict[str, int],
        calls: List[Dict[str, Any]],
    ) -> Tuple[int, str]:
        messages = [
            {
                "role": "system",
                "content": P.reasoning_prompt(experience, reflections, self.max_help),
            },
            {"role": "user", "content": P.TASK.format(review=review)},
        ]

        for _ in range(self.max_steps + 3):
            state["step"] += 1
            if state["step"] > self.max_steps:
                messages.append({"role": "user", "content": P.FORCE_ANSWER})
            text = self._invoke(self.llm, f"reasoning {state['step']}", messages, calls)
            messages.append({"role": "assistant", "content": text})
            kind, content = first_block(text)

            if kind == "answer":
                rating = parse_rating(content)
                if rating is not None:
                    return rating, self._transcript(messages)
                messages.append({"role": "user", "content": P.BAD_ANSWER})
            elif kind == "help" and state["step"] <= self.max_steps:
                first = state["help"] == 0
                state["help"] += 1
                evidence = self._help(review, content, first, shown, queries, calls)
                messages.append(
                    {"role": "user", "content": f"<evidence>\n{evidence}\n</evidence>"}
                )
            else:
                messages.append({"role": "user", "content": P.NO_BLOCK})
        raise RuntimeError("no valid answer within the step limit")

    def _help(
        self,
        review: str,
        request: str,
        first: bool,
        shown: set,
        queries: List[str],
        calls: List[Dict[str, Any]],
    ) -> str:
        if self.searcher is None:
            return P.NO_TOOL
        prompt = P.TOOL_PROMPT.format(
            previous="\n".join(queries) or "none", request=request
        )
        text = self._invoke(
            self.auxiliary_llm,
            "tool queries",
            [{"role": "user", "content": prompt}],
            calls,
        )
        new = [
            str(q)
            for q in (_parse_json(text) or {}).get("queries", [])
            if str(q).strip()
        ][:3] or [request]
        queries.extend(new)
        if first:
            new = [review] + new

        lines = []
        for query in new:
            for text_, rating in self.searcher.select(query):
                if text_ in shown or text_ == review:
                    continue
                shown.add(text_)
                lines.append(
                    f"Similar review (rated {rating} stars): {str(text_)[:MAX_REVIEW_CHARS]}"
                )
        return "\n\n".join(lines) or P.NO_EVIDENCE

    @staticmethod
    def _transcript(messages: List[Dict[str, str]]) -> str:
        return "\n\n".join(m["content"] for m in messages[2:])

    @staticmethod
    def _result(
        rating: int, transcript: str, state: Dict[str, int], reflections: List[str]
    ) -> Dict[str, Any]:
        return {
            "rating": rating,
            "reasoning": transcript[-4000:],
            "n_help": state["help"],
            "n_reflections": len(reflections),
        }

    def _summarize(
        self,
        review: str,
        result: Dict[str, Any],
        true_rating: int,
        experience: Optional[P.Experience],
        calls: List[Dict[str, Any]],
    ) -> Optional[str]:
        task = P.TASK_DESCRIPTION.format(
            review=review,
            reasoning=result["reasoning"],
            pred=result["rating"],
            true=true_rating,
        )
        text = self._invoke(
            self.llm,
            "experience",
            [{"role": "user", "content": P.experience_prompt(task, experience)}],
            calls,
        )
        return extract_last(text, "<updated_experience>", "</updated_experience>")

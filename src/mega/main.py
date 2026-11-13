import csv
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Collection, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .agent import Session
from .config import MASOut, Params
from .llm import get_llm_response
from .prompts import (
    KICKOFF_LEADER,
    KICKOFF_MEMBER,
    NUDGE_LEADER,
    PLAN_REQUEST,
    RELEASE_LEADER,
    TODO_REMINDER,
)

if TYPE_CHECKING:
    from ..llm_provider import LLMProvider

logger = logging.getLogger("mega_agent")


class PlanError(RuntimeError):
    pass


def params_from_provider(params: Params, llm: Optional["LLMProvider"]) -> Params:
    """Copy the model, token and temperature of an LLMProvider into Params."""
    if llm is None:
        return params
    if llm.provider != "openai":
        raise ValueError(
            f"The mega agent needs an OpenAI-compatible model with tool calling; "
            f"got provider {llm.provider!r}."
        )
    effort = getattr(llm, "reasoning_effort", None) or params.reasoning_effort
    return replace(
        params,
        model=llm.model_name,
        api_key=llm.token,
        temperature=llm.temperature,
        reasoning_effort=effort,
    )


class MegaRunner:
    """Runs the mega agent team, one isolated Session per review.

    The team plan (the leader's one-off call that recruits employees and
    writes their prompts) depends only on the task, so it is made once and
    reused for every review.

    Parameters
    ----------
    params : Params
        Team settings.
    llm : LLMProvider, optional
        Model to use. Its model name, token and temperature (and
        ``reasoning_effort``, if it has one) replace those in ``params``.
        Must be an "openai" provider: the agents rely on OpenAI-style tool calls.
    log_path : str, default "./exports/logs.out"
        Log file.
    run_dir : str, optional, default "./exports/mega_runs"
        Folder for each review's files, transcript and result. None saves nothing.
    display : bool, default False
        Also print the log.

    Attributes
    ----------
    plan_text : str or None
        The team plan.
    plan_input_token, plan_output_token : int
        Tokens spent on planning.
    calls_ : dict of str to list of dict
        Every request (label and messages) for the reviews named in
        ``run_many(record_ids=...)``.
    """

    def __init__(
        self,
        params: Params,
        llm: Optional["LLMProvider"] = None,
        log_path: str = "./exports/logs.out",
        run_dir: Optional[str] = "./exports/mega_runs",
        display: bool = False,
    ):
        self.params = params_from_provider(params, llm)
        self.log_path = log_path
        self.display = display
        self.run_dir = Path(run_dir) if run_dir else None
        self.plan_text: Optional[str] = None
        self.plan_input_token = 0
        self.plan_output_token = 0
        self.calls_: Dict[str, List[dict]] = {}
        self._plan_key: Optional[str] = None
        self._plan_lock = threading.Lock()
        self._ckpt_lock = threading.Lock()
        self._calls_lock = threading.Lock()
        self._init_logger()

    def _init_logger(self) -> None:
        if logger.handlers:
            return
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        fh = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        if self.display:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    def plan(self, task_instruction: str, force: bool = False) -> str:
        with self._plan_lock:
            reusable = self.plan_text is not None and self._plan_key in (
                None,
                task_instruction,
            )
            if reusable and not force:
                return self.plan_text
            messages = [
                {"role": "system", "content": task_instruction},
                {"role": "user", "content": PLAN_REQUEST},
            ]
            for attempt in range(2):
                out = get_llm_response(messages, self.params, enable_tools=False)
                self.plan_input_token += out.input_token
                self.plan_output_token += out.output_token
                text = out.response.get("content") or ""
                if len(self.parse_plan(text)) >= 2:
                    self.plan_text, self._plan_key = text, task_instruction
                    logger.info(f"Team plan:\n{text}")
                    return text
                logger.warning(
                    f"Plan attempt {attempt + 1} had fewer than 2 agents:\n{text}"
                )
            raise PlanError("The planner did not return at least 2 <agent> blocks.")

    def save_plan(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.plan_text or "", encoding="utf-8")

    def load_plan(self, path: str) -> str:
        """Reuse a saved plan for reproducible reruns."""
        self.plan_text = Path(path).read_text(encoding="utf-8")
        self._plan_key = None
        return self.plan_text

    def parse_plan(self, text: str) -> List[Tuple[str, str]]:
        """Return [(name, prompt)], leader first, at most ``max_subordinates`` others."""
        found, seen = [], set()
        for raw, prompt in re.findall(
            r'<agent name="([^"]+)">(.*?)</agent>', text or "", re.DOTALL
        ):
            name = re.sub(r"\W", "", raw)
            if name and name.lower() not in seen and prompt.strip():
                seen.add(name.lower())
                found.append((name, prompt.strip()))
        ceo = self.params.ceo_name
        leader = [(ceo, p) for n, p in found if n.lower() == ceo.lower()]
        if not leader:
            leader = [
                (
                    ceo,
                    f"You are {ceo}, the leader. Collect your employees' reports "
                    f"and make the final decision.",
                )
            ]
        members = [(n, p) for n, p in found if n.lower() != ceo.lower()][
            : self.params.max_subordinates
        ]
        return leader + members

    # ------------------------------------------------------------------ #
    # One review
    # ------------------------------------------------------------------ #
    def run(
        self,
        task_instruction: str,
        prompt: str,
        true: Optional[int] = None,
        sample_id: Optional[str] = None,
        review: str = "",
        record: bool = False,
    ) -> MASOut:
        """Run one review. Failures come back as pred=None with the reason in ``error``."""
        param = self.params
        team = self.parse_plan(self.plan(task_instruction))
        session = Session(
            param, task_context=prompt, sample_id=sample_id, record=record
        )
        leader = session.add_agent(team[0][0], team[0][1], is_leader=True)
        members = [session.add_agent(n, p) for n, p in team[1:]]

        start = time.time()
        deadline = time.monotonic() + param.timeout_s
        if param.parallel_kickoff:
            leader.hold = True
            for m in members:
                m.enqueue("user", KICKOFF_MEMBER)
        else:
            leader.enqueue("user", KICKOFF_LEADER)

        nudges = 0
        while not session.stopped:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                session.fail(f"timeout after {param.timeout_s:.0f}s")
                break
            if not session.wait(remaining) or session.stopped:
                continue
            if leader.hold:
                leader.release(RELEASE_LEADER)
                continue
            if nudges >= param.max_leader_nudges:
                session.fail("leader never called submit_rating")
                break
            nudges += 1
            if param.track_todos:
                for a in session.agents.values():
                    todo, _, _ = session.workspace.read(f"todo_{a.name}.txt")
                    if todo and a is not leader:
                        a.enqueue("user", TODO_REMINDER)
            leader.enqueue("user", NUDGE_LEADER)

        tin, tout, ncalls = session.totals()
        rating, reasoning = (
            (session.result[0], session.result[1]) if session.result else (None, "")
        )
        out = MASOut(
            pred=rating,
            true=int(true) if true is not None else None,
            reasoning=reasoning,
            total_input_token=tin,
            total_output_token=tout,
            duration=round(time.time() - start, 3),
            review=review or prompt,
            sample_id=session.id,
            n_llm_calls=ncalls,
            error=session.error,
        )
        logger.info(
            f"[{session.id}] pred={out.pred} true={out.true} calls={ncalls} "
            f"tokens={tin}/{tout} {out.duration}s error={out.error}"
        )
        if record:
            with self._calls_lock:
                self.calls_[session.id] = list(session.requests)
        self._save_artifacts(session, out)
        return out

    def _save_artifacts(self, session: Session, out: MASOut) -> None:
        if self.run_dir is None:
            return
        try:
            d = self.run_dir / session.id
            session.workspace.dump(d / "files")
            with open(d / "transcript.jsonl", "w", encoding="utf-8") as f:
                for ev in session.transcript():
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            (d / "result.json").write_text(
                json.dumps(asdict(out), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as e:
            logger.warning(f"[{session.id}] could not save artifacts: {e}")

    # ------------------------------------------------------------------ #
    # Many reviews
    # ------------------------------------------------------------------ #
    def run_many(
        self,
        task_instruction: str,
        reviews: Sequence[str],
        trues: Optional[Sequence[int]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        prompt_template: str = "{review}",
        max_workers: int = 4,
        checkpoint_path: Optional[str] = None,
        progress: bool = True,
        record_ids: Optional[Collection[str]] = None,
    ) -> pd.DataFrame:
        """Run every review in parallel threads; one row per review, in input order.

        Parameters
        ----------
        task_instruction : str
            Planning prompt from ``MegaPrompt.get_mega_instructions``.
        reviews : sequence of str
        trues : sequence of int, optional
            Author ratings, only logged.
        sample_ids : sequence of str, optional
            Ids used for logs, artifact folders and checkpoints. Defaults to
            ``s0000``, ``s0001``, ...
        prompt_template : str, default "{review}"
            Per-review project input; must contain ``{review}``.
        max_workers : int, default 4
        checkpoint_path : str, optional
            Each finished row is appended here immediately.
        progress : bool, default True
            Print one line per finished review.
        record_ids : collection of str, optional
            Keep every request for these sample ids in ``calls_``.
        """
        self.plan(task_instruction)
        reviews = list(reviews)
        trues = list(trues) if trues is not None else [None] * len(reviews)
        ids = (
            list(sample_ids)
            if sample_ids is not None
            else [f"s{i:04d}" for i in range(len(reviews))]
        )
        record_ids = set(record_ids or ())
        rows: List[Optional[MASOut]] = [None] * len(reviews)

        def job(i: int) -> Tuple[int, MASOut]:
            try:
                out = self.run(
                    task_instruction,
                    prompt_template.format(review=reviews[i]),
                    true=trues[i],
                    sample_id=ids[i],
                    review=reviews[i],
                    record=ids[i] in record_ids,
                )
            except Exception as e:
                out = MASOut(
                    None,
                    trues[i],
                    "",
                    0,
                    0,
                    0.0,
                    reviews[i],
                    ids[i],
                    0,
                    f"{type(e).__name__}: {e}",
                )
            self._checkpoint(checkpoint_path, out)
            return i, out

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for fut in as_completed([pool.submit(job, i) for i in range(len(reviews))]):
                i, out = fut.result()
                rows[i] = out
                done += 1
                if progress:
                    print(
                        f"[{done}/{len(reviews)}] {out.sample_id} pred={out.pred} true={out.true} "
                        f"{out.duration:.1f}s calls={out.n_llm_calls}"
                        + (f" ERROR={out.error}" if out.error else "")
                    )
        return pd.DataFrame([asdict(r) for r in rows])

    def _checkpoint(self, path: Optional[str], out: MASOut) -> None:
        if not path:
            return
        row = asdict(out)
        with self._ckpt_lock:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            new = not p.exists()
            with open(p, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)

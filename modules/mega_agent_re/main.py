import csv
import json
import logging
import re
import pandas as pd
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import Session, MASOut, Params, get_llm_response

logger = logging.getLogger("mega_agent")

KICKOFF_MEMBER = "[System] The project has started. Do your job now."
KICKOFF_LEADER = "[System] The project has started. Delegate the work now."
RELEASE_LEADER = (
    "[System] All employees have finished; their reports are in the messages "
    "above. Decide the final rating and call submit_rating."
)
NUDGE_LEADER = (
    "[System] Everyone is idle and no rating has been submitted. If you are "
    "missing a report, talk to that employee; otherwise call submit_rating now."
)
TODO_REMINDER = (
    "[System] Other agents have finished, but your TODO list is not empty. "
    "Finish it and clear it, or remind whoever you are waiting for."
)


class PlanError(RuntimeError):
    pass


class Runner:
    """Runs the MegaAgent pipeline, one isolated Session per sample.

    The team plan (the leader's "recruit employees and write their prompts"
    call) depends only on the task, not on the review, so it is generated
    once and reused for every sample: one fewer LLM call per sample, and the
    same team structure every time, which keeps predictions comparable.
    """

    def __init__(
        self,
        log_path: str = "./exports/logs.out",
        display: bool = False,
        run_dir: Optional[str] = "./exports/mega_runs",
    ):
        self.log_path = log_path
        self.display = display
        self.run_dir = Path(run_dir) if run_dir else None
        self.plan_text: Optional[str] = None
        self.plan_input_token = 0
        self.plan_output_token = 0
        self._plan_key: Optional[str] = None
        self._plan_lock = threading.Lock()
        self._ckpt_lock = threading.Lock()
        self.init_logger()

    # ---- logging ---------------------------------------------------------
    def init_logger(self):
        # One named logger, configured once. Nothing is opened per sample,
        # so there are no lingering handles on log files.
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

    # ---- planning --------------------------------------------------------
    def plan(self, params: Params, task_instruction: str, force: bool = False) -> str:
        with self._plan_lock:
            reusable = self.plan_text is not None and self._plan_key in (
                None,
                task_instruction,
            )
            if reusable and not force:
                return self.plan_text
            messages = [
                {"role": "system", "content": task_instruction},
                {
                    "role": "user",
                    "content": "Create the agents and assign the tasks for them.",
                },
            ]
            for attempt in range(2):
                out = get_llm_response(messages, params, enable_tools=False)
                self.plan_input_token += out.input_token
                self.plan_output_token += out.output_token
                text = out.message or ""
                if len(self.parse_plan(text, params)) >= 2:
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
        """Reuse a saved plan (any task_instruction) for reproducible reruns."""
        self.plan_text = Path(path).read_text(encoding="utf-8")
        self._plan_key = None
        return self.plan_text

    @staticmethod
    def parse_plan(text: str, params: Params) -> List[Tuple[str, str]]:
        """Return [(name, prompt)], leader first, at most max_subordinates others."""
        found, seen = [], set()
        for raw, prompt in re.findall(
            r'<agent name="([^"]+)">(.*?)</agent>', text or "", re.DOTALL
        ):
            name = re.sub(r"\W", "", raw)
            if name and name.lower() not in seen and prompt.strip():
                seen.add(name.lower())
                found.append((name, prompt.strip()))
        ceo = params.ceo_name
        leader = [(ceo, p) for n, p in found if n.lower() == ceo.lower()]
        if not leader:
            leader = [
                (
                    ceo,
                    f"You are {ceo}, the leader. Collect your employees' reports and make the final decision.",
                )
            ]
        members = [(n, p) for n, p in found if n.lower() != ceo.lower()][
            : params.max_subordinates
        ]
        return leader + members

    # ---- one sample ------------------------------------------------------
    def run(
        self,
        param: Params,
        task_instruction: str,
        prompt: str,
        true: Optional[int] = None,
        sample_id: Optional[str] = None,
        review: str = "",
    ) -> MASOut:
        """Run one sample. Never raises for per-sample failures: they come
        back as a row with pred=None and the reason in `error`."""
        team = self.parse_plan(self.plan(param, task_instruction), param)
        session = Session(param, task_context=prompt, sample_id=sample_id)
        leader = session.add_agent(team[0][0], team[0][1], is_leader=True)
        members = [session.add_agent(n, p) for n, p in team[1:]]

        start = time.time()
        deadline = time.monotonic() + param.timeout_s
        if param.parallel_kickoff:
            leader.hold = True  # buffer reports; wake once everyone is done
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
            # Everyone is idle and there is no rating yet.
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
        self._save_artifacts(session, out)
        return out

    def _save_artifacts(self, session: Session, out: MASOut) -> None:
        # Written once by the main thread after the run, into a directory
        # unique to this sample, so there is never contention on a filename.
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

    # ---- many samples ----------------------------------------------------
    def run_many(
        self,
        params: Params,
        task_instruction: str,
        reviews: Sequence[str],
        trues: Optional[Sequence[int]] = None,
        prompt_template: str = "{review}",
        max_workers: int = 4,
        checkpoint_path: Optional[str] = None,
        progress: bool = True,
    ):
        """Run every review (in parallel threads) and return a DataFrame with
        one row per review, in input order. Each finished row is appended to
        checkpoint_path immediately, so a long run survives interruptions."""

        self.plan(params, task_instruction)  # once, before the workers start
        reviews = list(reviews)
        trues = list(trues) if trues is not None else [None] * len(reviews)
        rows: List[Optional[MASOut]] = [None] * len(reviews)

        def job(i: int) -> Tuple[int, MASOut]:
            try:
                out = self.run(
                    params,
                    task_instruction,
                    prompt_template.format(review=reviews[i]),
                    true=trues[i],
                    sample_id=f"s{i:04d}",
                    review=reviews[i],
                )
            except Exception as e:  # defensive: a sample can never kill the batch
                out = MASOut(
                    None,
                    trues[i],
                    "",
                    0,
                    0,
                    0.0,
                    reviews[i],
                    f"s{i:04d}",
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

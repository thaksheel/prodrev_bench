import json
import threading
import time
import uuid
from typing import Dict, List, Optional

from . import Params, build_tools, get_llm_response, Workspace

NO_CALL_MSG = (
    "[System] Error: respond only with function calls. "
    "If you have nothing left to do, call terminate."
)
MAX_TOOL_OUTPUT = 1500


COMMON_RULES = (
    "Respond ONLY with function calls, never plain text. Be brief: keep talk "
    "messages under 60 words. Files are shared by the whole team."
)
LEADER_PARALLEL = (
    "How this project runs: your employees start immediately and their reports "
    "are delivered to you as messages. Do not redo their work. Once you have "
    "their reports, decide the final answer and call submit_rating(rating, "
    "reasoning); it records review.txt for you and ends the project. Only talk "
    "to an employee if a report is missing or unclear, then call terminate to "
    "wait for the reply."
)
LEADER_DELEGATE = (
    "How this project runs: assign work with talk, then call terminate to wait; "
    "replies wake you up. When you have what you need, call submit_rating("
    "rating, reasoning); it records review.txt for you and ends the project."
)
MEMBER_RULES = (
    "How to work: the project input is already above, so you do not need to "
    "read it from a file. Do your job, then send your result with talk (to "
    "whoever your instructions name, otherwise to {leader}) and call terminate "
    "in the same response. If your instructions mention writing a file, also "
    "put the key result in your talk message so nobody has to open the file."
)


def _clip(text, limit: int = MAX_TOOL_OUTPUT) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...(truncated)...\n" + text[-half:]


class Session:
    """Everything that belongs to one sample's run."""

    def __init__(
        self, params: Params, task_context: str, sample_id: Optional[str] = None
    ):
        self.params = params
        self.task_context = task_context
        self.id = sample_id or uuid.uuid4().hex[:10]
        self.agents: Dict[str, "Agent"] = {}
        self.leader: Optional["Agent"] = None
        self.workspace = Workspace()
        self.cond = threading.Condition()
        self.result = None  # (rating, reasoning, submitted_by)
        self.error: Optional[str] = None
        self.stopped = False
        self.t0 = time.monotonic()
        self._events: List[dict] = []
        self._events_lock = threading.Lock()
        self._chroma = None

    # ---- lifecycle -------------------------------------------------------
    def submit(self, rating: int, reasoning: str, by: str) -> bool:
        with self.cond:
            accepted = self.result is None and not self.stopped
            if accepted:
                self.result = (rating, reasoning, by)
            self.stopped = True
            self.cond.notify_all()
        return accepted

    def fail(self, message: str) -> None:
        with self.cond:
            if self.error is None and self.result is None:
                self.error = message
            self.stopped = True
            self.cond.notify_all()

    def notify(self) -> None:
        with self.cond:
            self.cond.notify_all()

    def all_idle(self) -> bool:
        return all(a.state == "idle" for a in list(self.agents.values()))

    def wait(self, timeout: float) -> bool:
        """Block until the run stops or every agent is idle."""
        with self.cond:
            return self.cond.wait_for(lambda: self.stopped or self.all_idle(), timeout)

    # ---- team ------------------------------------------------------------
    def add_agent(self, name: str, prompt: str, is_leader: bool = False) -> "Agent":
        agent = Agent(name, prompt, self.params, self, is_leader=is_leader)
        self.agents[name] = agent
        if is_leader:
            self.leader = agent
        return agent

    def lookup(self, name: str) -> Optional["Agent"]:
        name = str(name).strip()
        if name in self.agents:
            return self.agents[name]
        low = name.lower()
        for n, a in self.agents.items():
            if n.lower() == low:
                return a
        return None

    def roster(self) -> str:
        return ", ".join(
            f"{n} (leader)" if a.is_leader else n for n, a in self.agents.items()
        )

    def build_system_prompt(self, agent: "Agent") -> str:
        parts = [
            agent.initial_message.strip(),
            f"Team: {self.roster()}",
            f"PROJECT INPUT (shared with the whole team):\n{self.task_context}",
        ]
        if agent.is_leader:
            parts.append(
                LEADER_PARALLEL if self.params.parallel_kickoff else LEADER_DELEGATE
            )
        else:
            parts.append(
                MEMBER_RULES.format(
                    leader=self.leader.name if self.leader else "the leader"
                )
            )
        parts.append(COMMON_RULES)
        return "\n\n".join(parts)

    # ---- bookkeeping -----------------------------------------------------
    def log(self, agent: str, message: dict) -> None:
        event = {
            "t": round(time.monotonic() - self.t0, 3),
            "agent": agent,
            "role": message.get("role"),
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            event["tool_calls"] = [
                {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
                for tc in message["tool_calls"]
            ]
        with self._events_lock:
            self._events.append(event)

    def transcript(self) -> List[dict]:
        with self._events_lock:
            return list(self._events)

    def collection(self, agent_name: str):
        """Per-session chroma collection (only when use_retrieval_memory)."""
        if self._chroma is None:
            import chromadb  # lazy: import is slow and only needed here

            self._chroma = chromadb.EphemeralClient()
        return self._chroma.get_or_create_collection(
            name=f"s{self.id}_{agent_name}"[:63]
        )

    def totals(self):
        agents = list(self.agents.values())
        return (
            sum(a.total_input_token for a in agents),
            sum(a.total_output_token for a in agents),
            sum(a.n_calls for a in agents),
        )


class Memory:
    def __init__(
        self, agent_name: str, initial_message: str, session: Session, params: Params
    ):
        self.name = agent_name
        self.initial_message = initial_message
        self.session = session
        self.params = params
        self.history: List[dict] = []
        self.subordinates: Dict[str, str] = {}
        self._pool = (
            session.collection(agent_name) if params.use_retrieval_memory else None
        )

    def add_memory(self, memory: dict) -> None:
        self.history.append(memory)
        self.session.log(self.name, memory)
        if (
            self._pool is not None
            and memory.get("role") not in ("tool", "function")
            and memory.get("content")
        ):
            self._pool.add(documents=[memory["content"]], ids=[uuid.uuid4().hex])

    def get(self, max_memory: int) -> List[dict]:
        system = self.session.build_system_prompt(self)
        ws = self.session.workspace

        if self.params.track_todos:
            todo, _, _ = ws.read(f"todo_{self.name}.txt")
            done, _, _ = ws.read(f"status_{self.name}.txt")
            if todo:
                system += f"\n\nYour current TODO list: {todo}"
            if done:
                system += f"\n\nWhat you have done so far: {done}"

        if self.subordinates:
            system += "\n\nYour subordinates (already added): " + ", ".join(
                f"{n}: {d}" if d else n for n, d in self.subordinates.items()
            )

        if self._pool is not None and self.history and self.history[-1].get("content"):
            hit = self._pool.query(
                query_texts=[self.history[-1]["content"]], n_results=1
            )
            if hit and hit.get("documents") and hit["documents"][0]:
                system += f"\n\nA relevant memory:\n{hit['documents'][0][0]}"

        return [{"role": "system", "content": system}] + self.history[-max_memory:]


class Agent(Memory):
    def __init__(
        self,
        name: str,
        initial_message: str,
        params: Params,
        session: Session,
        is_leader: bool = False,
    ):
        super().__init__(name, initial_message, session, params)
        self.is_leader = is_leader
        self.tools = build_tools(is_leader, params)
        self.queue: List[dict] = []
        self.state = "idle"
        self.hold = False  # buffer messages without waking (leader in parallel mode)
        self.lock = threading.Lock()
        self.total_input_token = 0
        self.total_output_token = 0
        self.n_calls = 0

    # ---- scheduling ------------------------------------------------------
    def enqueue(self, speaker: str, message: str) -> None:
        with self.lock:
            self.queue.append({"role": speaker, "content": message})
            start = self._claim_start()
        if start:
            self._spawn()

    def release(self, message: Optional[str] = None) -> None:
        """Lift the hold and wake up with everything that was buffered."""
        with self.lock:
            self.hold = False
            if message:
                self.queue.append({"role": "user", "content": message})
            start = self._claim_start()
        if start:
            self._spawn()

    def _claim_start(self) -> bool:
        # Called with self.lock held. Flipping to "running" here, synchronously,
        # is what makes Session.all_idle() trustworthy: a recipient is already
        # "running" before the sender can go idle.
        if (
            self.state == "idle"
            and not self.hold
            and self.queue
            and not self.session.stopped
        ):
            self.state = "running"
            return True
        return False

    def _spawn(self) -> None:
        threading.Thread(
            target=self._run_safe, daemon=True, name=f"{self.session.id}-{self.name}"
        ).start()

    def _run_safe(self) -> None:
        try:
            while True:
                with self.lock:
                    # Checking the queue and going idle under one lock closes the
                    # lost-wakeup window of the original run()/enqueue() pair.
                    if not self.queue or self.session.stopped:
                        self.state = "idle"
                        break
                    batch, self.queue = self.queue, []
                self._handle(batch)
        except Exception as e:  # noqa: BLE001 - must never leave the run hanging
            with self.lock:
                self.state = "idle"
            self.session.fail(f"{self.name}: {type(e).__name__}: {e}")
        finally:
            self.session.notify()

    # ---- one wake-up -----------------------------------------------------
    def _call(self, req: List[dict]) -> dict:
        out = get_llm_response(
            req,
            self.params,
            agent_name=self.name,
            tools=self.tools,
            tool_choice=self.params.tool_choice,
        )
        self.total_input_token += out.input_token
        self.total_output_token += out.output_token
        self.n_calls += 1
        return out.response["choices"][0]["message"]

    def _handle(self, batch: List[dict]) -> None:
        req = self.get(self.params.max_memory)
        for m in batch:
            self.add_memory(m)
        req += batch

        for _ in range(self.params.max_rounds):
            if self.session.stopped:
                return
            msg = self._call(req)
            if self.session.stopped:  # e.g. timed out while we were waiting
                return
            self.add_memory(msg)
            req.append(msg)

            calls = msg.get("tool_calls") or []
            if not calls:
                nudge = {"role": "user", "content": NO_CALL_MSG}
                self.add_memory(nudge)
                req.append(nudge)
                continue

            # Execute EVERY call in the reply (the original dropped the last
            # reply's calls when max_rounds ran out, and stopped at terminate
            # even if it came before other calls).
            ended = False
            for tc in calls:
                name = (tc.get("function") or {}).get("name") or ""
                if name == "terminate":
                    ended = True
                    result = "Turn ended. You will be woken when a message arrives."
                else:
                    result = self._execute(
                        name, tc.get("function", {}).get("arguments")
                    )
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": result,
                }
                self.add_memory(tool_msg)
                req.append(tool_msg)
                if self.session.stopped:  # submit_rating succeeded
                    return
            if ended:
                return

    # ---- tools -----------------------------------------------------------
    def _execute(self, tool_name: str, raw_args) -> str:
        try:
            args = (
                json.loads(raw_args)
                if isinstance(raw_args, str) and raw_args.strip()
                else (raw_args or {})
            )
        except json.JSONDecodeError:
            return "Error: the arguments were not valid JSON."
        if not isinstance(args, dict):
            args = {}
        handler = {
            "read_file": self._t_read,
            "write_file": self._t_write,
            "talk": self._t_talk,
            "submit_rating": self._t_submit,
            "change_task_status": self._t_todo,
            "add_agent": self._t_add_agent,
        }.get(tool_name.lower())
        if handler is None:
            return f"Error: '{tool_name}' is not an available function."
        try:
            return _clip(handler(args))
        except Exception as e:  # tool errors go back to the model, not up the stack
            return f"Error: {type(e).__name__}: {e}"

    def _t_read(self, args) -> str:
        filename = args.get("filename", "")
        ws = self.session.workspace
        content, version, author = ws.read(filename)
        if content is None:
            return f"Error: {filename} does not exist. Existing files: {ws.list() or 'none'}"
        if not self.params.share_file and author not in (self.name, "", "system"):
            return f"Error: {filename} belongs to {author}; ask them with talk."
        return f"{filename} (version {version}):\n{content}"

    def _t_write(self, args) -> str:
        filename, content = args.get("filename"), args.get("content")
        if not filename or content is None:
            return "Error: filename and content are required."
        ok, current, version = self.session.workspace.write(
            filename, str(content), args.get("base_version") or None, author=self.name
        )
        if ok:
            return f"Saved {filename} (version {version})."
        return (
            f"Conflict: {filename} already exists at version {version}. To overwrite, "
            f'call write_file again with base_version="{version}". Current content:\n{_clip(current, 600)}'
        )

    def _t_talk(self, args) -> str:
        to = args.get("to") or []
        if isinstance(to, str):
            to = [t.strip() for t in to.replace(";", ",").split(",")]
        message = str(args.get("message", "")).strip()
        if not to or not message:
            return "Error: both 'to' and 'message' are required."
        sent, unknown = [], []
        for name in to:
            target = self.session.lookup(name)
            if target is None:
                unknown.append(name)
            elif target is not self:
                target.enqueue("user", f"{self.name}: {message}")
                sent.append(target.name)
        result = []
        if sent:
            result.append(
                f"Sent to {', '.join(sent)}. If you are waiting for a reply, call terminate now."
            )
        if unknown:
            result.append(
                f"Error: unknown teammate(s) {unknown}. Team: {self.session.roster()}"
            )
        return " ".join(result) or "Nothing sent (you cannot message yourself)."

    def _t_submit(self, args) -> str:
        if not self.is_leader:
            return "Error: only the leader can submit the rating. Send your result to the leader with talk."
        try:
            rating = int(str(args.get("rating")).strip())
        except (TypeError, ValueError):
            return "Error: rating must be an integer from 1 to 5."
        if not 1 <= rating <= 5:
            return "Error: rating must be an integer from 1 to 5."
        reasoning = " ".join(str(args.get("reasoning", "")).split())
        self.session.workspace.put(
            "review.txt", f"rating: {rating}, reasoning: {reasoning}", author="system"
        )
        if self.session.submit(rating, reasoning, by=self.name):
            return "Recorded. The project is complete."
        return "The project has already ended."

    def _t_todo(self, args) -> str:
        ws = self.session.workspace
        ws.put(f"todo_{self.name}.txt", str(args.get("todo", "")), author=self.name)
        ws.put(f"status_{self.name}.txt", str(args.get("done", "")), author=self.name)
        return "Success."

    def _t_add_agent(self, args) -> str:
        if len(self.subordinates) >= self.params.max_subordinates:
            return f"Error: you already have {len(self.subordinates)} subordinates (the maximum)."
        base = (
            "".join(ch for ch in str(args.get("name", "")) if ch.isalnum() or ch == "_")
            or "Agent"
        )
        name, i = base, 2
        while self.session.lookup(name) is not None:
            name, i = f"{base}{i}", i + 1
        prompt = (
            str(args.get("initial_prompt", "")) + f"\nYour supervisor is: {self.name}"
        )
        self.session.add_agent(name, prompt)
        self.subordinates[name] = str(args.get("description", ""))
        note = "" if name == base else f" ('{base}' was taken.)"
        return f"Added {name}.{note} Talk to them to start their work."

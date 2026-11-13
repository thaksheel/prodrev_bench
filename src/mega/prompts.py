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
NO_CALL_MSG = (
    "[System] Error: respond only with function calls. "
    "If you have nothing left to do, call terminate."
)
PLAN_REQUEST = "Create the agents and assign the tasks for them."

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


class MegaPrompt:
    """Builds the one-off planning prompt in which the leader designs the team."""

    def __init__(self):
        self.task_ins: str = None
        self.leader_name: str = None

    def get_mega_instructions(
        self, leader_name: str, task_ins: str, max_agents: int = 3
    ) -> str:
        self.leader_name = leader_name
        self.task_ins = task_ins
        return f"""You are {leader_name}, the leader of a highly effective company. Your company's current goal is to {task_ins}
You are now recruiting employees and assigning work to them. For each employee (including yourself), write a prompt that states their name (one word, no prefix), their job, and exactly what they must do. You MUST state all their collaborators' names and jobs in the prompt. Use exactly this format (the example is for Alice in an unrelated novel-writing project):

<agent name="Alice">
You are Alice, a novelist. Your job is to write a single chapter of a novel with 1000 words according to the outline from Carol, the architect designer, and send it to David, the editor. Please only follow this routine. Your collaborators include Bob (the Boss), Carol (the architect designer) and David (the editor).
</agent>

Every employee is lazy and will not do anything that is not stated in their prompt, so each job must be non-divisible, concrete and short. Together the employees (including yourself) must cover the whole SOP, and {leader_name} makes the final decision. Use between 2 and {max_agents} employees in total, including yourself: every extra employee adds time and cost. Keep each prompt under 80 words."""

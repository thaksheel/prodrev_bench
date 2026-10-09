import logging
import os
import time
import re
from typing import List
from pathlib import Path

from . import get_llm_response, agent_dict, git_lock, Agent, git_commit, Params, MASOut


class Runner:
    def __init__(self, log_path: str, display: bool = False):
        self.log_path = log_path
        self.display = display
        self.init_files()

    def init_logger(self):
        logger = logging.getLogger()
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

        formatter = logging.Formatter("%(asctime)s: %(message)s")
        f_handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

        if self.display:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)

    def close_loggers(self):
        # Close handlers on the root logger and every named logger
        loggers = [logging.getLogger()] + [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]
        for lg in loggers:
            for handler in lg.handlers[:]:
                handler.close()
                lg.removeHandler(handler)

    def init_files(self):
        git_commit("Initial commit")
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def clear_dir(self, folder_path: str):
        folder = Path(folder_path)
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()

    def init_agents(self, agents_instructions: str, param: Params, task_prompt: str):
        agent_pattern = re.compile(r'<agent name="(\w+)">(.*?)</agent>', re.DOTALL)
        agent_names_tasks = agent_pattern.findall(agents_instructions)
        for name, task in agent_names_tasks:
            if name == param.ceo_name:
                agent_dict[name] = Agent(name, task, param)
        for name, task in agent_names_tasks:
            if name != param.ceo_name:
                agent_dict[param.ceo_name].add_subordinate(
                    name=name,
                    description="",
                    initial_prompt=task,
                    params=param,
                    additional_prompt="",
                )
        ceo_agent = [a for a in agent_dict.values() if a.name == param.ceo_name][0]
        agents: List[Agent] = [ceo_agent]
        agents += [a for a in agent_dict.values() if a.name != param.ceo_name]
        ceo_agent.enqueue(
            "user",
            f"Now complete this task:\n\n{task_prompt}\n\n"
            "Coordinate with your subordinates as needed, and make sure the requested "
            "deliverable is written to the output file specified in your task instructions.",
        )
        return agents

    def is_idle(self, agents: List[Agent]):
        return all([agent.state == "idle" for agent in agents])

    def is_todos_complete(self, agents: List[Agent]):
        no_todos = True
        for agent in agents:
            try:
                with git_lock:
                    f = open(f"files/todo_{agent.name}.txt", "r")
                    content = f.read()
                    todos_to_complete = True if content else False
                    f.close()
            except FileNotFoundError:
                todos_to_complete = False
            if todos_to_complete is True:
                agent.enqueue(
                    "system",
                    "Other agents have terminated. However, you still have unfinished tasks in your TODO list. Please finish them and clear it. If you are waiting for someone, chances are that they have forgotten about you. Please remind them.",
                )
                no_todos = False
        return no_todos

    def wait_for_idle(self, agent: Agent, timeout: float = 60.0):
        deadline = time.monotonic() + timeout
        while agent.state != "idle":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.1, remaining))
        return True

    def run(self, param: Params, task_instruction: str, prompt: str):
        self.close_loggers()
        self.clear_dir(folder_path="./files")
        self.clear_dir(folder_path="./logs")
        self.init_logger()
        messages = [
            {"role": "system", "content": task_instruction},
            {
                "role": "user",
                "content": "Create the agents and assign the tasks for them.",
            },  # temp
        ]
        out = get_llm_response(
            messages=messages,
            enable_tools=False,
            params=param,
        )
        agents_instructions = out.message
        agents = self.init_agents(agents_instructions, param, task_prompt=prompt)
        start = time.time()
        while True:  # Wait until all agents become idle
            time.sleep(1)
            if self.is_idle(agents):
                no_todos = self.is_todos_complete(agents)
                if no_todos:
                    agents[0].enqueue(
                        speaker="system",
                        message=(
                            "All the agents have terminated. Please use read_file to browse"
                            "and proofread all the output files. Be sure to test them if needed, and ."
                            "check whether the project has been completed(do not leave placeholders!) "
                            "If the project is completed with total accuracy, please call the 'terminate' function;"
                            "if not, please assign the remaining tasks."
                        ),
                    )
                    if not self.wait_for_idle(agent=agents[0], timeout=60.0):
                        raise TimeoutError(
                            "The lead agent did not become idle within 60 seconds."
                        )
                    tit = sum([a.total_input_token for a in agents])
                    tot = sum([a.total_output_token for a in agents])
                    with open("./files/review.txt", encoding="cp1252") as f:
                        text = f.read()
                    pattern = r"rating:\s*(\d+),\s*reasoning:\s*(.*)"
                    match = re.search(pattern, text, re.DOTALL)
                    if not match:
                        raise ValueError(
                            "format of final rating prediction is not consistent."
                        )
                    rating = int(match.group(1))
                    if not 1 <= rating <= 5:
                        raise ValueError(
                            f"Predicted rating must be between 1 and 5; got {rating}."
                        )
                    reasoning = match.group(2).strip()
                    return MASOut(
                        pred=rating,
                        true=None,
                        reasoning=reasoning,
                        total_input_token=tit,
                        total_output_token=tot,
                        duration=time.time() - start,
                    )

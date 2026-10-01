import logging
import os
import time
import re

from . import (
    get_llm_response,
    agent_dict,
    git_lock,
    Agent,
    git_commit,
    Params,
)


class Runner:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.init_files()

    def init_logger(self):
        logger = logging.getLogger()
        formatter = logging.Formatter("%(asctime)s: %(message)s")
        f_handler = logging.FileHandler(self.log_path)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    def init_files(self):
        git_commit("Initial commit")
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        self.init_logger()

    def run(self, param: Params, task_instruction: str, prompt: str):
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
        agents_instructions = out['choices'][0]["message"]["content"]
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
        agent_dict[param.ceo_name].enqueue(
            "user",
            "Now let's start the project. Please split the task and talk to your subordinates to assign the tasks.",
        )
        # Wait until all agents become idle
        while True:
            time.sleep(1)
            if all(agent.state == "idle" for agent in agent_dict.values()):
                ok = True
                for agent in agent_dict.values():
                    try:
                        with git_lock:
                            f = open(f"files/todo_{agent.name}.txt", "r")
                            content = f.read()
                            f.close()
                    except FileNotFoundError:
                        content = ""
                    if content != "":
                        agent.enqueue(
                            "system",
                            "Other agents have terminated. However, you still have unfinished tasks in your TODO list. Please finish them and clear it. If you are waiting for someone, chances are that they have forgotten about you. Please remind them.",
                        )
                        ok = False
                if not ok:
                    continue
                else:
                    agent_dict[param.ceo_name].enqueue(
                        "system",
                        r"All the agents have terminated. Please use read_file to browse and proofread all the output files. Be sure to test them if needed, and check whether the project has been completed(do not leave placeholders!). If the project is completed with 100% accuracy, please call the 'terminate' function; if not, please assign the remaining tasks.",
                    )
                    while agent_dict[param.ceo_name].state != "idle":
                        time.sleep(1)
                    if all(agent.state == "idle" for agent in agent_dict.values()):
                        break
                    else:
                        continue

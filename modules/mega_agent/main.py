import logging
import os
import time
import re

from . import (
    get_llm_response,
    agent_dict,
    git_lock,
    Agent,
    input_token,
    output_token,
    git_commit,
    delete_all_files_in_folder,
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
        # delete_all_files_in_folder("logs")
        # delete_all_files_in_folder("files")
        git_commit("Initial commit")
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        self.init_logger()

    def run(self, param: Params, prompt: str):
        begin_time = time.time()
        meta_output = get_llm_response(
            messages=[{"role": "system", "content": prompt}],
            enable_tools=False,
            params=param,
        )["choices"][0]["message"]["content"]
        logging.info(meta_output)
        agent_pattern = re.compile(r'<agent name="(\w+)">(.*?)</agent>', re.DOTALL)
        agents = agent_pattern.findall(meta_output)
        for agent in agents:
            if agent[0] == param.ceo_name:
                agent_dict[agent[0]] = Agent(agent[0], agent[1], param)

        for agent in agents:
            if agent[0] != param.ceo_name:
                agent_dict[param.ceo_name].add_subordinate(
                    name=agent[0],
                    description="",
                    initial_prompt=agent[1],
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
        end_time = time.time()
        logging.info(f"Time elapsed: {end_time-begin_time} seconds")
        logging.info(f"Input tokens: {input_token}, output tokens: {output_token}")
        logging.info(f"Number of agents: {len(agent_dict)}")

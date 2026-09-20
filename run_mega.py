import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

from modules import Runner, Params

load_dotenv()
api_key = os.getenv("OPENAI_KEY")

params = Params(
    api_key=api_key,
    model="gpt-5.6-luna",
    reasoning_effort="none",
    max_memory=5,
    max_rounds=2,
    max_subordinates=2,
    share_file=True,
    ceo_name="Bob",
)
runner = Runner(log_path="./exports/logs.txt")
prompt = r"""
    You are Bob, the leader of a software development club. Your club's current goal is to develop a Gobang game with a very strong AI, no frontend, and can be executed by running 'main.py'. Remember to test it. You are now recruiting employees and assigning work to them. For each employee(including yourself), please write a prompt. Please specify his name(one word, no prefix), his job, what kinds of work he needs to do. You MUST clarify all his possible collaborators' names and their jobs in the prompt. The format should be like (The example is for Alice in another novel writing project):

    <agent name="Alice">
    You are Alice, a novelist. Your job is to write a single chapter of a novel with 1000 words according to the outline (outline.txt) from Carol, the architect designer, and pass it to David (chapter_x.txt), the editor. Please only follow this routine. Your collarborators include Bob(the Boss), Carol(the architect designer) and David(the editor).
    </agent>

    Please note that every employee is lazy, and will not care anything not mentioned by your prompt. To ensure the completion of your project, the work of each employee should be **non-divisable**, detailed in specific action(like what file to write. Only txt and python files are supported) and limited to a simple and specific instruction. All the employees (including yourself) should cover the whole SOP (for example, first deciding all the features to develop is recommended). Speed up the process by adding more employees to divide the work.
"""
runner.run(params, prompt=prompt)


print("END")

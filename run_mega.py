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
# NOTE: the prompt has a specific format before I can use it.
product_review = """I like it fine. The color is very very subtle. I don't think I will purchase this again."""  # 5
prompt: str = f"""

"""

runner.run(params, prompt=prompt)


print("END")

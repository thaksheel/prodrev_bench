import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import os

from modules import Runner, Params
from modules.mega_agent import MegaPrompt
from src import CloseModelSelection

load_dotenv()
api_key = os.getenv("OPENAI_KEY")
filename = "./data/sr.csv"
df = pd.read_csv(filename)
df = df.dropna(subset=["review"])
_, sample = train_test_split(df, test_size=200, random_state=42, stratify=df.rating)
sentences = sample.review.tolist()
groundtruth = sample.rating.to_numpy()

leader_name = "Erza"
params = Params(
    api_key=api_key,
    model=CloseModelSelection.gpt6_luna.value,
    reasoning_effort="none",
    max_memory=5,
    max_rounds=2,
    max_subordinates=2,
    share_file=True,
    ceo_name=leader_name,
)
runner = Runner(log_path="./exports/logs.out")
mp = MegaPrompt()
product_review = """I like it fine. The color is very very subtle. I don't think I will purchase this again."""  # 5
task_ins = "You are leading a product text reviews analysis agency and you need to recruit agents with different tasks and analyse the product reviews affectively. You need to rate the online product reviews text from 1-5 as accurately as possible in order to capture the sentiment of the customers."
prompt = f"score this customer online product review from 1-5: {product_review}"

out = runner.run(
    params,
    task_instruction=mp.get_mega_intrustions(leader_name, task_ins),
    prompt=prompt,
)

print(out)
print("END")

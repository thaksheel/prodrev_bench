import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import os

from modules import Runner, Params
from modules.mega_agent import MegaPrompt, MASOut
from src import CloseModelSelection

load_dotenv()
api_key = os.getenv("OPENAI_KEY")
filename = "./data/sr.csv"
size = 10
df = pd.read_csv(filename)
df = df.dropna(subset=["review"])
_, sample = train_test_split(df, test_size=size, random_state=42, stratify=df.rating)
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
task_ins = (
    "You are leading a product review analysis agency. Evaluate the supplied review "
    "and predict its overall star rating from 1 to 5 using only the review text. "
    "Write the final rating and a brief evidence-based rationale to review.txt. "
    "Do not use any label or comment outside the review text as evidence. "
    "Submit the final review answer in the format rating: int, reasoning: str"
)
outs: list[MASOut] = []
for true, review in zip(groundtruth, sentences):
    prompt = f"score this customer online product review from 1-5: <review> {review} </review>"
    out = runner.run(
        params,
        task_instruction=mp.get_mega_intrustions(leader_name, task_ins),
        prompt=prompt,
    )
    out.true = true
    outs.append(out) 
    print(out)
df_results = pd.DataFrame(outs)
df_results.to_excel(f"./exports/mega_results_{size}.xlsx")

print(f"input_tokens={sum([o.total_input_token for o in outs])}")
print(f"output_tokens={sum([o.total_output_token for o in outs])}")
print(f"duration={sum([o.duration for o in outs])}")
print("END")

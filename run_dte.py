import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

from modules.debate_train_evolve import dte
from quick_analysis import evaluate_ratings

df = pd.read_csv("./data/sr.csv")
df = df.dropna(subset=["review"])
_, sample = train_test_split(df, test_size=30, random_state=42, stratify=df.rating)
sentences = sample.review.tolist()
groundtruth = sample.rating.to_numpy()

model_name = "Qwen/Qwen2.5-7B-Instruct"
preds = []
for s in tqdm(sentences, total=len(sentences)):
    pred = dte.debate(
                query=s,
                model=model_name,
                num_agents=3,
                max_rounds=3,
                task_type="general",
            ).final_answer
    preds.append(int(pred))

df_results = pd.DataFrame(
    {
        "rating": groundtruth,
        "results": preds,
        "sentence": sentences,
    }
)
df_results.to_excel("./exports/rslt_qwen25_dte0.xlsx")
df_r = evaluate_ratings(df_results)
df_r["len"] = [len(df_results)] * len(df_r)

print(f"df_head={df_r.head()}")
print("END")
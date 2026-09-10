import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from sklearn.model_selection import train_test_split

from src.debate import Debate

modelname = "meta-llama/Llama-2-70b-hf"
modelname = "meta-llama/Llama-2-7b-hf"
modelname = "meta-llama/Llama-3.1-8B-Instruct"
modelname = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
modelname = "Qwen/Qwen2.5-7B-Instruct"
with open("./exports/references.json", "r", encoding="utf-8") as f:
    references = json.load(f)
filename = "./data/sr.csv"
df = pd.read_csv(filename)
df = df.dropna(subset=["review"])
_, sample = train_test_split(df, test_size=6000, random_state=42, stratify=df.rating)
sentences = sample.review.tolist()
groundtruth = sample.rating.to_numpy()
debate = Debate(
    model_name=modelname,
    device="cuda",
    max_new_token=1000,
    num_agents=5,
    num_rounds=3,
    references=references,
)
final_cls, reasoning = debate.start_debate(sentences[0])
results, reasonings = debate.simulate_debate(sentences)
df_results = pd.DataFrame({
    "rating": groundtruth, 
    "results": results, 
    "sentence": sentences, 
    "reasoning": reasonings,
})
df_results.to_excel("./exports/results1.xlsx")
results = np.array(results)
accurate = (groundtruth == results).sum()
debate.save_reasoning()
print(f"---> accuracy={accurate/len(groundtruth):.4f}")
print("END")

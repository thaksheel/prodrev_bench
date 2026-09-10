import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import torch 
import json 

from src.debate import Debate

modelname = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
with open("./exports/references.json", "r", encoding="utf-8") as f:
    references = json.load(f)
filename = "./data/sr.csv"
df = pd.read_csv(filename)
df = df.dropna(subset=["review"]).sample(100)
sentences = df.review.tolist()
groundtruth = df.rating.to_numpy()
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
results = np.array(results)
accurate = (groundtruth == results).sum()
debate.save_reasoning()
print(f"---> accuracy={accurate:.4f}")
print("END")

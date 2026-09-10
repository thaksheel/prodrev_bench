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
sentences = df.dropna(subset=["review"]).sample(100).review.tolist()
debate = Debate(
    model_name=modelname,
    device="cuda",
    max_new_token=1000,
    num_agents=5,
    num_rounds=4,
    references=references,
)
final_cls, reasoning = debate.start_debate(sentences[0])
results, reasonings = debate.simulate_debate(sentences)
debate.save_reasoning()
print(final_cls, reasoning)
print("END")

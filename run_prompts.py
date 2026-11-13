import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

from src import PromptStrategy, Prompt 

df = pd.read_csv("./data/sr.csv")
prompt = Prompt(
    review="<TEST_REVIEW>", 
    chain_of_thought=True, 
    num_examples=10, 
)
ps = PromptStrategy(
    prompt=prompt, 
    strategy="few-shots", 
    df=df, 
    few_shots_type="random", 
)
print(ps.get_prompt(df))
import numpy as np 
import pandas as pd 

from .modules.debate_train_evolve import dte 

model_name = "Qwen/Qwen2.5-7B-Instruct"
result = dte.debate(
    query="Why is the sky blue", 
    model=model_name, 
    num_agents=3, 
    max_rounds=3,
    task_type="general",
)

print(result.final_answer)
print(result.consensus_reached)

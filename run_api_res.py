import numpy as np 
import pandas as pd 
from dotenv import load_dotenv
import os 

from src import ApiResponse 


load_dotenv()
apikey = os.getenv("OPENAI_KEY")
ar = ApiResponse(
    model_name="gpt-5.6-luna", 
    provider="openai", 
    api_key=apikey,
)
messages = [{"role": "user", "content": "why is the sky blue"}] 
response = ar.response(messages)

print(response)
print("END")

import numpy as np 
import pandas as pd 
from typing import List, Dict 
from numpy.typing import NDArray
from enum import Enum 
from dataclasses import dataclass


class OpenModelSelection(Enum):
    tllama = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    qwen38_27b = "Qwen/Qwen3.8-27B"
    llama31_8b = "meta-llama/Llama-3.1-8B-Instruct"
    qwen25_7b = "Qwen/Qwen2.5-7B-Instruct"
    qwen3_8b = "Qwen/Qwen3-8B"


class CloseModelSelection(Enum):
    gpt6_luna = "gpt-6-luna"
    gpt6_sol = "gpt-6-sol"
    gpt6_astra = "gpt-6-astra"
    gpt56_sol = "gpt-5.6-sol"
    gpt56_luna = "gpt-5.6-luna"
    gpt56_terra = "gpt-5.6-terra"


@dataclass
class LLMOut:
    response: str
    input_token: int
    output_token: int
    model_name: str
    provider: str

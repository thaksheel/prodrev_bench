import numpy as np
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass
import pandas as pd


class Memory:
    def __init__(
        self,
    ):
        self.instruction: str = None 
        self.initial_stance: str = None 
        self.references: List[str] = None 
        self.arguments_histroy: List[str] = [] 

    def get_prompt(self):
        pass 


if __name__ == "__main__":
    pass

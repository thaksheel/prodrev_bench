import time
import os
import random
import numpy as np
from typing import Literal, Dict, List, Optional

from llm_response import LLMResponse


class Agent:
    def __init__(
        self,
        model_name: str,
        device: Literal["cpu", "cuda"],
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.memory_lst = []

        # private fields
        self.llmr = LLMResponse(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    def query(
        self,
        messages: List[Dict],
    ) -> str:
        response = self.llmr.response(messages)
        return response

    def set_meta_prompt(self, meta_prompt: str):
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_event(self, event: str):
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})

    def ask(self):
        return self.query(self.memory_lst)

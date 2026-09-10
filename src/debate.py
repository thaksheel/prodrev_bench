import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import torch
from typing import Literal, Dict, List, Optional, Tuple

from agent import Agent
from llm_response import LLMResponse


class Debate:
    def __init__(
        self,
        model_name: str,
        device: Literal["cpu", "cuda"],
        max_new_token: int,
        num_agents: int,
        num_rounds: int,
        references: Dict[str, List[str]],
    ):
        self.num_rounds = num_rounds
        self.num_agents = num_agents
        self.references = references
        # post init
        self.llm = LLMResponse(
            model_name=model_name,
            max_new_tokens=max_new_token,
            device=torch.device(device),
        )
        # null fields
        self.agents: List[Agent] = []
        self.judge: Agent = None

    def init_agent(self):
        self.agents = [
            Agent(
                agent_id=i,
                cls_lvl=i + 1,
                instructions=self.references["instructions"][i],
                prompt_rebuddle_tmpl=self.references["rebuddle_prompt_template"][i],
                prompt_tmpl=self.references["prompt_template"][i],
                references=self.references["references"][i],
                examples=self.references["examples"][str(i + 1)],
                llm_responder=self.llm,
                total_rounds=self.num_rounds,
            )
            for i in range(self.num_agents)
        ]
        return self

    def judgment(self, text: str) -> Tuple[int, str]:
        history = ""
        for r in range(self.num_rounds):
            his = [
                f"debater {agent.agent_id} opinion: {agent.opinions[r]}"
                for agent in self.agents
            ]
            history += f"round {r} ".join(his)
        judge = Agent(
            agent_id=-1,
            cls_lvl=1,
            instructions=None,
            prompt_tmpl=None,
            prompt_rebuddle_tmpl=None,
            references=None,
            examples=None,
            llm_responder=self.llm,
            total_rounds=self.num_rounds,
        )
        final_cls, reasoning = judge.final_judgement(text, history)
        return final_cls, reasoning

    def start_debate(self, text: str):
        self.init_agent()
        for r in range(self.num_rounds):
            rebuddle = None
            for agent in self.agents:
                if r > 0:
                    rebuddle = [agent.opinions[-1] for agent in self.agents]
                    rebuddle = " ".join(rebuddle)
                a = agent.debate(text, round_num=r, rebuddle=rebuddle)
        final_cls, reasoning = self.judgment(text)
        return final_cls, reasoning

    def simulate_debate(self, sentences: List[str]):
        results = []
        reasonings = []
        for text in sentences:
            r, re = self.start_debate(text)
            results.append(r)
            reasonings.append(re)
        return results, reasonings

    def save_reasoning(self):
        pass




from typing import Dict, List, Tuple
import json

from .llm_response import LLMResponse


class Agent:
    def __init__(
        self,
        agent_id: str,
        total_rounds: int,
        cls_lvl: int,
        instructions: str,
        prompt_rebuddle_tmpl: str,
        prompt_tmpl: str,
        references: str,
        examples: List[str],
        llm_responder: LLMResponse,
    ) -> None:
        self.cls_lvl = cls_lvl
        self.prompt_tmpl = prompt_tmpl
        self.prompt_rebuddle_tmpl = prompt_rebuddle_tmpl
        self.instructions = instructions
        self.references = references
        self.examples = examples
        self.agent_id = agent_id
        self.llm_responder = llm_responder

        # null fields
        self.opinions: List[str] = []
        self.memory_collection: Dict[int, List[str]] = dict(
            zip(
                [i for i in range(total_rounds)],
                [[] for _ in range(total_rounds)],
            )
        )

    def get_response(
        self,
        messages: List[Dict],
    ) -> str:
        response = self.llm_responder.response(messages)
        return response

    def add_instruction(self, round_num: int, ins: str):
        """Set instructions for the current task at hand, stored as `system`"""
        prompt = self.instructions.replace("##reference##", self.references)
        self.memory_collection[round_num].append(
            {"role": "system", "content": f"{prompt}"}
        )

    def add_topic(self, text: str, round_num: int):
        """Add prompts/events to generate a response by the LLM, stored at `user`"""
        prompt = self.prompt_tmpl.replace("##text##", text)
        self.memory_collection[round_num].append(
            {"role": "user", "content": f"{prompt}"}
        )

    def add_rebuddle(self, rebuddle: str, round_num: int):
        """Add prompts/events to generate a response by the LLM, stored at `user`"""
        prompt = self.prompt_tmpl.replace("##rebuddle##", rebuddle)
        self.memory_collection[round_num].append(
            {"role": "user", "content": f"{prompt}"}
        )

    def add_opinion(self, opinion: str, round_num: int):
        """Store the response by the LLMs as `assistant`"""
        self.memory_collection[round_num].append(
            {"role": "assistant", "content": f"{opinion}"}
        )

    def debate(self, text: str, rebuddle: str, round_num: int):
        if round_num == 0:
            self.add_instruction(round_num, text)
            self.add_topic(text, round_num)
            self.opinions.append(self.get_response(self.memory_collection[round_num]))
            self.add_opinion(self.opinions[-1], round_num)
        else:
            self.add_instruction(round_num, text)
            self.add_rebuddle(rebuddle, round_num)
            self.opinions.append(self.get_response(self.memory_collection[round_num]))
            self.add_opinion(self.opinions[-1], round_num)
        return self.opinions[-1]

    def final_judgement(self, text: str, history: str) -> Tuple[int, str]:
        prompt = """You are the Sentiment Rating Judge. The debaters have completed their arguments.
Review Text:
"##text##"

Debate History:
##history##

Your task:
1. Evaluate all arguments from the debaters.
2. Determine which star rating (1-5) is best supported by the evidence and reasoning.
3. Provide a final judgment strictly in JSON format.

JSON Output Format:
{"Rating": <1-5>, "Reason": "<one-sentence explanation>"}

Important:
- The "Rating" must be a single integer from 1 to 5.
- The "Reason" must be one concise sentence summarizing why that rating is correct.
- Do NOT include any text outside the JSON object.
"""
        prompt = prompt.replace("##text##", text)
        prompt = prompt.replace("##history##", history)
        messages = [
            {"role": "system", "content": "follow the given instructions."}, 
            {"role": "user", "content": f"{prompt}"},
        ]
        out = self.get_response(messages)
        start = out.find("{")
        end = out.rfind("}") + 1
        json_str = out[start:end]
        try: 
            d = json.loads(json_str)
            final_cls, reasoning = int(d["Rating"]), d["Reason"]
            return final_cls, reasoning
        except json.decoder.JSONDecoder: 
            print(f"---> `out` from final_judgement={out}")
            return None 

    def clear_memory(self):
        self.memory_collection: Dict[int, List[str]] = {}
        return self

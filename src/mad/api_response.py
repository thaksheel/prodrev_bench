from typing import List, Dict, Optional, Literal
import asyncio 
import json 
import litellm 


class ApiResponse: 
    def __init__(self, model_name: Literal["gpt-5.6-luna", "gpt-6", "gpt-5.6-sol"], api_key: str, provider: Literal["openai", "anthropic", "gemini"]):
        self.model_name = model_name
        self.api_key = api_key
        self.provider = provider

    def build_prompt(self):
        pass 

    def response(self, messages: Dict[str, str]): 
        """Synchronous response using messages in the format [{role: user, content: prompt}]"""
        response = litellm.completion(
            model=self.model_name, 
            messages=messages, 
            api_key=self.api_key, 
        )
        r = response.choices[0].message.content 
        return r 

    async def response_batch(self, messages_batch: List[Dict[str, str]]):
        """Async response using different providers."""
        # TODO: complete the implementation later to collect the batches and parse the responses needed z
        with open("prompt.jsonl", "w") as f:
            for i, messages in enumerate(messages_batch): 
                record = {
                    "custom_id": f"prompt-{i}", 
                    "method": "POST", 
                    "url": "v1/chat/completions", 
                    "body": {
                        "model": self.model_name, 
                        "messages": messages,
                    },
                }
                f.write(json.dumps(record) + "\n")
        with open("prompt.jsonl", "rb") as f:
            uploaded = await litellm.acreate_file(
                file=f, 
                purpose="batch", 
                custom_llm_provider=self.provider, 
            )
        batch = await litellm.acreate_batch(
            completion_window="24h", 
            endpoint="/v1/chat/completions", 
            input_file_id=uploaded.id, 
            custom_llm_provider=self.provider, 
        )
        return batch 
import numpy as np
import torch
from typing import Literal, Tuple, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)


class LLMResponse:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        device: Literal["cpu", "cuda"],
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self.initialize()

    def initialize(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        return model, tokenizer 

    def manual_build_prompt(self, messages: Dict[str, str]): 
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>\n{content}\n"
            elif role == "user":
                text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}\n"
            else:
                raise ValueError(f"Unknown role: {role}")
        # The final assistant tag tells the model to continue
        text += "<|assistant|>\n"
        return text

    def build_prompt(self, messages: Dict[str, str]):
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # fallback to manual serialization
            return self.manual_build_prompt(messages)

    def response(self, messages: Dict[str, str]):
        prompt = self.build_prompt(messages=messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return output


if __name__ == "__main__":
    modelname = "meta-llama/Llama-2-70b-hf"
    modelname = "Qwen/Qwen2.5-7B-Instruct"
    modelname = "meta-llama/Llama-2-7b-hf"
    modelname = "meta-llama/Llama-3.1-8B-Instruct"
    modelname = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llme = LLMResponse(
        model_name=modelname,
        max_new_tokens=2048,
        device="cpu",
    )
    out = llme.response(prompt="why is the sky blue")
    print(out)

import os
import torch
from typing import Literal, Optional, Dict, Union, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLLM

from . import LLMOut


class LLMProvider:
    def __init__(
        self,
        provider: Literal["hf", "openai"],
        model_name: str,
        token: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        load_in_4bit: bool = True,
    ):
        self.provider = provider
        self.model_name = model_name
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_4bit = load_in_4bit

        # null fields
        self.tokenizer = None
        self.llm: BaseLLM = None

        if provider == "hf":
            self.llm = self.init_hf(token)
        elif provider == "openai":
            self.llm = self.init_openai(token)

    def invoke(self, messages: List[Dict[str, str]], prompt: str = None) -> LLMOut:
        if prompt is None:
            if self.provider == "hf":
                p = self.build_prompt(messages)
                response = self.llm.invoke(p)
                return LLMOut(
                    response=response,
                    input_token=len(str(p)) / 4,
                    output_token=len(response) / 4,
                    model_name=self.model_name,
                    provider=self.provider,
                )
            elif self.provider == "openai":
                response = self.llm.invoke(messages)
                usage = response.response_metadata["token_usage"]
                response = response.content
                return LLMOut(
                    response=response,
                    input_token=usage["prompt_tokens"],
                    output_token=usage["completion_tokens"],
                    model_name=self.model_name,
                    provider=self.provider,
                )
            else:
                raise NotImplementedError
        else:
            response = self.llm.invoke(prompt)
            return LLMOut(
                response=response,
                input_token=len(prompt) / 4,  # crude approx.
                output_token=len(response) / 4,  # crude approx.
                model_name=self.model_name,
                provider=self.provider,
            )

    def init_openai(self, token: str):
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=token,
        )

    def init_hf(self, token):
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=pipe)

    def manual_build_prompt(self, messages: List[Dict[str, str]]):
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
            return self.manual_build_prompt(messages)



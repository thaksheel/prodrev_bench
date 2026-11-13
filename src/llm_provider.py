import torch
from typing import Literal, Optional, Dict, Union, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    TokenizersBackend,
)
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLLM

from . import LLMOut


class ThinkingBudge(LogitsProcessor):
    """Caps the <think> section of reasoning models inside ONE generate() call.
    Once thinking reaches `budget` tokens it forces Qwen's stop-thinking phrase
    and </think>, so the rest of max_new_tokens goes to the final answer."""

    STOP = (
        "\n\nConsidering the limited time by the user, I have to give the "
        "solution based on the thinking directly now.\n"
    )

    def __init__(self, tokenizer: TokenizersBackend, budget: int):
        self.start = tokenizer.convert_tokens_to_ids("<think>")
        self.end = tokenizer.convert_tokens_to_ids("</think>")
        self.budget = budget
        self.forced = tokenizer.encode(self.STOP, add_special_tokens=False) + [self.end]

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            if self.start not in ids:
                continue
            s = len(ids) - 1 - ids[::-1].index(self.start)
            if self.end in ids[s + 1 :]:
                continue
            k = len(ids) - s - 1 - self.budget
            if k >= 0:
                scores[b, :] = float("-inf")
                scores[b, self.forced[min(k, len(self.forced) - 1)]] = 0.0
        return scores


class LLMProvider:

    def __init__(
        self,
        provider: Literal["hf", "openai", "anthropic"],
        model_name: str,
        token: str,
        max_new_tokens: int = 2048,
        temperature: float = 1,
        load_in_4bit: bool = True,
        answer_reserve: int = 1024,
        reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "low",
        reasoning_summary: Literal["auto", "concise", "detailed"] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_4bit = load_in_4bit
        self.answer_reserve = answer_reserve
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary

        # null fields
        self.tokenizer: TokenizersBackend = None
        self.llm: BaseLLM = None

        if provider == "hf":
            self.llm = self.init_hf(token)
        elif provider == "openai":
            self.llm = self.init_openai(token)
        elif provider == "anthropic":
            self.llm = self.init_anthropic(token)

    @staticmethod
    def _text(msg) -> str:
        """Final answer text, with any thinking summary wrapped in <think> tags."""
        if isinstance(msg.content, str):
            return msg.content
        thinking, text = [], []
        for b in msg.content:
            if not isinstance(b, dict):
                continue
            if b.get("type") in ("text", "output_text"):
                text.append(b.get("text", ""))
            elif b.get("type") == "thinking" and b.get("thinking"):
                thinking.append(b["thinking"])
        answer = "".join(text)
        if not thinking:
            return answer
        joined = "\n".join(thinking)
        return f"<think>\n{joined}\n</think>\n\n{answer}"

    def invoke(self, messages: List[Dict[str, str]], prompt: str = None) -> LLMOut:
        if prompt is None:
            if self.provider == "hf":
                p = self.build_prompt(messages)
                response = self.llm.invoke(p)
                return LLMOut(
                    response=response,
                    input_token=len(self.tokenizer(str(p)).input_ids),
                    output_token=len(self.tokenizer(response).input_ids),
                    model_name=self.model_name,
                    provider=self.provider,
                )
            elif self.provider == "openai":
                msg = self.llm.invoke(messages)
                usage = msg.usage_metadata or {}
                details = usage.get("output_token_details") or {}
                return LLMOut(
                    response=self._text(msg),
                    input_token=usage.get("input_tokens", 0),
                    output_token=usage.get("output_tokens", 0),
                    model_name=self.model_name,
                    provider=self.provider,
                    reasoning_token=details.get("reasoning", 0),
                )
            elif self.provider == "anthropic":
                msg = self.llm.invoke(messages)
                usage = msg.usage_metadata or {}
                details = usage.get("output_token_details") or {}
                return LLMOut(
                    response=self._text(msg),
                    input_token=usage.get("input_tokens", 0),
                    output_token=usage.get("output_tokens", 0),
                    model_name=self.model_name,
                    provider=self.provider,
                    reasoning_token=details.get("reasoning", 0),
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
        kwargs = {"model": self.model_name, "api_key": token}
        REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
        is_reasoning = self.model_name.startswith(REASONING_PREFIXES)
        if is_reasoning and self.reasoning_effort is not None:
            reasoning = {"effort": self.reasoning_effort}
            if self.reasoning_summary:
                reasoning["summary"] = self.reasoning_summary
            kwargs["reasoning"] = reasoning
            kwargs["use_responses_api"] = True

        # reasoning models reject temperature unless reasoning is off
        if not is_reasoning or self.reasoning_effort == "none":
            kwargs["temperature"] = self.temperature
        return ChatOpenAI(**kwargs)

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
        gen_kwargs = {}
        vocab = self.tokenizer.get_vocab()
        if "<think>" in vocab and "</think>" in vocab:
            budget = self.max_new_tokens - self.answer_reserve
            gen_kwargs["logits_processor"] = LogitsProcessorList(
                [ThinkingBudge(self.tokenizer, budget)]
            )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            return_full_text=False,
            **gen_kwargs,
        )
        return HuggingFacePipeline(pipeline=pipe)

    def init_anthropic(self, token: str):
        NO_SAMPLING_PREFIXES = (
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-5",
            "claude-fable-5",
        )
        kwargs = {
            "model": self.model_name,
            "api_key": token,
            "max_tokens": self.max_new_tokens,
        }
        if self.reasoning_effort is not None:
            kwargs["effort"] = self.reasoning_effort
        if self.reasoning_summary:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if not self.model_name.startswith(NO_SAMPLING_PREFIXES):
            kwargs["temperature"] = self.temperature
        return ChatAnthropic(**kwargs)

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



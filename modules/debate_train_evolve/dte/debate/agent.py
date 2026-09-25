"""
Individual debate agent implementation with model weight sharing.

This module implements individual agents that participate in multi-agent debates.
When multiple agents use the same model (the default DTE paradigm), they
**share a single loaded model instance** through a module-level registry,
avoiding redundant memory usage.

The model registry is a simple cache keyed by ``(model_name, device)`` that
keeps a reference count. Models are automatically unloaded when no agents
reference them.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .prompts import DebatePromptManager, DebateResponse

# ---------------------------------------------------------------------------
# Module-level model registry for weight sharing
# ---------------------------------------------------------------------------


class _ModelRegistry:
    """Thread-safe registry that caches loaded models for weight sharing.

    Models are keyed by ``(model_name, device_str)`` and reference-counted.
    When the reference count drops to zero the model is released.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # key -> {"model": model, "tokenizer": tokenizer, "refcount": int}
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def acquire(
        self,
        model_name: str,
        device: torch.device,
        model_config: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """Load or retrieve a cached ``(model, tokenizer)`` pair.

        If the model is already loaded for the given device it is reused and
        the reference count is incremented. Otherwise the model is freshly
        loaded.

        Args:
            model_name: HuggingFace model name or local path.
            device: Target device.
            model_config: Extra kwargs forwarded to ``from_pretrained``.

        Returns:
            ``(model, tokenizer)`` tuple.
        """
        key = (model_name, str(device))
        with self._lock:
            if key in self._cache:
                self._cache[key]["refcount"] += 1
                return self._cache[key]["model"], self._cache[key]["tokenizer"]

        # Load outside the lock (may take a while)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_config.get("tokenizer", {}),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
            "device_map": "auto" if device.type == "cuda" else None,
            "trust_remote_code": True,
            **model_config.get("model", {}),
        }
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if model_kwargs.get("device_map") is None:
            model = model.to(device)
        model.eval()

        with self._lock:
            # Another thread may have loaded in parallel; prefer the earlier one
            if key in self._cache:
                self._cache[key]["refcount"] += 1
                return self._cache[key]["model"], self._cache[key]["tokenizer"]
            self._cache[key] = {
                "model": model,
                "tokenizer": tokenizer,
                "refcount": 1,
            }
        return model, tokenizer

    def release(self, model_name: str, device: torch.device) -> None:
        """Decrement the reference count and free memory when it hits zero.

        Args:
            model_name: HuggingFace model name or local path.
            device: Device the model was loaded on.
        """
        key = (model_name, str(device))
        with self._lock:
            if key not in self._cache:
                return
            self._cache[key]["refcount"] -= 1
            if self._cache[key]["refcount"] <= 0:
                entry = self._cache.pop(key)
                del entry["model"]
                del entry["tokenizer"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# Singleton registry shared by all agents
_model_registry = _ModelRegistry()


# ---------------------------------------------------------------------------
# DebateAgent
# ---------------------------------------------------------------------------


class DebateAgent:
    """Individual agent for multi-agent debate.

    Each agent maintains its own conversation history and performance
    statistics. When multiple agents share the same ``model_name`` and
    ``device``, a single model instance is loaded and shared across all
    of them via :class:`_ModelRegistry`, dramatically reducing GPU memory
    consumption for the standard DTE paradigm.

    Args:
        agent_id: Unique identifier for this agent (e.g. ``"1"``).
        model_name: HuggingFace model name or local path.
        device: Computation device (``"auto"``, ``"cpu"``, ``"cuda"``,
            ``"cuda:N"``, ``"mps"``).
        model_config: Extra configuration forwarded to model/tokenizer
            loading. May contain ``"tokenizer"`` and ``"model"`` sub-dicts.
        generation_config: Parameters for text generation (temperature,
            top_p, top_k, max_length, etc.).
    """

    def __init__(
        self,
        agent_id: str,
        model_name: str,
        device: str = "auto",
        model_config: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.device = self._setup_device(device)

        self.model_config = model_config or {}
        self.generation_config = generation_config or {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": None,
        }

        # Acquire shared model
        self.model, self.tokenizer = _model_registry.acquire(self.model_name, self.device, self.model_config)
        self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        # Agent state
        self.response_history: List[DebateResponse] = []
        self.current_round = 0

        # Prompt manager
        self.prompt_manager = DebatePromptManager()

        # Performance tracking
        self.generation_times: List[float] = []
        self.token_counts: List[int] = []

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_device(device: str) -> torch.device:
        """Resolve a device specification to a :class:`torch.device`."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def generate_initial_response(self, query: str, task_type: str = "math") -> DebateResponse:
        """Generate the initial response for round 0.

        Args:
            query: The problem/question to solve.
            task_type: Task type (``"math"``, ``"arc"``, ``"general"``).

        Returns:
            Structured :class:`DebateResponse`.
        """
        prompt = self.prompt_manager.create_initial_prompt(query, task_type)
        response_text = self._generate_text(prompt)
        response = self.prompt_manager.parse_response(
            response_text, agent_id=self.agent_id, round_num=0, task_type=task_type
        )
        self.response_history.append(response)
        self.current_round = 0
        return response

    def generate_debate_response(
        self,
        query: str,
        answers_so_far: Dict[str, str],
        round_num: int,
        task_type: str = "math",
    ) -> DebateResponse:
        """Generate a debate response for round > 0.

        Args:
            query: Original problem/question.
            answers_so_far: Mapping of agent IDs to their previous reasoning.
            round_num: Current round number (1-indexed).
            task_type: Task type.

        Returns:
            Structured :class:`DebateResponse`.
        """
        prompt = self.prompt_manager.create_debate_prompt(
            query=query,
            agent_id=self.agent_id,
            round_num=round_num,
            answers_so_far=answers_so_far,
            task_type=task_type,
        )
        response_text = self._generate_text(prompt)
        response = self.prompt_manager.parse_response(
            response_text,
            agent_id=self.agent_id,
            round_num=round_num,
            task_type=task_type,
        )
        self.response_history.append(response)
        self.current_round = round_num
        return response

    def _generate_text(self, prompt: str) -> str:
        """Generate text using the (shared) language model.

        Args:
            prompt: Input prompt for generation.

        Returns:
            Generated text response (stripped).
        """
        start_time = time.time()

        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            max_gen_length = self.generation_config.get("max_length", 2048)
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_gen_length - 500,
                padding=False,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_cfg = GenerationConfig(**self.generation_config)
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_cfg,
                    return_dict_in_generate=True,
                    output_scores=False,
                )

            generated_tokens = outputs.sequences[0][len(inputs["input_ids"][0]) :]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
            self.token_counts.append(len(generated_tokens))

            return response_text.strip()

        except Exception as e:
            print(f"Generation error for agent {self.agent_id}: {e}")
            return "Error: Failed to generate response"

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_current_response(self) -> Optional[DebateResponse]:
        """Return the most recent response, or ``None``."""
        return self.response_history[-1] if self.response_history else None

    def get_response_history(self) -> List[DebateResponse]:
        """Return a copy of the complete response history."""
        return self.response_history.copy()

    def reset_history(self) -> None:
        """Clear the agent's debate history for a fresh debate."""
        self.response_history.clear()
        self.current_round = 0

    # ------------------------------------------------------------------
    # Performance stats
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> Dict[str, Any]:
        """Return generation performance statistics.

        Returns:
            Dictionary with timing and token count metrics.
        """
        if not self.generation_times:
            return {
                "agent_id": self.agent_id,
                "total_generations": 0,
                "avg_generation_time": 0.0,
                "total_generation_time": 0.0,
                "avg_tokens_generated": 0.0,
                "total_tokens_generated": 0,
                "model_name": self.model_name,
                "device": str(self.device),
            }

        return {
            "agent_id": self.agent_id,
            "total_generations": len(self.generation_times),
            "avg_generation_time": sum(self.generation_times) / len(self.generation_times),
            "total_generation_time": sum(self.generation_times),
            "avg_tokens_generated": sum(self.token_counts) / len(self.token_counts) if self.token_counts else 0,
            "total_tokens_generated": sum(self.token_counts),
            "model_name": self.model_name,
            "device": str(self.device),
        }

    def update_generation_config(self, new_config: Dict[str, Any]) -> None:
        """Update generation parameters (e.g. temperature) for this agent.

        Args:
            new_config: Key-value pairs to merge into the generation config.
        """
        self.generation_config.update(new_config)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release this agent's reference to the shared model."""
        _model_registry.release(self.model_name, self.device)

    def __repr__(self) -> str:
        return (
            f"DebateAgent(id={self.agent_id}, model={self.model_name}, "
            f"round={self.current_round}, responses={len(self.response_history)})"
        )

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

import logging
import time
from functools import lru_cache

from . import LLMOut, Params, chat_completion, wrap_tools

logger = logging.getLogger("mega_agent")


class LLMError(RuntimeError):
    pass


_READ = {
    "name": "read_file",
    "description": "Read a team file. Returns its content and version.",
    "parameters": {
        "type": "object",
        "properties": {"filename": {"type": "string"}},
        "required": ["filename"],
    },
}
_WRITE = {
    "name": "write_file",
    "description": "Write a .txt file. To overwrite an existing file, pass the version you got from read_file as base_version.",
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {"type": "string"},
            "content": {"type": "string"},
            "base_version": {
                "type": "string",
                "description": "Required only to overwrite.",
            },
        },
        "required": ["filename", "content"],
    },
}
_TALK = {
    "name": "talk",
    "description": "Send a message to one or more teammates. They reply later by message.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact teammate names.",
            },
            "message": {"type": "string"},
        },
        "required": ["to", "message"],
    },
}
_TERMINATE = {
    "name": "terminate",
    "description": "End your turn. You will be woken up when a new message arrives.",
}
_SUBMIT = {
    "name": "submit_rating",
    "description": "Leader only. Submit the team's final star rating; this records review.txt and ends the project.",
    "parameters": {
        "type": "object",
        "properties": {
            "rating": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "reasoning": {
                "type": "string",
                "description": "Brief evidence from the review text (max 2 sentences).",
            },
        },
        "required": ["rating", "reasoning"],
    },
}
_TODO = {
    "name": "change_task_status",
    "description": "Update your TODO list (unfinished tasks) and done list. Clear the TODO list when finished.",
    "parameters": {
        "type": "object",
        "properties": {"todo": {"type": "string"}, "done": {"type": "string"}},
        "required": ["todo", "done"],
    },
}
_ADD_AGENT = {
    "name": "add_agent",
    "description": "Recruit a subordinate. initial_prompt must give their one-word name, job, exact task, and collaborators' exact names.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "initial_prompt": {"type": "string"},
        },
        "required": ["name", "description", "initial_prompt"],
    },
}


@lru_cache(maxsize=None)
def _tools(is_leader: bool, track_todos: bool, allow_recruiting: bool):
    bare = [_TALK, _READ, _WRITE, _TERMINATE]
    if is_leader:
        bare = [_SUBMIT] + bare
    if track_todos:
        bare.append(_TODO)
    if allow_recruiting:
        bare.append(_ADD_AGENT)
    return tuple(wrap_tools(bare))


def build_tools(is_leader: bool, params: Params):
    return list(_tools(is_leader, params.track_todos, params.allow_recruiting))


def get_llm_response(
    messages,
    params: Params,
    enable_tools: bool = True,
    agent_name: str = "",
    tools=None,
    tool_choice=None,
) -> LLMOut:
    """One request with retry/backoff. Raises LLMError instead of returning
    an error dict, so a failing call can never leave an agent half-dead."""
    tools = tools if enable_tools else None
    last_err = "unknown error"
    for attempt in range(max(1, params.llm_retries)):
        resp = chat_completion(
            messages,
            params.model,
            tools,
            api_key=params.api_key,
            base_url=params.base_url,
            reasoning_effort=params.reasoning_effort,
            tool_choice=tool_choice if tools else None,
            temperature=params.temperature,
            seed=params.seed,
        )
        if "error" not in resp and resp.get("choices"):
            usage = resp.get("usage") or {}
            return LLMOut(
                response=resp,
                message=resp["choices"][0]["message"].get("content"),
                input_token=usage.get("prompt_tokens") or 0,
                output_token=usage.get("completion_tokens") or 0,
                model_name=params.model,
                provider="openai",
            )
        last_err = resp.get("error", "response had no choices")
        logger.warning(
            f"[{agent_name or 'planner'}] LLM attempt {attempt + 1} failed: {last_err}"
        )
        time.sleep(min(0.5 * 2**attempt, 4.0))
    raise LLMError(f"{agent_name or 'planner'}: {last_err}")

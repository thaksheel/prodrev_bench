"""
RCR (Reflect-Critique-Refine) prompting system for DTE multi-agent debate.

This module implements the three-phase RCR prompting strategy described in the
DTE paper (EMNLP 2025). Each debate round after round 0 follows three explicit
phases:

1. **Reflect** -- The agent reflects on its own previous reasoning, identifying
   strengths and weaknesses.
2. **Critique** -- The agent critiques exactly 2 peer solutions with structured
   feedback covering correctness, reasoning quality, and completeness.
3. **Refine** -- The agent synthesizes all feedback (self-reflection and peer
   critiques) to produce a refined answer.

The initial round (round 0) uses a standard problem-solving prompt without
RCR phases.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..utils.answer_extraction import extract_final_answer


@dataclass
class DebateResponse:
    """Structured response from a debate agent.

    Attributes:
        answer: The raw answer text from the agent.
        reasoning: The full reasoning/response text.
        extracted_answer: The extracted final answer (e.g. numeric value).
        confidence: Optional confidence score (0-1).
        round_number: The debate round this response belongs to.
        agent_id: Identifier for the agent that produced this response.
    """

    answer: str
    reasoning: str
    extracted_answer: str
    confidence: Optional[float] = None
    round_number: int = 0
    agent_id: str = ""


class DebatePromptManager:
    """Manager for RCR (Reflect-Critique-Refine) debate prompts.

    This class implements the core RCR prompting innovation from the DTE paper.
    For round 0 it produces a standard problem-solving prompt. For subsequent
    rounds it produces a three-phase prompt that explicitly asks the agent to:

    1. Reflect on its own previous reasoning
    2. Critique exactly 2 peer solutions
    3. Refine its answer based on all feedback

    The number of critique targets is configurable but defaults to 2 as
    specified in the paper.

    Args:
        critique_pairs: Number of peer solutions each agent must critique
            per round. Defaults to 2 per the paper specification.
    """

    def __init__(self, critique_pairs: int = 2):
        """Initialize the RCR debate prompt manager.

        Args:
            critique_pairs: Number of peer solutions to critique per round.
                Must be at least 1. Defaults to 2 as in the DTE paper.

        Raises:
            ValueError: If critique_pairs is less than 1.
        """
        if critique_pairs < 1:
            raise ValueError(f"critique_pairs must be at least 1, got {critique_pairs}")
        self.critique_pairs = critique_pairs

    # ------------------------------------------------------------------
    # Initial prompt (round 0)
    # ------------------------------------------------------------------

    def create_initial_prompt(self, query: str, task_type: str = "math") -> str:
        """Generate the initial prompt for round 0 of debate.

        Round 0 does not use RCR -- agents solve the problem independently.

        Args:
            query: The question or problem to solve. For ARC tasks this
                should be a dict with ``question`` and ``choices`` keys.
            task_type: One of ``"math"``, ``"arc"``, or ``"general"``.

        Returns:
            Formatted initial prompt string.

        Raises:
            ValueError: If *task_type* is not recognized.
        """
        if task_type == "math":
            return self._create_math_initial_prompt(query)
        elif task_type == "arc":
            return self._create_arc_initial_prompt(query)
        elif task_type == "general":
            return self._create_general_initial_prompt(query)
        else:
            raise ValueError(f"Unknown task_type '{task_type}'. Expected one of: 'math', 'arc', 'general'.")

    def _create_math_initial_prompt(self, question: str) -> str:
        """Create initial prompt for mathematical reasoning tasks."""
        return (
            f"Can you solve this math problem?\n"
            f"Your final answer must be in the format \\boxed{{answer}} at the end.\n\n"
            f"Problem: {question}"
        )

    def _create_arc_initial_prompt(self, question_data) -> str:
        """Create initial prompt for ARC-Challenge tasks.

        Args:
            question_data: Either a string or a dict with ``question``,
                ``choices`` (with ``text`` and ``label`` lists) keys.
        """
        if isinstance(question_data, str):
            return (
                f"Answer the following multiple-choice question.\n"
                f"Read the question and all choices carefully before answering.\n\n"
                f"{question_data}\n\n"
                f"Please provide your reasoning and then give your final answer "
                f"in the format: Answer: [letter]"
            )

        question = question_data.get("question", "")
        choices = question_data.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])

        choices_text = ""
        for label, choice in zip(labels, texts):
            choices_text += f"{label}. {choice}\n"

        return (
            f"Answer the following multiple-choice question from the ARC Challenge dataset.\n"
            f"Read the question and all choices carefully before answering.\n\n"
            f"Question: {question}\n\n"
            f"Choices:\n{choices_text}\n"
            f"Please provide your reasoning and then give your final answer "
            f"in the format: Answer: [letter]"
        )

    def _create_general_initial_prompt(self, query: str) -> str:
        """Create initial prompt for general reasoning tasks."""
        return (
            f"Please solve the following problem step by step.\n"
            f"Provide clear reasoning and a definitive final answer.\n\n"
            f"Problem: {query}"
        )

    # ------------------------------------------------------------------
    # RCR debate prompt (round > 0)
    # ------------------------------------------------------------------

    def create_debate_prompt(
        self,
        query: str,
        agent_id: str,
        round_num: int,
        answers_so_far: Dict[str, str],
        task_type: str = "math",
    ) -> str:
        """Generate an RCR (Reflect-Critique-Refine) prompt for rounds > 0.

        The prompt contains three clearly delineated phases:

        1. **Reflect** -- Agent examines its own previous reasoning.
        2. **Critique** -- Agent critiques up to ``self.critique_pairs`` peer
           solutions with structured feedback.
        3. **Refine** -- Agent synthesizes all feedback into a refined answer.

        Args:
            query: The original question/problem.
            agent_id: ID of the current agent (e.g. ``"1"``).
            round_num: Current debate round (1-indexed).
            answers_so_far: Mapping of agent IDs to their most recent
                reasoning text.
            task_type: One of ``"math"``, ``"arc"``, or ``"general"``.

        Returns:
            Formatted RCR debate prompt string.
        """
        if task_type == "math":
            return self._create_math_rcr_prompt(query, agent_id, round_num, answers_so_far)
        elif task_type == "arc":
            return self._create_arc_rcr_prompt(query, agent_id, round_num, answers_so_far)
        else:
            return self._create_general_rcr_prompt(query, agent_id, round_num, answers_so_far)

    # -- Math RCR prompt --

    def _create_math_rcr_prompt(
        self,
        question: str,
        agent_id: str,
        round_num: int,
        answers_so_far: Dict[str, str],
    ) -> str:
        """Build the three-phase RCR prompt for math tasks."""
        # Separate own and peer responses
        own_previous = answers_so_far.get(agent_id, "")
        own_answer = extract_final_answer(own_previous) if own_previous else "N/A"

        peer_ids = [pid for pid in answers_so_far if pid != agent_id]
        # Select up to critique_pairs peers
        critique_targets = peer_ids[: self.critique_pairs]

        # Build peer section
        peer_section = ""
        for pid in critique_targets:
            peer_text = answers_so_far[pid]
            peer_ans = extract_final_answer(peer_text)
            peer_section += f"--- Agent {pid} ---\nSolution: {peer_text}\nExtracted answer: {peer_ans}\n\n"

        # Build the remaining peers as context (not required to critique)
        remaining_peers = [pid for pid in peer_ids if pid not in critique_targets]
        context_section = ""
        if remaining_peers:
            context_section = "\nAdditional peer solutions (for reference):\n"
            for pid in remaining_peers:
                peer_text = answers_so_far[pid]
                peer_ans = extract_final_answer(peer_text)
                context_section += f"Agent {pid} answer: {peer_ans}\n"

        prompt = f"""You are Agent {agent_id} in a multi-agent debate (round {round_num}).

Problem: {question}

Your previous solution:
{own_previous}

Your previous extracted answer: {own_answer}

=== PHASE 1: REFLECT ===
Carefully reflect on your own reasoning above. Identify any errors, gaps, or weak assumptions in your solution. Be honest about what you got right and what might be wrong.

=== PHASE 2: CRITIQUE ===
Now critique the following peer solutions. For each peer, evaluate:
- Is their final answer correct? Why or why not?
- Are there errors in their reasoning steps?
- Do they use any insights or approaches you missed?

{peer_section}{context_section}
=== PHASE 3: REFINE ===
Based on your self-reflection and peer critiques, produce your refined solution. If you still believe your original answer is correct, defend it with stronger reasoning. If you found errors, correct them.

Your final answer must be in the format \\boxed{{answer}} at the end."""

        return prompt

    # -- ARC RCR prompt --

    def _create_arc_rcr_prompt(
        self,
        question_data,
        agent_id: str,
        round_num: int,
        answers_so_far: Dict[str, str],
    ) -> str:
        """Build the three-phase RCR prompt for ARC tasks."""
        # Handle both string and dict question formats
        if isinstance(question_data, dict):
            question = question_data.get("question", "")
            choices = question_data.get("choices", {})
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choices_text = ""
            for label, choice in zip(labels, texts):
                choices_text += f"{label}. {choice}\n"
        else:
            question = question_data
            choices_text = ""

        own_previous = answers_so_far.get(agent_id, "")
        own_answer = self._extract_arc_answer(own_previous) if own_previous else "N/A"

        peer_ids = [pid for pid in answers_so_far if pid != agent_id]
        critique_targets = peer_ids[: self.critique_pairs]

        peer_section = ""
        for pid in critique_targets:
            peer_text = answers_so_far[pid]
            peer_ans = self._extract_arc_answer(peer_text)
            peer_section += f"--- Agent {pid} ---\nReasoning: {peer_text}\nAnswer: {peer_ans}\n\n"

        prompt = f"""You are Agent {agent_id} in a multi-agent debate (round {round_num}).

Question: {question}

Choices:
{choices_text}

Your previous reasoning:
{own_previous}

Your previous answer: {own_answer}

=== PHASE 1: REFLECT ===
Reflect on your own reasoning above. Did you consider all relevant scientific principles? Are there any logical gaps or misinterpretations of the question?

=== PHASE 2: CRITIQUE ===
Critique the following peer responses. For each, evaluate whether their scientific reasoning is sound and their choice is justified.

{peer_section}
=== PHASE 3: REFINE ===
Synthesize your self-reflection and peer critiques. Provide your refined reasoning and final answer.

Please give your final answer in the format: Answer: [letter]"""

        return prompt

    # -- General RCR prompt --

    def _create_general_rcr_prompt(
        self,
        query: str,
        agent_id: str,
        round_num: int,
        answers_so_far: Dict[str, str],
    ) -> str:
        """Build the three-phase RCR prompt for general reasoning tasks."""
        own_previous = answers_so_far.get(agent_id, "")

        peer_ids = [pid for pid in answers_so_far if pid != agent_id]
        critique_targets = peer_ids[: self.critique_pairs]

        peer_section = ""
        for pid in critique_targets:
            peer_text = answers_so_far[pid]
            peer_section += f"--- Agent {pid} ---\nResponse: {peer_text}\n\n"

        prompt = f"""You are Agent {agent_id} in a multi-agent debate (round {round_num}).

Problem: {query}

Your previous response:
{own_previous}

=== PHASE 1: REFLECT ===
Reflect on your previous reasoning. What are its strengths? What might be wrong or incomplete?

=== PHASE 2: CRITIQUE ===
Critique the following peer responses. Identify errors, strong points, and novel insights.

{peer_section}
=== PHASE 3: REFINE ===
Based on your reflection and critiques, provide your refined solution. Incorporate valid insights from peers and correct any errors you identified."""

        return prompt

    # ------------------------------------------------------------------
    # Answer extraction helpers
    # ------------------------------------------------------------------

    def _extract_arc_answer(self, response: str) -> str:
        """Extract ARC answer choice from response text.

        Args:
            response: Model response containing a letter choice.

        Returns:
            Extracted letter (A-D) or ``"Unable to Extract"``.
        """
        patterns = [
            r"(?:answer|choice)(?:\s+is)?\s*:?\s*([A-D])",
            r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
            r"\\boxed\{([A-D])\}",
            r"\(([A-D])\)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[-1].upper()

        return "Unable to Extract"

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(
        self,
        response_text: str,
        agent_id: str = "",
        round_num: int = 0,
        task_type: str = "math",
    ) -> DebateResponse:
        """Parse raw agent output into a structured :class:`DebateResponse`.

        Args:
            response_text: Raw text generated by the agent.
            agent_id: Identifier of the responding agent.
            round_num: Current debate round number.
            task_type: Task type (``"math"``, ``"arc"``, or ``"general"``).

        Returns:
            A :class:`DebateResponse` with extracted answer and metadata.
        """
        if task_type == "arc":
            extracted_answer = self._extract_arc_answer(response_text)
        else:
            extracted_answer = extract_final_answer(response_text)

        return DebateResponse(
            answer=extracted_answer,
            reasoning=response_text,
            extracted_answer=extracted_answer,
            round_number=round_num,
            agent_id=agent_id,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_response_format(self, response: DebateResponse, task_type: str = "math") -> List[str]:
        """Validate that a response follows the expected format.

        Args:
            response: Parsed response to validate.
            task_type: Task type for format-specific checks.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []

        if response.extracted_answer == "Unable to Extract":
            errors.append("Could not extract answer from response")

        if not response.reasoning or len(response.reasoning.strip()) < 20:
            errors.append("Reasoning too short or missing")

        if task_type == "math":
            if "\\boxed{" not in response.reasoning:
                errors.append("Missing \\boxed{} format for math answer")
        elif task_type == "arc":
            if not re.search(r"Answer:\s*[A-D]", response.reasoning, re.IGNORECASE):
                errors.append("Missing 'Answer: [letter]' format for ARC question")

        return errors

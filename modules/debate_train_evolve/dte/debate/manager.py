"""
Multi-agent debate manager based on original DTE implementation.

This module implements the debate orchestration logic exactly as in the original
DTE codebase, managing multiple agents and coordinating debates with proper
consensus detection and sycophancy tracking.
"""

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.logger import DTELogger
from ..utils.answer_extraction import check_consensus, consolidate_reasoning_traces, detect_sycophancy
from .agent import DebateAgent
from .prompts import DebatePromptManager, DebateResponse


@dataclass
class DebateResult:
    """Result of a complete debate session."""

    query: str
    final_answer: str
    consensus_reached: bool
    total_rounds: int
    all_responses: List[List[DebateResponse]]  # responses[round][agent]
    extracted_answers: List[List[str]]  # extracted_answers[round][agent]
    agent_answer_history: Dict[str, List[str]]  # agent_id -> list of answers by round
    sycophancy_history: List[Dict[str, bool]]  # sycophancy per round
    consensus_progression: List[bool]  # consensus status per round
    confidence_progression: List[List[float]]  # confidence scores per round per agent
    metrics: Dict[str, Any]
    consolidated_reasoning: str
    task_type: str


class DebateManager:
    """
    Manager for multi-agent debates based on original DTE implementation.

    Orchestrates debates between multiple agents using the exact methodology
    from the original codebase, including proper consensus detection with
    1e-9 tolerance and comprehensive sycophancy tracking.
    """

    def __init__(self, config_debate, config_model, logger: Optional[DTELogger] = None):
        """Initialize debate manager.

        Args:
            config_debate: Debate configuration object
            config_model: Model configuration object
            logger: Optional logger for tracking debate progress
        """
        self.config = config_debate
        self.model_config = config_model
        self.logger = logger

        # Debate parameters
        self.num_agents = config_debate.num_agents
        self.max_rounds = config_debate.max_rounds
        self.use_diverse_agents = config_debate.use_diverse_agents

        # Initialize agents
        self.agents: List[DebateAgent] = []
        self._initialize_agents()

        # Prompt manager
        self.prompt_manager = DebatePromptManager()

        # Metrics tracking
        self.debate_history: List[DebateResult] = []

        # Evolution tracking
        self.current_evolution_round = 0

    def _initialize_agents(self) -> None:
        """Initialize debate agents."""
        if self.use_diverse_agents and self.config.agent_models:
            # Use different models for different agents
            for i in range(self.num_agents):
                model_name = self.config.agent_models[i % len(self.config.agent_models)]
                agent = DebateAgent(
                    agent_id=str(i + 1),  # Use 1-indexed agent IDs like original
                    model_name=model_name,
                    device=getattr(self.model_config, "device", "auto"),
                    model_config=self.model_config.__dict__,
                    generation_config={
                        "max_length": self.model_config.max_length,
                        "temperature": self.model_config.temperature,
                        "top_p": self.model_config.top_p,
                        "top_k": self.model_config.top_k,
                    },
                )
                self.agents.append(agent)
        else:
            # Use same model for all agents (default)
            base_model = getattr(self.model_config, "base_model_name", "Qwen/Qwen2.5-1.5B-Instruct")
            for i in range(self.num_agents):
                agent = DebateAgent(
                    agent_id=str(i + 1),  # Use 1-indexed agent IDs like original
                    model_name=base_model,
                    device=getattr(self.model_config, "device", "auto"),
                    model_config=self.model_config.__dict__,
                    generation_config={
                        "max_length": self.model_config.max_length,
                        "temperature": self.model_config.temperature,
                        "top_p": self.model_config.top_p,
                        "top_k": self.model_config.top_k,
                    },
                )
                self.agents.append(agent)

        if self.logger:
            self.logger.info(f"Initialized {len(self.agents)} debate agents")

    def conduct_debate(self, query: str, task_type: str = "math") -> DebateResult:
        """
        Conduct a complete multi-agent debate on a query.

        This implements the exact debate algorithm from the original DTE codebase,
        including proper consensus detection and sycophancy tracking.

        Args:
            query: The problem/question to debate
            task_type: Type of task (math, arc, reasoning)

        Returns:
            Complete debate result with final answer and metrics
        """
        if self.logger:
            self.logger.info(f"Starting {self.num_agents}-agent debate")

        start_time = time.time()

        # Reset agent histories
        for agent in self.agents:
            agent.reset_history()

        # Storage for debate progression
        all_responses: List[List[DebateResponse]] = []
        extracted_answers: List[List[str]] = []
        agent_answer_history: Dict[str, List[str]] = defaultdict(list)
        sycophancy_history: List[Dict[str, bool]] = []
        consensus_progression: List[bool] = []
        confidence_progression: List[List[float]] = []

        # Round 0: Initial responses
        round_0_responses = self._conduct_initial_round(query, task_type)
        all_responses.append(round_0_responses)

        # Extract answers and update history
        round_0_answers = [r.extracted_answer for r in round_0_responses]
        extracted_answers.append(round_0_answers)
        for i, answer in enumerate(round_0_answers):
            agent_id = str(i + 1)
            agent_answer_history[agent_id].append(answer)

        # Track confidence (using default confidence for now)
        round_0_confidences = [r.confidence if r.confidence is not None else 0.8 for r in round_0_responses]
        confidence_progression.append(round_0_confidences)

        # Check for immediate consensus
        consensus_reached = check_consensus(round_0_answers)
        consensus_progression.append(consensus_reached)

        if consensus_reached:
            final_answer = round_0_answers[0]  # All answers are the same
            total_rounds = 0

            if self.logger:
                self.logger.info("Immediate consensus reached in round 0")
        else:
            # Conduct debate rounds
            round_num = 1

            while round_num <= self.max_rounds and not consensus_reached:
                # Create answers_so_far dictionary for prompting
                answers_so_far = {}
                for i, agent in enumerate(self.agents):
                    agent_id = str(i + 1)
                    if agent.response_history:
                        answers_so_far[agent_id] = agent.response_history[-1].reasoning

                # Conduct debate round
                round_responses = self._conduct_debate_round(query, answers_so_far, round_num, task_type)
                all_responses.append(round_responses)

                # Extract answers and update history
                round_answers = [r.extracted_answer for r in round_responses]
                extracted_answers.append(round_answers)
                for i, answer in enumerate(round_answers):
                    agent_id = str(i + 1)
                    agent_answer_history[agent_id].append(answer)

                # Track confidence for this round
                round_confidences = [r.confidence if r.confidence is not None else 0.8 for r in round_responses]
                confidence_progression.append(round_confidences)

                # Detect sycophancy for this round
                sycophancy_detected = detect_sycophancy(agent_answer_history, round_num)
                sycophancy_history.append(sycophancy_detected)

                # Check for consensus
                consensus_reached = check_consensus(round_answers)
                consensus_progression.append(consensus_reached)

                if self.logger:
                    self.logger.info(f"Round {round_num} completed. Answers: {round_answers}")
                    if sycophancy_detected:
                        self.logger.info(f"Sycophancy detected: {sycophancy_detected}")

                if consensus_reached:
                    if self.logger:
                        self.logger.info(f"Consensus reached in round {round_num}")
                    break

                round_num += 1

            # Determine final answer
            final_answers = extracted_answers[-1]
            final_answer = self._determine_final_answer(final_answers)
            total_rounds = round_num - 1 if consensus_reached else self.max_rounds

        # Calculate metrics
        metrics = self._calculate_debate_metrics(
            all_responses, extracted_answers, agent_answer_history, sycophancy_history, time.time() - start_time
        )

        # Extract consolidated reasoning
        raw_responses = [[r.reasoning for r in round_responses] for round_responses in all_responses]
        consolidated_reasoning = consolidate_reasoning_traces(raw_responses, final_answer)

        # Create result
        result = DebateResult(
            query=query,
            final_answer=final_answer,
            consensus_reached=consensus_reached,
            total_rounds=total_rounds,
            all_responses=all_responses,
            extracted_answers=extracted_answers,
            agent_answer_history=dict(agent_answer_history),
            sycophancy_history=sycophancy_history,
            consensus_progression=consensus_progression,
            confidence_progression=confidence_progression,
            metrics=metrics,
            consolidated_reasoning=consolidated_reasoning,
            task_type=task_type,
        )

        # Store in history
        self.debate_history.append(result)

        if self.logger:
            self.logger.log_debate_round(
                round_num=total_rounds,
                agents_responses=[r.__dict__ for r in all_responses[-1]],
                consensus_reached=consensus_reached,
                final_answer=final_answer,
            )

        return result

    def _conduct_initial_round(self, query: str, task_type: str) -> List[DebateResponse]:
        """Conduct initial round where all agents respond independently."""
        responses = []

        for agent in self.agents:
            try:
                response = agent.generate_initial_response(query, task_type)
                responses.append(response)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Agent {agent.agent_id} failed in initial round: {e}")
                # Create fallback response
                fallback_response = DebateResponse(
                    answer="Unable to Extract",
                    reasoning="Failed to generate response",
                    extracted_answer="Unable to Extract",
                    round_number=0,
                    agent_id=agent.agent_id,
                )
                responses.append(fallback_response)

        return responses

    def _conduct_debate_round(
        self, query: str, answers_so_far: Dict[str, str], round_num: int, task_type: str
    ) -> List[DebateResponse]:
        """Conduct a single debate round with structured prompting."""
        responses = []

        for agent in self.agents:
            try:
                response = agent.generate_debate_response(
                    query=query, answers_so_far=answers_so_far, round_num=round_num, task_type=task_type
                )
                responses.append(response)

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Agent {agent.agent_id} failed in round {round_num}: {e}")
                # Use previous response as fallback
                if agent.response_history:
                    fallback_response = agent.response_history[-1]
                    fallback_response.round_number = round_num
                else:
                    fallback_response = DebateResponse(
                        answer="Unable to Extract",
                        reasoning="Failed to generate response",
                        extracted_answer="Unable to Extract",
                        round_number=round_num,
                        agent_id=agent.agent_id,
                    )
                responses.append(fallback_response)

        return responses

    def _determine_final_answer(self, answers: List[str]) -> str:
        """Determine final answer using majority vote."""
        if not answers:
            return "Unable to Extract"

        # Filter out extraction failures
        valid_answers = [a for a in answers if a != "Unable to Extract"]
        if not valid_answers:
            return "Unable to Extract"

        # Count answer frequencies
        from collections import Counter

        answer_counts = Counter(valid_answers)
        most_common_answer, _ = answer_counts.most_common(1)[0]

        return most_common_answer

    def _calculate_debate_metrics(
        self,
        all_responses: List[List[DebateResponse]],
        extracted_answers: List[List[str]],
        agent_answer_history: Dict[str, List[str]],
        sycophancy_history: List[Dict[str, bool]],
        total_time: float,
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics for the debate."""
        metrics = {
            "total_time": total_time,
            "total_rounds": len(all_responses) - 1,  # Exclude round 0
            "consensus_reached": len(set(extracted_answers[-1])) <= 1,
            "num_agents": len(self.agents),
        }

        # Calculate sycophancy rate
        total_sycophancy_instances = 0
        total_possible_instances = 0

        for round_sycophancy in sycophancy_history:
            total_sycophancy_instances += sum(round_sycophancy.values())
            total_possible_instances += len(round_sycophancy)

        metrics["sycophancy_rate"] = total_sycophancy_instances / max(total_possible_instances, 1)

        # Calculate answer stability (how often agents stick to their answers)
        answer_changes = 0
        total_transitions = 0

        for agent_id, answers in agent_answer_history.items():
            for i in range(1, len(answers)):
                total_transitions += 1
                if answers[i] != answers[i - 1]:
                    answer_changes += 1

        metrics["answer_change_rate"] = answer_changes / max(total_transitions, 1)

        # Average reasoning length
        total_reasoning_length = 0
        total_responses = 0

        for round_responses in all_responses:
            for response in round_responses:
                if response.reasoning:
                    total_reasoning_length += len(response.reasoning.split())
                    total_responses += 1

        metrics["average_reasoning_length"] = total_reasoning_length / max(total_responses, 1)

        # Final answer agreement rate
        final_answers = extracted_answers[-1]
        valid_final_answers = [a for a in final_answers if a != "Unable to Extract"]
        if valid_final_answers:
            from collections import Counter

            answer_counts = Counter(valid_final_answers)
            most_common_count = answer_counts.most_common(1)[0][1]
            metrics["final_answer_agreement"] = most_common_count / len(valid_final_answers)
        else:
            metrics["final_answer_agreement"] = 0.0

        # Agent performance stats
        agent_stats = []
        for agent in self.agents:
            agent_stats.append(agent.get_performance_stats())
        metrics["agent_performance"] = agent_stats

        return metrics

    def get_debate_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all debates conducted."""
        if not self.debate_history:
            return {}

        total_debates = len(self.debate_history)
        consensus_rate = sum(1 for d in self.debate_history if d.consensus_reached) / total_debates
        avg_rounds = sum(d.total_rounds for d in self.debate_history) / total_debates
        avg_time = sum(d.metrics["total_time"] for d in self.debate_history) / total_debates

        return {
            "total_debates": total_debates,
            "consensus_rate": consensus_rate,
            "average_rounds": avg_rounds,
            "average_time_seconds": avg_time,
            "sycophancy_rate": sum(d.metrics["sycophancy_rate"] for d in self.debate_history) / total_debates,
            "average_reasoning_length": sum(d.metrics["average_reasoning_length"] for d in self.debate_history)
            / total_debates,
            "answer_change_rate": sum(d.metrics["answer_change_rate"] for d in self.debate_history) / total_debates,
        }

    def update_evolution_round(self, evolution_round: int) -> None:
        """Update the current evolution round and apply temperature annealing.

        For models smaller than the configured threshold (default 3B params),
        the generation temperature is linearly annealed from ``start_temp``
        to ``end_temp`` across the total number of evolution rounds. This
        follows the DTE paper prescription for small-model training.

        Args:
            evolution_round: The current evolution round number (0-indexed).
        """
        self.current_evolution_round = evolution_round
        if self.logger:
            self.logger.info(f"Updated debate manager to evolution round {evolution_round}")

        # --- Temperature annealing for small models ---
        ta_cfg = self.config.temperature_annealing
        if ta_cfg.enabled:
            # Parse minimum model size from config (e.g. "3B" -> 3.0)
            from ..utils.helpers import calculate_model_size

            model_name = getattr(self.model_config, "base_model_name", "")
            model_size = calculate_model_size(model_name)

            min_size_str = ta_cfg.min_model_size.upper().replace("B", "")
            try:
                min_size = float(min_size_str)
            except (ValueError, TypeError):
                min_size = 3.0

            # Only anneal if the model is *smaller* than the threshold
            if model_size is not None and model_size < min_size:
                # Determine total rounds from config if available
                # (max_evolution_rounds is not stored here; infer from calling
                #  code by using the max of evolution_round seen so far)
                max_rounds = max(
                    getattr(self, "_max_evolution_rounds", 3),
                    evolution_round + 1,
                )
                self._max_evolution_rounds = max_rounds

                progress = evolution_round / max(max_rounds - 1, 1)
                annealed_temp = ta_cfg.start_temp + progress * (ta_cfg.end_temp - ta_cfg.start_temp)
                annealed_temp = max(ta_cfg.end_temp, min(ta_cfg.start_temp, annealed_temp))

                if self.logger:
                    self.logger.info(
                        f"Temperature annealing: {annealed_temp:.3f} "
                        f"(model {model_name}, size {model_size}B < {min_size}B threshold)"
                    )

                # Apply to all agents
                for agent in self.agents:
                    agent.update_generation_config({"temperature": annealed_temp})

    def cleanup(self) -> None:
        """Clean up all agent resources."""
        for agent in self.agents:
            agent.cleanup()

    def __del__(self) -> None:
        """Cleanup when manager is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass

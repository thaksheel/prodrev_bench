"""
GPU integration test: load a real model and run multi-agent debates.

Uses Qwen/Qwen2.5-0.5B-Instruct (smallest model) on CUDA_VISIBLE_DEVICES=0.
Marked with @pytest.mark.gpu so it is skipped when no GPU is available.

Tests cover:
- Math debate produces result
- ARC-style debate (multiple choice)
- Consensus and sycophancy tracking
- Agent answer history
- Metrics population
- Consolidated reasoning
- Debate statistics
- Temperature annealing across evolution rounds
- Weight sharing verification (multiple agents, single model in memory)
- Debate with different task types (math, arc, general)
"""

import os

import pytest

# Pin to GPU 0 before any CUDA initialization (per spec: debate on GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dte.core.config import DebateConfig, ModelConfig, TemperatureAnnealingConfig
from dte.debate.agent import _model_registry
from dte.debate.manager import DebateManager

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.gpu
class TestDebateIntegration:
    """End-to-end debate integration tests on real hardware."""

    @pytest.fixture(scope="class")
    def debate_manager(self):
        """Create a DebateManager with 3 agents sharing the smallest model.

        Scoped to the class so the model is loaded only once for all tests.
        """
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        debate_cfg = DebateConfig(
            num_agents=3,
            max_rounds=3,
        )
        manager = DebateManager(debate_cfg, model_cfg, logger=None)
        yield manager
        manager.cleanup()

    def test_math_debate_produces_result(self, debate_manager):
        """A simple multiplication should produce a non-empty final answer."""
        result = debate_manager.conduct_debate("What is 15 * 24?", task_type="math")

        assert result.final_answer is not None
        assert result.final_answer != ""
        assert result.total_rounds >= 0
        assert len(result.all_responses) >= 1
        assert len(result.extracted_answers) >= 1

    def test_arc_style_debate(self, debate_manager):
        """An ARC-style multiple-choice question should produce a valid result."""
        arc_query = (
            "What is the main function of the roots of a plant?\n\n"
            "Choices:\n"
            "A. To absorb water and nutrients from the soil\n"
            "B. To produce seeds\n"
            "C. To perform photosynthesis\n"
            "D. To attract pollinators"
        )
        result = debate_manager.conduct_debate(arc_query, task_type="arc")

        assert result.final_answer is not None
        assert result.final_answer != ""
        assert result.task_type == "arc"
        assert len(result.all_responses) >= 1
        # Each round should have 3 agent responses
        for round_responses in result.all_responses:
            assert len(round_responses) == 3

    def test_general_task_type_debate(self, debate_manager):
        """A general reasoning question should produce a valid result."""
        general_query = (
            "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
        )
        result = debate_manager.conduct_debate(general_query, task_type="general")

        assert result.final_answer is not None
        assert result.final_answer != ""
        assert result.task_type == "general"
        assert len(result.all_responses) >= 1

    def test_consensus_tracking(self, debate_manager):
        """Verify consensus and sycophancy tracking structures are populated."""
        result = debate_manager.conduct_debate("What is 2 + 2?", task_type="math")

        assert isinstance(result.consensus_reached, bool)
        assert isinstance(result.consensus_progression, list)
        assert len(result.consensus_progression) >= 1
        assert isinstance(result.confidence_progression, list)
        assert len(result.confidence_progression) >= 1
        # Each round should have one confidence score per agent
        for round_confidences in result.confidence_progression:
            assert len(round_confidences) == 3

    def test_agent_answer_history(self, debate_manager):
        """Verify agent answer history is properly tracked."""
        result = debate_manager.conduct_debate("What is 3 * 7?", task_type="math")

        assert isinstance(result.agent_answer_history, dict)
        assert len(result.agent_answer_history) == 3
        for agent_id, history in result.agent_answer_history.items():
            assert len(history) >= 1  # At least initial round

    def test_metrics_populated(self, debate_manager):
        """Verify that debate metrics contain expected keys."""
        result = debate_manager.conduct_debate("What is 100 / 4?", task_type="math")

        assert "total_time" in result.metrics
        assert "sycophancy_rate" in result.metrics
        assert "answer_change_rate" in result.metrics
        assert "average_reasoning_length" in result.metrics
        assert result.metrics["total_time"] > 0

    def test_consolidated_reasoning(self, debate_manager):
        """Verify consolidated reasoning is generated."""
        result = debate_manager.conduct_debate("What is 10 + 5?", task_type="math")

        assert result.consolidated_reasoning is not None
        assert isinstance(result.consolidated_reasoning, str)
        assert len(result.consolidated_reasoning) > 0

    def test_debate_statistics(self, debate_manager):
        """After running debates, statistics should be available."""
        stats = debate_manager.get_debate_statistics()

        assert isinstance(stats, dict)
        if stats:  # Will have data from previous tests
            assert "total_debates" in stats
            assert stats["total_debates"] > 0


@pytest.mark.gpu
class TestTemperatureAnnealing:
    """Test temperature annealing across evolution rounds with a real model."""

    @pytest.fixture(scope="class")
    def annealing_manager(self):
        """Create a DebateManager with temperature annealing enabled."""
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(
            num_agents=2,
            max_rounds=1,
            temperature_annealing=TemperatureAnnealingConfig(
                enabled=True,
                start_temp=0.7,
                end_temp=0.3,
                min_model_size="3B",  # 0.5B < 3B, so annealing will apply
            ),
        )
        manager = DebateManager(debate_cfg, model_cfg, logger=None)
        yield manager
        manager.cleanup()

    def test_temperature_decreases_across_rounds(self, annealing_manager):
        """Temperature should decrease as evolution round increases."""
        # Record starting temperature
        initial_temps = [agent.generation_config.get("temperature", 0.7) for agent in annealing_manager.agents]

        # Advance to evolution round 1
        annealing_manager.update_evolution_round(1)
        mid_temps = [agent.generation_config.get("temperature", 0.7) for agent in annealing_manager.agents]

        # Advance to evolution round 2
        annealing_manager.update_evolution_round(2)
        late_temps = [agent.generation_config.get("temperature", 0.7) for agent in annealing_manager.agents]

        # Temperature should be monotonically non-increasing
        for i in range(len(annealing_manager.agents)):
            assert mid_temps[i] <= initial_temps[i] + 1e-6, (
                f"Agent {i}: mid temp {mid_temps[i]} > initial {initial_temps[i]}"
            )
            assert late_temps[i] <= mid_temps[i] + 1e-6, f"Agent {i}: late temp {late_temps[i]} > mid {mid_temps[i]}"

    def test_debate_works_after_annealing(self, annealing_manager):
        """Debate should still produce valid results after temperature change."""
        annealing_manager.update_evolution_round(2)
        result = annealing_manager.conduct_debate("What is 5 + 5?", task_type="math")

        assert result.final_answer is not None
        assert result.final_answer != ""
        assert len(result.all_responses) >= 1


@pytest.mark.gpu
class TestWeightSharing:
    """Verify that multiple agents share a single model in GPU memory."""

    def test_agents_share_model_object(self):
        """All agents in a debate should reference the same model object."""
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=3, max_rounds=1)
        manager = DebateManager(debate_cfg, model_cfg, logger=None)

        try:
            agents = manager.agents
            assert len(agents) == 3

            # All agents should share the exact same model object (same id())
            model_ids = [id(agent.model) for agent in agents]
            assert model_ids[0] == model_ids[1] == model_ids[2], "Agents do not share the same model object in memory"

            # All agents should share the same tokenizer object
            tokenizer_ids = [id(agent.tokenizer) for agent in agents]
            assert tokenizer_ids[0] == tokenizer_ids[1] == tokenizer_ids[2], (
                "Agents do not share the same tokenizer object in memory"
            )

            # Run a debate to verify they still work with shared weights
            result = manager.conduct_debate("What is 1 + 1?", task_type="math")
            assert result.final_answer is not None

        finally:
            manager.cleanup()

    def test_model_registry_refcount(self):
        """Model registry should correctly track reference counts."""
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=2, max_rounds=1)
        manager = DebateManager(debate_cfg, model_cfg, logger=None)

        try:
            # Find the registry entry for our model
            found_key = None
            for key, entry in _model_registry._cache.items():
                if MODEL_NAME in key[0]:
                    found_key = key
                    break

            assert found_key is not None, "Model not found in registry"
            # 2 agents should have refcount >= 2
            assert _model_registry._cache[found_key]["refcount"] >= 2

        finally:
            manager.cleanup()


@pytest.mark.gpu
class TestDebateTaskTypes:
    """Test debates with multiple task types on real hardware."""

    @pytest.fixture(scope="class")
    def manager(self):
        """Create a lightweight DebateManager."""
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=2, max_rounds=1)
        manager = DebateManager(debate_cfg, model_cfg, logger=None)
        yield manager
        manager.cleanup()

    def test_math_task_type(self, manager):
        """Math task type should work end-to-end."""
        result = manager.conduct_debate("What is 12 + 8?", task_type="math")
        assert result.task_type == "math"
        assert result.final_answer is not None

    def test_arc_task_type(self, manager):
        """ARC task type should work end-to-end."""
        query = (
            "Which of the following is the best conductor of electricity?\n\n"
            "Choices:\nA. Wood\nB. Rubber\nC. Copper\nD. Glass"
        )
        result = manager.conduct_debate(query, task_type="arc")
        assert result.task_type == "arc"
        assert result.final_answer is not None

    def test_reasoning_task_type(self, manager):
        """Reasoning task type should work end-to-end."""
        result = manager.conduct_debate(
            "Is it possible for a triangle to have two right angles? Explain.",
            task_type="reasoning",
        )
        assert result.task_type == "reasoning"
        assert result.final_answer is not None

    def test_multiple_debates_sequential(self, manager):
        """Running multiple debates sequentially should not leak state."""
        result1 = manager.conduct_debate("What is 1+1?", task_type="math")
        result2 = manager.conduct_debate("What is 2+2?", task_type="math")

        # Results should be independent
        assert result1.query == "What is 1+1?"
        assert result2.query == "What is 2+2?"
        # Both should have final answers
        assert result1.final_answer is not None
        assert result2.final_answer is not None

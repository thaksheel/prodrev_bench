"""Tests for RCR debate prompt generation and response parsing."""

import pytest

from dte.debate.prompts import DebatePromptManager, DebateResponse


class TestInitialPrompts:
    """Tests for initial prompt generation (round 0, no RCR)."""

    def test_math_prompt(self, prompt_manager):
        prompt = prompt_manager.create_initial_prompt("What is 5 + 3?", "math")
        assert "5 + 3" in prompt
        assert "\\boxed" in prompt

    def test_general_prompt(self, prompt_manager):
        prompt = prompt_manager.create_initial_prompt("Explain gravity", "general")
        assert "Explain gravity" in prompt
        assert "step by step" in prompt.lower()

    def test_arc_prompt_with_dict(self, prompt_manager):
        question_data = {
            "question": "What is the color of sky?",
            "choices": {
                "text": ["Red", "Blue", "Green", "Yellow"],
                "label": ["A", "B", "C", "D"],
            },
        }
        prompt = prompt_manager.create_initial_prompt(question_data, "arc")
        assert "color of sky" in prompt
        assert "A." in prompt
        assert "Blue" in prompt

    def test_arc_prompt_with_string(self, prompt_manager):
        prompt = prompt_manager.create_initial_prompt("What is photosynthesis?", "arc")
        assert "photosynthesis" in prompt

    def test_unknown_task_type_raises(self, prompt_manager):
        with pytest.raises(ValueError, match="Unknown task_type"):
            prompt_manager.create_initial_prompt("test", "unknown_type")


class TestRCRDebatePrompts:
    """Tests for RCR (Reflect-Critique-Refine) debate prompts."""

    def test_math_rcr_prompt_has_three_phases(self, prompt_manager):
        answers_so_far = {
            "1": "I computed 5+3=8 so the answer is \\boxed{8}",
            "2": "Adding gives \\boxed{7}",
        }
        prompt = prompt_manager.create_debate_prompt(
            query="What is 5+3?",
            agent_id="1",
            round_num=1,
            answers_so_far=answers_so_far,
            task_type="math",
        )
        # All three RCR phases must be present
        assert "PHASE 1: REFLECT" in prompt
        assert "PHASE 2: CRITIQUE" in prompt
        assert "PHASE 3: REFINE" in prompt

    def test_debate_prompt_contains_agent_info(self, prompt_manager):
        answers_so_far = {
            "1": "My answer is 8",
            "2": "My answer is 7",
        }
        prompt = prompt_manager.create_debate_prompt(
            query="5+3?",
            agent_id="1",
            round_num=1,
            answers_so_far=answers_so_far,
            task_type="math",
        )
        assert "Agent 1" in prompt
        assert "round 1" in prompt.lower()

    def test_debate_prompt_shows_own_previous(self, prompt_manager):
        answers_so_far = {
            "1": "My previous reasoning here",
            "2": "Peer reasoning here",
        }
        prompt = prompt_manager.create_debate_prompt(
            query="test",
            agent_id="1",
            round_num=1,
            answers_so_far=answers_so_far,
            task_type="math",
        )
        assert "My previous reasoning here" in prompt
        assert "Your previous" in prompt

    def test_debate_prompt_shows_peer_solutions(self, prompt_manager):
        answers_so_far = {
            "1": "My answer",
            "2": "Peer 2 answer",
            "3": "Peer 3 answer",
        }
        prompt = prompt_manager.create_debate_prompt(
            query="test",
            agent_id="1",
            round_num=1,
            answers_so_far=answers_so_far,
            task_type="math",
        )
        # Agent 2 and 3 should appear in critique section
        assert "Agent 2" in prompt
        assert "Agent 3" in prompt

    def test_critique_pairs_default_is_2(self):
        pm = DebatePromptManager()
        assert pm.critique_pairs == 2

    def test_critique_pairs_custom(self):
        pm = DebatePromptManager(critique_pairs=1)
        assert pm.critique_pairs == 1

    def test_critique_pairs_invalid_raises(self):
        with pytest.raises(ValueError, match="critique_pairs must be at least 1"):
            DebatePromptManager(critique_pairs=0)

    def test_general_rcr_prompt(self, prompt_manager):
        prompt = prompt_manager.create_debate_prompt(
            query="Explain gravity",
            agent_id="1",
            round_num=2,
            answers_so_far={"2": "Gravity is a force"},
            task_type="general",
        )
        assert "Agent 1" in prompt
        assert "PHASE 1: REFLECT" in prompt
        assert "PHASE 2: CRITIQUE" in prompt
        assert "PHASE 3: REFINE" in prompt

    def test_arc_rcr_prompt(self, prompt_manager):
        prompt = prompt_manager.create_debate_prompt(
            query="What is the color of the sky?",
            agent_id="1",
            round_num=1,
            answers_so_far={"2": "The sky is blue. Answer: B"},
            task_type="arc",
        )
        assert "PHASE 1: REFLECT" in prompt
        assert "PHASE 2: CRITIQUE" in prompt
        assert "PHASE 3: REFINE" in prompt
        assert "Answer: [letter]" in prompt


class TestResponseParsing:
    """Tests for parsing agent responses."""

    def test_parse_math_response(self, prompt_manager):
        text = "Let me solve this. 5 + 3 = 8. So the answer is \\boxed{8}."
        response = prompt_manager.parse_response(text, agent_id="1", round_num=0, task_type="math")
        assert isinstance(response, DebateResponse)
        assert response.extracted_answer == "8"
        assert response.agent_id == "1"
        assert response.round_number == 0

    def test_parse_arc_response(self, prompt_manager):
        text = "The sky is blue, so the Answer: B"
        response = prompt_manager.parse_response(text, agent_id="2", round_num=1, task_type="arc")
        assert response.extracted_answer == "B"

    def test_parse_unable_to_extract(self, prompt_manager):
        text = "I have no idea what the answer could be."
        response = prompt_manager.parse_response(text, task_type="math")
        assert response.extracted_answer == "Unable to Extract"


class TestResponseValidation:
    """Tests for response format validation."""

    def test_valid_math_response(self, prompt_manager):
        response = DebateResponse(
            answer="8",
            reasoning="First add 5+3=8, the answer is \\boxed{8} and I am confident.",
            extracted_answer="8",
        )
        errors = prompt_manager.validate_response_format(response, "math")
        assert len(errors) == 0

    def test_missing_boxed_format(self, prompt_manager):
        response = DebateResponse(
            answer="8",
            reasoning="The answer is 8, that is for sure, let me explain why.",
            extracted_answer="8",
        )
        errors = prompt_manager.validate_response_format(response, "math")
        assert any("boxed" in e.lower() for e in errors)

    def test_short_reasoning(self, prompt_manager):
        response = DebateResponse(
            answer="8",
            reasoning="8",
            extracted_answer="8",
        )
        errors = prompt_manager.validate_response_format(response, "math")
        assert any("short" in e.lower() or "missing" in e.lower() for e in errors)

    def test_unable_to_extract_error(self, prompt_manager):
        response = DebateResponse(
            answer="Unable to Extract",
            reasoning="Some reasoning here that is long enough to pass",
            extracted_answer="Unable to Extract",
        )
        errors = prompt_manager.validate_response_format(response, "math")
        assert any("extract" in e.lower() for e in errors)

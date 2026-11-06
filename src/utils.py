from typing import List, Dict, Literal, Optional
from dataclasses import dataclass, field
from numpy.typing import NDArray
from enum import Enum
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import warnings
import numpy as np


class OpenModelSelection(Enum):
    tllama = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    qwen38_27b = "Qwen/Qwen3.8-27B"
    llama31_8b = "meta-llama/Llama-3.1-8B-Instruct"
    qwen25_7b = "Qwen/Qwen2.5-7B-Instruct"
    qwen3_8b = "Qwen/Qwen3-8B"


class CloseModelSelection(Enum):
    gpt6_luna = "gpt-6-luna"
    gpt6_sol = "gpt-6-sol"
    gpt6_astra = "gpt-6-astra"
    gpt56_sol = "gpt-5.6-sol"
    gpt56_luna = "gpt-5.6-luna"
    gpt56_terra = "gpt-5.6-terra"


@dataclass
class LLMOut:
    response: str
    input_token: int
    output_token: int
    model_name: str
    provider: str


@dataclass
class PromptMethod:
    """One single-call prompting method from prompts.py.

    Parameters
    ----------
    strategy : {"zero-shot", "few-shots"}
        Prompting strategy passed to ``PromptStrategy``.
    few_shots_type : {"random", "kate"}, default "random"
        How few-shot examples are chosen. Ignored for zero-shot.
    chain_of_thought : bool, default False
        Add the step-by-step instruction to the prompt.
    num_examples : int, default 10
        Number of few-shot examples.
    output_format : str, default "JSON_REASONING"
        Member name of ``RatingReasoning`` that sets the answer format.
    """

    strategy: Literal["zero-shot", "few-shots"]
    few_shots_type: Literal["random", "kate"] = "random"
    chain_of_thought: bool = False
    num_examples: int = 10
    output_format: str = "JSON_REASONING"


@dataclass
class MadConfig:
    """Settings for the multi-agent debate method.

    Parameters
    ----------
    model_name : str
        Model used by every debate agent.
    provider : str, default "openai"
        Backend passed to ``Debate``.
    api_key : str, optional
        API key passed to ``Debate``.
    device : str, default "cuda"
        Device for local models.
    max_new_token : int, default 1000
        Generation budget per agent turn.
    num_agents : int, default 5
        Agents in the debate.
    num_rounds : int, default 3
        Debate rounds.
    references_path : str, default "./exports/references.json"
        JSON file with the debate references.
    """

    model_name: str
    provider: str = "openai"
    api_key: Optional[str] = None
    device: str = "cuda"
    max_new_token: int = 1000
    num_agents: int = 5
    num_rounds: int = 3
    references_path: str = "./exports/references.json"


@dataclass
class MegaConfig:
    """Settings for the mega agent method.

    Parameters
    ----------
    params : modules.mega_agent_re.Params
        Model and team settings for the mega agent.
    task_instruction : str
        Task description given to the team leader.
    prompt_template : str
        Per-review prompt; must contain ``{review}``.
    plan_path : str, default "./exports/mega_plan.txt"
        Saved team plan; created on the first run and reused afterwards.
    log_path : str, default "./exports/logs.out"
        Log file for the mega runner.
    run_dir : str, default "./exports/mega_runs"
        Folder for per-run files.
    checkpoint_path : str, optional
        Checkpoint CSV for ``run_many``.
    max_workers : int, default 4
        Reviews processed in parallel.
    """

    params: Any
    task_instruction: str = (
        "predict the overall star rating (1 to 5) a customer gave a product, using "
        "only the text of their online review. Do not use any label or information "
        "outside the review text as evidence. The final answer is an integer rating "
        "with a brief evidence-based rationale."
    )
    prompt_template: str = (
        "Score this customer online product review from 1-5: <review> {review} </review>"
    )
    plan_path: str = "./exports/mega_plan.txt"
    log_path: str = "./exports/logs.out"
    run_dir: str = "./exports/mega_runs"
    checkpoint_path: Optional[str] = None
    max_workers: int = 4


def default_prompt_methods() -> Dict[str, PromptMethod]:
    return {
        "zero_shot": PromptMethod("zero-shot"),
        "zero_shot_cot": PromptMethod("zero-shot", chain_of_thought=True),
        "few_shot_random": PromptMethod("few-shots", "random"),
        "few_shot_random_cot": PromptMethod(
            "few-shots", "random", chain_of_thought=True
        ),
        "few_shot_kate": PromptMethod("few-shots", "kate"),
    }


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Accuracy, balanced accuracy, macro F1, MAE and RMSE."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    err = (y_true - y_pred).astype(float)
    with warnings.catch_warnings():  # sklearn warns when a subset lacks a class
        warnings.simplefilter("ignore")
        bal_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": bal_acc,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


# ---------------------------------------------------------------------------
# Prompts for ReviewRatingJudge
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a careful annotator of product reviews. You judge how well a 1-5 "
    "star rating matches the opinion expressed in a review's text. Respond with "
    "a single JSON object and nothing else."
)

# C5 [kim2024prometheus; liu2023geval]
DEFAULT_RUBRIC = """\
1 star  - Strongly negative. The product failed, broke or was unusable, or the reviewer is angry or regrets buying it; would not recommend.
2 stars - Mostly negative. Clear disappointment or notable problems; any positives are minor.
3 stars - Neutral or mixed. "It's okay", average, does the job but nothing special; pros and cons roughly balanced.
4 stars - Mostly positive. Satisfied and would likely recommend, with minor reservations.
5 stars - Strongly positive. Enthusiastic, exceeded expectations, no meaningful complaints.

Guidance:
- Judge the reviewer's opinion of the PRODUCT as expressed in the text. Complaints only about shipping, packaging or the seller weigh less than comments on the product itself.
- Account for sarcasm, negation, hedging and comparisons (e.g. "no better than the cheap version").
- Short or vague reviews carry weak evidence: say so and lower your confidence rather than guessing an extreme.
- Punctuation and capitalisation can signal intensity but do not override the literal content."""

# C5: placeholders; replace with reviews from your data that annotators agreed on.
DEFAULT_ANCHORS: Dict[int, List[str]] = {
    1: [
        "Broke after two days and the company ignored my emails. Total waste of money."
    ],
    2: [
        "Smaller than pictured and the zipper sticks. I'll use it but wouldn't buy again."
    ],
    3: ["It does what it says. Nothing special, but fine for the price."],
    4: ["Really comfortable and looks great. Runs a little small, so size up."],
    5: ["Absolutely love this! Best blender I've owned, worth every penny."],
}

# C4 [zheng2023judging]
BLIND_TEMPLATE = """\
## Rating rubric
{rubric}

## Reference examples
{anchors}

## Review
\"\"\"{review}\"\"\"

## Task
1. "evidence": list the phrases that signal the reviewer's opinion and their intensity, noting any sarcasm, negation, hedging or comparisons.
2. "supported_ratings": every rating (1-5) the text plausibly supports.
3. "best_rating": the single most likely rating.
4. "confidence": "low", "medium" or "high".

Return only this JSON:
{{"evidence": "...", "supported_ratings": [..], "best_rating": <int>, "confidence": "..."}}"""

# C1, C3, C6 [zheng2023judging; wang2024fair; wei2022chain]
PAIRWISE_TEMPLATE = """\
## Rating rubric
{rubric}

## Reference examples
{anchors}

## Review
\"\"\"{review}\"\"\"
{blind_section}
## Candidate ratings
Two annotators proposed a rating for this review. Their identities are hidden and the order is arbitrary; do not let the order influence you.
Rating A: {rating_a}
Rating B: {rating_b}

## Task
1. "evidence": the phrases in the review that matter for this decision.
2. "best_rating": the single rating (1-5) the text best supports.
3. "reasoning": compare each candidate against the evidence and the rubric.
4. "verdict":
   - "A" if Rating A is better supported by the text,
   - "B" if Rating B is better supported,
   - "tie" if both are equally well supported (for example, equally distant from the best reading),
   - "neither" if the text clearly supports a different rating and both candidates are unacceptable.
5. "confidence": "low", "medium" or "high".

Return only this JSON:
{{"evidence": "...", "best_rating": <int>, "reasoning": "...", "verdict": "A|B|tie|neither", "confidence": "..."}}"""

BLIND_SECTION = """
## Your earlier independent analysis (made before you saw any candidate rating)
{analysis}
"""


@dataclass
class JudgeConfig:
    """Settings for :class:`ReviewRatingJudge`.

    Parameters
    ----------
    tie_policy : {"signal", "drop"}, default "signal"
        How ties are labelled in the adjudicated metrics. ``"signal"`` keeps
        tied rows, flagged with ``is_tie=True``, and gives them the author's
        rating. ``"drop"`` gives them no label, so they are excluded.
    aggregation : {"strict", "pooled"}, default "strict"
        How the two position orders are combined (C2). ``"strict"``: both
        orders must agree, otherwise the row is a tie. ``"pooled"``: majority
        vote over all votes from both orders.
    blind_first : bool, default True
        Ask the judge for its own rating before showing the candidates (C4).
    n_samples : int, default 1
        Judgments per position order (C7). Values above 1 give a majority
        vote; they only differ if the LLMProvider was created with
        ``temperature > 0``.
    rubric : str, default DEFAULT_RUBRIC
        Definition of each star level shown to the judge (C5).
    anchors : dict of int to list of str, default DEFAULT_ANCHORS
        Reference reviews for each star level (C5).
    labels : tuple of int, default (1, 2, 3, 4, 5)
        The rating scale.
    review_col : str, default "review"
        Column holding the review text.
    true_col : str, default "true"
        Column holding the author's rating.
    pred_col : str, default "pred"
        Column holding the predicted rating.
    id_col : str, default "sample_id"
        Column holding the row id.
    max_retries : int, default 2
        Extra attempts when a response cannot be parsed as JSON.
    max_workers : int, default 4
        Reviews judged in parallel. Use 1 for a local Hugging Face model.
    """

    tie_policy: Literal["signal", "drop"] = "signal"
    aggregation: Literal["strict", "pooled"] = "strict"
    blind_first: bool = True
    n_samples: int = 1
    rubric: str = DEFAULT_RUBRIC
    anchors: Dict[int, List[str]] = field(default_factory=lambda: dict(DEFAULT_ANCHORS))
    labels: tuple = (1, 2, 3, 4, 5)
    review_col: str = "review"
    true_col: str = "true"
    pred_col: str = "pred"
    id_col: str = "sample_id"
    max_retries: int = 2
    max_workers: int = 4


@dataclass
class RunnerConfig:
    """Settings for :class:`EvaluationRunner`.

    Parameters
    ----------
    data_path : str
        CSV of product reviews.
    review_col : str, default "review"
        Column holding the review text.
    rating_col : str, default "rating"
        Column holding the 1-5 star rating.
    sample_size : int, default 5000
        Size of the balanced evaluation slice; each rating gets
        ``sample_size // 5`` rows.
    eval_limit : int, optional
        Evaluate only a balanced subset of the slice of this size
        (``eval_limit // 5`` rows per rating). None evaluates the whole slice.
    fewshot_pool_size : int, optional, default 2000
        Rows outside the slice used as the few-shot example pool.
        None uses every row outside the slice.
    seed : int, default 42
        Random seed for the slice and the few-shot pool.
    prompt_methods : dict of str to PromptMethod
        Prompting methods to run, keyed by the name used in the results.
    mad : MadConfig, optional
        Multi-agent debate settings. None skips the method.
    mega : MegaConfig, optional
        Mega agent settings. None skips the method.
    judge : JudgeConfig
        Settings for the LLM judge applied to every method.
    max_workers : int, default 4
        Reviews processed in parallel for prompting methods. Use 1 for a
        local Hugging Face model.
    output_path : str, default "./exports/method_comparison.xlsx"
        Workbook with the comparison table and per-method details.
    slice_path : str, optional
        Also save the evaluation slice as CSV here.
    """

    data_path: str
    review_col: str = "review"
    rating_col: str = "rating"
    sample_size: int = 5000
    eval_limit: Optional[int] = None
    fewshot_pool_size: Optional[int] = 2000
    seed: int = 42
    prompt_methods: Dict[str, PromptMethod] = field(
        default_factory=default_prompt_methods
    )
    mad: Optional[MadConfig] = None
    mega: Optional[MegaConfig] = None
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    max_workers: int = 4
    output_path: str = "./exports/method_comparison.xlsx"
    slice_path: Optional[str] = None

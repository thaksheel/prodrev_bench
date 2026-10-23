"""
LLM-as-a-judge adjudication of star-rating labels for product reviews.

For each review where the model prediction and the human (author) rating
disagree, a single LLM judge decides which of the two ratings is better
supported by the review *text*. The class then recomputes classification
metrics against the adjudicated labels and flags judge outputs for manual
sanity checking.

Framing note for the paper: the author's star rating is what they actually
clicked, so the judge does not establish that it is "wrong". It establishes
which rating is more consistent with the text the author wrote.

Design components and their sources (BibTeX keys refer to references.bib;
call ``ReviewRatingJudge.design_citations()`` for this list as a table):

    C1  Pairwise comparison of two candidates, with an explicit tie option
        [zheng2023judging]
    C2  Position swapping: every pair is judged in both orders and an
        inconsistent verdict is treated as a tie ("strict"), or votes are
        pooled across both orders ("pooled")
        [zheng2023judging; wang2024fair (Balanced Position Calibration)]
    C3  Evidence-before-verdict chain-of-thought output format
        [wei2022chain; wang2024fair (Multiple Evidence Calibration);
        liu2023geval]
    C4  Blind first pass: the judge rates the review before it sees any
        candidate rating (reference-guided judging, with the judge producing
        its own reference)
        [zheng2023judging]
    C5  Score rubric plus anchored reference examples per rating level
        [kim2024prometheus; kim2024prometheus2; liu2023geval]
    C6  Anonymised candidates ("Rating A/B"), predictor reasoning hidden, to
        limit position and verbosity/persuasion bias
        [zheng2023judging; wang2024fair]
    C7  Self-consistency: several sampled judgments, majority vote
        [wang2023selfconsistency]
    C8  Judge from a different model family than the predictor
        (self-preference bias) [panickssery2024llm]
    C9  Model-label disagreement as the signal for candidate label errors
        [northcutt2021confident; chong2022detecting]
    C10 Metrics for 5-level ordinal labels: accuracy, balanced accuracy,
        macro/weighted F1, Cohen's kappa, linear/quadratic weighted kappa,
        MAE, macro-averaged MAE, off-by-one accuracy, Spearman's rho
        [brodersen2010balanced; cohen1960coefficient; cohen1968weighted;
        baccianella2009evaluation; pedregosa2011scikit]
"""

import json
import logging
import random
import re
import threading
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

LLMFn = Callable[[str, str, float], str]
LABELS: tuple = (1, 2, 3, 4, 5)


def openai_llm(
    model: str, client=None, pass_temperature: bool = True, **create_kwargs
) -> LLMFn:
    """Wrap the OpenAI Chat Completions API as an LLMFn.

    OpenAI reasoning models reject a non-default temperature: pass
    pass_temperature=False for them. Extra keyword arguments (for example
    max_completion_tokens) go straight to chat.completions.create.
    """
    from openai import OpenAI

    client = client or OpenAI()

    def _call(system: str, user: str, temperature: float) -> str:
        extra = {"temperature": temperature} if pass_temperature else {}
        resp = client.chat.completions.create(
            model=model,
            **extra,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **create_kwargs,
        )
        return resp.choices[0].message.content or ""

    return _call


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a careful annotator of product reviews. You judge how well a 1-5 "
    "star rating matches the opinion expressed in a review's text. Respond with "
    "a single JSON object and nothing else."
)

# C5: score rubric [kim2024prometheus; liu2023geval]
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

# C5: anchored reference examples. These are placeholders written for this
# template. Replace them with reviews from your own data on which several
# human annotators agreed, and report how they were chosen.
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

# C4: blind pass, before any candidate rating is shown [zheng2023judging]
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

# C1, C3, C6: anonymised pairwise judgment, evidence before verdict
# [zheng2023judging; wang2024fair; wei2022chain]
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


_MOJIBAKE = re.compile(
    "[\u00c2\u00c3\u00e2][\u0080-\u00bf\u20ac\u2018-\u201e\u2122\u0152\u0153\u0161\u0178]"
)


def repair_mojibake(text):
    """Fix UTF-8 text that was decoded as Windows-1252 (e.g. "itâ€™s" -> "it’s").

    Uses ftfy when it is installed, otherwise a cp1252 round trip. Text without
    the telltale character pairs is returned unchanged.
    """
    if not isinstance(text, str) or not _MOJIBAKE.search(text):
        return text
    try:
        import ftfy

        return ftfy.fix_text(text)
    except ImportError:
        pass
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _format_anchors(anchors: Dict[int, Sequence[str]]) -> str:
    lines = []
    for star in sorted(anchors):
        for text in anchors[star]:
            lines.append(f'- {star} star{"s" if star != 1 else ""}: "{text}"')
    return "\n".join(lines) if lines else "(none)"


def _parse_json(text: Optional[str]) -> Optional[dict]:
    """Extract the first {...} block from an LLM response."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _to_int_rating(
    value, labels: Sequence[int], round_fractional: bool = True
) -> Optional[int]:
    """Parse a rating to an int in ``labels``; None if missing or out of range.

    With round_fractional=False, non-integer values such as 4.5 return None.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    if not round_fractional and not f.is_integer():
        return None
    v = int(np.floor(f + 0.5))  # half-up, so 4.5 -> 5 (Python's round() would give 4)
    return v if v in labels else None


def _llm_rating(value, labels: Sequence[int]) -> Optional[int]:
    """Parse a rating produced by the judge, accepting forms such as 4, "4" or "4 stars"."""
    r = _to_int_rating(value, labels)
    if r is None and isinstance(value, str):
        m = re.search(r"\d+", value)
        r = _to_int_rating(m.group(), labels) if m else None
    return r


def _parse_verdict(value) -> Optional[str]:
    """Normalise a verdict ("A", "Rating A", "a)", "Tie", "neither") to A / B / TIE / NEITHER."""
    v = re.sub(r"[^A-Z ]", " ", str(value).upper()).split()
    v = [w for w in v if w not in ("RATING", "CANDIDATE", "OPTION")]
    if not v:
        return None
    return v[0] if v[0] in ("A", "B", "TIE", "NEITHER") else None


def _mode(values: Sequence, default=None):
    values = [v for v in values if v is not None]
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Metrics (C10)
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, labels: Sequence[int] = LABELS) -> dict:
    """Nominal and ordinal metrics for 5-level star ratings.

    Nominal: accuracy, balanced accuracy [brodersen2010balanced], macro and
    weighted F1, Cohen's kappa [cohen1960coefficient].
    Ordinal: linear and quadratic weighted kappa [cohen1968weighted], MAE,
    RMSE, macro-averaged MAE [baccianella2009evaluation], off-by-one accuracy,
    Spearman's rho.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(labels)
    n = len(y_true)
    if n == 0:
        return {"n": 0}

    err = np.abs(y_true - y_pred)
    # Macro-MAE: mean of per-true-class MAEs, robust to the usual skew towards 5 stars.
    per_class_mae = [err[y_true == c].mean() for c in labels if np.any(y_true == c)]

    def _spearman(a, b):
        ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # sklearn warns on classes absent from a small subset
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        out = {
            "n": n,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
            "weighted_f1": f1_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            ),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
            "linear_weighted_kappa": cohen_kappa_score(
                y_true, y_pred, labels=labels, weights="linear"
            ),
            "quadratic_weighted_kappa": cohen_kappa_score(
                y_true, y_pred, labels=labels, weights="quadratic"
            ),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(np.mean(err.astype(float) ** 2))),
            "macro_mae": float(np.mean(per_class_mae)),
            "off_by_one_accuracy": float(np.mean(err <= 1)),
            "spearman_rho": _spearman(y_true, y_pred),
        }
    out["per_class"] = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s},
        index=pd.Index(labels, name="stars"),
    )
    out["confusion_matrix"] = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=pd.Index(labels, name="label"),
        columns=pd.Index(labels, name="pred"),
    )
    return out


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------
class ReviewRatingJudge:
    """Pairwise LLM judge that adjudicates human vs. predicted star ratings.

    Parameters
    ----------
    llm : LLMFn
        Function (system_prompt, user_prompt, temperature) -> response text.
    judge_model_name, predictor_model_name : str
        Recorded in the config. If both share a model family, a
        self-preference warning is raised (C8, [panickssery2024llm]).
    tie_policy : "signal" | "drop"
        "signal": ties stay in the evaluation, flagged with ``is_tie=True``;
        their adjudicated label falls back to the human rating (conservative).
        "drop": ties get no adjudicated label and are excluded from the
        adjudicated metrics.
    n_samples : int
        Judgments per position order (C7). Values above 1 enable majority
        voting and need temperature > 0.
    temperature : float, optional
        Defaults to 0.0 when n_samples == 1, otherwise 0.7.
    blind_first : bool
        Run the blind rating pass before the pairwise pass (C4).
    aggregation : "strict" | "pooled"
        "strict": both position orders must agree, otherwise tie
        [zheng2023judging]. "pooled": majority over votes from both orders
        [wang2024fair].
    rubric, anchors :
        Rating rubric and reference examples per star level (C5).
    audit_rate : float
        Share of judged rows randomly flagged for manual audit, in addition
        to the rule-based flags.
    columns : dict
        Map from {"review", "true", "pred", "id"} to your column names.
    round_fractional_ratings : bool
        Ratings must be whole numbers in ``labels``. By default a value such
        as 4.5 is treated as invalid input (excluded, with a warning); set
        True to round it to the nearest star instead.
    preflight : bool
        Make one test call before evaluating and raise with the real cause
        if the backend is broken (wrong model name, missing key, ...).
    max_error_rate : float
        Raise after evaluation if more than this share of judged rows
        failed. Failed rows get judge_choice='error' and are always
        excluded from the adjudicated metrics; they are never counted as ties.
    """

    PROMPT_VERSION = "1.0"

    def __init__(
        self,
        llm: LLMFn,
        *,
        judge_model_name: str = "unspecified",
        predictor_model_name: Optional[str] = None,
        tie_policy: Literal["signal", "drop"] = "signal",
        n_samples: int = 1,
        temperature: Optional[float] = None,
        blind_first: bool = True,
        aggregation: Literal["strict", "pooled"] = "strict",
        rubric: str = DEFAULT_RUBRIC,
        anchors: Optional[Dict[int, Sequence[str]]] = None,
        labels: Sequence[int] = LABELS,
        max_retries: int = 2,
        max_workers: int = 4,
        audit_rate: float = 0.05,
        seed: int = 0,
        columns: Optional[Dict[str, str]] = None,
        round_fractional_ratings: bool = False,
        preflight: bool = True,
        max_error_rate: float = 0.2,
    ):
        if tie_policy not in ("signal", "drop"):
            raise ValueError("tie_policy must be 'signal' or 'drop'")
        if aggregation not in ("strict", "pooled"):
            raise ValueError("aggregation must be 'strict' or 'pooled'")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        self.llm = llm
        self.tie_policy = tie_policy
        self.n_samples = n_samples
        self.temperature = (
            temperature if temperature is not None else (0.0 if n_samples == 1 else 0.7)
        )
        self.blind_first = blind_first
        self.aggregation = aggregation
        self.rubric = rubric
        self.anchors = anchors if anchors is not None else DEFAULT_ANCHORS
        self.labels = tuple(labels)
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.audit_rate = audit_rate
        self.seed = seed
        self.round_fractional_ratings = round_fractional_ratings
        self.preflight = preflight
        self.max_error_rate = max_error_rate
        self.columns = {
            "review": "review",
            "true": "true",
            "pred": "pred",
            "id": "sample_id",
        }
        if columns:
            self.columns.update(columns)

        if n_samples > 1 and self.temperature == 0:
            warnings.warn(
                "n_samples > 1 with temperature 0 gives identical samples; raise the temperature."
            )

        # C8: self-preference bias [panickssery2024llm]
        if predictor_model_name and judge_model_name != "unspecified":
            fam = lambda s: re.split(r"[-_/ :.]", s.lower().strip())[0]
            if fam(predictor_model_name) == fam(judge_model_name):
                warnings.warn(
                    f"Judge ({judge_model_name}) and predictor ({predictor_model_name}) appear to share "
                    "a model family; LLM judges tend to favour their own family's outputs "
                    "(Panickssery et al., 2024). Consider a judge from another family."
                )

        self.config = {
            "prompt_version": self.PROMPT_VERSION,
            "judge_model_name": judge_model_name,
            "predictor_model_name": predictor_model_name,
            "tie_policy": tie_policy,
            "n_samples": n_samples,
            "temperature": self.temperature,
            "blind_first": blind_first,
            "aggregation": aggregation,
            "audit_rate": audit_rate,
            "seed": seed,
            "round_fractional_ratings": round_fractional_ratings,
        }
        self.raw_log: List[dict] = []
        self._log_lock = threading.Lock()
        self.results_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Citations
    # ------------------------------------------------------------------ #
    @staticmethod
    def design_citations() -> pd.DataFrame:
        """Each design component with the BibTeX keys it draws on."""
        rows = [
            ("C1", "Pairwise comparison with explicit tie option", "zheng2023judging"),
            (
                "C2",
                "Position swap; strict consistency or pooled votes",
                "zheng2023judging; wang2024fair",
            ),
            (
                "C3",
                "Evidence-before-verdict chain-of-thought",
                "wei2022chain; wang2024fair; liu2023geval",
            ),
            (
                "C4",
                "Blind independent rating before seeing candidates",
                "zheng2023judging",
            ),
            (
                "C5",
                "Score rubric with anchored reference examples",
                "kim2024prometheus; kim2024prometheus2; liu2023geval",
            ),
            (
                "C6",
                "Anonymised candidates, predictor reasoning hidden",
                "zheng2023judging; wang2024fair",
            ),
            ("C7", "Self-consistency majority voting", "wang2023selfconsistency"),
            ("C8", "Cross-family judge to limit self-preference", "panickssery2024llm"),
            (
                "C9",
                "Model-label disagreement flags candidate label errors",
                "northcutt2021confident; chong2022detecting",
            ),
            (
                "C10",
                "Nominal and ordinal metrics for 5-level labels",
                "brodersen2010balanced; cohen1960coefficient; cohen1968weighted; baccianella2009evaluation; pedregosa2011scikit",
            ),
        ]
        return pd.DataFrame(rows, columns=["component", "description", "bibtex_keys"])

    # ------------------------------------------------------------------ #
    # LLM plumbing
    # ------------------------------------------------------------------ #
    def _call_json(self, prompt: str, required: set, meta: dict):
        """Call the judge, parse JSON, retry on failure.

        Returns (parsed | None, n_calls, error_detail | None).
        """
        calls, detail = 0, None
        for attempt in range(self.max_retries + 1):
            calls += 1
            text, error = None, None
            try:
                text = self.llm(SYSTEM_PROMPT, prompt, self.temperature)
            except Exception as exc:  # bad model name, auth, rate limits, network ...
                error = f"{type(exc).__name__}: {exc}"
            parsed = _parse_json(text)
            ok = parsed is not None and required <= set(parsed)
            if not ok:
                if error:
                    detail = f"API call raised {error}"
                elif not text:
                    detail = "Empty response from the model"
                elif parsed is None:
                    detail = f"Response is not JSON: {text[:200]!r}"
                else:
                    detail = f"JSON missing keys {sorted(required - set(parsed))}: {text[:200]!r}"
            with self._log_lock:
                self.raw_log.append(
                    {
                        **meta,
                        "attempt": attempt,
                        "prompt": prompt,
                        "response": text,
                        "error": error,
                        "parsed_ok": ok,
                        "detail": detail,
                    }
                )
            if ok:
                return parsed, calls, None
        return None, calls, detail

    def _blind_prompt(self, review: str) -> str:
        return BLIND_TEMPLATE.format(
            rubric=self.rubric, anchors=_format_anchors(self.anchors), review=review
        )

    def _pairwise_prompt(
        self, review: str, a: int, b: int, blind: Optional[dict]
    ) -> str:
        section = ""
        if blind is not None:
            analysis = json.dumps(
                {
                    k: blind.get(k)
                    for k in ("evidence", "supported_ratings", "best_rating")
                },
                ensure_ascii=False,
            )
            section = BLIND_SECTION.format(analysis=analysis)
        return PAIRWISE_TEMPLATE.format(
            rubric=self.rubric,
            anchors=_format_anchors(self.anchors),
            review=review,
            blind_section=section,
            rating_a=a,
            rating_b=b,
        )

    # ------------------------------------------------------------------ #
    # Core adjudication of one (review, rating_x, rating_y) pair
    # ------------------------------------------------------------------ #
    def _adjudicate(self, review: str, x: int, y: int, sample_id) -> dict:
        """Judge candidate x against candidate y. Winner is 'x', 'y', 'tie', 'neither' or 'error'."""
        calls = 0
        blind_parses: List[dict] = []
        errors: List[str] = []

        # C4: blind pass
        if self.blind_first:
            for k in range(self.n_samples):
                parsed, n, err = self._call_json(
                    self._blind_prompt(review),
                    {"best_rating"},
                    {"sample_id": sample_id, "stage": "blind", "sample": k},
                )
                calls += n
                if parsed is not None:
                    blind_parses.append(parsed)
                else:
                    errors.append(f"blind: {err}")
        blind_ratings = [
            _llm_rating(p.get("best_rating"), self.labels) for p in blind_parses
        ]
        blind_mode = _mode(blind_ratings)
        blind_ref = next(
            (p for p, r in zip(blind_parses, blind_ratings) if r == blind_mode), None
        )

        # C2 + C7: both orders, n_samples each
        votes: Dict[str, List[str]] = {"xy": [], "yx": []}
        confidences, pair_ratings, first = [], [], None
        for order in ("xy", "yx"):
            a, b = (x, y) if order == "xy" else (y, x)
            to_slot = {"A": order[0], "B": order[1], "TIE": "tie", "NEITHER": "neither"}
            for k in range(self.n_samples):
                parsed, n, err = self._call_json(
                    self._pairwise_prompt(review, a, b, blind_ref),
                    {"verdict"},
                    {
                        "sample_id": sample_id,
                        "stage": f"pairwise_{order}",
                        "sample": k,
                        "rating_a": a,
                        "rating_b": b,
                    },
                )
                calls += n
                if parsed is None:
                    votes[order].append("error")
                    errors.append(f"pairwise_{order}: {err}")
                    continue
                verdict = _parse_verdict(parsed.get("verdict"))
                if verdict is None:
                    errors.append(
                        f"pairwise_{order}: unrecognised verdict {parsed.get('verdict')!r}"
                    )
                votes[order].append(to_slot.get(verdict, "error"))
                confidences.append(str(parsed.get("confidence", "")).lower() or None)
                pair_ratings.append(_llm_rating(parsed.get("best_rating"), self.labels))
                first = first or parsed

        winner, consistent, reason = self._aggregate(votes)
        return {
            "winner": winner,
            "position_consistent": consistent,
            "tie_reason": reason,
            "votes_xy": votes["xy"],
            "votes_yx": votes["yx"],
            "judge_rating": (
                blind_mode if blind_mode is not None else _mode(pair_ratings)
            ),
            "judge_supported_ratings": (
                blind_ref.get("supported_ratings") if blind_ref else None
            ),
            "judge_confidence": _mode(confidences),
            "judge_evidence": (blind_ref or first or {}).get("evidence"),
            "judge_reasoning": (first or {}).get("reasoning"),
            "n_judge_calls": calls,
            "judge_error_detail": errors[0] if errors else None,
        }

    def _aggregate(self, votes: Dict[str, List[str]]):
        """Combine votes into (winner, position_consistent, tie_reason)."""

        def majority(vs):
            vs = [v for v in vs if v != "error"]
            if not vs:
                return "error"
            top = Counter(vs).most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                return "tie"
            return top[0][0]

        m_xy, m_yx = majority(votes["xy"]), majority(votes["yx"])
        if m_xy == "error" and m_yx == "error":
            return "error", False, "judge_error"
        consistent = m_xy == m_yx

        if self.aggregation == "strict":  # [zheng2023judging]
            if "error" in (
                m_xy,
                m_yx,
            ):  # one order failed: position consistency can't be checked
                return "error", False, "judge_error"
            winner = m_xy if consistent else "tie"
            if not consistent:
                return "tie", False, "position_inconsistent"
        else:  # pooled [wang2024fair]
            winner = majority(votes["xy"] + votes["yx"])

        reason = {
            "tie": "judge_tie",
            "neither": "judge_neither",
            "error": "judge_error",
        }.get(winner)
        return winner, consistent, reason

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Judge every disagreement in ``df``; return one row per input row.

        Key output columns
        ------------------
        judge_choice      'pred' | 'true' | 'tie' | 'error' (judge failed) | 'agree' (no judging
                          needed) | 'invalid_input'
        selected_rating   rating chosen by the judge (NaN for ties)
        true_rating       human rating as a whole-number Int64 (NA if unusable)
        pred_rating       predicted rating as a whole-number Int64 (NA if unusable)
        is_tie            True for ties, rejections of both and position-inconsistent verdicts
        tie_reason        judge_tie | judge_neither | position_inconsistent (judge_error for errors)
        judge_error_detail why the judge failed on that row
        adjudicated_label label used for adjudicated metrics (see tie_policy)
        sanity_flag       True if the row should be inspected by hand
        sanity_reasons    why it was flagged
        """
        c = self.columns
        missing = [c[k] for k in ("review", "true", "pred") if c[k] not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        work = df.reset_index(drop=True).copy()
        work["true_rating"] = self._normalise_ratings(work[c["true"]], c["true"])
        work["pred_rating"] = self._normalise_ratings(work[c["pred"]], c["pred"])
        ids = (
            work[c["id"]]
            if c["id"] in work.columns
            else pd.Series([f"row{i}" for i in range(len(work))])
        )
        records = [
            {"sample_id": sid, "review": rev, "true": t, "pred": p}
            for sid, rev, t, p in zip(
                ids, work[c["review"]], work["true_rating"], work["pred_rating"]
            )
        ]

        if self.preflight:
            first = next(
                (
                    r["review"]
                    for r in records
                    if not pd.isna(r["true"])
                    and not pd.isna(r["pred"])
                    and r["true"] != r["pred"]
                    and isinstance(r["review"], str)
                    and r["review"].strip()
                ),
                None,
            )
            if first is not None:
                self.check_backend(first)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            judged = list(pool.map(self._evaluate_one, records))

        jdf = pd.DataFrame(judged)
        out = pd.concat([work, jdf.drop(columns=["sample_id"])], axis=1)
        out = self._apply_tie_policy(out)
        out = self._flag_for_sanity(out)
        self.results_ = out

        n_judged = int(out["judge_choice"].isin(["pred", "true", "tie", "error"]).sum())
        n_err = int((out["judge_choice"] == "error").sum())
        if n_err:
            top = (
                out.loc[out["judge_choice"] == "error", "judge_error_detail"]
                .value_counts()
                .head(3)
            )
            msg = (
                f"{n_err} of {n_judged} judged rows failed (judge_choice='error'); they are excluded from "
                f"the adjudicated metrics. Most common causes:\n"
                + "\n".join(f"  [{k}x] {d}" for d, k in top.items())
                + "\nSee judge.error_report() for details."
            )
            if n_judged and n_err / n_judged > self.max_error_rate:
                raise RuntimeError(
                    msg
                    + f"\nError rate is above max_error_rate={self.max_error_rate}. "
                    "Partial results are in judge.results_."
                )
            warnings.warn(msg)
        return out

    def check_backend(self, review: str = "It works fine, nothing special.") -> dict:
        """Make one real judge call and raise with the actual cause if it fails."""
        parsed, _, err = self._call_json(
            self._blind_prompt(review),
            {"best_rating"},
            {"sample_id": "preflight", "stage": "preflight", "sample": 0},
        )
        if parsed is None:
            raise RuntimeError(
                "Judge backend check failed before evaluation started.\n"
                f"Cause: {err}\n"
                "Common fixes: use a valid model name; set the API key environment variable; "
                "pass pass_temperature=False if the model rejects temperature; raise max_tokens "
                "if responses are cut off; reduce max_workers if you hit rate limits."
            )
        return parsed

    def error_report(self) -> pd.DataFrame:
        """Failed judge calls grouped by cause."""
        failed = [r for r in self.raw_log if not r.get("parsed_ok")]
        if not failed:
            return pd.DataFrame(columns=["detail", "n_calls", "example_sample_id"])
        f = pd.DataFrame(failed)
        return (
            f.groupby("detail", dropna=False)
            .agg(n_calls=("detail", "size"), example_sample_id=("sample_id", "first"))
            .sort_values("n_calls", ascending=False)
            .reset_index()
        )

    def _normalise_ratings(self, col: pd.Series, name: str) -> pd.Series:
        """Convert a rating column to nullable Int64, warning about values that can't be used."""
        parsed = [
            _to_int_rating(v, self.labels, self.round_fractional_ratings) for v in col
        ]
        bad = [v for v, q in zip(col, parsed) if q is None and not pd.isna(v)]
        rounded = (
            [
                v
                for v, q in zip(col, parsed)
                if q is not None and not pd.isna(v) and float(v) != q
            ]
            if self.round_fractional_ratings
            else []
        )
        if bad:
            warnings.warn(
                f"Column '{name}': {len(bad)} value(s) are not whole ratings in {list(self.labels)} and are "
                f"treated as invalid_input (examples: {bad[:5]}). Pass round_fractional_ratings=True "
                "to round them instead."
            )
        if rounded:
            warnings.warn(
                f"Column '{name}': rounded {len(rounded)} fractional value(s) (examples: {rounded[:5]})."
            )
        return pd.Series(pd.array(parsed, dtype="Int64"), index=col.index)

    def _evaluate_one(self, rec: dict) -> dict:
        t = None if pd.isna(rec["true"]) else int(rec["true"])
        p = None if pd.isna(rec["pred"]) else int(rec["pred"])
        base = {
            "sample_id": rec["sample_id"],
            "judge_choice": None,
            "tie_reason": None,
            "position_consistent": None,
            "votes_pred_first": None,
            "votes_true_first": None,
            "judge_rating": None,
            "judge_supported_ratings": None,
            "judge_confidence": None,
            "judge_evidence": None,
            "judge_reasoning": None,
            "n_judge_calls": 0,
            "judge_error_detail": None,
        }
        if (
            t is None
            or p is None
            or not isinstance(rec["review"], str)
            or not rec["review"].strip()
        ):
            return {**base, "judge_choice": "invalid_input"}
        if t == p:
            return {**base, "judge_choice": "agree"}

        # C9: only disagreements reach the judge. x = prediction, y = human label.
        res = self._adjudicate(rec["review"], x=p, y=t, sample_id=rec["sample_id"])
        choice = {"x": "pred", "y": "true", "error": "error"}.get(res["winner"], "tie")
        slot = {
            "x": "pred",
            "y": "true",
            "tie": "tie",
            "neither": "neither",
            "error": "error",
        }
        return {
            **base,
            "judge_choice": choice,
            "tie_reason": res["tie_reason"],
            "position_consistent": res["position_consistent"],
            "votes_pred_first": [slot[v] for v in res["votes_xy"]],
            "votes_true_first": [slot[v] for v in res["votes_yx"]],
            "judge_rating": res["judge_rating"],
            "judge_supported_ratings": res["judge_supported_ratings"],
            "judge_confidence": res["judge_confidence"],
            "judge_evidence": res["judge_evidence"],
            "judge_reasoning": res["judge_reasoning"],
            "n_judge_calls": res["n_judge_calls"],
            "judge_error_detail": res["judge_error_detail"],
        }

    def _apply_tie_policy(self, out: pd.DataFrame) -> pd.DataFrame:
        true = out["true_rating"].astype("Float64")
        pred = out["pred_rating"].astype("Float64")
        choice = out["judge_choice"]

        selected = pd.Series(np.nan, index=out.index)
        selected[choice == "pred"] = pred[choice == "pred"]
        selected[choice.isin(["true", "agree"])] = true[choice.isin(["true", "agree"])]
        out["selected_rating"] = selected.astype("Int64")
        out["is_tie"] = choice.eq("tie")
        out["is_judge_error"] = choice.eq("error")

        adjudicated = selected.copy()  # errors stay NA under both policies
        if self.tie_policy == "signal":
            adjudicated[out["is_tie"]] = true[out["is_tie"]]  # conservative fallback
        out["adjudicated_label"] = adjudicated.astype("Int64")
        out["excluded_from_adjudicated"] = out["adjudicated_label"].isna()
        out["label_changed"] = (
            (out["adjudicated_label"] != out["true_rating"]).fillna(False).astype(bool)
        )
        return out

    def _flag_for_sanity(self, out: pd.DataFrame) -> pd.DataFrame:
        """Rule-based plus random flags for manual inspection."""
        rng = random.Random(self.seed)
        flags, reasons = [], []
        for _, row in out.iterrows():
            r = []
            if row["judge_choice"] == "error":
                r.append("judge_error")
            elif row["judge_choice"] in ("pred", "true", "tie"):
                t, p, j = (
                    int(row["true_rating"]),
                    int(row["pred_rating"]),
                    row["judge_rating"],
                )
                if row["judge_confidence"] == "low":
                    r.append("low_confidence")
                if row["tie_reason"] == "position_inconsistent":
                    r.append("position_inconsistent")
                if row["tie_reason"] == "judge_neither":
                    r.append("judge_rejected_both")
                if abs(t - p) >= 3:
                    r.append("large_gap")
                if j is not None and not pd.isna(j):
                    if row["judge_choice"] == "pred" and abs(j - p) > abs(j - t):
                        r.append("verdict_contradicts_own_rating")
                    if row["judge_choice"] == "true" and abs(j - t) > abs(j - p):
                        r.append("verdict_contradicts_own_rating")
                if rng.random() < self.audit_rate:
                    r.append("random_audit")
            flags.append(bool(r))
            reasons.append(";".join(r))
        out["sanity_flag"] = flags
        out["sanity_reasons"] = reasons
        return out

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def judge_summary(self, results: Optional[pd.DataFrame] = None) -> dict:
        r = self._results(results)
        c = self.columns
        judged = r[r["judge_choice"].isin(["pred", "true", "tie"])]
        n = len(judged)
        share = lambda k: (
            float((judged["judge_choice"] == k).mean()) if n else float("nan")
        )
        summary = {
            "n_rows": len(r),
            "n_agree": int((r["judge_choice"] == "agree").sum()),
            "n_invalid_input": int((r["judge_choice"] == "invalid_input").sum()),
            "n_judged_disagreements": n,
            "n_judge_errors": int((r["judge_choice"] == "error").sum()),
            "share_pred_favored": share("pred"),
            "share_true_favored": share("true"),
            "share_tie": share("tie"),
            "tie_reasons": judged["tie_reason"].value_counts().to_dict(),
            "position_consistency_rate": (
                float(judged["position_consistent"].astype(float).mean())
                if n
                else float("nan")
            ),
            "confidence_distribution": judged["judge_confidence"]
            .value_counts()
            .to_dict(),
            "n_labels_changed": int(r["label_changed"].sum()),
            "n_excluded_ties": int(
                (r["excluded_from_adjudicated"] & r["is_tie"]).sum()
            ),
            "n_judge_calls": int(r["n_judge_calls"].sum()),
        }
        if n:
            gap = (
                (judged["pred_rating"].astype(int) - judged["true_rating"].astype(int))
                .abs()
                .rename("abs_gap")
            )
            summary["choice_by_gap"] = pd.crosstab(
                gap, judged["judge_choice"], normalize="index"
            )
            summary["choice_by_human_label"] = pd.crosstab(
                judged["true_rating"].astype(int).rename("human_label"),
                judged["judge_choice"],
                normalize="index",
            )
        return summary

    def compute_metrics(self, results: Optional[pd.DataFrame] = None) -> dict:
        """Metrics before and after adjudication, plus judge diagnostics.

        pred_vs_human            original evaluation
        pred_vs_adjudicated      evaluation against judge-adjudicated labels
        pred_vs_human_same_rows  original evaluation on the same rows as
                                 pred_vs_adjudicated (differs only under "drop")
        judge_rating_vs_human    the judge's own blind rating as a rater, on the
                                 judged (disagreement) rows only, which are the
                                 hardest cases, so expect it to look low

        Caveat for the paper: adjudication can only move labels towards the
        prediction, so pred_vs_adjudicated is optimistic by construction.
        Report it alongside pred_vs_human and the manual sanity-check results.
        """
        r = self._results(results)
        valid = r[r["judge_choice"] != "invalid_input"]
        adj = valid[valid["adjudicated_label"].notna()]
        judged = valid[valid["judge_rating"].notna()]
        return {
            "judge_summary": self.judge_summary(r),
            "pred_vs_human": classification_metrics(
                valid["true_rating"], valid["pred_rating"], self.labels
            ),
            "pred_vs_adjudicated": classification_metrics(
                adj["adjudicated_label"], adj["pred_rating"], self.labels
            ),
            "pred_vs_human_same_rows": classification_metrics(
                adj["true_rating"], adj["pred_rating"], self.labels
            ),
            "judge_rating_vs_human": classification_metrics(
                judged["true_rating"], judged["judge_rating"].astype(int), self.labels
            ),
        }

    @staticmethod
    def metrics_table(metrics: dict) -> pd.DataFrame:
        """Scalar metrics side by side, ready for a paper table."""
        keys = [
            "pred_vs_human",
            "pred_vs_human_same_rows",
            "pred_vs_adjudicated",
            "judge_rating_vs_human",
        ]
        cols = {
            k: {m: v for m, v in metrics[k].items() if np.isscalar(v)}
            for k in keys
            if k in metrics
        }
        return pd.DataFrame(cols)

    # ------------------------------------------------------------------ #
    # Sanity checking
    # ------------------------------------------------------------------ #
    def sample_for_review(
        self,
        results: Optional[pd.DataFrame] = None,
        n_per_choice: int = 10,
        max_flagged: int = 40,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Build a sheet of judge outputs to inspect by hand.

        Contains a stratified random sample per judge_choice (pred / true / tie)
        plus rule-flagged rows, with blank columns to fill in:
          annotator_rating            your own 1-5 rating of the review
          annotator_agrees_with_judge yes / no / unsure
          annotator_notes             free text
        Hide the human/pred columns while you rate if you want a blind check.
        """
        r = self._results(results)
        seed = self.seed if seed is None else seed
        judged = r[r["judge_choice"].isin(["pred", "true", "tie"])]
        parts = []
        for choice, grp in judged.groupby("judge_choice"):
            take = grp.sample(min(n_per_choice, len(grp)), random_state=seed)
            parts.append(take.assign(sheet_reason="stratified_" + choice))
        flagged = judged[judged["sanity_flag"]]
        flagged = flagged.sample(min(max_flagged, len(flagged)), random_state=seed)
        parts.append(
            flagged.assign(sheet_reason="flagged:" + flagged["sanity_reasons"])
        )
        sheet = pd.concat(parts) if parts else judged.head(0).assign(sheet_reason=None)
        sheet = sheet[~sheet.index.duplicated(keep="first")]

        c = self.columns
        keep = [
            col
            for col in [
                c["id"],
                c["review"],
                "true_rating",
                "pred_rating",
                "judge_choice",
                "selected_rating",
                "tie_reason",
                "judge_rating",
                "judge_confidence",
                "position_consistent",
                "judge_evidence",
                "judge_reasoning",
                "sanity_reasons",
                "sheet_reason",
            ]
            if col in sheet.columns
        ]
        sheet = sheet[keep].copy()
        sheet["annotator_rating"] = ""
        sheet["annotator_agrees_with_judge"] = ""
        sheet["annotator_notes"] = ""
        return sheet

    def export_sanity_sheet(self, path: str, **kwargs) -> pd.DataFrame:
        sheet = self.sample_for_review(**kwargs)
        sheet.to_csv(path, index=False)
        return sheet

    def score_sanity_sheet(self, sheet: pd.DataFrame) -> dict:
        """Score a filled-in sanity sheet: how often you agree with the judge,
        and whose label (human or prediction) your own rating matches."""
        out = {}
        agree = sheet["annotator_agrees_with_judge"].astype(str).str.strip().str.lower()
        answered = agree.isin(["yes", "no"])
        out["n_answered"] = int(answered.sum())
        out["agreement_with_judge"] = (
            float((agree[answered] == "yes").mean()) if answered.any() else float("nan")
        )

        rated = pd.to_numeric(sheet["annotator_rating"], errors="coerce")
        m = rated.notna()
        if m.any():
            ann = rated[m].astype(int)
            out["n_rated"] = int(m.sum())
            out["annotator_matches_human"] = float(
                (ann == sheet.loc[m, "true_rating"].astype(int)).mean()
            )
            out["annotator_matches_pred"] = float(
                (ann == sheet.loc[m, "pred_rating"].astype(int)).mean()
            )
            j = pd.to_numeric(sheet.loc[m, "judge_rating"], errors="coerce")
            jm = j.notna()
            if jm.sum() > 1:
                out["qwk_annotator_vs_judge_rating"] = cohen_kappa_score(
                    ann[jm],
                    j[jm].astype(int),
                    labels=list(self.labels),
                    weights="quadratic",
                )
            nt = sheet.loc[m, "judge_choice"].isin(["pred", "true"])
            if nt.any():
                sel = sheet.loc[m, "selected_rating"][nt].astype(int)
                out["annotator_matches_judge_selection"] = float(
                    (ann[nt] == sel).mean()
                )
        return out

    def run_controls(
        self, df: pd.DataFrame, n: int = 20, seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Canary test: pit the label of an agreement row (pred == human)
        against an obviously wrong decoy (1 for labels >= 3, else 5).
        A sound judge should almost always pick the real label."""
        c = self.columns
        seed = self.seed if seed is None else seed
        t = self._normalise_ratings(df[c["true"]], c["true"])
        p = self._normalise_ratings(df[c["pred"]], c["pred"])
        pool = df.assign(_t=t.values)[(t == p).fillna(False).values]
        pool = pool.sample(min(n, len(pool)), random_state=seed)
        rows = []
        for idx, row in pool.iterrows():
            label = int(row["_t"])
            decoy = 1 if label >= 3 else 5
            sid = row[c["id"]] if c["id"] in pool.columns else idx
            res = self._adjudicate(
                row[c["review"]], x=decoy, y=label, sample_id=f"control_{sid}"
            )
            rows.append(
                {
                    "sample_id": sid,
                    "review": row[c["review"]],
                    "label": label,
                    "decoy": decoy,
                    "judge_winner": {"x": "decoy", "y": "label"}.get(
                        res["winner"], res["winner"]
                    ),
                    "passed": res["winner"] == "y",
                    "judge_rating": res["judge_rating"],
                    "judge_error_detail": res["judge_error_detail"],
                }
            )
        ctrl = pd.DataFrame(rows)
        if len(ctrl):
            ok = ctrl[ctrl["judge_winner"] != "error"]
            logger.info(
                "Control pass rate: %.1f%% of %d completed cases (%d judge errors)",
                100 * ok["passed"].mean() if len(ok) else float("nan"),
                len(ok),
                len(ctrl) - len(ok),
            )
        return ctrl

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_raw_log(self, path: str) -> None:
        """Every prompt and raw response as JSON lines, for reproducibility."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"config": self.config}) + "\n")
            for rec in self.raw_log:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _results(self, results: Optional[pd.DataFrame]) -> pd.DataFrame:
        r = results if results is not None else self.results_
        if r is None:
            raise RuntimeError("Call evaluate() first or pass a results DataFrame.")
        return r

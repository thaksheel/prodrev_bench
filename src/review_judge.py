"""
LLM-as-a-judge adjudication of star-rating labels for product reviews.

For each review where the predicted rating and the author's rating disagree,
an LLM judge decides which of the two ratings the review text supports better.
Metrics are then computed against the original and the adjudicated labels.

Design components and their sources (BibTeX keys refer to references.bib):

    C1  Pairwise comparison with an explicit tie option [zheng2023judging]
    C2  Position swapping; "strict" (orders must agree) or "pooled" votes
        [zheng2023judging; wang2024fair]
    C3  Evidence-before-verdict chain-of-thought
        [wei2022chain; wang2024fair; liu2023geval]
    C4  Blind rating before the candidates are shown [zheng2023judging]
    C5  Score rubric with anchored reference examples
        [kim2024prometheus; kim2024prometheus2; liu2023geval]
    C6  Anonymised candidates, predictor reasoning hidden
        [zheng2023judging; wang2024fair]
    C7  Self-consistency majority voting [wang2023selfconsistency]
    C8  Model-label disagreement as the signal for candidate label errors
        [northcutt2021confident; chong2022detecting]
    C9  Metrics: accuracy, balanced accuracy, macro F1, MAE, RMSE
        [brodersen2010balanced; pedregosa2011scikit]
"""

import json, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import pandas as pd

from . import (
    LLMProvider,
    JudgeConfig,
    PAIRWISE_TEMPLATE,
    BLIND_SECTION,
    BLIND_TEMPLATE,
    SYSTEM_PROMPT,
    classification_metrics
)


def _format_anchors(anchors: Dict[int, Sequence[str]]) -> str:
    return "\n".join(
        f'- {star} star{"s" if star != 1 else ""}: "{text}"'
        for star in sorted(anchors)
        for text in anchors[star]
    )


def _parse_json(text: Optional[str]) -> Optional[dict]:
    """Extract the first {...} block from an LLM response."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", str(text))
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_rating(value, labels: Sequence[int]) -> Optional[int]:
    """Read a rating such as 4, "4" or "4 stars"."""
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m and int(m.group()) in labels else None


def _parse_verdict(value) -> Optional[str]:
    """Normalise "A", "Rating A", "tie", "neither" to A / B / TIE / NEITHER."""
    words = [
        w
        for w in re.sub(r"[^A-Z ]", " ", str(value).upper()).split()
        if w not in ("RATING", "CANDIDATE", "OPTION")
    ]
    return words[0] if words and words[0] in ("A", "B", "TIE", "NEITHER") else None


def _mode(values: Sequence):
    values = [v for v in values if v is not None]
    return Counter(values).most_common(1)[0][0] if values else None


class ReviewRatingJudge:
    """Pairwise LLM judge that adjudicates author vs. predicted star ratings.

    Parameters
    ----------
    provider : LLMProvider
        The model used as the judge; queried through ``provider.invoke``.
    config : JudgeConfig, optional
        Judge settings. Defaults to ``JudgeConfig()``.

    Attributes
    ----------
    results_ : pandas.DataFrame or None
        Row-level results of the last :meth:`evaluate` call.
    """

    def __init__(self, provider: "LLMProvider", config: Optional[JudgeConfig] = None):
        self.provider = provider
        self.config = config or JudgeConfig()
        self._anchors = _format_anchors(self.config.anchors)
        self.results_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # LLM calls
    # ------------------------------------------------------------------ #
    def _call(self, prompt: str, required_key: str):
        """Query the judge; returns (parsed JSON or None, input tokens, output tokens)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tok_in = tok_out = 0
        for _ in range(self.config.max_retries + 1):
            try:
                out = self.provider.invoke(messages=messages)
            except Exception:  # transient API failure: retry
                continue
            tok_in += out.input_token or 0
            tok_out += out.output_token or 0
            parsed = _parse_json(out.response)
            if parsed is not None and required_key in parsed:
                return parsed, tok_in, tok_out
        return None, tok_in, tok_out

    def _blind_prompt(self, review: str) -> str:
        return BLIND_TEMPLATE.format(
            rubric=self.config.rubric, anchors=self._anchors, review=review
        )

    def _pairwise_prompt(
        self, review: str, a: int, b: int, blind: Optional[dict]
    ) -> str:
        section = ""
        if blind is not None:
            keep = {
                k: blind.get(k)
                for k in ("evidence", "supported_ratings", "best_rating")
            }
            section = BLIND_SECTION.format(
                analysis=json.dumps(keep, ensure_ascii=False)
            )
        return PAIRWISE_TEMPLATE.format(
            rubric=self.config.rubric,
            anchors=self._anchors,
            review=review,
            blind_section=section,
            rating_a=a,
            rating_b=b,
        )

    # ------------------------------------------------------------------ #
    # Adjudication of one review
    # ------------------------------------------------------------------ #
    def _adjudicate(self, review: str, pred: int, true: int) -> dict:
        cfg = self.config
        tok_in = tok_out = 0

        # C4: blind pass
        blind_parses = []
        if cfg.blind_first:
            for _ in range(cfg.n_samples):
                parsed, i, o = self._call(self._blind_prompt(review), "best_rating")
                tok_in, tok_out = tok_in + i, tok_out + o
                if parsed is not None:
                    blind_parses.append(parsed)
        blind_ratings = [
            _parse_rating(p.get("best_rating"), cfg.labels) for p in blind_parses
        ]
        blind_mode = _mode(blind_ratings)
        blind_ref = next(
            (p for p, r in zip(blind_parses, blind_ratings) if r == blind_mode), None
        )

        # C2 + C7: both orders, n_samples each
        votes = {"pred_first": [], "true_first": []}
        confidences, pair_ratings, first = [], [], None
        for order in votes:
            a, b = (pred, true) if order == "pred_first" else (true, pred)
            slot = {
                "A": "pred" if order == "pred_first" else "true",
                "B": "true" if order == "pred_first" else "pred",
                "TIE": "tie",
                "NEITHER": "neither",
            }
            for _ in range(cfg.n_samples):
                parsed, i, o = self._call(
                    self._pairwise_prompt(review, a, b, blind_ref), "verdict"
                )
                tok_in, tok_out = tok_in + i, tok_out + o
                if parsed is None:
                    votes[order].append("error")
                    continue
                votes[order].append(
                    slot.get(_parse_verdict(parsed.get("verdict")), "error")
                )
                confidences.append(str(parsed.get("confidence", "")).lower() or None)
                pair_ratings.append(
                    _parse_rating(parsed.get("best_rating"), cfg.labels)
                )
                first = first or parsed

        choice, tie_reason = self._aggregate(votes)
        return {
            "judge_choice": choice,
            "tie_reason": tie_reason,
            "votes_pred_first": ", ".join(votes["pred_first"]),
            "votes_true_first": ", ".join(votes["true_first"]),
            "judge_rating": (
                blind_mode if blind_mode is not None else _mode(pair_ratings)
            ),
            "judge_confidence": _mode(confidences),
            "judge_evidence": (blind_ref or first or {}).get("evidence"),
            "judge_reasoning": (first or {}).get("reasoning"),
            "judge_input_tokens": tok_in,
            "judge_output_tokens": tok_out,
        }

    def _aggregate(self, votes: Dict[str, List[str]]):
        """Combine votes into (judge_choice, tie_reason)."""

        def majority(vs):
            vs = [v for v in vs if v != "error"]
            if not vs:
                return "error"
            top = Counter(vs).most_common()
            return "tie" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]

        m1, m2 = majority(votes["pred_first"]), majority(votes["true_first"])
        if self.config.aggregation == "strict":  # [zheng2023judging]
            if "error" in (m1, m2):
                return "error", None
            if m1 != m2:
                return "tie", "position_inconsistent"
            winner = m1
        else:  # pooled [wang2024fair]
            winner = majority(votes["pred_first"] + votes["true_first"])

        if winner == "tie":
            return "tie", "judge_tie"
        if winner == "neither":
            return "tie", "judge_neither"
        return winner, None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Judge every disagreement in ``df``; return one row per review.

        Added columns
        -------------
        judge_choice       'agree' (not judged), 'pred', 'true', 'tie' or 'error'
        tie_reason         judge_tie, judge_neither or position_inconsistent
        selected_rating    rating chosen by the judge (empty for ties and errors)
        adjudicated_label  label used for the adjudicated metrics
        label_changed      True where the adjudicated label differs from the author's
        """
        cfg = self.config
        out = df.reset_index(drop=True).copy()
        true = out[cfg.true_col].astype(int)
        pred = out[cfg.pred_col].astype(int)

        def judge_row(i):
            if true[i] == pred[i]:
                return {"judge_choice": "agree"}
            return self._adjudicate(
                out.at[i, cfg.review_col], int(pred[i]), int(true[i])
            )

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            judged = pd.DataFrame(list(pool.map(judge_row, range(len(out)))))
        out = pd.concat([out, judged], axis=1)

        choice = out["judge_choice"]
        selected = pd.Series(pd.NA, index=out.index, dtype="Int64")
        selected[choice == "pred"] = pred[choice == "pred"]
        selected[choice.isin(["true", "agree"])] = true[choice.isin(["true", "agree"])]
        out["selected_rating"] = selected
        out["is_tie"] = choice.eq("tie")

        adjudicated = selected.copy()  # errors stay empty under both policies
        if cfg.tie_policy == "signal":
            adjudicated[out["is_tie"]] = true[out["is_tie"]]
        out["adjudicated_label"] = adjudicated
        out["label_changed"] = (adjudicated != true).fillna(False).astype(bool)

        self.results_ = out
        return out

    def summary(self, results: Optional[pd.DataFrame] = None):
        """Return (metrics, judge_counts) as DataFrames.

        ``metrics`` compares the predictions with the author's labels
        ("original") and with the adjudicated labels ("adjudicated"), both on
        the rows that have an adjudicated label, so the two columns are always
        computed on the same reviews.
        """
        r = self.results_ if results is None else results
        cfg = self.config
        rows = r[r["adjudicated_label"].notna()]
        metrics = pd.DataFrame(
            {
                "original": classification_metrics(
                    rows[cfg.true_col], rows[cfg.pred_col]
                ),
                "adjudicated": classification_metrics(
                    rows["adjudicated_label"], rows[cfg.pred_col]
                ),
            }
        )

        judged = r[r["judge_choice"] != "agree"]
        counts = {
            "n_reviews": len(r),
            "n_agree": int((r["judge_choice"] == "agree").sum()),
            "n_disagreements": len(judged),
            "n_pred_favored": int((judged["judge_choice"] == "pred").sum()),
            "n_true_favored": int((judged["judge_choice"] == "true").sum()),
            "n_ties": int(judged["is_tie"].sum()),
            "n_errors": int((judged["judge_choice"] == "error").sum()),
            "n_labels_changed": int(r["label_changed"].sum()),
        }
        for reason in ("judge_tie", "judge_neither", "position_inconsistent"):
            counts[f"ties_{reason}"] = int((judged["tie_reason"] == reason).sum())
        counts_df = pd.Series(counts, name="value").to_frame()
        return metrics, counts_df

    def save_excel(self, path, results: Optional[pd.DataFrame] = None) -> Path:
        """Write the summary and the row-level details to one workbook.

        Sheet "summary": the metrics table, with the judge counts below it.
        Sheet "details": one row per review.
        """
        r = self.results_ if results is None else results
        metrics, counts = self.summary(r)
        path = Path(path).with_suffix(".xlsx")
        path.parent.mkdir(parents=True, exist_ok=True)

        # strip control characters Excel rejects (can appear in LLM output)
        bad = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
        details = r.apply(
            lambda c: (
                c.map(lambda v: bad.sub("", v) if isinstance(v, str) else v)
                if c.dtype == object
                else c
            )
        )

        with pd.ExcelWriter(path) as xw:
            metrics.round(4).to_excel(xw, sheet_name="summary", startrow=0)
            counts.to_excel(xw, sheet_name="summary", startrow=len(metrics) + 3)
            details.to_excel(xw, sheet_name="details", index=False)
        return path

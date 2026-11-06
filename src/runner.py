"""
Runner that evaluates several LLM methods for 1-5 star rating prediction on
one balanced slice of a review dataset, and scores every method both against
the author's ratings and against labels adjudicated by the LLM judge.

Methods
-------
prompt  single-call prompting strategies from prompts.py, run through LLMProvider
mad     multi-agent debate (modules.mad_predict_re.Debate)
mega    mega agent team (modules.mega_agent_re.Runner)
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.prompts import Prompt, PromptStrategy, RatingReasoning
from . import LLMProvider, ReviewRatingJudge, PromptMethod, RunnerConfig

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "mae", "rmse"]


def parse_rating(text) -> float:
    if text is None:
        return np.nan
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S)
    m = re.search(r'"?rating"?\s*[:=]\s*"?([1-5])', text, re.I)
    if m is None:
        m = re.search(r"\b([1-5])\b", text)
    return float(m.group(1)) if m else np.nan


def parse_reasoning(text) -> Optional[str]:
    text = re.sub(r"```(?:json)?", "", str(text))
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s : e + 1]).get("reasoning")
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


class Runner:
    """Evaluate prompting and agentic methods on one balanced slice.

    Parameters
    ----------
    predictor : LLMProvider
        Model used by the prompting methods.
    judge_provider : LLMProvider
        Model used by the LLM judge.
    config : RunnerConfig
        Runner settings.

    Attributes
    ----------
    slice_ : pandas.DataFrame
        The balanced slice of ``sample_size`` rows.
    eval_df_ : pandas.DataFrame
        The rows actually evaluated (the slice, or its balanced subset).
    predictions_ : dict of str to pandas.DataFrame
        Raw predictions per method.
    judged_ : dict of str to pandas.DataFrame
        Judge results per method.
    comparison_ : pandas.DataFrame
        One row per method with author-label and judge-label metrics.
    """

    def __init__(
        self,
        predictor: "LLMProvider",
        judge_provider: "LLMProvider",
        config: RunnerConfig,
    ):
        self.predictor = predictor
        self.config = config
        judge_cfg = replace(
            config.judge,
            review_col="review",
            true_col="true",
            pred_col="pred",
            id_col="sample_id",
        )
        self.judge = ReviewRatingJudge(judge_provider, judge_cfg)
        self.predictions_: Dict[str, pd.DataFrame] = {}
        self.judged_: Dict[str, pd.DataFrame] = {}
        self.comparison_: Optional[pd.DataFrame] = None
        self._load_data()

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def _load_data(self) -> None:
        cfg = self.config
        df = pd.read_csv(cfg.data_path).dropna(subset=[cfg.review_col, cfg.rating_col])
        df[cfg.rating_col] = df[cfg.rating_col].astype(int)
        per_class = cfg.sample_size // 5

        sl = (
            df.groupby(cfg.rating_col, group_keys=False)
            .sample(n=per_class, random_state=cfg.seed)
            .sample(frac=1, random_state=cfg.seed)
        )
        pool = df.drop(index=sl.index)
        pool = pool[~pool[cfg.review_col].isin(set(sl[cfg.review_col]))]
        if cfg.fewshot_pool_size and len(pool) > cfg.fewshot_pool_size:
            pool = pool.sample(n=cfg.fewshot_pool_size, random_state=cfg.seed)

        sl = sl.reset_index().rename(columns={"index": "source_index"})
        sl.insert(0, "sample_id", [f"s{i:05d}" for i in range(len(sl))])
        self.slice_ = sl
        self.pool_ = pool.reset_index(drop=True)

        if cfg.eval_limit:
            ev = sl.groupby(cfg.rating_col, group_keys=False).head(cfg.eval_limit // 5)
            self.eval_df_ = ev.sort_values("sample_id").reset_index(drop=True)
        else:
            self.eval_df_ = sl
        if cfg.slice_path:
            Path(cfg.slice_path).parent.mkdir(parents=True, exist_ok=True)
            sl.to_csv(cfg.slice_path, index=False)

    def _frame(
        self,
        preds,
        reasonings=None,
        in_tok=None,
        out_tok=None,
        durations=None,
        errors=None,
    ) -> pd.DataFrame:
        cfg = self.config
        n = len(self.eval_df_)
        return pd.DataFrame(
            {
                "sample_id": self.eval_df_["sample_id"],
                "review": self.eval_df_[cfg.review_col],
                "true": self.eval_df_[cfg.rating_col].astype(int),
                "pred": pd.to_numeric(pd.Series(preds), errors="coerce").where(
                    lambda s: s.isin([1, 2, 3, 4, 5])
                ),
                "reasoning": reasonings if reasonings is not None else [None] * n,
                "input_tokens": in_tok if in_tok is not None else [np.nan] * n,
                "output_tokens": out_tok if out_tok is not None else [np.nan] * n,
                "duration": durations if durations is not None else [np.nan] * n,
                "error": errors if errors is not None else [None] * n,
            }
        )

    # ------------------------------------------------------------------ #
    # Prompting methods
    # ------------------------------------------------------------------ #
    def run_prompt_method(self, name: str, method: PromptMethod) -> pd.DataFrame:

        fmt = RatingReasoning[method.output_format]
        embeddings = None

        def build(review):
            prompt = Prompt(
                review=review,
                chain_of_thought=method.chain_of_thought,
                rreasoning=fmt,
                num_examples=method.num_examples,
            )
            return PromptStrategy(
                prompt,
                method.strategy,
                self.pool_,
                method.few_shots_type,
                embeddings=embeddings,
            )

        def predict(review):
            t0 = time.time()
            try:
                ps = build(review)
                obj = (
                    ps.zero_shot()
                    if method.strategy == "zero-shot"
                    else ps.few_shots(self.pool_)
                )
                out = self.predictor.invoke(
                    messages=[
                        {"role": "system", "content": obj["system"]},
                        {"role": "user", "content": obj["user"]},
                    ]
                )
                return (
                    parse_rating(out.response),
                    parse_reasoning(out.response),
                    out.input_token,
                    out.output_token,
                    time.time() - t0,
                    None,
                )
            except Exception as exc:
                return (
                    np.nan,
                    None,
                    np.nan,
                    np.nan,
                    time.time() - t0,
                    f"{type(exc).__name__}: {exc}",
                )

        reviews = self.eval_df_[self.config.review_col].tolist()
        rows = []
        if method.strategy == "few-shots" and method.few_shots_type == "kate":
            ps = build(reviews[0])
            ps.few_shots(self.pool_)
            embeddings = ps.embeddings
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            rows = list(pool.map(predict, reviews))

        p, r, i, o, d, e = map(list, zip(*rows))
        return self._frame(p, r, i, o, d, e)

    # ------------------------------------------------------------------ #
    # Agentic methods
    # ------------------------------------------------------------------ #
    def run_mad(self) -> pd.DataFrame:
        from modules.mad_predict_re import Debate

        m = self.config.mad
        with open(m.references_path, "r", encoding="utf-8") as f:
            references = json.load(f)
        debate = Debate(
            model_name=m.model_name,
            device=m.device,
            max_new_token=m.max_new_token,
            num_agents=m.num_agents,
            num_rounds=m.num_rounds,
            references=references,
            provider=m.provider,
            api_key=m.api_key,
        )
        reviews = self.eval_df_[self.config.review_col].tolist()
        t0 = time.time()
        results, reasonings = debate.simulate_debate(reviews)
        per_review = (time.time() - t0) / max(len(reviews), 1)
        debate.save_reasoning()
        return self._frame(
            results, [str(r) for r in reasonings], durations=[per_review] * len(reviews)
        )

    def run_mega(self) -> pd.DataFrame:
        from modules.mega_agent_re import MegaPrompt, Runner

        m = self.config.mega
        task = MegaPrompt().get_mega_instructions(
            m.params.ceo_name,
            m.task_instruction,
            max_agents=m.params.max_subordinates + 1,
        )
        runner = Runner(log_path=m.log_path, run_dir=m.run_dir)
        if Path(m.plan_path).exists():
            runner.load_plan(m.plan_path)
        else:
            runner.plan(m.params, task)
            runner.save_plan(m.plan_path)

        out = runner.run_many(
            m.params,
            task,
            self.eval_df_[self.config.review_col].tolist(),
            trues=self.eval_df_[self.config.rating_col].to_numpy(),
            prompt_template=m.prompt_template,
            max_workers=m.max_workers,
            checkpoint_path=m.checkpoint_path,
        ).reset_index(drop=True)
        col = lambda c: out[c].tolist() if c in out.columns else None
        return self._frame(
            out["pred"],
            col("reasoning"),
            col("total_input_token"),
            col("total_output_token"),
            col("duration"),
            col("error"),
        )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def _judge_method(self, name: str, preds: pd.DataFrame) -> Dict[str, Any]:
        valid = preds[preds["pred"].notna()].copy()
        valid["pred"] = valid["pred"].astype(int)
        judged = self.judge.evaluate(valid) if len(valid) else valid
        self.judged_[name] = preds.drop(columns=["pred"]).merge(
            judged.drop(
                columns=[
                    "review",
                    "true",
                    "reasoning",
                    "input_tokens",
                    "output_tokens",
                    "duration",
                    "error",
                ],
                errors="ignore",
            ),
            on="sample_id",
            how="left",
        )
        row = {
            "method": name,
            "n": len(preds),
            "n_valid": len(valid),
            "coverage": len(valid) / len(preds) if len(preds) else np.nan,
        }
        if len(valid):
            metrics, counts = self.judge.summary(judged)
            for m in METRICS:
                row[f"{m}_human"] = metrics.loc[m, "original"]
                row[f"{m}_judge"] = metrics.loc[m, "adjudicated"]
            c = counts["value"]
            row.update(
                {
                    k: c[k]
                    for k in (
                        "n_disagreements",
                        "n_pred_favored",
                        "n_true_favored",
                        "n_ties",
                        "n_errors",
                        "n_labels_changed",
                    )
                }
            )
        row.update(
            {
                "mean_input_tokens": preds["input_tokens"].mean(),
                "mean_output_tokens": preds["output_tokens"].mean(),
                "mean_duration_s": preds["duration"].mean(),
            }
        )
        return row

    def run(self, methods: Optional[List[str]] = None) -> pd.DataFrame:
        """Run the selected methods (default: every configured one) and judge each."""
        cfg = self.config
        jobs = {
            n: (lambda n=n, m=m: self.run_prompt_method(n, m))
            for n, m in cfg.prompt_methods.items()
        }
        if cfg.mad is not None:
            jobs["mad"] = self.run_mad
        if cfg.mega is not None:
            jobs["mega"] = self.run_mega
        if methods is not None:
            jobs = {n: jobs[n] for n in methods}

        rows = []
        for name, job in jobs.items():
            print(f"[{name}] predicting {len(self.eval_df_)} reviews")
            self.predictions_[name] = job()
            print(f"[{name}] judging")
            rows.append(self._judge_method(name, self.predictions_[name]))
        self.comparison_ = pd.DataFrame(rows).set_index("method")
        return self.comparison_

    def save_excel(self, path: Optional[str] = None) -> Path:
        """Sheet "comparison", sheet "eval_rows", then one sheet per method."""
        path = Path(path or self.config.output_path).with_suffix(".xlsx")
        path.parent.mkdir(parents=True, exist_ok=True)
        bad = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
        clean = lambda d: d.apply(
            lambda c: (
                c.map(lambda v: bad.sub("", v) if isinstance(v, str) else v)
                if c.dtype == object
                else c
            )
        )
        with pd.ExcelWriter(path) as xw:
            self.comparison_.round(4).to_excel(xw, sheet_name="comparison")
            clean(self.eval_df_).to_excel(xw, sheet_name="eval_rows", index=False)
            for name, d in self.judged_.items():
                clean(d).to_excel(xw, sheet_name=name[:31], index=False)
        return path

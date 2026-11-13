import json, re, time 
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.prompts import BertEncoder, KATEExamples, PromptBuilder, RandomExamples
from .mad import Debate
from .mega import MegaRunner, MegaPrompt
from . import LLMProvider, ReviewRatingJudge, PromptMethod, RunnerConfig

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "mae", "rmse"]
LABELS = [1, 2, 3, 4, 5]
_EXCEL_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


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


def per_label_accuracy(judged: pd.DataFrame) -> Dict[str, float]:
    """Accuracy within each author rating, against author and adjudicated labels."""
    rows = judged[judged["adjudicated_label"].notna()]
    out = {}
    for label in LABELS:
        group = rows[rows["true"] == label]
        if len(group) == 0:
            out[f"acc_label{label}_human"] = np.nan
            out[f"acc_label{label}_judge"] = np.nan
            continue
        pred = group["pred"].astype(int)
        out[f"acc_label{label}_human"] = float(
            (pred == group["true"].astype(int)).mean()
        )
        out[f"acc_label{label}_judge"] = float(
            (pred == group["adjudicated_label"].astype(int)).mean()
        )
    return out


def _column(values, n: int, blank) -> list:
    if values is None:
        return [blank] * n
    return list(values)


def _optional_column(df: pd.DataFrame, name: str) -> Optional[list]:
    if name in df.columns:
        return df[name].tolist()
    return None


def _seed_path(path: Optional[str], seed: int) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    return str(p.with_name(f"{p.stem}_seed{seed}{p.suffix}"))


def _excel_clean_value(value):
    if isinstance(value, str):
        return _EXCEL_ILLEGAL.sub("", value)
    return value


def _excel_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(_excel_clean_value)
    return out


class Runner:
    """Evaluate prompting and agentic methods on one balanced sample per seed.

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
    slices_ : dict of int to pandas.DataFrame
        The balanced sample for each seed.
    pools_ : dict of int to pandas.DataFrame
        The few-shot example pool for each seed (rows outside its sample).
    predictions_ : dict of (int, str) to pandas.DataFrame
        Raw predictions per seed and method.
    judged_ : dict of (int, str) to pandas.DataFrame
        Predictions plus judge results per seed and method.
    prompts_ : list of dict
        Prompts recorded per seed and method.
    comparison_ : pandas.DataFrame
        One row per seed and method.
    summary_ : pandas.DataFrame
        Mean and standard deviation across seeds, one row per method.
    """

    def __init__(
        self,
        predictor: LLMProvider,
        judge_provider: LLMProvider,
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

        self.df_ = self._load_data()
        self.slices_: Dict[int, pd.DataFrame] = {}
        self.pools_: Dict[int, pd.DataFrame] = {}
        for seed in config.seeds:
            self.slices_[seed], self.pools_[seed] = self._draw_sample(seed)
        if config.slice_path:
            Path(config.slice_path).parent.mkdir(parents=True, exist_ok=True)
            pd.concat(self.slices_.values()).to_csv(config.slice_path, index=False)

        self.predictions_: Dict[Tuple[int, str], pd.DataFrame] = {}
        self.judged_: Dict[Tuple[int, str], pd.DataFrame] = {}
        self.prompts_: List[Dict[str, Any]] = []
        self.comparison_: Optional[pd.DataFrame] = None
        self.summary_: Optional[pd.DataFrame] = None

        self._encoder = None
        self._debate = None
        self._mad_references = None
        self._mega_runner = None
        self._mega_task = None

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def _load_data(self) -> pd.DataFrame:
        cfg = self.config
        df = pd.read_csv(cfg.data_path).dropna(subset=[cfg.review_col, cfg.rating_col])
        df[cfg.rating_col] = df[cfg.rating_col].astype(int)
        return df

    def _draw_sample(self, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        per_class = cfg.sample_size // 5
        sample = (
            self.df_.groupby(cfg.rating_col, group_keys=False)
            .sample(n=per_class, random_state=seed)
            .sample(frac=1, random_state=seed)
        )
        pool = self.df_.drop(index=sample.index)
        pool = pool[~pool[cfg.review_col].isin(set(sample[cfg.review_col]))]
        if cfg.fewshot_pool_size and len(pool) > cfg.fewshot_pool_size:
            pool = pool.sample(n=cfg.fewshot_pool_size, random_state=seed)

        sample = sample.reset_index().rename(columns={"index": "source_index"})
        sample.insert(
            0, "sample_id", [f"seed{seed}_s{i:04d}" for i in range(len(sample))]
        )
        sample.insert(1, "seed", seed)
        return sample, pool.reset_index(drop=True)

    def _frame(
        self,
        seed: int,
        preds,
        n_llm_calls=None,
        reasonings=None,
        in_tok=None,
        out_tok=None,
        durations=None,
        errors=None,
    ) -> pd.DataFrame:
        cfg = self.config
        data = self.slices_[seed]
        n = len(data)
        pred = pd.to_numeric(pd.Series(list(preds)), errors="coerce")
        pred = pred.where(pred.isin(LABELS))
        return pd.DataFrame(
            {
                "seed": seed,
                "sample_id": data["sample_id"],
                "review": data[cfg.review_col],
                "true": data[cfg.rating_col].astype(int),
                "pred": pred,
                "reasoning": _column(reasonings, n, None),
                "n_llm_calls": _column(n_llm_calls, n, np.nan),
                "input_tokens": _column(in_tok, n, np.nan),
                "output_tokens": _column(out_tok, n, np.nan),
                "duration": _column(durations, n, np.nan),
                "error": _column(errors, n, None),
            }
        )

    # ------------------------------------------------------------------ #
    # Prompt log
    # ------------------------------------------------------------------ #
    def _record_prompt(
        self,
        seed: int,
        method: str,
        sample_id: str,
        system: Optional[str],
        user: Optional[str],
        note: Optional[str] = None,
    ) -> None:
        self.prompts_.append(
            {
                "seed": seed,
                "method": method,
                "sample_id": sample_id,
                "system": system,
                "user": user,
                "note": note,
            }
        )

    def _get_encoder(self) -> BertEncoder:
        if self._encoder is None:
            cfg = self.config
            self._encoder = BertEncoder(
                cfg.kate_model_path, cfg.kate_device, cfg.kate_batch_size
            )
        return self._encoder

    def _make_builder(self, seed: int, method: PromptMethod) -> PromptBuilder:
        cfg = self.config
        if method.strategy == "zero-shot":
            return PromptBuilder("zero-shot", method.chain_of_thought)
        pool = self.pools_[seed]
        if method.few_shots_type == "kate":
            examples = KATEExamples(
                pool,
                self._get_encoder(),
                method.num_examples,
                cfg.review_col,
                cfg.rating_col,
            )
        else:
            examples = RandomExamples(
                pool, method.num_examples, seed, cfg.review_col, cfg.rating_col
            )
        return PromptBuilder("few-shots", method.chain_of_thought, examples)

    def _predict_one(self, builder: PromptBuilder, review: str) -> Dict[str, Any]:
        start = time.time()
        result = {
            "pred": np.nan,
            "reasoning": None,
            "input_tokens": np.nan,
            "output_tokens": np.nan,
            "n_llm_calls": 0,
            "duration": np.nan,
            "error": None,
            "messages": None,
        }
        try:
            messages = builder.messages(review)
            result["messages"] = messages
            result["n_llm_calls"] = 1
            out = self.predictor.invoke(messages=messages)
            result["pred"] = parse_rating(out.response)
            result["reasoning"] = parse_reasoning(out.response)
            result["input_tokens"] = out.input_token
            result["output_tokens"] = out.output_token
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["duration"] = time.time() - start
        return result

    def run_prompt_method(
        self, seed: int, name: str, method: PromptMethod
    ) -> pd.DataFrame:
        data = self.slices_[seed]
        reviews = data[self.config.review_col].tolist()
        builder = self._make_builder(seed, method)
        builder.prepare(reviews)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._predict_one, builder, review)
                for review in reviews
            ]
            results = [future.result() for future in futures]

        if self.config.prompt_log == "all":
            to_log = len(results)
        else:
            to_log = 1
        for sample_id, result in zip(data["sample_id"][:to_log], results[:to_log]):
            messages = result["messages"] or [{"content": None}, {"content": None}]
            self._record_prompt(
                seed, name, sample_id, messages[0]["content"], messages[1]["content"]
            )

        frame = pd.DataFrame(results)
        return self._frame(
            seed,
            frame["pred"],
            frame["n_llm_calls"],
            frame["reasoning"],
            frame["input_tokens"],
            frame["output_tokens"],
            frame["duration"],
            frame["error"],
        )

    # ------------------------------------------------------------------ #
    # Agentic methods
    # ------------------------------------------------------------------ #
    def _get_debate(self) -> Debate:
        if self._debate is None:
            m = self.config.mad
            with open(m.references_path, "r", encoding="utf-8") as f:
                self._mad_references = json.load(f)
            self._debate = Debate(
                model_name=m.model_name,
                device=m.device,
                max_new_token=m.max_new_token,
                num_agents=m.num_agents,
                num_rounds=m.num_rounds,
                references=self._mad_references,
                provider=m.provider,
                api_key=m.api_key,
            )
        return self._debate

    def run_mad(self, seed: int) -> pd.DataFrame:
        m = self.config.mad
        data = self.slices_[seed]
        reviews = data[self.config.review_col].tolist()
        debate = self._get_debate()

        self._record_prompt(
            seed,
            "mad",
            data["sample_id"].iloc[0],
            system=json.dumps(self._mad_references, ensure_ascii=False),
            user=reviews[0],
            note=f"Debate builds its agent prompts internally; recorded here are its inputs "
            f"(references and review) for {m.num_agents} agents over {m.num_rounds} rounds.",
        )

        start = time.time()
        results, reasonings = debate.simulate_debate(reviews)
        per_review = (time.time() - start) / max(len(reviews), 1)
        debate.save_reasoning()

        n = len(reviews)
        calls = [m.num_agents * m.num_rounds] * n
        return self._frame(
            seed,
            results,
            calls,
            [str(r) for r in reasonings],
            durations=[per_review] * n,
        )

    def _get_mega_runner(self) -> MegaRunner:
        if self._mega_runner is None:
            m = self.config.mega
            self._mega_task = MegaPrompt().get_mega_instructions(
                m.params.ceo_name,
                m.task_instruction,
                max_agents=m.params.max_subordinates + 1,
            )
            self._mega_runner = MegaRunner(log_path=m.log_path, run_dir=m.run_dir)
            if Path(m.plan_path).exists():
                self._mega_runner.load_plan(m.plan_path)
            else:
                self._mega_runner.plan(m.params, self._mega_task)
                self._mega_runner.save_plan(m.plan_path)
        return self._mega_runner

    def run_mega(self, seed: int) -> pd.DataFrame:
        m = self.config.mega
        data = self.slices_[seed]
        reviews = data[self.config.review_col].tolist()
        runner = self._get_mega_runner()

        self._record_prompt(
            seed,
            "mega",
            data["sample_id"].iloc[0],
            system=self._mega_task,
            user=m.prompt_template.format(review=reviews[0]),
            note="Task instruction and per-review prompt given to the team leader; "
            "subordinate agents build their own prompts from these.",
        )

        out = runner.run_many(
            m.params,
            self._mega_task,
            reviews,
            trues=data[self.config.rating_col].to_numpy(),
            prompt_template=m.prompt_template,
            max_workers=m.max_workers,
            checkpoint_path=_seed_path(m.checkpoint_path, seed),
        ).reset_index(drop=True)

        return self._frame(
            seed,
            out["pred"],
            _optional_column(out, "n_llm_calls"),
            _optional_column(out, "reasoning"),
            _optional_column(out, "total_input_token"),
            _optional_column(out, "total_output_token"),
            _optional_column(out, "duration"),
            _optional_column(out, "error"),
        )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def _judge_method(
        self, seed: int, name: str, preds: pd.DataFrame
    ) -> Dict[str, Any]:
        valid = preds[preds["pred"].notna()].copy()
        valid["pred"] = valid["pred"].astype(int)
        judged = self.judge.evaluate(valid) if len(valid) else valid

        judge_columns = ["sample_id"] + [
            c for c in judged.columns if c not in preds.columns
        ]
        self.judged_[(seed, name)] = preds.merge(
            judged[judge_columns], on="sample_id", how="left"
        )

        row = {
            "seed": seed,
            "method": name,
            "n": len(preds),
            "n_valid": len(valid),
            "coverage": len(valid) / len(preds) if len(preds) else np.nan,
        }
        if len(valid):
            metrics, counts = self.judge.summary(judged)
            for metric in METRICS:
                row[f"{metric}_human"] = metrics.loc[metric, "original"]
                row[f"{metric}_judge"] = metrics.loc[metric, "adjudicated"]
            row.update(per_label_accuracy(judged))
            for key in (
                "n_disagreements",
                "n_pred_favored",
                "n_true_favored",
                "n_ties",
                "n_errors",
                "n_labels_changed",
            ):
                row[key] = counts.loc[key, "value"]
        row["mean_llm_calls"] = preds["n_llm_calls"].mean()
        row["mean_input_tokens"] = preds["input_tokens"].mean()
        row["mean_output_tokens"] = preds["output_tokens"].mean()
        row["mean_duration_s"] = preds["duration"].mean()
        return row

    def available_methods(self) -> List[str]:
        """Names of every configured method, in run order."""
        names = list(self.config.prompt_methods)
        if self.config.mad is not None:
            names.append("mad")
        if self.config.mega is not None:
            names.append("mega")
        return names

    def predict(self, seed: int, name: str) -> pd.DataFrame:
        """Run one method by name on one seed's sample and return its predictions."""
        if name == "mad":
            return self.run_mad(seed)
        if name == "mega":
            return self.run_mega(seed)
        return self.run_prompt_method(seed, name, self.config.prompt_methods[name])

    def _summarize(self, comparison: pd.DataFrame) -> pd.DataFrame:
        grouped = comparison.drop(columns="seed").groupby("method", sort=False)
        stats = grouped.agg(["mean", "std"])
        stats.columns = [f"{column}_{stat}" for column, stat in stats.columns]
        stats.insert(0, "n_seeds", grouped.size())
        return stats

    def run(self, methods: Optional[List[str]] = None) -> pd.DataFrame:
        """Run the selected methods (default: every configured one) on every seed and judge each.

        Returns the summary across seeds; per-seed rows are in ``comparison_``.
        """
        if methods is None:
            methods = self.available_methods()

        rows = []
        for seed in self.config.seeds:
            for name in methods:
                print(
                    f"[seed {seed}] [{name}] predicting {len(self.slices_[seed])} reviews"
                )
                predictions = self.predict(seed, name)
                self.predictions_[(seed, name)] = predictions

                print(f"[seed {seed}] [{name}] judging")
                rows.append(self._judge_method(seed, name, predictions))

        self.comparison_ = pd.DataFrame(rows)
        self.summary_ = self._summarize(self.comparison_)
        return self.summary_

    def save_excel(self, path: Optional[str] = None) -> Path:
        """Sheets: summary, by_seed, prompts, samples, then one sheet per method (all seeds)."""
        path = Path(path or self.config.output_path).with_suffix(".xlsx")
        path.parent.mkdir(parents=True, exist_ok=True)
        method_names = list(dict.fromkeys(name for _, name in self.judged_))

        with pd.ExcelWriter(path) as xw:
            self.summary_.round(4).to_excel(xw, sheet_name="summary")
            self.comparison_.round(4).to_excel(xw, sheet_name="by_seed", index=False)
            _excel_clean(pd.DataFrame(self.prompts_)).to_excel(
                xw, sheet_name="prompts", index=False
            )
            _excel_clean(pd.concat(self.slices_.values())).to_excel(
                xw, sheet_name="samples", index=False
            )
            for name in method_names:
                frames = [
                    self.judged_[(seed, name)]
                    for seed in self.config.seeds
                    if (seed, name) in self.judged_
                ]
                _excel_clean(pd.concat(frames)).to_excel(
                    xw, sheet_name=name[:31], index=False
                )
        return path

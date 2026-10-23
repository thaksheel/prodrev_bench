import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv 
import pandas as pd

from src.review_judge import ReviewRatingJudge, openai_llm, repair_mojibake

load_dotenv()

CONFIG = {
    # --- Data ---------------------------------------------------------------
    "INPUT_PATH": "./exports/mega_results_200.xlsx",
    "SHEET": 0,  # Excel sheet name or index
    "REVIEW_COL": "review",
    "TRUE_COL": "true",
    "PRED_COL": "pred",
    "ID_COL": "sample_id",
    "LIMIT": None,  # first N rows only; None = all rows
    "FIX_ENCODING": True,  # repair mojibake such as "itâ€™s"
    "ROUND_FRACTIONAL": False,  # True rounds ratings such as 4.5; False excludes them
    # --- Judge model ----------------------------------------------------------
    "PROVIDER": "openai",  # "anthropic" or "openai"
    "JUDGE_MODEL": "gpt-6-luna",  # exact model id, required
    "PREDICTOR_MODEL": None,  # model that produced the predictions (self-preference check)
    "PASS_TEMPERATURE": True,  # False for models that reject the temperature parameter
    "MAX_TOKENS": 1024,  # Anthropic only
    # --- Judging protocol -----------------------------------------------------
    "TIE_POLICY": "signal",  # "signal" keeps ties (human label); "drop" excludes them
    "AGGREGATION": "strict",  # "strict" (orders must agree) or "pooled"
    "N_SAMPLES": 1,  # judgments per position order; >1 enables majority voting
    "TEMPERATURE": 1,  # None: 0.0 if N_SAMPLES == 1, else 0.7
    "BLIND_FIRST": True,  # blind rating pass before the pairwise comparison
    # --- Reliability ----------------------------------------------------------
    "MAX_WORKERS": 4,  # parallel requests; lower this if you hit rate limits
    "MAX_RETRIES": 2,
    "MAX_ERROR_RATE": 0.2,  # stop if more than this share of judged rows fail
    # --- Sanity checking ------------------------------------------------------
    "CONTROLS": 20,  # decoy control cases; 0 = skip
    "SHEET_N": 10,  # sanity-sheet rows per judge choice
    "AUDIT_RATE": 0.05,  # share of judged rows randomly flagged for audit
    "SEED": 0,
    # --- Output ---------------------------------------------------------------
    "OUTPUT_DIR": "./exports",  # None = ./judge_outputs/<input-name>_<timestamp>
}

try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except ImportError:
    ILLEGAL_CHARACTERS_RE = None

HERE = Path(__file__).resolve().parent
log = logging.getLogger("run_judge")


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else HERE / p


def validate_config(cfg: dict) -> None:
    """Catch configuration mistakes before any API call is made."""
    missing = [k for k in CONFIG if k not in cfg]
    if missing:
        raise KeyError(f"CONFIG is missing keys: {missing}")
    model = str(cfg["JUDGE_MODEL"] or "").strip()
    if not model or "YOUR" in model.upper() or model.startswith("<"):
        raise ValueError("Set CONFIG['JUDGE_MODEL'] to a real model id before running.")
    if cfg["PROVIDER"] not in ("anthropic", "openai"):
        raise ValueError("CONFIG['PROVIDER'] must be 'anthropic' or 'openai'.")
    if cfg["TIE_POLICY"] not in ("signal", "drop"):
        raise ValueError("CONFIG['TIE_POLICY'] must be 'signal' or 'drop'.")
    if cfg["AGGREGATION"] not in ("strict", "pooled"):
        raise ValueError("CONFIG['AGGREGATION'] must be 'strict' or 'pooled'.")
    if not _resolve(cfg["INPUT_PATH"]).exists():
        raise FileNotFoundError(f"Input file not found: {_resolve(cfg['INPUT_PATH'])}")


def load_table(path, sheet=0) -> pd.DataFrame:
    path = _resolve(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Lists become strings and characters Excel rejects are removed."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda v: (
                    json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, dict))
                    else v
                )
            )
            if ILLEGAL_CHARACTERS_RE is not None:
                out[col] = out[col].map(
                    lambda v: (
                        ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
                    )
                )
    return out


def write_metrics(path: Path, metrics: dict) -> None:
    comparisons = [
        "pred_vs_human",
        "pred_vs_human_same_rows",
        "pred_vs_adjudicated",
        "judge_rating_vs_human",
    ]
    summary = metrics["judge_summary"]
    scalars = {
        k: (json.dumps(v) if isinstance(v, dict) else v)
        for k, v in summary.items()
        if not isinstance(v, pd.DataFrame)
    }
    with pd.ExcelWriter(path) as xw:
        ReviewRatingJudge.metrics_table(metrics).to_excel(xw, sheet_name="summary")
        pd.Series(scalars, name="value").to_frame().to_excel(
            xw, sheet_name="judge_summary"
        )
        for key in ("choice_by_gap", "choice_by_human_label"):
            if key in summary:
                summary[key].to_excel(xw, sheet_name=key[:31])
        for comp in comparisons:
            m = metrics.get(comp, {})
            if "per_class" in m:
                m["per_class"].to_excel(xw, sheet_name=f"perclass_{comp}"[:31])
                m["confusion_matrix"].to_excel(xw, sheet_name=f"cm_{comp}"[:31])
        ReviewRatingJudge.design_citations().to_excel(
            xw, sheet_name="design_citations", index=False
        )


def print_summary(metrics: dict, outdir: Path) -> None:
    s = metrics["judge_summary"]
    table = ReviewRatingJudge.metrics_table(metrics)
    print("\n" + "=" * 70)
    print(
        f"Disagreements judged: {s['n_judged_disagreements']}   judge errors: {s['n_judge_errors']}"
    )
    print(
        f"Judge favoured prediction: {s['share_pred_favored']:.1%}   human label: "
        f"{s['share_true_favored']:.1%}   tie: {s['share_tie']:.1%}"
    )
    print(
        f"Labels changed: {s['n_labels_changed']}   position consistency: {s['position_consistency_rate']:.1%}"
    )
    rows = [
        r
        for r in [
            "n",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "quadratic_weighted_kappa",
            "macro_mae",
        ]
        if r in table.index
    ]
    print(table.loc[rows].round(3).to_string())
    print(f"\nOutputs written to {outdir.resolve()}")


def run(config: Optional[dict] = None, llm: Optional[Callable] = None) -> dict:
    """Run the full pipeline.

    Parameters
    ----------
    config : dict, optional
        Settings in the same shape as CONFIG; defaults to CONFIG.
    llm : callable, optional
        A ready-made (system, user, temperature) -> text function. If given,
        PROVIDER and JUDGE_MODEL are only recorded, not used to build a client
        (handy for testing with a mock judge).

    Returns
    -------
    dict with "results", "metrics", "controls", "judge" and "outdir".
    Raises if the backend is broken or too many judge calls fail; any
    partial outputs and the raw log are written first.
    """
    cfg = dict(CONFIG if config is None else config)
    validate_config(cfg)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_table(cfg["INPUT_PATH"], cfg["SHEET"])
    if cfg["LIMIT"]:
        df = df.head(int(cfg["LIMIT"]))
    if cfg["FIX_ENCODING"]:
        before = df[cfg["REVIEW_COL"]].copy()
        df[cfg["REVIEW_COL"]] = df[cfg["REVIEW_COL"]].map(repair_mojibake)
        log.info(
            "Repaired encoding in %d reviews",
            int((before != df[cfg["REVIEW_COL"]]).sum()),
        )

    if llm is None:
        if cfg["PROVIDER"] == "anthropic":
            llm = None
        else:
            llm = openai_llm(
                cfg["JUDGE_MODEL"], pass_temperature=cfg["PASS_TEMPERATURE"]
            )

    judge = ReviewRatingJudge(
        llm,
        judge_model_name=cfg["JUDGE_MODEL"],
        predictor_model_name=cfg["PREDICTOR_MODEL"],
        tie_policy=cfg["TIE_POLICY"],
        n_samples=cfg["N_SAMPLES"],
        temperature=cfg["TEMPERATURE"],
        blind_first=cfg["BLIND_FIRST"],
        aggregation=cfg["AGGREGATION"],
        max_retries=cfg["MAX_RETRIES"],
        max_workers=cfg["MAX_WORKERS"],
        audit_rate=cfg["AUDIT_RATE"],
        seed=cfg["SEED"],
        round_fractional_ratings=cfg["ROUND_FRACTIONAL"],
        max_error_rate=cfg["MAX_ERROR_RATE"],
        columns={
            "review": cfg["REVIEW_COL"],
            "true": cfg["TRUE_COL"],
            "pred": cfg["PRED_COL"],
            "id": cfg["ID_COL"],
        },
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = (
        _resolve(cfg["OUTPUT_DIR"])
        if cfg["OUTPUT_DIR"]
        else HERE / "judge_outputs" / f"{Path(cfg['INPUT_PATH']).stem}_{stamp}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(
        json.dumps(
            {**judge.config, "run_config": cfg, "run_started": stamp},
            indent=2,
            default=str,
        )
    )

    n_dis = int((df[cfg["TRUE_COL"]] != df[cfg["PRED_COL"]]).sum())
    log.info(
        "Judging %d rows (%d disagreements) with %s", len(df), n_dis, cfg["JUDGE_MODEL"]
    )

    failure = None
    try:
        results = judge.evaluate(df)
    except RuntimeError as exc:
        failure = exc
        results = judge.results_  # None if the preflight check failed
    finally:
        judge.save_raw_log(outdir / "raw_log.jsonl")
        report = judge.error_report()
        if len(report):
            report.to_csv(outdir / "error_report.csv", index=False)

    if results is None:  # nothing was judged
        log.error("Raw log and error report: %s", outdir)
        raise failure

    excel_safe(results).to_excel(outdir / "results.xlsx", index=False)
    metrics = judge.compute_metrics(results)
    write_metrics(outdir / "metrics.xlsx", metrics)
    judge.export_sanity_sheet(
        str(outdir / "sanity_sheet.csv"), n_per_choice=cfg["SHEET_N"]
    )

    controls = None
    if cfg["CONTROLS"] and failure is None:
        controls = judge.run_controls(df, n=cfg["CONTROLS"])
        controls.to_csv(outdir / "controls.csv", index=False)

    print_summary(metrics, outdir)
    if failure is not None:  # too many errors: outputs saved, still stop loudly
        log.error("Partial outputs saved to %s", outdir)
        raise failure

    return {
        "results": results,
        "metrics": metrics,
        "controls": controls,
        "judge": judge,
        "outdir": outdir,
    }


if __name__ == "__main__":
    warnings.simplefilter("default")
    output = run()
    print("END")

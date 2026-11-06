import os

import pandas as pd
from dotenv import load_dotenv

from src.llm_provider import LLMProvider
from src.review_judge import JudgeConfig, ReviewRatingJudge

load_dotenv()

INPUT_PATH = "./exports/mega_results_200.xlsx"
OUTPUT_FILE = "./exports/judge_gpt6luna_signal"  # .xlsx is added if missing

provider = LLMProvider(
    provider="openai",               # "openai" or "hf"
    model_name="gpt-6-luna",
    token=os.getenv("OPENAI_TOKEN"), # os.getenv("HF_TOKEN") for "hf"
    temperature=1,
)

config = JudgeConfig(
    tie_policy="signal",   # "signal" or "drop"
    aggregation="strict",  # "strict" or "pooled"
    blind_first=True,
    n_samples=1,
    max_workers=4,         # use 1 for a local Hugging Face model
)

if __name__ == "__main__":
    df = pd.read_excel(INPUT_PATH)

    judge = ReviewRatingJudge(provider=provider, config=config)
    results = judge.evaluate(df)
    metrics, counts = judge.summary()

    print(metrics.round(3))
    print(counts)
    print("Saved to", judge.save_excel(OUTPUT_FILE))

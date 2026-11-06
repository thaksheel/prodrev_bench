import os

from dotenv import load_dotenv

from modules.mega_agent_re import Params
from src import CloseModelSelection
from src.runner import EvaluationRunner, MadConfig, MegaConfig, RunnerConfig
from src.llm_provider import LLMProvider
from src.review_judge import JudgeConfig

load_dotenv()

predictor = LLMProvider(
    provider="openai",
    model_name="gpt-6-luna",
    token=os.getenv("OPENAI_TOKEN"),
    temperature=0,
)
judge_provider = LLMProvider(
    provider="openai",
    model_name="gpt-6-luna",
    token=os.getenv("OPENAI_TOKEN"),
    temperature=0,
)

config = RunnerConfig(
    data_path="./data/sr.csv",
    sample_size=5000,
    eval_limit=20,
    seed=42,
    max_workers=4,
    mad=MadConfig(
        model_name="gpt-5.6-luna",
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        num_agents=5,
        num_rounds=3,
    ),
    mega=MegaConfig(
        params=Params(
            api_key=os.getenv("OPENAI_KEY"),
            model=CloseModelSelection.gpt6_luna.value,
            reasoning_effort="none",
            ceo_name="Erza",
            max_memory=6,
            max_rounds=3,
            max_subordinates=2,
            parallel_kickoff=True,
            timeout_s=60,
        ),
    ),
    judge=JudgeConfig(tie_policy="signal", aggregation="strict"),
    output_path="./exports/method_comparison_test20.xlsx",
    slice_path="./exports/eval_slice_5000.csv",
)

if __name__ == "__main__":
    runner = EvaluationRunner(predictor, judge_provider, config)
    comparison = runner.run()
    cols = [c for c in comparison.columns if c.endswith(("_human", "_judge"))]
    print(comparison[["n_valid"] + cols].round(3).T)
    print("Saved to", runner.save_excel())

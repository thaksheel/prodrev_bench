import os
from dataclasses import replace

from dotenv import load_dotenv

from src import (
    CloseModelSelection,
    JudgeConfig,
    LLMProvider,
    MadConfig,
    MegaConfig,
    Runner,
    RunnerConfig,
    MetaConfig,
    PromptMethod,
)
from src.mega import Params

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
llm_model = CloseModelSelection.gpt6_luna.value

TEST_RUN = False

predictor = LLMProvider(
    provider="openai",
    model_name=llm_model,
    token=openai_key,
    temperature=1,
    reasoning_effort="none",
)
judge_provider = LLMProvider(
    provider="openai",
    model_name=llm_model,
    token=openai_key,
    temperature=1,
    reasoning_effort="medium",
)

config = RunnerConfig(
    data_path="./data/sr.csv",
    sample_size=500,
    seeds=tuple(range(42, 42 + 10)),
    max_workers=4,
    prompt_methods={
        "zero_shot": PromptMethod("zero-shot"),
        "zero_shot_cot": PromptMethod("zero-shot", chain_of_thought=True),
        "few_shot_random": PromptMethod("few-shots", "random"),
        "few_shot_random_cot": PromptMethod(
            "few-shots", "random", chain_of_thought=True
        ),
        "few_shot_kate": PromptMethod("few-shots", "kate"),
    },
    mad=MadConfig(
        references_path="./data/references.json",
        num_agents=5,
        num_rounds=3,
        max_workers=4,
    ),
    mega=MegaConfig(
        params=Params(
            ceo_name="Erza",
            max_memory=6,
            max_rounds=3,
            max_subordinates=2,
            parallel_kickoff=True,
            timeout_s=60,
        ),
        plan_path="./exports/w46/mega_plan.txt",
        checkpoint_path="./exports/w46/mega_checkpoint.csv",
        max_workers=4,
    ),
    meta=MetaConfig(
        llm=predictor,
        max_steps=15,
    ),
    judge=JudgeConfig(tie_policy="signal", aggregation="strict"),
    output_path="./exports/w46/results_500_10_w46.xlsx",
    slice_path="./exports/w46/samples_500_10_w46.csv",
)

if TEST_RUN:
    config = replace(
        config,
        sample_size=5,
        seeds=config.seeds[:3],
        output_path="./exports/w46/test_results_20_1.xlsx",
        slice_path=None,
    )

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "mae", "rmse"]

if __name__ == "__main__":
    runner = Runner(predictor, judge_provider, config)
    summary = runner.run()

    cols = ["n_valid_mean", "coverage_mean", "mean_llm_calls_mean"]
    cols += [f"{m}_{label}_mean" for m in METRICS for label in ("human", "judge")]
    print(summary[cols].round(3).T)
    print("Saved to", runner.save_excel())
    print("END")

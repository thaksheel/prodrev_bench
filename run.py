import os
from dotenv import load_dotenv

from modules.mega_agent_re import Params
from src import (
    CloseModelSelection,
    Runner,
    MadConfig,
    MegaConfig,
    RunnerConfig,
    LLMProvider,
    JudgeConfig,
)

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
llm_model = CloseModelSelection.gpt6_luna.value

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
    seeds=[i for i in range(42, 42+10)],
    max_workers=4,
    mad=None,
    mad=MadConfig(
        model_name=llm_model,
        provider="openai",
        api_key=openai_key,
        num_agents=5,
        num_rounds=3,
        references_path="./data/references.json"
    ),
    mega=MegaConfig(
        params=Params(
            api_key=openai_key,
            model=llm_model,
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
    output_path="./exports/w45/results_500_10_w46.xlsx", #results_samples_seeds
)

if __name__ == "__main__":
    runner = Runner(predictor, judge_provider, config)
    comparison = runner.run()
    cols = [c for c in comparison.columns if c.endswith(("_human", "_judge"))]
    print(comparison[["n_valid"] + cols].round(3).T)
    print("Saved to", runner.save_excel())
    print("END")

# Checkpoint: mad_predict has error for json decoding in ./made_predict_re/agent.py ln 117
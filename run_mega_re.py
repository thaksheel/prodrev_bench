import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from modules.mega_agent_re import MegaPrompt, Params, Runner
from src import CloseModelSelection

load_dotenv()

SIZE = 200 
MAX_WORKERS = 4  
PLAN_PATH = "./exports/mega_plan.txt"  
LEADER = "Erza"

df = pd.read_csv("./data/sr.csv").dropna(subset=["review"])
_, sample = train_test_split(df, test_size=SIZE, random_state=44, stratify=df.rating)
reviews = sample.review.tolist()
groundtruth = sample.rating.to_numpy()

params = Params(
    api_key=os.getenv("OPENAI_KEY"),
    model=CloseModelSelection.gpt6_luna.value,
    reasoning_effort="none",
    ceo_name=LEADER,
    max_memory=6,
    max_rounds=3,
    max_subordinates=2, 
    parallel_kickoff=True, 
    timeout_s=60,
)

task_ins = (
    "predict the overall star rating (1 to 5) a customer gave a product, using "
    "only the text of their online review. Do not use any label or information "
    "outside the review text as evidence. The final answer is an integer rating "
    "with a brief evidence-based rationale."
)
mp = MegaPrompt()
task_instruction = mp.get_mega_instructions(
    LEADER, task_ins, max_agents=params.max_subordinates + 1
)

runner = Runner(log_path="./exports/logs.out", run_dir="./exports/mega_runs")
if Path(PLAN_PATH).exists():
    runner.load_plan(PLAN_PATH)
else:
    runner.plan(params, task_instruction)
    runner.save_plan(PLAN_PATH)
print("Team:", [name for name, _ in runner.parse_plan(runner.plan_text, params)])

df_results = runner.run_many(
    params,
    task_instruction,
    reviews,
    trues=groundtruth,
    prompt_template="Score this customer online product review from 1-5: <review> {review} </review>",
    max_workers=MAX_WORKERS,
    checkpoint_path=f"./exports/mega_results_{SIZE}_checkpoint.csv",
)
df_results.to_csv(f"./exports/mega_results_{SIZE}.csv", index=False)
df_results.to_excel(f"./exports/mega_results_{SIZE}.xlsx", index=False)

ok = df_results.dropna(subset=["pred"])
print(f"\nvalid predictions: {len(ok)}/{len(df_results)}")
if len(ok):
    print(
        f"accuracy={(ok.pred == ok.true).mean():.3f}  MAE={(ok.pred - ok.true).abs().mean():.3f}"
    )
print(
    f"duration: mean={df_results.duration.mean():.1f}s  median={df_results.duration.median():.1f}s  "
    f"max={df_results.duration.max():.1f}s"
)
print(
    f"tokens per sample: in={df_results.total_input_token.mean():.0f}  out={df_results.total_output_token.mean():.0f}"
)
print(
    f"total tokens: in={df_results.total_input_token.sum()}  out={df_results.total_output_token.sum()}  "
    f"(+ planning in={runner.plan_input_token} out={runner.plan_output_token})"
)
if df_results.error.notna().any():
    print("errors:\n", df_results.error.value_counts())
print("END")

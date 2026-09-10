import numpy as np
import pandas as pd
import json

filename = "./data/sr.csv"
df = pd.read_csv(filename)
df.dropna(subset=['review'])
stars = [i + 1 for i in range(5)]
limit = 10
stars_examples = dict(zip(stars, [[] for _ in stars]))
for s in stars:
    stars_examples[s] = (
        df[df.rating == s].review.sample(n=limit, replace=False).tolist()
    )

references = {
    "num_classes": len(stars),
    "cls_lvl": stars,
    "instructions": [
        "You are a debater in a multi-agent sentiment classification debate. The topic is: identify the correct star rating (1-5) for a given product review. You argue that the correct rating is **1 star**. Your role is to defend the position that the review expresses ##references##",
        "You are a debater in a multi-agent sentiment classification debate. The topic is: identify the correct star rating (1-5) for a given product review. You argue that the correct rating is **2 star**. Your role is to defend the position that the review expresses ##references##",
        "You are a debater in a multi-agent sentiment classification debate. The topic is: identify the correct star rating (1-5) for a given product review. You argue that the correct rating is **3 star**. Your role is to defend the position that the review expresses ##references##",
        "You are a debater in a multi-agent sentiment classification debate. The topic is: identify the correct star rating (1-5) for a given product review. You argue that the correct rating is **4 star**. Your role is to defend the position that the review expresses ##references##",
        "You are a debater in a multi-agent sentiment classification debate. The topic is: identify the correct star rating (1-5) for a given product review. You argue that the correct rating is **5 star**. Your role is to defend the position that the review expresses ##references##",
    ],
    "references": [
        "Reviews in this category express strong dissatisfaction, often describing products as unusable, harmful, or a complete waste. Users report severe issues such as “it didn’t lay smoothly and was splotchy,” “this stuff burned my eyes,” and “this product dried out my skin!!!” The tone is consistently frustrated, with buyers feeling regret, discomfort, or even physical irritation. A 1‑star reference should emphasize failure of core functionality, harsh reactions, and clear disappointment, leaving the user unwilling to continue using or repurchase the product.",
        "Two star reviews show clear dissatisfaction, but with some redeeming qualities or partial functionality. Users often mention problems like greasiness, mismatched shades, smudging, or mild irritation — “starts to smudge before my work day is done,” “felt so greasy,” “coverage was splotchy,” “made me look really oily.” These reviews acknowledge that the product works in limited ways (e.g., color payoff, smoothing, evening tone), but overall performance is disappointing. A 2‑star reference should highlight noticeable flaws, inconsistent results, and limited usefulness, while stopping short of total failure.",
        "Three star reviews are balanced, describing products that work adequately but fall short of expectations. Users often say the product is fine but not impressive — “It’s good at what it does, cleanses,” “helps slightly,” “kind of thick but doesn’t feel heavy,” “I noticed a little improvement.” Sentiment is neither strongly positive nor strongly negative. There are mild benefits (softening, moisturizing, slight brightening) paired with drawbacks (heaviness, breakouts, dryness, strange scent). A 3‑star reference should reflect moderate satisfaction, minor improvements, and clear limitations, resulting in a neutral or lukewarm overall impression.",
        "Four star reviews express strong satisfaction, with users praising effectiveness, texture, shade matching, or hydration — “great facial wash,” “very moisturizing,” “love the buildable coverage,” “super hydrating formula.” However, they also mention small drawbacks such as heaviness, oiliness, or short wear time — “doesn’t last long,” “causes my t‑zone to become super oily.” A 4‑star reference should emphasize overall positive performance, usefulness, and pleasing results, while acknowledging minor flaws that prevent a perfect rating.",
        "Five star reviews show enthusiastic satisfaction, often describing products as reliable, transformative, or essential. Users highlight strong performance — “very good coverage,” “would highly recommend,” “best lip balm ever,” “this is the ONLY product that actually helped!” These reviews frequently mention long-term loyalty, repurchase intent, and standout benefits like hydration, longevity, or visible improvement. A 5‑star reference should capture clear delight, strong endorsement, and consistently excellent results, with no meaningful complaints.",
    ],
    "rating": stars,
    "examples": stars_examples,
    "prompt_template": [
        "Product Review: ##text##.You believe the review is a 1-star review. Reference: '##reference##'. Based on the Reference, give a reason why you think the review deserves 1 stars. If you don't have a reference, give a reason based on the text itself. Please do so in one sentence.",
        "Product Review: ##text##.You believe the review is a 2-star review. Reference: '##reference##'. Based on the Reference, give a reason why you think the review deserves 2 stars. If you don't have a reference, give a reason based on the text itself. Please do so in one sentence.",
        "Product Review: ##text##.You believe the review is a 3-star review. Reference: '##reference##'. Based on the Reference, give a reason why you think the review deserves 3 stars. If you don't have a reference, give a reason based on the text itself. Please do so in one sentence.",
        "Product Review: ##text##.You believe the review is a 4-star review. Reference: '##reference##'. Based on the Reference, give a reason why you think the review deserves 4 stars. If you don't have a reference, give a reason based on the text itself. Please do so in one sentence.",
        "Product Review: ##text##.You believe the review is a 5-star review. Reference: '##reference##'. Based on the Reference, give a reason why you think the review deserves 5 stars. If you don't have a reference, give a reason based on the text itself. Please do so in one sentence.",
    ],
    "rebuddle_prompt_template": [
        "The opposing agent's argument is: ##rebuddle##. Based on your position, agree or rebut the opposing argument and explain your reasoning in one sentence.",
        "The opposing agent's argument is: ##rebuddle##. Based on your position, agree or rebut the opposing argument and explain your reasoning in one sentence.",
        "The opposing agent's argument is: ##rebuddle##. Based on your position, agree or rebut the opposing argument and explain your reasoning in one sentence.",
        "The opposing agent's argument is: ##rebuddle##. Based on your position, agree or rebut the opposing argument and explain your reasoning in one sentence.",
        "The opposing agent's argument is: ##rebuddle##. Based on your position, agree or rebut the opposing argument and explain your reasoning in one sentence.",
    ],
}

outfile = "./exports/references.json"
with open(outfile, "w") as f:
    json.dump(references, f)

print(stars_examples)
print(references)
print("END")

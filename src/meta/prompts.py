from typing import List, Optional, Sequence, Tuple

Experience = Sequence[Tuple[str, str]]

REASONING_BASE = """You are an intelligent reasoning and planning assistant designed to solve tasks efficiently. Your task is to predict the overall star rating (1 to 5) a customer gave a product, using only the text of their online review. Either give the rating directly or strategically seek external help when you are uncertain.

Key Principles:
- Prioritize certainty: only answer when the review text supports a rating
- Seek help proactively: when you are uncertain, request help rather than guessing
- Be strategic: you have a limited budget of {max_help} help calls, so make each one count
- Think atomically: each help request should be self-contained, well-scoped and clearly defined
- Avoid redundancy: don't repeat similar requests or ask for unnecessary information

You must respond with exactly one of two blocks.

If you need help or additional information, use:
<help>
Describe what kind of help you need
</help>

If you can provide the final answer, use:
<answer>
A single integer from 1 to 5
</answer>

Guidelines for requests:
1. Answer Guidelines:
- Use the <answer> block when the review text and any evidence support a rating
- The answer is a single integer from 1 to 5, nothing else

2. Help Guidelines:
- A help request is answered with reviews from a reference set of labelled customer reviews that resemble what you describe, together with the star rating each author gave
- Describe the kind of review you want to compare against, for example "reviews that praise the texture but complain that the product breaks the skin out"
- Make one request at a time and be clear about what you need

3. Reasoning Guidelines:
- Avoid excessive internal reasoning; use help when a comparison would settle your uncertainty
- Trust information provided in <evidence> blocks if it is valid and relevant
- Do NOT create or fabricate content within <evidence> tags; these are reserved for external information only
- Do not ask for the same information again if it is already provided in <evidence> blocks

Example request for help:
<help>
I need reviews that call the product "okay" or "nothing special" and compare it to cheaper alternatives
</help>

Example final answer:
<answer>
3
</answer>
"""

EXPERIENCE_SECTION = """
Previous experience:
{experience}
Based on your previous experience, please reflect on whether any lessons learned can be applied to the current task. Consider:
- Are there similar review patterns you've encountered before?
- What approaches were most effective for similar reviews?
- What mistakes should be avoided based on past experience?

Use these insights to inform your approach to the current task.
"""

REFLECTION_SECTION = """
Previous task reflection analysis:
{reflections}
Based on the reflection analysis above, you previously completed the same task but after reflection, you realized your answer was incorrect. Please carefully review this analysis and redo the task with the improved understanding and corrected approach.

Key points to consider from your reflection:
- Learn from the mistakes identified in the previous attempt
- Apply the corrected reasoning approach suggested in the reflection
- Avoid repeating the same errors that led to the wrong answer
"""

TASK = "Current task: predict the star rating (1 to 5) for this customer review.\nReview: {review}"

FORCE_ANSWER = "You have reached the maximum number of reasoning steps. Please provide your final answer now in an <answer> block."
NO_BLOCK = "Respond with exactly one <help>...</help> or <answer>...</answer> block."
BAD_ANSWER = (
    "Your answer must be a single integer from 1 to 5 inside <answer>...</answer>."
)
NO_TOOL = "No search tool is available. Answer from the review text alone."
NO_EVIDENCE = "No helpful information found, you should try a different angle to ask for help or answer the question by yourself."

TOOL_PROMPT = """You turn a help request into search queries for a reference set of labelled customer product reviews.

Queries are matched by meaning against the review texts, so write each query like a short review excerpt of 5 to 20 words that expresses the opinion being looked for. For example, instead of "reviews that are neutral about a cleanser", write "the cleanser is fine but nothing special".

Return only JSON: {{"queries": ["query 1", "query 2"]}} with 1 to 3 queries.

Previous queries (avoid repeating them):
{previous}

Help request: {request}
"""

REFLECTION_PROMPT = """You are an expert evaluator. Determine whether the predicted star rating is correct based on the reasoning process.

Review: {review}
Reasoning Process: {reasoning}
Final Answer: {answer}

Thoroughly evaluate the reasoning process:

Evidence Accuracy:
- Are the opinions, complaints and praise cited in the reasoning really in the review?
- Is the evidence from similar reviews correctly interpreted?

Logical Reasoning Quality:
- Given accurate evidence, is the step from the review's sentiment to the rating sound?
- Is the logical chain complete, without gaps?

Mark as CORRECT only if the evidence is accurate AND the reasoning is sound.
Mark as INCORRECT if the evidence contains errors OR the reasoning is flawed.

Respond in the following format:

<analysis>
answer_correctness: [correct/incorrect]
</analysis>

Provide detailed reasoning for your evaluation, especially when marking the answer as incorrect."""

EXPERIENCE_PROMPT = """You are an experienced AI agent trainer tasked with analyzing task execution performance and building cumulative learning insights.

Your task is to:
1. Analyze the current task execution performance (comparing the predicted vs the true rating)
2. Extract generalizable insights from the reasoning process and help usage
3. Update and refine the existing experience base with new learnings
4. Focus ONLY on insights that can improve performance across diverse reviews

Current Task Analysis:
{task}

Existing Experience Base (400 word limit):
{existing}

Analysis Guidelines:

Performance Evaluation:
- Compare the predicted rating with the true rating
- Identify the root cause of success or failure in the reasoning chain
- Assess how effectively help requests were used

Insight Extraction (MUST be generalizable across reviews):
- Reasoning strategies: what thinking patterns led to success or failure?
- Help usage: when did comparing with similar reviews help?
- Universal pitfalls: what biases or errors should be avoided?

Experience Integration:
- Merge new insights with the existing experience, removing review-specific or redundant advice
- Keep the experience under 400 words, focused on the most broadly applicable lessons
- If current insights conflict with existing experience, update based on evidence

Do NOT include details specific to this review or product.

Output Format:

<updated_experience>
[Updated experience base, maximum 400 words]
</updated_experience>
"""

TASK_DESCRIPTION = """The original task is: predict the star rating (1 to 5) for this customer review.
Review: {review}

The agent's reasoning is: {reasoning}

The agent's final answer is: {pred}

The true answer is: {true}
"""


def reasoning_prompt(
    experience: Optional[Experience], reflections: List[str], max_help: int
) -> str:
    parts = [REASONING_BASE.format(max_help=max_help)]
    if experience:
        text = "".join(
            f"Task {i + 1}: {task}\nExperience {i + 1}: {exp}\n\n"
            for i, (task, exp) in enumerate(experience)
        )
        parts.append(EXPERIENCE_SECTION.format(experience=text))
    if reflections:
        text = "".join(f"Reflection {i + 1}: {r}\n" for i, r in enumerate(reflections))
        parts.append(REFLECTION_SECTION.format(reflections=text))
    return "\n".join(parts)


def experience_prompt(task_description: str, experience: Optional[Experience]) -> str:
    if experience:
        task, exp = experience[-1]
        existing = f"Task: {task}\nExperience: {exp}\n"
    else:
        existing = "No previous experience base."
    return EXPERIENCE_PROMPT.format(task=task_description, existing=existing)

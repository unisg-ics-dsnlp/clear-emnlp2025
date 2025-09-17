import random


def roulette(
        prompts_w_scores: list[tuple[str, float]],
        select_n: int
) -> list[str]:
    total_score = sum([score for _, score in prompts_w_scores])
    probs = [score / total_score for _, score in prompts_w_scores]
    selected_prompts = random.choices(prompts_w_scores, weights=probs, k=select_n)
    return [prompt for prompt, _ in selected_prompts]

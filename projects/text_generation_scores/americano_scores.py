import re
import numpy as np

from util.template import PromptTemplate

PROMPT_COHERENCE = '''You are a lecturer of the writing class. You are given the following proposition on a controversial topic. You need to carefully read the proposition and evaluate it based on the criteria:
- Clarity
- Relevance
- Logical consistency
- Validity of reasoning
Now you need to assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria. Note, you should be very strict when giving the score.'''

PROMPT_PERSUASION = '''You are a lecturer of the writing class. You are given the following proposition on a controversial topic. You need to carefully read the proposition and evaluate it based on the criteria:
- Language and rhetoric
- Addressing opposing viewpoints
- Credibility
- Overall effectiveness
Now you need to assign a score for persuasion on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria. Note, you should be very strict when giving the score.'''


def extract_first_number(text):
    match = re.search(r'\d+[.,]?\d*', text)
    if match:
        number_str = match.group().replace(',', '.')
        return float(number_str)
    return None


def calculate_americano_scores(
        template: PromptTemplate,
        num_runs: int,
        texts: list[str]
):
    formatted_prompts_coherence = []
    formatted_prompts_persuasion = []
    for t in texts:
        coherence = template.format_prompt(PROMPT_COHERENCE, t, '', [], [])
        persuasion = template.format_prompt(PROMPT_PERSUASION, t, '', [], [])
        for i in range(num_runs):
            formatted_prompts_coherence.append(coherence)
            formatted_prompts_persuasion.append(persuasion)
    results_coherence = template.do_prompting(formatted_prompts_coherence)
    results_persuasion = template.do_prompting(formatted_prompts_persuasion)

    scores_coherence = [extract_first_number(t) for t in results_coherence]
    scores_persuasion = [extract_first_number(t) for t in results_persuasion]

    avgs_coherence = []
    for i in range(0, len(scores_coherence), num_runs):
        batch = scores_coherence[i:i + num_runs]
        batch_scores = [s for s in batch if s is not None]
        average_score = sum(batch_scores) / len(batch_scores) if len(batch_scores) > 0 else np.nan
        avgs_coherence.append(average_score)
    avgs_persuasion = []
    for i in range(0, len(scores_persuasion), num_runs):
        batch = scores_persuasion[i:i + num_runs]
        batch_scores = [s for s in batch if s is not None]
        average_score = sum(batch_scores) / len(batch_scores) if len(batch_scores) > 0 else np.nan
        avgs_persuasion.append(average_score)

    return formatted_prompts_coherence, formatted_prompts_persuasion, results_coherence, results_persuasion, scores_coherence, scores_persuasion, avgs_coherence, avgs_persuasion

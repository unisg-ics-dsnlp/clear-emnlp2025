from typing import Optional

from tqdm import tqdm

from projects.evoprompt.evolution_template import EvolutionPromptingTemplate
from projects.evoprompt.selection import roulette

from .template_de import v3


def get_evolve_prompt(
        prompts: list[str],
        strategy: str
) -> str:
    template = v3[strategy]
    filled_template = template.replace(
        '<prompt1>', prompts[0]).replace('<prompt2>', prompts[1]).replace('<prompt3>', prompts[2])
    return filled_template


class DifferentialEvolutionPrompting(EvolutionPromptingTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_prompting(
            self,
            prompt: str,
            demonstrations: Optional[list[str]] = None,
            *args,
            **kwargs
    ) -> str:
        for _ in tqdm(range(self.budget), desc='DE prompting'):
            new_candidates = self._evolve()
            scores = self.scoring_function([candidates for candidates, _original, _original_score in new_candidates],
                                           self.dev_set)
            for new_score, (new_candidate, original_prompt, original_score) in zip(scores, new_candidates):
                if new_score > original_score:
                    # remove original prompt from populatio
                    self.population = [(p, s) for p, s in self.population if p != original_prompt]
                    self.population.append((new_candidate, new_score))
            # TODO: check if this works, this call should do literally nothing at this point
            # self._evaluate_population()

            self._save_population()
        return self.population[0][0]

    def _evolve(self) -> list[tuple[str, str, float]]:
        formatted_prompts = []
        for i in range(self.n):
            # every prompt in population is tried as a base
            for prompt, score in self.population:
                if self.evolution_selection_strategy == 'roulette':
                    # from the remaining ones, do selection
                    remaining_prompts = [(p, s) for p, s in self.population if p != prompt]
                    selected_prompts = roulette(remaining_prompts, 2)
                    selected_and_best = selected_prompts + [prompt]
                    evolve_prompt = get_evolve_prompt(selected_and_best, self.evolution_perform_strategy)
                    # TODO: adjust for other models
                    formatted_llama3 = self.format_prompt('', evolve_prompt)
                    formatted_prompts.append((formatted_llama3, prompt, score))
        result = self.send_prompt_to_model([x[0] for x in formatted_prompts])
        new_candidate_prompts = []
        for cand, (_, original_prompt, original_score) in zip(result, formatted_prompts):
            # get everything between <prompt> and </prompt>
            try:
                cand_text = cand.split('<prompt>')[-1].split('</prompt>')[0]
            except IndexError:
                continue  # output not nicely formatted, skip
            new_candidate_prompts.append((cand_text, original_prompt, original_score))
        return new_candidate_prompts

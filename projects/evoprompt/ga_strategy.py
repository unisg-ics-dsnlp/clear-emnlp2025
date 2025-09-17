from typing import Optional

from tqdm import tqdm

from projects.evoprompt.evolution_template import EvolutionPromptingTemplate
from projects.evoprompt.selection import roulette
from projects.evoprompt.template_ga import templates_2


def get_evolve_prompt(
        prompts: list[str],
        strategy: str
) -> str:
    template = templates_2[strategy]
    filled_template = template.replace('<prompt1>', prompts[0]).replace('<prompt2>', prompts[1])
    return filled_template


class GeneticAlgorithmPrompting(EvolutionPromptingTemplate):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def do_prompting(
            self,
            prompt: str,
            demonstrations: Optional[list[str]] = None,
            *args,
            **kwargs
    ) -> str:
        for _ in tqdm(range(self.budget), desc='GA prompting'):
            new_candidates = self._evolve()
            self.population.extend([(prompt, None) for prompt in new_candidates])
            self._evaluate_population()
            self._save_population()
        return self.population[0][0]

    def _evolve(self) -> list[str]:
        formatted_prompts = []
        for i in range(self.n):
            if self.evolution_selection_strategy == 'roulette':
                selected_prompts = roulette(self.population, 2)
                evolve_prompt = get_evolve_prompt(selected_prompts, self.evolution_perform_strategy)
                # TODO: adjust for other models
                formatted_llama3 = self.format_prompt('', evolve_prompt)
                formatted_prompts.append(formatted_llama3)
        result = self.send_prompt_to_model(formatted_prompts)
        new_candidate_prompts = []
        for cand in result:
            # get everything between <prompt> and </prompt>
            try:
                cand_text = cand.split('<prompt>')[-1].split('</prompt>')[0]
            except IndexError:
                continue  # output not nicely formatted, skip
            new_candidate_prompts.append(cand_text)
        return new_candidate_prompts

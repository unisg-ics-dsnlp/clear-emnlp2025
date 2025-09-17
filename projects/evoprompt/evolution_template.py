import abc
import os
from typing import Optional

import pandas as pd

from util.template import PromptTemplate


class EvolutionPromptingTemplate(PromptTemplate, abc.ABC):
    def __init__(
            self,
            model: str,
            initial_population: list[tuple[str, Optional[float]]],
            dev_set: list[tuple[str, str]],
            evolution_selection_strategy: str,
            evolution_perform_strategy: str,
            checkpoint_path: str,
            task: str,
            scoring_function: callable,
            budget: int,
            n: int,
            population_size: int,
            temperature: float = 0.7,
            out_file_name: Optional[str] = None,
            use_already_evaluated_for_initial: bool = False,
            always_evaluate_initial: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(model, temperature=temperature, *args, **kwargs)
        self.evolution_selection_strategy = evolution_selection_strategy
        self.evolution_perform_strategy = evolution_perform_strategy
        self.n = n
        self.population: list[tuple[str, float]] = initial_population
        self.task = task
        self.checkpoint_path = checkpoint_path
        if out_file_name is None:
            self.out_file_name = f'{self.task}_scored_prompts.csv'
        else:
            self.out_file_name = out_file_name
        if use_already_evaluated_for_initial:
            existing_scores = self._load_evaluated_prompts()
            # update population, use scores if available
            self.population = existing_scores + [(prompt, score) for prompt, score in self.population if
                                                 prompt not in [prompt for prompt, _ in existing_scores]]
        self.dev_set = dev_set
        self.scoring_function = scoring_function
        self.budget = budget
        self.current_budget = 0
        self.population_size = population_size
        self.always_evaluate_initial = always_evaluate_initial
        self._evaluate_population(initial_evaluation=True, *args, **kwargs)
        self._save_population()

    @abc.abstractmethod
    def _evolve(self) -> list[str]:
        pass

    def _evaluate_population(self, initial_evaluation: bool = False, *args, **kwargs):
        """
        Evaluate the current population, and then keep only the top population_size.
        """
        scored_population = [(prompt, score) for prompt, score in self.population if score is not None]
        if len(scored_population) >= self.population_size and (initial_evaluation and not self.always_evaluate_initial):
            self.population = sorted(scored_population, key=lambda x: x[1], reverse=True)[:self.population_size]
            return
        unevaluated = [prompt for prompt, score in self.population if score is None]
        if not unevaluated:
            return
        scores = self.scoring_function(unevaluated, self.dev_set, *args, **kwargs)
        # append to population
        self.population = [(prompt, score) for prompt, score in self.population if score is not None]
        self.population.extend([(prompt, score) for prompt, score in zip(unevaluated, scores)])
        # order by score, descending
        self.population = sorted(self.population, key=lambda x: x[1], reverse=True)
        # keep only the top population_size
        self.population = self.population[:self.population_size]

    def _save_population(self):
        """
        Save the populations to a .csv file. Appends only new ones.
        """
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if os.path.exists(os.path.join(self.checkpoint_path, self.out_file_name)):
            existing_scores = self._load_evaluated_prompts()
        else:
            existing_scores = []
        new_entries = [(prompt, score) for prompt, score in self.population if
                       prompt not in [prompt for prompt, _ in existing_scores]]
        if not new_entries:
            return
        df = pd.DataFrame(new_entries, columns=['prompt', 'score'])
        if existing_scores:
            existing_df = pd.DataFrame(existing_scores, columns=['prompt', 'score'])
            df = pd.concat([df, existing_df])
        # order by score
        df = df.sort_values(by='score', ascending=False)
        df.to_csv(os.path.join(self.checkpoint_path, self.out_file_name), index=False)

    def _load_evaluated_prompts(self) -> list[tuple[str, float]]:
        """
        Load the evaluated prompts from the checkpoint file.
        :return: list of tuples of prompt and score.
        """
        if not os.path.exists(os.path.join(self.checkpoint_path, self.out_file_name)):
            return []
        existing_scores = pd.read_csv(os.path.join(self.checkpoint_path, self.out_file_name))
        existing_scores = [(prompt, score) for prompt, score in
                           zip(existing_scores['prompt'], existing_scores['score'])]
        return existing_scores

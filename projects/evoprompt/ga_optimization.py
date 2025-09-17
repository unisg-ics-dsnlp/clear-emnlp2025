from typing import Optional

from projects.evoprompt.ga_strategy import GeneticAlgorithmPrompting


def optimize_prompt_ga(
        dev_set: list[tuple[str, str]],
        initial_prompts: list[str],
        checkpoint_path: str,
        task_name: str,
        scoring_function: callable,
        evoluation_selection_strategy: str,
        evolution_perform_strategy: str,
        budget: int,
        n: int,
        population_size: int,
        out_file_name: Optional[str] = None,
        *args,
        **kwargs
):
    initial_population = [(prompt, None) for prompt in initial_prompts if prompt]
    ga = GeneticAlgorithmPrompting(
        model='llama3',
        initial_population=initial_population,
        dev_set=dev_set,
        evolution_selection_strategy=evoluation_selection_strategy,
        evolution_perform_strategy=evolution_perform_strategy,
        checkpoint_path=checkpoint_path,
        task=task_name,
        scoring_function=scoring_function,
        budget=budget,
        n=n,
        population_size=population_size,
        use_already_evaluated_for_initial=True,
        always_evaluate_initial=False,
        out_file_name=out_file_name,
        *args,
        **kwargs
    )
    optimized_prompt = ga.do_prompting('')
    return optimized_prompt

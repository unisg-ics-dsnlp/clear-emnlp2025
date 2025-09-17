import os
import re
from functools import partial
from typing import Optional

import pandas as pd
from tqdm import tqdm
from vllm import LLM

from bsm.bsm import BSMFindBranches
from direct.direct import DirectPrompt
from little_brother.little_brother import LittleBrotherPrompting
from projects.evoprompt.de_strategy import DifferentialEvolutionPrompting
from projects.evoprompt.ga_strategy import GeneticAlgorithmPrompting
from projects.improvement_pipeline.arg_rewrite_stuff.arg_revisions import get_demonstrations
from self_discover.self_discover import SelfDiscover
from util.template import phi_format_func, llama3_format_func


def load_microtext_df() -> pd.DataFrame:
    """
    Load microtexts from csv file.
    :return: A pandas DataFrame with columns ['topic_id', 'graph_id', 'stance', 'argument'].
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    microtexts_path = os.path.join(current_file_dir, 'microtexts.csv')
    df = pd.read_csv(microtexts_path)
    # filter out those where topic_id is nan
    df = df[df['topic_id'].notna()]
    return df


def bsm_formatter(
        topic: str,
        prompt: str
):
    return prompt.replace('>topic<', topic)


def prepare_microtext_prompts(
        argument_df: pd.DataFrame,
        demonstrations: Optional[list[str]] = None
) -> list[tuple[str, str, str]]:
    if demonstrations is None:
        demonstrations = []
    unformatted_prompts = []
    for i, row in tqdm(argument_df.iterrows(), total=len(argument_df)):
        topic = row['topic_id']
        stance = row['stance']
        argument = row['argument']
        system_prompt = f"You are given an argument about the topic \"{topic}\". Your task is to improve it. Respond only with the improved argument wrapped in @ symbols and nothing else. Here are some examples of improvements:\n\n"
        system_prompt += ''.join(demonstrations)
        user_prompt = f'Original: {argument}'
        assistant_prompt = f'Improved: @'
        unformatted_prompts.append((system_prompt, user_prompt, assistant_prompt))
    return unformatted_prompts


def improve_microtexts_direct(
        argument_df: pd.DataFrame,
        model: str,
        num_demonstrations: int = 3,
        format_func: Optional[callable] = None,
        use_vllm: bool = False,
        vllm_model_instance: Optional[LLM] = None,
        *args,
        **kwargs
) -> list[str]:
    demonstration_args = get_demonstrations(num_demonstrations)
    demonstrations = []
    for old, new in demonstration_args:
        demonstrations.append(f'Original: {old.strip()}\nImproved: @{new.strip()}@\n\n')

    unformatted_prompts = prepare_microtext_prompts(argument_df, demonstrations)
    direct = DirectPrompt(
        model,
        *args,
        format_func=format_func,
        use_vllm=use_vllm,
        vllm_model_instance=vllm_model_instance,
        **kwargs
    )
    formatted_model_prompts = [direct.format_prompt(system_prompt, user_prompt, assistant_prompt) for
                               system_prompt, user_prompt, assistant_prompt in unformatted_prompts]
    outputs = direct.do_prompting(formatted_model_prompts)

    improved_out = []
    for (system_prompt, user_prompt, assistant_prompt), formatted_model_prompt, output in zip(unformatted_prompts,
                                                                                              formatted_model_prompts,
                                                                                              outputs):
        full_output = assistant_prompt + output
        if (stop := kwargs.get('stop')) is not None:
            full_output += stop
        pattern = r'@(.*?)@'
        substrings = re.findall(pattern, full_output)
        try:
            improved_argument = max(substrings, key=len)
        except ValueError:
            improved_argument = full_output[len(assistant_prompt):]
        print(f'Improved argument: {improved_argument}')
        improved_out.append(improved_argument)
    return improved_out


def improve_microtexts_bsm(
        argument_df: pd.DataFrame,
        model: str,
        use_vllm: bool,
        vllm_model_instance: Optional[LLM] = None
):
    bsm_branch_prompt = f'''You are given an argument about the topic >topic<. Your task is to improve it. In order to do so, your task is to first propose certain aspects of the argument that can be improved, and then divide the aspects into two groups such that the argument can be improved individually for all aspects in the groups. Your output should be in the format:
Group 1: <aspects here>
Group 2: <aspects here>'''
    bsm_solve_prompt = f'''Improve the following argument by focussing on the specific aspects. Respond with the improved argument wrapped in @ symbols. Try to keep the length of the improved argument similar to the original one.
Argument: >task<
Aspects: >group<'''
    bsm_merge_prompt = f'''Given two arguments about the topic >topic<, your task is to merge them into a single argument. Respond with the merged argument wrapped in @ symbols.'''

    tasks = []
    branch_formatters = []
    solve_formatters = []
    merge_formatters = []
    for i, row in tqdm(argument_df.iterrows(), total=len(argument_df)):
        topic = row['topic_id']
        stance = row['stance']
        argument = row['argument']
        branch_formatters.append(partial(bsm_formatter, topic))
        solve_formatters.append(lambda x: x)
        merge_formatters.append(partial(bsm_formatter, topic))
        tasks.append(argument)

    bsm = BSMFindBranches(
        model=model,
        branch_prompt=bsm_branch_prompt,
        solve_prompt=bsm_solve_prompt,
        merge_prompt=bsm_merge_prompt,
        num_branches=2,
        branch_formatters=branch_formatters,
        solve_formatters=solve_formatters,
        merge_formatters=merge_formatters,
        use_vllm=use_vllm,
        vllm_model_instance=vllm_model_instance
    )
    out = bsm.do_prompting(tasks)
    return out


def improve_self_discover(
        argument_df: pd.DataFrame,
        model: str,
        num_demonstrations: int,
        reasoning_example: str,
        reasoning_example_plan: str,
        use_vllm: bool,
        vllm_model_instance: Optional[LLM] = None
) -> tuple[list[str], list[str]]:
    """
    Improve microtexts using SelfDiscover algorithm.
    :param argument_df: The DataFrame containing the microtexts.
    :param model: The model to use.
    :param num_demonstrations: The number of demonstrations to use.
    :param reasoning_example: An example task.
    :param reasoning_example_plan: A plan for how to solve the example task.
    :return:
    """
    demonstration_args = get_demonstrations(num_demonstrations)
    demonstrations = []
    for old, new in demonstration_args:
        demonstrations.append(f'Original: {old.strip()}\nImproved: @{new.strip()}@\n\n')
    unformatted_prompts = prepare_microtext_prompts(argument_df, demonstrations)
    selfdiscover = SelfDiscover(model, use_vllm=use_vllm, vllm_model_instance=vllm_model_instance)
    formatted_model_prompts = [selfdiscover.format_prompt(system_prompt, user_prompt, assistant_prompt) for
                               system_prompt, user_prompt, assistant_prompt in unformatted_prompts]
    outputs, _prompts = selfdiscover.do_prompting(
        formatted_model_prompts,
        demonstrations,
        reasoning_example=reasoning_example,
        reasoning_example_plan=reasoning_example_plan
    )
    improved_arguments = []
    original_arguments = []
    for (system_prompt, user_prompt, assistant_prompt), formatted_model_prompt, output in zip(unformatted_prompts,
                                                                                              formatted_model_prompts,
                                                                                              outputs):
        full_output = assistant_prompt + output
        try:
            tmp = full_output.split('@')
            # take longest as output
            improved_argument = max(tmp, key=len)
        except IndexError:
            improved_argument = ''
        print(f'Improved argument: {improved_argument}')
        improved_arguments.append(improved_argument)
        original_arguments.append(user_prompt)
    return improved_arguments, original_arguments


def preference_based_scoring(
        prompts: list[str],
        dev_set: list[tuple[str, str]]
):
    direct = DirectPrompt('llama3')
    system_prompt_compare = '''You are given two arguments. Your task is to choose the better one. Respond with @First@ if you prefer the first one, and with @Second@ if you prefer the second one.'''
    user_prompt_compare_template = '''Argument 1: {argument1}
Argument 2: {argument2}'''
    assistant_prompt_compare = '@'
    scores_out = []
    for prompt in prompts:
        llm_solutions = []
        solve_prompts = []
        for task, _ in dev_set:
            solve_prompt = direct.format_prompt(prompt, task)
            solve_prompts.append(solve_prompt)
        out = direct.send_prompt_to_model(solve_prompts)
        llm_solutions.extend(out)
        first_prefer = 0
        second_prefer = 0
        evaluate_prompts = []
        for llm_solution, (task, dev_improved_arg) in zip(llm_solutions, dev_set):
            user_prompt_compare = user_prompt_compare_template.format(
                argument1=dev_improved_arg, argument2=llm_solution
            )
            evaluate_prompt = direct.format_prompt(
                system_prompt_compare, user_prompt_compare, assistant_prompt_compare
            )
            evaluate_prompts.append(evaluate_prompt)
        out = direct.send_prompt_to_model(evaluate_prompts)
        for response in out:
            if 'first@' in response.lower():
                first_prefer += 1
            elif 'second@' in response.lower():
                second_prefer += 1
            else:
                continue  # do nothing if unclear
        # calculate how often the second one was preferred
        scores_out.append(second_prefer / (first_prefer + second_prefer))
    return scores_out


def aggressive_relative_scoring(
        prompts: list[str],
        dev_set: list[tuple[str, str]]
):
    direct = DirectPrompt('llama3')
    system_prompt_compare = '''You are given an initial version of an argument, and an improved version of it. Your task is to decide whether the improvement was successful. Be very critical and aggressive in your assessment. If the improved argument is not significantly better, respond with @Failure@. If it significantly improved, respond with @Success@.'''
    user_prompt_compare_template = '''Argument 1: {argument1}
Argument 2: {argument2}'''
    assistant_prompt_compare = '@'
    scores_out = []
    for prompt in prompts:
        llm_solutions = []
        solve_prompts = []
        for task, _ in dev_set:
            solve_prompt = direct.format_prompt(prompt, task)
            solve_prompts.append(solve_prompt)
        out = direct.send_prompt_to_model(solve_prompts)
        llm_solutions.extend(out)
        failure = 0
        success = 0
        evaluate_prompts = []
        for llm_solution, (task, dev_improved_arg) in zip(llm_solutions, dev_set):
            user_prompt_compare = user_prompt_compare_template.format(
                argument1=dev_improved_arg, argument2=llm_solution
            )
            evaluate_prompt = direct.format_prompt(
                system_prompt_compare, user_prompt_compare, assistant_prompt_compare
            )
            evaluate_prompts.append(evaluate_prompt)
        out = direct.send_prompt_to_model(evaluate_prompts)
        for response in out:
            if 'failure@' in response.lower():
                failure += 1
            elif 'success@' in response.lower():
                success += 1
            else:
                continue  # do nothing if unclear
        # calculate success rate
        scores_out.append(success / (failure + success))
    return scores_out


def improve_microtexts_ga(
        argument_df: pd.DataFrame,
        model: str,
        format_func: callable,
        use_vllm: bool,
        num_demonstrations: int = 3,
        vllm_model_instance: Optional[LLM] = None,
        budget: int = 3,
        n: int = 3,
        population_size: int = 5,
        dev_set_genetic: Optional[list[tuple[str, str]]] = None,
        demonstration_args: Optional[list[tuple[str, str]]] = None
):
    if dev_set_genetic is None:
        dev_set_genetic = get_demonstrations(50)
    scoring_function = preference_based_scoring
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_file_dir, 'ga_checkpoints')
    initial_population = [
        ('Improve the following argument', None),
        ('Make the following argument better', None),
        ('Enhance the following argument', None),
        ('Make the next argument not suck', None)
    ]

    ga = GeneticAlgorithmPrompting(
        model='llama3',
        initial_population=initial_population,
        dev_set=dev_set_genetic,
        evolution_selection_strategy='roulette',
        evolution_perform_strategy='sim',
        checkpoint_path=checkpoint_path,
        task='microtext_improvement_ga',
        scoring_function=scoring_function,
        budget=budget,
        n=n,
        population_size=population_size,
        use_already_evaluated_for_initial=True,
        always_evaluate_initial=False,
        vllm_model_instance=vllm_model_instance
    )
    optimized_prompt = ga.do_prompting('')
    if demonstration_args is None:
        demonstration_args = get_demonstrations(num_demonstrations)
    demonstrations = []
    for old, new in demonstration_args:
        demonstrations.append(f'Original: {old.strip()}\nImproved: @{new.strip()}@\n\n')
    unformatted_prompts = prepare_microtext_prompts(argument_df, demonstrations)
    direct = DirectPrompt(model, format_func=format_func, use_vllm=use_vllm, vllm_model_instance=vllm_model_instance)
    # use optimized prompt to improve the arguments, instead of the system prompt from unformatted_prompts
    formatted_model_prompts = [direct.format_prompt(optimized_prompt, user_prompt, assistant_prompt) for
                               _, user_prompt, assistant_prompt in unformatted_prompts]
    outputs = direct.do_prompting(formatted_model_prompts)
    improved_out = []
    for (system_prompt, user_prompt, assistant_prompt), formatted_model_prompt, output in zip(unformatted_prompts,
                                                                                              formatted_model_prompts,
                                                                                              outputs):
        full_output = assistant_prompt + output
        pattern = r'@(.*?)@'
        substrings = re.findall(pattern, full_output)
        try:
            improved_argument = max(substrings, key=len)
        except ValueError:
            improved_argument = full_output[len(assistant_prompt):]
        print(f'Improved argument: {improved_argument}')
        improved_out.append(improved_argument)
    return improved_out, optimized_prompt


def improve_microtexts_de(
        argument_df: pd.DataFrame,
        model: str,
        format_func: callable,
        use_vllm: bool,
        num_demonstrations: int = 3,
        vllm_model_instance: Optional[LLM] = None,
        budget: int = 5,
        n: int = 10,
        population_size: int = 10,
        dev_set_genetic: Optional[list[tuple[str, str]]] = None,
        demonstration_args: Optional[list[tuple[str, str]]] = None
):
    if dev_set_genetic is None:
        dev_set_genetic = get_demonstrations(50)
    scoring_function = aggressive_relative_scoring
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_file_dir, 'de_checkpoints')
    initial_population = [
        ('Improve the following argument', None),
        ('Make the following argument better', None),
        ('Enhance the following argument', None),
        ('Make the next argument not suck', None)
    ]

    ga = DifferentialEvolutionPrompting(
        model='llama3',
        initial_population=initial_population,
        dev_set=dev_set_genetic,
        evolution_selection_strategy='roulette',
        evolution_perform_strategy='sim',
        checkpoint_path=checkpoint_path,
        task='microtext_improvement_de',
        scoring_function=scoring_function,
        budget=budget,
        n=n,
        population_size=population_size,
        use_already_evaluated_for_initial=True,
        always_evaluate_initial=False,
        vllm_model_instance=vllm_model_instance
    )
    optimized_prompt = ga.do_prompting('')
    if demonstration_args is None:
        demonstration_args = get_demonstrations(num_demonstrations)
    demonstrations = []
    for old, new in demonstration_args:
        demonstrations.append(f'Original: {old.strip()}\nImproved: @{new.strip()}@\n\n')
    unformatted_prompts = prepare_microtext_prompts(argument_df, demonstrations)
    direct = DirectPrompt(model, format_func=format_func, use_vllm=use_vllm, vllm_model_instance=vllm_model_instance)
    # use optimized prompt to improve the arguments, instead of the system prompt from unformatted_prompts
    formatted_model_prompts = [direct.format_prompt(optimized_prompt, user_prompt, assistant_prompt) for
                               _, user_prompt, assistant_prompt in unformatted_prompts]
    outputs = direct.do_prompting(formatted_model_prompts)
    improved_out = []
    for (system_prompt, user_prompt, assistant_prompt), formatted_model_prompt, output in zip(unformatted_prompts,
                                                                                              formatted_model_prompts,
                                                                                              outputs):
        full_output = assistant_prompt + output
        pattern = r'@(.*?)@'
        substrings = re.findall(pattern, full_output)
        try:
            improved_argument = max(substrings, key=len)
        except ValueError:
            improved_argument = full_output[len(assistant_prompt):]
        print(f'Improved argument: {improved_argument}')
        improved_out.append(improved_argument)
    return improved_out, optimized_prompt


def little_brother_answer_parser(
        answers: list[str]
) -> list[str]:
    out = []
    for answer in answers:
        pattern = r'@(.*?)@'
        substrings = re.findall(pattern, answer)
        # find longest substring
        try:
            improved_argument = max(substrings, key=len)
        except ValueError:
            improved_argument = answer
        out.append(improved_argument)
    return out


def improve_microtexts_little_brother(
        argument_df: pd.DataFrame,
        model: str,
        use_vllm: bool,
        num_demonstrations: int = 3,
        vllm_model_instance: Optional[LLM] = None
) -> list[str]:
    demonstration_args = get_demonstrations(num_demonstrations)
    demonstrations = []
    for old, new in demonstration_args:
        demonstrations.append(f'Original: {old.strip()}\nImproved: @{new.strip()}@\n\n')

    unformatted_prompts = prepare_microtext_prompts(argument_df, demonstrations)
    raw_tasks = []
    for i, row in tqdm(argument_df.iterrows(), total=len(argument_df)):
        topic = row['topic_id']
        stance = row['stance']
        argument = row['argument']
        system_prompt = f"You are given an argument about the topic \"{topic}\". Your task is to improve it. Respond only with the improved argument wrapped in @ symbols and nothing else. Here are some examples of improvements:\n\n"
        system_prompt += ''.join(demonstrations)
        raw_tasks.append(system_prompt)

    little_brother = LittleBrotherPrompting(
        model=model,
        little_brother_model='llama3',
        answer_parser=little_brother_answer_parser,
        raw_tasks=raw_tasks,
        little_brother_prompt=None,
        use_vllm=use_vllm,
        vllm_model_instance=vllm_model_instance
    )
    formatted_model_prompts = [little_brother.format_prompt(system_prompt, user_prompt, assistant_prompt) for
                               system_prompt, user_prompt, assistant_prompt in unformatted_prompts]
    outputs = little_brother.do_prompting(formatted_model_prompts)

    arguments_out = []
    for (system_prompt, user_prompt, assistant_prompt), formatted_model_prompt, output in zip(unformatted_prompts,
                                                                                              formatted_model_prompts,
                                                                                              outputs):
        full_output = assistant_prompt + output
        pattern = r'@(.*?)@'
        substrings = re.findall(pattern, full_output)
        # find longest substring
        try:
            improved_argument = max(substrings, key=len)
        except ValueError:
            improved_argument = full_output
        print(f'Improved argument: {improved_argument}')
        arguments_out.append(improved_argument)
    return arguments_out


if __name__ == '__main__':
    _df = load_microtext_df()
    # improve_microtexts_direct(
    #     _df,
    #     'microsoft/Phi-3-mini-4k-instruct',
    #     format_func=phi_format_func,
    #     stop='@'
    # )

    improve_microtexts_direct(
        _df,
        'google/flan-t5-xxl',
        num_demonstrations=0,
        max_new_tokens=2048,
        use_vllm=False
    )

    improve_microtexts_bsm(_df, 'llama3')

    _reasoning_path_example = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples',
                                           'reasoning_example.txt')
    with open(_reasoning_path_example, 'r') as file:
        _reasoning_example = file.read()

    _reasoning_example_plan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples',
                                                'reasoning_example_plan.txt')
    with open(_reasoning_example_plan_path, 'r') as file:
        _reasoning_example_plan = file.read()

    improve_self_discover(
        _df,
        'llama3',
        3,
        _reasoning_example,
        _reasoning_example_plan_path
    )

    improve_microtexts_ga(_df, 'llama3', llama3_format_func, False)
    improve_microtexts_de(_df, 'llama3', llama3_format_func, False)
    improve_microtexts_little_brother(_df, 'llama3')

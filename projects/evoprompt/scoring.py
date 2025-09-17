from direct.direct import DirectPrompt


def direct_scoring(
        prompts: list[str],
        dev_set: list[tuple[str, str]],
        *args,
        **kwargs
):
    direct = DirectPrompt('llama3')
    system_prompt_verify = '''You are given both a task with its intended solution, as well as the solution by a student. Your task is to evaluate whether the student solved the task correctly. Respond with @Yes@ if it was solved correctly, and with @No@ if it was not.'''
    user_prompt_verify_template = '''[Task]
{task}
[/Task]
[Intended Solution]
{solution}
[/Intended Solution]
-----
[Student Solution]
{student_solution}
[/Student Solution]'''
    assistant_prompt_verify = '@'
    scores_out = []
    for prompt in prompts:
        llm_solutions = []

        # solve dev set tasks
        solve_prompts = []
        for task, dev_solution in dev_set:
            if kwargs.get('format_into_prompt', None):
                tmp = prompt.replace(kwargs['format_into_prompt'], task)
                solve_prompt = direct.format_prompt(tmp, '')
            else:
                # solve the task, with prompt
                solve_prompt = direct.format_prompt(prompt, task)
            solve_prompts.append(solve_prompt)
        out = direct.send_prompt_to_model(solve_prompts)
        llm_solutions.extend(out)

        # evaluate solutions
        correct = 0
        wrong = 0
        evaluate_prompts = []
        for llm_solution, (task, correct_solution) in zip(llm_solutions, dev_set):
            user_prompt_verify = user_prompt_verify_template.format(
                task=task, solution=correct_solution, student_solution=llm_solution
            )
            evaluate_prompt = direct.format_prompt(
                system_prompt_verify, user_prompt_verify, assistant_prompt_verify
            )
            evaluate_prompt = evaluate_prompt.replace('{task}', task).replace('{solution}', correct_solution).replace(
                '{student_solution}', llm_solution)
            evaluate_prompts.append(evaluate_prompt)
        out = direct.send_prompt_to_model(evaluate_prompts)
        for response in out:
            if 'yes@' in response.lower():
                correct += 1
            elif 'no@' in response.lower():
                wrong += 1
            else:
                wrong += 1

        accuracy = correct / (correct + wrong)
        scores_out.append(accuracy)
    return scores_out

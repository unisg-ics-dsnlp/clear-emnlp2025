# some approaches/models add fluff like 'here is your improved argument:' - this script aims to remvoe that
import json
import os

from direct.direct import DirectPrompt
from util.template import llama3_format_func

improved_arguments_path = '../../localFiles/essays-Phi-3-mini-4k-instruct.json'
with open(improved_arguments_path, 'r') as f:
    improved = json.load(f)

system_prompt = 'You are given an improved argumentative text. The input may contain fluff that is not related to the argumentative text. Your task is to wrap the improved argumentative text in @ symbols. For example, if the argumentative text is "The sky is blue.", you should wrap it as "@The sky is blue.@". Ignore all other fluff and respond only with the improved argumentative text wrapped in @. If there is no argumentative text then respond with @INVALID@.'

print(os.listdir('.'))
essays = os.listdir(os.path.join('projects', 'improvement_pipeline', 'ArgumentAnnotatedEssays-2.0', 'brat-project-final', 'brat-project-final'))
essays = sorted([essay for essay in essays if essay.endswith('.txt')])

direct = DirectPrompt(
    'llama3',
    format_func=llama3_format_func,
    use_vllm=False
)
cleaned_out = []
for approach, args in improved:
    if approach in ['de', 'ga']:
        args, improved_prompt = args
    print(f'> Cleaning for approach {approach}')
    prepared_prompts = []
    assistant_prompt = f'Improved argument: @'
    for arg in args:
        user_prompt = f'<|Improved argument|>: {arg}'
        prompt = direct.format_prompt(system_prompt, user_prompt, assistant_prompt)
        prepared_prompts.append(prompt)

    cleaned = direct.do_prompting(prepared_prompts)
    # extract improved arguments
    improved_args = []
    for (i, solution), essay in zip(enumerate(cleaned), essays):
        essay_path = os.path.join('projects', 'improvement_pipeline', 'ArgumentAnnotatedEssays-2.0', 'brat-project-final', 'brat-project-final', essay)
        with open(essay_path, 'r') as f:
            essay_text = f.read()
        solution = '@' + solution
        try:
            tmp = solution.split('@')
            # take longest element
            solution_parsed = max(tmp, key=len)
        except IndexError:
            solution_parsed = ''
        improved_args.append((solution_parsed, essay_path, essay_text))
    cleaned_out.append((approach, improved_args))

with open('../../localFiles/essays-Phi-3-mini-4k-instruct-cleaned.json', 'w') as f:
    json.dump(cleaned_out, f, indent=4)

import os
from typing import Optional, Union

from util.template import PromptTemplate


class SelfDiscover(PromptTemplate):
    def __init__(
            self,
            model: str,
            modules: Optional[list[str]] = None,
            *args,
            **kwargs
    ):
        super().__init__(
            model,
            *args,
            **kwargs
        )
        self.modules = modules
        if self.modules is None:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_file_dir, 'REASONING_MODULES.txt')
            with open(file_path, 'r', encoding='utf-8') as file:
                self.modules = file.read().splitlines()
        self.plan = None

    def select_reasoning_modules(
            self,
            modules: list[str],
            examples: list[str],
    ):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_file_dir, 'SELECT_prompt.txt')
        with open(file_path, 'r') as file:
            prompt = file.read()
        prompt = prompt.replace('[MODULES]', ', '.join(modules))
        prompt = prompt.replace('[EXAMPLES]', '\n'.join(examples))

        prompt = self.format_prompt(prompt, '')
        prompt_response = self.send_prompt_to_model(prompt)
        return prompt_response, prompt

    def adapt_reasoning_modules(
            self,
            modules: str,
            examples: list[str],
    ):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_file_dir, 'ADAPT_prompt.txt')
        with open(file_path, 'r') as file:
            prompt = file.read()
        prompt = prompt.replace('[MODULES]', modules)
        prompt = prompt.replace('[EXAMPLES]', ', '.join(examples))
        prompt = self.format_prompt(prompt, '')
        prompt_response = self.send_prompt_to_model(prompt)
        return prompt_response, prompt

    def implement_reasoning_modules(
            self,
            reasoning_example: str,
            reasoning_example_plan: str,
            modules: str,
            examples: list[str],
    ):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_file_dir, 'IMPLEMENT_prompt.txt')
        with open(file_path, 'r') as file:
            prompt = file.read()
        prompt = prompt.replace('[REASONING EXAMPLE]', reasoning_example)
        prompt = prompt.replace('[REASONING EXAMPLE PLAN]', reasoning_example_plan)
        prompt = prompt.replace('[MODULES]', modules)
        prompt = prompt.replace('[EXAMPLES]', ', '.join(examples))
        prompt = self.format_prompt(prompt, '')
        prompt_response = self.send_prompt_to_model(prompt)
        return prompt_response, prompt

    def execute_reasoning_plan(
            self,
            reasoning_plan: str,
            task_instance: Union[str, list[str]],
    ):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_file_dir, 'EXECUTE_prompt.txt')
        with open(file_path, 'r') as file:
            prompt = file.read()
        prompt = prompt.replace('[REASONING_PLAN]', reasoning_plan)
        if isinstance(task_instance, list):
            prepared_prompts = []
            for task in task_instance:
                prepared_prompts.append(prompt.replace('[TASK_INSTANCE]', task))
            prompt_response = self.send_prompt_to_model(prepared_prompts)
        else:
            prompt = prompt.replace('[TASK_INSTANCE]', task_instance)
            prompt_response = self.send_prompt_to_model(prompt)
        return prompt_response, prompt

    def do_prompting(
            self,
            prompt: Union[str, list[str]],
            demonstrations: Optional[list[str]] = None,
            *args,
            **kwargs
    ) -> tuple[str, list[str]]:
        select_prompt = adapt_prompt = implement_prompt = ''
        if self.plan is None or kwargs.get('update_plan', False):
            print('> Selecting reasoning modules')
            selected_modules, select_prompt = self.select_reasoning_modules(
                self.modules,
                demonstrations
            )

            print('> Adapting reasoning modules')
            adapted_modules, adapt_prompt = self.adapt_reasoning_modules(
                '\n'.join(selected_modules),
                demonstrations
            )
            adapted_modules = '\n'.join(adapted_modules)

            reasoning_example = kwargs['reasoning_example']
            reasoning_example_plan = kwargs['reasoning_example_plan']
            print('> Implementing reasoning modules')
            plan, implement_prompt = self.implement_reasoning_modules(
                reasoning_example,
                reasoning_example_plan,
                adapted_modules,
                demonstrations
            )
            plan = '\n'.join(plan)
            self.plan = plan

        print('> Executing reasoning plan')
        execution, execute_prompt = self.execute_reasoning_plan(
            self.plan,
            prompt,
        )
        return execution, [select_prompt, adapt_prompt, implement_prompt, execute_prompt]

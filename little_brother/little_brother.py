from typing import Union, Optional

from direct.direct import DirectPrompt
from util.template import PromptTemplate


class LittleBrotherPrompting(PromptTemplate):
    def __init__(
            self,
            model: str,
            little_brother_model: str,
            answer_parser: callable,
            raw_tasks: list[str],
            little_brother_prompt: Optional[str] = None,
            temperature: float = 0.7,
            *args,
            **kwargs
    ):
        super().__init__(model, temperature=temperature, *args, **kwargs)
        self.little_brother_model = little_brother_model
        self.little_brother = DirectPrompt(model=little_brother_model, temperature=temperature)
        self.little_brother_answer_parser = answer_parser
        self.raw_tasks = raw_tasks
        if little_brother_prompt is None:
            system_prompt = '''Solve this task: {task}. Your little brother has solved this task like this previously:
[PREVIOUS]
{previous}
[/PREVIOUS]'''
            user_prompt = '''Check if your little brother's solution is correct. If it is not, teach them where they made a mistake, and correct it. If it is correct, state the solution and explain it. Put the corrected solution into @ symbols.'''
            self.little_brother_prompt = self.format_prompt(system_prompt, user_prompt)
        else:
            self.little_brother_prompt = little_brother_prompt

    def do_prompting(
            self,
            prompt: Union[str, list[str]],
            *args,
            **kwargs
    ) -> str:
        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) != len(self.raw_tasks):
            raise ValueError(
                f'Number of prompts ({len(prompt)}) does not match number of tasks ({len(self.raw_tasks)})'
            )
        little_brother_solutions = self.little_brother.do_prompting(prompt)
        little_brother_solutions_parsed = self.little_brother_answer_parser(little_brother_solutions)
        prepared_prompts = []
        for task, little_brother_solution in zip(self.raw_tasks, little_brother_solutions_parsed):
            tmp = self.little_brother_prompt.replace('{task}', task)
            tmp = tmp.replace('{previous}', little_brother_solution)
            prepared_prompts.append(tmp)
        return self.send_prompt_to_model(prepared_prompts)

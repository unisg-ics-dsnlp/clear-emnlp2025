import os
import re
from typing import Optional, Union

from util.template import PromptTemplate


class BSM(PromptTemplate):
    def __init__(
            self,
            model: str,
            branch_candidates: str,
            branch_prompt: str,
            solve_prompt: str,
            merge_prompt: str,
            num_branches: int = 5,
            temperature: float = 0.7,
            keyword: Optional[str] = None,
            *args,
            **kwargs
    ):
        super().__init__(model, temperature=temperature, *args, **kwargs)
        self.branch_candidates = branch_candidates
        self.branch_prompt = branch_prompt
        self.solve_prompt = solve_prompt
        self.merge_prompt = merge_prompt
        self.num_branches = num_branches
        self.keyword = keyword

    def do_prompting(
            self,
            prompt: str,
            demonstrations: Optional[list[str]] = None,
            *args,
            **kwargs
    ) -> str | list[str]:
        branches = self.do_branching()
        solved_branches = self.do_solving(branches)
        merged_stories = self.do_merging(solved_branches)
        return merged_stories

    def do_branching(
            self
    ) -> list[tuple[list[str], str | None]]:
        pattern = r'^([a-zA-Z]+\s?)[0-9]*:?'
        prompt = self.branch_prompt.format(concepts=self.branch_candidates)
        branches = []
        for i in range(self.num_branches):
            response = self.send_prompt_to_model(prompt)
            group_cands = []
            extracted_keyword = None
            for line in response.split('\n'):
                line = line.strip()
                if line.lower().startswith('group'):
                    candidate = re.sub(pattern, r'\1', line.lower())
                    candidate = candidate.split('group')[1]
                    if not candidate.strip():
                        continue
                    group_cands.append(candidate)
                if self.keyword is not None:
                    if self.keyword in line.lower():
                        extracted_keyword = line.lower().split(self.keyword)[1].strip()
                        if extracted_keyword.startswith(':'):
                            extracted_keyword = extracted_keyword[1:].strip()
            if len(group_cands) < 2:
                continue  # did not properly make/find groups
            branches.append((group_cands, extracted_keyword))
        return branches

    def do_solving(
            self,
            branches_all: list[tuple[list[str], str | None]]
    ) -> list[list[tuple[str, str]]]:
        solved_branches_all = []
        for branches, keyword in branches_all:
            branch_prompts = []
            solved_branches = []
            for branch in branches:
                if keyword is not None:
                    tmp = self.solve_prompt.format(branch=branch, keyword=keyword)
                else:
                    tmp = self.solve_prompt.format(branch=branch)
                branch_prompts.append((tmp, branch))
            for branch_prompt, branch in branch_prompts:
                response = self.send_prompt_to_model(branch_prompt)
                solved_branches.append((response, branch))
            solved_branches_all.append(solved_branches)
        return solved_branches_all

    def do_merging(
            self,
            solved_branches_all: list[list[tuple[str, str]]]
    ) -> list[str]:
        out = []
        for solved_branches in solved_branches_all:
            tmp = {}
            for i, (story, concepts) in enumerate(solved_branches):
                tmp[f'story{i + 1}'] = story
                tmp[f'concepts{i + 1}'] = concepts
            prompt = self.merge_prompt.format(**tmp)
            response = self.send_prompt_to_model(prompt)
            out.append(response)
        return out


class BSMFindBranches(PromptTemplate):
    def __init__(
            self,
            branch_prompt: str,
            solve_prompt: str,
            merge_prompt: str,
            num_branches: int,
            branch_formatters: list[callable],
            solve_formatters: list[callable],
            merge_formatters: list[callable],
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.branch_prompt = branch_prompt
        self.formatted_branch_prompts = []
        self.solve_prompt = solve_prompt
        self.formatted_solve_prompts = []
        self.merge_prompt = merge_prompt
        self.formatted_merge_prompts = []
        self.num_branches = num_branches
        self.branch_formatters = branch_formatters
        self.solve_formatters = solve_formatters
        self.merge_formatters = merge_formatters

    def _do_branching(
            self,
            tasks: list[str]
    ) -> list[tuple[str, ...]]:
        branch_prompts = []
        for task, formatter in zip(tasks, self.branch_formatters):
            tmp = formatter(self.branch_prompt)
            tmp = self.format_prompt(tmp, task, 'Group 1: ')
            branch_prompts.append(tmp)
        self.formatted_branch_prompts = branch_prompts
        branch_solutions = self.send_prompt_to_model(branch_prompts)
        pattern = r'^([a-zA-Z]+\s?)[0-9]*:?'
        branch_solutions_parsed: list[tuple[str, ...]] = []
        for solution in branch_solutions:
            solution = 'Group 1: ' + solution
            group_cands = []
            for line in solution.split('\n'):
                line = line.strip()
                if line.lower().startswith('group'):
                    candidate = re.sub(pattern, r'\1', line.lower())
                    candidate = candidate.split('group')[1]
                    if not candidate.strip():
                        continue
                    group_cands.append(candidate)
            if len(group_cands) < self.num_branches:
                branch_solutions_parsed.append(tuple('' for _ in range(self.num_branches)))
            elif len(group_cands) > self.num_branches:
                # use only as many as needed
                branch_solutions_parsed.append(tuple(group_cands[:self.num_branches]))
            else:
                branch_solutions_parsed.append(tuple(group_cands))
        return branch_solutions_parsed

    def _do_solving(
            self,
            branches_all: list[tuple[str, ...]],
            tasks: list[str]
    ) -> list[list[str]]:
        solve_prompts: list[str] = []
        for task, branches in zip(tasks, branches_all):
            tmp = []
            for branch, formatter in zip(branches, self.solve_formatters):
                tmp2 = self.solve_prompt.replace('>group<', branch)
                tmp.append(self.format_prompt(formatter(tmp2), task, '@'))
            solve_prompts.extend(tmp)
        self.formatted_solve_prompts = solve_prompts
        solve_solutions = self.send_prompt_to_model(solve_prompts)
        solve_solutions_parsed = []
        for solution in solve_solutions:
            solution = '@' + solution
            try:
                tmp = solution.split('@')
                # take longest element
                solution_parsed = max(tmp, key=len)
            except IndexError:
                solution_parsed = ''
            solve_solutions_parsed.append(solution_parsed)
        out = []
        for i in range(0, len(solve_solutions), self.num_branches):
            out.append(solve_solutions_parsed[i:i + self.num_branches])
        return out

    def _do_merging(
            self,
            solutions_all: list[list[str]],
            tasks: list[str]
    ) -> list[str]:
        merge_prompts = []
        for solutions, formatter, task in zip(solutions_all, self.merge_formatters, tasks):
            tmp = ''
            for i, solution in enumerate(solutions):
                tmp += f'Solution {i + 1}: {solution}\n'
            tmp2 = formatter(self.merge_prompt)
            tmp2 = self.format_prompt(tmp2, tmp, '@')
            merge_prompts.append(tmp2)
        self.formatted_merge_prompts = merge_prompts
        merge_solutions = self.send_prompt_to_model(merge_prompts)
        return merge_solutions

    def do_prompting(
            self,
            prompts: list[str],
            *args,
            **kwargs
    ) -> list[str]:
        branch = self._do_branching(prompts)
        solve = self._do_solving(branch, prompts)
        merge = self._do_merging(solve, prompts)
        parsed = []
        for solution in merge:
            tmp = '@' + solution
            try:
                tmp2 = tmp.split('@')
                # take longest element
                parsed.append(max(tmp2, key=len))
            except IndexError:
                parsed.append('')
        return parsed

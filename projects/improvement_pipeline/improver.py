import os
from typing import Optional

import pandas as pd
from vllm import LLM

from projects.improvement_pipeline.improve_arguments import improve_arguments_direct, improve_arguments_bsm, \
    improve_arguments_selfdiscover, improve_arguments_ga, improve_arguments_de, improve_arguments_little_brother, \
    ArgumentDataFrame


class ArgumentImprover:
    def __init__(
            self,
            arguments: ArgumentDataFrame,
            model: str,
            use_vllm: bool,
            format_func: callable,
            vllm_model_instance: Optional[LLM] = None,
            budget: int = 5,
            n: int = 2,
            population_size: int = 10,
            dev_set_genetic: Optional[list[tuple[str, str]]] = None,
            demonstration_args: Optional[list[tuple[str, str]]] = None,
            use_feedback: Optional[bool] = False
    ):
        self.arguments = arguments
        self.model = model
        self.use_vllm = use_vllm
        self.format_func = format_func
        self.vllm_model_instance = vllm_model_instance
        self.budget = budget
        self.n = n
        self.population_size = population_size
        self.dev_set_genetic = dev_set_genetic
        self.demonstration_args = demonstration_args
        self.bsm_formatted_branch_prompts = []
        self.bsm_formatted_solve_prompts = []
        self.bsm_formatted_merge_prompts = []
        self.selfdiscover_select_prompt = ''
        self.selfdiscover_adapt_prompt = ''
        self.selfdiscover_implement_prompt = ''
        self.selfdiscover_execute_prompt = ''
        self.use_feedback = use_feedback

    def improve_direct(
            self,
            num_demonstrations: int = 0,
            max_new_tokens: int = 2048
    ) -> list[str]:
        return improve_arguments_direct(
            self.arguments,
            self.model,
            num_demonstrations=num_demonstrations,
            format_func=self.format_func,
            use_vllm=self.use_vllm,
            vllm_model_instance=self.vllm_model_instance,
            use_feedback=self.use_feedback,
            max_new_tokens=max_new_tokens,
        )

    def improve_bsm(
            self,
    ) -> list[str]:
        improved_args, bsm = improve_arguments_bsm(
            self.arguments,
            self.model,
            use_vllm=self.use_vllm,
            vllm_model_instance=self.vllm_model_instance,
            use_feedback=self.use_feedback
        )
        self.bsm_formatted_branch_prompts = bsm.formatted_branch_prompts
        self.bsm_formatted_solve_prompts = bsm.formatted_solve_prompts
        self.bsm_formatted_merge_prompts = bsm.formatted_merge_prompts
        return improved_args

    def improve_self_discover(
            self,
            num_demonstrations: int = 3,
            demonstration_args: Optional[list[tuple[str, str]]] = None
    ) -> list[str]:
        reasoning_example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples',
                                              'reasoning_example.txt')
        reasonong_example_plan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples',
                                                   'reasoning_example_plan.txt')
        with open(reasoning_example_path, 'r') as f:
            reasoning_example = f.read()

        with open(reasonong_example_plan_path, 'r') as f:
            reasoning_example_plan = f.read()

        improved, _original, (
            select_prompt,
            adapt_prompt,
            implement_prompt,
            execute_prompt
        ) = improve_arguments_selfdiscover(
            self.arguments,
            self.model,
            reasoning_example=reasoning_example,
            reasoning_example_plan=reasoning_example_plan,
            use_vllm=self.use_vllm,
            vllm_model_instance=self.vllm_model_instance,
            num_demonstrations=num_demonstrations,
            demonstration_args=demonstration_args,
            use_feedback=self.use_feedback
        )
        self.selfdiscover_select_prompt = select_prompt
        self.selfdiscover_adapt_prompt = adapt_prompt
        self.selfdiscover_implement_prompt = implement_prompt
        self.selfdiscover_execute_prompt = execute_prompt
        return improved

    def improve_ga(
            self
    ) -> tuple[list[str], str]:
        return improve_arguments_ga(
            self.arguments,
            self.model,
            self.format_func,
            self.use_vllm,
            vllm_model_instance=self.vllm_model_instance,
            budget=self.budget,
            n=self.n,
            population_size=self.population_size,
            dev_set_genetic=self.dev_set_genetic,
            demonstration_args=self.demonstration_args,
            use_feedback=self.use_feedback
        )

    def improve_de(
            self
    ) -> tuple[list[str], str]:
        return improve_arguments_de(
            self.arguments,
            self.model,
            self.format_func,
            self.use_vllm,
            vllm_model_instance=self.vllm_model_instance,
            budget=self.budget,
            n=self.n,
            population_size=self.population_size,
            dev_set_genetic=self.dev_set_genetic,
            demonstration_args=self.demonstration_args
        )

    def improve_little_brother(self) -> list[str]:
        return improve_arguments_little_brother(
            self.arguments,
            self.model,
            use_vllm=self.use_vllm,
            vllm_model_instance=self.vllm_model_instance,
            use_feedback=self.use_feedback
        )

    def do_all(
            self,
            num_demonstrations: int = 3,
            max_new_tokens: int = 2048
    ):
        improveds = []
        out = []
        print('> Improving with direct method')
        try:
            direct_improved = self.improve_direct(num_demonstrations=num_demonstrations, max_new_tokens=max_new_tokens)
            tmp = {
                'approach': 'direct',
                'original_arguments': self.arguments['argument'].tolist(),
                'topic_id': self.arguments['topic_id'].tolist(),
                'graph_id': self.arguments['graph_id'].tolist(),
                'stance': self.arguments['stance'].tolist(),
                'improved_arguments': direct_improved,
                'bsm_formatted_branch_prompts': 'N/A',
                'bsm_formatted_solve_prompts': 'N/A',
                'bsm_formatted_merge_prompts': 'N/A',
                'selfdiscover_select_prompt': 'N/A',
                'selfdiscover_adapt_prompt': 'N/A',
                'selfdiscover_implement_prompt': 'N/A',
                'selfdiscover_execute_prompt': 'N/A',
                'optimized_prompt': 'N/A',
            }
            # get the additional columns from self.arguments df
            for col in self.arguments.columns:
                if col not in tmp:
                    tmp[col] = self.arguments[col].tolist()

            out.append(tmp)
            improveds.append(('direct', direct_improved))
        except Exception as e:
            print(e)

        print('> Improving with bsm method')
        try:
            bsm_improved = self.improve_bsm()
            tmp = {
                'approach': 'bsm',
                'original_arguments': self.arguments['argument'].tolist(),
                'topic_id': self.arguments['topic_id'].tolist(),
                'graph_id': self.arguments['graph_id'].tolist(),
                'stance': self.arguments['stance'].tolist(),
                'improved_arguments': bsm_improved,
                'bsm_formatted_branch_prompts': self.bsm_formatted_branch_prompts,
                'bsm_formatted_solve_prompts': self.bsm_formatted_solve_prompts,
                'bsm_formatted_merge_prompts': self.bsm_formatted_merge_prompts,
                'selfdiscover_select_prompt': 'N/A',
                'selfdiscover_adapt_prompt': 'N/A',
                'selfdiscover_implement_prompt': 'N/A',
                'selfdiscover_execute_prompt': 'N/A',
                'optimized_prompt': 'N/A',
            }
            # get the additional columns from self.arguments df
            for col in self.arguments.columns:
                if col not in tmp:
                    tmp[col] = self.arguments[col].tolist()
            out.append(tmp)
            improveds.append(('bsm', bsm_improved))
        except Exception as e:
            print(e)

        print('> Improving with self_discover method')
        try:
            self_discover_improved = self.improve_self_discover(num_demonstrations=num_demonstrations)
            tmp = {
                'approach': 'self_discover',
                'original_arguments': self.arguments['argument'].tolist(),
                'topic_id': self.arguments['topic_id'].tolist(),
                'graph_id': self.arguments['graph_id'].tolist(),
                'stance': self.arguments['stance'].tolist(),
                'improved_arguments': self_discover_improved,
                'bsm_formatted_branch_prompts': 'N/A',
                'bsm_formatted_solve_prompts': 'N/A',
                'bsm_formatted_merge_prompts': 'N/A',
                'selfdiscover_select_prompt': self.selfdiscover_select_prompt,
                'selfdiscover_adapt_prompt': self.selfdiscover_adapt_prompt,
                'selfdiscover_implement_prompt': self.selfdiscover_implement_prompt,
                'selfdiscover_execute_prompt': self.selfdiscover_execute_prompt,
                'optimized_prompt': 'N/A',
            }
            # get the additional columns from self.arguments df
            for col in self.arguments.columns:
                if col not in tmp:
                    tmp[col] = self.arguments[col].tolist()
            out.append(tmp)
            improveds.append(('self_discover', self_discover_improved))
        except Exception as e:
            print(e)

        print('> Improving with ga method')
        try:
            ga_improved, optimized_prompt = self.improve_ga()
            tmp = {
                'approach': 'ga',
                'original_arguments': self.arguments['argument'].tolist(),
                'topic_id': self.arguments['topic_id'].tolist(),
                'graph_id': self.arguments['graph_id'].tolist(),
                'stance': self.arguments['stance'].tolist(),
                'improved_arguments': ga_improved,
                'bsm_formatted_branch_prompts': 'N/A',
                'bsm_formatted_solve_prompts': 'N/A',
                'bsm_formatted_merge_prompts': 'N/A',
                'selfdiscover_select_prompt': 'N/A',
                'selfdiscover_adapt_prompt': 'N/A',
                'selfdiscover_implement_prompt': 'N/A',
                'selfdiscover_execute_prompt': 'N/A',
                'optimized_prompt': optimized_prompt,
            }
            # get the additional columns from self.arguments df
            for col in self.arguments.columns:
                if col not in tmp:
                    tmp[col] = self.arguments[col].tolist()
            out.append(tmp)
            improveds.append(('ga', ga_improved))
        except Exception as e:
            print(e)

        # TODO: if commenting in, make sure it adds the columns as needed (see the others), it may be outdated by now
        # print('> Improving with de method')
        # try:
        #     de_improved, optimized_prompt = self.improve_de()
        #     out.append({
        #         'approach': 'de',
        #         'original_arguments': self.arguments['argument'],
        #         'topic_id': self.arguments['topic_id'],
        #         'graph_id': self.arguments['graph_id'],
        #         'stance': self.arguments['stance'],
        #         'improved_arguments': de_improved,
        #         'bsm_formatted_branch_prompts': 'N/A',
        #         'bsm_formatted_solve_prompts': 'N/A',
        #         'bsm_formatted_merge_prompts': 'N/A',
        #         'selfdiscover_select_prompt': 'N/A',
        #         'selfdiscover_adapt_prompt': 'N/A',
        #         'selfdiscover_implement_prompt': 'N/A',
        #         'selfdiscover_execute_prompt': 'N/A',
        #         'optimized_prompt': optimized_prompt,
        #     })
        #     improveds.append(('de', de_improved))
        # except Exception as e:
        #     print(e)

        print('> Improving with little_brother method')
        try:
            little_brother_improved = self.improve_little_brother()
            tmp = {
                'approach': 'little_brother',
                'original_arguments': self.arguments['argument'].tolist(),
                'topic_id': self.arguments['topic_id'].tolist(),
                'graph_id': self.arguments['graph_id'].tolist(),
                'stance': self.arguments['stance'].tolist(),
                'improved_arguments': little_brother_improved,
                'bsm_formatted_branch_prompts': 'N/A',
                'bsm_formatted_solve_prompts': 'N/A',
                'bsm_formatted_merge_prompts': 'N/A',
                'selfdiscover_select_prompt': 'N/A',
                'selfdiscover_adapt_prompt': 'N/A',
                'selfdiscover_implement_prompt': 'N/A',
                'selfdiscover_execute_prompt': 'N/A',
                'optimized_prompt': 'N/A',
            }
            # get the additional columns from self.arguments df
            for col in self.arguments.columns:
                if col not in tmp:
                    tmp[col] = self.arguments[col].tolist()
            out.append(tmp)
            improveds.append(('little_brother', little_brother_improved))
        except Exception as e:
            print(e)
        return improveds, out

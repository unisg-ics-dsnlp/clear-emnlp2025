import argparse
import json
import os
from enum import Enum
from typing import Optional

import pandas as pd
from vllm import LLM
import torch.cuda

from direct.direct import DirectPrompt
from projects.improvement_pipeline.improve_arguments import ArgumentDataFrame
from projects.improvement_pipeline.improver import ArgumentImprover
from util.template import llama3_format_func, phi_format_func, mixtral_format_func, gemma_format_func

file_dir = os.path.dirname(os.path.abspath(__file__))
ESSAYS = os.listdir(os.path.join(file_dir, 'ArgumentAnnotatedEssays-2.0', 'brat-project-final', 'brat-project-final'))
ESSAYS = sorted([essay for essay in ESSAYS if essay.endswith('.txt')])

FORMAT_FUNCTIONS = [
    llama3_format_func,
    phi_format_func,
    gemma_format_func,
    mixtral_format_func,
    None
]


class SupportedDatasets(Enum):
    ESSAYS = 'essays'
    REVISIONS_DRAFT1 = 'revisions_draft1'
    REVISIONS_DRAFT2 = 'revisions_draft2'
    MICROTEXTS = 'microtexts'
    REVISIONS_DRAFT3 = 'revisions_draft3'
    REVISIONS_DRAFT1_FEEDBACK = 'revisions_draft1_feedback'
    REVISIONS_DRAFT2_FEEDBACK = 'revisions_draft2_feedback'
    REVISIONS_DRAFT3_FEEDBACK = 'revisions_draft3_feedback'


DATASETS = [
    SupportedDatasets.ESSAYS,
    SupportedDatasets.REVISIONS_DRAFT1,
    SupportedDatasets.REVISIONS_DRAFT2,
    SupportedDatasets.MICROTEXTS,
    SupportedDatasets.REVISIONS_DRAFT3,
    SupportedDatasets.REVISIONS_DRAFT1_FEEDBACK,
    SupportedDatasets.REVISIONS_DRAFT2_FEEDBACK,
    SupportedDatasets.REVISIONS_DRAFT3_FEEDBACK
]


def prepare_microtext_df() -> ArgumentDataFrame:
    """
    Load microtexts from csv file.
    :return: A pandas DataFrame with columns ['topic_id', 'graph_id', 'stance', 'argument'].
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    microtexts_path = os.path.join(current_file_dir, 'microtexts.csv')
    df = pd.read_csv(microtexts_path)
    df = df[df['topic_id'].notna()]
    df = ArgumentDataFrame(df)
    return df


def prepare_microtexts_both_df():
    """
    Load microtexts from both languages.
    :return: A tuple of two pandas DataFrames with columns ['topic_id', 'graph_id', 'stance', 'argument'].
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    microtexts_en_path = os.path.join(current_file_dir, 'microtexts.csv')
    microtexts_de_path = os.path.join(current_file_dir, 'microtexts_de.csv')
    df_en = pd.read_csv(microtexts_en_path)
    df_de = pd.read_csv(microtexts_de_path)
    df_en = df_en[df_en['topic_id'].notna()]
    df_en['topic_id'] = df_en['topic_id'].apply(lambda x: x + ' (en)')
    df_de = df_de[df_de['topic_id'].notna()]
    df_de['topic_id'] = df_de['topic_id'].apply(lambda x: x + ' (de)')
    df_merged = pd.concat([df_en, df_de], ignore_index=True)
    df_merged = ArgumentDataFrame(df_merged)
    return df_merged


def prepare_essays_df() -> ArgumentDataFrame:
    essays_df = []
    for essay in ESSAYS:
        with open(os.path.join(file_dir, 'ArgumentAnnotatedEssays-2.0', 'brat-project-final', 'brat-project-final',
                               essay), 'r') as f:
            text = f.read()
        topic = text.split('\n')[0].strip()
        argument = '\n'.join(text.split('\n')[1:]).strip()
        graph_id = essay
        stance = 'N/A'
        essays_df.append({
            'topic_id': topic,
            'graph_id': graph_id,
            'stance': stance,
            'argument': argument
        })
    essays_df = ArgumentDataFrame(pd.DataFrame(essays_df))
    return essays_df


def load_arg_rewrite_df() -> pd.DataFrame:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_file_dir, 'arg_rewrite_stuff', 'ArgRewrite', 'essays')
    feedback_path = os.path.join(current_file_dir, 'arg_rewrite_stuff', 'ArgRewrite', 'meta-data', 'ExpertFeedbackTxt')
    draft1_path = os.path.join(file_path, 'Draft1')
    draft2_path = os.path.join(file_path, 'Draft2')
    draft3_path = os.path.join(file_path, 'Draft3')
    draft1_files = os.listdir(draft1_path)
    tmp = []
    for file in draft1_files:
        if not file.endswith('.txt'):
            continue
        draft1_file_name = file[len('draft1_2018'):]
        with open(os.path.join(draft1_path, file), 'r') as f:
            draft1_text = f.read().strip()
        draft2_file_name = 'draft2_2018' + draft1_file_name
        draft3_file_name = '2018' + draft1_file_name
        with open(os.path.join(draft2_path, draft2_file_name), 'r') as f:
            draft2_text = f.read().strip()
        with open(os.path.join(draft3_path, draft3_file_name), 'r') as f:
            draft3_text = f.read().strip()
        feedback_file = 'feedback_' + file[len('draft1_2018argrewrite_'): -len('.txt')] + '.txt'
        with open(os.path.join(feedback_path, feedback_file), 'r') as f:
            feedback_text = f.read().strip()
        tmp.append({
            'draft1': draft1_text,
            'draft2': draft2_text,
            'draft3': draft3_text,
            'feedback': feedback_text,
            'file': file
        })
    df = pd.DataFrame(tmp)
    return df


def prepare_revisions_df(
        revisions_df: pd.DataFrame
) -> tuple[ArgumentDataFrame, ArgumentDataFrame, ArgumentDataFrame]:
    draft1_args = []
    draft2_args = []
    draft3_args = []
    for i, row in revisions_df.iterrows():
        draft1_args.append({
            'topic_id': 'Self-driving cars',
            'graph_id': row['file'],
            'stance': 'N/A',
            'argument': row['draft1'],
            'feedback': row['feedback'],
            'draft1': row['draft1'],
            'draft2': row['draft2'],
            'draft3': row['draft3']
        })
        draft2_args.append({
            'topic_id': 'Self-driving cars',
            'graph_id': row['file'],
            'stance': 'N/A',
            'argument': row['draft2'],
            'feedback': row['feedback'],
            'draft1': row['draft1'],
            'draft2': row['draft2'],
            'draft3': row['draft3']
        })
        draft3_args.append({
            'topic_id': 'Self-driving cars',
            'graph_id': row['file'],
            'stance': 'N/A',
            'argument': row['draft3'],
            'feedback': row['feedback'],
            'draft1': row['draft1'],
            'draft2': row['draft2'],
            'draft3': row['draft3']
        })
    draft1_args = ArgumentDataFrame(pd.DataFrame(draft1_args))
    draft2_args = ArgumentDataFrame(pd.DataFrame(draft2_args))
    draft3_args = ArgumentDataFrame(pd.DataFrame(draft3_args))
    return draft1_args, draft2_args, draft3_args


class ArgumentImprovementPipeline:
    def __init__(
            self,
            argument_df: ArgumentDataFrame,
            model: str,
            force_vllm_preload: bool,
            trust_remote_code: bool,
            use_vllm: bool,
            improve_out_file_path: str,
            cleaned_out_file_path: str,
            format_func: callable,
            improved_arguments: Optional[list[tuple[str, list[tuple[str, str, str]]]]] = None,
            clean_prompt: Optional[str] = None,
            use_feedback: Optional[bool] = False
    ):
        self.argument_df = argument_df
        self.model = model
        self.force_vllm_preload = force_vllm_preload
        self.trust_remote_code = trust_remote_code
        self.use_vllm = use_vllm
        self.improve_out_file_path = improve_out_file_path
        self.cleaned_out_file_path = cleaned_out_file_path
        self.format_func = format_func
        self.improved_arguments = improved_arguments
        if clean_prompt is None:
            self.clean_prompt = 'You are given an improved argumentative text. The input may contain fluff that is not related to the argumentative text. Your task is to wrap the improved argumentative text in @ symbols. For example, if the argumentative text is "The sky is blue.", you should wrap it as "@The sky is blue.@". Ignore all other fluff and respond only with the improved argumentative text wrapped in @. If there is no argumentative text then respond with @INVALID@.'
        self.cleaned_arguments = None
        self.use_feedback = use_feedback

    def improve_arguments(self):
        if self.force_vllm_preload:
            os.environ['FORCE_VLLM_PRELOAD'] = '1'
            model_instance = LLM(
                model=self.model,
                dtype='half',
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=self.trust_remote_code,
                max_num_seqs=32
            )
        else:
            model_instance = None
        arg_improver = ArgumentImprover(
            self.argument_df,
            self.model,
            use_vllm=self.use_vllm,
            format_func=self.format_func,
            vllm_model_instance=model_instance,
            use_feedback=self.use_feedback
        )
        improved, out = arg_improver.do_all()
        with open(os.path.join(file_dir, self.improve_out_file_path), 'w') as f:
            json.dump(out, f, indent=4)
        self.improved_arguments = out
        return improved

    def clean_improved_arguments(self):
        direct = DirectPrompt(
            'llama3',
            format_func=llama3_format_func,
            use_vllm=False
        )
        cleaned_out = []
        arg_file = os.path.join(file_dir, self.improve_out_file_path)
        with open(arg_file, 'r') as f:
            data = json.load(f)
        for entry in data:
            approach = entry['approach']
            original_arguments = entry['original_arguments']
            topic_id = entry['topic_id']
            graph_id = entry['graph_id']
            stance = entry['stance']
            improved_arguments = entry['improved_arguments']
            bsm_formatted_branch_prompts = entry['bsm_formatted_branch_prompts']
            bsm_formatted_solve_prompts = entry['bsm_formatted_solve_prompts']
            bsm_formatted_merge_prompts = entry['bsm_formatted_merge_prompts']
            selfdiscover_select_prompt = entry['selfdiscover_select_prompt']
            selfdiscover_adapt_prompt = entry['selfdiscover_adapt_prompt']
            selfdiscover_implement_prompt = entry['selfdiscover_implement_prompt']
            selfdiscover_execute_prompt = entry['selfdiscover_execute_prompt']
            optimized_prompt = entry['optimized_prompt']
            assistant_prompt = f'Improved argument: @'
            prepared_prompts = []
            for improved_argument in improved_arguments:
                user_prompt = f'<|Improved argument|>: {improved_argument}'
                prompt = direct.format_prompt(self.clean_prompt, user_prompt, assistant_prompt)
                prepared_prompts.append(prompt)
            cleaned = direct.do_prompting(prepared_prompts)
            cleaned_arguments = []
            for clean in cleaned:
                solution = '@' + clean
                try:
                    tmp = solution.split('@')
                    # take longest element
                    solution_parsed = max(tmp, key=len)
                except IndexError:
                    solution_parsed = ''
                cleaned_arguments.append(solution_parsed)
            tmp = {
                'approach': approach,
                'original_arguments': original_arguments,
                'improved_arguments': improved_arguments,
                'cleaned_arguments': cleaned_arguments,
                'topic_id': topic_id,
                'graph_id': graph_id,
                'stance': stance,
                'bsm_formatted_branch_prompts': bsm_formatted_branch_prompts,
                'bsm_formatted_solve_prompts': bsm_formatted_solve_prompts,
                'bsm_formatted_merge_prompts': bsm_formatted_merge_prompts,
                'selfdiscover_select_prompt': selfdiscover_select_prompt,
                'selfdiscover_adapt_prompt': selfdiscover_adapt_prompt,
                'selfdiscover_implement_prompt': selfdiscover_implement_prompt,
                'selfdiscover_execute_prompt': selfdiscover_execute_prompt,
                'optimized_prompt': optimized_prompt
            }
            # add all the other columns from data to cleaned_out that dont already exist
            for key, value in entry.items():
                if key not in tmp:
                    tmp[key] = value
            cleaned_out.append(tmp)
        self.cleaned_arguments = cleaned_out
        with open(os.path.join(file_dir, self.cleaned_out_file_path), 'w') as f:
            json.dump(cleaned_out, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='The model to use for the improvement pipeline.'
    )
    parser.add_argument(
        '--force-vllm-preload',
        action='store_true',
        help='Whether to force the VLLM model to preload.'
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Whether to trust remote code.'
    )
    parser.add_argument(
        '--use-vllm',
        action='store_true',
        help='Whether to use VLLM.'
    )
    parser.add_argument(
        '--improve-out-file-path',
        type=str,
        required=True,
        help='The file path to save the improved arguments.'
    )
    parser.add_argument(
        '--cleaned-out-file-path',
        type=str,
        required=True,
        help='The file path to save the cleaned arguments.'
    )
    parser.add_argument(
        '--format-func',
        type=int,
        required=True,
        help='The format function to use for the improvement pipeline.'
    )
    parser.add_argument(
        '--dataset',
        type=int,
        required=True,
        help='The dataset to use for the improvement pipeline.'
    )

    args = parser.parse_args()
    _use_feedback = False

    format_func_ = FORMAT_FUNCTIONS[args.format_func]
    dataset = DATASETS[args.dataset]
    if dataset == SupportedDatasets.ESSAYS:
        _essay_df = prepare_essays_df()
    elif dataset == SupportedDatasets.REVISIONS_DRAFT1:
        _revisions_df = load_arg_rewrite_df()
        _essay_df, _, _ = prepare_revisions_df(_revisions_df)
    elif dataset == SupportedDatasets.REVISIONS_DRAFT2:
        _revisions_df = load_arg_rewrite_df()
        _, _essay_df, _ = prepare_revisions_df(_revisions_df)
    elif dataset == SupportedDatasets.REVISIONS_DRAFT3:
        _revisions_df = load_arg_rewrite_df()
        _, _, _essay_df = prepare_revisions_df(_revisions_df)
    elif dataset == SupportedDatasets.MICROTEXTS:
        _essay_df = prepare_microtexts_both_df()
    elif dataset == SupportedDatasets.REVISIONS_DRAFT1_FEEDBACK:
        _revisions_df = load_arg_rewrite_df()
        _essay_df, _, _ = prepare_revisions_df(_revisions_df)
        _use_feedback = True
    elif dataset == SupportedDatasets.REVISIONS_DRAFT2_FEEDBACK:
        _revisions_df = load_arg_rewrite_df()
        _, _essay_df, _ = prepare_revisions_df(_revisions_df)
        _use_feedback = True
    elif dataset == SupportedDatasets.REVISIONS_DRAFT3_FEEDBACK:
        _revisions_df = load_arg_rewrite_df()
        _, _, _essay_df = prepare_revisions_df(_revisions_df)
        _use_feedback = True
    else:
        raise ValueError('Invalid dataset.')

    _pipeline = ArgumentImprovementPipeline(
        argument_df=_essay_df,
        model=args.model,
        force_vllm_preload=args.force_vllm_preload,
        trust_remote_code=args.trust_remote_code,
        use_vllm=args.use_vllm,
        improve_out_file_path=args.improve_out_file_path,
        cleaned_out_file_path=args.cleaned_out_file_path,
        format_func=format_func_,
        use_feedback=_use_feedback
    )
    print('> Starting improve, with df length:', len(_essay_df))
    _pipeline.improve_arguments()
    _pipeline.clean_improved_arguments()

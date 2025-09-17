import enum
import json
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.errors import InvalidIndexError
from scipy.stats import kruskal, pearsonr
from pyfiglet import Figlet
import warnings

from scipy.stats._axis_nan_policy import SmallSampleWarning

from projects.improvement_pipeline.improvement_pipeline import load_arg_rewrite_df, prepare_essays_df, \
    prepare_microtexts_both_df
from projects.text_generation_scores.americano_scores import extract_first_number

warnings.simplefilter('ignore', SmallSampleWarning)
warnings.filterwarnings('ignore')

models = [
    'bloomz-3b-cleaned',
    'bloomz-560m-cleaned',
    'llama3_nemotron_cleaned',
    'OLMo-7B-0724-Instruct-hf-cleaned',
    'Phi-3-medium-4k-instruct-cleaned',
    'Phi-3-mini-4k-instruct-cleaned'
]

approaches = [
    'direct',
    'bsm',
    'ga',
    'little_brother',
    'self_discover'
]

datasets = [
    'revision1__REVISIONS',
    'revision2__REVISIONS',
    'revision3__REVISIONS',
    '_ESSAYS',
    '_MICROTEXTS',
]

current_file_dir = os.path.dirname(os.path.abspath(__file__))


def get_revision_lengths():
    revision_df = load_arg_rewrite_df()
    draft1_lens = revision_df['draft1'].apply(lambda x: len(x))
    draft2_lens = revision_df['draft2'].apply(lambda x: len(x))
    draft3_lens = revision_df['draft3'].apply(lambda x: len(x))
    return draft1_lens, draft2_lens, draft3_lens


def get_essays_lengths():
    essay_df = prepare_essays_df()
    lens = essay_df['argument'].apply(lambda x: len(x))
    return lens


def get_microtext_lengths():
    microtexts_df = prepare_microtexts_both_df()
    lens = microtexts_df['argument'].apply(lambda x: len(x))
    return lens


def get_improved_lens_all():
    _files_all = [
        ('MICROTEXTS', 'bloomz-3b-cleaned.json'),
        ('MICROTEXTS', 'bloomz-560m-cleaned.json'),
        ('MICROTEXTS', 'llama3_nemotron_cleaned.json'),
        ('MICROTEXTS', 'OLMo-7B-0724-Instruct-hf-cleaned.json'),
        ('MICROTEXTS', 'Phi-3-medium-4k-instruct-cleaned.json'),
        ('MICROTEXTS', 'Phi-3-mini-4k-instruct-cleaned.json'),
        ('ESSAYS', 'bloomz-3b-cleaned.json'),
        ('ESSAYS', 'bloomz-560m-cleaned.json'),
        ('ESSAYS', 'llama3_nemotron_cleaned.json'),
        ('ESSAYS', 'OLMo-7B-0724-Instruct-hf-cleaned.json'),
        ('ESSAYS', 'Phi-3-medium-4k-instruct-cleaned.json'),  # this was missing, now exists
        ('ESSAYS', 'Phi-3-mini-4k-instruct-cleaned.json'),
        ('REVISIONS', 'bloomz-3b-cleaned_revision1.json'),
        ('REVISIONS', 'bloomz-3b-cleaned_revision2.json'),
        ('REVISIONS', 'bloomz-3b-cleaned_revision3.json'),
        ('REVISIONS', 'bloomz-560m-cleaned_revision1.json'),
        ('REVISIONS', 'bloomz-560m-cleaned_revision2.json'),
        ('REVISIONS', 'bloomz-560m-cleaned_revision3.json'),
        ('REVISIONS', 'llama3_nemotron_cleaned_revision1.json'),
        ('REVISIONS', 'llama3_nemotron_cleaned_revision2.json'),
        ('REVISIONS', 'llama3_nemotron_cleaned_revision3.json'),  # this was missing, now exists
        ('REVISIONS', 'OLMo-7B-0724-Instruct-hf-cleaned_revision1.json'),
        ('REVISIONS', 'OLMo-7B-0724-Instruct-hf-cleaned_revision2.json'),
        ('REVISIONS', 'OLMo-7B-0724-Instruct-hf-cleaned_revision3.json'),
        ('REVISIONS', 'Phi-3-medium-4k-instruct-cleaned_revision1.json'),
        ('REVISIONS', 'Phi-3-medium-4k-instruct-cleaned_revision2.json'),
        ('REVISIONS', 'Phi-3-medium-4k-instruct-cleaned_revision3.json'),
        ('REVISIONS', 'Phi-3-mini-4k-instruct-cleaned_revision1.json'),
        ('REVISIONS', 'Phi-3-mini-4k-instruct-cleaned_revision2.json'),
        ('REVISIONS', 'Phi-3-mini-4k-instruct-cleaned_revision3.json'),
    ]
    out = {}
    empties = {}
    for folder, file_name in _files_all:
        _test_file = os.path.join(current_file_dir, '..', 'improved_out', folder, file_name)
        try:
            with open(_test_file, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            with open(f'{_test_file}BACKUP', 'r') as file:
                data = json.load(file)
        if folder != 'REVISIONS':
            model_name = file_name[:-len('.json')]
            if folder == 'MICROTEXTS':
                dataset_name = '_MICROTEXTS'
            else:
                dataset_name = '_ESSAYS'
        else:
            model_name = file_name[:-len('_revision1.json')]
            if 'revision1' in file_name:
                dataset_name = 'revision1__REVISIONS'
            elif 'revision2' in file_name:
                dataset_name = 'revision2__REVISIONS'
            else:
                dataset_name = 'revision3__REVISIONS'
        for approach in data:
            approach_name = approach['approach']
            improved = approach['improved_arguments']
            empty_improved = [i for i, x in enumerate(improved) if not x.strip()]
            improved_lens = [len(i) for i in improved]
            if approach_name not in out:
                out[approach_name] = {}
            if dataset_name not in out[approach_name]:
                out[approach_name][dataset_name] = {}
            if approach_name not in empties:
                empties[approach_name] = {}
            if dataset_name not in empties[approach_name]:
                empties[approach_name][dataset_name] = {}
            out[approach_name][dataset_name][model_name] = improved_lens
            empties[approach_name][dataset_name][model_name] = empty_improved
    return out, empties


essays_lens = get_essays_lengths()
microtexts_lens = get_microtext_lengths()
revision1_lens, revision2_lens, revision3_lens = get_revision_lengths()

DATASET_LENS = [
    revision1_lens,
    revision2_lens,
    revision3_lens,
    essays_lens,
    microtexts_lens
]


def calculate_kruskal_score(
        dfs: list[pd.DataFrame]
):
    cols = dfs[0].columns
    significance_counts = {col: 0 for col in cols}
    effect_sizes = {col: [] for col in cols}
    for col in cols:
        data = [df[col].values for df in dfs]
        try:
            stat, p_value = kruskal(*data)
        except ValueError as e:
            # print(f'Error for {col}: {e}')
            # happens when all values are equal
            continue
        if p_value < 0.05:
            N = sum(len(group) for group in data)
            k = len(data)
            eta_squared = (stat - k + 1) / (N - k)
            effect_sizes[col].append(eta_squared)
            significance_counts[col] += 1
    avg_effect_sizes = {col: (sum(effect_sizes[col]) / len(effect_sizes[col]) if effect_sizes[col] else 0) for col in
                        cols}
    # for col in cols:
    #     print(f'Column: {col}, Significance: {significance_counts[col]}, Avg. Effect Size: {avg_effect_sizes[col]}')
    return significance_counts, avg_effect_sizes


def metrics_analysis(
        metric_identifier: str,
        feng_hirst_ngram_limit: int = 1,
        do_deltas: Optional[bool] = True
):
    comparison_dfs_all = []
    corpus_dfs_all = []
    correlations = {}
    model_lens, empties = get_improved_lens_all()
    for approach in approaches:
        for dataset, lens in zip(datasets, DATASET_LENS):
            comparison_delta_dfs = []
            corpus_dfs = []
            for model in models:
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')
                    comparison_delta_dfs.append(None)
                    corpus_dfs.append(None)
                    continue
                comparison_file_name = f'{model}_{dataset}__{approach}_comparison.csv'
                corpus_file_name = f'{model}_{dataset}__{approach}_corpus.csv'
                try:
                    comparison_df = pd.read_csv(
                        os.path.join(current_file_dir, 'scores_out', metric_identifier, comparison_file_name),
                        index_col=0
                    )
                except FileNotFoundError:
                    print(
                        '> remove this later, some files are missing. should only be phi-3-medium stuff, currently generating',
                        comparison_file_name)
                    comparison_delta_dfs.append(None)
                    corpus_dfs.append(None)
                    continue
                # do syllable column fix here - original/improved may have different max syllable counts
                # this leads to missing columns, for original/improved
                try:
                    max_syllables_original = max(
                        [int(col[len('linguaf_number_of_'):-len('_syllable_words_original')]) for col in
                         comparison_df.columns if 'syllable_words_original' in col])
                except ValueError:
                    # no syllable columns
                    max_syllables_original = 0
                try:
                    max_syllables_improved = max(
                        [int(col[len('linguaf_number_of_'):-len('_syllable_words_improved')]) for col in
                         comparison_df.columns if 'syllable_words_improved' in col])
                except ValueError:
                    # no syllable columns
                    max_syllables_improved = 0
                if max_syllables_original > max_syllables_improved:
                    # need to fill up improved columns
                    for i in range(max_syllables_improved + 1, max_syllables_original + 1):
                        comparison_df[f'linguaf_number_of_{i}_syllable_words_improved'] = 0
                elif max_syllables_improved > max_syllables_original:
                    # need to fill up original columns
                    for i in range(max_syllables_original + 1, max_syllables_improved + 1):
                        comparison_df[f'linguaf_number_of_{i}_syllable_words_original'] = 0

                corpus_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', metric_identifier, corpus_file_name), index_col=0
                )
                corpus_df_before = corpus_df.copy()
                comparison_df_before = comparison_df.copy()
                # for e in empty_json:
                #     # insert empty row at index e
                #     empty_row = pd.DataFrame([np.nan] * comparison_df.shape[1],
                #                              index=comparison_df.columns).T
                #     comparison_df = pd.concat(
                #         [comparison_df.iloc[:e], empty_row, comparison_df.iloc[e:]]
                #     ).reset_index(drop=True)
                #
                #     empty_row = pd.DataFrame([np.nan] * corpus_df.shape[1], index=corpus_df.columns).T
                #     corpus_df = pd.concat(
                #         [corpus_df.iloc[:e], empty_row, corpus_df.iloc[e:]]
                #     ).reset_index(drop=True)
                relevant_lens = model_lens[approach][dataset][model]
                zero_indices = empties[approach][dataset][model]
                if approach == 'ga' and 'revision3' in dataset and 'medium' in model:
                    zero_indices = [18, 46]  # manual check

                for i in zero_indices:
                    empty_row = pd.DataFrame([np.nan] * comparison_df.shape[1],
                                             index=comparison_df.columns).T
                    comparison_df = pd.concat(
                        [comparison_df.iloc[:i], empty_row, comparison_df.iloc[i:]]
                    ).reset_index(drop=True)

                    empty_row = pd.DataFrame([np.nan] * corpus_df.shape[1], index=corpus_df.columns).T
                    corpus_df = pd.concat(
                        [corpus_df.iloc[:i], empty_row, corpus_df.iloc[i:]]
                    ).reset_index(drop=True)

                deltas = []
                percentage_changes = []
                comparison_cols = comparison_df.columns
                if metric_identifier == 'FENG_HIRST_SCORES':
                    comparison_cols = [x for x in comparison_cols if not x.count(';') > feng_hirst_ngram_limit]

                for col in comparison_cols:
                    if col.endswith('_improved'):
                        original_col = col.replace('_improved', '_original')
                        deltas.append((col.replace('_improved', ''), comparison_df[col] - comparison_df[original_col]))
                        tmp = (comparison_df[col] - comparison_df[original_col]) / comparison_df[original_col]
                        percentage_changes.append((col.replace('_improved', ''), tmp))

                if do_deltas:
                    # print('> Doing deltas')
                    deltas = deltas
                else:
                    # print('> Doing percentage changes')
                    deltas = percentage_changes
                comparison_deltas_df = pd.DataFrame({k: v for k, v in deltas})
                comparison_deltas_df['length'] = lens
                try:
                    comparison_deltas_df['improved_length'] = model_lens[approach][dataset][model]
                except ValueError:
                    print(f'> ERROR FOR MODEL ', model, approach, dataset)
                    break
                corpus_df['length'] = lens
                corpus_df['improved_length'] = model_lens[approach][dataset][model]

                for col in comparison_deltas_df.columns:
                    if col == 'length':
                        continue
                    corr = comparison_deltas_df['length'].corr(comparison_deltas_df[col])
                    correlations[(model, dataset, approach, col)] = corr
                for col in corpus_df.columns:
                    if col == 'length':
                        continue
                    corr = corpus_df['length'].corr(corpus_df[col])
                    correlations[(model, dataset, approach, col)] = corr
                comparison_delta_dfs.append(comparison_deltas_df)
                corpus_dfs.append(corpus_df)
            comparison_dfs_all.append((approach, dataset, comparison_delta_dfs, models))
            corpus_dfs_all.append((approach, dataset, corpus_dfs, models))

    all_dfs = []

    # After the loops where you build `comparison_deltas_df`
    all_dfs.append(comparison_deltas_df)

    # Then concatenate
    full_df = pd.concat(all_dfs, axis=0)

    # Compute correlation
    correlation_series = full_df.corr(numeric_only=True)['length']
    print(correlation_series)

    max_corr = max(correlations.items(), key=lambda k: k[1])
    min_corr = min(correlations.items(), key=lambda k: k[1])
    print(f'Max correlation: {max_corr[0]}: {max_corr[1]}')
    print(f'Min correlation: {min_corr[0]}: {min_corr[1]}')

    corrs_sorted = sorted(correlations.items(), key=lambda k: k[1], reverse=True)

    comparion_score_names = comparison_dfs_all[0][2][0].columns
    corpus_score_names = corpus_dfs_all[0][2][0].columns
    # for approach, dataset, comparison_delta_dfs, models_ in comparison_dfs_all:
    #     for m_, df in zip(models_, comparison_delta_dfs):
    #         if df is None:
    #             continue
    #         df.to_csv(os.path.join(current_file_dir, 'csvs',
    #                                f'{m_}_{dataset}__{approach}_{metric_identifier}_comparison_deltas.csv'))
    # for approach, dataset, corpus_dfs, models_ in corpus_dfs_all:
    #     for m_, df in zip(models_, corpus_dfs):
    #         if df is None:
    #             continue
    #         df.to_csv(
    #             os.path.join(current_file_dir, 'csvs', f'{m_}_{dataset}__{approach}_{metric_identifier}_corpus.csv'))

    return comparion_score_names, corpus_score_names, comparison_dfs_all, corpus_dfs_all


def bertalign_analysis():
    comparison_dfs_all = []
    corpus_dfs_all = []
    correlations = {}
    model_lens = get_improved_lens_all()
    revision_lens = get_revision_lengths()

    for approach in approaches:
        for dataset, lens in zip(datasets, DATASET_LENS):
            comparison_dfs = []
            bertalign_corpus_dfs = []
            for model in models:
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')
                    comparison_dfs.append(None)
                    bertalign_corpus_dfs.append(None)
                    continue
                # load bertalign scores
                bertalign_comparison = f'{model}_{dataset}__{approach}_comparison.csv'
                bertalign_corpus = f'{model}_{dataset}__{approach}_corpus.csv'
                bertalign_comparison_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'BERTALIGN_SCORES', bertalign_comparison), index_col=0)
                bertalign_corpus_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'BERTALIGN_SCORES', bertalign_corpus), index_col=0)

                for e in empty_json:
                    # insert empty row at index e
                    empty_row = pd.DataFrame([np.nan] * bertalign_comparison_df.shape[1],
                                             index=bertalign_comparison_df.columns).T
                    bertalign_comparison_df = pd.concat(
                        [bertalign_comparison_df.iloc[:e], empty_row, bertalign_comparison_df.iloc[e:]]).reset_index(
                        drop=True)

                    empty_row = pd.DataFrame([np.nan] * bertalign_corpus_df.shape[1],
                                             index=bertalign_corpus_df.columns).T
                    bertalign_corpus_df = pd.concat(
                        [bertalign_corpus_df.iloc[:e], empty_row, bertalign_corpus_df.iloc[e:]]).reset_index(drop=True)

                deltas = []
                # calculate delta between _original and _improved columns
                for col in bertalign_comparison_df.columns:
                    if col.endswith('_improved'):
                        original_col = col.replace('_improved', '_original')
                        deltas.append((col.replace('_improved', ''),
                                       bertalign_comparison_df[col] - bertalign_comparison_df[original_col]))
                bertalign_comparison_deltas_df = pd.DataFrame({k: v for k, v in deltas})

                # add lens as column
                bertalign_comparison_deltas_df['length'] = lens
                bertalign_comparison_deltas_df['improved_length'] = model_lens[approach][dataset][model]
                # add lens as column
                bertalign_corpus_df['length'] = lens
                bertalign_corpus_df['improved_length'] = model_lens[approach][dataset][model]

                # calculate correlation between length and all other columns
                for col in bertalign_comparison_deltas_df.columns:
                    if col == 'length':
                        continue
                    corr = bertalign_comparison_deltas_df['length'].corr(bertalign_comparison_deltas_df[col])
                    correlations[(model, dataset, approach, col)] = corr
                for col in bertalign_corpus_df.columns:
                    if col == 'length':
                        continue
                    corr = bertalign_corpus_df['length'].corr(bertalign_corpus_df[col])
                    correlations[(model, dataset, approach, col)] = corr

                comparison_dfs.append(bertalign_comparison_deltas_df)
                bertalign_corpus_dfs.append(bertalign_corpus_df)
            # do analysis
            # significance_counts, avg_effect_sizes = calculate_kruskal_score([x.dropna() for x in comparison_dfs])
            comparison_dfs_all.append((approach, dataset, comparison_dfs, models))
            corpus_dfs_all.append((approach, dataset, bertalign_corpus_dfs, models))
    print('foo?')


def feng_hirst_analysis():
    for approach in approaches:
        for model in models:
            for dataset in datasets:
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')
                # load feng hirst scores
                feng_hirst_comparison = f'{model}_{dataset}__{approach}_comparison.csv'
                feng_hirst_corpus = f'{model}_{dataset}__{approach}_corpus.csv'
                feng_hirst_comparison_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'FENG_HIRST_SCORES', feng_hirst_comparison),
                    index_col=0)
                feng_hirst_corpus_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'FENG_HIRST_SCORES', feng_hirst_corpus), index_col=0)
                for e in empty_json:
                    # insert empty row at index e
                    empty_row = pd.DataFrame([np.nan] * feng_hirst_comparison_df.shape[1],
                                             index=feng_hirst_comparison_df.columns).T
                    feng_hirst_comparison_df = pd.concat(
                        [feng_hirst_comparison_df.iloc[:e], empty_row, feng_hirst_comparison_df.iloc[e:]]).reset_index(
                        drop=True)

                    empty_row = pd.DataFrame([np.nan] * feng_hirst_corpus_df.shape[1],
                                             index=feng_hirst_corpus_df.columns).T
                    feng_hirst_corpus_df = pd.concat(
                        [feng_hirst_corpus_df.iloc[:e], empty_row, feng_hirst_corpus_df.iloc[e:]]).reset_index(
                        drop=True)

                deltas = []
                # calculate delta between _original and _improved columns
                for col in feng_hirst_comparison_df.columns:
                    if col.endswith('_improved'):
                        original_col = col.replace('_improved', '_original')
                        deltas.append((col.replace('_improved', ''),
                                       feng_hirst_comparison_df[col] - feng_hirst_comparison_df[original_col]))
                feng_hirst_comparison_deltas_df = pd.DataFrame({k: v for k, v in deltas})


def gruen_analysis():
    for approach in approaches:
        for model in models:
            for dataset in datasets:
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')

                # load gruen scores
                gruen_comparison = f'{model}_{dataset}__{approach}_comparison.csv'
                gruen_corpus = f'{model}_{dataset}__{approach}_corpus.csv'
                gruen_comparison_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'GRUEN_SCORES', gruen_comparison), index_col=0)
                gruen_corpus_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'GRUEN_SCORES', gruen_corpus),
                    index_col=0)


def levenshtein_analysis():
    for approach in approaches:
        for model in models:
            for dataset in datasets:
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')
                # load levenshtein scores
                levenshtein_comparison = f'{model}_{dataset}__{approach}_comparison.csv'
                levenshtein_corpus = f'{model}_{dataset}__{approach}_corpus.csv'
                levenshtein_comparison_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'LEVENSHTEIN_SCORES', levenshtein_comparison),
                    index_col=0)
                levenshtein_corpus_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'LEVENSHTEIN_SCORES', levenshtein_corpus), index_col=0)


def lexical_analysis():
    for approach in approaches:
        for model in models:
            for dataset in datasets:
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')
                # load lexical scores
                lexical_comparison = f'{model}_{dataset}__{approach}_comparison.csv'
                lexical_corpus = f'{model}_{dataset}__{approach}_corpus.csv'
                lexical_comparison_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'LEXICAL_SCORES', lexical_comparison), index_col=0)
                lexical_corpus_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'LEXICAL_SCORES', lexical_corpus), index_col=0)


def linguaf_analysis():
    for approach in approaches:
        for model in models:
            for dataset in datasets:
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')
                # load linguaf scores
                linguaf_comparison = f'{model}_{dataset}__{approach}_comparison.csv'
                linguaf_corpus = f'{model}_{dataset}__{approach}_corpus.csv'
                linguaf_syllable_scores = f'{model}_{dataset}__{approach}_syllable_changes.json'
                with open(os.path.join(current_file_dir, 'scores_out', 'LINGUAF_SCORES', linguaf_syllable_scores),
                          'r') as f:
                    linguaf_syllable_changes = json.loads(f.read())
                linguaf_comparison_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'LINGUAF_SCORES', linguaf_comparison), index_col=0)
                linguaf_corpus_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'LINGUAF_SCORES', linguaf_corpus), index_col=0)


def semantic_similarity_analysis():
    for approach in approaches:
        for model in models:
            for dataset in datasets:
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')
                # load semantic similarity scores
                semantic_similarity_comparison = f'{model}_{dataset}__{approach}_comparison.csv'
                semantic_similarity_corpus = f'{model}_{dataset}__{approach}_corpus.csv'
                semantic_similarity_comparison_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'SEMANTIC_SIMILARITY_SCORES',
                                 semantic_similarity_comparison), index_col=0)
                semantic_similarity_corpus_df = pd.read_csv(
                    os.path.join(current_file_dir, 'scores_out', 'SEMANTIC_SIMILARITY_SCORES',
                                 semantic_similarity_corpus),
                    index_col=0)


def sentiment_analysis_effect_size():
    # analysis with deltas and effect size - not used for the linguistic level analysis
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__essays_sentiment_sentiment.json'), 'r') as f:
        human_essay_sentiment_scores = json.loads(f.read())
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__human_draft1_original_sentiment_sentiment.json'), 'r') as f:
        human_revision1_sentiment_scores = json.loads(f.read())
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__human_draft2_improved_sentiment_sentiment.json'), 'r') as f:
        human_revision2_sentiment_scores = json.loads(f.read())
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__human_draft3_improved_sentiment_sentiment.json'), 'r') as f:
        human_revision3_sentiment_scores = json.loads(f.read())
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__microtexts_sentiment_sentiment.json'), 'r') as f:
        human_microtext_sentiment_scores = json.loads(f.read())

    human_scores = [
        human_revision1_sentiment_scores,
        human_revision2_sentiment_scores,
        human_revision3_sentiment_scores,
        human_essay_sentiment_scores,
        human_microtext_sentiment_scores
    ]

    delta_cols_english = [
        'polarity',
        'subjectivity'
    ]
    delta_cols_german = [
        'german_proba_positive',
        'german_proba_negative',
        'german_proba_neutral'
    ]

    dfs_english_all = []
    dfs_german_all = []

    for approach in approaches:
        for model in models:
            f = Figlet(font='slant')
            print(f.renderText(f'{model}   {approach}'))
            sentiment_scores_model = {}
            for dataset, hs in zip(datasets, human_scores):
                english_identifier = f'{dataset}-english'
                german_identifier = f'{dataset}-german'
                sentiment_scores_model[english_identifier] = {}
                sentiment_scores_model[german_identifier] = {}
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')
                sentiment_file = f'{model}_{dataset}__{approach}_improved_sentiment_sentiment.json'

                try:
                    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES', sentiment_file),
                              'r') as f:
                        sentiment_scores = json.loads(f.read())
                except FileNotFoundError:
                    print(f'> MISSING FILE: {sentiment_file}')
                    continue

                sentiment_df = pd.DataFrame(sentiment_scores)
                human_sentiment_df = pd.DataFrame(hs)

                english_model_data = sentiment_df[~sentiment_df['is_german']][delta_cols_english]
                german_model_data = sentiment_df[sentiment_df['is_german']][delta_cols_german]

                english_human_data = human_sentiment_df[~human_sentiment_df['is_german']][delta_cols_english]
                german_human_data = human_sentiment_df[human_sentiment_df['is_german']][delta_cols_german]

                deltas_english = english_model_data - english_human_data
                deltas_german = german_model_data - german_human_data

                sc_e, avg_es_e = calculate_kruskal_score([english_model_data, english_human_data])
                sc_g, avg_es_g = calculate_kruskal_score([german_model_data, german_human_data])

                print('==================================')
                print_flag_e = False
                for k, v in sc_e.items():
                    # if v:
                    #     if not print_flag_e:
                    #         print(f'> Effect size, significance count for {model}, {dataset}, {approach} - ENGLISH <')
                    #         print_flag_e = True
                    #     print(f'{k:<22}: {v:>5.2f} - {avg_es_e[k]:>5.2f}')
                    sentiment_scores_model[english_identifier][f'{k}_significance'] = v
                    sentiment_scores_model[english_identifier][f'{k}_effect_size'] = avg_es_e[k]
                for k, v in deltas_english.sum().to_dict().items():
                    sentiment_scores_model[english_identifier][f'{k}_delta'] = v

                print_flag_g = False
                for k, v in sc_g.items():
                    # if v:
                    #     if not print_flag_g:
                    #         print(f'\n> Effect size, significance count for {model}, {dataset}, {approach} - GERMAN <')
                    #         print_flag_g = True
                    #     print(f'{k:<22}: {v:>5.2f} - {avg_es_g[k]:>5.2f}')
                    sentiment_scores_model[german_identifier][f'{k}_significance'] = v
                    sentiment_scores_model[german_identifier][f'{k}_effect_size'] = avg_es_g[k]
                for k, v in deltas_german.sum().to_dict().items():
                    sentiment_scores_model[german_identifier][f'{k}_delta'] = v
                if not print_flag_e and not print_flag_g:
                    print(f'> No significant differences for {model}, {dataset}, {approach}')
                print('==================================\n\n')

            english_rows = []
            german_rows = []
            for k, v in sentiment_scores_model.items():
                if 'german' in k and not 'MICROTEXTS' in k:
                    continue
                if 'english' in k:
                    if not v:
                        v['polarity_significance'] = np.nan
                        v['polarity_effect_size'] = np.nan
                        v['subjectivity_significance'] = np.nan
                        v['subjectivity_effect_size'] = np.nan
                        v['polarity_delta'] = np.nan
                        v['subjectivity_delta'] = np.nan
                    tmp = (k, v['polarity_significance'], v['polarity_effect_size'], v['subjectivity_significance'],
                           v['subjectivity_effect_size'], v['polarity_delta'], v['subjectivity_delta'])
                    english_rows.append(tmp)
                elif 'german' in k:
                    if not v:
                        v['german_proba_positive_significance'] = np.nan
                        v['german_proba_positive_effect_size'] = np.nan
                        v['german_proba_negative_significance'] = np.nan
                        v['german_proba_negative_effect_size'] = np.nan
                        v['german_proba_neutral_significance'] = np.nan
                        v['german_proba_neutral_effect_size'] = np.nan
                        v['german_proba_positive_delta'] = np.nan
                        v['german_proba_negative_delta'] = np.nan
                        v['german_proba_neutral_delta'] = np.nan
                    tmp = (k, v['german_proba_positive_significance'], v['german_proba_positive_effect_size'],
                           v['german_proba_negative_significance'], v['german_proba_negative_effect_size'],
                           v['german_proba_neutral_significance'], v['german_proba_neutral_effect_size'],
                           v['german_proba_positive_delta'], v['german_proba_negative_delta'],
                           v['german_proba_neutral_delta'])
                    german_rows.append(tmp)

            # make dataframe out of tmp
            english_df = pd.DataFrame(english_rows, columns=['dataset', 'polarity_significance', 'polarity_effect_size',
                                                             'subjectivity_significance', 'subjectivity_effect_size',
                                                             'polarity_delta', 'subjectivity_delta'])
            german_df = pd.DataFrame(german_rows, columns=['dataset', 'german_proba_positive_significance',
                                                           'german_proba_positive_effect_size',
                                                           'german_proba_negative_significance',
                                                           'german_proba_negative_effect_size',
                                                           'german_proba_neutral_significance',
                                                           'german_proba_neutral_effect_size',
                                                           'german_proba_positive_delta', 'german_proba_negative_delta',
                                                           'german_proba_neutral_delta'])
            dfs_english_all.append((approach, model, english_df))
            dfs_german_all.append((approach, model, german_df))
    polarity_deltas = [x[2]['polarity_delta'] for x in dfs_english_all]
    # concat all series together
    polarity_deltas_concat = pd.concat(polarity_deltas, axis=0)
    print(polarity_deltas_concat.describe())

    german_concat = pd.concat([x[2] for x in dfs_german_all], axis=0)
    print(german_concat.dropna().describe())


def sentiment_analysis():
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__essays_sentiment_sentiment.json'), 'r') as f:
        human_essay_sentiment_scores = json.loads(f.read())
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__human_draft1_original_sentiment_sentiment.json'), 'r') as f:
        human_revision1_sentiment_scores = json.loads(f.read())
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__human_draft2_improved_sentiment_sentiment.json'), 'r') as f:
        human_revision2_sentiment_scores = json.loads(f.read())
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__human_draft3_improved_sentiment_sentiment.json'), 'r') as f:
        human_revision3_sentiment_scores = json.loads(f.read())
    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES',
                           'HUMAN_SENTIMENT__microtexts_sentiment_sentiment.json'), 'r') as f:
        human_microtext_sentiment_scores = json.loads(f.read())

    human_scores = [
        human_revision1_sentiment_scores,
        human_revision2_sentiment_scores,
        human_revision3_sentiment_scores,
        human_essay_sentiment_scores,
        human_microtext_sentiment_scores
    ]

    delta_cols_english = [
        'polarity',
        'subjectivity'
    ]
    delta_cols_german = [
        'german_proba_positive',
        'german_proba_negative',
        'german_proba_neutral'
    ]

    dfs_english_all = []
    dfs_german_all = []

    scores_human_all_german = []
    scores_human_all_english = []
    scores_model_all_german = []
    scores_model_all_english = []

    for approach in approaches:
        for dataset, hs in zip(datasets, human_scores):
            sentiment_scores_model = {}
            tmp_e_h = []
            tmp_g_h = []
            tmp_e_m = []
            tmp_g_m = []
            for model in models:
                english_identifier = f'{dataset}-english'
                german_identifier = f'{dataset}-german'
                sentiment_scores_model[english_identifier] = {}
                sentiment_scores_model[german_identifier] = {}
                empty_file = f'{model}_{dataset}__{approach}_empty_json'
                try:
                    with open(os.path.join(current_file_dir, 'scores_out', empty_file), 'r') as f:
                        empty_json = json.loads(f.read())
                except FileNotFoundError:
                    # TODO: generate and score the missing ones
                    print(f'Not found: {empty_file}')
                sentiment_file = f'{model}_{dataset}__{approach}_improved_sentiment_sentiment.json'

                try:
                    with open(os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES', sentiment_file),
                              'r') as f:
                        sentiment_scores = json.loads(f.read())
                except FileNotFoundError:
                    print(f'> MISSING FILE: {sentiment_file}')
                    continue

                sentiment_df = pd.DataFrame(sentiment_scores)
                human_sentiment_df = pd.DataFrame(hs)

                for e in empty_json:
                    # add empty row at index e to sentiment_df only
                    empty_row = pd.DataFrame([None] * sentiment_df.shape[1],
                                             index=sentiment_df.columns).T
                    # get value of is_german for two previous row
                    is_german = sentiment_df.iloc[e - 1]['is_german']
                    # get majority
                    if e > 1:
                        is_german = sentiment_df.iloc[e - 2:e]['is_german'].mode().values[0]
                    empty_row['is_german'] = is_german
                    sentiment_df = pd.concat(
                        [sentiment_df.iloc[:e], empty_row, sentiment_df.iloc[e:]]
                    ).reset_index(drop=True)

                # original is the HUMAN data
                # only improved is used for model - scores are measured, NOT the deltas!
                # get deltas by doing (improved - human)
                english_model_data = sentiment_df[~sentiment_df['is_german']][delta_cols_english]
                german_model_data = sentiment_df[sentiment_df['is_german']][delta_cols_german]

                english_human_data = human_sentiment_df[~human_sentiment_df['is_german']][delta_cols_english]
                german_human_data = human_sentiment_df[human_sentiment_df['is_german']][delta_cols_german]

                tmp_e_h.append(english_human_data)
                tmp_g_h.append(german_human_data)
                tmp_e_m.append(english_model_data)
                tmp_g_m.append(german_model_data)

            scores_human_all_german.append((approach, dataset, tmp_g_h, models))
            scores_human_all_english.append((approach, dataset, tmp_e_h, models))

            scores_model_all_german.append((approach, dataset, tmp_g_m, models))
            scores_model_all_english.append((approach, dataset, tmp_e_m, models))
    return scores_human_all_german, scores_human_all_english, scores_model_all_german, scores_model_all_english


def measure_sentiment_changes():
    scores_human_all_german, scores_human_all_english, scores_model_all_german, scores_model_all_english = sentiment_analysis()
    print('FOO!')
    direction_shifts = {}
    shifts = []
    human_positive = []
    human_negative = []
    human_neutral = []
    pol_ms = []
    for sme, she in zip(scores_model_all_english, scores_human_all_english):
        a_m, ds_m, dfs_m, models_m = sme
        a_h, ds_h, dfs_h, models_h = she
        if a_m != 'direct':
            continue
        df_m = dfs_m[2]
        df_h = dfs_h[2]
        # count direction of shift
        # if ds_h polarity is negative, and delta is negative, then its a negative shift
        pols = df_h['polarity'].tolist()
        pols_m = df_m['polarity'].tolist()

        # shifts = []
        for pol_h, pol_m in zip(pols, pols_m):
            # calculate shift magnitude
            delta = pol_m - pol_h
            # if original is +0.1, then delta of -0.1 is a 100% shift to negative
            if pol_h == 0 and delta != 0:
                shift = float('inf')
            else:
                shift = delta / pol_h
            if pol_h > 0.2:
                human_positive.append(pol_h)
            elif pol_h < -0.2:
                human_negative.append(pol_h)
            else:
                human_neutral.append(pol_h)

            shift_percentage = (delta / abs(pol_h)) * 100 if pol_h != 0 else float('inf')
            shifts.append((pol_h, pol_m, delta, shift, shift_percentage))
            pol_ms.append(pol_m)
    # count how often the shift is positive, negative, or neutral
    positive_shifts = len([x for x in shifts if x[4] > 20])
    negative_shifts = len([x for x in shifts if x[4] < -20])
    neutral_shifts = len([x for x in shifts if -20 <= x[4] <= 20])
    # sum up human pols and get mean human pol
    mean_human_pol = np.mean(human_positive + human_negative + human_neutral)

    # calculate mean shift value, exclude infinity
    mean_shift = np.mean([x[4] for x in shifts if x[4] != float('inf')])
    median_shift = np.median([x[4] for x in shifts if x[4] != float('inf')])
    print('FOO!')

    # calculate mean pol_m
    mean_pol_m = np.mean(pol_ms)
    print('FOO!')


def get_columns_all():
    bertalign_scores_comparison, bertalign_scores_corpus = metrics_analysis('BERTALIGN_SCORES')
    feng_hirst_scores, feng_hirst_scores_corpus = metrics_analysis('FENG_HIRST_SCORES')
    gruen_scores, gruen_scores_corpus = metrics_analysis('GRUEN_SCORES')
    levenshtein_scores, levenshtein_scores_corpus = metrics_analysis('LEVENSHTEIN_SCORES')
    lexical_scores, lexical_scores_corpus = metrics_analysis('LEXICAL_SCORES')
    linguaf_scores, linguaf_scores_corpus = metrics_analysis('LINGUAF_SCORES')
    comparison_scores = set()
    for col_list in [bertalign_scores_comparison, feng_hirst_scores, gruen_scores, levenshtein_scores, lexical_scores,
                     linguaf_scores]:
        for col in col_list:
            comparison_scores.add(col)
    print(comparison_scores)
    print('\n\n\n====================================\n\n\n')
    corpus_scores = set()
    for col_list in [bertalign_scores_corpus, feng_hirst_scores_corpus, gruen_scores_corpus, levenshtein_scores_corpus,
                     lexical_scores_corpus, linguaf_scores_corpus]:
        for col in col_list:
            corpus_scores.add(col)
    for s in sorted(corpus_scores):
        print(s)


class LinguisticLevel(enum.Enum):
    SYNTACTIC = 'syntactic'
    SEMANTIC = 'semantic'
    PRAGMATIC = 'pragmatic'
    LEXICAL = 'lexical'
    ARGMINING = 'argmining'


def filter_scores_dfs(
        scores,
        approach_filter: Optional[str] = None,
        dataset_filter: Optional[str] = None,
        model_filter: Optional[str] = None
):
    # get all relevant scores
    tmp = []
    is_microtexts = []
    for approach, dataset, dfs, models_ in scores:
        if approach_filter is not None and approach != approach_filter:
            continue
        if dataset_filter is not None and dataset != dataset_filter:
            continue
        if model_filter is not None:
            # get index of model in models list
            model_idx = models_.index(model_filter)
            try:
                dfs = [dfs[model_idx]]
            except IndexError:
                print('??')
        else:
            dfs = [x for x in dfs if x is not None]
        is_microtexts.append('microtext' in dataset.lower())
        tmp.extend(dfs)
    return tmp, is_microtexts


def linguistic_level_analysis(
        level: LinguisticLevel,
        approach_filter: Optional[str] = None,
        dataset_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        do_deltas: Optional[bool] = True
):
    if level == LinguisticLevel.SYNTACTIC:
        # complex scores
        comparion_score_names, corpus_score_names, comparison_dfs_all, corpus_dfs_all = metrics_analysis(
            'BERTALIGN_SCORES',
            do_deltas=do_deltas
        )
        comparison_cols = [
            'complex_annotation_scores_num_adv_mod',
            'complex_annotation_scores_num_advcl',
            'complex_annotation_scores_num_appos',
            'complex_annotation_scores_num_coordNP',
            'complex_annotation_scores_num_coordVP',
            'complex_annotation_scores_num_coord_cl',
            'complex_annotation_scores_num_part',
            'complex_annotation_scores_num_prep',
            'complex_annotation_scores_num_relcl',
            'complex_annotation_scores_num_speech'
        ]
        corpus_cols = [
            'bertalign_scores_add',
            'bertalign_scores_copy',
            'bertalign_scores_delete',
            'bertalign_scores_fusion',
            'bertalign_scores_merge',
            'bertalign_scores_other'
        ]
        relevant_comparison, _ = filter_scores_dfs(comparison_dfs_all, approach_filter, dataset_filter, model_filter)
        relevant_comparison_filtered = [x[comparison_cols] for x in relevant_comparison if
                                        x is not None]  # skipping missing ones

        # concat all of them together - number of cols should be consistent
        relevant_comparison_concat = pd.concat(relevant_comparison_filtered, axis=0)

        relevant_corpus, _ = filter_scores_dfs(corpus_dfs_all, approach_filter, dataset_filter, model_filter)
        relevant_corpus_filtered = [x[corpus_cols] for x in relevant_corpus if x is not None]
        # concat all of them together - number of cols should be consistent
        relevant_corpus_concat = pd.concat(relevant_corpus_filtered, axis=0)

        # these are the relevant ones
        relevant_corpus_concat
        relevant_comparison_concat

        tmp = []
        for approach_, ds_, dfs_, _ in comparison_dfs_all:
            l = len(dfs_[0])
            tmp.append((approach_, ds_, l))
        return relevant_comparison_concat, relevant_corpus_concat, tmp
    elif level == LinguisticLevel.SEMANTIC:
        # corpus has length, improved length
        # these are already the deltas!!
        comparion_score_names, corpus_score_names, comparison_dfs_all, corpus_dfs_all = metrics_analysis(
            'FENG_HIRST_SCORES',
            do_deltas=do_deltas
        )
        # comparison_cols = [col.replace(';', '_') for col in comparison_cols]
        relevant_comparison, _ = filter_scores_dfs(comparison_dfs_all, approach_filter, dataset_filter, model_filter)
        feng_hirst_cols = set()
        # get all feng hirst cols
        for _, _, dfs, _ in comparison_dfs_all:
            for df in dfs:
                if df is not None:
                    feng_hirst_cols.update(df.columns)
        for feng_hirst_col in feng_hirst_cols:
            for _, _, dfs, _ in comparison_dfs_all:
                for df in dfs:
                    if df is not None and feng_hirst_col not in df.columns:
                        # add a 0 col
                        df[feng_hirst_col] = 0
        relevant_comparison_filtered = [x[list(feng_hirst_cols)] for x in relevant_comparison if x is not None]
        # concat all of them together - number of cols should be consistent
        relevant_comparison_concat = pd.concat(relevant_comparison_filtered, axis=0)

        # these are also already the deltas!!
        comparion_score_names_gruen, corpus_score_names_gruen, comparison_dfs_all_gruen, corpus_dfs_all_gruen = metrics_analysis(
            'GRUEN_SCORES',
            do_deltas=do_deltas
        )
        relevant_comparison_gruen, _ = filter_scores_dfs(comparison_dfs_all_gruen, approach_filter, dataset_filter,
                                                         model_filter)
        relevant_comparison_gruen_filtered = [x['gruen_scores'] for x in relevant_comparison_gruen if x is not None]
        # concat all of them together - number of cols should be consistent

        relevant_comparison_concat_gruen = pd.concat(relevant_comparison_gruen_filtered, axis=0)
        # add gruen_scores col to feng hirst cols

        # reindex - added for mini, check if needed
        relevant_comparison_concat = relevant_comparison_concat.reset_index(drop=True)
        relevant_comparison_concat_gruen = relevant_comparison_concat_gruen.reset_index(drop=True)

        relevant_comparison_concat_all = pd.concat([relevant_comparison_concat, relevant_comparison_concat_gruen],
                                                   axis=1)

        # sentiment scores here
        scores_human_all_german, scores_human_all_english, scores_model_all_german, scores_model_all_english = sentiment_analysis()

        # must filter human scores to only keep the relevant ones
        if approach_filter is not None:
            scores_h_g_filtered = [x for x in scores_human_all_german if x[0] == approach_filter]
            scores_h_e_filtered = [x for x in scores_human_all_english if x[0] == approach_filter]
        else:
            scores_h_g_filtered = [x for x in scores_human_all_german]
            scores_h_e_filtered = [x for x in scores_human_all_english]
        if dataset_filter is not None:
            scores_h_g_filtered = [x for x in scores_h_g_filtered if x[1] == dataset_filter]
            scores_h_e_filtered = [x for x in scores_h_e_filtered if x[1] == dataset_filter]
        else:
            scores_h_g_filtered = [x for x in scores_h_g_filtered]
            scores_h_e_filtered = [x for x in scores_h_e_filtered]

        # filtering of model scores is done here - after this, we keep only the model scores that fit the filters
        relevant_m_g, is_microtexts_g = filter_scores_dfs(scores_model_all_german, approach_filter, dataset_filter,
                                                          model_filter)
        relevant_m_e, is_microtexts_e = filter_scores_dfs(scores_model_all_english, approach_filter, dataset_filter,
                                                          model_filter)

        # now both human and model scores are filtered

        # the last in each is microtexts - the ONLY one with german/english!!

        # ORIGINAL for sentiment is the HUMAN TEXTS
        # for delta have to use the HUMAN sentiment scores and model scores! model scores - human scores!!

        # concat all in relevant_m_e together, row-wise
        # then add relevant_m_g rows as well
        sentiment_m_merged = pd.concat(relevant_m_e, axis=0)
        sentiment_m_merged = pd.concat([sentiment_m_merged, pd.concat(relevant_m_g, axis=0)], axis=0)

        # for human scores, dfs are repeated - can just use the first one, they are all the same
        # first merge english together
        sentiment_h_merged = pd.concat([x[2][0] for x in scores_h_e_filtered], axis=0)
        # then merge german together
        sentiment_h_merged = pd.concat([sentiment_h_merged, pd.concat([x[2][0] for x in scores_h_g_filtered], axis=0)],
                                       axis=0)

        if do_deltas:
            sentiment_deltas = sentiment_m_merged - sentiment_h_merged
        else:
            # percentage changes
            sentiment_deltas = (sentiment_m_merged - sentiment_h_merged) / sentiment_h_merged

        # get original human scores, to check sentiment scores
        # concat scores_h_e_filtered together
        # sentiment_h_merged = pd.concat([x[2][0] for x in scores_h_e_filtered], axis=0)
        # sentiment_h_merged = pd.concat([sentiment_h_merged, pd.concat([x[2][0] for x in scores_h_g_filtered], axis=0)],
        #                                axis=0)

        # TODO: use sentiment_h_merged to check (for the english cols) how often the sentiment was pushed into one direction (based on the deltas in sentiment_deltas)

        # deltas + this are important
        relevant_comparison_concat_all
        sentiment_deltas

        # reindex relevant_comparison_concat_all, to fix misalignment
        relevant_comparison_concat_all = relevant_comparison_concat_all.reset_index(drop=True)
        sentiment_deltas = sentiment_deltas.reset_index(drop=True)

        # merge them together
        relevant_comparison_concat_all = pd.concat([relevant_comparison_concat_all, sentiment_deltas], axis=1)

        tmp = []
        for approach_, ds_, dfs_, _ in comparison_dfs_all:
            l = len(dfs_[0])
            tmp.append((approach_, ds_, l))

        return relevant_comparison_concat_all, pd.DataFrame(), tmp
    elif level == LinguisticLevel.PRAGMATIC:
        comparion_score_names, corpus_score_names, comparison_dfs_all, corpus_dfs_all = metrics_analysis(
            'AMERICANO_SCORES',
            do_deltas=do_deltas
        )
        comparison_cols = [
            'americano_coherence_avgs',
            'americano_persuasion_avgs'
        ]
        relevant_comparison, _ = filter_scores_dfs(comparison_dfs_all, approach_filter, dataset_filter, model_filter)
        relevant_comparison_filtered = [x[comparison_cols] for x in relevant_comparison if x is not None]
        # concat all of them together - number of cols should be consistent
        relevant_comparison_concat = pd.concat(relevant_comparison_filtered, axis=0)
        # scores can only be between 1 and 5
        # filter out those where abs(score) is not in range
        # scores are deltas, so 0 is also valid - just means no change
        # if a change goes from 5 to 1, delta is 4 - range must be (0, 4)
        relevant_comparison_concat_valids = relevant_comparison_concat[
            (relevant_comparison_concat.abs() >= 0) & (relevant_comparison_concat.abs() <= 4)
            ]

        relevant_comparison_concat_valids
        tmp = []
        for approach_, ds_, dfs_, _ in comparison_dfs_all:
            l = len(dfs_[0])
            tmp.append((approach_, ds_, l))

        return relevant_comparison_concat_valids, pd.DataFrame(), tmp
    elif level == LinguisticLevel.LEXICAL:
        comparion_score_names, _, comparison_dfs_all, _ = metrics_analysis(
            'LINGUAF_SCORES',
            do_deltas=do_deltas
        )
        relevant_comparison, _ = filter_scores_dfs(comparison_dfs_all, approach_filter, dataset_filter, model_filter)
        # find largest number of syllable col
        max_syllables = 0
        for df in relevant_comparison:
            if df is None:
                continue
            for col in df.columns:
                if 'syllable_words' in col:
                    syls = extract_first_number(col)
                    max_syllables = int(max(max_syllables, syls))
        for df in relevant_comparison:
            if df is None:
                continue
            syllable_cols = [col for col in df.columns if 'syllable_words' in col]
            max_syllables_df = int(max([extract_first_number(col) for col in syllable_cols]))
            for i in range(max_syllables_df, max_syllables + 1):
                name = f'linguaf_number_of_{i}_syllable_words'
                # add col with all 0s to df
                df[name] = 0
        relevant_cols = [col for col in relevant_comparison[0].columns]
        relevant_comparison_filtered = [x[relevant_cols] for x in relevant_comparison if x is not None]
        # concat all of them together - number of cols should be consistent
        relevant_comparison_concat = pd.concat(relevant_comparison_filtered, axis=0)
        relevant_comparison_concat['length_delta'] = relevant_comparison_concat['improved_length'] - \
                                                     relevant_comparison_concat['length']

        # lexical scores
        comparion_score_names_lex, _, comparison_dfs_all_lex, _ = metrics_analysis(
            'LEXICAL_SCORES',
            do_deltas=do_deltas
        )
        relevant_comparison_lex, _ = filter_scores_dfs(comparison_dfs_all_lex, approach_filter, dataset_filter,
                                                       model_filter)
        relevant_cols_lex = [col for col in comparion_score_names_lex if col not in ['length', 'improved_length']]
        relevant_comparison_lex_filtered = [x[relevant_cols_lex] for x in relevant_comparison_lex if x is not None]
        # concat all of them together - number of cols should be consistent
        relevant_comparison_lex_concat = pd.concat(relevant_comparison_lex_filtered, axis=0)

        # levenshtein scores
        _, corpus_score_names_lev, _, corpus_dfs_all_lev = metrics_analysis(
            'LEVENSHTEIN_SCORES',
            do_deltas=do_deltas
        )
        relevant_corpus_lev, _ = filter_scores_dfs(corpus_dfs_all_lev, approach_filter, dataset_filter, model_filter)
        relevant_cols_lev = [col for col in corpus_score_names_lev]
        relevant_corpus_lev_filtered = [x[relevant_cols_lev] for x in relevant_corpus_lev if x is not None]

        # these two are what is relevant for this step!
        relevant_corpus_lev_concat = pd.concat(relevant_corpus_lev_filtered, axis=0)
        relevant_comparison_concat_all = pd.concat([relevant_comparison_concat, relevant_comparison_lex_concat], axis=1)

        relevant_corpus_lev_concat
        relevant_comparison_concat_all
        tmp = []
        for approach_, ds_, dfs_, _ in comparison_dfs_all:
            l = len(dfs_[0])
            tmp.append((approach_, ds_, l))
        return relevant_comparison_concat_all, relevant_corpus_lev_concat, tmp
    else:
        print('wat')


def make_overview_table(
        relevant_models: Optional[list[str]] = None
):
    if relevant_models is None:
        relevant_models = models

    out_compare = []
    out_corpus = []

    for model in relevant_models:
        for level in LinguisticLevel:
            comparison_df, corpus_df, _ = linguistic_level_analysis(
                level,
                model_filter=model,
                approach_filter='direct',
                do_deltas=False
            )
            # remove length and improved_length columns from dfs, if they exist
            if 'length' in comparison_df.columns:
                comparison_df = comparison_df.drop(columns=['length'])
            if 'improved_length' in comparison_df.columns:
                comparison_df = comparison_df.drop(columns=['improved_length'])
            if 'length' in corpus_df.columns:
                corpus_df = corpus_df.drop(columns=['length'])
            if 'improved_length' in corpus_df.columns:
                corpus_df = corpus_df.drop(columns=['improved_length'])

            # count how many NaN and infinite are in each column
            nans_compare = comparison_df.isna().sum()
            infs_compare = comparison_df.isin([np.inf, -np.inf]).sum()
            nans_corpus = corpus_df.isna().sum()
            infs_corpus = corpus_df.isin([np.inf, -np.inf]).sum()

            comparison_df = comparison_df.replace([np.inf, -np.inf], np.nan)
            corpus_df = corpus_df.replace([np.inf, -np.inf], np.nan)
            # calculate mean for each column, but skip NaN and infs
            col_means_compare = {col: comparison_df[~comparison_df[col].isna()][col].mean() for col in
                                 comparison_df.columns}
            col_means_corpus = {col: corpus_df[~corpus_df[col].isna()][col].mean() for col in corpus_df.columns}

            try:
                mean_change_compare = sum(col_means_compare.values()) / len(col_means_compare)
            except ZeroDivisionError:
                mean_change_compare = 0
            try:
                mean_change_corpus = sum(col_means_corpus.values()) / len(col_means_corpus)
            except ZeroDivisionError:
                mean_change_corpus = 0
            out_compare.append((model, level.value, mean_change_compare, nans_compare, infs_compare))
            out_corpus.append((model, level.value, mean_change_corpus, nans_corpus, infs_corpus))

    out_compare_df = pd.DataFrame(out_compare, columns=['model', 'level', 'mean_change', 'nans', 'infs'])
    out_corpus_df = pd.DataFrame(out_corpus, columns=['model', 'level', 'mean_change', 'nans', 'infs'])

    out_compare_df.to_csv('overview_compare.csv')
    out_corpus_df.to_csv('overview_corpus.csv')

    # rows: models
    # cols: linguistic levels


def load_analysis_files_new(
        model_filter: str,
        approach: Optional[str] = None,
        level=None,
        ds=None
):
    if isinstance(model_filter, str):
        model_filter = [model_filter]
    csv_file_dir = os.path.join(current_file_dir, 'analysis_csvs_separate')
    needed_files = []
    if level is None:
        needed_levels = [level for level in LinguisticLevel if level != LinguisticLevel.ARGMINING]
    else:
        needed_levels = [level]
    if approach is None:
        needed_approaches = approaches
    else:
        needed_approaches = [approach]
    if ds is None:
        needed_datasets = datasets
    else:
        needed_datasets = [ds]

    dfs_deltas = []
    dfs_percentages = []
    for model in model_filter:
        for level in needed_levels:
            for need_ds in needed_datasets:
                for a in needed_approaches:
                    f = f'{model}_{level}_{need_ds}_{a}_comparison_percentages.csv'
                    df_comp = pd.read_csv(os.path.join(csv_file_dir, f), index_col=0)
                    f = f'{model}_{level}_{need_ds}_{a}_corpus_percentages.csv'
                    df_corpus = pd.read_csv(os.path.join(csv_file_dir, f), index_col=0)
                    dfs_percentages.append((level, df_comp, df_corpus))

                    f = f'{model}_{level}_{need_ds}_{a}_comparison_deltas.csv'
                    df_comp = pd.read_csv(os.path.join(csv_file_dir, f), index_col=0)
                    f = f'{model}_{level}_{need_ds}_{a}_corpus_deltas.csv'
                    df_corpus = pd.read_csv(os.path.join(csv_file_dir, f), index_col=0)

                    dfs_deltas.append((level, df_comp, df_corpus))
    return dfs_deltas, dfs_percentages


def OLDprocess_llama3_for_analysis(
        approach: str
):
    dfs_deltas = []
    dfs_percentages = []
    out_dir = os.path.join(current_file_dir, 'analysis_csvs')
    # check if files exist
    out_files_all = [os.path.join(out_dir, f'LLAMA3_{level}_{ds}_{approach}_comparison_percentages.csv') for level in
                     LinguisticLevel for ds in datasets]
    out_files_all.extend(
        [os.path.join(out_dir, f'LLAMA3_{level}_{ds}_{approach}_corpus_percentages.csv') for level in LinguisticLevel
         for ds in datasets])

    if all([os.path.exists(f) for f in out_files_all]):
        # load files
        print('> Files found, loading files for llama3')
        for level in LinguisticLevel:
            for ds in datasets:
                comparison = pd.read_csv(
                    os.path.join(out_dir, f'LLAMA3_{level}_{ds}_{approach}_comparison_percentages.csv'), index_col=0)
                corpus = pd.read_csv(os.path.join(out_dir, f'LLAMA3_{level}_{ds}_{approach}_corpus_percentages.csv'),
                                     index_col=0)
                dfs_percentages.append((level, comparison, corpus))
                comparison = pd.read_csv(os.path.join(out_dir, f'LLAMA3_{level}_{ds}_{approach}_comparison_deltas.csv'),
                                         index_col=0)
                corpus = pd.read_csv(os.path.join(out_dir, f'LLAMA3_{level}_{ds}_{approach}_corpus_deltas.csv'),
                                     index_col=0)
                dfs_deltas.append((level, comparison, corpus))
    else:
        print('> Files not found, generating files for llama3')
        for level in LinguisticLevel:
            print('>>> LEVEL:', level)
            for ds in datasets:
                comparison, corpus, tmp = linguistic_level_analysis(
                    level,
                    model_filter='llama3_nemotron_cleaned',
                    dataset_filter=ds,
                    do_deltas=False,
                    approach_filter=approach
                )
                dfs_percentages.append((level, ds, comparison, corpus))
                comparison, corpus, tmp = linguistic_level_analysis(
                    level,
                    model_filter='llama3_nemotron_cleaned',
                    do_deltas=True,
                    dataset_filter=ds,
                    approach_filter=approach,
                )
                dfs_deltas.append((level, ds, comparison, corpus))
            for level_p, ds_p, comparison, corpus in dfs_percentages:
                comparison.to_csv(
                    os.path.join(out_dir, f'LLAMA3_{level_p}_{ds_p}_{approach}_comparison_percentages.csv'))
                corpus.to_csv(os.path.join(out_dir, f'LLAMA3_{level_p}_{ds_p}_{approach}_corpus_percentages.csv'))
            for level_d, ds_d, comparison, corpus in dfs_deltas:
                comparison.to_csv(os.path.join(out_dir, f'LLAMA3_{level_d}_{ds_d}_{approach}_comparison_deltas.csv'))
                corpus.to_csv(os.path.join(out_dir, f'LLAMA3_{level_d}_{ds_d}_{approach}_corpus_deltas.csv'))
    return dfs_deltas, dfs_percentages


def process_model_for_analysis(
        approach: str,
        model_out_name: str,
        model_filter: str
):
    dfs_deltas = []
    dfs_percentages = []
    out_dir = os.path.join(current_file_dir, 'analysis_csvs')
    # check if files exist
    out_files_all = [os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_comparison_percentages.csv') for
                     level in
                     LinguisticLevel for ds in datasets]
    out_files_all.extend(
        [os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_corpus_percentages.csv') for level in
         LinguisticLevel
         for ds in datasets])

    if all([os.path.exists(f) for f in out_files_all]):
        # load files
        print(f'> Files found, loading files for {model_out_name}')
        for level in LinguisticLevel:
            for ds in datasets:
                comparison = pd.read_csv(
                    os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_comparison_percentages.csv'),
                    index_col=0)
                corpus = pd.read_csv(
                    os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_corpus_percentages.csv'),
                    index_col=0)
                dfs_percentages.append((level, comparison, corpus))
                comparison = pd.read_csv(
                    os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_comparison_deltas.csv'),
                    index_col=0)
                corpus = pd.read_csv(
                    os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_corpus_deltas.csv'),
                    index_col=0)
                dfs_deltas.append((level, comparison, corpus))
    else:
        print(f'> Files not found, generating files for {model_out_name}')
        for level in LinguisticLevel:
            print('>>> LEVEL:', level)
            for ds in datasets:
                comparison, corpus, tmp = linguistic_level_analysis(
                    level,
                    model_filter=model_filter,
                    dataset_filter=ds,
                    do_deltas=False,
                    approach_filter=approach
                )
                dfs_percentages.append((level, ds, comparison, corpus))
                comparison, corpus, tmp = linguistic_level_analysis(
                    level,
                    model_filter=model_filter,
                    do_deltas=True,
                    dataset_filter=ds,
                    approach_filter=approach,
                )
                dfs_deltas.append((level, ds, comparison, corpus))
            for level_p, ds_p, comparison, corpus in dfs_percentages:
                comparison.to_csv(
                    os.path.join(out_dir, f'{model_out_name}_{level_p}_{ds_p}_{approach}_comparison_percentages.csv'))
                corpus.to_csv(
                    os.path.join(out_dir, f'{model_out_name}_{level_p}_{ds_p}_{approach}_corpus_percentages.csv'))
            for level_d, ds_d, comparison, corpus in dfs_deltas:
                comparison.to_csv(
                    os.path.join(out_dir, f'{model_out_name}_{level_d}_{ds_d}_{approach}_comparison_deltas.csv'))
                corpus.to_csv(os.path.join(out_dir, f'{model_out_name}_{level_d}_{ds_d}_{approach}_corpus_deltas.csv'))
    return dfs_deltas, dfs_percentages


def process_model_for_analysis_SAVE_ALL(
        model_out_name: str,
        model_filter: str,
        approach_filter: Optional = None
):
    """
    debug function that saves everything at once for a model - load each file and concat as needed later
    """
    dfs_deltas = []
    dfs_percentages = []
    out_dir = os.path.join(current_file_dir, 'analysis_csvs_separate')
    # check if files exist
    print(f'> Files not found, generating files for {model_out_name}')
    for level in LinguisticLevel:
        print('>>> LEVEL:', level)
        if level == LinguisticLevel.ARGMINING:
            continue
        comparison, corpus, tmp = linguistic_level_analysis(
            level,
            model_filter=model_filter,
            do_deltas=False,
            approach_filter=approach_filter
        )
        counter = 0
        for approach, ds, l in tmp:
            r_comp = comparison[counter:counter + l]
            r_corpus = corpus[counter:counter + l]
            r_comp.to_csv(
                os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_comparison_percentages.csv'))
            r_corpus.to_csv(os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_corpus_percentages.csv'))
            counter += l

        # dfs_percentages.append((level, ds, comparison, corpus))
        comparison, corpus, tmp = linguistic_level_analysis(
            level,
            model_filter=model_filter,
            do_deltas=True,
        )
        counter = 0
        for approach, ds, l in tmp:
            r_comp = comparison[counter:counter + l]
            r_corpus = corpus[counter:counter + l]
            r_comp.to_csv(os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_comparison_deltas.csv'))
            r_corpus.to_csv(os.path.join(out_dir, f'{model_out_name}_{level}_{ds}_{approach}_corpus_deltas.csv'))
            counter += l

        # dfs_deltas.append((level, ds, comparison, corpus))
        # for level_p, ds_p, comparison, corpus in dfs_percentages:
        #     comparison.to_csv(
        #         os.path.join(out_dir, f'{model_out_name}_{level_p}_{ds_p}_{approach}_comparison_percentages.csv'))
        #     corpus.to_csv(os.path.join(out_dir, f'{model_out_name}_{level_p}_{ds_p}_{approach}_corpus_percentages.csv'))
        # for level_d, ds_d, comparison, corpus in dfs_deltas:
        #     comparison.to_csv(os.path.join(out_dir, f'{model_out_name}_{level_d}_{ds_d}_{approach}_comparison_deltas.csv'))
        #     corpus.to_csv(os.path.join(out_dir, f'{model_out_name}_{level_d}_{ds_d}_{approach}_corpus_deltas.csv'))
    return dfs_deltas, dfs_percentages


import plotly.express as px


def create_heatmap_plotly(
        df,
        name,
        cell_width=70,
        cell_height=50,
):
    row_count, col_count = df.shape  # Get number of rows and columns

    # Compute figure dimensions based on cell size
    fig_width = 500
    fig_height = 500

    fig = px.imshow(
        df.values,
        labels=dict(x='Dataset', y='Metric', color='% change'),
        x=df.columns,
        y=df.index,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        width=fig_width,  # Adjust width to make it more compact
        height=fig_height,  # Adjust height
        margin=dict(l=0, r=0, t=0, b=0),  # Adjust margins,
        yaxis_title=None,
        xaxis_title=None,
        autosize=True
    )
    fig.update_coloraxes(showscale=False)
    # fig.show()

    # save fig as png
    fig.write_image(
        os.path.join(current_file_dir, 'heatmaps', f'{name}.png'),
        scale=1
    )

    return fig


import matplotlib.pyplot as plt
import seaborn as sns


def create_heatmap(df, name, base_cell_width=40, base_cell_height=30):
    df = df.drop(['improved_length', 'original_length'], errors='ignore')
    row_count, col_count = df.shape

    # Set figure size based on cell count
    fig_width = col_count * (base_cell_width / 100)
    fig_height = row_count * (base_cell_height / 100)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap without colorbar
    heatmap = sns.heatmap(
        df,
        annot=False,
        cmap='coolwarm',
        cbar=False,  # Disable seaborn’s default colorbar
        xticklabels=True,
        yticklabels=True,
        ax=ax,
        square=False,
        center=0,
        vmax=100,
        vmin=-100
    )

    # Create a separate axis for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)  # 5% width, pad between heatmap and colorbar
    cbar = fig.colorbar(heatmap.get_children()[0], cax=cax)
    cbar.set_label('', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Rotate labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    # Save
    output_dir = os.path.join(current_file_dir, 'heatmaps_version2')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{name}.pdf')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.05, format='pdf')
    plt.close(fig)
    return fig


def create_barchart(
        df,
        name
):
    pass


def llama_analysis_table():
    # deltas, percentages = OLDprocess_llama3_for_analysis(
    #     'direct'
    # )
    deltas, percentages = load_analysis_files_new(
        'llama3_nemotron_cleaned',
        approach='direct'
    )
    bertalign_cols = [
        'bertalign_scores_add',
        'bertalign_scores_copy',
        'bertalign_scores_delete',
        'bertalign_scores_fusion',
        'bertalign_scores_merge',
        'bertalign_scores_other'
    ]
    semantic_cols = [
        'feng_hirst_depth',
        'feng_hirst_relation_count_Attribution',
        'feng_hirst_relation_count_Background',
        'feng_hirst_relation_count_Cause',
        'feng_hirst_relation_count_Comparison',
        'feng_hirst_relation_count_Condition',
        'feng_hirst_relation_count_Contrast',
        'feng_hirst_relation_count_Elaboration',
        'feng_hirst_relation_count_Enablement',
        'feng_hirst_relation_count_Evaluation',
        'feng_hirst_relation_count_Explanation',
        'feng_hirst_relation_count_Joint',
        'feng_hirst_relation_count_Manner-Means',
        'feng_hirst_relation_count_Summary',
        'feng_hirst_relation_count_Temporal',
        'feng_hirst_relation_count_Topic-Change',
        'feng_hirst_relation_count_Topic-Comment',
        'feng_hirst_relation_count_same-unit',
        'gruen_scores',
        'polarity',
        'subjectivity',
        'german_proba_positive',
        'german_proba_negative',
        'german_proba_neutral'
    ]
    lexical_cols = [
        'linguaf_avg_word_length',
        'linguaf_char_count',
        'linguaf_digit_count',
        'linguaf_letter_count',
        'linguaf_avg_sentence_length',
        'linguaf_avg_words_per_sentence',
        'length',
        'improved_length',
        'lexical_ttr',
        'linguaf_flesch_kincaid_grade',
        'linguaf_flesch_reading_ease',
    ]
    lexical_cols_corpus = [
        'levenshtein_levenshtein'
    ]
    ds_names = ['Rev1', 'Rev2', 'Rev3', 'Essays', 'MT']

    for level in LinguisticLevel:
        relevent_deltas, relevant_percs = [], []
        for delta, percs in zip(deltas, percentages):
            if level != delta[0]:
                continue
            relevent_deltas.append(delta)
            relevant_percs.append(percs)
        # deltas_rev1, deltas_rev2, deltas_rev3, deltas_essays, deltas_microtexts = relevent_deltas
        # percs_rev1, percs_rev2, percs_rev3, percs_essays, percs_microtexts = relevant_percs
        if level == LinguisticLevel.SYNTACTIC:
            bertaligns = []
            complexs = []

            # for bertalign, percentage and deltas are the same (reference-based metric, cant do a percentage measure)
            for r in relevant_percs:
                bertalign_scores = r[2][bertalign_cols]
                # rename cols to remove 'bertalign_scores' from them
                bertalign_scores.columns = [col.replace('bertalign_scores_', '') for col in bertalign_scores.columns]
                complex_cols = [col for col in r[1].columns if col.startswith('complex_annotation_scores')]
                complex_scores = r[1][complex_cols]
                # rename cols to remove 'complex_annotation_scores' from them
                complex_scores.columns = [col.replace('complex_annotation_scores_', '') for col in
                                          complex_scores.columns]
                bertaligns.append(bertalign_scores.mean())

                # replace infinite with NaN
                complex_scores.replace([np.inf, -np.inf], np.nan, inplace=True)
                complexs.append(complex_scores.mean() * 100)
            bertaligns_dict = {ds: df for ds, df in zip(ds_names, bertaligns)}
            complexs_dict = {ds: df for ds, df in zip(ds_names, complexs)}

            bertaligns_df = pd.DataFrame(bertaligns_dict)
            complexs_df = pd.DataFrame(complexs_dict)
            final_df = pd.concat([bertaligns_df, complexs_df])

            # round all scores to 2 decimals
            final_df = final_df.round(2)
            bertalign_final_df = final_df.iloc[:6]
            complex_final_df = final_df.iloc[6:]
            create_heatmap(bertalign_final_df, 'syntactic_bertalign')
            create_heatmap(complex_final_df, 'syntactic_complex')
            # convert all floats to string with 2 decimals
            final_df = final_df.applymap(lambda x: f'{x:.2f}')

            final_df.index.name = 'score_name'
            # escape all _ in df
            final_df.reset_index(inplace=True)
            final_df['score_name'] = final_df['score_name'].apply(lambda x: x.replace('_', '\_'))
            final_df.columns = [col.replace('_', '\_') for col in final_df.columns]
            print('========== SYNTACTIC TABLE ==========\n\n')
            print(final_df.to_latex(index=False))
            print('\n\n\n')
        elif level == LinguisticLevel.SEMANTIC:
            scores = []
            for r in relevant_percs:
                tmp = r[1][semantic_cols]
                tmp.replace([np.inf, -np.inf], np.nan, inplace=True)

                # remove feng_hirst_relation_count_ from column names
                tmp.columns = [col.replace('feng_hirst_relation_count_', '') for col in tmp.columns]
                # multiple all cols except the german_ ones by 100
                # filter out those where value is above 1000
                tmp = tmp[tmp < 100]

                scores.append(tmp.mean() * 100)
            semantic_dict = {ds: df for ds, df in zip(ds_names, scores)}
            semantic_df = pd.DataFrame(semantic_dict)
            semantic_df = semantic_df.round(2)

            # rename gruen_scores to gruen_score
            semantic_df.rename(columns={'gruen_scores': 'gruen_score', 'feng_hirst_depth': 'depth of RST tree'},
                               inplace=True)

            create_heatmap(semantic_df, 'semantic')

            semantic_df = semantic_df.applymap(lambda x: f'{x:.2f}')
            semantic_df.index.name = 'score_name'
            semantic_df.reset_index(inplace=True)
            semantic_df['score_name'] = semantic_df['score_name'].apply(lambda x: x.replace('_', '\_'))
            semantic_df.columns = [col.replace('_', '\_') for col in semantic_df.columns]
            print('========== SEMANTIC TABLE ==========\n\n')
            print(semantic_df.to_latex(index=False))
            print('\n\n\n')
        elif level == LinguisticLevel.PRAGMATIC:
            scores = []
            for r in relevant_percs:
                tmp = r[1][['americano_coherence_avgs', 'americano_persuasion_avgs']]
                tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
                scores.append(tmp.mean() * 100)
            pragmatic_dict = {ds: df for ds, df in zip(ds_names, scores)}
            pragmatic_df = pd.DataFrame(pragmatic_dict)
            pragmatic_df = pragmatic_df.round(2)
            create_heatmap(pragmatic_df, 'pragmatic')

            pragmatic_df = pragmatic_df.applymap(lambda x: f'{x:.2f}')
            pragmatic_df.index.name = 'score_name'
            pragmatic_df.reset_index(inplace=True)
            pragmatic_df['score_name'] = pragmatic_df['score_name'].apply(lambda x: x.replace('_', '\_'))
            pragmatic_df.columns = [col.replace('_', '\_') for col in pragmatic_df.columns]
            print('========== PRAGMATIC TABLE ==========\n\n')
            print(pragmatic_df.to_latex(index=False))
            print('\n\n\n')
        elif level == LinguisticLevel.LEXICAL:
            scores_compare = []
            scores_l = []
            for r in relevant_percs:
                tmp = r[1][lexical_cols]
                levenshtein = r[2][['levenshtein_levenshtein']]
                # also get all syllable_count cols
                counts = [col for col in r[1].columns if 'linguaf_number_of_' in col]
                # get max syllable count
                max_syllables = int(max(extract_first_number(col) for col in counts))
                count1to3 = 0
                count4to6 = 0
                count7to10 = 0
                count10plus = 0
                print('foo!')
                cols1to3 = [f'linguaf_number_of_{n}_syllable_words' for n in range(1, 4)]
                count1to3 = r[1][cols1to3].sum(axis=1) / 3

                cols4to6 = [f'linguaf_number_of_{n}_syllable_words' for n in range(4, 7)]
                count4to6 = r[1][cols4to6].sum(axis=1) / 3

                cols7to10 = [f'linguaf_number_of_{n}_syllable_words' for n in range(7, min(11, max_syllables))]
                count7to10 = r[1][cols7to10].sum(axis=1) / 3

                cols10plus = [f'linguaf_number_of_{n}_syllable_words' for n in range(10, max_syllables + 1)]
                count10plus = r[1][cols10plus].sum(axis=1) / 3

                tmp['count1to3'] = count1to3
                tmp['count4to6'] = count4to6
                tmp['count7to10'] = count7to10
                tmp['count10plus'] = count10plus

                tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
                means = tmp.mean()
                # multiply all except length and improved_length by 100
                means = means * 100
                means['length'] /= 100
                means['improved_length'] /= 100

                # calculate length change in percentage
                means['length_change'] = ((means['improved_length'] - means['length']) / means['length']) * 100
                # remove length cols from means
                means = means.drop(['length', 'improved_length'])

                scores_compare.append(means)

                levenshtein.replace([np.inf, -np.inf], np.nan, inplace=True)
                scores_l.append(levenshtein.mean())
            compare_dict = {ds: df for ds, df in zip(ds_names, scores_compare)}
            compare_df = pd.DataFrame(compare_dict)
            levenshtein_df = pd.DataFrame({ds: df for ds, df in zip(ds_names, scores_l)})
            create_heatmap(compare_df, 'lexical_compare')
            create_heatmap(levenshtein_df, 'lexical_levenshtein')
            final_df = pd.concat([compare_df, levenshtein_df])

            final_df = final_df.round(2)
            final_df = final_df.applymap(lambda x: f'{x:.2f}')

            final_df.index.name = 'score_name'
            final_df.reset_index(inplace=True)

            final_df['score_name'] = final_df['score_name'].apply(lambda x: x.replace('_', '\_'))
            final_df.columns = [col.replace('_', '\_') for col in final_df.columns]
            print('========== LEXICAL TABLE ==========\n\n')
            print(final_df.to_latex(index=False))
            print('\n\n\n')

    print('FOO!')


def calculate_pearsonr_correlation(
        df: pd.DataFrame,
):
    length_col = 'length'
    # if length is not in cols, use original_length - they are the same
    if length_col not in df.columns:
        length_col = 'original_length'
    correlations = {}
    p_values = {}

    for col in df.columns:
        if col == length_col:
            continue
        x = df[length_col]
        y = df[col]
        valid = x.notna() & y.notna()
        if valid.sum() > 1:
            r, p = pearsonr(x[valid], y[valid])
            correlations[col] = r
            p_values[col] = p

    correlations = sorted([(k, v) for k, v in correlations.items()], key=lambda x: abs(x[1]), reverse=True)
    return correlations, p_values


def general_analysis_table(
        model_out_name: Union[str, list[str]],
        approach_filter: Optional[str] = None,
        levels: Optional = LinguisticLevel
):
    if len(levels) > 1:
        deltas, percentages = load_analysis_files_new(
            model_out_name,
            approach=approach_filter
        )
    else:
        deltas = []
        percentages = []
    bertalign_cols = [
        'bertalign_scores_add',
        'bertalign_scores_copy',
        'bertalign_scores_delete',
        'bertalign_scores_fusion',
        'bertalign_scores_merge',
        'bertalign_scores_other'
    ]
    semantic_cols = [
        'feng_hirst_depth',
        'feng_hirst_relation_count_Attribution',
        'feng_hirst_relation_count_Background',
        'feng_hirst_relation_count_Cause',
        'feng_hirst_relation_count_Comparison',
        'feng_hirst_relation_count_Condition',
        'feng_hirst_relation_count_Contrast',
        'feng_hirst_relation_count_Elaboration',
        'feng_hirst_relation_count_Enablement',
        'feng_hirst_relation_count_Evaluation',
        'feng_hirst_relation_count_Explanation',
        'feng_hirst_relation_count_Joint',
        'feng_hirst_relation_count_Manner-Means',
        'feng_hirst_relation_count_Summary',
        'feng_hirst_relation_count_Temporal',
        'feng_hirst_relation_count_Topic-Change',
        'feng_hirst_relation_count_Topic-Comment',
        'feng_hirst_relation_count_same-unit',
        'gruen_scores',
        'polarity',
        'subjectivity',
        'german_proba_positive',
        'german_proba_negative',
        'german_proba_neutral'
    ]
    lexical_cols = [
        'linguaf_avg_word_length',
        'linguaf_char_count',
        'linguaf_digit_count',
        'linguaf_letter_count',
        'linguaf_avg_sentence_length',
        'linguaf_avg_words_per_sentence',
        'length',
        'improved_length',
        'lexical_ttr',
        'linguaf_flesch_kincaid_grade',
        'linguaf_flesch_reading_ease',
    ]
    lexical_cols_corpus = [
        'levenshtein_levenshtein'
    ]
    ds_names = ['Rev1', 'Rev2', 'Rev3', 'Essays', 'MT']
    lexical_final = None
    syntactic_final = None
    semantic_final = None
    pragmatic_final = None
    argmining_final = None

    correlations_per_level = {}

    for level in levels:
        if level not in correlations_per_level:
            correlations_per_level[level] = {}
        relevent_deltas, relevant_percs = [], []
        for delta, percs in zip(deltas, percentages):
            if level != delta[0]:
                continue
            relevent_deltas.append(delta)
            relevant_percs.append(percs)

        lens, _ = get_improved_lens_all()

        # datasets
        if not isinstance(model_out_name, list):
            model_out_name = [model_out_name]
        expanded_datasets = datasets * len(model_out_name)
        expanded_lens = DATASET_LENS * len(model_out_name)

        for ds, r, ol in zip(expanded_datasets, relevant_percs, expanded_lens):
            rl = lens[approach_filter][ds][model_out_name[0]]
            r[1]['improved_length'] = rl
            r[2]['improved_length'] = rl
            r[1]['original_length'] = ol
            r[2]['original_length'] = ol

        # deltas_rev1, deltas_rev2, deltas_rev3, deltas_essays, deltas_microtexts = relevent_deltas
        # percs_rev1, percs_rev2, percs_rev3, percs_essays, percs_microtexts = relevant_percs
        if level == LinguisticLevel.SYNTACTIC:
            bertaligns = []
            complexs = []

            # for bertalign, percentage and deltas are the same (reference-based metric, cant do a percentage measure)
            complex_alls = []
            bertaligns_all = []
            for r in relevant_percs:
                bertalign_scores = r[2][bertalign_cols]
                # rename cols to remove 'bertalign_scores' from them
                bertalign_scores.columns = [col.replace('bertalign_scores_', '') for col in bertalign_scores.columns]
                # add lengths to bertalign_scores
                bertalign_scores['improved_length'] = r[2]['improved_length']
                bertalign_scores['original_length'] = r[2]['original_length']

                complex_cols = [col for col in r[1].columns if col.startswith('complex_annotation_scores')]
                complex_cols += ['improved_length', 'original_length']
                complex_scores = r[1][complex_cols]
                # rename cols to remove 'complex_annotation_scores' from them
                complex_scores.columns = [col.replace('complex_annotation_scores_', '') for col in
                                          complex_scores.columns]
                bertaligns_all.append(bertalign_scores)
                bertaligns.append(bertalign_scores.mean() * 100)

                # replace infinite with NaN
                complex_scores.replace([np.inf, -np.inf], np.nan, inplace=True)
                complex_alls.append(complex_scores)
                foo = complex_scores.mean() * 100
                foo['improved_length'] /= 100
                foo['original_length'] /= 100
                complexs.append(foo)

            # correlation check here
            complex_alls = pd.concat(complex_alls)
            # correlation: complex_alls.corr()
            syntatic_corrs_all = complex_alls.corr()

            # corrs, ps = calculate_pearsonr_correlation(syntatic_corrs_all)
            syntactic_corrs, syntactic_ps = calculate_pearsonr_correlation(complex_alls)
            syntactic_means = complex_alls.mean()

            bertaligns_all = pd.concat(bertaligns_all)
            bertalign_corrs, bertalign_ps = calculate_pearsonr_correlation(bertaligns_all)
            # correlation: bertaligns_all.corr()
            bertalign_corrs_all = bertaligns_all.corr()
            bertalign_corrs, bertalign_ps = calculate_pearsonr_correlation(bertaligns_all)
            bertalign_means = bertaligns_all.mean()

            bertaligns_dict = {ds: df for ds, df in zip(ds_names, bertaligns)}
            complexs_dict = {ds: df for ds, df in zip(ds_names, complexs)}

            bertaligns_df = pd.DataFrame(bertaligns_dict)
            complexs_df = pd.DataFrame(complexs_dict)

            final_df = pd.concat([bertaligns_df, complexs_df])

            # round all scores to 2 decimals
            final_df = final_df.round(2)
            bertalign_final_df = final_df.iloc[:6]
            complex_final_df = final_df.iloc[6:]

            bertalign_df_table = bertalign_final_df.copy()
            bertalign_df_table.reset_index(inplace=True)
            # round numeric cols to 2 decimals and convert to str
            bertalign_df_table.iloc[:, 1:] = bertalign_df_table.iloc[:, 1:].applymap(lambda x: f'{x:.2f}')
            # bertalign_df_table['score_name'] = bertalign_df_table['score_name'].apply(lambda x: x.replace('_', '\_'))
            bertalign_df_table.columns = [col.replace('_', '\_') for col in bertalign_df_table.columns]
            print('========== BERTALIGN TABLE ==========\n\n')
            print(bertalign_df_table.to_latex(index=False))

            create_heatmap(bertalign_final_df, 'syntactic_bertalign')
            create_heatmap(complex_final_df, 'syntactic_complex')

            # convert all floats to string with 2 decimals
            final_df = final_df.applymap(lambda x: f'{x:.2f}')

            final_df.index.name = 'score_name'
            # escape all _ in df
            final_df.reset_index(inplace=True)
            final_df['score_name'] = final_df['score_name'].apply(lambda x: x.replace('_', '\_'))
            final_df.columns = [col.replace('_', '\_') for col in final_df.columns]
            print('========== SYNTACTIC TABLE ==========\n\n')
            print(final_df.to_latex(index=False))
            print('\n\n\n')
            syntactic_final = final_df
        elif level == LinguisticLevel.SEMANTIC:
            scores = []

            tmps_all = []
            for r in relevant_percs:
                tmp = r[1][semantic_cols]
                # add length and improved length to tmp
                if 'original_length' not in r[1].columns:
                    r[1]['original_length'] = r[1]['length']
                tmp['original_length'] = r[1]['original_length']
                tmp['improved_length'] = r[1]['improved_length']
                tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
                tmps_all.append(tmp.copy())

                # remove feng_hirst_relation_count_ from column names
                tmp.columns = [col.replace('feng_hirst_relation_count_', '') for col in tmp.columns]
                # multiple all cols except the german_ ones by 100
                # filter out those where value is above 1000
                tmp = tmp[tmp < 100]

                scores.append(tmp.mean() * 100)
            tmps_all = pd.concat(tmps_all)
            # correlation: tmps_all.corr()
            semantic_corrs_all = tmps_all.corr()
            semantic_corrs, semantic_ps = calculate_pearsonr_correlation(tmps_all)
            semantic_means = tmps_all.mean()
            semantic_dict = {ds: df for ds, df in zip(ds_names, scores)}
            semantic_df = pd.DataFrame(semantic_dict)
            semantic_df = semantic_df.round(2)

            # rename gruen_scores to gruen_score
            semantic_df.rename(columns={'gruen_scores': 'gruen_score', 'feng_hirst_depth': 'depth of RST tree'},
                               inplace=True)

            create_heatmap(semantic_df, 'semantic')

            semantic_df = semantic_df.applymap(lambda x: f'{x:.2f}')
            semantic_df.index.name = 'score_name'
            semantic_df.reset_index(inplace=True)
            semantic_df['score_name'] = semantic_df['score_name'].apply(lambda x: x.replace('_', '\_'))
            semantic_df.columns = [col.replace('_', '\_') for col in semantic_df.columns]
            print('========== SEMANTIC TABLE ==========\n\n')
            print(semantic_df.to_latex(index=False))
            print('\n\n\n')
            semantic_final = semantic_df
        elif level == LinguisticLevel.PRAGMATIC:
            scores = []

            tmp_merge_all = []
            for r in relevant_percs:
                tmp = r[1][
                    ['americano_coherence_avgs', 'americano_persuasion_avgs', 'original_length', 'improved_length']]
                tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
                tmp_merge_all.append(tmp.copy())
                tmp = tmp.mean() * 100
                tmp['original_length'] /= 100
                tmp['improved_length'] /= 100
                scores.append(tmp)
            tmp_merge_all = pd.concat(tmp_merge_all)
            # correlation: tmp_merge_all.corr()
            pragmatic_corrs_all = tmp_merge_all.corr()

            pragmatic_corrs, pragmatic_ps = calculate_pearsonr_correlation(tmp_merge_all)
            pragmatic_means = tmp_merge_all.mean()

            pragmatic_dict = {ds: df for ds, df in zip(ds_names, scores)}
            pragmatic_df = pd.DataFrame(pragmatic_dict)
            pragmatic_df = pragmatic_df.round(2)
            create_heatmap(pragmatic_df, 'pragmatic')

            pragmatic_df = pragmatic_df.applymap(lambda x: f'{x:.2f}')
            pragmatic_df.index.name = 'score_name'
            pragmatic_df.reset_index(inplace=True)
            pragmatic_df['score_name'] = pragmatic_df['score_name'].apply(lambda x: x.replace('_', '\_'))
            pragmatic_df.columns = [col.replace('_', '\_') for col in pragmatic_df.columns]
            print('========== PRAGMATIC TABLE ==========\n\n')
            print(pragmatic_df.to_latex(index=False))
            print('\n\n\n')
            pragmatic_final = pragmatic_df
        elif level == LinguisticLevel.LEXICAL:
            scores_compare = []
            scores_l = []
            tmps_all = []
            for r in relevant_percs:
                tmp = r[1][lexical_cols]
                # add length and improved length to tmp
                # check if original_length is in cols - if not, add it, copy values from "length" col
                if 'original_length' not in r[1].columns:
                    r[1]['original_length'] = r[1]['length']
                tmp['original_length'] = r[1]['original_length']
                tmp['improved_length'] = r[1]['improved_length']

                levenshtein = r[2][['levenshtein_levenshtein']]

                # also get all syllable_count cols
                counts = [col for col in r[1].columns if 'linguaf_number_of_' in col]
                # get max syllable count
                max_syllables = int(max(extract_first_number(col) for col in counts))
                count1to3 = 0
                count4to6 = 0
                count7to10 = 0
                count10plus = 0
                cols1to3 = [f'linguaf_number_of_{n}_syllable_words' for n in range(1, 4)]
                count1to3 = r[1][cols1to3].sum(axis=1) / 3

                cols4to6 = [f'linguaf_number_of_{n}_syllable_words' for n in range(4, 7)]
                count4to6 = r[1][cols4to6].sum(axis=1) / 3

                cols7to10 = [f'linguaf_number_of_{n}_syllable_words' for n in range(7, min(11, max_syllables))]
                count7to10 = r[1][cols7to10].sum(axis=1) / 3

                cols10plus = [f'linguaf_number_of_{n}_syllable_words' for n in range(10, max_syllables + 1)]
                count10plus = r[1][cols10plus].sum(axis=1) / 3

                tmp['count1to3'] = count1to3
                tmp['count4to6'] = count4to6
                tmp['count7to10'] = count7to10
                tmp['count10plus'] = count10plus

                tmp.replace([np.inf, -np.inf], np.nan, inplace=True)

                tmps_all.append(tmp.copy())

                means = tmp.mean()
                # multiply all except length and improved_length by 100
                means = means * 100
                means['length'] /= 100
                means['improved_length'] /= 100

                # calculate length change in percentage
                means['length_change'] = ((means['improved_length'] - means['length']) / means['length']) * 100
                # remove length cols from means
                means = means.drop(['length', 'improved_length'])

                scores_compare.append(means)

                levenshtein.replace([np.inf, -np.inf], np.nan, inplace=True)
                scores_l.append(levenshtein.mean())

            # concat tmps_all
            tmps_all = pd.concat(tmps_all)
            # correlation: tmps_all.corr()
            lexical_corrs_all = tmps_all.corr()
            # corrs, ps = calculate_pearsonr_correlation(tmps_all)
            lexical_corrs, lexical_ps = calculate_pearsonr_correlation(tmps_all)
            lexical_means = tmps_all.mean()

            compare_dict = {ds: df for ds, df in zip(ds_names, scores_compare)}
            compare_df = pd.DataFrame(compare_dict)
            levenshtein_df = pd.DataFrame({ds: df for ds, df in zip(ds_names, scores_l)})

            levenshtein_df_table = levenshtein_df.copy()
            levenshtein_df_table.reset_index(inplace=True)
            # round numeric cols to 2 decimals and convert to str
            levenshtein_df_table.iloc[:, 1:] = levenshtein_df_table.iloc[:, 1:].applymap(lambda x: f'{x:.2f}')
            # levenshtein_df_table['score_name'] = levenshtein_df_table['score_name'].apply(lambda x: x.replace('_', '\_'))
            levenshtein_df_table.columns = [col.replace('_', '\_') for col in levenshtein_df_table.columns]
            print('========== LEXICAL LEVENSHTEIN TABLE ==========\n\n')
            print(levenshtein_df_table.to_latex(index=False))

            create_heatmap(compare_df, 'lexical_compare')
            create_heatmap(levenshtein_df, 'lexical_levenshtein')
            final_df = pd.concat([compare_df, levenshtein_df])

            final_df = final_df.round(2)
            final_df = final_df.applymap(lambda x: f'{x:.2f}')

            final_df.index.name = 'score_name'
            final_df.reset_index(inplace=True)

            final_df['score_name'] = final_df['score_name'].apply(lambda x: x.replace('_', '\_'))
            final_df.columns = [col.replace('_', '\_') for col in final_df.columns]
            print('========== LEXICAL TABLE ==========\n\n')
            print(final_df.to_latex(index=False))
            print('\n\n\n')
            lexical_final = final_df
        elif level == LinguisticLevel.ARGMINING:
            # get relevant files
            relevant_files = []
            if isinstance(approach_filter, str):
                approach_filter = [approach_filter]
            for model_name in model_out_name:
                for approach in approach_filter:
                    for dataset in datasets:
                        relevant_files.append(
                            (model_name, approach, dataset, f'{model_name}_{dataset}__{approach}_comparison.csv'))

            human_files = []
            for dataset in datasets:
                human_files.append((dataset, f'HUMAN_{dataset}__comparison.csv'))
            human_dfs = {}
            for dataset, hf in human_files:
                p = os.path.join(current_file_dir, 'scores_out', 'ARG_MINING_SCORES', hf)
                df = pd.read_csv(p, index_col=0)
                human_dfs[dataset] = df

            deltas = []
            cols = ['Claim', 'Premise', 'MajorClaim', 'None']
            model_dfs = []
            for model_name, approach, dataset, fname in relevant_files:
                # open file
                p = os.path.join(current_file_dir, 'scores_out', 'ARG_MINING_SCORES', fname)
                df = pd.read_csv(p, index_col=0)
                # check if a col is missing
                for col in cols:
                    if col not in df.columns:
                        df[col] = 0
                delta = df - human_dfs[dataset]
                deltas.append(delta.mean())
                model_dfs.append(df)
            for d, ds_name in zip(deltas, ds_names):
                d['dataset'] = ds_name
            # concat them all
            final_df = pd.concat(deltas, axis=1).T
            final_df = final_df.set_index('dataset').T

            # round all values to 2 decimals
            final_df = final_df.round(2)
            final_df = final_df.applymap(lambda x: f'{x:.2f}')
            print('========== ARGMINING TABLE ==========\n\n')
            print(final_df.to_latex(index=True))

            argmining_final = final_df
    print('FOO!')
    return lexical_final, syntactic_final, semantic_final, pragmatic_final, argmining_final


if __name__ == '__main__':
    # sentiment_analysis()
    # metrics_analysis('BERTALIGN_SCORES')
    # metrics_analysis('FENG_HIRST_SCORES')
    # metrics_analysis('GRUEN_SCORES')
    # metrics_analysis('LEVENSHTEIN_SCORES')
    # metrics_analysis('LEXICAL_SCORES')
    # metrics_analysis('LINGUAF_SCORES')
    # metrics_analysis('AMERICANO_SCORES')
    # get_columns_all()
    # linguistic_level_analysis(
    #     LinguisticLevel.SEMANTIC,
    #     model_filter='llama3_nemotron_cleaned',
    #     approach_filter='direct',
    #     do_deltas=False
    # )
    # make_overview_table()
    # linguistic_level_analysis(LinguisticLevel.SEMANTIC, model_filter='llama3_nemotron_cleaned')

    # process_model_for_analysis('direct', 'olmo', 'OLMo-7B-0724-Instruct-hf-cleaned')
    # process_model_for_analysis('direct', 'bloomz3b', 'bloomz-3b-cleaned')
    # process_model_for_analysis('direct', 'bloomz560m', 'bloomz-560m-cleaned')
    #
    # for a_ in approaches:
    #     print(f'> DOING ANALYSIS LOOP {a_}')
    #     process_model_for_analysis(a_, 'LLAMA3', 'llama3_nemotron_cleaned')
    #     process_model_for_analysis(a_, 'olmo', 'OLMo-7B-0724-Instruct-hf-cleaned')
    #     process_model_for_analysis(a_, 'bloomz3b', 'bloomz-3b-cleaned')
    #     process_model_for_analysis(a_, 'bloomz560m', 'bloomz-560m-cleaned')

    # llama_analysis_table()

    # get_revision_lengths()
    # get_essays_lengths()
    # get_microtext_lengths()
    # get_improved_lens_all()

    # this is done!!
    # process_model_for_analysis_SAVE_ALL('llama3_nemotron_cleaned', 'llama3_nemotron_cleaned')
    # process_model_for_analysis_SAVE_ALL('OLMo-7B-0724-Instruct-hf-cleaned', 'OLMo-7B-0724-Instruct-hf-cleaned')
    # process_model_for_analysis_SAVE_ALL('bloomz-3b-cleaned', 'bloomz-3b-cleaned')
    # process_model_for_analysis_SAVE_ALL('bloomz-560m-cleaned', 'bloomz-560m-cleaned')

    # RUN THIS LATER WHEN SCORES ARE DONE
    # process_model_for_analysis_SAVE_ALL('Phi-3-medium-4k-instruct-cleaned', 'Phi-3-medium-4k-instruct-cleaned')
    # process_model_for_analysis_SAVE_ALL('llama3_nemotron_cleaned', 'llama3_nemotron_cleaned', approach_filter='direct')

    # process_model_for_analysis_SAVE_ALL('OLMo-7B-0724-Instruct-hf-cleaned', 'OLMo-7B-0724-Instruct-hf-cleaned', approach_filter='direct')
    # raise Exception('DONE')

    # measure_sentiment_changes()
    # process_model_for_analysis_SAVE_ALL('OLMo-7B-0724-Instruct-hf-cleaned', 'OLMo-7B-0724-Instruct-hf-cleaned')

    # metrics_analysis('LEXICAL_SCORES')

    # CORRELATIONS COME FROM THIS !!!
    lexical_final_llama, syntactic_final_llama, semantic_final_llama, pragmatic_final_llama, argmining_final_llama = general_analysis_table(
        ['llama3_nemotron_cleaned', 'OLMo-7B-0724-Instruct-hf-cleaned', 'Phi-3-medium-4k-instruct-cleaned', 'Phi-3-mini-4k-instruct-cleaned'],
        approach_filter='direct',
        levels=[LinguisticLevel.LEXICAL, LinguisticLevel.SYNTACTIC, LinguisticLevel.SEMANTIC, LinguisticLevel.PRAGMATIC]
    )
    print('!!')

    # USE THIS FOR TABLES AND HEATMAPS
    lexical_final_llama, syntactic_final_llama, semantic_final_llama, pragmatic_final_llama, argmining_final_llama = general_analysis_table(
        ['llama3_nemotron_cleaned'],
        approach_filter='direct',
        levels=[LinguisticLevel.LEXICAL, LinguisticLevel.SYNTACTIC, LinguisticLevel.SEMANTIC, LinguisticLevel.PRAGMATIC]
    )
    # lexical_final_olmo, syntactic_final_olmo, semantic_final_olmo, pragmatic_final_olmo, argmining_final_olmo = general_analysis_table(
    #     ['OLMo-7B-0724-Instruct-hf-cleaned'],
    #     approach_filter='little_brother',
    #     # levels=[LinguisticLevel.ARGMINING]
    # )
    # lexical_final_bloomz3b, syntactic_final_bloomz3b, semantic_final_bloomz3b, pragmatic_final_bloomz3b, argmining_final_bloomz3b = general_analysis_table(
    #     ['bloomz-3b-cleaned'],
    #     approach_filter='little_brother',
    #     # levels=[LinguisticLevel.ARGMINING]
    # )
    # lexical_final_bloomz560m, syntactic_final_bloomz560m, semantic_final_bloomz560m, pragmatic_final_bloomz560m, argmining_final_bloomz560m = general_analysis_table(
    #     ['bloomz-560m-cleaned'],
    #     approach_filter='little_brother',
    #     # levels=[LinguisticLevel.ARGMINING]
    # )
    # lexical_final_phi3mini, syntactic_final_phi3mini, semantic_final_phi3mini, pragmatic_final_phi3mini, argmining_final_phi3mini = general_analysis_table(
    #     ['Phi-3-mini-4k-instruct-cleaned'],
    #     approach_filter='self_discover',
    #     # levels=[LinguisticLevel.ARGMINING]
    # )
    # lexical_final_phi3medium, syntactic_final_phi3medium, semantic_final_phi3medium, pragmatic_final_phi3medium, argmining_final_phi3medium = general_analysis_table(
    #     ['Phi-3-medium-4k-instruct-cleaned'],
    #     approach_filter='little_brother',
    #     # levels=[LinguisticLevel.ARGMINING]
    # )
    print('FOO!')

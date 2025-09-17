import json
import os

import pandas as pd
from tqdm import tqdm

from direct.direct import DirectPrompt
from projects.improvement_pipeline.improvement_pipeline import load_arg_rewrite_df
from projects.text_generation_scores.bertalign_scores.align_improvements import do_bertalign_metrics
import spacy

from projects.text_generation_scores.complex_structures.annotate import calculate_complex_scores, \
    ComplexSentenceAnnotations
from projects.text_generation_scores.feng_hirst_scores import feng_hirst_scores
from projects.text_generation_scores.gruen_score.score import calculate_gruen_score
from projects.text_generation_scores.levenshtein_scores import levenshtein_scores
from projects.text_generation_scores.lexical_richness_scores import lexical_richness_scores
from projects.text_generation_scores.linguaf_scores import linguaf_scores
from projects.text_generation_scores.semantic_text_similarity_scores import semantic_text_similarity_scores
from util.template import llama3_format_func

current_file_dir = os.path.dirname(os.path.abspath(__file__))


def save_intermediate_results(
        out: dict,
        out_dir: str,
        out_file_name: str
):
    comparison_cols = [k for k in out.keys() if k.endswith('original') or k.endswith('improved')]
    corpus_cols = [k for k in out.keys() if k not in comparison_cols]
    comparison_df = pd.DataFrame({k: out[k] for k in comparison_cols})
    corpus_df = pd.DataFrame({k: out[k] for k in corpus_cols if
                              not k.endswith('syllable_words_percentage_change')})
    syllable_change_dict = {k: out[k] for k in corpus_cols if k.endswith('syllable_words_percentage_change')}
    comparison_df.to_csv(os.path.join(out_dir, f'{out_file_name}_comparison.csv'))
    corpus_df.to_csv(os.path.join(out_dir, f'{out_file_name}_corpus.csv'))
    with open(os.path.join(out_dir, f'{out_file_name}_syllable_changes.json'), 'w') as f:
        json.dump(syllable_change_dict, f, indent=4)


def check_already_done(
        out_dir: str,
        out_identifier: str
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return os.path.exists(os.path.join(out_dir, f'{out_identifier}_comparison.csv')) and \
        os.path.exists(os.path.join(out_dir, f'{out_identifier}_corpus.csv')) and \
        os.path.exists(os.path.join(out_dir, f'{out_identifier}_syllable_changes.json'))


def do_bertalign_scores(
        original_texts: list[str],
        improved_texts: list[str],
        out_identifier: str,
        nlp: spacy.Language,
        src_lng: str = 'en',
        tgt_lng: str = 'en',
):
    # does bertalign and complex scores
    out_dir = os.path.join(current_file_dir, 'scores_out', 'BERTALIGN_SCORES')
    done = check_already_done(out_dir, out_identifier)
    if done:
        print(f'{out_identifier} already exists, skipping, bertalign')
        return

    bertalign_scores = []
    complex_annotations_original_all = []
    complex_annotations_improved_all = []
    out = {}
    for i, (original, improved) in enumerate(zip(original_texts, improved_texts)):
        if not original.strip() or not improved.strip():
            bertalign_scores.append(None)
            complex_annotations_original_all.append(None)
            complex_annotations_improved_all.append(None)
            continue
        try:
            bertalign_score = do_bertalign_metrics(
                original=original,
                improved=improved,
                src_lng=src_lng,
                tgt_lng=tgt_lng,
            )
        except IndexError:
            bertalign_scores.append(None)
            complex_annotations_original_all.append(None)
            complex_annotations_improved_all.append(None)
            continue
        complex_annotations_original = calculate_complex_scores(original, nlp)
        complex_annotations_improved = calculate_complex_scores(improved, nlp)
        bertalign_scores.append(bertalign_score)
        complex_annotations_original_all.append(complex_annotations_original)
        complex_annotations_improved_all.append(complex_annotations_improved)

    out['bertalign_scores_add'] = []
    out['bertalign_scores_copy'] = []
    out['bertalign_scores_delete'] = []
    out['bertalign_scores_fusion'] = []
    out['bertalign_scores_merge'] = []
    out['bertalign_scores_other'] = []
    for score in bertalign_scores:
        if score is None:
            out['bertalign_scores_add'].append(None)
            out['bertalign_scores_copy'].append(None)
            out['bertalign_scores_delete'].append(None)
            out['bertalign_scores_fusion'].append(None)
            out['bertalign_scores_merge'].append(None)
            out['bertalign_scores_other'].append(None)
        else:
            d = score.to_dict()
            out['bertalign_scores_add'].append(d['add'])
            out['bertalign_scores_copy'].append(d['copy'])
            out['bertalign_scores_delete'].append(d['delete'])
            out['bertalign_scores_fusion'].append(d['fusion'])
            out['bertalign_scores_merge'].append(d['merge'])
            out['bertalign_scores_other'].append(d['other'])
    for score in complex_annotations_original_all:
        if score is None:
            score = ComplexSentenceAnnotations(
                num_relcl=None,
                num_advcl=None,
                num_appos=None,
                num_prep=None,
                num_coordNP=None,
                num_coord_cl=None,
                num_coordVP=None,
                num_speech=None,
                num_adv_mod=None,
                num_part=None,
                complex_dict={},
                sent=None
            )
        tmp = score.to_dict()
        for k, v in tmp.items():
            if not k.startswith('num'):
                continue
            if f'complex_annotation_scores_{k}_original' not in out:
                out[f'complex_annotation_scores_{k}_original'] = []
            out[f'complex_annotation_scores_{k}_original'].append(v)
    for score in complex_annotations_improved_all:
        if score is None:
            score = ComplexSentenceAnnotations(
                num_relcl=None,
                num_advcl=None,
                num_appos=None,
                num_prep=None,
                num_coordNP=None,
                num_coord_cl=None,
                num_coordVP=None,
                num_speech=None,
                num_adv_mod=None,
                num_part=None,
                complex_dict={},
                sent=None
            )
        tmp = score.to_dict()
        for k, v in tmp.items():
            if not k.startswith('num'):
                continue
            if f'complex_annotation_scores_{k}_improved' not in out:
                out[f'complex_annotation_scores_{k}_improved'] = []
            out[f'complex_annotation_scores_{k}_improved'].append(v)
    save_intermediate_results(
        out=out,
        out_dir=out_dir,
        out_file_name=out_identifier
    )


def do_gruen_scores(
        original_texts: list[str],
        improved_texts: list[str],
        nlp: spacy.Language,
        out_identifier: str
):
    out_dir = os.path.join(current_file_dir, 'scores_out', 'GRUEN_SCORES')
    done = check_already_done(out_dir, out_identifier)
    if done:
        print(f'{out_identifier} already exists, skipping, GRUEN scores')
        return
    out = {}
    # gruen_scores_all: list of scores, one score per sentence
    # gruen_scores_per_text: list of lists, one list per text, then one score per sentence
    gruen_scores_all_orig, gruen_scores_per_text_orig = calculate_gruen_score(original_texts, nlp)
    gruen_scores_per_text_orig_average = [sum(scores) / len(scores) for scores in gruen_scores_per_text_orig]
    out['gruen_scores_original'] = gruen_scores_per_text_orig_average

    gruen_scores_all_imp, gruen_scores_per_text_imp = calculate_gruen_score(improved_texts, nlp)
    gruen_scores_per_text_imp_average = [sum(scores) / len(scores) for scores in gruen_scores_per_text_imp]
    out['gruen_scores_improved'] = gruen_scores_per_text_imp_average
    save_intermediate_results(
        out=out,
        out_dir=out_dir,
        out_file_name=out_identifier
    )


def do_feng_hirst_scores(
        original_texts: list[str],
        improved_texts: list[str],
        feng_hirst_output_dir: str,
        feng_hirst_identifier: str,
        out_identifier: str
):
    out_dir = os.path.join(current_file_dir, 'scores_out', 'FENG_HIRST_SCORES')
    done = check_already_done(out_dir, out_identifier)
    if done:
        print(f'{out_identifier} already exists, skipping, FENG HIRST scores')
        return
    out = {}
    # dataframe, do each row and then to_dict()?
    # tmp_dfs = []
    # for count, (o, i) in enumerate(zip(original_texts, improved_texts)):
    #     print(f'> Feng Hirst one by one - count: {count} (total: {len(original_texts)}')
    #     # this WILL overwrite the feng hirst files - should not be relevant for us
    #     tmp = feng_hirst_scores([o], [i], feng_hirst_output_dir, feng_hirst_identifier)
    #     tmp_dfs.append(tmp)
    # # merge all dataframes
    # feng_hirst_df = pd.concat(tmp_dfs, ignore_index=True)

    # old - commented out because it randomly freezes - trying new approach above - doing it one by one
    # after the fixes, this should be fine and work properly
    feng_hirst_df = feng_hirst_scores(original_texts, improved_texts, feng_hirst_output_dir, feng_hirst_identifier)

    for i, row in feng_hirst_df.iterrows():
        tmp = row.to_dict()
        for k, v in tmp.items():
            if 'original' in k:
                identifier = f'feng_hirst_{k}_original'
            elif 'improved' in k:
                identifier = f'feng_hirst_{k}_improved'
            else:
                raise ValueError('Neither original nor improved in column name')
            if identifier not in out:
                out[identifier] = []
            out[identifier].append(v)
    save_intermediate_results(
        out=out,
        out_dir=out_dir,
        out_file_name=out_identifier
    )


def do_levenshtein_scores(
        original_texts: list[str],
        improved_texts: list[str],
        out_identifier: str
):
    out_dir = os.path.join(current_file_dir, 'scores_out', 'LEVENSHTEIN_SCORES')
    done = check_already_done(out_dir, out_identifier)
    if done:
        print(f'{out_identifier} already exists, skipping, levenshtein scores')
        return
    out = {}
    levenshtein_dict = levenshtein_scores(original_texts, improved_texts)
    for k, v in levenshtein_dict.items():
        out[f'levenshtein_{k}'] = v
    save_intermediate_results(
        out=out,
        out_dir=out_dir,
        out_file_name=out_identifier
    )


def do_lexical_scores(
        original_texts: list[str],
        improved_texts: list[str],
        out_identifier: str
):
    out_dir = os.path.join(current_file_dir, 'scores_out', 'LEXICAL_SCORES')
    done = check_already_done(out_dir, out_identifier)
    if done:
        print(f'{out_identifier} already exists, skipping, lexical scores')
        return
    out = {}
    lexical_percentage_changes, lexical_absolute_changes, scores_original, scores_improved = lexical_richness_scores(
        original_texts, improved_texts
    )
    for k, v in scores_original.items():
        out[f'lexical_{k}_original'] = v
    for k, v in scores_improved.items():
        out[f'lexical_{k}_improved'] = v
    save_intermediate_results(
        out=out,
        out_dir=out_dir,
        out_file_name=out_identifier
    )


def do_linguaf_scores(
        original_texts: list[str],
        improved_texts: list[str],
        out_identifier: str,
        src_lng: str = 'en',
        tgt_lng: str = 'en',
):
    out_dir = os.path.join(current_file_dir, 'scores_out', 'LINGUAF_SCORES')
    done = check_already_done(out_dir, out_identifier)
    if done:
        print(f'{out_identifier} already exists, skipping, linguaf scores')
        return
    out = {}
    # all are dicts - number_of_n_syllable_words_all is another dict
    linguaf_scores_original, linguaf_scores_improved, linguaf_percentage_changes, linguaf_absolute_changes, scores_original_singles_dct, scores_improved_singles_dct = linguaf_scores(
        original_texts, improved_texts, src_lng, tgt_lng
    )
    # number_of_n_syllable_words_all
    for metric, score in scores_original_singles_dct.items():
        out[f'linguaf_{metric}_original'] = score
    for metric, score in scores_improved_singles_dct.items():
        out[f'linguaf_{metric}_improved'] = score

    # find max number of syllables
    max_syllables = max(
        linguaf_percentage_changes['number_of_n_syllable_words_all'].keys()
    )
    for i in range(0, max_syllables + 1):
        count = linguaf_percentage_changes['number_of_n_syllable_words_all'].get(i, 0)
        out[f'linguaf_number_of_{i}_syllable_words_percentage_change'] = count
    save_intermediate_results(
        out=out,
        out_dir=out_dir,
        out_file_name=out_identifier
    )


def do_semantic_similarity_scores(
        original_texts: list[str],
        improved_texts: list[str],
        out_identifier: str
):
    out_dir = os.path.join(current_file_dir, 'scores_out', 'SEMANTIC_SIMILARITY_SCORES')
    done = check_already_done(out_dir, out_identifier)
    if done:
        print(f'{out_identifier} already exists, skipping, semantic similarity scores')
        return
    out = {}
    # list of scores
    semantic_sim_scores = semantic_text_similarity_scores(original_texts, improved_texts)
    out['semantic_similarity_scores'] = semantic_sim_scores
    save_intermediate_results(
        out=out,
        out_dir=out_dir,
        out_file_name=out_identifier
    )


def do_americano_scores(
        original_texts: list[str],
        improved_texts: list[str],
        out_identifier: str,
        model_name: str,
        format_func: callable,
        num_runs: int = 5,
        prompttemplate=None
):
    out_dir = os.path.join(current_file_dir, 'scores_out', 'AMERICANO_SCORES')
    done = check_already_done(out_dir, out_identifier)
    if done:
        print(f'{out_identifier} already exists, skipping, americano scores')
        return
    os.environ['FORCE_VLLM_PRELOAD'] = '1'

    out = {}
    print('> running americano scores pipeline for original texts')
    formatted_prompts_coherence, formatted_prompts_persuasion, results_coherence, results_persuasion, scores_coherence, scores_persuasion, avgs_coherence, avgs_persuasion = calculate_americano_scores(
        template=prompttemplate,
        num_runs=num_runs,
        texts=original_texts
    )
    prompts_out_dir = os.path.join(current_file_dir, 'scores_out', 'AMERICANO_PROMPTS')
    o_df = pd.DataFrame({
        'coherence_prompts': formatted_prompts_coherence,
        'persuasion_prompts': formatted_prompts_persuasion,
        'coherence_results': results_coherence,
        'persuasion_results': results_persuasion,
    })
    o_df.to_csv(os.path.join(prompts_out_dir, f'{out_identifier}_prompts_original.csv'))
    out['americano_coherence_avgs_original'] = avgs_coherence
    out['americano_persuasion_avgs_original'] = avgs_persuasion

    print('> running americano scores pipeline for improved texts')
    formatted_prompts_coherence, formatted_prompts_persuasion, results_coherence, results_persuasion, scores_coherence, scores_persuasion, avgs_coherence, avgs_persuasion = calculate_americano_scores(
        template=prompttemplate,
        num_runs=num_runs,
        texts=improved_texts
    )
    i_df = pd.DataFrame({
        'coherence_prompts': formatted_prompts_coherence,
        'persuasion_prompts': formatted_prompts_persuasion,
        'coherence_results': results_coherence,
        'persuasion_results': results_persuasion,
    })
    i_df.to_csv(os.path.join(prompts_out_dir, f'{out_identifier}_prompts_improved.csv'))
    out['americano_coherence_avgs_improved'] = avgs_coherence
    out['americano_persuasion_avgs_improved'] = avgs_persuasion
    save_intermediate_results(
        out=out,
        out_dir=out_dir,
        out_file_name=out_identifier
    )


def do_all_scores_separately(
        original_texts: list[str],
        improved_texts: list[str],
        feng_hirst_output_dir: str,
        feng_hirst_identifier: str,
        out_identifier: str,
        src_lng: str = 'en',
        tgt_lng: str = 'en',
        spacy_model_name='en_core_web_lg',
        prompttemplate=None
):
    nlp = spacy.load(spacy_model_name)
    # print('> BERTALIGN scores')
    # do_bertalign_scores(
    #     original_texts=original_texts,
    #     improved_texts=improved_texts,
    #     out_identifier=out_identifier,
    #     nlp=nlp,
    #     src_lng=src_lng,
    #     tgt_lng=tgt_lng,
    # )
    # print('> GRUEN scores')
    # do_gruen_scores(
    #     original_texts=original_texts,
    #     improved_texts=improved_texts,
    #     nlp=nlp,
    #     out_identifier=out_identifier
    # )
    # print('> Feng Hirst scores')
    # do_feng_hirst_scores(
    #     original_texts=original_texts,
    #     improved_texts=improved_texts,
    #     feng_hirst_output_dir=feng_hirst_output_dir,
    #     feng_hirst_identifier=feng_hirst_identifier,
    #     out_identifier=out_identifier
    # )
    # print('> Levenshtein scores')
    # do_levenshtein_scores(
    #     original_texts=original_texts,
    #     improved_texts=improved_texts,
    #     out_identifier=out_identifier
    # )
    # print('> Lexical scores')
    # do_lexical_scores(
    #     original_texts=original_texts,
    #     improved_texts=improved_texts,
    #     out_identifier=out_identifier
    # )
    # print('> Linguaf scores')
    # do_linguaf_scores(
    #     original_texts=original_texts,
    #     improved_texts=improved_texts,
    #     out_identifier=out_identifier,
    #     src_lng=src_lng,
    #     tgt_lng=tgt_lng,
    # )
    # print('> Semantic similarity scores')
    # do_semantic_similarity_scores(
    #     original_texts=original_texts,
    #     improved_texts=improved_texts,
    #     out_identifier=out_identifier
    # )
    print('> AMERICANO scores')
    do_americano_scores(
        original_texts=original_texts,
        improved_texts=improved_texts,
        out_identifier=out_identifier,
        model_name='llama3',
        format_func=llama3_format_func,
        prompttemplate=prompttemplate
    )


if __name__ == '__main__':
    rewrite_df = load_arg_rewrite_df()
    folder = 'HUMAN_REVISIONS'
    file_name1 = 'human_revision1to2.json'
    file_name2 = 'human_revision2to3.json'

    original_1to2 = rewrite_df['draft1'].tolist()
    improved_1to2 = rewrite_df['draft2'].tolist()

    original_2to3 = rewrite_df['draft2'].tolist()
    improved_2to3 = rewrite_df['draft3'].tolist()

    direct = DirectPrompt(
        model='llama3',
        dtype='half',
        temperature=0.7,
        use_vllm=False
    )

    do_all_scores_separately(
        original_1to2,
        improved_1to2,
        'feng_hirst_tmp',
        f'{folder}_HUMAN_REVISION_1to2',
        f'HUMAN_REVISION_1to2__{folder}',
        prompttemplate=direct
    )

    do_all_scores_separately(
        original_2to3,
        improved_2to3,
        'feng_hirst_tmp',
        f'{folder}_HUMAN_REVISION_2to3',
        f'HUMAN_REVISION_2to3__{folder}',
        prompttemplate=direct
    )

    # _test_file = os.path.join(current_file_dir, '..', 'improved_out', 'MICROTEXTS', 'llama3_nemotron_cleaned.json')
    # with open(_test_file, 'r') as file:
    #     data = json.load(file)
    # _direct = data[0]
    #
    # _original = _direct['original_arguments'][:3]
    # _improved = _direct['cleaned_arguments'][:3]
    # do_all_scores(_original, _improved, 'feng_hirst_tmp', 'test')

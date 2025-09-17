import json
import os
from collections import defaultdict
from functools import partial

import math
from linguaf import descriptive_statistics as ds
from linguaf import lexical_diversity as ld
from linguaf import readability as r
from linguaf import syntactical_complexity as sc


def linguaf_scores_single(
        texts: list[str],
        lang: str = 'en'
):
    score_funcs = [
        ds.char_count,
        ds.letter_count,
        ds.punctuation_count,
        ds.digit_count,
        ds.syllable_count,
        ds.sentence_count,
        ds.number_of_n_syllable_words_all,
        ds.avg_word_length,
        ds.avg_sentence_length,
        ds.avg_words_per_sentence,
        ld.lexical_density,
        ld.type_token_ratio,
        ld.log_type_token_ratio,
        ld.summer_index,
        ld.root_type_token_ratio,
        r.flesch_kincaid_grade,
        r.flesch_reading_ease,
        r.automated_readability_index,
        r.automated_readability_index_simple,
        r.coleman_readability,
        r.easy_listening,
        # sc.mean_dependency_distance,
    ]
    scores = []
    for score_func in score_funcs:
        # check if func has a 'lang' param
        if 'lang' in score_func.__code__.co_varnames:
            try:
                score_original = score_func(texts, lang=lang)
            except Exception:
                score_original = None
            scores.append(score_original)
            continue
        try:
            scores.append(score_func(texts))
        except Exception:
            scores.append(None)
    scores_dict = {
        'char_count': scores[0],
        'letter_count': scores[1],
        'punctuation_count': scores[2],
        'digit_count': scores[3],
        'syllable_count': scores[4],
        'sentence_count': scores[5],
        'number_of_n_syllable_words_all': scores[6],
        'avg_word_length': scores[7],
        'avg_sentence_length': scores[8],
        'avg_words_per_sentence': scores[9],
        'lexical_density': scores[10],
        'type_token_ratio': scores[11],
        'log_type_token_ratio': scores[12],
        'summer_index': scores[13],
        'root_type_token_ratio': scores[14],
        'flesch_kincaid_grade': scores[15],
        'flesch_reading_ease': scores[16],
        'automated_readability_index': scores[17],
        'automated_readability_index_simple': scores[18],
        'coleman_readability': scores[19],
        'easy_listening': scores[20],
        # 'mean_dependency_distance': scores[21]
    }
    return scores_dict


def linguaf_scores(
        original_texts: list[str],
        improved_texts: list[str],
        src_lng: str = 'en',
        tgt_lng: str = 'en',
):
    scores_original_all = linguaf_scores_single(original_texts, lang=src_lng)
    scores_improved_all = linguaf_scores_single(improved_texts, lang=tgt_lng)
    scores_original_singles = []
    scores_improved_singles = []
    for ot in original_texts:
        scores_original_singles.append(linguaf_scores_single([ot], lang=src_lng))
    for it in improved_texts:
        scores_improved_singles.append(linguaf_scores_single([it], lang=tgt_lng))
    scores_original_singles_dct = {}
    scores_improved_singles_dct = {}

    # original and improved share metrics - keys are the same, can use the shortcut here to initialize empty dicts
    for metric in scores_original_all.keys():
        if not metric in scores_original_singles_dct:
            scores_original_singles_dct[metric] = []
            scores_improved_singles_dct[metric] = []
        for i, score in enumerate(scores_original_singles):
            scores_original_singles_dct[metric].append(score[metric])
        for i, score in enumerate(scores_improved_singles):
            scores_improved_singles_dct[metric].append(score[metric])


    # find max number of syllables
    max_syllables_original = -1
    for counts in scores_original_singles_dct['number_of_n_syllable_words_all']:
        max_syllables_original = max(max_syllables_original, max(counts.keys()))

    syllable_data_original = scores_original_singles_dct.pop('number_of_n_syllable_words_all')
    syllable_counts_flattened = {}
    for counts_dct in syllable_data_original:
        for i in range(1, max_syllables_original + 1):
            count = counts_dct.get(i, 0)
            if i not in syllable_counts_flattened:
                syllable_counts_flattened[i] = []
            syllable_counts_flattened[i].append(count)
    for k, v in syllable_counts_flattened.items():
        scores_original_singles_dct[f'number_of_{k}_syllable_words'] = v

    # find max number of syllables
    max_syllables_improved = -1
    for counts in scores_improved_singles_dct['number_of_n_syllable_words_all']:
        if counts is None:
            max_syllables_improved = max(max_syllables_improved, -1)
        else:
            max_syllables_improved = max(max_syllables_improved, max(counts.keys()))

    syllable_data_improved = scores_improved_singles_dct.pop('number_of_n_syllable_words_all')
    syllable_counts_flattened = {}
    for counts_dct in syllable_data_improved:
        for i in range(1, max_syllables_improved + 1):
            if counts_dct is None:
                count = 0
            else:
                count = counts_dct.get(i, 0)
            if i not in syllable_counts_flattened:
                syllable_counts_flattened[i] = []
            syllable_counts_flattened[i].append(count)
    for k, v in syllable_counts_flattened.items():
        scores_improved_singles_dct[f'number_of_{k}_syllable_words'] = v

    percentage_changes = []
    absolute_changes = []
    for metric in scores_original_all.keys():
        score_original = scores_original_all[metric]
        score_improved = scores_improved_all[metric]
        if metric == 'number_of_n_syllable_words_all':
            syllable_counts_all = set(score_original.keys()) | set(score_improved.keys())
            count_percentage_changes = defaultdict(int)
            count_absolute_changes = defaultdict(int)
            total_count_original = sum(score_original.values())
            total_count_improved = sum(score_improved.values())
            for syllable_count in syllable_counts_all:
                count_original = score_original[syllable_count]
                count_original_percent = count_original / total_count_original * 100

                count_improved = score_improved[syllable_count]
                count_improved_percent = count_improved / total_count_improved * 100
                if count_original == 0:
                    percentage_change = math.inf
                else:
                    percentage_change = (count_improved - count_original) / count_original * 100
                count_percentage_changes[syllable_count] = percentage_change

                # note - this counts IN PERCENT how it has changed
                # if originally 20% were 1-syllable words, and now it's 30%, then this value will be +10%
                count_absolute_changes[syllable_count] = count_improved_percent - count_original_percent
            percentage_changes.append(count_percentage_changes)
            absolute_changes.append(count_absolute_changes)
            continue
        try:
            percentage_change = (score_improved - score_original) / score_original * 100
        except ZeroDivisionError:
            if score_improved > 0:
                percentage_change = math.inf
            elif score_improved < 0:
                percentage_change = -math.inf
            else:
                percentage_change = 0
        except TypeError:
            if score_original is None and score_improved is not None:
                percentage_change = math.inf
            elif score_original is not None and score_improved is None:
                percentage_change = -math.inf
            else:
                percentage_change = 0
        percentage_changes.append(percentage_change)
        if score_improved is None or score_original is None:
            absolute_changes.append(None)
        else:
            absolute_changes.append(score_improved - score_original)
    percentage_changes = {
        'char_count': percentage_changes[0],
        'letter_count': percentage_changes[1],
        'punctuation_count': percentage_changes[2],
        'digit_count': percentage_changes[3],
        'syllable_count': percentage_changes[4],
        'sentence_count': percentage_changes[5],
        'number_of_n_syllable_words_all': percentage_changes[6],
        'avg_word_length': percentage_changes[7],
        'avg_sentence_length': percentage_changes[8],
        'avg_words_per_sentence': percentage_changes[9],
        'lexical_density': percentage_changes[10],
        'type_token_ratio': percentage_changes[11],
        'log_type_token_ratio': percentage_changes[12],
        'summer_index': percentage_changes[13],
        'root_type_token_ratio': percentage_changes[14],
        'flesch_kincaid_grade': percentage_changes[15],
        'flesch_reading_ease': percentage_changes[16],
        'automated_readability_index': percentage_changes[17],
        'automated_readability_index_simple': percentage_changes[18],
        'coleman_readability': percentage_changes[19],
        'easy_listening': percentage_changes[20],
        # 'mean_dependency_distance': percentage_changes[21]
    }
    absolute_changes = {
        'char_count': absolute_changes[0],
        'letter_count': absolute_changes[1],
        'punctuation_count': absolute_changes[2],
        'digit_count': absolute_changes[3],
        'syllable_count': absolute_changes[4],
        'sentence_count': absolute_changes[5],
        'number_of_n_syllable_words_all': absolute_changes[6],
        'avg_word_length': absolute_changes[7],
        'avg_sentence_length': absolute_changes[8],
        'avg_words_per_sentence': absolute_changes[9],
        'lexical_density': absolute_changes[10],
        'type_token_ratio': absolute_changes[11],
        'log_type_token_ratio': absolute_changes[12],
        'summer_index': absolute_changes[13],
        'root_type_token_ratio': absolute_changes[14],
        'flesch_kincaid_grade': absolute_changes[15],
        'flesch_reading_ease': absolute_changes[16],
        'automated_readability_index': absolute_changes[17],
        'automated_readability_index_simple': absolute_changes[18],
        'coleman_readability': absolute_changes[19],
        'easy_listening': absolute_changes[20],
        # 'mean_dependency_distance': absolute_changes[21]
    }
    return scores_original_all, scores_improved_all, percentage_changes, absolute_changes, scores_original_singles_dct, scores_improved_singles_dct

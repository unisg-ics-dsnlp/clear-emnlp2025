import json
import os

import math
from lexicalrichness import LexicalRichness


def lexical_richness_scores_single(
        texts: list[str],
):
    score_dict = {
        'words': [],
        'terms': [],
        'ttr': [],
        'rttr': [],
        'cttr': [],
        'msttr': [],
        'mattr': [],
        'mtld': [],
        'hdd': [],
        'vocd': [],
        'herdan': [],
        'summer': [],
        'dugast': [],
        'maas': [],
        'yulek': [],
        'yulei': [],
        'herdanvm': [],
        'simpsond': []
    }
    for text in texts:
        if not text.strip():
            scores = {
                'words': 0,
                'terms': 0,
                'ttr': 0,
                'rttr': 0,
                'cttr': 0,
                'msttr': 0,
                'mattr': 0,
                'mtld': 0,
                'hdd': 0,
                'vocd': 0,
                'herdan': 0,
                'summer': 0,
                'dugast': 0,
                'maas': 0,
                'yulek': 0,
                'yulei': 0,
                'herdanvm': 0,
                'simpsond': 0
            }
            for key, value in scores.items():
                score_dict[key].append(value)
            continue
        lex = LexicalRichness(text)

        try:
            msttr = lex.msttr(segment_window=5)
        except ValueError:
            msttr = 0
        try:
            mattr = lex.mattr(window_size=5)
        except ValueError:
            mattr = 0

        try:
            hdd = lex.hdd(draws=5)
        except ValueError:
            hdd = 0

        try:
            herdan = lex.Herdan
        except (ZeroDivisionError, ValueError):
            herdan = 0

        try:
            summer = lex.Summer
        except (ZeroDivisionError, ValueError):
            summer = 0

        try:
            dugast = lex.Dugast
        except (ZeroDivisionError, ValueError):
            dugast = 0

        try:
            maas = lex.Maas
        except (ZeroDivisionError, ValueError):
            maas = 0

        try:
            yulek = lex.yulek
        except (ZeroDivisionError, ValueError, KeyError):
            yulek = 0

        try:
            yulei = lex.yulei
        except (ZeroDivisionError, ValueError, KeyError):
            yulei = 0

        try:
            ttr = lex.ttr
        except (ZeroDivisionError, ValueError):
            ttr = 0

        try:
            rttr = lex.rttr
        except (ZeroDivisionError, ValueError):
            rttr = 0

        try:
            cttr = lex.cttr
        except (ZeroDivisionError, ValueError):
            cttr = 0

        try:
            mtld = lex.mtld(threshold=0.72)
        except (ZeroDivisionError, ValueError):
            mtld = 0

        try:
            herdanvm = lex.herdanvm
        except (ZeroDivisionError, ValueError, KeyError):
            herdanvm = 0

        try:
            simpsond = lex.simpsond
        except (ZeroDivisionError, ValueError, KeyError):
            simpsond = 0

        scores = {
            'words': lex.words,
            'terms': lex.terms,
            'ttr': ttr,
            'rttr': rttr,
            'cttr': cttr,
            'msttr': msttr,
            'mattr': mattr,
            'mtld': mtld,
            'hdd': hdd,
            'vocd': 0,  # lex.vocd(ntokens=10, within_sample=100, iterations=3),
            'herdan': herdan,
            'summer': summer,
            'dugast': dugast,
            'maas': maas,
            'yulek': yulek,
            'yulei': yulei,
            'herdanvm': herdanvm,
            'simpsond': simpsond
        }
        for key, value in scores.items():
            score_dict[key].append(value)
    return score_dict


def lexical_richness_scores(
        original_texts: list[str],
        improved_texts: list[str],
):
    scores_original = lexical_richness_scores_single(original_texts)
    scores_improved = lexical_richness_scores_single(improved_texts)
    percentage_changes = []
    absolute_changes = []
    for metric in scores_original.keys():
        score_original = scores_original[metric]
        score_improved = scores_improved[metric]

        score_original_sum = sum(score_original)
        score_improved_sum = sum(score_improved)

        score_original_avg = score_original_sum / len(score_original)
        score_improved_avg = score_improved_sum / len(score_improved)

        try:
            percentage_change = (score_improved_sum - score_original_sum) / score_original_sum * 100
        except (ZeroDivisionError, ValueError):
            if score_improved_sum == 0:
                percentage_change = 0
            else:
                percentage_change = math.inf
        percentage_changes.append(percentage_change)
        absolute_changes.append(score_improved_sum - score_original_sum)
    percentage_changes = {
        'words': percentage_changes[0],
        'terms': percentage_changes[1],
        'ttr': percentage_changes[2],
        'rttr': percentage_changes[3],
        'cttr': percentage_changes[4],
        'msttr': percentage_changes[5],
        'mattr': percentage_changes[6],
        'mtld': percentage_changes[7],
        'hdd': percentage_changes[8],
        'vocd': percentage_changes[9],
        'herdan': percentage_changes[10],
        'summer': percentage_changes[11],
        'dugast': percentage_changes[12],
        'maas': percentage_changes[13],
        'yulek': percentage_changes[14],
        'yulei': percentage_changes[15],
        'herdanvm': percentage_changes[16],
        'simpsond': percentage_changes[17]
    }
    absolute_changes = {
        'words': absolute_changes[0],
        'terms': absolute_changes[1],
        'ttr': absolute_changes[2],
        'rttr': absolute_changes[3],
        'cttr': absolute_changes[4],
        'msttr': absolute_changes[5],
        'mattr': absolute_changes[6],
        'mtld': absolute_changes[7],
        'hdd': absolute_changes[8],
        'vocd': absolute_changes[9],
        'herdan': absolute_changes[10],
        'summer': absolute_changes[11],
        'dugast': absolute_changes[12],
        'maas': absolute_changes[13],
        'yulek': absolute_changes[14],
        'yulei': absolute_changes[15],
        'herdanvm': absolute_changes[16],
        'simpsond': absolute_changes[17]
    }
    return percentage_changes, absolute_changes, scores_original, scores_improved

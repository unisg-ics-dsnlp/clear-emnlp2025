import json
import os
from Levenshtein import distance, ratio, hamming, jaro, jaro_winkler


def levenshtein_scores(
        texts_original: list[str],
        texts_improved: list[str],
):
    scores = {
        'levenshtein': [],
        'ratio': [],
        'hamming': [],
        'jaro': [],
        'jaro_winkler': []
    }
    for text_original, text_improved in zip(texts_original, texts_improved):
        d = distance(text_original, text_improved)
        r = ratio(text_original, text_improved)
        h = hamming(text_original, text_improved)
        j = jaro(text_original, text_improved)
        jw = jaro_winkler(text_original, text_improved)
        scores['levenshtein'].append(d)
        scores['ratio'].append(r)
        scores['hamming'].append(h)
        scores['jaro'].append(j)
        scores['jaro_winkler'].append(jw)
    avgs = {key: sum(value) / len(value) for key, value in scores.items()}
    return scores

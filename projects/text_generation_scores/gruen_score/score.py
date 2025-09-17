import json
import os

import spacy
from tqdm import tqdm

from projects.text_generation_scores.gruen_score.main import get_gruen


def calculate_gruen_score(
        texts: list[str],
        nlp: spacy.language.Language
):
    gruen_scores_all = []
    gruen_scores_per_text = []
    for text in tqdm(texts, desc='Calculating GRUEN score'):
        doc = nlp(text)
        candidates = []
        for sent in doc.sents:
            candidates.append(sent.text.strip())
        if not candidates:
            # set to nan
            gruen_score = [float('nan')]
        else:
            try:
                gruen_score = get_gruen(candidates)
            except ValueError:
                gruen_score = [float('nan')]
        gruen_scores_all.extend(gruen_score)
        gruen_scores_per_text.append(gruen_score)
    print('avg GRUEN score: ', sum(gruen_scores_all) / len(gruen_scores_all))
    return gruen_scores_all, gruen_scores_per_text


if __name__ == '__main__':
    _nlp = spacy.load('en_core_web_sm')
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    local_dir = os.path.join(current_file_dir, '..', '..', '..', 'localFiles')
    essays_example_file = os.path.join(local_dir, 'essays-gemma-7b-it-cleaned.json')

    with open(essays_example_file, 'r') as file:
        data = json.load(file)

    _, direct_data = data[0]
    essays = [x[4] for x in direct_data[:10]]
    _scores_all, _scores_per_text = calculate_gruen_score(essays, _nlp)
    print(_scores_all)

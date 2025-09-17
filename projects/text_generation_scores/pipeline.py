import json
import os

import spacy

from projects.text_generation_scores.bertalign_scores.align_improvements import do_bertalign_metrics
from projects.text_generation_scores.complex_structures.annotate import calculate_complex_scores
from projects.text_generation_scores.gruen_score.score import calculate_gruen_score


def evaluation_pipeline(
        original_texts: list[str],
        improved_texts: list[str],
        src_lng: str = 'en',
        tgt_lng: str = 'en',
):
    nlp = spacy.load('en_core_web_sm')

    # BERTAlign
    bertalign_scores = []
    for original, improved in zip(original_texts, improved_texts):
        bertalign = do_bertalign_metrics(original, improved, src_lng, tgt_lng)
        bertalign_scores.append(bertalign)

    # GRUEN
    gruen_all, gruen_per_text = calculate_gruen_score(improved_texts, nlp)
    print('GRUEN scores:', gruen_all)
    print('GRUEN scores per text:', gruen_per_text)

    # Complex structures
    for score in bertalign_scores:
        targets = score.targets
        cmplx_anns = []
        for sent in targets:
            cmplx_ann = calculate_complex_scores(sent, nlp)
            cmplx_anns.append(cmplx_ann)
        total_num_relcl = sum([ann.num_relcl for ann in cmplx_anns])
        total_num_advcl = sum([ann.num_advcl for ann in cmplx_anns])
        total_num_appos = sum([ann.num_appos for ann in cmplx_anns])
        total_num_prep = sum([ann.num_prep for ann in cmplx_anns])
        total_num_coordNP = sum([ann.num_coordNP for ann in cmplx_anns])
        total_num_coord_cl = sum([ann.num_coord_cl for ann in cmplx_anns])
        total_num_coordVP = sum([ann.num_coordVP for ann in cmplx_anns])
        total_num_speech = sum([ann.num_speech for ann in cmplx_anns])
        total_num_adv_mod = sum([ann.num_adv_mod for ann in cmplx_anns])
        total_num_part = sum([ann.num_part for ann in cmplx_anns])
        print(f'total_num_relcl: {total_num_relcl}')
        print(f'total_num_advcl: {total_num_advcl}')
        print(f'total_num_appos: {total_num_appos}')
        print(f'total_num_prep: {total_num_prep}')
        print(f'total_num_coordNP: {total_num_coordNP}')
        print(f'total_num_coord_cl: {total_num_coord_cl}')
        print(f'total_num_coordVP: {total_num_coordVP}')
        print(f'total_num_speech: {total_num_speech}')
        print(f'total_num_adv_mod: {total_num_adv_mod}')
        print(f'total_num_part: {total_num_part}')
        print(f'total_num_sentences: {len(cmplx_anns)}')

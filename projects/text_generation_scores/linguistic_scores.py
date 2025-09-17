import json
from collections import defaultdict
from copy import deepcopy

import spacy


def spacy_scores(
        text: str,
        model: spacy.Language
):
    nlp = model(text)
    total_pos_tags = {
        'ADJ': 0,
        'ADP': 0,
        'ADV': 0,
        'AUX': 0,
        'CCONJ': 0,
        'DET': 0,
        'INTJ': 0,
        'NOUN': 0,
        'NUM': 0,
        'PART': 0,
        'PRON': 0,
        'PROPN': 0,
        'PUNCT': 0,
        'SCONJ': 0,
        'SPACE': 0,
        'SYM': 0,
        'VERB': 0,
        'X': 0
    }
    sentence_pos_tags = []
    # labels from here: https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
    total_dep_tags = {
        'ROOT': 0,
        'acl': 0,
        'acomp': 0,
        'advcl': 0,
        'advmod': 0,
        'agent': 0,
        'amod': 0,
        'appos': 0,
        'attr': 0,
        'aux': 0,
        'auxpass': 0,
        'case': 0,
        'cc': 0,
        'ccomp': 0,
        'compound': 0,
        'conj': 0,
        'csubj': 0,
        'csubjpass': 0,
        'dative': 0,
        'dep': 0,
        'det': 0,
        'dobj': 0,
        'expl': 0,
        'intj': 0,
        'mark': 0,
        'meta': 0,
        'neg': 0,
        'nmod': 0,
        'npadvmod': 0,
        'nsubj': 0,
        'nsubjpass': 0,
        'nummod': 0,
        'oprd': 0,
        'parataxis': 0,
        'pcomp': 0,
        'pobj': 0,
        'poss': 0,
        'preconj': 0,
        'predet': 0,
        'prep': 0,
        'prt': 0,
        'punct': 0,
        'quantmod': 0,
        'relcl': 0,
        'xcomp': 0
    }
    sentence_dep_tags = []
    total_ner = {
        'CARDINAL': 0,
        'DATE': 0,
        'EVENT': 0,
        'FAC': 0,
        'GPE': 0,
        'LANGUAGE': 0,
        'LAW': 0,
        'LOC': 0,
        'MONEY': 0,
        'NORP': 0,
        'ORDINAL': 0,
        'ORG': 0,
        'PERCENT': 0,
        'PERSON': 0,
        'PRODUCT': 0,
        'QUANTITY': 0,
        'TIME': 0,
        'WORK_OF_ART': 0,
        '': 0
    }
    sentence_ner = []
    num_sents = len(list(nlp.sents))
    for sent in nlp.sents:
        sent_pos_tags = deepcopy(total_pos_tags)
        sent_dep_tags = deepcopy(total_dep_tags)
        sent_ner = deepcopy(total_ner)
        for token in sent:
            total_pos_tags[token.pos_] += 1
            sent_pos_tags[token.pos_] += 1
            total_dep_tags[token.dep_] += 1
            sent_dep_tags[token.dep_] += 1
            total_ner[token.ent_type_] += 1
            sent_ner[token.ent_type_] += 1
        sentence_pos_tags.append(dict(sent_pos_tags))
        sentence_dep_tags.append(dict(sent_dep_tags))
        sentence_ner.append(dict(sent_ner))
    total_pos_tags = dict(total_pos_tags)
    total_dep_tags = dict(total_dep_tags)
    return total_pos_tags, sentence_pos_tags, total_dep_tags, sentence_dep_tags, total_ner, sentence_ner, num_sents

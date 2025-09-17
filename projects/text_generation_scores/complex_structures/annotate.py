from dataclasses import dataclass

import spacy


@dataclass
class ComplexSentenceAnnotations:
    num_relcl: int
    num_advcl: int
    num_appos: int
    num_prep: int
    num_coordNP: int
    num_coord_cl: int
    num_coordVP: int
    num_speech: int
    num_adv_mod: int
    num_part: int
    complex_dict: dict
    sent: str

    def to_dict(self):
        tmp = {
            'num_relcl': self.num_relcl,
            'num_advcl': self.num_advcl,
            'num_appos': self.num_appos,
            'num_prep': self.num_prep,
            'num_coordNP': self.num_coordNP,
            'num_coord_cl': self.num_coord_cl,
            'num_coordVP': self.num_coordVP,
            'num_speech': self.num_speech,
            'num_adv_mod': self.num_adv_mod,
            'num_part': self.num_part,
            'sent': self.sent
        }
        for k, v in self.complex_dict.items():
            tmp[f'complex_{k}'] = v
        return tmp


def calculate_complex_scores(
        sent: str,
        nlp: spacy.Language
) -> ComplexSentenceAnnotations:
    num_relcl = 0
    num_advcl = 0
    num_appos = 0
    num_prep = 0
    num_coordNP = 0
    num_coord_cl = 0
    num_coordVP = 0
    num_speech = 0
    num_adv_mod = 0
    num_part = 0
    complex_dict = {}
    count = 0
    sents = [s.text for s in nlp(sent).sents]
    all_sents = []
    for elem in sents:
        all_sents.append(elem)
    for sent in all_sents:
        doc = nlp(sent)
        complex_sentence_elements = []
        complex_elements = {'source': sent}
        for token in doc:
            if token.dep_ == 'relcl':
                rel_clause = ' '.join([t.text for t in token.subtree])
                complex_sentence_elements.append((rel_clause.strip(), 'relative clause'))
                num_relcl += 1
            elif token.dep_ == 'advcl':
                adverbial_clause = ' '.join([t.text for t in token.subtree])
                complex_sentence_elements.append((adverbial_clause.strip(), 'adverbial clause'))
                num_advcl += 1
            elif token.dep_ == 'appos':
                apposition = ' '.join([t.text for t in token.subtree])
                complex_sentence_elements.append((apposition.strip(), 'appositive phrase'))
                num_appos += 1
            elif token.dep_ == 'prep':
                if token.head.pos_ == 'VERB' or token.pos_ == 'VERB' or token.head.dep_ == 'prep' or token.dep_ == 'prep':
                    for t in token.subtree:
                        if t.dep_ == 'pobj':
                            prep = ' '.join([t.text for t in token.subtree])
                            complex_sentence_elements.append((prep.strip(), 'prepositional phrase'))
                            num_prep += 1
            elif token.dep_ == 'conj':
                head = ' '.join([t.text for t in token.head.subtree])
                if token.pos_ in ('NOUN', 'PROPN'):
                    num_coordNP += 1
                    complex_sentence_elements.append((head.strip(), 'coordinate noun phrases'))
                else:
                    coordinate_clause = False
                    for tok in token.children:
                        if tok.dep_ == 'nsubj':
                            complex_sentence_elements.append((head.strip(), 'coordinate clauses'))
                            coordinate_clause = True
                            num_coord_cl += 1
                    if not coordinate_clause:
                        complex_sentence_elements.append((head.strip(), 'coordinate verb phrases'))
                        num_coordVP += 1
            elif token.dep_ == 'ccomp':
                speech = ' '.join([t.text for t in token.subtree])
                complex_sentence_elements.append((speech.strip(), '(in)direct speech'))
                num_speech += 1
            elif token.dep_ == 'advmod':
                advmod = ' '.join([t.text for t in token.subtree])
                complex_sentence_elements.append((advmod.strip(), 'adverbial modifier'))
                num_adv_mod += 1
            elif token.dep_ == 'pcomp':
                part_phrases = ' '.join([t.text for t in token.head.subtree])
                complex_sentence_elements.append((part_phrases.strip(), 'participial phrases'))
                num_part += 1
            elif token.dep_ == 'acl':
                part_phrases = ' '.join([t.text for t in token.subtree])
                complex_sentence_elements.append((part_phrases.strip(), 'adjectival clause'))
                num_part += 1
        complex_sentence_elements = list(dict.fromkeys(complex_sentence_elements))
        complex_elements['complex_elements'] = complex_sentence_elements
        complex_dict[count] = complex_elements
        count += 1
    return ComplexSentenceAnnotations(
        num_relcl=num_relcl,
        num_advcl=num_advcl,
        num_appos=num_appos,
        num_prep=num_prep,
        num_coordNP=num_coordNP,
        num_coord_cl=num_coord_cl,
        num_coordVP=num_coordVP,
        num_speech=num_speech,
        num_adv_mod=num_adv_mod,
        num_part=num_part,
        complex_dict=complex_dict,
        sent=sent
    )


if __name__ == '__main__':
    _sent = '''Competition and cooperation are two fundamental aspects of life that have a profound impact on an individual's success.'''
    _nlp = spacy.load('en_core_web_sm')
    _out = calculate_complex_scores(_sent, _nlp)
    print(_out)

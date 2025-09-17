import json
import os

import spacy
# do not delete this import - script fails without it!! from spacytextblob.spacytextblob import SpacyTextBlob
from spacytextblob.spacytextblob import SpacyTextBlob
from germansentiment import SentimentModel
from tqdm import tqdm

from projects.improvement_pipeline.improvement_pipeline import load_arg_rewrite_df, prepare_essays_df, \
    prepare_microtexts_both_df

current_file_dir = os.path.dirname(os.path.abspath(__file__))


def do_sentiment_w_spacy(
        texts: list[str],
        out_identifier: str,
        is_german_lst: list[bool],
        nlp: spacy.language.Language,
        german_model: SentimentModel
):
    out_dir = os.path.join(current_file_dir, 'scores_out', 'SENTIMENT_SCORES')
    out_file_path = os.path.join(out_dir, f'{out_identifier}_sentiment.json')
    if os.path.exists(out_file_path):
        print(f'{out_file_path} already exists')
        return

    out = {
        'polarity': [],
        'subjectivity': [],
        'is_german': is_german_lst,
        'german_sentiment': [],
        'german_proba_positive': [],
        'german_proba_negative': [],
        'german_proba_neutral': []
    }
    for text, is_german in zip(texts, is_german_lst):
        if is_german:
            result, probas = german_model.predict_sentiment([text], output_probabilities=True)
            for i, proba_lst in enumerate(probas):
                sentiment = max(proba_lst, key=lambda x: x[1])[0]
                out['german_sentiment'].append(sentiment)
                for sentiment, proba in proba_lst:
                    if sentiment == 'positive':
                        out['german_proba_positive'].append(proba)
                    elif sentiment == 'negative':
                        out['german_proba_negative'].append(proba)
                    else:
                        out['german_proba_neutral'].append(proba)
            out['polarity'].append(None)
            out['subjectivity'].append(None)
        else:
            doc = nlp(text)
            polarity = doc._.blob.polarity
            subjectivity = doc._.blob.subjectivity
            out['polarity'].append(polarity)
            out['subjectivity'].append(subjectivity)
            out['german_sentiment'].append(None)
            out['german_proba_positive'].append(None)
            out['german_proba_negative'].append(None)
            out['german_proba_neutral'].append(None)
    with open(out_file_path, 'w') as f:
        json.dump(out, f, indent=4)


if __name__ == '__main__':
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
        ('REVISIONS_FEEDBACK', 'bloomz-3b-cleaned_revision1.json'),
        ('REVISIONS_FEEDBACK', 'bloomz-3b-cleaned_revision2.json'),
        ('REVISIONS_FEEDBACK', 'bloomz-3b-cleaned_revision3.json'),
        ('REVISIONS_FEEDBACK', 'bloomz-560m-cleaned_revision1.json'),
        ('REVISIONS_FEEDBACK', 'bloomz-560m-cleaned_revision2.json'),
        ('REVISIONS_FEEDBACK', 'bloomz-560m-cleaned_revision3.json'),
        ('REVISIONS_FEEDBACK', 'OLMo-7B-0724-Instruct-hf-cleaned_revision1.json'),
        ('REVISIONS_FEEDBACK', 'OLMo-7B-0724-Instruct-hf-cleaned_revision2.json'),
        ('REVISIONS_FEEDBACK', 'OLMo-7B-0724-Instruct-hf-cleaned_revision3.json'),
        ('REVISIONS_FEEDBACK', 'Phi-3-medium-4k-instruct-cleaned_revision1.json'),
        ('REVISIONS_FEEDBACK', 'Phi-3-medium-4k-instruct-cleaned_revision2.json'),
        ('REVISIONS_FEEDBACK', 'Phi-3-medium-4k-instruct-cleaned_revision3.json'),
        ('REVISIONS_FEEDBACK', 'Phi-3-mini-4k-instruct-cleaned_revision1.json'),
        ('REVISIONS_FEEDBACK', 'Phi-3-mini-4k-instruct-cleaned_revision2.json'),
        ('REVISIONS_FEEDBACK', 'Phi-3-mini-4k-instruct-cleaned_revision3.json'),
    ]
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('spacytextblob')
    german_model = SentimentModel()

    for folder, file_name in tqdm(_files_all, desc='Processing score files'):
        out_name = file_name[:len(file_name) - len('.json')]
        if not 'medium' in out_name.lower():
            print(f'File name {file_name} does not contain "medium" - UGLY WORKAROUND, FIX LATER!!')
            continue
        _test_file = os.path.join(current_file_dir, '..', 'improved_out', folder, file_name)
        with open(_test_file, 'r') as file:
            data = json.load(file)
        for approach in data:
            approach_name = approach['approach']
            original = approach['original_arguments']
            improved = approach['improved_arguments']
            if folder == 'MICROTEXTS':
                topic_ids = approach['topic_id']
                is_german_lst = [topic_id.endswith('(de)') for topic_id in topic_ids]
            else:
                is_german_lst = [False] * len(improved)

            # print(approach_name, folder, file_name, len(original), len(improved))

            # get indicies where improved.strip() is empty
            empty_improved = [i for i, x in enumerate(improved) if not x.strip()]
            # exclude those from original and improved
            original = [x for i, x in enumerate(original) if i not in empty_improved]
            improved = [x for i, x in enumerate(improved) if i not in empty_improved]
            is_german_lst = [x for i, x in enumerate(is_german_lst) if i not in empty_improved]

            out_dir = os.path.join(current_file_dir, 'scores_out')
            out_file_name = f'{out_name}__{folder}__{approach_name}_empty_sentiment_json'

            # save empty ones
            with open(os.path.join(out_dir, out_file_name), 'w') as f:
                json.dump(empty_improved, f, indent=4)

            do_sentiment_w_spacy(
                improved,
                f'{out_name}__{folder}__{approach_name}_improved_sentiment',
                is_german_lst,
                nlp,
                german_model
            )
    #
    # # human scores
    # folder = 'HUMAN_SENTIMENT'
    #
    # rewrite_df = load_arg_rewrite_df()
    # file_name1 = 'human_revision1to2.json'
    # file_name2 = 'human_revision2to3.json'
    #
    # original_1to2 = rewrite_df['draft1'].tolist()
    # improved_1to2 = rewrite_df['draft2'].tolist()
    #
    # original_2to3 = rewrite_df['draft2'].tolist()
    # improved_2to3 = rewrite_df['draft3'].tolist()
    #
    # do_sentiment_w_spacy(
    #     original_1to2,
    #     f'{folder}__human_draft1_original_sentiment',
    #     [False] * len(original_1to2),
    #     nlp,
    #     german_model
    # )
    # do_sentiment_w_spacy(
    #     improved_1to2,
    #     f'{folder}__human_draft2_improved_sentiment',
    #     [False] * len(improved_1to2),
    #     nlp,
    #     german_model
    # )
    # do_sentiment_w_spacy(
    #     improved_2to3,
    #     f'{folder}__human_draft3_improved_sentiment',
    #     [False] * len(improved_2to3),
    #     nlp,
    #     german_model
    # )
    #
    # essay_df = prepare_essays_df()
    # essays = essay_df['argument'].tolist()
    # do_sentiment_w_spacy(
    #     essays,
    #     f'{folder}__essays_sentiment',
    #     [False] * len(essays),
    #     nlp,
    #     german_model
    # )
    #
    # microtexts_df = prepare_microtexts_both_df()
    # topic_ids = microtexts_df['topic_id']
    # is_german_lst = [topic_id.endswith('(de)') for topic_id in topic_ids]
    # microtexts = microtexts_df['argument'].tolist()
    # do_sentiment_w_spacy(
    #     microtexts,
    #     f'{folder}__microtexts_sentiment',
    #     is_german_lst,
    #     nlp,
    #     german_model
    # )

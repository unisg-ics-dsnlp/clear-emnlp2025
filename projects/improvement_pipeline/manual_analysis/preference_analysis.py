import json
import os

import pandas as pd

CURRENT_FILE_DIR = os.getcwd()
IMPROVED_DIR = '../improved_out'
MICROTEXTS_PATH = os.path.join(IMPROVED_DIR, 'MICROTEXTS', 'llama3_nemotron_cleaned.json')
with open(MICROTEXTS_PATH) as f:
    microtext_data = json.load(f)

ESSAYS_PATH = os.path.join(IMPROVED_DIR, 'ESSAYS', 'llama3_nemotron_cleaned.json')
with open(ESSAYS_PATH) as f:
    essay_data = json.load(f)

REVISIONS1_PATH = os.path.join(IMPROVED_DIR, 'REVISIONS', 'llama3_nemotron_cleaned_revision1.json')
with open(REVISIONS1_PATH) as f:
    revision1_data = json.load(f)

REVISIONS2_PATH = os.path.join(IMPROVED_DIR, 'REVISIONS', 'llama3_nemotron_cleaned_revision2.json')
with open(REVISIONS2_PATH) as f:
    revision2_data = json.load(f)

REVISIONS3_PATH = os.path.join(IMPROVED_DIR, 'REVISIONS', 'llama3_nemotron_cleaned_revision3.json')
with open(REVISIONS3_PATH) as f:
    revision3_data = json.load(f)

REVISIONS1_FEEDBACK_PATH = os.path.join(IMPROVED_DIR, 'REVISIONS_FEEDBACK', 'llama3_nemotron_cleaned_revision1.json')
with open(REVISIONS1_FEEDBACK_PATH) as f:
    revision1_feedback_data = json.load(f)

REVISIONS2_FEEDBACK_PATH = os.path.join(IMPROVED_DIR, 'REVISIONS_FEEDBACK', 'llama3_nemotron_cleaned_revision2.json')
with open(REVISIONS2_FEEDBACK_PATH) as f:
    revision2_feedback_data = json.load(f)

REVISIONS3_FEEDBACK_PATH = os.path.join(IMPROVED_DIR, 'REVISIONS_FEEDBACK', 'llama3_nemotron_cleaned_revision3.json')
with open(REVISIONS3_FEEDBACK_PATH) as f:
    revision3_feedback_data = json.load(f)

data_all = [
    microtext_data,
    microtext_data,
    essay_data,
    revision1_data,
    revision2_data,
    revision3_data,
    revision1_feedback_data,
    revision2_feedback_data,
    revision3_feedback_data
]
data_all_names = [
    'microtext_de',
    'microtext_en',
    'essay',
    'revision1',
    'revision2',
    'revision3',
    'revision1_feedback',
    'revision2_feedback',
    'revision3_feedback'
]

microtext_english_ids = [
    'micro_b011',
    'micro_b055,',
    'micro_b001',
    'micro_b046',
    'micro_b019',
    'micro_b031',
    'micro_b023',
    'micro_b051',
    'micro_b013',
    'micro_k003'
]
microtext_german_ids = [
    'micro_b011',
    'micro_b055',
    'micro_b001',
    'micro_b046',
    'micro_b019',
    'micro_b031',
    'micro_b023',
    'micro_b051',
    'micro_b013',
    'micro_k003'
]
essays_ids = [
    'essay226.txt',
    'essay282.txt',
    'essay085.txt',
    'essay286.txt',
    'essay095.txt',
    'essay034.txt',
    'essay127.txt',
    'essay094.txt',
    'essay212.txt',
    'essay392.txt'
]
rev1_ids = [
    'draft1_2018argrewrite_9.txt',
    'draft1_2018argrewrite_16.txt',
    'draft1_2018argrewrite_26.txt',
    'draft1_2018argrewrite_65.txt',
    'draft1_2018argrewrite_63.txt',
    'draft1_2018argrewrite_46.txt',
    'draft1_2018argrewrite_101.txt',
    'draft1_2018argrewrite_103.txt',
    'draft1_2018argrewrite_70.txt',
    'draft1_2018argrewrite_19.txt'
]
rev2_ids = [
    'draft1_2018argrewrite_9.txt',
    'draft1_2018argrewrite_16.txt',
    'draft1_2018argrewrite_26.txt',
    'draft1_2018argrewrite_65.txt',
    'draft1_2018argrewrite_63.txt',
    'draft1_2018argrewrite_46.txt',
    'draft1_2018argrewrite_101.txt',
    'draft1_2018argrewrite_103.txt',
    'draft1_2018argrewrite_70.txt',
    'draft1_2018argrewrite_19.txt'
]
rev3_ids = [
    'draft1_2018argrewrite_17.txt',
    'draft1_2018argrewrite_11.txt',
    'draft1_2018argrewrite_19.txt',
    'draft1_2018argrewrite_74.txt',
    'draft1_2018argrewrite_65.txt',
    'draft1_2018argrewrite_78.txt',
    'draft1_2018argrewrite_40.txt',
    'draft1_2018argrewrite_101.txt',
    'draft1_2018argrewrite_76.txt',
    'draft1_2018argrewrite_38.txt'
]

out = []

for data, data_name in zip(data_all, data_all_names):
    for approach_data in data:
        original_arguments = approach_data['original_arguments']
        improved_arguments = approach_data['improved_arguments']
        graph_ids = approach_data['graph_id']
        topic_ids = approach_data['topic_id']
        tmp_df = pd.DataFrame({
            'original_arguments': original_arguments,
            'improved_arguments': improved_arguments,
            'graph_id': graph_ids,
            'topic_id': topic_ids,
            'approach': [approach_data['approach']] * len(original_arguments),
            'dataset_name': [data_name] * len(original_arguments),
        })
        # remove @ from end of improved_arguments if its at the very end
        tmp_df['improved_arguments'] = tmp_df['improved_arguments'].str.rstrip('@')

        if data_name == 'microtext_de':
            tmp_df = tmp_df[tmp_df['topic_id'].str.endswith('(de)')]
        elif data_name == 'microtext_en':
            tmp_df = tmp_df[tmp_df['topic_id'].str.endswith('(en)')]

        # filter out those where graph id is in the list of ids
        if data_name == 'microtext_en':
            tmp_df = tmp_df[~tmp_df['graph_id'].isin(microtext_english_ids)]
        elif data_name == 'microtext_de':
            tmp_df = tmp_df[~tmp_df['graph_id'].isin(microtext_german_ids)]
        elif data_name == 'essay':
            tmp_df = tmp_df[~tmp_df['graph_id'].isin(essays_ids)]
        elif data_name == 'revision1':
            tmp_df = tmp_df[~tmp_df['graph_id'].isin(rev1_ids)]
        elif data_name == 'revision2':
            tmp_df = tmp_df[~tmp_df['graph_id'].isin(rev2_ids)]
        elif data_name == 'revision3':
            tmp_df = tmp_df[~tmp_df['graph_id'].isin(rev3_ids)]
        elif data_name == 'revision1_feedback':
            tmp_df = tmp_df[~tmp_df['graph_id'].isin(rev1_ids)]
        elif data_name == 'revision2_feedback':
            tmp_df = tmp_df[~tmp_df['graph_id'].isin(rev2_ids)]
        elif data_name == 'revision3_feedback':
            tmp_df = tmp_df[~tmp_df['graph_id'].isin(rev3_ids)]

        # pick 20 for each dataset
        samples = tmp_df.sample(20, random_state=42)
        out.append(samples)

out_df = pd.concat(out, ignore_index=True)
out_df = out_df.sort_values(by=['approach', 'dataset_name'])


def save_files(
        mask_file,
        check_df,
        t1t2_df,
        identifier
):
    with open(os.path.join(CURRENT_FILE_DIR, 'preference_manual', f'{identifier}_mask.json'), 'w') as f:
        json.dump(mask_file, f, indent=4)
    check_df.to_csv(os.path.join(CURRENT_FILE_DIR, 'preference_manual', f'{identifier}_check_df.csv'), index=False)
    t1t2_df.to_csv(os.path.join(CURRENT_FILE_DIR, 'preference_manual', f'{identifier}_t1t2_df.csv'), index=False)


microtexts_english = out_df[out_df['dataset_name'] == 'microtext_en']
check_mt_english = microtexts_english[microtexts_english['approach'] == 'direct']

# create boolean of len(check_mt_english), randomly
import random

check_mt_english_mask = random.choices([True, False], k=len(check_mt_english))

# add columns text1 and text2 - use original_arguments for text1 of mask is True, else improved_arguments
check_mt_english['text1'] = check_mt_english['original_arguments'].where(check_mt_english_mask,
                                                                         check_mt_english['improved_arguments'])
check_mt_english['text2'] = check_mt_english['improved_arguments'].where(check_mt_english_mask,
                                                                         check_mt_english['original_arguments'])
t1t2_mt_english = check_mt_english[['topic_id', 'graph_id', 'text1', 'text2']].copy()
save_files(check_mt_english_mask, check_mt_english, t1t2_mt_english, 'microtext_en')

microtexts_german = out_df[out_df['dataset_name'] == 'microtext_de']
check_mt_german = microtexts_german[microtexts_german['approach'] == 'direct']
# create boolean of len(check_mt_german), randomly
check_mt_german_mask = random.choices([True, False], k=len(check_mt_german))
# add columns text1 and text2 - use original_arguments for text1 of mask is True, else improved_arguments
check_mt_german['text1'] = check_mt_german['original_arguments'].where(check_mt_german_mask,
                                                                       check_mt_german['improved_arguments'])
check_mt_german['text2'] = check_mt_german['improved_arguments'].where(check_mt_german_mask,
                                                                       check_mt_german['original_arguments'])
t1t2_mt_german = check_mt_german[['topic_id', 'graph_id', 'text1', 'text2']].copy()
save_files(check_mt_german_mask, check_mt_german, t1t2_mt_german, 'microtext_de')

essays = out_df[out_df['dataset_name'] == 'essay']
check_essays = essays[essays['approach'] == 'direct']
# create boolean of len(check_essays), randomly
check_essays_mask = random.choices([True, False], k=len(check_essays))
# add columns text1 and text2 - use original_arguments for text1 of mask is True, else improved_arguments
check_essays['text1'] = check_essays['original_arguments'].where(check_essays_mask,
                                                                 check_essays['improved_arguments'])
check_essays['text2'] = check_essays['improved_arguments'].where(check_essays_mask,
                                                                 check_essays['original_arguments'])
t1t2_essays = check_essays[['topic_id', 'graph_id', 'text1', 'text2']].copy()
save_files(check_essays_mask, check_essays, t1t2_essays, 'essay')

revisions1 = out_df[out_df['dataset_name'] == 'revision1']
check_revisions1 = revisions1[revisions1['approach'] == 'direct']
# create boolean of len(check_revisions1), randomly
check_revisions1_mask = random.choices([True, False], k=len(check_revisions1))
# add columns text1 and text2 - use original_arguments for text1 of mask is True, else improved_arguments
check_revisions1['text1'] = check_revisions1['original_arguments'].where(check_revisions1_mask,
                                                                         check_revisions1['improved_arguments'])
check_revisions1['text2'] = check_revisions1['improved_arguments'].where(check_revisions1_mask,
                                                                         check_revisions1['original_arguments'])
t1t2_revisions1 = check_revisions1[['topic_id', 'graph_id', 'text1', 'text2']].copy()
save_files(check_revisions1_mask, check_revisions1, t1t2_revisions1, 'revision1')

revisions2 = out_df[out_df['dataset_name'] == 'revision2']
check_revisions2 = revisions2[revisions2['approach'] == 'direct']
# create boolean of len(check_revisions2), randomly
check_revisions2_mask = random.choices([True, False], k=len(check_revisions2))
# add columns text1 and text2 - use original_arguments for text1 of mask is True, else improved_arguments
check_revisions2['text1'] = check_revisions2['original_arguments'].where(check_revisions2_mask,
                                                                         check_revisions2['improved_arguments'])
check_revisions2['text2'] = check_revisions2['improved_arguments'].where(check_revisions2_mask,
                                                                         check_revisions2['original_arguments'])
t1t2_revisions2 = check_revisions2[['topic_id', 'graph_id', 'text1', 'text2']].copy()
save_files(check_revisions2_mask, check_revisions2, t1t2_revisions2, 'revision2')

revisions3 = out_df[out_df['dataset_name'] == 'revision3']
check_revisions3 = revisions3[revisions3['approach'] == 'direct']
# create boolean of len(check_revisions3), randomly
check_revisions3_mask = random.choices([True, False], k=len(check_revisions3))
# add columns text1 and text2 - use original_arguments for text1 of mask is True, else improved_arguments
check_revisions3['text1'] = check_revisions3['original_arguments'].where(check_revisions3_mask,
                                                                         check_revisions3['improved_arguments'])
check_revisions3['text2'] = check_revisions3['improved_arguments'].where(check_revisions3_mask,
                                                                         check_revisions3['original_arguments'])
t1t2_revisions3 = check_revisions3[['topic_id', 'graph_id', 'text1', 'text2']].copy()
save_files(check_revisions3_mask, check_revisions3, t1t2_revisions3, 'revision3')

import json
import os

from util.template import llama3_format_func
from direct.direct import DirectPrompt

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

approaches = [microtext_data[i]['approach'] for i in range(len(microtext_data))]
approaches

data_all = [
    microtext_data,
    microtext_data,
    essay_data,
    revision1_data,
    revision2_data,
    revision3_data,
]
data_all_names = [
    'microtext_de',
    'microtext_en',
    'essay',
    'revision1',
    'revision2',
    'revision3',
]

import pandas as pd

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
        if data_name == 'microtext_de':
            tmp_df = tmp_df[tmp_df['topic_id'].str.endswith('(de)')]
        elif data_name == 'microtext_en':
            tmp_df = tmp_df[tmp_df['topic_id'].str.endswith('(en)')]
        samples = tmp_df.sample(10, random_state=42)
        out.append(samples)

out_df = pd.concat(out, ignore_index=True)
out_df = out_df.sort_values(by=['approach', 'dataset_name'])
out_df = out_df[out_df['approach'] == 'direct']
print(out_df)

direct = DirectPrompt(
    'llama3',
    format_func=llama3_format_func,
    use_vllm=False
)

formatted_prompts = []
for i, row in out_df.iterrows():
    p = direct.format_prompt('Summarize the changes that were made to the original argument.',
                             f'[ORIGINAL]{row["original_arguments"]}[END ORIGINAL]\n\n[IMPROVED]{row["improved_arguments"]}[END IMPROVED]')
    formatted_prompts.append(p)

out = direct.do_prompting(formatted_prompts)

out_dct = [
    {
        'graph_id': row['graph_id'],
        'approach': row['approach'],
        'topic_id': row['topic_id'],
        'dataset_name': row['dataset_name'],
        'analysis': out[j]
    }
    for j, (i, row) in enumerate(out_df.iterrows())
]

with open('llama3_nemotron_direct_summary.json', 'w') as f:
    json.dump(out_dct, f, indent=4)
